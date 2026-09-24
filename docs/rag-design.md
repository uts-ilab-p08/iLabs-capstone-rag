# RAG Layer — Design Document

**Project:** 08-01 Interactive Surveillance Video Querying Using LLMs and Multi-Camera CCTV Datasets
**Component:** Vector index, retrieval and answer generation (the proposal's "RAG layer" and Stage 8)
**Status:** 2026-09-24. Describes what is built and running, not a plan.

---

## 1. Purpose and scope

The annotation pipeline turns raw MEVA footage into **structured events** — a
short natural-language description of one moment, with its location, camera,
timespan and detected objects — and stores them in a Supabase Postgres database.

This component sits on top of that store and answers questions about the footage:

1. **Indexing** — read events from Postgres, turn each into a vector, and store
   it in Qdrant with its metadata.
2. **Retrieval** — turn a question into a vector, optionally narrow the search
   with metadata filters, and return the closest events.
3. **Answer generation** — give those events to an LLM and return a grounded
   answer that cites them.

### In scope

Indexing, retrieval, metadata filtering, answer generation, and the evaluation of
retrieval and answer quality.

### Out of scope

Video processing, detection, tracking and captioning; the Postgres schema and
raw-video storage; the HTTP API; the frontend. This component is a Python package
that the backend imports and calls; it knows nothing about HTTP, users or
sessions.

### Boundaries

| Direction | Interface |
|---|---|
| Input | The `bronze` schema in Supabase Postgres (read-only; indexing only) |
| Input | A question as a plain string |
| Output | A dict containing the answer and the events it was based on |

---

## 2. Decisions

| Area | Choice | Why |
|---|---|---|
| Vector database | **Qdrant**, hosted free tier | Filterable HNSW keeps recall when filters are selective. Hosted rather than local so the backend can reach the same index and several processes can share it. |
| Embedding model | **BAAI/bge-base-en-v1.5**, 768-dim | Free, open, strong on short English text, and has distinct query and document prompts. |
| Embedding runtime | **fastembed** (ONNX) | Roughly 100 MB installed and no PyTorch, so the repo stays light enough for others to clone and run. Also ships the models needed for the planned benchmark. |
| LLM access | **OpenRouter**, with local **Ollama** as an alternative | Both speak the OpenAI chat-completions API, so the model is three settings in `.env` rather than a code change. Swapping models for comparison costs nothing. |
| Query filtering | **Rule-based**, not LLM-extracted | See §6.1. Deterministic, instant, and fails safe. |
| Language | Python 3.11+ | Matches the backend. |

---

## 3. The data

### 3.1 Source schema (`bronze`)

Five tables. Indexing reads the first four; `geometries` is deliberately not read.

```
videos
  video_id PK, video_name (MEVA clip filename), camera_id, scene (location),
  capture_start_local, capture_end_local, capture_time_zone,
  duration_seconds, fps, frame_width, frame_height, video_url, ...

events
  event_id PK, video_id FK, event_name (short free-text label),
  description (the caption), start_frame, end_frame,
  start_seconds, end_seconds, source_status

event_objects
  event_id FK, object_id FK          -- many-to-many junction

objects
  object_id PK, video_id FK, source_object_id, detection_count,
  first_seen_seconds, last_seen_seconds,
  label_details (jsonb list of competing labels with confidences)

geometries
  geometry_id PK, object_id FK, frame_index, timestamp_seconds,
  label, confidence, spatial_position, bounding_box_pixels
```

`geometries` holds per-frame bounding boxes — spatial-evaluation and
frontend-overlay data. It is large, useless for semantic search, and never read
by this component.

### 3.2 Characteristics that shaped the design

These are properties of the real data, each of which forced a design decision.

**Several cameras watch each location, synchronised to GPS time.** MEVA's ground
cameras have overlapping fields of view. In the current index, **113 pairs of
events share a location and an overlapping wall-clock time while coming from
different cameras** — the same occurrence filmed from two angles. Any rule of the
form "different camera means different object" is therefore wrong, and wrong on
the majority of the data. This is why the prompt reasons about location and clock
time instead (§7).

**Clock time is only recoverable from the clip filename.** Events store an offset
within their clip. Clips are named `2018-03-05.13-20-00.13-25-00.school.G424`, so
the absolute time is the clip start plus the offset. Without that, two events at
"0:41" in different clips are indistinguishable from simultaneous ones.

**Events are duplicated.** Roughly 13% of event rows are exact duplicates —
identical caption, identical timespan, same video, different `event_id`. Some
share the same objects (one occurrence inserted more than once); others differ
only in which object they name. Indexing collapses them (§4).

**Object labels are unreliable.** `label_details` is a list of competing
hypotheses: the same white SUV appears as `car` in one event and `truck` in
another. Object type is therefore never used as a filter, and the prompt is told
to distrust the labels.

**Activity labels are free text.** `event_name` is model-generated prose
("Vehicle movement", "Person exits a white SUV"), not the 37-class MEVA
taxonomy, and casing is inconsistent. There is no controlled vocabulary to
filter on.

**All current footage is from one date**, so the date carries no information and
is omitted from the prompt unless retrieved events actually span more than one.

### 3.3 Qdrant point schema

One point per de-duplicated event.

```
id       = uuid5(NAMESPACE_URL, event_id)
vector   = 768 floats, cosine distance
payload  = event_id, video_id, video_name, camera_id, scene,
           event_name, description, start_seconds, end_seconds,
           object_types, source_status, video_url, composed_text
```

**Point IDs are derived from `event_id` by a pure function.** The same event
always produces the same point ID, so re-indexing overwrites rather than
duplicates and can be run at any time. Qdrant requires IDs to be integers or
UUIDs, and `event_id` is a 64-character hex string, hence the hash.

Payload indexes exist on `camera_id`, `scene`, `object_types`, `source_status`,
`start_seconds` and `end_seconds`. They are required for efficient filtering and
for the facet queries that build the filter vocabulary (§6.1).

`composed_text` is stored only so it is possible to see exactly what was
embedded when a result looks wrong.

---

## 4. Indexing

`scripts/index_events.py`: fetch → compose → embed → upsert.

- **Fetch** joins `events` to `videos`, and collapses exact duplicates with
  `DISTINCT ON (video_id, description, start_seconds, end_seconds)`. This is a
  working convenience so results are readable, not a data-cleaning layer; it can
  be removed once de-duplication happens upstream.
- Object labels are resolved in Python rather than SQL: each object's primary
  label is the one maximising `average confidence × detection_count`, so a label
  backed by 24 frames beats a more confident single-frame guess.
- **Embedding** is batched, and **upload** is batched separately at 100 points
  per request — a single request carrying every point will time out against a
  hosted cluster.
- Safe to re-run at any time; the index is derived state and can be deleted and
  rebuilt in about a minute.

---

## 5. What gets embedded

Not the bare caption. Each event is rendered as:

```
{description}
Activity: {event_name}. Objects: {object_types}.
Camera: {camera_id} ({scene}).
```

A question like "a vehicle at the bus stop" has to match on location and object
type, and neither necessarily appears in the caption text. Folding the structured
fields into the embedded string makes them reachable by semantic search.

The raw `description` is stored separately and is what the user and the LLM see.
The composed string is machine input only.

---

## 6. Retrieval

### 6.1 Metadata filtering

A question naming a location or a camera is filtered before ranking.

**The vocabulary is read from Qdrant, not Postgres**, using the facet API to get
distinct values server-side. This matters: the vocabulary must describe what is
*searchable*. A location present in Postgres but not yet indexed would otherwise
become a filter that matches nothing. It is cached per process, so a long-running
service will not see newly indexed locations until it restarts.

Matching is rule-based against that vocabulary, plus a small synonym map
(`campus`→school, `clinic`→hospital, `depot`→bus). Camera IDs are matched as
exact tokens and discarded if unknown, so a typo cannot become a filter that
guarantees zero results.

**Every rule errs toward filtering on nothing**, because the two failure modes
are not symmetric. A missed filter leaves results slightly noisy. A wrong filter
reports "no matching footage" about footage that exists, and nothing downstream
can recover it. So:

- An ambiguous question ("was there a school bus?", which matches two locations)
  applies no location filter.
- A negated one ("anything except the school") applies no filter rather than
  guessing at the inverse.
- Both record why in `filters.notes`.

Object type is deliberately not filtered on, because the labels disagree with
themselves (§3.2).

Rules were chosen over an LLM extractor because the vocabulary is tiny, the
result is deterministic — which matters while measuring retrieval — and it adds
no latency or failure mode. The evaluation will show whether rules miss enough to
justify an LLM fallback.

### 6.2 The similarity threshold

Unfiltered searches drop results below a cosine similarity of 0.6.

**When a filter matches, the floor is removed entirely.** The filter has become
the relevance signal: if the user asked for the hospital, hospital events are
relevant regardless of how the text reads.

This is not a tuning detail, it is what makes scope questions work. "What
happened in the hospital clip" is a *summary* request, and no individual event
description resembles a summary request — so every hospital event scores around
0.45–0.50 even though the ranking is perfectly correct. A fixed floor discards
all of them and the system reports no footage for footage it holds. The score
measures textual resemblance, not relevance.

---

## 7. Answer generation

Retrieved events are rendered into a prompt and sent to the LLM. The prompt is
versioned (`PROMPT_VERSION`); evaluation results are only comparable within one
version.

The rules exist because of specific observed failures:

- **Judge by location and clock time, not camera name.** Cameras overlap, so the
  camera alone means nothing. Same location and overlapping times is one
  occurrence; same location at different times is two; different locations are
  different objects. Without this the model merged vehicles from different sites
  into one journey; with an earlier, cruder version of it, the model split one
  person filmed by two cameras into two people.
- **Be careful counting**, and say so when it cannot be determined.
- **Answer from the closest match first**, because the frontend shows that event
  most prominently.
- **Distrust the detector labels**, which contradict each other.
- **When the answer is no, say so in one sentence and stop.** A long answer to a
  question with nothing to report is just room for mistakes — an early version
  padded a "no helipad" answer with invented detail.
- **Style:** no preamble, no repetition of metadata already on screen, clock
  times only, square brackets reserved for citations, no closing restatement.

Events are rendered with location and time first and camera last, because those
are the fields the model must compare, and reading the camera first invites the
wrong conclusion.

**Failure handling.** If no model is configured, or the call fails, the answer
degrades to a plain summary rather than raising, so the backend never returns an
error because of the model. The fallback wording distinguishes filtered from
unfiltered results, since filtered events are "from the place you asked about"
rather than "matching your question".

---

## 8. Public interface

The backend imports one function:

```python
from rag.pipeline import answer_query
response = answer_query(query)      # query: str
```

```json
{
  "query": "the question as received",
  "answer": "generated answer, citing events as [1], [2]",
  "sources": [
    {"score": 0.649, "annotation": "...", "video_id": "...", "event_id": "..."}
  ],
  "filters": {"scenes": ["hospital"], "cameras": [], "notes": []}
}
```

- `sources` holds at most five events, best first, and is empty when nothing
  matches. Anything beyond these four fields is looked up by the backend in
  Postgres using `event_id`.
- `filters` is diagnostic and may be ignored.
- Citations are not guaranteed — weaker models sometimes omit them — so the
  frontend must not depend on their presence.

Richer fields (location, camera, timestamps) exist internally because the prompt
is built from them; they are trimmed at this boundary.

---

## 9. Configuration and deployment

Configuration comes from the **process environment**. A local `.env` file is a
convenience that is copied into the environment at startup; deployed services
supply the same variables through their platform instead. `.env` is never
committed.

```
DATABASE_URL     indexing only
QDRANT_URL, QDRANT_API_KEY
LLM_BASE_URL, LLM_MODEL, LLM_API_KEY
EMBED_MODEL, EMBED_DIM, QDRANT_COLLECTION   (defaults are fine)
```

`DATABASE_URL` is **optional**. Only indexing reads Postgres; answering a
question needs Qdrant and the LLM alone. A query-only deployment therefore needs
no database credentials at all, and indexing without it fails with an explanatory
error rather than a crash on import.

Deployment considerations not yet validated: the embedding model runs in-process
and needs meaningful memory, and model weights (~130 MB) are downloaded on first
use, so a platform with an ephemeral filesystem re-downloads them after every
restart.

---

## 10. Evaluation

Planned, not yet built. A set of 25 questions with their expected footage, used
to measure:

- **Retrieval quality** — whether a correct event appears in the top 1, 3 or 5,
  plus questions whose correct answer is that nothing matches.
- **How often filters fire**, and whether rule-based extraction misses enough to
  justify an LLM extractor.
- **Model comparison** — the same questions across several models, the reason
  temperature is fixed at 0.
- **Embedding comparison** — bge-base against bge-large and arctic-embed-l, a
  proposal deliverable.
- **Whether the LLM layer earns its place** — the same questions answered by the
  top annotation alone, by the LLM over five events, and by the LLM with
  filtering.

Results must record the prompt version, embedding model and LLM model, or they
are not comparable.

---

## 11. Not built

Temporal-ordering queries ("what happened after the van arrived"), hybrid
keyword-plus-vector search, LLM-based filter extraction, chronological
whole-clip summarisation as a distinct retrieval mode, and automated tests.

---

## 12. Open questions

1. Will a controlled activity vocabulary ever replace the free-text
   `event_name`? It would make activity a filterable field.
2. Is upstream de-duplication planned? The ingest-time collapse can then be
   removed.
3. `capture_time_zone` is `unknown` on every row, so time-of-day filtering
   ("anything this morning") is not possible.
4. Is a cleaned layer planned on top of `bronze`? If so this component should
   read from it, since `bronze` is explicitly a landing layer.
5. Should indexing use its own read-only database role rather than a shared
   credential?
6. Is cross-camera object identity available anywhere? `object_id` is currently
   scoped to a single video, so a person cannot be followed between cameras.

---

## 13. Milestones

| Week | Date | This component |
|---|---|---|
| 6 | 31 Aug – 6 Sep | Schema exploration; repo; first fetch |
| 7 | 7–13 Sep | Indexing pipeline and semantic search |
| 8 | 14–20 Sep | Answer generation; hosted vector database |
| 9 | 21–27 Sep | Metadata filtering; prompt design; **Progress Report** |
| 10 | 28 Sep – 4 Oct | Evaluation; embedding and model comparisons |
| 11–12 | 5–18 Oct | Remaining features as evaluation directs; **Final Report** |
