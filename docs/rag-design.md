# RAG Layer — Design Document

**Project:** 08-01 Interactive Surveillance Video Querying Using LLMs and Multi-Camera CCTV Datasets
**Component:** Vector index + query interpretation + Q&A (proposal "RAG layer" and Stage 8)
**Owners:** Abhishek Chopda (RAG Lead), with Saurabh Sabharwal, Gourika Sood (RAG support)
**Status:** Draft v0.1 — 2026-09-03. For team review (Juan: schema fit; Maria: evaluation-interface fit).

---

## 1. Purpose and scope

### 1.1 What this component does

The annotation pipeline (Juan, stages 1–6) converts video into **structured events**
and stores them, with Qwen-VL captions and metadata, in a Supabase Postgres
database. This component sits on top of that store and provides:

1. **Indexing** — read events + captions + metadata from Postgres, embed them, and
   store the vectors plus a structured payload in a vector database (Qdrant).
2. **Query interpretation (Stage 8)** — take a user's natural-language question and
   use an LLM to extract structured filters (time window, cameras, object types,
   activity types, temporal-ordering constraints) plus a semantic search string.
3. **Retrieval** — run a filtered vector search against Qdrant, optionally
   re-rank for temporal-ordering queries, and return a ranked list of events.
4. **Answer generation** — assemble the retrieved events into context and generate
   a grounded natural-language answer with citations to clip + timespan.

### 1.2 In scope

- Ingestion pipeline from Supabase → Qdrant.
- LLM-based query → structured-filter parsing.
- Filtered semantic retrieval and result ranking.
- Grounded answer generation, including the "nothing matches" case.
- An embedding-model benchmark (proposal deliverable).
- A retrieval-quality evaluation harness that feeds Stage 9 (Maria).

### 1.3 Out of scope (handled by other members)

- Video → events, detection, tracking, captioning (stages 1–6, Juan / Saurabh).
- The Postgres schema itself and raw-video object storage (Gourika / Juan).
- Independent annotation-layer evaluation, stage 7 (Maria).
- React frontend and auth (Nelkit); FastAPI app wiring (Gourika) — we expose
  Python functions / a small router they call.

### 1.4 Boundary / hand-offs

| Direction | Interface | Owner on the other side |
|---|---|---|
| Input | Supabase Postgres tables (events, captions, tracks, geometry, clip metadata) | Juan / Gourika |
| Output → backend | `answer_query(text) -> QueryResult` (ranked events + generated answer + citations) | Gourika (FastAPI) |
| Output → evaluation | Retrieval results serialised in an ActEV-compatible shape | Maria (Stage 9) |

---

## 2. Locked decisions

| Area | Choice | Rationale |
|---|---|---|
| Vector DB | **Qdrant**, local Docker for dev; managed/cluster later if needed | Filterable HNSW keeps recall under selective filters; native named vectors for the embedding comparison; light single container; clean, separable "RAG component" for the thesis. |
| Embedding model (default) | **BAAI/bge-base-en-v1.5** (768-dim) | Free/open (MIT), runs on CPU / Apple Silicon, query/passage instruction prompts, best-documented embedder. |
| Embedding benchmark | bge-base vs **bge-large-en-v1.5** vs **Snowflake/snowflake-arctic-embed-l-v2.0** | Proposal asks for an embedding comparison; near-free once the harness exists. |
| LLM provider | **Deferred** — abstract behind `LLMClient`; bake-off before the Progress Report (27 Sep) | Proposal also leaves this "to be finalised". Candidates: Anthropic Claude, OpenAI GPT, local Qwen/Llama via Ollama. |
| Language / stack | Python 3.11+, `sentence-transformers`, `qdrant-client`, `psycopg` / `supabase-py`, `pydantic`, `typer` CLI, `fastapi` later | Matches team stack. |

---

## 3. Data model

### 3.1 Source records (from Supabase — **confirmed 2026-09-05, live against `bronze`**)

The team's landing schema, inspected directly (`scripts/inspect_schema.py`,
`scripts/explore_bronze.py`). Snapshot at inspection time: 10 videos, 123 events,
151 objects, 214 event↔object links, 27,637 per-frame geometries — Stage-1-scale,
as expected. This is a **bronze (landing) layer**; Juan's team expects to keep
developing it, so we read it through a thin access layer we control, not ad hoc
SQL scattered through our code.

```
bronze.videos
  video_id              text, PK
  video_name            e.g. 2018-03-05.13-15-00.13-20-00.bus.G340  (MEVA clip filename)
  camera_id             e.g. G340
  scene                 site label, e.g. bus | hospital | school
  capture_start_local / capture_end_local   timestamp without tz
  capture_time_zone     currently 'unknown' for all rows — blocks true time-of-day
  duration_seconds, frame_count, fps, frame_width, frame_height
  source_video_path     original local path (not usable by us)
  storage_bucket / storage_path / video_url   Cloudflare R2 (public mp4 URL)
  source_metadata        jsonb, duplicates several of the above
  created_at / updated_at

bronze.events
  event_id              text, PK
  video_id               FK -> videos
  event_name             SHORT FREE-TEXT label, e.g. "Vehicle movement",
                          "Person standing near vehicle" — NOT the MEVA 37-class
                          taxonomy (no underscores, inconsistent casing/wording,
                          e.g. "Vehicle Movement" vs "Vehicle movement" both exist)
  description             longer free-text caption (this is our caption text)
  start_frame / end_frame
  start_seconds / end_seconds   (0–300 range per 5-min clip; use this for filtering)
  source_status           currently 100% 'generated' — no KPF/'truth' rows land
                          here; KPF is read separately at evaluation time, per
                          the proposal's design. activity_source in our schema
                          (§3.2) can just mirror source_status.

bronze.event_objects      pure junction: (event_id, object_id) — many-to-many

bronze.objects
  object_id              text, PK
  video_id                FK -> videos
  source_object_id        original tracker/track id (not globally unique)
  detection_count         = count of this object's geometries (verified 1:1)
  first_seen_seconds / last_seen_seconds
  label_details           jsonb LIST — an object can carry >1 candidate label,
                          e.g. [{label:"car", confidence:{avg,min,max},
                          descriptions:[...], detection_count, spatial_positions:[...]},
                          {label:"truck", ...}]. Track identity is genuinely
                          multi-hypothesis in this pipeline; we should treat
                          label_details[0] (highest detection_count/confidence)
                          as the primary type and keep the rest as alternates.

bronze.geometries
  geometry_id            text, PK
  object_id               FK -> objects
  source_geometry_id      per-frame-local detection index
  frame_index, timestamp_seconds
  label                   single label at that instant (can differ frame-to-frame
                          from the object's "primary" label — class flicker)
  confidence
  spatial_position         coarse 9-cell quadrant string (top-left … bottom-right),
                          not a continuous position
  bounding_box_pixels      jsonb {x_min, x_max, y_min, y_max} — corner form, same
                          family as KPF g0 but different key names/shape; a
                          detected-object label vocabulary that includes stray
                          COCO classes (broccoli, bird, chair — clearly detector
                          noise, not MEVA's 6 track types)
```

**Key differences from our original assumption (kept for the record):**
- There is **no activity field constrained to the 37 MEVA classes** in this
  bronze layer — `event_name`/`description` are free-text, model-generated. This
  actually matches the proposal's own framing (an LVLM annotator producing
  free-text descriptions, reconciled against the 37-class taxonomy later by an
  LLM judge, §3, Table 15) — but it means our query parser cannot filter on
  `activity IN (37 labels)` against this table as-is; see §12.
- Camera/site/time live on `videos`, not `events` — every event needs a join to
  filter by camera or derive time-of-day, and `capture_time_zone` is unresolved
  ('unknown'), so time-of-day is not yet reliably derivable.
- `objects` is multi-label per track (see `label_details`), not one clean type —
  our `object_types` payload field should use the primary label, with alternates
  kept for debugging, not surfaced as hard filters yet.
- `geometries.bounding_box_pixels` uses `{x_min,x_max,y_min,y_max}` — different
  key names from KPF's `g0` but the same corner-coordinate family. Any transform
  we write for spatial data must handle this shape explicitly, not assume KPF's.

**Data-quality finding — duplicate events:** five distinct `event_id`s in the same
video, linked to the *same two* `object_id`s, with identical `event_name`,
`description`, and `start_seconds`/`end_seconds` (260.0–272.0, "Person standing
near vehicle"). This looks like the annotation pipeline emitting the same
real-world event once per overlapping sampling window without de-duplication
(proposal §4.1 describes optionally-overlapping windows). **Flag to Juan/Saurabh**
— left unhandled, this pollutes retrieval (near-duplicate hits in every top-k)
and breaks the event-count-agreement metric (Table 12). Until fixed upstream, our
ingestion should de-duplicate on `(video_id, round(start_seconds), round(end_seconds),
sorted(object_ids))` before embedding, and log how many duplicates it collapsed.

### 3.2 Qdrant point schema

One point per event. Named vectors so multiple embeddings coexist (default model,
benchmark models, and a future sparse vector) without separate collections.

```
point.id       = deterministic UUIDv5 of event_id   (idempotent upsert / re-ingest)
point.vectors  = {
    "caption_bge_base"  : <768>,   # default: embedding of the composed text (§4)
    # added by the benchmark run, not in the v1 ingest path:
    "caption_bge_large" : <1024>,
    "caption_arctic_l"  : <1024>,
    # reserved for hybrid v2:
    "caption_sparse"    : <sparse> # BM25-style over caption tokens
}
point.payload  = {
    "event_id", "clip_id", "camera", "site", "date",
    "start_ts", "end_ts", "start_frame", "end_frame",
    "activity", "activity_source",          # "kpf" | "generated"
    "object_types": [...],
    "actor_track_ids": [...],
    "time_of_day": "morning|afternoon|evening|night",   # derived from clip start time
    "caption": "<raw caption text>",
    "composed_text": "<what we embedded>",   # stored for debugging / eval
    "ingested_at", "embed_model"
}
```

Payload indexes (Qdrant) on: `camera`, `site`, `activity`, `activity_source`,
`object_types`, `time_of_day`, `start_ts`, `end_ts`, `date`. These are the fields
the query parser filters on.

### 3.3 Why store the composed text and raw caption in the payload

- Debuggability: we can see exactly what was embedded for any hit.
- The answer generator uses the **raw caption**, not the composed string.
- The evaluation harness needs `activity`, `start_ts/end_ts`, `actor_track_ids`,
  `clip_id` to match retrieved events against the reference set.

---

## 4. Text composition (what we embed)

We do **not** embed the bare caption. We embed a templated composition so that
structured-attribute queries ("a vehicle turning left at the school gate in the
morning") retrieve well regardless of embedding model.

**Template:**

```
{caption}
Activity: {activity_label_human}. Objects: {object_types}.
Camera: {camera} ({site}). Time of day: {time_of_day}.
```

**Worked example:**

- Raw caption: *"A man in a white shirt gets out of a white sedan and walks toward
  the building entrance."*
- Event metadata: `activity = person_exits_vehicle`, `object_types = [person,
  vehicle]`, `camera = G336`, `site = school`, `time_of_day = afternoon`.
- Composed text:

  > A man in a white shirt gets out of a white sedan and walks toward the building
  > entrance.
  > Activity: person exits vehicle. Objects: person, vehicle.
  > Camera: G336 (school). Time of day: afternoon.

`activity_label_human` = the MEVA label with underscores replaced by spaces; for
out-of-vocabulary generated activities we use the caption's own phrasing.

For **queries**, we embed the semantic string the LLM produces (see §5) with the
model's query prompt (`bge` uses an instruction prefix for queries). Passage and
query prompts must match the model — encapsulated in the `EmbeddingModel` wrapper.

---

## 5. Query flow (Stage 8)

```
NL query
  │
  ▼
LLMClient.parse(query) ─────────────► QuerySpec
  │                                     ├─ semantic_text: str          (for vector search)
  │                                     ├─ cameras: list[str] | None
  │                                     ├─ sites: list[str] | None
  │                                     ├─ activities: list[str] | None (mapped to the 37 labels)
  │                                     ├─ object_types: list[str] | None
  │                                     ├─ time_window: {start, end} | None
  │                                     ├─ time_of_day: list[str] | None
  │                                     ├─ ordering: list[OrderingConstraint]  ("A after B")
  │                                     └─ expect_empty: bool          (negative-query hint)
  ▼
build Qdrant filter  ◄── payload fields from QuerySpec
  │
  ▼
Qdrant filtered vector search (named vector = default model), top-N (N ≈ 50)
  │
  ▼
temporal post-processing  ── if ordering constraints: keep event pairs/sequences
  │                            where event B's start_ts > event A's end_ts on the
  │                            same camera (or across cameras per constraint)
  ▼
rank + truncate to top-k (k ≈ 10)
  │
  ├───────────────► QueryResult.events   (to backend / frontend timeline)
  ▼
LLMClient.generate_answer(query, retrieved_events)
  │   context = raw captions + activity + clip_id + timespan for each event
  │   instruction: answer only from context; cite clip_id + [start_ts–end_ts];
  │                if context is empty or irrelevant, say nothing matches
  ▼
QueryResult.answer  +  QueryResult.citations
```

### 5.1 QuerySpec parsing notes

- The LLM is given the **fixed vocabularies** in the prompt: 37 activity labels,
  6 object types, the camera / site list (queried once from Postgres and cached).
  It must map free text to these ("someone got in a car" → `person_enters_vehicle`)
  or return `null` for a dimension it can't ground.
- Output is validated against a pydantic `QuerySpec`; on validation failure we
  retry once with the error, then fall back to pure semantic search with no
  filters.
- `expect_empty` lets the answer stage be stricter about not inventing a result.

### 5.2 Negative queries

The evaluation set includes queries whose correct answer is "nothing matches"
(proposal §4.3.3). Handling:

- If the filter yields zero candidates, or all similarity scores are below a
  calibrated floor, return an empty `events` list and an answer stating no
  matching footage was found.
- The similarity floor is tuned on the development split only, never the held-out
  10%.

### 5.3 Temporal-ordering queries

"Did anyone enter the building after the white van arrived?" →
two sub-events (`vehicle` arriving, `person_enters_scene_through_structure`), an
ordering constraint between them. We retrieve candidates for each sub-event, then
keep combinations that satisfy `B.start_ts > A.end_ts` within a time budget. This
mirrors the compositional queries the proposal targets (§2.1.5).

---

## 6. Interfaces (Python)

```python
class EmbeddingModel(Protocol):
    name: str
    dim: int
    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...

class VectorStore(Protocol):
    def ensure_collection(self, vectors: dict[str, VectorParams]) -> None: ...
    def upsert(self, points: list[EventPoint]) -> None: ...
    def search(self, vector_name: str, query_vec: list[float],
               flt: Filter | None, limit: int) -> list[ScoredEvent]: ...
    def count(self) -> int: ...

class LLMClient(Protocol):
    def parse(self, query: str, vocab: Vocab) -> QuerySpec: ...
    def generate_answer(self, query: str, events: list[Event]) -> Answer: ...
```

- `VectorStore` is implemented by `QdrantStore`. A `PgVectorStore` remains a
  possible alternative implementation if Qdrant proves awkward — the interface
  keeps that option open at low cost.
- `LLMClient` is implemented per provider (`ClaudeClient`, `OpenAIClient`,
  `OllamaClient`); provider chosen by config.

---

## 7. Hybrid (keyword + semantic) search

**Decision: defer to v2.** Rationale:

- The structured dimensions (camera, activity, object type, time) are exact-match
  **filters**, not keyword search — already handled.
- Pure keyword value is limited to rare tokens in free-text captions (specific
  colours, "backpack" vs "duffel"). bge dense retrieval handles most of this.
- Qdrant supports a sparse named vector, and the point schema (§3.2) already
  reserves `caption_sparse`, so adding BM25 + fusion later is additive, not a
  migration.

Revisit after the first retrieval evaluation if recall on descriptive queries is
weak.

---

## 8. Embedding-model benchmark

**Goal:** report, for the Progress / Final report, which open embedding model best
serves surveillance-caption retrieval.

- **Models:** bge-base-en-v1.5 (default), bge-large-en-v1.5, arctic-embed-l-v2.0.
  All free, all local.
- **Query set:** built from the audited reference set — for a sample of events,
  write 2–3 natural-language queries whose known-correct answer is that event (and
  a set of negative queries). Target ~150–200 labelled query→event pairs. Shared
  with Maria so it aligns with the Stage 9 evaluation query set.
- **Procedure:** embed all events with each model into its own named vector; run
  each query against each model's vector; no metadata filter (isolate embedding
  quality).
- **Metrics:** Recall@{1,5,10,20}, MRR, nDCG@10. Report per query category
  (single-activity, descriptive, negative).
- **Also report:** index size, embedding throughput on CPU and on Apple Silicon.
- **Dimension lock-in:** the production collection commits to one model's
  dimension. Changing the default model later = full re-embed + re-index. The
  benchmark vectors are added alongside, so the comparison itself needs no
  re-ingest.

---

## 9. Evaluation hooks (feeds Stage 9 — Maria)

Retrieval results are serialised so Maria's ActEV-style scoring can consume them
without adapting to our internals:

```
retrieval_run.json
  query_id
  query_text
  retrieved: [
    { clip_id, activity, start_ts, end_ts, start_frame, end_frame,
      actor_track_ids, camera, score, rank }
  ]
```

- Matching to the reference set is done on Maria's side using the proposal's rule
  (label agreement + temporal IoU ≥ 0.5, Hungarian assignment — Table 11).
- We provide `activity_source` on every event so annotation-layer failures
  ("known FN for this class") can be separated from retrieval failures
  (proposal §4.3.4, Table 17).
- Metrics we own the plumbing for: precision / recall / F1 of retrieved activity
  instances, plus retrieval-only measures (Recall@k, MRR) for our own tuning.

---

## 10. Configuration and secrets

```
.env                         # git-ignored
  DATABASE_URL=              # Supabase Postgres connection string (read-only role)
  # or SUPABASE_URL / SUPABASE_KEY
  QDRANT_URL=http://localhost:6333
  QDRANT_API_KEY=            # empty for local
  EMBED_MODEL=BAAI/bge-base-en-v1.5
  LLM_PROVIDER=              # deferred
  LLM_API_KEY=
```

- `config.py` uses `pydantic-settings`; no secret is ever committed or logged.
- `.env.example` committed with blank values.
- Local Qdrant via `docker compose up qdrant` (see repo `docker-compose.yml`).

---

## 11. Local development setup

```bash
cd rag
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env          # fill in DATABASE_URL from Juan
docker compose up -d qdrant   # Qdrant on :6333
rag inspect-schema            # dump Supabase schema
rag fetch-sample --limit 20   # eyeball real events + captions
```

---

## 12. Open questions (for team review)

**Resolved by direct inspection (2026-09-05):**
- ~~Exact table/column names~~ — see §3.1. Five tables in `bronze`: `videos`,
  `events`, `objects`, `geometries`, `event_objects`. No pre-joined view; we join
  in our access layer.
- ~~Read-only role vs REST API~~ — we were handed a direct Postgres pooler
  connection string. **Still to confirm with Juan/Gourika:** should we get our
  *own* read-only role instead of sharing the same credential everyone uses, and
  is this bronze schema expected to be stable enough to build on, or will tables
  be renamed/restructured as they iterate (it's explicitly a landing layer)?

**Still open, now more specific:**
1. **Juan:** `events.event_name` / `description` are free text, not the 37-class
   MEVA taxonomy (§3.1). Will a controlled activity label land in `bronze` (or a
   later `silver`/`gold` schema) before we build the query filter, or should our
   query parser match against free text / an LLM-derived label from day one? This
   materially changes Stage 8's filter design.
2. **Juan/Saurabh:** the duplicate-event finding in §3.1 — same object pair, same
   timespan, 5 separate `event_id`s. Can you confirm whether this is the
   overlapping-window sampling emitting the same event repeatedly, and is a fix
   planned upstream, or should we de-duplicate defensively on ingest?
3. **Juan:** `videos.capture_time_zone = 'unknown'` for every row so far — will
   real timezone data land later? We need it (or a documented assumption) for
   `time_of_day` filters.
4. **Juan:** is there a plan for a `silver`/`gold` schema on top of `bronze`
   (naming strongly suggests medallion architecture)? If so, should we target
   that instead once it exists, to avoid rebuilding our access layer against a
   moving target?
5. **Maria:** confirm the `retrieval_run.json` shape in §9 works for Stage 9, and
   let's co-own the labelled query set (§8) so it serves both the embedding
   benchmark and the retrieval evaluation.
6. **Team:** shared monorepo or a separate `rag` repo? Where does this `rag/`
   folder live long-term? (It's currently its own local git repo, not yet pushed
   anywhere.)
7. **Team / Ali:** any preference or constraint on LLM provider (cost, data
   residency, "must be reproducible / offline")?
8. Do we need cross-camera identity (same person across cameras) for v1 temporal
   queries, or is single-camera ordering enough to start? Current `bronze` data
   shows no cross-camera object linkage — `object_id` is scoped to one `video_id`.

---

## 13. Milestone alignment

| Week | Date | This component |
|---|---|---|
| 6 | 31 Aug – 6 Sep | Design doc (this); repo skeleton; Supabase connection + first fetch |
| 7 | 7–13 Sep | Ingestion pipeline (Supabase → compose → bge-base → Qdrant) |
| 8 | 14–20 Sep | Query parser + filtered retrieval + answer generation, end to end |
| 9 | 21–27 Sep | **Progress Report**: working RAG loop + LLM provider chosen |
| 10 | 28 Sep – 4 Oct | Embedding benchmark; retrieval evaluation with Maria; tuning |
| 11–12 | 5–18 Oct | Hybrid search if needed; final evaluation; Final Report + presentation |
