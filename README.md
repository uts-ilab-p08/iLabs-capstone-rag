# RAG layer — interactive surveillance video querying

Vector index and natural-language retrieval over the MEVA event annotations in
Supabase. Part of capstone project 08-01.

Design notes and schema map: [`docs/rag-design.md`](docs/rag-design.md).

## Setup

```bash
uv sync
cp .env.example .env
```

`DATABASE_URL` is needed only for indexing. Answering questions uses Qdrant and
the LLM alone, so a query-only deployment needs no database credentials.

Qdrant is chosen by `QDRANT_URL` in `.env`:

- **a hosted cluster URL** — what the project uses, so the backend and your
  laptop share one index. The API key goes in `QDRANT_API_KEY`.
- **`http://localhost:6333`** — the container in `docker-compose.yml`
  (`docker compose up -d qdrant`), with a dashboard at `/dashboard`.
- **empty** — runs embedded against `./qdrant_data`, no Docker. Single process
  only, and payload indexes are ignored, so filtering falls back to scanning.

## Use

```bash
uv run scripts/index_events.py                            # build the index
uv run scripts/search.py "person getting out of a car"    # search it
uv run scripts/search.py "white truck" --limit 10
```

`index_events.py` is safe to re-run — point IDs are derived from `event_id`, so a
second run overwrites rather than duplicates. The index is derived state and can
be deleted and rebuilt at any time.

## Using it from the backend

The backend only needs one function. It owns all HTTP; this package never does.

```python
from rag.pipeline import answer_query

query = ...                      # INPUT: question text from the query API
response = answer_query(query)   # OUTPUT: dict to hand to the response API
```

`response` is JSON-serialisable:

```json
{
  "query": "anything at the hospital on March 5 between 1pm and 2pm?",
  "answer": "A white SUV left the parking area at 13:16 [1]...",
  "sources": [
    {
      "score": 0.649,
      "annotation": "A person in a dark jacket is seen exiting a white SUV...",
      "video_id": "20c38d5f...",
      "event_id": "b159b039..."
    }
  ],
  "filters": {
    "scenes": ["hospital"],
    "cameras": [],
    "dates": ["2018-03-05"],
    "time_of_day": [46800, 50400],
    "notes": []
  }
}
```

- `sources` is the best `TOP_K` events, best first. Anything beyond these four
  fields is looked up in Postgres by `event_id` — location, camera and
  timestamps exist internally, because the prompt is built from them, but are
  trimmed here.
- `annotation` is the caption — bronze's `events.description`.
- `answer` cites the events it used as `[1]`, `[2]`, matching `sources` order.
  Citations are not guaranteed: weaker models sometimes omit them, so the
  frontend must not depend on their presence.
- No match → `"sources": []` and an answer saying no footage was found.
- If the LLM is unreachable, `answer` degrades to a plain summary rather than
  raising, so the endpoint never fails because of the model.
- `filters` is diagnostic: what was filtered on, with `time_of_day` as seconds
  since midnight, plus notes on anything deliberately ignored. Safe to drop, but
  useful for telling the user "searched hospital only".

## Metadata filtering

A question naming a location, camera, date or time of day is filtered before
ranking. Several combine: *"anything at the hospital on March 5 between 1pm and
2pm"* applies all three at once.

| Names | Filters on | Understands |
|---|---|---|
| a location | `scene` | `school`, plus synonyms `campus`, `clinic`, `depot` |
| a camera | `camera_id` | `G341`, several at once |
| a date | `capture_date` | `2018-03-07`, `March 7`, `7 March`, `the 7th` |
| a time | `start_time_of_day` | `2-4pm`, `between 13:00 and 14:00`, `after 2pm`, `before 11am`, `at 2pm` |

All rule-based — no model call, so it adds no latency and is deterministic.

The vocabulary of locations, cameras and dates is read from Qdrant rather than
Postgres, because it must describe what is *searchable*: a location that exists
upstream but is not yet indexed would otherwise become a filter matching
nothing. It is read fresh on every query, so re-indexing makes a new location
filterable immediately, with no restart.

Time of day has its own field rather than reusing a timestamp range, because
"2-4pm" across two days is two disjoint windows. `start_seconds` is an offset
inside a clip and cannot answer "what happened at 2pm".

Every rule errs toward filtering on nothing, because a wrong filter reports
"no matching footage" about footage that exists, while a missed filter merely
leaves results slightly noisy. So these all fall back to unfiltered search and
say why in `filters.notes`:

- **ambiguous location** — "was there a school bus?" matches two locations
- **negation** — "anything except the school"
- **ambiguous time** — "2-4" could be 02:00 or 14:00; only an explicit `am`/`pm`
  or an hour of 13+ is unambiguous
- **ambiguous date** — "the 7th" when several indexed dates fall on a 7th

A fully written date that *isn't* in the index still filters, giving no results —
"nothing was recorded that day" is the honest answer.

**When a filter matches, the similarity floor is dropped.** The filter is then
the relevance signal. This is what makes scope questions work: no single event
resembles "what happened in this clip", so every hospital event scores ~0.45-0.50
and a fixed floor would discard all of them.

Object types are deliberately **not** filtered on — the same vehicle appears as
`car` in one event and `truck` in another, so filtering would drop correct results.

## Choosing the LLM

OpenRouter and Ollama both speak the OpenAI API, so the model is configuration,
not code. Set three values in `.env`:

| | OpenRouter (hosted) | Ollama (local) |
|---|---|---|
| `LLM_BASE_URL` | `https://openrouter.ai/api/v1` | `http://localhost:11434/v1` |
| `LLM_MODEL` | `<vendor>/<model>` | `qwen3:4b` |
| `LLM_API_KEY` | `sk-or-...` | leave empty |

Leave `LLM_MODEL` empty to skip the LLM; answers fall back to a plain summary.

Reasoning models spend tokens thinking before answering, so `max_tokens` is set
generously (2000). Too low and they hit the cap mid-thought and return nothing.

## Tuning how many results come back

`TOP_K` (default 5) sets how many events a query returns, and how many the LLM
sees as context. It is read from the environment, so a deployment can change it
without a new release — set it in `.env` locally, or as an environment variable
on the host.

The retrieval evaluation found that a correct result almost always ranks first,
so 5 is generous rather than tight. Raising it mostly adds context for the LLM
to read, at proportional cost in tokens and latency.

To try it without the backend: `uv run scripts/ask.py "your question"`.

The backend installs this package straight from a git tag in its
`requirements.txt` (distribution name `ilabs-cctv-rag`, module name `rag`):

```
ilabs-cctv-rag @ git+https://github.com/uts-ilab-p08/iLabs-capstone-rag.git@vX.Y.Z
```

## Publishing a new version

A version is just a git tag the backend pins to. There is no PyPI upload — pip
clones the repo at that tag and builds the wheel itself, so **whatever is in the
tagged commit is exactly what gets installed**.

1. **Bump `version` in `pyproject.toml`** and refresh the lock file. Keep it equal
   to the tag you are about to create (tag `v0.1.3` ↔ `version = "0.1.3"`), so
   `pip show ilabs-cctv-rag` on the backend reports the real version.

   ```bash
   uv lock
   ```

2. **Check it installs from outside the repo.** `uv run` imports straight from
   `src/` and never builds a wheel, so it can't catch packaging mistakes — this
   can:

   ```bash
   rm -rf dist && uv build
   python -m venv /tmp/rag-check && /tmp/rag-check/bin/pip install dist/*.whl
   DATABASE_URL=postgresql://x:y@localhost/z \
     /tmp/rag-check/bin/python -c "from rag.pipeline import answer_query; print('RAG package OK')"
   ```

   (`DATABASE_URL` is no longer needed for this check — it is optional, and only
   indexing reads Postgres. Setting it does no harm.)

3. **Commit and push first, then tag.** A tag points at a commit, not at your
   working tree: tagging before committing publishes the *previous* commit.

   ```bash
   git add pyproject.toml uv.lock   # plus whatever changed
   git commit -m "chore(release): v0.1.3"
   git push origin main
   git tag v0.1.3
   git push origin v0.1.3
   ```

4. **Confirm the tag points at your commit** before telling anyone about it:

   ```bash
   git ls-remote --tags origin      # v0.1.3 must show the same hash as:
   git rev-parse HEAD
   ```

5. **Bump the pin in the backend** (`requirements.txt`: `@v0.1.2` → `@v0.1.3`)
   and reinstall there with `pip install -r requirements.txt`. Render picks the
   new version up on its next deploy.

Never move or delete a tag that's already published — the backend (or someone's
venv) may already be pinned to it. If a release is broken, publish the next
patch version instead.

If a change alters the vector size (a different `EMBED_MODEL`/`EMBED_DIM`) or
what gets indexed, the Qdrant collection has to be rebuilt with
`scripts/index_events.py` against the remote Qdrant **before** the backend is
bumped to that version.

## Layout

```
src/rag/
  config.py   settings from .env
  db.py       fetch events from Supabase bronze schema
  text.py     compose the string that gets embedded
  embed.py    bge-base-en-v1.5 via fastembed (ONNX, no PyTorch)
  store.py    Qdrant collection, payload schema, upsert, search
  filters.py  pull location and camera filters out of a question
  temporal.py pull dates and clock ranges out of a question
  llm.py      one client for OpenRouter and Ollama
  pipeline.py answer_query(query) -> response   <- what the backend calls
scripts/
  index_events.py    fetch -> compose -> embed -> store
  ask.py             run answer_query() on one question, print the response
  search.py          raw ranked search results, for debugging retrieval
  inspect_schema.py  read-only: list tables, columns, row counts
  explore_bronze.py  read-only: distincts, joins, sample JSON
```

## Not built yet

Temporal-ordering queries ("what happened after the van arrived"), hybrid
keyword-plus-vector search, LLM-based filter extraction, whole-clip
chronological summarisation, the embedding-model benchmark, the retrieval
evaluation harness, and automated tests. See `docs/rag-design.md`.
