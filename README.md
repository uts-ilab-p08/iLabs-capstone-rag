# RAG layer — interactive surveillance video querying

Vector index and natural-language retrieval over the MEVA event annotations in
Supabase. Part of capstone project 08-01.

Design notes and schema map: [`docs/rag-design.md`](docs/rag-design.md).

## Setup

```bash
uv sync
cp .env.example .env     # fill in DATABASE_URL
```

Qdrant runs one of two ways, decided by `QDRANT_URL` in `.env`:

- **empty** — runs embedded against `./qdrant_data`. No Docker needed.
- **`http://localhost:6333`** — connects to the container in `docker-compose.yml`
  (`docker compose up -d qdrant`). Adds the dashboard at
  <http://localhost:6333/dashboard> and lets several processes share one index.

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
  "query": "did anyone get out of a vehicle?",
  "answer": "Yes, a person got out of a vehicle in events [1], [2] and [3]...",
  "sources": [
    {
      "score": 0.649,
      "annotation": "A person in a dark jacket is seen exiting a white SUV...",
      "video_id": "20c38d5f...",
      "event_id": "b159b039...",

      "event_name": "Person exits a white SUV",
      "video_name": "2018-03-05.13-20-00.13-25-00.school.G336",
      "camera_id": "G336",
      "scene": "school",
      "start_seconds": 41.844,
      "end_seconds": 52.306,
      "object_types": ["car", "person"],
      "video_url": "https://..."
    }
  ],
  "filters": {"scenes": ["hospital"], "cameras": [], "notes": []}
}
```

- `sources` holds at most 5 events, best first. The first four fields are the
  agreed contract; the rest are included because the frontend needs them to play
  the exact moment without querying Postgres again. Drop them if unwanted.
- `annotation` is the caption — bronze's `events.description`.
- `answer` cites the events it used as `[1]`, `[2]`, matching `sources` order.
- No match → `"sources": []` and an answer saying no footage was found.
- If the LLM is unreachable, `answer` degrades to a plain summary rather than
  raising, so the endpoint never fails because of the model.
- `filters` is diagnostic: which metadata filter was applied and anything
  deliberately ignored. Safe to drop from the API response, but useful for
  telling the user "searched hospital only".

## Metadata filtering

A question naming a location or camera is filtered before ranking, so
"what happened in the hospital clip" searches only hospital footage.

Rule-based against the vocabulary actually present in the index (read from
Qdrant, not Postgres — it must describe what is *searchable*). Handles
synonyms (`campus`, `clinic`, `depot`) and rejects camera IDs that do not exist.

Every rule errs toward filtering on nothing, because a wrong filter reports
"no matching footage" about footage that exists, while a missed filter merely
leaves results slightly noisy. So an ambiguous question ("was there a school
bus?") and a negated one ("anything except the school") both fall back to
unfiltered search, and say so in `filters.notes`.

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

To try it without the backend: `uv run scripts/ask.py "your question"`.

## Layout

```
src/rag/
  config.py   settings from .env
  db.py       fetch events from Supabase bronze schema
  text.py     compose the string that gets embedded
  embed.py    bge-base-en-v1.5 via fastembed (ONNX, no PyTorch)
  store.py    Qdrant collection, upsert, search
  pipeline.py answer_query(query) -> response   <- what the backend calls
scripts/
  index_events.py    fetch -> compose -> embed -> store
  ask.py             run answer_query() on one question, print the response
  search.py          raw ranked search results, for debugging retrieval
  inspect_schema.py  read-only: list tables, columns, row counts
  explore_bronze.py  read-only: distincts, joins, sample JSON
```

## Not built yet

LLM query parsing into metadata filters, answer generation, temporal-ordering
queries, hybrid search, the embedding-model benchmark, and the retrieval
evaluation harness. See `docs/rag-design.md`.
