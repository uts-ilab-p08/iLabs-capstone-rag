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
  "query": "person getting out of a car",
  "answer": "natural-language answer",
  "sources": [
    {"score": 0.788, "description": "...", "event_name": "...", "video_id": "...",
     "video_name": "...", "camera_id": "G336", "scene": "school",
     "start_seconds": 40.0, "end_seconds": 52.0, "object_types": ["car", "person"],
     "video_url": "https://..."}
  ]
}
```

No match → `"sources": []` and an answer saying no footage was found.
`answer` is placeholder text until the LLM provider is chosen; its shape won't change.

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
