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

   (`DATABASE_URL` only needs to be set, not valid — `rag.config` reads it at
   import time.)

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
