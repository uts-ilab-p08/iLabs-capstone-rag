"""The single entry point the backend calls: a question in, a response out.

    from rag.pipeline import answer_query
    response = answer_query(query)

The backend API owns everything HTTP. Nothing here knows about requests,
endpoints or users — it takes a plain string and returns a plain,
JSON-serialisable dict.
"""

from __future__ import annotations

import atexit
from functools import lru_cache

from qdrant_client import QdrantClient

from rag import store
from rag.embed import Embedder

TOP_K = 5

# Hits below this cosine similarity are treated as unrelated and dropped.
# Chosen from early testing: real matches scored ~0.72–0.82, a nonsense query
# ~0.50–0.52. Recalibrate once we have a labelled query set.
MIN_SCORE = 0.60


# Loading the model takes a few seconds and opening Qdrant holds a file lock, so
# both are created once and reused for every question, not rebuilt per request.
@lru_cache(maxsize=1)
def _embedder() -> Embedder:
    return Embedder()


@lru_cache(maxsize=1)
def _client() -> QdrantClient:
    client = store.connect()
    # Close cleanly when the process exits, instead of during interpreter
    # teardown (which prints a harmless but alarming traceback).
    atexit.register(client.close)
    return client


def _mmss(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


def retrieve(query: str, limit: int = TOP_K) -> list[dict]:
    """Embed the question and return the matching events, best first."""
    vector = _embedder().embed_query(query)
    hits = store.search(_client(), vector, limit=limit)

    sources = []
    for hit in hits:
        if hit.score < MIN_SCORE:
            continue
        p = hit.payload
        sources.append(
            {
                "score": round(hit.score, 3),
                "description": p.get("description"),
                "event_name": p.get("event_name"),
                "video_id": p.get("video_id"),
                "video_name": p.get("video_name"),
                "camera_id": p.get("camera_id"),
                "scene": p.get("scene"),
                "start_seconds": p.get("start_seconds"),
                "end_seconds": p.get("end_seconds"),
                "object_types": p.get("object_types") or [],
                "video_url": p.get("video_url"),
            }
        )
    return sources


def generate_answer(query: str, sources: list[dict]) -> str:
    """Turn retrieved events into a natural-language answer.

    PLACEHOLDER: the LLM provider has not been chosen yet, so this builds a
    plain summary from the best match. Replace the body with an LLM call once
    the provider is decided — the signature and return type stay the same, so
    nothing that calls answer_query() needs to change.
    """
    if not sources:
        return "No matching footage was found for this question."

    best = sources[0]
    return (
        f"Found {len(sources)} matching moment(s). Closest match: {best['description']} "
        f"(camera {best['camera_id']}, clip {best['video_name']}, "
        f"{_mmss(best['start_seconds'])}–{_mmss(best['end_seconds'])})."
    )


def answer_query(query: str) -> dict:
    """Run the full RAG flow for one question.

    Returns:
        {
          "query":   the question as received,
          "answer":  natural-language answer (placeholder text until the LLM is added),
          "sources": the events the answer is based on, best first — caption,
                     camera, clip, timespan, objects and a playable video_url
        }
    """
    query = (query or "").strip()
    if not query:
        return {"query": query, "answer": "Please enter a question.", "sources": []}

    sources = retrieve(query)
    return {"query": query, "answer": generate_answer(query, sources), "sources": sources}
