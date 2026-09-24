"""The single entry point the backend calls: a question in, a response out.

    from rag.pipeline import answer_query
    response = answer_query(query)

The backend API owns everything HTTP. Nothing here knows about requests,
endpoints or users — it takes a plain string and returns a plain,
JSON-serialisable dict.
"""

from __future__ import annotations

import atexit
import logging
from datetime import datetime, timedelta
from functools import lru_cache

from qdrant_client import QdrantClient

from rag import filters, llm, store
from rag.embed import Embedder

log = logging.getLogger(__name__)

TOP_K = 5

# Hits below this cosine similarity are treated as unrelated and dropped.
# Chosen from early testing: real matches scored ~0.72–0.82, a nonsense query
# ~0.50–0.52. Recalibrate once we have a labelled query set.
MIN_SCORE = 0.6 #keep as 0.60


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


def retrieve(query: str, limit: int = TOP_K) -> tuple[list[dict], filters.FilterSpec]:
    """Embed the question and return (matching events best first, filter used)."""
    client = _client()
    spec = filters.extract(query, *filters.vocabulary(client))
    query_filter = filters.to_qdrant_filter(spec)

    vector = _embedder().embed_query(query)
    hits = store.search(client, vector, limit=limit, query_filter=query_filter)

    # A similarity floor only makes sense for unfiltered search. Once a filter
    # has matched, the filter *is* the relevance signal: the user asked for the
    # hospital, so hospital events are relevant even when they read nothing like
    # the question. Scope questions ("what happened in the hospital clip") score
    # low against every event because no single event resembles a summary
    # request, and a fixed floor would throw all of them away.
    floor = 0.0 if query_filter is not None else MIN_SCORE
    if spec.notes or not spec.is_empty:
        log.info("query filter: %s", spec.describe())

    sources = []
    for hit in hits:
        if hit.score < floor:
            continue
        p = hit.payload
        sources.append(
            {
                # The four fields the backend contract requires, first.
                # "annotation" is the caption, bronze's events.description.
                "score": round(hit.score, 3),
                "annotation": p.get("description"),
                "video_id": p.get("video_id"),
                "event_id": p.get("event_id"),
                # Extras, already available and useful to the frontend for
                # playing the exact moment without a round-trip to Postgres.
                "event_name": p.get("event_name"),
                "video_name": p.get("video_name"),
                "camera_id": p.get("camera_id"),
                "scene": p.get("scene"),
                "start_seconds": p.get("start_seconds"),
                "end_seconds": p.get("end_seconds"),
                "object_types": p.get("object_types") or [],
                "video_url": p.get("video_url"),
            }
        )
    return sources, spec


NO_MATCH = "No matching footage was found for this question."

# Bump when the prompt changes. Evaluation results are only comparable within
# the same version, so record it alongside any numbers we report.
PROMPT_VERSION = "2026-09-24.v5"

_SYSTEM_PROMPT = """You answer questions about CCTV surveillance footage.

You are given numbered events found by searching the footage. Each event is one
camera's description of one moment, with its location, camera and timestamp.

JUDGE BY LOCATION AND CLOCK TIME, NOT BY CAMERA NAME.
Several cameras watch each location from different angles and their clocks are
synchronised, so the camera name alone tells you nothing. Use the timestamps.

- Same location, overlapping times: almost certainly ONE real occurrence, either
  filmed from two angles or described more than once. Report it once. Never turn
  one person into two because two cameras saw them.
- Same location, clearly different times: separate occurrences.
- Different locations: different objects and different occurrences, however
  alike the descriptions sound. Two white vehicles at two locations are two
  vehicles. Never join them into one journey.

COUNTING.
Only give a number when the times and locations clearly support it. If you
cannot tell whether two events are the same occurrence, say so instead of
counting them.

ANSWER FROM EVENT [1] FIRST.
Event [1] is the closest match to the question and is the one the user sees
first. Lead with what [1] shows. Mention other events only afterwards, and only
as separate, additional sightings.

OTHER RULES.
- Answer the question that was asked. Do not simply retell an event's
  description back to the user.
- Use only the events provided. Never invent events, objects, people or details.
- Cite each event you use by its number, like [1] or [3].
- The detector labels are automatic and frequently wrong: one vehicle may be
  labelled "car" in one event and "truck" in another. Trust the written
  description over the label, and never treat disagreeing labels as meaningful.
- The descriptions themselves are machine-generated and may be imprecise. Do not
  overstate what they establish.
WHEN THE ANSWER IS NO.
If the events do not contain what was asked about, say so in ONE sentence and
stop. Do not go on to describe what the events show instead, do not explain how
you checked, and do not speculate about what the footage means. The events are
already on screen beside your answer; the user can look. A long answer to a
question with nothing to report is just room for mistakes.
Good: "No helipad appears in any of the hospital footage."
Bad: "No, there is no evidence of a helipad. All five events describe regular
pedestrian and vehicle activity around the hospital building..."

STYLE.
The user is reviewing footage, not reading an essay. Be direct.
- Open with the answer itself. Never begin with a preamble such as "Based on the
  retrieved events" or "According to the footage".
- Each event is already shown on screen beside your answer, with its location,
  camera and full timestamp. Do not repeat that metadata in prose. Cite the
  number and let the user look.
- Give clock times only, such as 13:16. Never write a calendar date in your
  answer, even if one appears in the event data.
- Two to four sentences. If you are reporting several distinct occurrences, a
  short bulleted list is clearer: one line each, no sub-points.
- Stop when you have answered. Do not add a closing sentence that restates what
  you just said."""


def absolute_window(video_name: str, start: float | None, end: float | None):
    """Clip-relative seconds -> absolute datetimes, from the clip filename.

    MEVA clips are named e.g. 2018-03-05.13-20-00.13-25-00.school.G424, so the
    recording start is in the name and an offset within the clip gives the real
    time. The cameras are synchronised to GPS time, which is what makes times
    from different cameras directly comparable — and comparing them is the only
    way to tell one event filmed twice from two separate events.

    Returns (None, None) if the name does not parse, so callers can fall back.
    """
    try:
        date_part, start_part, *_ = video_name.split(".")
        base = datetime.strptime(f"{date_part} {start_part}", "%Y-%m-%d %H-%M-%S")
    except (ValueError, AttributeError):
        return None, None
    if start is None or end is None:
        return None, None
    return base + timedelta(seconds=start), base + timedelta(seconds=end)


def _format_events(sources: list[dict]) -> str:
    """Render events for the prompt.

    Location and absolute time lead each entry, because those are what the model
    must compare to decide whether two events are the same occurrence. Camera
    comes last: several cameras cover each location, so the camera name is the
    least informative field and reading it first invites wrong conclusions.
    """
    windows = [
        absolute_window(s["video_name"], s["start_seconds"], s["end_seconds"]) for s in sources
    ]
    # Show the date only when the events actually span more than one, which is
    # the only time it helps decide whether two events are the same occurrence.
    # Otherwise it is noise the model tends to copy into its answer.
    dates = {first.date() for first, _ in windows if first}
    show_date = len(dates) > 1

    lines = []
    for i, (s, (first, last)) in enumerate(zip(sources, windows), start=1):
        marker = "  <- closest match" if i == 1 else ""
        labels = ", ".join(s["object_types"]) or "none"
        if first and last:
            stamp = "%Y-%m-%d %H:%M:%S" if show_date else "%H:%M:%S"
            when = f"{first:{stamp}} to {last:%H:%M:%S}"
        else:
            when = f"{_mmss(s['start_seconds'])}-{_mmss(s['end_seconds'])} into the clip"
        lines.append(
            f"[{i}] location: {s['scene']} | time: {when}{marker}\n"
            f"    description: {s['annotation']}\n"
            f"    camera: {s['camera_id']} (one of several covering {s['scene']})\n"
            f"    detector labels (unreliable): {labels}"
        )
    return "\n\n".join(lines)


def _fallback_answer(sources: list[dict], spec: filters.FilterSpec) -> str:
    """Used when no model is configured or the model call fails.

    The backend should still get a usable response rather than an error, so a
    failed LLM degrades to a plain summary instead of breaking the endpoint.

    The wording depends on how these events were selected. When a metadata
    filter supplied them there is no similarity floor, so they are events from
    the right place rather than events that answer the question — claiming a
    "match" would be false for something like "is there a helicopter at the
    hospital?", where every hospital event comes back and none is relevant.
    """
    best = sources[0]
    where = f" ({spec.describe()})" if not spec.is_empty else ""
    lead = (
        f"Showing {len(sources)} event(s) from the footage you asked about{where}. "
        if not spec.is_empty
        else f"Found {len(sources)} matching moment(s). "
    )
    return (
        f"{lead}First result: {best['annotation']} "
        f"(camera {best['camera_id']}, clip {best['video_name']}, "
        f"{_mmss(best['start_seconds'])}–{_mmss(best['end_seconds'])})."
    )


def generate_answer(query: str, sources: list[dict], spec: filters.FilterSpec) -> str:
    """Turn retrieved events into a natural-language answer."""
    if not sources:
        # Deterministic, and saves a pointless model call.
        return NO_MATCH

    if not llm.is_configured():
        return _fallback_answer(sources, spec)

    user_prompt = f"Question: {query}\n\nRetrieved events:\n{_format_events(sources)}"
    try:
        return llm.complete(_SYSTEM_PROMPT, user_prompt)
    except llm.LLMError as exc:
        log.warning("LLM call failed (%s); falling back to a plain summary", exc)
        return _fallback_answer(sources, spec)


def answer_query(query: str) -> dict:
    """Run the full RAG flow for one question.

    Returns:
        {
          "query":   the question as received,
          "answer":  LLM-generated answer, citing its sources as [1], [2], ...
          "sources": up to TOP_K events the answer is based on, best first, each
                     with score, annotation, video_id and event_id. The backend
                     looks up anything further in Postgres by event_id.
          "filters": which metadata filter was applied, and any notes about
                     what was deliberately ignored. Diagnostic — safe to drop
                     from the API response, but useful for showing the user
                     "searched hospital only" and for the evaluation harness.
        }

    Never raises for an unreachable model: if the LLM fails, `answer` falls back
    to a plain summary so the backend still gets a valid response.
    """
    query = (query or "").strip()
    if not query:
        return {
            "query": query,
            "answer": "Please enter a question.",
            "sources": [],
            "filters": filters.FilterSpec().as_dict(),
        }

    sources, spec = retrieve(query)
    answer = generate_answer(query, sources, spec)

    # Trim at the boundary, not in retrieve(). The prompt is built from the
    # location, camera and timestamps, so those fields have to exist internally
    # — they are what lets the model tell one event filmed twice from two
    # separate events. The backend just does not need them: it looks the rest up
    # in Postgres by event_id.
    return {
        "query": query,
        "answer": answer,
        "sources": [
            {
                "score": s["score"],
                "annotation": s["annotation"],
                "video_id": s["video_id"],
                "event_id": s["event_id"],
            }
            for s in sources
        ],
        "filters": spec.as_dict(),
    }
