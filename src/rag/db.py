"""Read events out of the Supabase `bronze` schema.

Only the columns the RAG layer actually needs. Geometries (per-frame bounding
boxes) are deliberately not fetched — they are spatial-evaluation and frontend
overlay data, not retrieval data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import psycopg

from rag.config import DATABASE_URL, DB_SCHEMA


@dataclass
class Event:
    """One retrievable moment. This is the unit we embed and store."""

    event_id: str
    video_id: str
    event_name: str | None
    description: str | None
    start_seconds: float | None
    end_seconds: float | None
    source_status: str
    video_name: str
    camera_id: str | None
    scene: str | None
    capture_start_local: object | None
    video_url: str | None
    object_types: list[str] = field(default_factory=list)


# DISTINCT ON is Postgres-specific: it keeps one row per unique combination of
# the listed columns, choosing which one via ORDER BY. We use it to collapse the
# exact-duplicate events in bronze (same video, same caption, same timespan,
# different event_id).
#
# This is a working convenience so retrieval results are readable — NOT a data
# cleaning layer. Delete it once Juan's silver layer handles de-duplication.
_EVENTS_SQL = f"""
SELECT DISTINCT ON (e.video_id, e.description, e.start_seconds, e.end_seconds)
    e.event_id,
    e.video_id,
    e.event_name,
    e.description,
    e.start_seconds,
    e.end_seconds,
    e.source_status,
    v.video_name,
    v.camera_id,
    v.scene,
    v.capture_start_local,
    v.video_url
FROM {DB_SCHEMA}.events e
JOIN {DB_SCHEMA}.videos v USING (video_id)
ORDER BY e.video_id, e.description, e.start_seconds, e.end_seconds, e.event_id
"""

# Object labels come back as a jsonb array of competing hypotheses, so we pull
# them separately and resolve the winner in Python — an argmax over a jsonb
# array is possible in SQL but unreadable.
_OBJECTS_SQL = f"""
SELECT eo.event_id, o.label_details
FROM {DB_SCHEMA}.event_objects eo
JOIN {DB_SCHEMA}.objects o USING (object_id)
"""

_TOTAL_EVENTS_SQL = f"SELECT count(*) FROM {DB_SCHEMA}.events"


def primary_label(label_details: list[dict] | None) -> str | None:
    """Pick the most trustworthy label for one object.

    An object can claim several labels, e.g. `car` at 0.55 average confidence
    over 24 frames and `truck` at 0.30 over 1 frame. We rank by
    `average confidence x detection_count` rather than confidence alone, because
    a label backed by many frames is more reliable than a confident one-off.
    """
    if not label_details:
        return None

    def score(entry: dict) -> float:
        confidence = (entry.get("confidence") or {}).get("average") or 0.0
        count = entry.get("detection_count") or 0
        return float(confidence) * float(count)

    best = max(label_details, key=score)
    return best.get("label")


def fetch_events() -> tuple[list[Event], int]:
    """Return (de-duplicated events, total event rows before de-duplication)."""
    with psycopg.connect(DATABASE_URL) as conn, conn.cursor() as cur:
        cur.execute(_TOTAL_EVENTS_SQL)
        total_rows = cur.fetchone()[0]

        cur.execute(_EVENTS_SQL)
        columns = [d.name for d in cur.description]
        events = {
            row_dict["event_id"]: Event(**row_dict)
            for row_dict in (dict(zip(columns, row)) for row in cur.fetchall())
        }

        cur.execute(_OBJECTS_SQL)
        for event_id, label_details in cur.fetchall():
            event = events.get(event_id)
            if event is None:
                continue  # object belongs to an event dropped by de-duplication
            label = primary_label(label_details)
            if label and label not in event.object_types:
                event.object_types.append(label)

    for event in events.values():
        event.object_types.sort()

    return list(events.values()), total_rows
