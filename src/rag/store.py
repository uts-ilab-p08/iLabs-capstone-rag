"""Qdrant wrapper: create the collection, upsert events, search.

Connection mode is decided by QDRANT_URL in .env:
  set    -> connect to a Qdrant server (Docker container, or a hosted cluster)
  empty  -> run embedded against ./qdrant_data, no Docker required

Everything else in this file is identical either way.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from qdrant_client import QdrantClient, models

from rag.config import (
    COLLECTION,
    EMBED_DIM,
    QDRANT_API_KEY,
    QDRANT_LOCAL_PATH,
    QDRANT_URL,
)
from rag.db import Event

# Qdrant point IDs must be an unsigned integer or a UUID. Our event_id is a
# 64-char hex string, which is neither, so we derive a UUID from it. uuid5 is a
# pure function of its input: the same event_id always produces the same point
# ID, which is what makes re-indexing idempotent — a second run overwrites the
# same points instead of inserting duplicates.
_ID_NAMESPACE = uuid.NAMESPACE_URL


@dataclass
class Hit:
    score: float
    payload: dict


def point_id(event_id: str) -> str:
    return str(uuid.uuid5(_ID_NAMESPACE, event_id))


def connect() -> QdrantClient:
    if QDRANT_URL:
        # The default timeout is a few seconds, which is fine for a container on
        # localhost but not for a hosted cluster on the far side of the network.
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
    QDRANT_LOCAL_PATH.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(QDRANT_LOCAL_PATH))


def describe_connection() -> str:
    return f"server {QDRANT_URL}" if QDRANT_URL else f"embedded at {QDRANT_LOCAL_PATH}"


# Qdrant can only filter on fields it has an index for, so every field the query
# layer filters on must appear here.
PAYLOAD_INDEXES = (
    ("camera_id", models.PayloadSchemaType.KEYWORD),
    ("scene", models.PayloadSchemaType.KEYWORD),
    ("source_status", models.PayloadSchemaType.KEYWORD),
    ("object_types", models.PayloadSchemaType.KEYWORD),
    ("capture_date", models.PayloadSchemaType.KEYWORD),
    ("start_seconds", models.PayloadSchemaType.FLOAT),
    ("end_seconds", models.PayloadSchemaType.FLOAT),
    ("start_timestamp", models.PayloadSchemaType.FLOAT),
    ("start_time_of_day", models.PayloadSchemaType.INTEGER),
)


def ensure_collection(client: QdrantClient) -> bool:
    """Create the collection if missing, and make sure every index exists.

    Indexes are ensured on every run, not only at creation. Adding a filterable
    field to an existing collection otherwise leaves it unindexed, and filtering
    on it fails long after the change that introduced it.
    """
    created = not client.collection_exists(COLLECTION)
    if created:
        # Cosine distance is the standard choice for text embeddings: it
        # compares direction rather than magnitude, so a long caption is not
        # penalised against a short one. bge vectors are L2-normalised, so
        # cosine and dot product are equivalent here anyway.
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(size=EMBED_DIM, distance=models.Distance.COSINE),
        )

    # Embedded mode ignores payload indexes (it scans instead), so we only
    # create them against a real server to avoid a misleading warning.
    if QDRANT_URL:
        for field_name, schema in PAYLOAD_INDEXES:
            client.create_payload_index(
                collection_name=COLLECTION, field_name=field_name, field_schema=schema
            )
    return created


def to_payload(event: Event, composed_text: str) -> dict:
    """What we store alongside the vector.

    `description` is the human-readable caption we display and later feed to the
    LLM. `composed_text` is kept only so we can debug what was actually embedded.
    """
    payload = {
        "event_id": event.event_id,
        "video_id": event.video_id,
        "video_name": event.video_name,
        "camera_id": event.camera_id,
        "scene": event.scene,
        "event_name": event.event_name,
        "description": event.description,
        "start_seconds": event.start_seconds,
        "end_seconds": event.end_seconds,
        "object_types": event.object_types,
        "source_status": event.source_status,
        "video_url": event.video_url,
        "composed_text": composed_text,
    }
    payload.update(_temporal_fields(event))
    return payload


def _temporal_fields(event: Event) -> dict:
    """When the event happened, in forms Qdrant can filter on.

    `start_seconds` is an offset inside its clip, so it cannot answer "what
    happened at 2pm". These three can:

      capture_date       "2018-03-07"            which day
      start_time_of_day  seconds since midnight  time of day, across all days
      start_timestamp    epoch seconds           absolute point in time

    Time of day needs its own field because it cannot be expressed as a range
    over timestamps: 2-4pm on two different days is two disjoint windows.

    The epoch value is pinned to UTC rather than using the machine's local
    timezone, so an index built on one machine and queried from another agrees.
    The source timestamps carry no timezone (capture_time_zone is "unknown"
    throughout), so this is a consistent convention, not a claim about the real
    local time.
    """
    start = event.capture_start_local
    if not isinstance(start, datetime) or event.start_seconds is None:
        return {"capture_date": None, "start_time_of_day": None, "start_timestamp": None}

    at = start + timedelta(seconds=event.start_seconds)
    return {
        "capture_date": at.strftime("%Y-%m-%d"),
        "start_time_of_day": at.hour * 3600 + at.minute * 60 + at.second,
        "start_timestamp": at.replace(tzinfo=timezone.utc).timestamp(),
    }


def upsert(client: QdrantClient, points: list[models.PointStruct]) -> None:
    client.upsert(collection_name=COLLECTION, points=points)


def count(client: QdrantClient) -> int:
    return client.count(collection_name=COLLECTION, exact=True).count


def search(
    client: QdrantClient,
    vector: list[float],
    limit: int = 5,
    query_filter: models.Filter | None = None,
) -> list[Hit]:
    response = client.query_points(
        collection_name=COLLECTION,
        query=vector,
        limit=limit,
        with_payload=True,
        query_filter=query_filter,
    )
    return [Hit(score=p.score, payload=p.payload or {}) for p in response.points]
