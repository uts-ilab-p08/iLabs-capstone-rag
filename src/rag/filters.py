"""Pull metadata filters out of a plain-language question.

Rule-based on purpose. The vocabulary is tiny — three locations and thirteen
camera IDs — and filtering is destructive: it drops candidates before ranking.
A missed filter leaves us with today's slightly noisy results, but a wrong
filter confidently reports "no matching footage" about footage that exists and
nothing downstream can recover it. So every rule here errs toward doing nothing.

Swapping this for an LLM extractor later means reimplementing `extract()`; the
rest of the pipeline only sees FilterSpec.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from qdrant_client import QdrantClient, models

from rag.config import COLLECTION

# Everyday words for the locations MEVA labels more tersely.
SYNONYMS = {
    "campus": "school",
    "classroom": "school",
    "clinic": "hospital",
    "medical centre": "hospital",
    "medical center": "hospital",
    "depot": "bus",
    "bus stop": "bus",
    "bus station": "bus",
}

# If one of these sits just before a location word, the user is excluding it,
# not asking for it. We cannot express that reliably, so we filter on nothing.
NEGATIONS = ("not", "no", "except", "excluding", "other than", "besides", "without", "apart from")

_CAMERA_RE = re.compile(r"\b[gG]\d{3}\b")


@dataclass(frozen=True)
class FilterSpec:
    """What we decided to filter on, and why."""

    scenes: tuple[str, ...] = ()
    cameras: tuple[str, ...] = ()
    notes: tuple[str, ...] = field(default=())

    @property
    def is_empty(self) -> bool:
        return not self.scenes and not self.cameras

    def describe(self) -> str:
        parts = []
        if self.scenes:
            parts.append("location " + " or ".join(self.scenes))
        if self.cameras:
            parts.append("camera " + " or ".join(self.cameras))
        if not parts:
            parts.append("none")
        if self.notes:
            parts.append("(" + "; ".join(self.notes) + ")")
        return ", ".join(parts)

    def as_dict(self) -> dict:
        return {
            "scenes": list(self.scenes),
            "cameras": list(self.cameras),
            "notes": list(self.notes),
        }


@lru_cache(maxsize=1)
def vocabulary(client: QdrantClient) -> tuple[frozenset[str], frozenset[str]]:
    """Distinct scenes and camera IDs that are actually in the index.

    Read from Qdrant rather than Postgres on purpose: the vocabulary has to
    describe what is *searchable*. A location that exists upstream but has not
    been indexed yet would otherwise become a filter that matches nothing.

    Uses the facet API, which computes distinct values server-side off the
    payload indexes. Falls back to a scroll for embedded mode, where payload
    indexes are not created.

    Cached for the life of the process, so a long-running server will not notice
    newly indexed locations until it restarts.
    """
    scenes, cameras = set(), set()
    try:
        for key, sink in (("scene", scenes), ("camera_id", cameras)):
            for hit in client.facet(collection_name=COLLECTION, key=key, limit=200).hits:
                if hit.value:
                    sink.add(str(hit.value))
    except Exception:  # noqa: BLE001 - embedded mode, or no payload index
        scenes, cameras = set(), set()
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=COLLECTION,
                limit=500,
                offset=offset,
                with_payload=["scene", "camera_id"],
                with_vectors=False,
            )
            for p in points:
                if p.payload.get("scene"):
                    scenes.add(str(p.payload["scene"]))
                if p.payload.get("camera_id"):
                    cameras.add(str(p.payload["camera_id"]))
            if offset is None:
                break
    return frozenset(scenes), frozenset(cameras)


def _negated(text: str, start: int) -> bool:
    """True if a negation word appears shortly before position `start`."""
    window = text[max(0, start - 30) : start]
    return any(neg in window for neg in NEGATIONS)


def extract(query: str, known_scenes: frozenset[str], known_cameras: frozenset[str]) -> FilterSpec:
    text = (query or "").lower()
    notes: list[str] = []

    # Cameras: exact tokens like G341. Unknown IDs are discarded rather than
    # turned into a filter that is guaranteed to match nothing.
    cameras, unknown = set(), set()
    for match in _CAMERA_RE.finditer(query):
        found = match.group(0).upper()
        if found in known_cameras:
            if _negated(text, match.start()):
                notes.append(f"ignored camera {found}: looks excluded")
            else:
                cameras.add(found)
        else:
            unknown.add(found)
    if unknown:
        notes.append("unknown camera id " + ", ".join(sorted(unknown)))

    # Locations: the scene name itself, or an everyday synonym for it.
    candidates = {scene: scene for scene in known_scenes}
    candidates.update({word: scene for word, scene in SYNONYMS.items() if scene in known_scenes})

    scenes = set()
    for word, scene in candidates.items():
        match = re.search(rf"\b{re.escape(word)}\b", text)
        if not match:
            continue
        if _negated(text, match.start()):
            notes.append(f"ignored location {scene}: looks excluded")
            continue
        scenes.add(scene)

    # "was there a school bus?" matches both school and bus. Rules cannot tell
    # that from a genuine two-location question, and an unfiltered search still
    # returns something whereas a wrong filter returns nothing.
    if len(scenes) > 1:
        notes.append("ambiguous location (" + ", ".join(sorted(scenes)) + "), not filtering")
        scenes = set()

    return FilterSpec(tuple(sorted(scenes)), tuple(sorted(cameras)), tuple(notes))


def to_qdrant_filter(spec: FilterSpec) -> models.Filter | None:
    """FilterSpec -> Qdrant filter. Values within a field are OR, fields are AND."""
    if spec.is_empty:
        return None
    must = []
    if spec.scenes:
        must.append(models.FieldCondition(key="scene", match=models.MatchAny(any=list(spec.scenes))))
    if spec.cameras:
        must.append(
            models.FieldCondition(key="camera_id", match=models.MatchAny(any=list(spec.cameras)))
        )
    return models.Filter(must=must)
