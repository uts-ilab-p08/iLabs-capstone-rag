"""Turn an Event into the single string we embed.

We do not embed the raw caption on its own. A question like "a vehicle at the
bus stop" needs to match on location and object type, and neither necessarily
appears in the caption text. Folding the structured fields into the embedded
string makes them semantically reachable.

The raw caption stays in the Qdrant payload separately — that is what we show
the user, and what the LLM will read as context later. This composed string is
machine input only; never display it.
"""

from __future__ import annotations

from rag.db import Event


def compose_text(event: Event) -> str:
    lines: list[str] = []

    if event.description:
        lines.append(event.description.strip())

    details: list[str] = []
    if event.event_name:
        details.append(f"Activity: {event.event_name.strip().lower()}.")
    if event.object_types:
        details.append(f"Objects: {', '.join(event.object_types)}.")
    if details:
        lines.append(" ".join(details))

    if event.camera_id:
        location = f"Camera: {event.camera_id}"
        if event.scene:
            location += f" ({event.scene})"
        lines.append(location + ".")

    return "\n".join(lines)
