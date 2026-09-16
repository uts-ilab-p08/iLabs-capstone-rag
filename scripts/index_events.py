"""Build the vector index: fetch -> compose -> embed -> store.

    uv run scripts/index_events.py

Safe to re-run. Point IDs are derived deterministically from event_id, so a
second run overwrites the same points rather than duplicating them.
"""

from __future__ import annotations

from qdrant_client import models
from rich.console import Console

from rag import store
from rag.db import fetch_events
from rag.embed import Embedder
from rag.text import compose_text

console = Console()
BATCH_SIZE = 64


def main() -> None:
    console.rule("[bold]Indexing MEVA events")

    console.print("[bold]1/5[/bold] Fetching events from Supabase...")
    events, total_rows = fetch_events()
    collapsed = total_rows - len(events)
    console.print(f"      {total_rows} event rows in bronze")
    console.print(f"      {collapsed} collapsed as exact duplicates")
    console.print(f"      [green]{len(events)} events to index[/green]\n")

    if not events:
        console.print("[red]Nothing to index. Stopping.[/red]")
        return

    console.print("[bold]2/5[/bold] Composing text for embedding...")
    texts = [compose_text(event) for event in events]
    console.print("      example of what gets embedded:")
    for line in texts[0].splitlines():
        console.print(f"        [dim]{line}[/dim]")
    console.print()

    console.print("[bold]3/5[/bold] Loading embedding model...")
    console.print("      [dim](first run downloads ~130MB, then it is cached)[/dim]")
    embedder = Embedder()
    console.print(f"      [green]{embedder.model_name}[/green]\n")

    console.print("[bold]4/5[/bold] Embedding...")
    vectors: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        vectors.extend(embedder.embed_documents(batch))
        console.print(f"      {len(vectors)}/{len(texts)} embedded")
    console.print(f"      [green]{len(vectors[0])} dimensions per vector[/green]\n")

    console.print("[bold]5/5[/bold] Storing in Qdrant...")
    console.print(f"      connection: {store.describe_connection()}")
    client = store.connect()
    created = store.ensure_collection(client)
    console.print(f"      collection: {'created' if created else 'already existed'}")

    points = [
        models.PointStruct(
            id=store.point_id(event.event_id),
            vector=vector,
            payload=store.to_payload(event, text),
        )
        for event, text, vector in zip(events, texts, vectors)
    ]
    store.upsert(client, points)
    console.print(f"      upserted {len(points)} points")
    console.print(f"      [green]collection now holds {store.count(client)} points[/green]\n")

    console.rule("[bold green]Done")
    console.print('Try: [bold]uv run scripts/search.py "person getting out of a car"[/bold]')


if __name__ == "__main__":
    main()
