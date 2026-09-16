"""Semantic search over the indexed events.

    uv run scripts/search.py "person getting out of a car"
    uv run scripts/search.py "white truck" --limit 10

Pure vector search: no metadata filters and no LLM yet. This is the tool that
tells us whether retrieval itself is any good, before we build layers on top
that could hide its failures.
"""

from __future__ import annotations

import argparse

from rich.console import Console

from rag import store
from rag.embed import Embedder

console = Console()


def format_span(start: float | None, end: float | None) -> str:
    if start is None or end is None:
        return "unknown span"

    def mmss(seconds: float) -> str:
        return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"

    return f"{mmss(start)}-{mmss(end)}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic search over indexed MEVA events")
    parser.add_argument("query", help="what to search for, in plain language")
    parser.add_argument("--limit", type=int, default=5, help="how many results (default 5)")
    args = parser.parse_args()

    console.rule(f"[bold]Searching: {args.query}")

    console.print("[bold]1/3[/bold] Loading embedding model...")
    embedder = Embedder()

    console.print("[bold]2/3[/bold] Embedding the query...")
    console.print("      [dim](queries get BGE's instruction prefix, documents do not)[/dim]")
    vector = embedder.embed_query(args.query)

    console.print("[bold]3/3[/bold] Searching Qdrant...")
    console.print(f"      connection: {store.describe_connection()}")
    client = store.connect()
    hits = store.search(client, vector, limit=args.limit)
    console.print(f"      searched {store.count(client)} indexed events\n")

    if not hits:
        console.print("[yellow]No results. Has the index been built?[/yellow]")
        return

    console.rule(f"[bold]Top {len(hits)} results")
    for rank, hit in enumerate(hits, start=1):
        p = hit.payload
        console.print(f"[bold]{rank}. [green]{hit.score:.3f}[/green]  {p.get('description')}")
        console.print(
            f"   [dim]{p.get('video_name')}"
            f"  |  camera {p.get('camera_id')} ({p.get('scene')})"
            f"  |  {format_span(p.get('start_seconds'), p.get('end_seconds'))}"
            f"  |  objects: {', '.join(p.get('object_types') or []) or 'none'}[/dim]"
        )
        if p.get("video_url"):
            console.print(f"   [dim]{p['video_url']}[/dim]")
        console.print()


if __name__ == "__main__":
    main()
