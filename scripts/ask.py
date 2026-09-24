"""Run the RAG pipeline on one question, exactly as the backend will.

    uv run scripts/ask.py "person getting out of a car"
"""

from __future__ import annotations

import argparse

from rich.console import Console

from rag import llm
from rag.pipeline import answer_query

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question over the indexed MEVA events")
    parser.add_argument("query", help="the question, in plain language")
    args = parser.parse_args()

    # INPUT — in the backend, the query API sets this from the frontend request.
    query = args.query

    console.rule(f"[bold]Question: {query}")
    console.print("[bold]1/2[/bold] Embedding the question and searching Qdrant...")
    console.print("      [dim](first question loads the model, a few seconds)[/dim]")
    console.print(f"      LLM: {llm.describe()}")

    # OUTPUT — in the backend, this dict is what gets handed to the response API.
    response = answer_query(query)

    f = response["filters"]
    applied = ", ".join(f["scenes"] + f["cameras"]) or "none"
    console.print(f"      filter applied: {applied}")
    for note in f["notes"]:
        console.print(f"      [yellow]note:[/yellow] {note}")
    console.print(f"[bold]2/2[/bold] Answer built from {len(response['sources'])} matching event(s)\n")
    console.rule("[bold]response")
    console.print_json(data=response)


if __name__ == "__main__":
    main()

