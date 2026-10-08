"""Collect answers from every candidate model, for the model comparison.

    uv run eval/collect_responses.py

Writes eval/model_responses.json. Safe to re-run: work already done is kept and
only what is missing gets collected, which matters because the free endpoints
fail often enough that a clean run end to end is unlikely.

Two design points:

* **Retrieval happens once per question** and the identical context is given to
  every model, so the model is the only thing that varies. Letting each model
  trigger its own retrieval would compare model-plus-retrieval and make a
  difference impossible to attribute.

* **Every attempt is recorded, not just the successful one.** `llm.complete`
  makes a single request and does not retry, so retrying happens here where the
  failures can be counted. That keeps both numbers: how often one call succeeds,
  and how often an answer is obtained once retries are allowed.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag import llm  # noqa: E402
from rag.config import EMBED_MODEL  # noqa: E402
from rag.pipeline import (  # noqa: E402
    PROMPT_VERSION,
    TOP_K,
    _SYSTEM_PROMPT,
    _format_events,
    retrieve,
)

MODELS = [
    # Order matters: the Nvidia models draw on OpenRouter's free-model daily
    # quota (50/day), the Gemmas use a personal Google key and do not. So the
    # quota-consuming models go first, and the one with no data at all leads.
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "google/gemma-4-26b-a4b-it:free",
    "google/gemma-4-31b-it:free",
]

RUNS_PER_QUESTION = 2      # second run exists only to measure consistency
MAX_ATTEMPTS = 3           # per run, to separate "model is bad" from "model was busy"
BACKOFF_SECONDS = 3
PAUSE_BETWEEN_CALLS = 0.5  # be a reasonable neighbour on a shared free pool

HERE = Path(__file__).resolve().parent
QUESTIONS = HERE / "model_questions.csv"
OUTPUT = HERE / "model_responses.json"

console = Console()


def call_once(model: str, system: str, user: str) -> tuple[str | None, str | None, float]:
    """One attempt. Returns (answer, error, seconds)."""
    # llm.complete() reads the model from config at call time, so pointing it at
    # a different model is a module-level assignment. The HTTP client itself is
    # model-independent, so it does not need rebuilding.
    llm.LLM_MODEL = model
    started = time.perf_counter()
    try:
        # complete() makes exactly one request and does not retry, so each call
        # here is a genuine single attempt. Retrying is this script's job,
        # because it needs to count the failures rather than absorb them.
        answer = llm.complete(system, user, temperature=0.0)
        return answer, None, time.perf_counter() - started
    except llm.LLMError as exc:
        return None, str(exc)[:200], time.perf_counter() - started


def load_existing() -> dict:
    if OUTPUT.exists():
        return json.loads(OUTPUT.read_text())
    return {"metadata": {}, "questions": {}, "responses": []}


def main() -> None:
    rows = list(csv.DictReader(QUESTIONS.open()))
    data = load_existing()
    done = {(r["model"], r["question_id"], r["run"]) for r in data["responses"] if r.get("answer")}
    if done:
        console.print(f"[dim]resuming: {len(done)} answers already collected[/dim]\n")

    console.rule("[bold]1/2  Retrieving context (once per question)")
    for row in rows:
        if row["id"] in data["questions"]:
            continue
        sources, spec = retrieve(row["question"])
        data["questions"][row["id"]] = {
            "question": row["question"],
            "probes": row["probes"],
            "filter": spec.describe(),
            "n_sources": len(sources),
            "context": _format_events(sources) if sources else "",
            "sources": [
                {k: s[k] for k in ("score", "annotation", "video_name", "scene",
                                   "camera_id", "start_seconds", "end_seconds")}
                for s in sources
            ],
        }
        console.print(f"  {row['id']}  {len(sources)} sources  filter: {spec.describe()[:46]}")

    answerable = [r for r in rows if data["questions"][r["id"]]["n_sources"] > 0]
    skipped = [r["id"] for r in rows if data["questions"][r["id"]]["n_sources"] == 0]
    if skipped:
        console.print(
            f"\n[yellow]skipping {skipped}: no context retrieved, so the pipeline "
            f"never calls a model and every model would return identical text[/yellow]"
        )

    total = len(MODELS) * len(answerable) * RUNS_PER_QUESTION
    console.rule(f"[bold]2/2  Collecting {total} answers "
                 f"({len(MODELS)} models x {len(answerable)} questions x {RUNS_PER_QUESTION} runs)")

    for model in MODELS:
        ok = fail = 0
        console.print(f"\n[bold]{model}[/bold]")
        for row in answerable:
            qid = row["id"]
            q = data["questions"][qid]
            user = f"Question: {q['question']}\n\nRetrieved events:\n{q['context']}"
            for run in range(1, RUNS_PER_QUESTION + 1):
                if (model, qid, str(run)) in done:
                    continue
                attempts = []
                answer = None
                for attempt in range(1, MAX_ATTEMPTS + 1):
                    answer, error, seconds = call_once(model, _SYSTEM_PROMPT, user)
                    attempts.append({"ok": answer is not None, "error": error,
                                     "seconds": round(seconds, 2)})
                    if answer is not None:
                        break
                    if attempt < MAX_ATTEMPTS:
                        time.sleep(BACKOFF_SECONDS * attempt)
                data["responses"].append({
                    "model": model,
                    "question_id": qid,
                    "run": str(run),
                    "answer": answer,
                    "attempts": attempts,
                    "n_attempts": len(attempts),
                    "seconds": attempts[-1]["seconds"] if answer else None,
                    "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                })
                ok, fail = (ok + 1, fail) if answer else (ok, fail + 1)
                mark = "[green]ok[/green]" if answer else "[red]FAILED[/red]"
                console.print(f"   {qid} run{run}  {mark}  "
                              f"{len(attempts)} attempt(s)  {attempts[-1]['seconds']:.1f}s")
                OUTPUT.write_text(json.dumps(data, indent=1))  # checkpoint every answer
                time.sleep(PAUSE_BETWEEN_CALLS)
        console.print(f"   [bold]{ok} collected, {fail} failed[/bold]")

    data["metadata"] = {
        "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "models": MODELS,
        "runs_per_question": RUNS_PER_QUESTION,
        "max_attempts": MAX_ATTEMPTS,
        "prompt_version": PROMPT_VERSION,
        "embed_model": EMBED_MODEL,
        "top_k": TOP_K,
        "questions_total": len(rows),
        "questions_answerable": len(answerable),
        "questions_skipped": skipped,
    }
    OUTPUT.write_text(json.dumps(data, indent=1))
    console.rule("[bold green]Done")
    console.print(f"{len(data['responses'])} responses -> {OUTPUT}")


if __name__ == "__main__":
    main()
