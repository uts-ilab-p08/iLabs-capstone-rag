"""Automatic measures over collected model answers.

Everything here is computed without a language model. The only models available
are the ones under test, and a model scoring its own output has a known
self-preference bias — so judgement is left to the human pass, and this file
covers what can be checked mechanically.

Most measures are derived from rules the prompt already states, which makes
them objective: the prompt says to cite events, so citations can be counted.

    from eval.measures import score_all
    df = score_all()
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

RESPONSES = HERE / "model_responses.json"
QUESTIONS = HERE / "model_questions.csv"

CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
ANY_BRACKET = re.compile(r"\[([^\]]*)\]")
ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
MONTH_DATE = re.compile(
    r"\b(?:\d{1,2}\s+)?(january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\b(?:\s+\d{1,2})?", re.I)
CAMERA = re.compile(r"\b[Gg]\d{3}\b")
PREAMBLES = ("based on", "according to", "looking at", "from the retrieved",
             "the retrieved events", "in the provided", "based upon")
SENTENCE = re.compile(r"[.!?]+(?:\s|$)")
STOPWORDS = {
    "the", "and", "with", "that", "this", "they", "their", "from", "into", "then",
    "there", "where", "which", "while", "would", "could", "about", "after", "before",
    "being", "been", "have", "has", "had", "does", "not", "but", "for", "are", "was",
    "were", "any", "all", "also", "appear", "appears", "seen", "shows", "show", "event",
    "events", "footage", "camera", "location", "clip", "none", "nothing", "does",
}


def citations(answer: str) -> list[int]:
    """Every event number cited, including grouped forms like [1, 2]."""
    out = []
    for group in CITATION.findall(answer):
        out += [int(n) for n in re.split(r"\s*,\s*", group)]
    return out


def brackets_only_citations(answer: str) -> bool:
    """True when nothing but event numbers appears inside square brackets.

    The prompt reserves brackets for citations; a model writing [G341] breaks
    any frontend that parses them to link results.
    """
    return all(re.fullmatch(r"[\d\s,]+", inside or "") for inside in ANY_BRACKET.findall(answer))


def mentions_date(answer: str, question: str) -> bool:
    """A calendar date in the answer that the question did not already contain.

    Echoing a date the user supplied is excusable; introducing one is what the
    prompt forbids.
    """
    for pattern in (ISO_DATE, MONTH_DATE):
        for found in pattern.findall(answer):
            text = found if isinstance(found, str) else found[0]
            if text and text.lower() not in question.lower():
                return True
    return False


def has_preamble(answer: str) -> bool:
    return answer.strip().lower().startswith(PREAMBLES)


def sentence_count(answer: str) -> int:
    return len([s for s in SENTENCE.split(answer.strip()) if s.strip()])


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in STOPWORDS}


def ungrounded_mentions(answer: str, context: str) -> list[str]:
    """Locations and cameras named in the answer but absent from the context.

    A direct hallucination check: we know exactly what each model was shown.
    """
    out = []
    ctx = context.lower()
    for cam in set(CAMERA.findall(answer)):
        if cam.lower() not in ctx:
            out.append(cam)
    for place in ("school", "hospital", "bus", "admin"):
        if re.search(rf"\b{place}\b", answer, re.I) and place not in ctx:
            out.append(place)
    return sorted(out)


def extractiveness(answer: str, context: str) -> float:
    """Share of the answer's content words that appear in the context.

    Low values mean the model is introducing vocabulary of its own, which is
    not wrong by itself but is where unsupported claims tend to live.
    """
    words = content_words(answer)
    if not words:
        return 0.0
    ctx = content_words(context)
    return round(len(words & ctx) / len(words), 3)


def cosine(a, b) -> float:
    return round(float(sum(x * y for x, y in zip(a, b))), 3)


def score_all(responses_path: Path | None = None):
    """One row per collected answer, with every automatic measure."""
    import csv

    import pandas as pd

    from rag.embed import Embedder

    data = json.loads((responses_path or RESPONSES).read_text())
    probes = {r["id"]: r["probes"] for r in csv.DictReader(QUESTIONS.open())}
    embedder = Embedder()

    # Embed once per distinct text; the same answers are reused across measures.
    texts = {}
    for r in data["responses"]:
        if r.get("answer"):
            texts[r["answer"]] = None
    for qid, q in data["questions"].items():
        texts[q["question"]] = None
        if q["context"]:
            texts[q["context"]] = None
    keys = list(texts)
    for key, vec in zip(keys, embedder.embed_documents(keys)):
        texts[key] = vec

    rows = []
    for r in data["responses"]:
        q = data["questions"][r["question_id"]]
        is_negative = "negative" in probes.get(r["question_id"], "")
        row = {
            "model": r["model"],
            "question_id": r["question_id"],
            "run": r["run"],
            "probes": probes.get(r["question_id"], ""),
            "is_negative": is_negative,
            "answered": bool(r.get("answer")),
            "n_attempts": r["n_attempts"],
            "seconds": r.get("seconds"),
        }
        if r.get("answer"):
            answer, context = r["answer"], q["context"]
            cited = citations(answer)
            n_sources = q["n_sources"]
            ungrounded = ungrounded_mentions(answer, context)
            row.update({
                "words": len(answer.split()),
                "sentences": sentence_count(answer),
                "cites_anything": bool(cited),
                "cites_in_range": bool(cited) and all(1 <= c <= n_sources for c in cited),
                "brackets_clean": brackets_only_citations(answer),
                "no_date": not mentions_date(answer, q["question"]),
                "no_preamble": not has_preamble(answer),
                "concise_refusal": (sentence_count(answer) == 1) if is_negative else None,
                "ungrounded": ", ".join(ungrounded),
                "is_grounded": not ungrounded,
                "extractiveness": extractiveness(answer, context),
                "relevancy_to_question": cosine(texts[answer], texts[q["question"]]),
                "relevancy_to_context": cosine(texts[answer], texts[context]) if context else None,
            })
        rows.append(row)

    df = pd.DataFrame(rows)

    # Consistency: how alike the two runs of the same question are.
    consistency = {}
    for (model, qid), group in df[df.answered].groupby(["model", "question_id"]):
        answers = [
            r["answer"] for r in data["responses"]
            if r["model"] == model and r["question_id"] == qid and r.get("answer")
        ]
        if len(answers) == 2:
            consistency[(model, qid)] = cosine(texts[answers[0]], texts[answers[1]])
    df["consistency"] = [
        consistency.get((m, q)) for m, q in zip(df.model, df.question_id)
    ]
    return df


if __name__ == "__main__":
    import pandas as pd

    pd.set_option("display.width", 200)
    df = score_all()
    print(f"{len(df)} rows\n")
    answered = df[df.answered]
    cols = ["cites_anything", "cites_in_range", "brackets_clean", "no_date",
            "no_preamble", "is_grounded"]
    summary = answered.groupby("model")[cols].mean().round(2)
    summary["n"] = answered.groupby("model").size()
    print(summary)
