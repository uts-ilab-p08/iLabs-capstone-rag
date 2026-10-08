# Model comparison — which LLM should generate the answers

Companion to [`model_comparison.ipynb`](model_comparison.ipynb), which holds the
code, tables and charts behind these findings.

---

## Recommendation

**Use `nvidia/nemotron-3-ultra-550b-a55b`.** It produced the best answers by a
clear margin (4.91 of 5 against 4.29 for the runner-up) and was the only model
that never failed a single call during collection.

**Keep `google/gemma-4-26b-a4b` configured as the fast alternative.** It is three
times quicker, perfectly deterministic, and the strongest on every automatic
measure — a better fit for iteration, demos, or any setting where a 12-second
wait is unacceptable.

**Do not use `google/gemma-4-31b`.** It needed 167 API calls to produce 13
answers and never completed the question set.

---

## What was compared

Four models, the same 15 questions, the same retrieved evidence.

The context is **retrieved once per question and given identically to every
model**, so the model is the only thing that varies. Had each model triggered its
own retrieval, any difference would have been impossible to attribute.

Questions were chosen to stress particular rules in the prompt rather than
sampled at random — several exist because an earlier version of the system failed
on them. They cover scope summaries, conflation across locations, counting,
refusals where the context looks superficially relevant, unreliable detector
labels, cross-camera identity, and temporal ordering the system was never built
to do.

Two runs per question per model: the first is scored, the second measures
consistency. 151 responses in total.

### How quality was judged

**By an LLM judge (Claude), not by a human.** Scores are correctness,
faithfulness and usefulness, each 1–5, assigned from the question, the supplied
context and the answer alone.

Using one of the four candidates as judge was rejected — a model scoring its own
output carries a self-preference bias. Claude is outside the candidate set, so
that objection does not apply.

Every score carries a written justification in `judge_scores.csv`, and the full
set was reviewed and accepted by the project author before being used.

---

## Results

| Model | calls per answer | median s | rule compliance | consistency | judge mean |
|---|---|---|---|---|---|
| **nemotron-550b** | **1.00** | 11.5 | 0.967 | 0.924 | **4.91** |
| gemma-4-26b | 1.13 | **3.8** | **0.978** | **1.000** | 4.29 |
| gemma-4-31b | **12.85** | 16.4 | 0.897 | 0.909 | 3.81 |
| nemotron-30b | 1.40 | 8.9 | 0.944 | 0.835 | 3.69 |

### Quality, in detail

| Model | correctness | faithfulness | usefulness | n |
|---|---|---|---|---|
| nemotron-550b | 4.93 | 4.93 | **4.87** | 15 |
| gemma-4-26b | 4.27 | 4.40 | 4.20 | 15 |
| gemma-4-31b | 3.71 | 4.14 | 3.57 | 7 |
| nemotron-30b | 3.67 | 4.47 | **2.93** | 15 |

---

## Findings

### The automatic measures and the quality scores disagree

`gemma-4-26b` leads on rule compliance, consistency, extractiveness and speed.
The 550B wins on quality by 0.6 of a point. Both results are real, and the
disagreement is the most useful thing in this evaluation: **following the rules
and answering well are not the same thing.**

The 550B breaks two rules the others keep. It is the only model that writes
calendar dates (13% of answers, against zero elsewhere), and it never once gave a
one-sentence refusal where the prompt asks for one — 0 of 6. It still produced
the best answers.

### Terseness read as discipline in the metrics and as uselessness in the scoring

`nemotron-30b` scores **4.47 on faithfulness but 2.93 on usefulness** — the
widest gap in the set. It rarely says anything unsupported because it says very
little: a median of 17 words against 64 for the 550B.

Asked whether there was a truck at the school, having been given four, it
answered *"Yes. Event [1] shows a dark pickup truck at the school."* That is
faithful, compliant, and close to useless.

No automatic measure can separate admirable concision from unhelpful thinness.
This is the specific gap the quality scoring exists to fill.

### The 550B earns its score on the hard questions

Asked what happened *after* the white SUV arrived — temporal ordering the system
never implemented — it challenged the premise, gave precise timing, and stated
that it could not be determined whether a later sighting was the same vehicle
returning or a different one.

Asked whether the same person appeared on more than one camera, it reasoned that
the three events covered two locations on different days, and that the only
same-location pair came from a single camera, so no cross-camera match existed.

Both are cases where a confident wrong answer was the easy path.

### Two answers were outright wrong, from two different models

- One claimed a bag was left **unattended** while stating in the same sentence
  that the person *remained near the bench*. A false positive on an
  abandoned-bag query is the worst error type in this domain.
- One claimed a person appeared on more than one camera and then named camera
  **G505 twice**.

Both contradict evidence they themselves cite. Neither was caught by any
automatic measure, because both are fluent, well-cited and grounded in supplied
events.

### gemma-4-31b is unusable, and this is not a bad day

**167 API calls for 13 answers.** It runs on a personal Google key with no shared
quota involved, so the failures are provider-side: HTTP 504 aborts at a
consistent ~11.5 seconds, and HTTP 502 relaying a genuine internal error from
Google AI Studio. The 26B runs on the same key through the same provider at 1.13
calls per answer.

Its quality scores rest on 7 answers against 15 for the others and are not
directly comparable.

### Availability could not be used as a ranking criterion

The original design treated availability as a gate. The data does not support
that. The 550B recorded zero successes on one day and 30 of 30 the next; the
first was the OpenRouter account hitting its free-model daily limit, so those
calls never reached a provider. Those records were removed from the dataset, and
the removal is noted in `model_responses.json`.

What survives is narrower but sound: three models are viable when quota exists,
and one fails at the provider regardless.

---

## Limitations

**Fifteen questions, four models.** Enough to establish that one model cites and
another does not. Not enough to support "model A is 12% better".

**Unequal samples.** gemma-4-31b contributes 7 scored answers against 15 for the
others.

**Availability is confounded** by an account-level quota that was mixed in with
genuine provider failures during collection.

**Free-tier models only.** A paid model was not evaluated and might change the
recommendation, particularly on latency and reliability.

---

## Reproducing

```bash
cd rag/eval
uv run collect_responses.py        # slow, flaky, resumable; writes model_responses.json
uv run python measures.py          # automatic measures, printed summary
uv run jupyter lab model_comparison.ipynb
```

Collection is the only non-deterministic step. The notebook reads the cached
responses, so it reproduces exactly and can be re-run without touching an API.
