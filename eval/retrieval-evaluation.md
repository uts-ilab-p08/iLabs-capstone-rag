# Retrieval evaluation — results

Companion to [`evaluation.ipynb`](evaluation.ipynb), which contains the code,
tables and charts these findings come from.

---

## Summary

Metadata filtering is the single largest factor in retrieval accuracy. With it
enabled the system found the correct footage for **20 of 22** answerable
questions; without it, **9 of 22**. Mean reciprocal rank more than doubled, from
0.409 to 0.909.

The system also declines to answer when it should: **4 of 5** questions about
things absent from the footage returned nothing, and **3 of 3** ambiguous or
negated questions were correctly refused with the reason recorded.

Two questions failed, both for reasons worth fixing rather than accepting.

---

## What was measured

**Retrieval only** — whether the right footage comes back. Every number here is
computed from the events returned by the search, before any language model is
involved, so the results are independent of which LLM is configured and are
reproducible run to run.

Answer quality is deliberately excluded. Measuring them together makes a bad
answer impossible to attribute: retrieval may have missed the footage, or the
model may have fumbled footage it was given.

### How correctness was decided

Each question carries criteria describing what a correct answer looks like —
caption terms plus optional location, camera, date and time constraints. The
answer key is built by scanning the caption text of all 2,013 indexed events.

**The system under test never sees those criteria.** It receives only the
question. Building the answer key from the system's own search results would
grade it against itself.

The key is lexical; the retrieval being tested is semantic. The system therefore
has to bridge a vocabulary gap the answer key does not, which makes these
scores a **lower bound** — a correct result whose caption wording differs from
the key is counted as a miss. Understating is the safer direction to err.

### Setup

| | |
|---|---|
| Events indexed | 2,013 |
| Videos | 71 |
| Locations | 4 (school, bus, hospital, admin) |
| Cameras | 23 |
| Embedding model | BAAI/bge-base-en-v1.5, 768 dimensions |
| Results per query (`TOP_K`) | 5 |
| Similarity floor (`MIN_SCORE`) | 0.6 |

**30 questions**: 22 answerable (plain lookup, location, camera, date, time and
combined filters), 5 negative, 3 refusal. Difficulty is uneven on purpose — one
question has 7 relevant events in the entire corpus, another has 1,621.

---

## Results

| Condition | Hit@1 | Hit@3 | Hit@5 | out of | MRR | returned nothing |
|---|---|---|---|---|---|---|
| **Filters on** | 20 | 20 | 20 | 22 | **0.909** | 2 |
| **Filters off** | 9 | 9 | 9 | 22 | 0.409 | 12 |

### Finding 1 — filtering more than doubles accuracy

20 of 22 against 9 of 22. At this sample size a difference of one or two
questions would mean nothing; a difference of eleven is real.

**The mechanism is visible in the last column.** With filters off, 12 of 22
questions returned *nothing at all* — not wrong footage, no footage. The
similarity floor of 0.6 rejected every result.

This is the behaviour the design anticipated. Similarity measures textual
resemblance, not relevance, and a question like "what happened at the hospital?"
resembles no individual caption, because no caption is phrased as a summary of a
clip. Every hospital event scores around 0.45–0.50 while being perfectly
relevant. When a filter matches, the floor is dropped and the filter supplies
relevance instead — which is exactly why those 11 questions recover.

Every location, camera, date and combined question failed without filtering and
succeeded with it.

### Finding 2 — when it finds the footage, it finds it first

Hit@1, Hit@3 and Hit@5 are identical in both conditions. The system never
recovers at ranks 2 through 5: either the correct video is the top result, or it
is not in the list at all.

Useful practically — a user does not need to scan five results — and it means
`TOP_K` could be reduced without losing accuracy, at some cost to the context
the language model receives.

### Finding 3 — negative questions: 4 of 5

Questions about an ambulance, a helicopter, a dog, an umbrella and children.
None of these words appears in any caption, verified directly against the
corpus. A system that always returns something would score well on recall-style
measures while being useless in practice, so this is measured explicitly.

Four returned nothing, correctly. **One failed, and the failure is instructive.**

"Did an ambulance arrive at the hospital?" returned five hospital events at
scores around 0.59. The word *hospital* triggered the location filter, the floor
was dropped as designed, and hospital footage came back — for a question about
an ambulance that does not exist.

This is the cost of Finding 1, not a separate bug. Dropping the floor is what
rescues eleven questions; it is also what lets this one through. The two are the
same mechanism.

### Finding 4 — refusals: 3 of 3

The filter extractor declines to guess when a question is ambiguous, and records
why:

| Question | Recorded |
|---|---|
| "Was there a school bus?" | ambiguous location (bus, school), not filtering |
| "What happened at 2-4?" | ignored time: no am/pm given |
| "Show me anything except at the school" | ignored location school: looks excluded |

All three fall back to unfiltered search rather than filtering wrongly. A wrong
filter reports "no matching footage" about footage that exists, and nothing
downstream can recover it; a missed filter merely leaves results broad.

---

## The two failures

### q03 — "Did anyone use a phone?"

Returned nothing, in both conditions, despite 121 relevant events.

The question names no location, so no filter fires, so the 0.6 floor applies —
and every result fell below it. The cause is **question phrasing**. The same
content scores very differently depending on wording:

| Query | Top score | Survives the floor? |
|---|---|---|
| "bicycle" | 0.689 | yes |
| "a person riding a bicycle" | 0.691 | yes |
| "Did anyone ride a bicycle?" | 0.571 | no |

Interrogative phrasing costs roughly 0.12 of similarity, because captions are
declarative descriptions and nothing in the corpus is phrased as a question.
Conversational questions — exactly what the system invites — are penalised
relative to keyword-style ones.

### q17 — "What happened between 11am and noon?"

No time filter was extracted, so the question was answered by unfiltered search
and the floor rejected everything.

The clock parser handles only numeric readings. **"noon", "midday" and
"midnight" are not recognised**, and they fail *silently* — the parser returns
nothing rather than refusing, so no note appears in the output explaining why
the filter did not fire. "between 11am and 12pm" works correctly.

The same gap affects q22 ("what did camera G341 record between 11am and noon?"),
where the time filter was also dropped. That question still succeeded, because
the camera filter carried it.

---

## What this suggests

**The similarity floor is the weak point.** Both failures trace to it, and so
does the one negative-query failure. A fixed threshold works for
description-shaped questions and fails for question-shaped ones, and relaxing it
for filtered queries creates the ambulance problem. Worth investigating:
embedding questions and captions more symmetrically, or deciding "nothing
matches" from the gap between consecutive scores rather than an absolute cut-off.

**Word-based times should be parsed or explicitly refused.** Silent failure is
the worst option; even a note saying the time was not understood would make this
visible to a user.

**`TOP_K` could be lowered.** Nothing was ever found below rank 1.

---

## Limitations

**Sample size.** 30 questions. A difference of one or two between conditions is
noise; only large gaps should be read as real. The filtering result is large
enough to trust, the others are indicative.

**The answer key is lexical.** A correct result whose caption wording differs
from the key counts as a miss, so these numbers understate true performance.

**This measures retrieval over captions, not over video.** If a caption
describes footage incorrectly, both the answer key and the search inherit the
error. Whether captions are faithful to the video is the annotation layer's
evaluation, deliberately separate from this one.

**Credit is given at video level.** A result counts as correct if it comes from
a video containing a relevant event, even when that specific event is not the
relevant one. Finer scoring would need event-level ground truth.

**Questions were written against the indexed captions.** They are grounded in
what the corpus actually contains, which avoids asking about things the system
could never find — but it is not an independent, blind question set.

---

## Reproducing

```bash
cd rag/eval
uv run jupyter lab evaluation.ipynb      # or: uv run jupyter nbconvert --execute --inplace --to notebook evaluation.ipynb
```

No language model is called, so the run is deterministic and free. The
notebook records the corpus size, embedding model, `TOP_K` and `MIN_SCORE` of
the run that produced it — results from different settings are not comparable.
The corpus has grown several times during the project, so re-running on a larger
index will not reproduce these exact numbers.
