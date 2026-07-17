# Collecting a StudyBench-style validation set for self-quizzing

Date: 2026-07-16

## Objective

Build `data/smalldspy_ourvalidationset.jsonl` — our own set of
`react_agents_and_tools` questions with gold answers and rubrics, schema-
identical to `data/smalldspy.jsonl` — by replicating the StudyBench
coding-suite construction pipeline (paper Appendix A) as closely as
possible, seeded from real user-submitted questions. This is the first
ingredient of the self-quizzing studying method: fixed validation (and
later training) questions whose distribution matches the held-out test set
via the "resolving user-submitted questions" prior, without using the test
questions themselves.

## Pipeline

Implementation in `data_collection/` (one module per paper stage; complete
API logs under `data_collection/logs/api/`; every divergence recorded in
`data_collection/README.md`'s fidelity table). Commands in that README.

## What happened at each stage

- **s0 collect**: 1,634 stanfordnlp/dspy GitHub issues (1,331 closed)
  snapshotted via GraphQL on 2026-07-16. The paper's DSPy seeds came from
  private community QA sessions; we substitute its OpenClaw recipe (closed
  issues).
- **s1 filter (paper Stage 1)**: closed + created before the pinned corpus
  commit date (2026-03-31) + length + English + question-form + exact and
  MinHash (128 perms, Jaccard 0.7) dedup → **302 seeds**. The question-form
  filter is the big cut (989 dropped): most issues are declarative bug
  reports. Funnel in `artifacts/filter_report.json`.
- **s2a embed**: Qwen3-Embedding-8B (revision `1d8ad4c`) on one L40S via
  vLLM, reconstructed domain instruct prefix, 302 seeds + the 5 held-out
  test questions (flagged, never clustered).
- **s2b cluster (paper Stage 2)**: UMAP(10, n_neighbors=15, cosine, seeded)
  + HDBSCAN. Paper-literal (30,5) yields 3 clusters + 122 noise; pool-scaled
  (10,5) yields 6 clusters + 92 noise, GPT-5.4-labeled: documentation/repo
  maintenance, prompt customization, integrations/extensibility, custom LM
  backends, environment setup, bug troubleshooting.

## Finding: no react_agents_and_tools cluster exists in DSPy issues

The held-out topic does not emerge at any tested granularity
(min_cluster_size 30/10/5): the ~26 react/tool/agent-flavored seeds scatter
across noise (7), integrations (8), bug reports (6), and others. The five
test questions embed tightly together and inside none of the issue
clusters (`artifacts/clusters.png`). Interpretation: the paper's community-
session source (users asking how to use DSPy) is materially different from
public GitHub issues (bug reports, integrations, setup), so distribution-
matching the StudyBench topic from issues alone is not possible via
literal cluster selection. Selection therefore fell back to
label-conditioned keywords — 25 seeds, pattern and members recorded in
`artifacts/clusters.json` — of which 20 were sampled (seed 20260716) as
generation anchors (~15/20 on-topic on manual inspection; the A.2 prompt
is explicitly designed for noisy seeds and the critic drops off-label
items).

## Controls (both passed before any bundle was accepted)

- **Sandbox control**: all 5 released SmallDSPy gold answers run clean in
  `.venv-dspy` (dspy at the pinned corpus commit `9cdb0aa`).
- **Self-grade control**: all 5 released golds score exactly **100/100**
  against their own rubrics under our paper-contract GPT-5.4 judge — the
  Stage-5 "gold must self-score 100" gate is calibrated; failures on our
  bundles indicate real bundle defects, not harness noise.
- **Prompt fidelity**: 10-probe verbatim spot-check of the A.2/A.3/A.4
  transcriptions against docs/paper.md passes.

## Results

`data/smalldspy_ourvalidationset.jsonl`: **5 questions**, all
`react_agents_and_tools`, schema- and key-order-identical to the released
dataset (verified against `load_questions`-equivalent checks from disk).

- **s3 generator** (GPT-5.4 xhigh, Responses API, 80 tool calls, ~2.6M input
  tokens): 12 candidates, zero correction rounds; all behaviorally framed
  and traceably anchored on seed issues (#157 duplicate-tool collision,
  #1518 ReAct state saving, #8896 async tools, #8195 History-in-ReAct).
- **s3 critic** (66 tool calls, re-verified candidates against source): 5
  finalists spanning trajectory truncation, sync/async tool execution,
  stage-specific History handling, tool-name collisions, and multimodal
  trajectory preservation; written rejection rationale (single-locator and
  off-label items dropped).
- **s4 rubrics**: 4-6 claims each, core weight 80-95, 5-10 spans each, zero
  drops, all excerpts byte-exact against the pinned corpus.
- **s5**: all 5 passed. One bundle needed one revision round (two fenced
  blocks -> one). Self-grades all exactly 100/100. Textual decontamination:
  zero shared 3-word shingles with any test question (computation verified
  live: identical-text jaccard = 1.0, non-empty shingle sets).
- **Embedding audit** (`audit_similarity.py`, same embedding model): max
  cosine to any test question 0.829-0.916, all under the 0.95 flag
  threshold; each generated question sits near a real seed (0.889-0.899),
  consistent with genuine seed anchoring. The 0.916 pair (our
  History-visibility question vs the test per-user-history chatbot) is
  topically adjacent but mechanistically distinct - the intended
  "same distribution, not a duplicate" regime.

Manual adversarial verification performed on the artifacts:

- Three bundles deep-verified claim-by-claim against corpus source
  (truncation: 3-attempt cap, 4-key pop, override point, ValueError
  edge; tool collision: name-keyed dict, `type(func).__name__` fallback,
  explicit-name precedence, reserved `finish`; async: coroutine
  ValueError, `allow_tool_async_sync_conversion`, `acall` await path).
  Zero false claims found.
- All five gold programs' *printed outputs* checked against the answers'
  behavioral claims using the sandbox stdout recorded in
  `validation_report.json` (the exit-0 gate alone would not catch a wrong
  printed claim): all five match (truncation drops step-0 keys; sync error
  + async `5`; `False True`; collision key sets; `2` image blocks).
- Calibration controls: released golds run clean in the pinned sandbox
  (5/5) and self-grade 100/100 (5/5) under the same judge contract.

## Open item

`dspyval_143f185af3a4`'s gold program contains stylistically poor
defensive code introduced by the Stage-5 revision model (an unnecessary
`make_dummy_lm()` try/except ladder and a `hasattr(dspy, "configure")`
branch; the demo never invokes the LM). It runs clean, scores 100, and the
rubric does not reference it, so correctness is unaffected. Awaiting
Omar's call on sending it through one human-review revision round versus
keeping it as-is.

## Next

1. Decide train-set generation (rerun s3 with a different sample seed
   and/or the remaining seeds) now that the validation pipeline is sound.
2. Implement the self-quizzing study loop that consumes these questions
   (study object updated from verifier feedback on training questions;
   this validation set held fixed for reporting).
