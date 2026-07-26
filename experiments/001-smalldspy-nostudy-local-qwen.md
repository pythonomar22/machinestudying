# SmallDSPy no-study baseline with local Qwen grading

Date: 2026-07-16

## Objective

Exercise the complete minimal workflow on the custom five-question SmallDSPy
development subset: retain the existing no-study rollout population, grade it
offline, and verify that the resulting report is complete and internally
consistent.

## Inputs

- Rollouts: `runs/smalldspy-nostudy-20260715/smalldspy`
- Questions: `data/smalldspy.jsonl`
- Corpus: `corpora/smalldspy` at DSPy commit
  `9cdb0aac28b2a04b064e40697ccd301872cf6a43`
- Generator: Qwen3.5-9B revision
  `c202236235762e1c871ad0ccb60c8ee5ba337b9a`
- Population: 5 questions × 4 budgets × 3 rollouts = 60 episodes

## Grading

Command:

```bash
sbatch scripts/grading.sbatch runs/smalldspy-nostudy-20260715
```

The job used four L40S GPUs as two TP=2 vLLM 0.24.0 replicas. The judge was the
same pinned Qwen3.5-9B revision, with temperature 0, seed 0, thinking enabled,
a hard 4,000-token thinking budget, one request per replica, one semantic
attempt, and a five-minute request timeout. The five-cell smoke passed before
the remaining 55 judgments ran.

## Persisted result

| Budget | Mean lenient | Mean generated tokens |
|---|---:|---:|
| direct | 10.07 | 3,984.9 |
| k5 | 21.40 | 4,464.4 |
| k20 | 19.60 | 6,453.9 |
| forced k20 | 41.93 | 25,111.7 |

Expertise: **17.6474**

Artifacts:

- `grades/smalldspy-nostudy-20260715/qwen35-9b-thinking-local/smalldspy/grade.json`
- `grades/smalldspy-nostudy-20260715/qwen35-9b-thinking-local/smalldspy/report.json`

## Validation

- Exactly 60 grade artifacts persisted, 15 per budget.
- Every response completed in one attempt with `finish_reason=stop` and
  `needs_regrade=false`.
- Completion usage ranged from 943 to 4,325 tokens; the upper tail is
  consistent with 4,000 thinking tokens plus the final structured verdict.
- Raw responses, prompts, schemas, claims, arithmetic, run provenance, and
  report aggregates all revalidated from disk.
- The two replicas served exactly 28 and 32 judgments; no request, schema,
  model, timeout, OOM, GPU, or ECC failure occurred.

## Interpretation

This is a development diagnostic, not a paper-comparable StudyBench number.
SmallDSPy is a custom five-question subset, and Qwen is judging outputs from
the same model family. GPT-5.4 with the paper contract remains the
paper-comparable judge.

An adversarial manual audit checked the lowest- and highest-scoring answer for
each of the five questions. Five of those ten anchors were materially
misgraded, including four of the five highest-scoring answers:

- Text adventure `k20/r2`: 100 should be 80; the answer omitted the required
  output field, so its undeclared `goal_completed` value is discarded.
- Per-user history `k20f/r2`: 12 should be 40 under the rubric's independent
  static claims; the judge incorrectly imported another claim's
  `dspy.History` requirement.
- Typed carpool `k20f/r0`: 25 should be 0; there is no typed rider signature
  input and the dict-based confirmation cannot produce the required booking.
- JSONAdapter planner `k20f/r1`: 88 should be 33 strictly, or at most 68 under
  generous interpretation; required inputs are missing, tool construction is
  invalid, and the answer recommends the known-broken adapter path.
- Nested router `k5/r0`: 75 should be 15; the program does not parse or run,
  budgeting and dispatch are malformed, and the router-layer tool requirement
  is not met.

This was systematic rather than isolated: for example, the judge awarded the
typed-input carpool claim in 11 of 12 episodes largely from seeing Pydantic
models, without tracing the actual agent input and data flow. Correcting only
the five audited anchors would move the displayed expertise from 17.65 to
about 15.32, but that is not a corrected population estimate because other
misgrades remain. The persisted local scores therefore validate the pipeline
and expose judge failure modes; they are not reliable for exact method
comparisons or research claims without an independent paper-tier judge and a
calibrated manual audit.

An earlier otherwise equivalent local pass printed expertise 19.3544 before
its artifacts were accidentally removed by a teardown-script bug. Its budget
means were 7.60, 25.07, 20.93, and 40.93. Because those raw grades were not
retained, this is not a second auditable estimate, but the difference from the
persisted rerun is a warning that GPU local judging is not bitwise
reproducible even at temperature 0. Treat single-run local differences as
screening signals, and calibrate promising findings with repeated judging or
GPT-5.4 before making a research claim.
