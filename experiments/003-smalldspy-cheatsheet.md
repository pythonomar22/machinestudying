# SmallDSPy cheatsheet study and evaluation

Date: 2026-07-16

## Objective

Run the paper-style cheatsheet condition on the custom five-question
SmallDSPy subset, save the study artifact beside its rollouts, and grade the
same 60-episode evaluation population with both GPT-5.4 and the local
Qwen3.5-9B 10k-thinking judge.

## Commands

These are the complete user-facing workflow:

```bash
SB_RUN_ID=smalldspy-cheatsheet-20260716 SB_SEED=20260715 \
sbatch scripts/cheatsheet.sbatch

set -a; source .env; set +a; uv run --frozen python -m studybench.grade \
  runs/smalldspy-cheatsheet-20260716 --judge gpt

SB_JUDGE=local10k \
sbatch scripts/grading.sbatch runs/smalldspy-cheatsheet-20260716
```

GPT and local grading can run in parallel after the rollout command finishes.
For this execution, the batch scripts were run inside the existing four-L40S
interactive Slurm allocation with `bash`; the public commands above remain
`sbatch`. The local launch also set `SB_DEBUG=1`, retaining the complete raw
judge requests and responses.

## Study artifact

The run was generated from clean commit
`6f43adddf6671e54c0c1b2a75d837710a80b474f` and wrote:

- `runs/smalldspy-cheatsheet-20260716/smalldspy/study.json`
- `runs/smalldspy-cheatsheet-20260716/smalldspy/cheatsheet.md`
- `runs/smalldspy-cheatsheet-20260716/smalldspy/run.json`
- 60 evaluation episodes below `episodes/`

The study used only the pinned 66-file SmallDSPy corpus, whose snapshot hash is
`edfd5e412afa87ff13e24c1515157c71199fa3e92c1a95e06bde4372ff450b5a`.
It completed exactly 50 accepted ReAct turns: 48 repository calls and two
correctly intercepted early `finish` attempts. The 53 LM calls comprise those
50 turns, two malformed-format fallback calls, and one final extraction. The
study generated 50,193 tokens. Its 4,602-byte cheatsheet exactly matches the
answer stored in `study.json`, and its hash is
`a7a8fe21df9e144a6775f7cd50040467ec8183526418a60539b09a4f452f838c`.

An independent audit replayed every repository observation, reconstructed the
corpus and dataset hashes, and checked for held-out-question leakage. It found
zero leakage or provenance errors.

## Evaluation integrity

The evaluation contains exactly 15 episodes for each of `direct`, `k5`,
`k20`, and forced `k20`, all with `status=ok`. Every forced episode contains
exactly 20 turns. Evaluation seeds exactly match the paired no-study run; only
the intended cheatsheet prefix changes the question hash.

The first pass produced 59 valid episodes and one explicit DSPy
`AdapterParseError` when Qwen omitted required action fields. Rerunning the
same cheatsheet command and run ID preserved the study and 59 valid episodes,
selected exactly one pending episode, and completed it successfully with the
same intended seed. No invalid rollout was accepted or hidden.

Mean evaluation generation tokens were:

| Budget | Mean generated tokens |
|---|---:|
| direct | 2,795.9 |
| k5 | 4,216.3 |
| k20 | 5,286.3 |
| forced k20 | 21,404.0 |

Study tokens are recorded separately and are not included on this evaluation
token axis.

## GPT-5.4 paper-tier result

| Budget | No study | Cheatsheet | Change |
|---|---:|---:|---:|
| direct | 7.13 | 15.40 | +8.27 |
| k5 | 14.80 | 15.13 | +0.33 |
| k20 | 16.07 | 19.73 | +3.67 |
| forced k20 | 26.47 | 29.13 | +2.67 |

- No-study expertise: **12.3533**
- Cheatsheet expertise: **19.1767**
- Difference: **+6.8234**

One GPT verdict had internally consistent claim judgments but faulty redundant
arithmetic: weights 45, 15, and 10 were awarded, while GPT wrote
`question_score=60` instead of 70. The raw response and raw 60 are preserved in
the grade artifact; the canonical lenient score is the mechanically recomputed
weighted claim sum, 70. This follows the paper and Jacob's clarification that
lenient grading is the pure weighted sum. Using the raw 60 would change
expertise only from 19.1767 to 19.0832, so this does not explain the measured
gain.

The paper prompt, response schema, and claim decisions were not changed.

## Local Qwen 10k diagnostic result

| Budget | No study | Cheatsheet | Change |
|---|---:|---:|---:|
| direct | 9.60 | 17.73 | +8.13 |
| k5 | 23.73 | 18.40 | -5.33 |
| k20 | 17.13 | 34.40 | +17.27 |
| forced k20 | 42.60 | 35.73 | -6.87 |

- No-study local expertise: **18.9785**
- Cheatsheet local expertise: **27.4746**
- Difference: **+8.4961**

All 60 local verdicts completed in one attempt with `finish_reason=stop` and
`needs_regrade=false`. The two replicas served 28 and 32 verdicts. Completion
lengths were 863 / 3,113.75 / 10,364 tokens (minimum / mean / maximum). A vLLM
`EngineDeadError` appeared only after the final HTTP 200 during deliberate
abort-mode server teardown; all grades and the report had already completed.
The batch job's five-verdict end-to-end smoke completed before the remaining
55 verdicts were requested.

Local Qwen substantially overgrades several nested-router answers relative to
GPT, including k20 disagreements of 40-60 points. Treat 27.4746 as a diagnostic
pipeline result, not a paper-quality method estimate. Across the 60 episodes,
the judges agree exactly on 34; local scores 21 higher and five lower, with a
mean local bias of +6.72 points and mean absolute disagreement of 8.88. For
example, local awards 75 versus GPT's 15 to one nested-router answer while
incorrectly claiming it constructs ReActs in `__init__`, uses bare callables,
and implements composed `forward` delegation.

## Adversarial interpretation

The GPT result is an exact measurement for these artifacts, but not a robust
general finding:

- SmallDSPy has only five questions, all from one
  `react_agents_and_tools` topic cluster.
- The gain is concentrated in the JSONAdapter question, which contributes 302
  raw points across its 12 episodes; three of five questions decline overall.
- The cheatsheet explicitly mentions `JSONAdapter()` and scoped
  `dspy.context`, so that concentrated gain is mechanistically plausible.
- Across paired episodes, 19 improve, 18 worsen, and 23 tie.
- Leave-one-question-out gains remain positive but range from about +2.42 to
  +10.09 expertise points; a diagnostic cluster bootstrap spans roughly
  -0.84 to +20.05.
- Lenient grading scores rubric claims without compilation. Some high-scoring
  nested-router programs are not runnable, which is expected for this metric
  but limits what the score means.

The honest conclusion is: this particular SmallDSPy cheatsheet run scores
higher than its paired no-study run under both judges, and the paper-tier GPT
measurement is 19.1767 versus 12.3533. More questions, study seeds, or repeated
study runs are required before claiming a stable studying improvement.
