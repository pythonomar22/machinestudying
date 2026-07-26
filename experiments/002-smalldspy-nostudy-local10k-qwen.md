# SmallDSPy no-study baseline with the 10k local Qwen judge

Date: 2026-07-16

## Objective

Regrade the existing five-question SmallDSPy no-study rollout population with
the local Qwen3.5-9B judge using a 10,000-token thinking budget. This is an
offline diagnostic measurement and the paired local reference for the
cheatsheet run; GPT-5.4 remains the paper-tier judge.

## Inputs

- Rollouts: `runs/smalldspy-nostudy-20260715/smalldspy`
- Population: 5 questions × 4 budgets × 3 rollouts = 60 episodes
- Generator: Qwen3.5-9B revision
  `c202236235762e1c871ad0ccb60c8ee5ba337b9a`
- Run source: clean commit
  `7c3b317dbf87f919289db8d10e483f1f25c1f931`

## Command

```bash
SB_JUDGE=local10k \
sbatch scripts/grading.sbatch runs/smalldspy-nostudy-20260715
```

The job used four L40S GPUs as two TP=2 vLLM 0.24.0 replicas. The judge used
temperature 0, seed 0, thinking enabled, a hard 10,000-token thinking budget,
one request per replica, and a five-minute timeout per question.

## Result

| Budget | Mean lenient | Mean generated tokens |
|---|---:|---:|
| direct | 9.60 | 3,984.9 |
| k5 | 23.73 | 4,464.4 |
| k20 | 17.13 | 6,453.9 |
| forced k20 | 42.60 | 25,111.7 |

Expertise: **18.9785**

Artifacts:

- `grades/smalldspy-nostudy-20260715/qwen35-9b-thinking10k-local/smalldspy/grade.json`
- `grades/smalldspy-nostudy-20260715/qwen35-9b-thinking10k-local/smalldspy/report.json`

## Validation and interpretation

- Exactly 60 grades persisted, 15 per budget.
- Every verdict completed in one attempt with `finish_reason=stop` and
  `needs_regrade=false`.
- The two replicas served 28 and 32 verdicts.
- Judge completion lengths were 982 / 3,066.6 / 10,357 tokens
  (minimum / mean / maximum), consistent with the 10k thinking cap plus the
  structured verdict.
- No request, timeout, schema, GPU, OOM, or ECC failure occurred.
- An earlier launch was interrupted by an accidental live edit to its running
  batch script. Incremental persistence retained its five valid grades; the
  stable relaunch validated and skipped them, then completed the remaining
  population. No partial or invalid report was retained.

This number is not paper-comparable. It is a same-model-family diagnostic on a
custom five-question subset, and prior manual audits show that local Qwen can
materially overgrade answers. Its intended use is pipeline validation and
coarse comparison with other local-Qwen runs, with GPT and manual inspection
used for any research interpretation.
