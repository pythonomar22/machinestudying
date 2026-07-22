# 008 — Full Study-DSPy baselines: no-study and cheatsheet

**Status: in progress (2026-07-22).**

## Objective

Move from the 5-question SmallDSPy development subset to the real
30-question Study-DSPy benchmark (`data/dspy.jsonl`, the paper's Table-1
DSPy column) against the full pinned DSPy corpus (`corpora/dspy`, commit
`9cdb0aac28b2a04b064e40697ccd301872cf6a43`, 622 files). Produce the two
paper-condition baselines that every fulldspy studying method will be
paired against:

- **no-study**: direct/k5/k20/forced-k20, no note;
- **cheatsheet**: 50 forced study iterations over the full corpus, note
  prepended to every evaluation question.

This is the one-time full-corpus baseline pair ledgered as open item 3 of
experiments/005; statistical power was the binding constraint identified
in 006/007 (n=5 cannot rank methods; n=30 is the paper's own population).

## Paper targets (Table 1, DSPy column)

| Condition | direct | k5 | k20 | forced k20 | Expertise |
|---|---:|---:|---:|---:|---:|
| base | 3.3 (4.1k) | 8.6 (7.9k) | 9.6 (8.6k) | 29.4 (34.6k) | 6.49 |
| cheatsheet | 6.3 (3.9k) | 14.4 (6.1k) | 14.1 (7.1k) | 23.1 (29.9k) | 9.65 |

## Setup

Code changes for this experiment (no behavior change for smalldspy):

- `scripts/nostudying.sbatch` gains `SB_TASK` (default `smalldspy`);
  `studybench.react` was already task-parametric (`--task dspy` with the
  pinned dataset hash and corpus registered in `studybench/dataset.py`).
- `studybench.report` gains `--tasks` so a DSPy-only paper comparison no
  longer requires an OpenClaw population.

Protocol identical to the smalldspy runs: master seed 20260715, Qwen3.5-9B
revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, 4 budgets × 3
rollouts, two TP=2 vLLM 0.24.0 replicas on four L40S, paper sampling,
same three repository tools and caps. Population per condition:
30 questions × 4 budgets × 3 rollouts = 360 episodes.

## Commands

```bash
# smokes (1 question; baseline direct+k5, cheatsheet 2-iteration study + direct)
SB_RUN_ID=smoke-fulldspy-base SB_SEED=20260715 SB_TASK=dspy \
SB_BUDGETS=direct,k5 SB_ROLLOUTS=1 SB_SMOKE=1 SB_LIMIT=1 bash scripts/nostudying.sbatch
SB_RUN_ID=smoke-fulldspy-cheat SB_SEED=20260715 SB_TASK=dspy \
SB_BUDGETS=direct SB_ROLLOUTS=1 SB_SMOKE=1 SB_LIMIT=1 bash scripts/cheatsheet.sbatch

# full runs. matx1/matx2 were NOT_RESPONDING at launch time, so instead of
# sbatch these ran sequentially inside the 2-day 7×L40S interactive
# allocation on matx3 (job 16299512) with the public 4-GPU shape, exactly
# like the smalldspy runs of experiments/003 (bash, CUDA_VISIBLE_DEVICES
# limited to four GPUs):
SB_RUN_ID=dspy-nostudy-20260722 SB_SEED=20260715 SB_TASK=dspy sbatch scripts/nostudying.sbatch
SB_RUN_ID=dspy-cheatsheet-20260722 SB_SEED=20260715 SB_TASK=dspy sbatch scripts/cheatsheet.sbatch

# paper-tier grading + report
set -a; source .env; set +a
uv run --frozen python -m studybench.grade runs/dspy-nostudy-20260722 --judge gpt
uv run --frozen python -m studybench.grade runs/dspy-cheatsheet-20260722 --judge gpt
uv run --frozen python -m studybench.report \
  --baseline-run dspy-nostudy-20260722 --cheatsheet-run dspy-cheatsheet-20260722 --tasks dspy
```

## Runs

- `runs/dspy-nostudy-20260722` — pending
- `runs/dspy-cheatsheet-20260722` — pending

## Results

Pending.

## Interpretation

Pending.
