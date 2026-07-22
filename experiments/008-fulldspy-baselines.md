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

- `runs/dspy-nostudy-20260722` — complete. 360/360 episodes `ok`. First
  pass produced 359 valid episodes plus one transient k5
  `AdapterParseError` (`dspy_136c0583928c`, the known temp-1.0
  malformed-action mode from experiments/003); rerunning the same command
  completed that single episode with its intended seed. Mean generated
  tokens: direct 3.0k, k5 5.7k, k20 7.7k, k20f 20.5k.
- `runs/dspy-cheatsheet-20260722` — complete. Study: 50 forced iterations
  exactly (38 repository calls + 12 intercepted `finish` attempts), 41,181
  generated tokens, 13,536-byte cheatsheet identical to the `study.json`
  answer and to the manifest `note_sha256`. Evaluation: 359 `ok` + 1
  `no_answer` (k20/r2 `dspy_a5b116f00083`, empty answer after 3
  iterations — a valid scored-as-written outcome, kept by design). First
  pass left 4 transient `AdapterParseError` episodes (2×k5-class parse
  failures, 2×k20f forced-trajectory aborts at the same parse mode);
  one rerun of the same command completed all 4 with their intended
  seeds. Mean generated tokens: direct 2.8k, k5 5.4k, k20 7.5k,
  k20f 24.1k.

**Debug-flag restart (ledgered).** The first cheatsheet launch inherited
`scripts/cheatsheet.sbatch`'s `SB_DEBUG=1` default while the completed
baseline ran `debug=false`; `debug` is in the reporter's arm-compatibility
contract (the smalldspy pair ran both arms `debug=true`). The partial
debug-mode run (study + 77 episodes) was deleted ~30 minutes in and
relaunched with `SB_DEBUG=0` to match the baseline — cheaper than
redoing the finished 80-minute baseline at `debug=true`, and it avoids
per-episode debug-history bloat across 360 episodes. `study.json` still
records the full 50-turn study trajectory without debug mode. The
deleted run's 10,795-byte note was produced from the same study seed;
the relaunched study re-samples (GPU inference is not bitwise
reproducible; experiments/001), so the retained note is simply the one
the retained run produced.

One relaunch attempt was refused with "full evaluations require a clean
committed source tree": in-progress edits to this very file had dirtied
the tree, and committing them was not an option because both arms must
record the same `source_commit` (57fd39d). The doc edits were stashed
for the launch and restored immediately after; both arms therefore ran
identical code at 57fd39d.

## Pre-grading verification (2026-07-22, passed in full)

Checked mechanically from disk before any grading:

1. All 18 reporter-COMPATIBLE manifest fields identical across arms
   (source commit 57fd39d, debug false, corpus snapshot, dataset hash,
   model revision, runtime/GPU shape, sampling, tools, budgets, rollouts,
   master seed, question ids).
2. Identical 360-episode grids; per-episode seeds equal pairwise across
   arms at every (budget, rollout, question) key.
3. Every episode valid: statuses above, all 90 forced episodes exactly
   20 iterations in both arms, all token ledgers integral.
4. Note delivery verified by hash: all 360 baseline episodes saw the raw
   question; all 360 cheatsheet episodes saw NOTE_PREFIX + note +
   question (`question_sha256` recomputed from data + cheatsheet.md).
5. Generation-token regime is sane against the paper (base direct 3.0k
   vs paper 4.1k; k20f 20.5k vs 34.6k — same compression we observed on
   smalldspy, where local k20f was 25.1k).

## Results

Rollouts complete and verified; **grading deliberately not started** —
Omar asked for the runs to be ready but ungraded (2026-07-22). The
staged grading commands are in the Commands section above. Both GPT-5.4
gradings should run in one batch era to avoid the judge-drift caveat
from the 006 audit.

## Interpretation

Pending grading.
