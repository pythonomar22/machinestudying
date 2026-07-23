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

## Results (GPT-5.4 paper judge, graded 2026-07-22 19:14–19:19 UTC-adj, one batch era)

Grading ran after Omar's go-ahead, both arms back-to-back in a single
contiguous window with the same served judge snapshot
(`gpt-5.4-2026-03-05`), eliminating the 006 judge-drift caveat for this
pair. Zero regrades; all 720 lenient scores rebuild byte-exact from the
claim verdicts.

| Condition | direct | k5 | k20 | forced k20 | Expertise |
|---|---:|---:|---:|---:|---:|
| no-study | 4.5 (3.0k) | 8.2 (5.7k) | 9.8 (7.7k) | 24.0 (20.5k) | 9.04 |
| cheatsheet | 10.4 (2.8k) | 14.3 (5.4k) | 19.5 (7.5k) | 29.5 (24.1k) | **15.90** |
| paper base | 3.3 (4.1k) | 8.6 (7.9k) | 9.6 (8.6k) | 29.4 (34.6k) | 6.49 |
| paper cheatsheet | 6.3 (3.9k) | 14.4 (6.1k) | 14.1 (7.1k) | 23.1 (29.9k) | 9.65 |

**Cheatsheet − no-study = +6.85 expertise, paired cluster bootstrap over
the 30 questions (20k resamples): 95% CI [+3.00, +10.83],
P(diff ≤ 0) = 0.0001.** Direction consistent in all three rollouts
(16.5/8.1, 16.2/7.9, 15.1/10.7). This is the project's first
statistically separable condition difference — n=30 delivers what n=5
could not (006/007).

## Adversarial audit (workflow `wf_9bf5b3fb-d63`, 5 auditors + verify)

Everything material survived. Full findings in the workflow transcript;
the load-bearing points:

- **Pairing/contamination clean**: all 720 seeds recomputed from the
  master seed; note delivery re-proven by hash; study confined to the
  corpus by construction (tool layer); max note↔question overlap is a
  4-token phrase; both arms at commit 57fd39d with matching manifests.
- **Grades mechanically exact**: lenient = weighted claim sum in 720/720;
  grades hash-bound to episode bytes; single grading era. On 19/720
  verdicts the judge's redundant freetext `question_score` disagrees with
  its own claim verdicts; the pre-registered contract (claim-weighted
  sum) resolves these, and the alternative resolution would give +6.53 —
  direction and significance unchanged.
- **The gain is distributed, not a lottery** (contrast 007): 23/30
  questions positive, wins>losses at every budget (153W/65L/142T),
  leave-one-question-out range [+6.09, +7.65], top question carries
  10.6% of the effect (vs ~97% for the same JSONAdapter question on
  smalldspy). Caveats: top-5 questions carry ~51% of the magnitude;
  signature_schema topic is mildly negative (−1.45 E).
- **Accuracy, not token-axis**: freezing the token axis at baseline
  attributes +6.73 of the +6.85 to accuracy; the token channel is ≤0.12.
  The smalldspy answer-shortening concern does not apply here.
- **Judge leniency bounded**: adversarial spot-check of the 15 highest-
  leverage episodes found 2 confirmed over-credits, both in the cheat arm
  (87 rubric points total); zeroing every identified over-credit moves
  the gap to ≈+6.5, and an extreme extrapolated haircut leaves +4.56,
  inside the CI. Judge is condition-blind; note-parroting is not the
  scoring driver.
- **Mechanism is note-matched at direct**: +5.90 of the +5.97 direct-
  budget gain lands on rubric claims naming APIs that appear verbatim in
  the cheatsheet; the one topic the note never covers goes negative.
  ~12% of the total gain sits on GEPA claims the note never mentions
  (tool budgets only — indirect channel). The note itself is partly
  pretraining recall (some hallucinated APIs; 3 of 6 exam topics absent),
  so describe the effect as "note-surfaced knowledge", not pure corpus
  extraction.

## Interpretation

1. **The paper's Table-1 DSPy finding replicates on our infrastructure
   and is separable from zero**: cheatsheet > no-study at every budget,
   +6.85 E (paper: +3.16). Our absolute numbers run higher than the
   paper's (9.04/15.90 vs 6.49/9.65), consistent with our persistently
   shorter forced-k20 answers (20.5–24.1k vs ~30–35k tokens) shifting
   AUC mass; cross-paper absolute comparisons remain out of scope.
2. These two runs are now the fulldspy anchor pair for the self-quizzing
   line: any studying method we propose must beat **15.90** under these
   exact seeds, budgets, and judge contract to claim an improvement over
   the paper's own study baseline.
3. Known biases to carry forward: ~0.3–0.4 E plausible judge-leniency
   inflation on the gap (bounded by the audit), a mildly heavy top-5
   tail, and one negative topic. None threaten sign or significance.

## Next

1. Re-run the self-quizzing loop (with the 007 iteration-3 levers) at
   fulldspy scale against this anchor pair.
2. Decide the 007 open design question (validation-drop revert guard)
   before iteration 3.
3. Optional: pin the dspy corpus snapshot hash in `dataset.py` the way
   smalldspy is pinned (audit hygiene note).
