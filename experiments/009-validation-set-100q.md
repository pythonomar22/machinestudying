# 009 — Validation set rev-3: 100 questions (artifacts2)

**Status: in progress (2026-07-22).**

## Objective

Scale the StudyBench-replica validation set from 32 to 100 questions for
the fulldspy self-quizzing line (experiments/008 established the anchor
baselines; 32 validation questions are too few for per-round validation
signal — 007's post-mortem P5 found validation-direct was ~90% zeros at
n=7). Same pipeline, same paper ratios, new artifacts tree
`data_collection/artifacts2/` selected via `DC_ARTIFACTS`.

## Design (full fidelity ledger: data_collection/FIDELITY.md, "Rev-3")

- **Cloned**: stage 1–3 artifacts byte-for-byte; stage-3a batch 0 = the
  rev-2 generation outputs (batch-0 seeds and prompt asserted
  byte-identical before adoption).
- **New**: `DC_GEN_BATCHES=3` → 60 candidates/topic (3 proven 20-candidate
  codex sessions, centroid-rank-partitioned seeds, anti-dup context);
  critic keeps round(60·5/12) = 25/topic → 100 finalists; stages 6–7
  unchanged (rubrics, sandbox + self-grade-to-100 optimization).

## Commands

```bash
cd /matx/u/omarah/ms2
DC_ARTIFACTS=artifacts2 DC_GEN_BATCHES=3 uv run data_collection/4_generate_candidates.py fulldspy
DC_ARTIFACTS=artifacts2 DC_GEN_BATCHES=3 uv run data_collection/5_critic_selection.py fulldspy
DC_ARTIFACTS=artifacts2 DC_GEN_BATCHES=3 DC_WORKERS=6 uv run data_collection/6_build_rubrics.py fulldspy
DC_ARTIFACTS=artifacts2 DC_GEN_BATCHES=3 DC_WORKERS=6 uv run data_collection/7_optimize_bundles.py fulldspy
```

All stages idempotent per topic/bundle; reruns pick up failures.

## Results (complete, 2026-07-22)

**`data_collection/artifacts2/7_optimize_bundles/7_fulldspy_validation.jsonl`
— 100 questions, 25 per topic, zero drops.**

- Stage 4: 240/240 candidates (60 × 4 topics), all unique. Batch-0
  clone adopted after byte-identity assertions (seeds and prompt equal
  to rev-2); 3 of 8 new batch sessions needed one corrective retry
  (single-fence violations — same class rev-2 saw). SmallDSPy-scope
  candidates: 46 (rev-2: 21).
- Stage 5: 100 finalists (25/60 per topic, paper's 5/12 selectivity),
  all unique, 53 hard / 47 very_hard, median question 653 chars, kept
  indices span all three batches in every topic.
- Stage 6: 100/100 rubric bundles first-pass valid population-wide
  (weights sum 100, core majority everywhere; median 5 claims, 6 spans).
- Stage 7: **100/100 pass — zero drops**; every gold program compiles,
  runs to exit 0 in the pinned sandbox, and self-scores exactly 100;
  9/100 needed one revision round (rev-2: 3/32).
- Derived SmallDSPy-scope validation set: **14 questions** (rev-2: 7),
  written alongside as `7_smalldspy_validation.jsonl`.
- Decontamination screen vs the 30 held-out Study-DSPy test questions:
  max 3-gram Jaccard **0.028** (drop threshold 0.35) — nothing close to
  contamination; no question dropped.
- Released-format check: all 100 rows pass a
  `studybench.dataset.load_questions`-equivalent contract validation
  (field set, unique ids, weight sums, claim types, span references,
  path roots).

The rev-2 32-question set in `artifacts/` remains untouched and is
still what experiments 006–007 reference. Rev-3 (this set) is the one
future fulldspy self-quizzing iterations should consume.

## Residuals (carried honestly from rev-2, unchanged)

No human review (declared gap, no substitute); question lengths run
below the released benchmark register (median ~654 vs 856–1,848);
self-grading uses the same model family that wrote the answers. New in
rev-3: batch 2 of the 22-member evaluation_metrics topic reuses 8
batch-0 seeds (wrap-around) with anti-duplication carried by the prompt
and the critic; cross-set overlap with rev-2's 32 questions is expected
by construction (shared batch-0 candidates) — the sets are alternatives,
not independent draws.
