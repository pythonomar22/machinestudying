# 007 — Self-quizzing runs (pre-registered 2026-07-06, before any results)

Design: experiments/005 (v1.2 operational spec). Code: `studybench/selfquiz.py`
(+ `scripts/selfquiz.sbatch`). Harness: the faithful react stack throughout
(experiments/006). Judge: fugu. Metric: pure-sum lenient + WAUC.

## Pre-registration (frozen before round 1 results)

**Procedure config:** K=4 chapters/round, M=5 questions/chapter (1 → dev exam),
retest fraction 20% (r≥2), quiz/derive episodes ≤15 tool iterations, derive
ensemble = 1 for DSPy (execution-grounded) / 2-with-agreement for OpenClaw,
note soft cap 4k tokens, quote gate ±2 lines. Chapters = first-level dirs under
corpus roots, LOC-descending, test dirs excluded (DSPy: 15 chapters; OpenClaw:
169 — fine-grained src/, the wrap-around syllabus covers the largest 4r by
round r; documented consequence of the corpus-agnostic rule, not tuned).

**Study budgets & milestones:** rounds 1, 2, 4, 8 snapshot the note
(`study-selfquiz/{task}/note-r{r}.md`) and get a milestone eval: the standard
4-budget × 3-rollout react grid with the note prepended (identical mechanics to
react-cheatsheet), fugu-graded.

**Comparisons (all paired per question, same judge, same harness):**
1. selfquiz-r* vs react base — does studying help at all?
2. selfquiz-r* vs react-cheatsheet — does error-driven note construction beat
   read-and-summarize at the same artifact type? (THE claim)
3. WAUC vs study-tokens curve across r (cheatsheet = one point at ~64k).

**Success criterion:** paired WAUC delta vs react-cheatsheet > 0 with the 95%
paired bootstrap CI excluding 0, on DSPy, at any pre-registered milestone.
OpenClaw is reported but pre-registered as underpowered/lower-trust (n=20, no
execution grounding).

**Pre-registered effect ordering:** selfquiz-vs-cheatsheet delta larger on DSPy
(stale-prior correction + execution) than OpenClaw. The reverse falsifies the
error-delta mechanism story and will be reported as such.

**Gates before results count:**
- Smoke (1 chapter × 3 questions, DSPy): human-read every artifact — question
  quality, attempt, blind derivation + probes, verdict, note entry.
- Round-1 dev-verdict audit: human agreement ≥80% on ~30 sampled verdicts
  before the dev metric may steer any procedure change; re-audit after any
  VERIFY prompt change.
- Leakage checklist per experiments/005 §4 (study tools mount corpus roots
  only — no data/, runs/, grades/, experiments/ reachable; no benchmark
  phrasing in study prompts; no design change motivated by test-score
  movement).
- Health bands per round: train error rate 30-70% healthy; quote-gate bounce
  rate logged (rising bounce = verifier degradation); dev pool accumulates.

**Iteration policy:** any tweak to prompts/config between rounds must cite
dev-exam or artifact-inspection evidence in this file, never milestone scores.

## Run log

- 2026-07-06: smoke submitted (job 25397: DSPy, 1 chapter, 3 questions, 1 GPU).
