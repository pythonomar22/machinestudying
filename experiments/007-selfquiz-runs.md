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

- 2026-07-06: smoke job 25397 crashed on a dspy.ReAct kwarg collision (QuizSig
  input named `module` shadows ReAct internals) — renamed to `chapter`; also made
  zero-item rounds fail loudly. Smoke 25409 passed; all 3 artifacts hand-read:
  the closed-book model REINVENTED LabeledFewShot with numpy (Jacob's
  "reinventing wheels" failure, caught and corrected with a true cite), the
  max_errors pitfall produced a real cross-file correction (None → inherits
  settings.max_errors=10), the dev item was rightly judged correct. repo_map bug
  found in the note (root chapter listed subdir files) — fixed.
- 2026-07-06 round 1 (job 25439, 8 GPUs, both tasks):
  - DSPy: 20 items, train error 87.5% (above the healthy band top — the model
    genuinely lacks these chapters closed-book; fine for note-building),
    8 entries admitted / 6 bounced; 279k study tokens.
  - OpenClaw: 20 items, train error 93.8% (never-seen library, as pre-registered),
    11 admitted / 4 bounced, 0 ensemble downgrades; 407k study tokens.
  - **Iteration (artifact-cited, per policy):** manual read of all bounced
    entries showed two classes — true fabrications (prose quoted as code:
    correctly dead) and format failures (real source quoted across lines or
    inside backticks). quote_gate now normalizes to the first non-empty line and
    strips markdown wrapping before the exact-match check; unit-tested on the
    actual bounced strings (real ones recover, prose still bounces). Recovery
    pass over items.jsonl: +3 entries per task (marked entry_recovered).
    Also: repo map capped at 20 chapters (OpenClaw's 169-line map bloated the
    note to 18.8k chars). Final r1 notes: dspy 11 entries/6.3k chars,
    openclaw 14 entries/8.9k chars.
  - Entry quality (manual read of all 25): precise, real-cite corrections —
    e.g. TwoStepAdapter's `extraction_model` arg, `rollout_id` stripped at
    temperature=0, ReAct's Tool-wrapping (dspy); gateway auth token precedence
    via `??`, strict-lowercase "session_expired", dispatcher unregister-at-zero
    (openclaw).
  - **Dev-verdict audit: 8/8 human agreement** (one arguable partial-vs-wrong).
    Accumulated 8 of the ~30 required before dev metrics may steer decisions.
  - Milestone eval r1 submitted: jobs 25455 (dspy) + 25456 (openclaw), variant
    react-selfquiz-r1, fugu grading armed.
