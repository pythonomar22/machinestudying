# 006 — Self-quizzing iteration 1 (SmallDSPy)

**Status: in progress (2026-07-21).** Code in `studying/`, launched via
`scripts/selfquiz.sbatch`.

## Hypothesis

A quiz→attempt→verify→distill loop over the SmallDSPy corpus produces a
better study object (cheatsheet) than the paper's single-pass exploration
cheatsheet, measured as expertise on the 5 held-out SmallDSPy test
questions under the paper-contract GPT-5.4 judge, paired against
`smalldspy-nostudy-20260715` (12.35) and `smalldspy-cheatsheet-20260716`
(19.18) with the same master seed 20260715.

## Method

Per round (5 rounds, 6 questions each):

1. **Quizmaster** (GPT-5.4 xhigh in `codex exec`, read-only at the pinned
   66-file corpus) writes practice questions in the register of our
   validation set (`data/smalldspy_validation.jsonl`, provenance in
   `data_collection/`, textual decontamination vs the test set: max
   3-gram Jaccard 0.010). Gold answers are single fenced offline programs;
   each is validated deterministically and must run to exit 0 in the pinned
   `.venv-dspy` sandbox before the question is accepted. From round 2 the
   prompt includes prior rounds' per-question outcomes so questions target
   observed weaknesses. Generated questions are additionally
   decontaminated-by-dropping against the test set (Jaccard > 0.35).
2. **Studier** (Qwen3.5-9B, paper ReAct harness + tools, current cheatsheet
   prepended) attempts each question, up to 20 voluntary iterations.
3. **Verifier**: the attempt's fenced program runs in the same sandbox;
   GPT-5.4 grades the attempt against the gold program plus both sandbox
   outcomes and extracts mistakes and corpus-grounded lessons.
4. **Distiller**: the studier itself (Qwen3.5-9B) rewrites its cheatsheet
   from the findings (verdicts, mistakes, lessons, gold programs for missed
   questions), capped at 12,000 chars.
5. **Validation report**: the fixed 7-question validation set is evaluated
   (direct + k5, 1 rollout) and graded with the paper-contract GPT judge.
   Reporting only — results never reach the quizmaster, verifier, or
   distiller.

Final evaluation: `studybench.react --condition selfquiz` (new condition:
the note comes from the study loop; `study.json` is hash-bound into the run
manifest), full 4-budget × 3-rollout protocol, GPT-5.4 paper judge,
identical eval seeds to the paired baseline/cheatsheet runs.

## Declared cheats and caveats (ledger)

- **External teacher**: the quizmaster and verifier are GPT-5.4 — the same
  class of cheat the paper ledgered for its DeepSeek-written SFT questions.
  Knowledge flows from verified gold programs into the studier's cheatsheet.
  A "pure" self-quizzing variant (studier writes its own questions) is the
  follow-up. The *distiller* is deliberately the studier itself, so the
  study object has the same author as the baseline cheatsheet.
- **Validation-seeded training questions** (user-directed design): training
  questions are generated to match the validation register, so per-round
  validation gains are optimistic and the validation trajectory is a
  diagnostic, not an unbiased transfer estimate. The test measurement is
  unaffected (test questions touched only by decontamination and final
  eval).
- Study compute is recorded (`study.json`) but excluded from the evaluation
  token axis, exactly as the paper treats the cheatsheet's 50 study steps.
  This method spends far more study compute than the baseline cheatsheet;
  expertise comparisons are at matched *inference* budgets, and studying
  intelligence (expertise per study compute) is out of scope here.
- The cheatsheet cap (12,000 chars) exceeds the baseline cheatsheet's
  ~4.6 KB; note size is prompt-side and free on the generated-token axis,
  same as the baseline condition.

## Runs

- `runs/smoke-selfquiz` — smoke (1 round, 2 questions, k5 attempts,
  direct-only validation, 1-question eval). First attempt exposed a real
  bug: quizmaster gold programs tried to introspect the repository at
  runtime and failed in the isolated sandbox. Fixed by adding an explicit
  execution contract to the quiz prompt and folding sandbox verification
  into the corrective-retry loop. Deleted after each iteration.
- `runs/smalldspy-selfquiz1-20260721` — the full run (pending).

## Results

(pending)

## Interpretation

(pending)
