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

Study run `smalldspy-selfquiz1-20260721` (5 rounds; one mid-run resume after
round 4 generation initially yielded 2/6 surviving questions — fixed by
allowing a third corrective codex attempt, commit `f6d6e7d`; rounds 1–3
replayed from cache). Per-round training verdicts: 0 correct anywhere;
3 partial in round 1, 1 in round 2. Studier study compute: 232k generated
tokens across attempts + 18k distillation (external teacher compute
excluded, recorded in artifacts).

Validation trajectory (mean lenient, 7 questions × 1 rollout — noisy):

| round | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|
| direct | 0.00 | 5.71 | 0.00 | 2.86 | 0.00 | 2.86 |
| k5 | 2.86 | **25.00** | 15.00 | 2.86 | 12.14 | 5.71 |

Held-out test (GPT-5.4 paper judge, seeds exactly paired with both
baselines; lenient accuracy / mean generated tokens):

| Condition | direct | k5 | k20 | forced k20 | Expertise |
|---|---:|---:|---:|---:|---:|
| no-study | 7.13/4.0k | 14.80/4.5k | 16.07/6.5k | 26.47/25.1k | 12.35 |
| cheatsheet | 15.40/2.8k | 15.13/4.2k | 19.73/5.3k | 29.13/21.4k | **19.18** |
| selfquiz (final note) | 11.40/3.4k | 14.67/4.9k | 17.73/7.0k | 21.27/21.3k | 13.75 |

## Interpretation (iteration 1, honest)

The specified protocol — evolve for 5 rounds, evaluate the final note —
**does not beat the exploration cheatsheet** (13.75 vs 19.18); it lands
between baseline and cheatsheet. Two mechanisms visible in the artifacts:

1. **Full-rewrite distillation is unstable.** The note is rewritten from
   scratch each round at temperature 1.0; validation peaked at round 1 and
   degraded as later rewrites dropped earlier content. The final note is
   not the best note the loop produced.
2. **Training difficulty outran the studier.** Verdict counts collapsed to
   all-incorrect by rounds 3–5, so later rounds distilled from failures on
   very hard questions rather than consolidating basics.
3. At forced k20 the selfquiz note scores below even no-study (21.27 vs
   26.47), consistent with the paper's observation that long search
   amortizes notes away — and suggesting this note actively misdirects
   long searches.

**Selection variant (declared before running):** because round selection by
*self-generated* validation score is test-blind and part of a legitimate
studying algorithm, we also evaluate the validation-selected note
(round 1, argmax of mean validation lenient across budgets: 15.4 vs ≤6.1
for all other rounds) as `smalldspy-selfquiz1sel-20260721`. This is the
second and final test-set evaluation of this method family in this
iteration; both results are reported regardless of outcome.
