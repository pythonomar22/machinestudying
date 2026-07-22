# 007 — Self-quizzing iteration 2 (SmallDSPy)

**Status: designed 2026-07-21, from a forensic read of iteration 1's
artifacts (experiments/006, `runs/smalldspy-selfquiz1-20260721`). Written
before any iteration-2 code.**

## What the iteration-1 artifacts actually show

**D1 — The note taught the wrong things.** The winning exploration
cheatsheet (4.6 KB) is a compact tutorial of the *core* user-facing API:
Signatures, Example/Prediction, basic Predict, custom Modules,
configure/context, adapters, DummyLM mocking, save/load. Our selfquiz
notes are catalogs of whatever exotic mechanisms the quizmaster probed
that round (KNN order sensitivity, MultiChain legacy aliases, BestOfN
winning traces, image observation blocks). Meanwhile the studier's 27
training attempts failed overwhelmingly on *basics*: 10+ answers did not
even compile or run (SyntaxErrors, NameErrors, invalid walrus-in-assert),
plus stale/hallucinated imports (`from dspy.predict import Predictor`,
`dspy.settings.warn_on_type_mismatch`, `dspy.adpater...`) and offline-LM
harness misuse (custom `MockLM(dspy.Module)` instead of the shipped
DummyLM). The note never consolidated these foundations: DummyLM mentions
oscillated (5→4→2→2→7), no verified-imports section ever existed, and no
answer-format guidance was ever distilled.

**D2 — Full-rewrite distillation destroyed content.** The round-3 rewrite
collapsed the note from 23 headings to 7 (6,578 → 5,702 chars),
discarding the round-1/2 History and Parallel material. Rewriting the
whole note at temperature 1.0 each round makes retention a matter of
sampling luck.

**D3 — The curriculum ran far past the studier's frontier.** All 27
verdicts across rounds 3–5 were incorrect (27 total: 0 correct, 4
partial, 23 incorrect); the quizmaster kept generating benchmark-grade
hard/very_hard questions because its prompt inherited the benchmark
pipeline's difficulty floor. A studier that cannot yet produce a
compiling DummyLM harness learns nothing durable from failing an
image-observation-blocks question.

**D4 — Validation was noise and its ranking anti-transferred.** Each
trajectory point was 7 questions × 1 rollout at temp 1.0. The round-1
"peak" (k5=25.0) is one question (`dspy_296a3a478f07`) scoring 100 once;
per-question rows are otherwise mostly zeros. Validation ranked round 1
(mean 15.4) far above round 5 (4.3); the test inverted this (E 13.04 vs
13.75; direct 4.40 vs 11.40). Selection on this validation signal was
worse than doing nothing.

**D5 — The quizmaster escaped its sandbox.** Codex `-s read-only`
prevents writes but allows filesystem-wide *reads*: rounds 4–5 ran
`find /matx/u/omarah`, read full-DSPy files outside the 66-file corpus,
and executed a stray venv's python. Audited: zero test-set reads, no
leakage — but "corpus-only study" was not enforced, only assumed.
(Iteration-2's detector, run retroactively over all iteration-1 event
logs, found escapes in round 1 as well — probing the *installed* full
dspy package under site-packages — so 6 of 10 sessions escaped: rounds
1, 4, and 5.)
(For the record: the studier's attempt programs and all gold programs
*did* run inside the isolated `.venv-dspy` sandbox with stripped secrets
and timeouts; the escape was only the quizmaster's exploration shell.)

**D6 — The expertise metric rewards concise direct answers.** ~1.5–2 of
the cheatsheet's 5.4 E-point lead flows through shorter direct answers
(2.8k vs 3.4k tokens), a legitimate channel of E that our notes never
exploited: no note ever told the studier how to *answer*.

## Iteration-2 design (each change mapped to a diagnosis)

- **C1 (D1, D3) Exploration-initialized study object.** Round 0 is the
  studier's own 50-step forced exploration cheatsheet — the identical
  procedure to the baseline cheatsheet condition, fresh seed. Quiz rounds
  then verify and patch that note instead of building an exotica catalog
  from scratch. The tested hypothesis becomes: *does self-quizzing add
  expertise on top of exploration?* (Study compute is therefore 50 steps
  + 5 quiz rounds vs the baseline's 50 steps — recorded, and comparisons
  remain at matched inference budgets.)
- **C2 (D2) Structured notebook with deterministic edits.** The note is a
  list of titled sections with stable IDs (the exploration note is parsed
  into sections by heading). The distiller (still Qwen itself) emits
  schema-constrained JSON edit operations — add / replace / delete /
  append-to section — applied deterministically. Full rewrites are
  impossible; every round's diff is auditable; the 12,000-char cap is
  enforced by targeted compression requests, not silent truncation.
- **C3 (D3) Curriculum control.** A `medium` difficulty tier is added
  (the hard/very_hard floor was a benchmark-pipeline inheritance; practice
  questions are not benchmark items). The mix adapts to the last round's
  outcomes: correct rate < 1/3 → mostly medium probing core public API;
  1/3–2/3 → balanced; > 2/3 → escalate. Up to 2 slots per round are
  *retests*: a fresh, simpler scenario on a previously failed mechanism,
  so the loop measures whether a distilled lesson actually landed.
- **C4 (D1, D6) Foundations-first feedback routing.** The verifier now
  classifies every mistake: `answer_format`, `stale_or_hallucinated_api`,
  `offline_harness_misuse`, `mechanism_misunderstanding`, `other`. The
  distiller is instructed to route the first three classes into standing
  sections (answer format & style; verified imports / API surface;
  offline testing idiom) rather than minting exotic mechanism sections.
- **C5 (D4) Honest validation.** 2 rollouts per validation point (28
  episodes/round), report-only as before, and **no selection variant**:
  iteration 2 makes exactly one test-set evaluation (the final note).
- **C6 (D5) Enforced corpus-only quizmaster.** Every codex session's
  event log is parsed; any `command_execution` touching an absolute path
  outside the corpus checkout (plus `~` and `..` escapes; /bin,/usr,/dev,
  /proc,/etc allowlisted) rejects the batch, triggers a corrective retry,
  and is recorded. The prompt states the restriction explicitly.
- **C7 (D1) Coverage prior over the corpus surface.** The quizmaster
  prompt now contains the deterministic list of the corpus's public
  modules (dspy/predict, dspy/adapters, dspy/primitives, tests/predict)
  with the instruction to spread questions across that surface; the
  validation exemplars define *register only* (iteration 1's
  "distribution to match" wording is removed — it skewed topics away
  from major corpus components like ReAct).

Unchanged, deliberately: quizmaster/verifier remain GPT-5.4 (ledgered
external teacher), the distiller remains Qwen itself, 5 rounds × 6
questions, k20-voluntary attempts, 12k-char cap, test
decontamination-by-dropping, master seed 20260715, paper-protocol final
eval paired with both baselines.

## Run

- `runs/smalldspy-selfquiz2-20260721` — completed 2026-07-21, single
  uninterrupted job (no resumes, no code changes mid-study). Every
  quizmaster round passed the corpus-only read check with zero escapes
  and generated 6/6 questions on the first attempt (9–15 min/round at
  medium-weighted difficulty vs 27–47 min in iteration 1). The notebook
  evolved by edits only: 11 → 12 → 14 → 13 → 13 → 12 sections; the cap
  compression at rounds 3–5 engaged as designed. Training verdicts:
  0 correct anywhere; 2/1/1/2 partial in rounds 1/3/4/5 (curriculum
  pinned difficulty at its floor — mostly `medium` — from round 2 on).
  Studier study compute: 50-step exploration (49k gen tokens) + 231k
  attempt tokens + 23k distillation tokens.

Validation trajectory (mean lenient, 7 questions × 2 rollouts):

| round | 0 (explore) | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|
| direct | 0.00 | 2.14 | 5.36 | 4.64 | 2.86 | 0.00 |
| k5 | 5.36 | 2.86 | 6.43 | 10.36 | 8.21 | 3.00 |

Gentler and more monotone than iteration 1 through round 3, then a drop
at round 5 coinciding with the largest cap compression (10.3k → 7.5k
chars). Still noise-dominated at this population size.

## Results

Paper-tier GPT-5.4 grading (run after credits were restored, 2026-07-21
18:49; grading of the other four runs predates it — the 006 audit's
judge-drift caution applies across grading batches):

| Condition | direct | k5 | k20 | forced k20 | Expertise |
|---|---:|---:|---:|---:|---:|
| no-study | 7.1/4.0k | 14.8/4.5k | 16.1/6.5k | 26.5/25.1k | 12.35 |
| cheatsheet | 15.4/2.8k | 15.1/4.2k | 19.7/5.3k | 29.1/21.4k | **19.18** |
| selfquiz1 (final note) | 11.4/3.4k | 14.7/4.9k | 17.7/7.0k | 21.3/21.3k | 13.75 |
| selfquiz1 (r1 note) | 4.4/3.0k | 12.3/5.2k | 15.0/8.0k | 38.5/22.9k | 13.04 |
| **selfquiz2 (explore+quiz)** | 4.3/2.5k | 13.8/3.9k | 17.6/4.7k | 26.5/24.8k | **15.19** |

Paired cluster bootstraps over the 5 questions (20k resamples):
E(cheatsheet)−E(selfquiz2) = +3.65, 95% CI **[−8.67, +20.41]**;
E(selfquiz2)−E(no-study) = +3.52, CI [−1.64, +9.96];
E(selfquiz2)−E(selfquiz1) = +1.61, CI [−6.77, +7.37]. Nothing separates
from zero.

Interim diagnostic: all five conditions graded with the pinned local
Qwen3.5-9B 10k-thinking judge (same judge, same contract, one batch era —
known overgrading bias, not paper-comparable; see experiments/002):

| Condition | direct | k5 | k20 | forced k20 | local E |
|---|---:|---:|---:|---:|---:|
| no-study | 9.6/4.0k | 23.7/4.5k | 17.1/6.5k | 42.6/25.1k | 18.98 |
| cheatsheet | 17.7/2.8k | 18.4/4.2k | 34.4/5.3k | 35.7/21.4k | **27.47** |
| selfquiz1 (final note) | 8.4/3.4k | 26.3/4.9k | 32.4/7.0k | 30.7/21.3k | 20.89 |
| selfquiz1sel (r1 note) | 14.1/3.0k | 23.9/5.2k | 20.5/8.0k | 46.3/22.9k | 22.68 |
| selfquiz2 (explore+quiz) | 7.6/2.5k | 10.8/3.9k | 25.9/4.7k | 49.1/24.8k | 22.46 |

Local-judge reading (diagnostic only): iteration 2 sits above no-study
and iteration-1-final but still below the exploration cheatsheet. Its
per-budget shape is distinctive — the *worst* k5 in the table but the
*highest* forced-k20 cell of the whole project (49.1) — i.e. the
quiz-patched note appears to help sustained search and not cheap
answers, which the expertise anchor penalizes. Local orderings between
nearby conditions are not trustworthy (iteration 1 showed local/GPT
rank inversions); treat everything here as screening signal.

## Interpretation

- **Point estimates**: iteration 2 is the best self-quizzing variant so
  far (15.19 vs 13.75/13.04) and sits above no-study (+2.83), but still
  below the exploration cheatsheet (−3.99). All gaps have CIs spanning
  zero; these are descriptive statements about this 5-question set.
- **The decisive per-budget fact**: selfquiz2's direct collapsed to 4.33
  vs the cheatsheet's 15.40 — *even though iteration 2's note started
  from an exploration note of the same kind*. On this run, quizzing on
  top of exploration subtracted value at cheap budgets: the final
  (round-5, cap-compressed) note conditions direct answers worse than a
  plain exploration note does, while k5/k20/k20f are near-baseline. The
  validation round-5 drop after the big compression pointed the same
  way. The local judge foreshadowed the shape (worst k5) but wildly
  overgraded k20f (49.1 vs GPT 26.5) — another local/GPT divergence.
- **Mechanics all held**: zero corpus escapes across 5 rounds, first-
  attempt generation every round, edit-ops with no churn, foundations
  sections created and grown, single uninterrupted job.
- **The studier remains the bottleneck**: 0 correct in 30 attempts even
  at `medium` difficulty (57 total across both iterations). The
  quiz→lesson→note channel cannot compound while the studier solves
  nothing; the note fills with corrections for errors the studier keeps
  making anyway at answer time.
- Study compute: selfquiz2 spent ~303k studier-generated tokens (49k
  exploration + 231k attempts + 23k distillation) vs the cheatsheet
  condition's ~50k, for less measured expertise — iteration 2 is also a
  negative result on studying *intelligence* (expertise per study
  compute), stated descriptively.

## Next

1. Regrade `smalldspy-selfquiz2-20260721` with GPT-5.4 when quota is
   restored (and ideally all five runs in one batch, per the 006 audit's
   judge-drift caution).
2. If the GPT ordering matches: iteration 3 should attack the studier's
   frontier directly — drill-level questions (single mechanism, few
   lines), or note ablations isolating whether quiz-added content helps
   or harms cheap-budget answering.
