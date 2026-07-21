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

- `runs/smalldspy-selfquiz2-20260721` (pending; smoke first).

## Results

(pending)
