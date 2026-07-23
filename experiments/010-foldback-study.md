# 010 — Fold-back study: distill high-budget competence into a k=0 object

**Status: designed 2026-07-22, with Omar (interview in session). Code in
`studying/foldback/`.**

## First-principles motivation

Expertise is WAUC on a log-token axis anchored at 3k generated tokens,
with weight halving per compute doubling (paper Appendix C). In our 008
cheatsheet run the weight decomposition is: direct 0.447, k5 0.151, k20
0.278, k20f 0.124 — **the direct point carries ~45% of all weight**, and
its answers already sit at the anchor (2.8k tokens), so the only channel
left there is accuracy. Meanwhile forced-k20 reaches 24.0 (no-study) /
29.5 (cheatsheet) accuracy — competence that exists in the model+corpus
system but is bought with tokens the metric discounts ~4×.

**Hypothesis:** the direct↔k20f accuracy gap is largely *location and
verification* knowledge that search buys at answer time. Mining
high-budget trajectories on practice questions and folding the
discoveries into a prepended study object recovers part of that gap at
k=0 (and shortens/sharpens k5/k20 search), raising expertise.

Counterfactual ceilings (008 cheatsheet arm, token axis fixed):
direct = k20's 19.5 → E 20.7; direct = k20f's 29.5 → E 29.5. Secondary
channel (k5/k20 answers shortened to ~4k/5.5k) → +1.5 E alone. Anchor
pair to beat: no-study 9.04, cheatsheet **15.90**.

## Design (decisions from the interview, all four = recommended options)

1. **Study object: layered.** A content playbook (verified idioms, API
   facts, behaviors, pitfalls — feeds the direct head) plus a
   navigational map (mechanism → file/symbol + one-line note — feeds
   k5/k20). Prompt-side text, so free on the metric's completion-token
   axis (008 audit confirmed the convention).
2. **Authorship: Qwen mines and writes; GPT-5.4 fact-checks only.** The
   studier (same author as the baseline cheatsheet) produces every
   entry; GPT-5.4 (codex, read-only at the corpus) marks entries
   verified/false/unverifiable and non-verified entries are DROPPED
   deterministically — the teacher never writes content (bounded,
   ledgered assist).
3. **Study signal: trajectories + sandbox + golds.** Attempts run at
   forced k=20, bare (no note) — the exact budget whose competence we
   fold back; each attempt's program and each gold program runs in the
   pinned sandbox; mining sees the trajectory, both sandbox outcomes,
   and the verified gold program (ledgered teacher signal, as in 006).
4. **Evaluation discipline: 70/30 split, one test shot.** The 100-question
   rev-3 set (`data/dspy_validation.jsonl`, promoted from
   `data_collection/artifacts2/`) splits stratified-by-topic into 70
   study / 30 dev. Object variants iterate offline on dev-30
   (direct+k5, paper-contract GPT judge) against two dev baselines
   (bare, 008 cheatsheet note). Exactly ONE fulldspy test evaluation
   (4 budgets × 3 rollouts, seeds paired with 008) for the single
   declared-best object. Declared before any test contact.

Deliberate calls, ledgered:
- k=50f is a declared escalation only if k20f mining looks
  signal-starved; v1 stays at the measured target budget.
- v1 object is built purely from trajectory mining (no
  exploration-note seeding) for clean attribution; layering exploration
  content in is a declared v2 variant.
- **No iterative rewrite loop.** Assembly is a single-pass map-reduce
  (per-question lesson records → per-topic reduce → deterministic
  merge). The 007 destruction dynamic (repeated self-rewrites of one
  artifact at temp 1.0) is structurally impossible.
- Mining/assembly calls use temperature 0.2 (study-time reliability;
  recorded in the manifest). Evaluation sampling is untouched.

## Pipeline (studying/foldback/, one idempotent phase per module)

split → gold sandbox runs → attempts (70 × forced k20, bare, full turn
logs) → attempt sandbox → mine (Qwen per question: items[kind, claim,
code?, files] + map entries) → assemble (Qwen per topic: dedup/rank to
≤15 items + ≤20 map entries; one answering-protocol reduce; deterministic
render) → factcheck (GPT-5.4 codex verdict per entry id; drops applied
deterministically, recorded) → object.md/object.json + study.json →
devval (variant × dev-30, direct+k5 × 2 rollouts, paper judge).

## Success criteria (declared)

- Offline gate: object beats the 008 cheatsheet note on dev-30 direct
  mean lenient (primary) without losing k5 (secondary).
- Test claim: E > 15.90 on the one declared test evaluation, paired
  bootstrap CI reported; adversarial audit workflow before any claim.

## Runs

Pending.
