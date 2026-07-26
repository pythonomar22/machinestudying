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

- `runs/dspy-teachermini-20260722` — **teacher trajectory collection,
  complete (2026-07-23), ungraded by instruction.** codex gpt-5.4-mini
  (ChatGPT-account auth, effort medium, read-only at the pinned fulldspy
  checkout) answered all 70 study-slice questions at forced exploration
  budgets k=5f / k=20f / k=50f — 210 sessions, 10 workers, zero
  failures, ~2.6 h wall. Every session archives the full codex event
  stream (ground-truth commands), the model's self-reported per-step
  tool_log (command, motivation, discovery), the answer, and a sandbox
  execution of its program.
  - Budget adherence (from event streams, not self-report): median
    commands exactly 5 / 20 / 51; exact-k compliance 54/41/23 of 70
    (mini overshoots by a few commands at higher k — recorded, not
    enforced).
  - Answer contract: 209/210 single-fence answers.
  - **Sandbox pass rises monotonically with search budget: 57 → 58 → 62
    of 70 (81% → 83% → 89%)** — first structural evidence in this line
    that more exploration buys more working programs, before any
    rubric grading.
  - Escape scan: 3 of 210 sessions touched out-of-corpus paths
    (logged per record for the analysis; diagnostics are not rejected).
  - Lenient grading deliberately NOT run at collection time; graded on
    Omar's go the same night (below).

## Teacher grading (paper-contract GPT-5.4, one era, 2026-07-23)

First grading against our own rev-3 rubrics at scale (previously only
eval runs on the StudyBench test questions were graded). 210/210
verdicts; artifacts under `runs/dspy-teachermini-20260722/dspy/grades/`.

| budget | mean lenient | mean gen tokens (codex output incl. reasoning) |
|---|---:|---:|
| direct (closed-book, added 2026-07-23) | 22.00 | 11.3k |
| k5f | 70.50 | 9.9k |
| k20f | 77.14 | 15.1k |
| k50f | 78.21 | 31.9k |

The direct pass (70 sessions, 63/70 executed exactly zero commands; the
rest 1–2) exposes the search gap directly: closed-book, only 6/70
programs run in the sandbox (vs 57–62/70 with search) and lenient falls
to 22.0 — while COSTING more tokens than k5f (11.3k vs 9.9k; the model
reasons longer when it cannot look, and it does not help). The direct
point is therefore strictly dominated by k5f and the measured-point WAUC
is unchanged at 22.83; the axis head below 9.9k stays floored (~70% of
the weight). Figure: `runs/dspy-teachermini-20260722/dspy/grades/
expertise_curves.png`. **The single most important number for the
fold-back thesis: five exploration commands buy +48.5 lenient points
(22.0 → 70.5).**

WAUC over the measured points (same Appendix-C integral,
cross-validated against studybench.weighted_auc on the 008 data):
**22.83**, with ~70% of the axis weight floored to zero below the
cheapest useful point. Not comparable to test-set expertise numbers
(different model, harness, and question set — and these rubrics are our
own replication, gold-authored by the same model family as the judge).

Per-question deltas: k5→k20 28 up / 15 down / 27 tie; k20→k50 15 up /
17 down / 38 tie. 21/70 perfect scores at k50f; no question is
all-zero at every budget. Biggest k5→k50 gainers (the trajectories to
read first): dspy_0fb766e45959 (10→100), dspy_387f99a1348d (0→75),
dspy_076b51a85fa2 (40→100), dspy_c99213024029 (25→85),
dspy_0767302a1fa2 (5→55).

Reading: the strong searcher is already at 70.5 with FIVE commands —
most of its advantage is prior knowledge plus efficient targeting, not
search volume; returns then diminish steeply (+6.6 for 5→20, +1.1 for
20→50) and k20→k50 is net noise (15 up, 17 down). The distillation
signal is concentrated in the k5→k20 flips and the five big gainers,
not in bulk k50f volume.

## Fold-back v1: object built and examined (2026-07-24)

Object (`runs/dspy-foldbackv1/dspy/cheatsheet.md`, 66KB): built by
`studying/foldback/teachermine.py` from the 70 practice k5f trajectories
+ direct attempts + practice golds (test set untouched). 70 mining
sessions (497 anchored facts, 241 prior-diffs, 440 excerpt nominations)
→ one organization pass → GPT-5.4 corpus fact-check (7 entries dropped)
→ deterministic render: answer contract, 40 prior-correction cards, 4
fact sections, 25 verbatim excerpts (pulled by code, not model memory).
Dev sanity (10 held-out dev questions, closed-book): bare 15.5 → object
26.5, tokens slightly down.

Exam (`dspy-codexmini-foldback-20260724`, 30 test questions × 4 budgets
× 1 rollout, paper judge):

| budget | lenient | tokens | vs no-study (3-rollout) |
|---|---:|---:|---|
| direct | **39.80** | 13.0k | 25.82 @ 10.8k (**+14.0 accuracy**) |
| k5 | 65.53 | 17.8k | 64.91 @ 13.5k |
| k20 | 67.83 | 22.2k | 67.10 @ 20.0k |
| k20f | 62.93 | 22.9k | 71.06 @ 20.3k (−8.1; single rollout) |
| **E** | **13.81** | | **16.78 no-study / 18.28 cheatsheet** |

**Split verdict.** The knowledge folded: +14 closed-book accuracy on
held-out test questions is the largest direct-budget transfer measured
in this project. But expertise FELL, for a reason the token
decomposition makes exact: answers stayed the same length (message
~1,050 tokens at direct, unchanged) while REASONING grew ~+3.3k at
every budget (direct 8.7k → 12.0k) — the model deliberates over the
66KB note instead of trusting it, shifting every point right on the
log axis; that weight loss plus the k20f dip outweighed the accuracy
gain. The 012 prediction (knowledge substitutes for rumination)
inverted at this note size for this model.

Iteration-2 levers, in priority order: (1) trust enforcement — answer
contract rewritten to "do not re-derive or cross-check the notes;
answer immediately from them" and/or reasoning_effort=low at direct
(the note exists precisely so thinking is unnecessary); (2) slim the
object (cards + contract + top excerpts; ablate the 66KB); (3) fix the
k20f interference (the note's "notes win" clashes with forced-search
instructions); (4) component ablations to attribute the +14.

## Qwen + fold-back object on the test set (2026-07-24, Omar-directed)

Run `dspy-qwenfoldback-20260724`: Qwen3.5-9B, paper harness, `foldback`
condition (the 66KB codex-mined object prepended; no study phase), full
protocol, seeds paired with the 008 anchors. 360/360 valid after two
sweep passes (18 transients on pass 1 — elevated AdapterParse/timeout
rate under the 17k-token note prefix, all cleared). Note delivery
hash-verified. Grading: paper judge, one era (grade.py extended to
accept the foldback condition, validated like selfquiz).

| budget | lenient | tokens | no-study | cheatsheet |
|---|---:|---:|---|---|
| direct | 9.19 | 3.6k | 4.48 @ 3.0k | 10.40 @ 2.8k |
| k5 | 13.11 | 5.4k | 8.17 @ 5.7k | 14.30 @ 5.4k |
| k20 | 14.39 | 6.0k | 9.82 @ 7.7k | 19.46 @ 7.5k |
| k20f | 20.42 | 31.3k | 23.97 @ 20.5k | 29.50 @ 24.1k |
| **E** | **11.02** | | 9.04 | 15.90 |

Paired cluster bootstraps (30 questions, 20k resamples):
foldback − no-study = **+2.08, CI [−1.27, +6.03]** (not separable);
foldback − cheatsheet = **−4.74, CI [−8.97, −0.04]** (significantly
worse than Qwen's own 4.6KB self-written cheatsheet).

Reading: the teacher-mined knowledge transfers to Qwen only weakly —
cheap budgets all rise ~+5 accuracy vs no-study (a real, consistent
pattern) at healthy token costs (direct 3.6k), but k20f REGRESSES
(24.0 → 20.4) with generation exploding to 31.3k tokens: the 66KB
prefix makes Qwen's forced searches longer and worse, echoing the codex
reasoning-inflation effect in a different channel. Meanwhile the same
object gave its own author (codex) +14 at direct. The object is
student-mismatched: a 9B cannot digest 66KB of another model's notes
as well as a small self-written note it authored. Iteration-2 lever
for the Qwen arm: radically slim, student-matched rendering (the
cheatsheet's 4.6KB is the existence proof of the right size), and/or
have Qwen rewrite the teacher content in its own register.
