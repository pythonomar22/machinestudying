# 015 — Automated fold-back distillation: the studier designs its own object

**Status: complete (2026-07-26). Study object built and verified;
next declared step is dev-30 evaluation, then the one test shot.**

## Objective

Replace 010's hand-shaped, trajectory-only fold-back with a fully
automated analyze→build pair in which the studier (gpt-5.4-mini)
authors everything AND chooses the object's form itself — note, lookup
tool, or hybrid — under the true metric mechanics, with no cheatsheet
prior baked in (Omar's directive). Inputs beyond 014's trajectories:
both grade sets at claim level (the flip structure), judge rationales,
the verified practice golds, rubric evidence paths, and the k20f
trajectory turns that touched each claim's evidence.

## The claim-level flip structure that shaped the design (014 data)

Of 7,000 rubric weight on the 70-question study slice: **FLIP+ 46.6%**
(earned only with tools — trajectory-minable), **BOTH0 37.4%** (missed
even at k20f — recoverable ONLY from the golds; a trajectory-only miner
caps at k20f's 59.5), BOTH1 12.9% (already known at direct — excluded
from teaching), FLIP− 3.1% (noise). And the metric math on this model's
measured tokens: k20f (3.5k tokens) carries **0.853** of the two-point
area vs direct's 0.147 — so raising tool-budget accuracy matters more
than fixing direct alone, and note-delivered knowledge raises both.
(The Sonnet-era "direct dominates" intuition is backwards for mini;
caught by computing, not assuming.)

## Pipeline (commit ab0a867, adversarially design-reviewed pre-spend)

1. `studying/foldback/analyze.py` — deterministic claim-flip layer +
   studier-as-analyst extraction per recoverable claim (lesson, code,
   source ∈ {trajectory, gold, both, prior}, grounding paths,
   generality), misconception fixes mined from the judge's rationales
   on the direct misses, map entries; one targeted follow-up call per
   question with uncovered recoverable claims. Config-stamped records,
   partial runs write `summary_partial.json` only (no cache poisoning).
2. `studying/foldback/build.py` — the studier drafts 3 independent
   object designs under a mechanics block computed from its own
   artifacts (best-so-far-area formula, per-budget marginal weights,
   scenario table, finish semantics, per-budget token headroom), then
   critiques and commits; per-topic assembly with claim statements +
   BOTH1 "don't re-teach" digest in view; drop-only verification
   (deterministic path pruning, then GPT-5.4 verdict-only fact-check
   against the union of cited sources); form-faithful render preserving
   the studier's section/entry order.
3. `studybench/react.py` — `study_lookup(key)` tool serving
   `study/build/lookup_store.json` under `--condition foldback`, so the
   tool/hybrid forms are real, not theater. (Eval manifests record the
   extra tool; foldback-vs-baseline comparisons must note the tooling
   delta.)

Pre-spend review: workflow wf_87816d7e-5c0, 5 lenses + adversarial
verify (39 agents) — 34/34 serious findings verified real and fixed,
including a backwards metric claim in the mechanics, a form choice the
code couldn't honor, trajectory starvation (400-char observations),
misconception fixes bypassing fact-check, and smoke-cache poisoning.

## Results

Analysis (70/70 questions, ~$2): **291 lessons — 283 reusable, 248
gold-sourced — covering 100% of recoverable claim weight** (FLIP+
3265/3265, BOTH0 2620/2620, uncovered 0), + misconception fixes and
map entries.

Build: the studier committed to **hybrid** with a numerically grounded
rationale (gold-derived BOTH0 content in the note because it lifts all
budgets; FLIP+ retrieval depth in a `study_lookup` store for tool
budgets; pure-note rejected because 46.6% benefits from tool-time
retrieval, pure-store rejected because BOTH0 must reach direct). Eight
sections led by a budget-aware answer-protocol capsule (classify →
answer tight at direct → index-first lookups + early stop at tool
budgets → claim-dense bullets, 3k-anchor awareness).

Verification: 148 assembled entries → **110 kept** (32 unverified, 1
contradicted, 5 uncited dropped; ~19% of served claim weight — the
heaviest drops are listed in the build log; drop-only, never
rewritten). Final object: 95-entry, 44,626-char note + 15-key lookup
store; only 1 question-specific entry kept. `cheatsheet.md` +
`study.json` sit at the run root in the exact contract
`studybench.react --condition foldback` consumes.

## Honest notes for the audit trail

- Studier temperature is 1.0 (gpt-5.x pins it; 010's convention was
  0.2) — ledgered in summary and study.json.
- The verifier's 32 "unverified" drops may include false negatives
  (strict sourcing); the drop-only rule keeps authorship clean at the
  cost of some recall. Dropped ids + weights are in the build ledger.
- kept-entry weight_hint sums (6,160) exceed recoverable weight (5,885)
  because entries legitimately serve overlapping claims across
  questions; coverage claims should cite the analysis layer (100% at
  lesson level), not weight_hint sums.
- Nothing here touched the dev-30 slice or the 30-question test set.

## Test shot (2026-07-27, Omar's call: test first, baselines after)

`studybench.react --condition foldback --model gptmini`, 30 test
questions × 4 budgets × 1 rollout, seed 20260715, 120/120 ok first
pass, GPT-5.4 judge:

| budget | mean lenient | mean gen tokens | behavior |
|---|---:|---:|---|
| direct | 36.97 | 857 | note only |
| k5 | 33.13 | 1,406 | median 2 iters (voluntary stop), lookup in 15/30 |
| k20 | 42.33 | 1,481 | median 2 iters, lookup in 15/30 |
| k20f | 43.67 | 3,825 | forced 20, lookup 2.6×/ep in 30/30 |

**Expertise = 43.38** — the highest E in the project (same judge and
questions: Sonnet no-study 32.94 / +cheatsheet 34.36, codex-mini
16.78/18.28, Qwen 9.04/15.90). Three of four budget points sit inside
the 3k anchor (weights 1/1/1/0.78): the answer-protocol's early
stopping collapsed k5/k20 to ~1.4k gen tokens, exactly the lever the
mechanics block priced.

**Attribution is NOT yet claimable.** vs codex-mini's 16.78 this
conflates the harness change with studying; the same-harness no-study
and cheatsheet baselines (next step) are required before attributing
any of the 43.38 to the study object. Also note the foldback arm's
tool set includes `study_lookup` at tool budgets (manifested), 1
rollout, n=30. Full adversarial audit deferred to the baseline
comparison, where the actual claims get made.

## Next

gptmini `baseline` and `cheatsheet` test runs in the dspy harness
(1 rollout, gpt judge), then the attribution audit.
