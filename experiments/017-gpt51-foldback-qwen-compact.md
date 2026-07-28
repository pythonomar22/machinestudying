# 017 — GPT-5.1 fold-back (headline) + Qwen v2-compact (length-tax refund)

**Status: in flight (2026-07-27). This header section is the pre-registered
plan, written before any result below it existed.**

## Why these two (from 016)

- **GPT-5.1** is the fold-back-shaped GPT model: no-study E=18.31 with
  direct 11.43 (stale 2024 priors — the blog's deprecated-API failure
  mode), best-through-k20 only 15.10. Direct owns c≈0.17 of the weight
  axis outright and best-so-far carry extends a direct fix across
  ~0.64. The pipeline has twice moved +13…19 into direct (mini +12.9…
  +19.0 at test scale; qwen +6.5 dev); a mini-sized transfer projects
  E ≈ 31–35, a near-doubling, and would land the blog's "can studying
  correct stale knowledge" claim within-model.
- **Qwen v2-compact**: v2 tied the cheatsheet (16.02 vs 15.90; CI
  ±3.8) with a k20f deficit of −3.5 that the 016 audit traced to a
  monotone **note-length tax on forced search** (13.5KB→29.5,
  45.9KB→26.0, 66KB→20.4). The fix the audit suggests: same verified
  entries, drop-only rank re-render to ~18KB. If the tax refunds even
  half of −3.5, v2-compact beats the cheatsheet and closes the
  founding target.

## Pre-registered design

### A. GPT-5.1 fold-back — `dspy-gpt51foldback-20260727`

Pipeline identical to 016's qwen v2 (stage0 → analyze → build →
dev-30 gate → one declared test), with gpt-5.1 as its own studier
(profile added to stage0/analyze/devrun; persona parameterized;
temperature pinned 1.0 by the gpt-5.x surface, ledgered). Judge
gpt-5.4 (paper contract) throughout. Seed 20260715.

- Stage-0: 70-question study slice, direct + k20f, bare.
- Dev-30 gate (bare vs object, direct+k5, 2 rollouts): **pass =
  direct up by ≥ 5 points and k5 not down by more than its noise**
  (same both-budgets-healthy bar 016 used for qwen).
- Test: 3 rollouts × 30q × 4 budgets, condition `foldback`.
- Controls, both 1 rollout (the tier every hosted-model control has
  used since 013): existing `dspy-gpt51-nostudy-20260727` (E=18.31)
  and a new **own-cheatsheet arm** `dspy-gpt51-cheatsheet-20260727`
  (50-iteration forced study, same contract as mini/sonnet/qwen
  cheatsheets). Claim target: fold-back beats no-study AND cheatsheet
  within-model (paired bootstrap, seed 20260715).

### B. Qwen v2-compact — `dspy-qwenfoldback2c-20260727`

- Render: `variants.py compact` — drop-only rank-prefix of the v2
  build's 153 verified entries (reusable first, claim-weight-served
  desc, ties by assembly order; presentation order and protocol kept
  verbatim; no new authorship, no new model calls). Result: 57
  entries, 17,695 chars (target 18k).
- Dev-sanity (not a gate in the 016 sense): compact arm on dev-30,
  direct+k5, 2 rollouts vs the existing v2 `object_v2` and `bare`
  reports. Pass = direct and k5 hold (do not collapse toward bare).
- **Honesty note, pre-registered: dev runs direct+k5 only, so the
  k20f recovery this test is designed to capture cannot be previewed
  on dev. The test shot is mechanism-motivated (016 audit's length
  tax), not dev-gated.**
- Test: 3 rollouts × 30q × 4 budgets, same seeds/judge/questions as
  the 008 anchors and v2. Anchors: no-study 9.04, cheatsheet 15.90,
  v2 16.02. Success = beats cheatsheet separably; k20f moving
  26.0 → ≥27.8 (half the tax) without direct/k5 giving back would
  already support the mechanism.

Both tracks: adversarial audit workflow before any claim is worded.

## Infra event — OpenAI quota outage (2026-07-27 ~22:33)

Mid-launch, the lab OpenAI key returned 429 `insufficient_quota`
(billing, not rate limiting) and every OpenAI-dependent call died:
gpt-5.1 episodes AND the gpt-5.4 paper judge. State at outage +
salvage:

- `dspy-gpt51foldback-20260727` stage-0: all 70 direct episodes ok
  (mean 1,358 gen tokens); k20f 11 ok, 59 quota-poisoned `gave_up`
  **deleted** (stage-0's `_episode_ok` treats `gave_up` as done — a
  rerun would have silently kept them). **Resume: rerun the same
  stage0 command**; it redoes the 59 attempts + all 140 grades.
- `dspy-gpt51-cheatsheet-20260727`: the 50-iteration forced study
  **completed before the outage** (cheatsheet.md + study.json valid);
  all 120 eval episodes were 0-token 429 stubs, deleted. **Resume:
  rerun the same react command**; study is reused.
- Qwen dev-sanity (`dev/qwen-compact`): episodes completing on vLLM;
  grading (gpt-5.4) blocked. **Resume: rerun the same devrun
  command** (idempotent per episode/grade).
- **Decision under the outage:** the compact test episodes were
  launched before dev-sanity could be graded — the GPU allocation
  (ends 07-28 06:34) is the only perishable resource and the compact
  object is a frozen deterministic render (nothing dev could have
  iterated). Evaluation order is preserved: dev-sanity will be graded
  and read BEFORE any test grade is looked at. Ledgered here rather
  than hidden.
- Resume order when quota returns: dev-sanity grades → read →
  compact-test grades; stage-0 rerun → analyze → build → dev-30 gate
  → gpt51 test; cheatsheet-arm rerun → grade.

## B. Dev-sanity result (read before any test grade existed)

| arm | direct | k5 |
|---|---:|---:|
| bare | 0.50 @ 3122 | 6.80 @ 6030 |
| object_v2 | 6.97 @ 2527 | 11.08 @ 5647 |
| compact | 2.92 @ 2462 | **11.08** @ 5528 |

k5 holds exactly. Direct does NOT hold: paired compact−v2 = **−4.05,
CI [−8.30, −0.63]** (10 questions down / 2 up / 18 tied) — the 96
dropped entries were earning direct claims on dev. **The
pre-registered sanity bar (direct holds) FAILED.**

Weight-math recalculation (recorded before test grades): at v2's test
token profile the k20f point carries only w≈0.12, so even a FULL
+3.5 k20f tax refund adds only ~+0.4 E. Scenario E (v2 test points,
audit tax slope −0.17/KB → compact k20f ≈ +2.8):

- direct holds, k20f +2.8 → E ≈ 16.3 (vs v2 16.02 — inside noise)
- direct −4 (dev-sized), k20f +2.8 → E ≈ 14.1 (worse than v2)

**The founding-target framing ("refund half the tax → beat 15.90")
does not survive the weights — k20f moves E through a 0.12 weight.
The test shot is therefore reinterpreted, before grading, as the
causal mechanism probe**: same entries, same protocol, only note
size differs (45.9KB → 17.7KB), so the k20f (and k20) budget points
directly test the 016 length-tax hypothesis within-object. E is
expected ≤ v2; any "beats cheatsheet" claim is off the table for
this arm.

## Results

(pending)
