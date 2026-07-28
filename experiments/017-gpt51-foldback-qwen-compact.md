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

## Results

(pending)
