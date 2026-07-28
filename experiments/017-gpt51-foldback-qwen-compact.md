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
  **Deviation (2026-07-28, decided after the dev gate, before any test
  contact): 1 rollout, not 3 — Omar's call on cost (the 96.6KB note
  rides every prompt; 3 rollouts projected ~$80–130). Silver lining:
  all three 5.1 arms are now rollout-matched (1 each), so every
  within-model contrast passes report.py check_pair; the CI will be
  wider than the konly precedent.**
- Endpoints (pre-registered in 018 before any test grade existed):
  primary E(3000); secondary closed-book direct accuracy (the
  transfer construct, immune to the 018 censoring band); anchor sweep
  reported alongside.
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

## A. GPT-5.1 results (audited wf_6f99d2fe-59e — all 6 checks pass;
## wording below is the audit's)

Pipeline (all 2026-07-27/28): stage-0 direct 7.14 vs k20f 35.21 on the
70-question practice slice → analyze 335 lessons / **317 misconception
fixes** / 100% recoverable-weight coverage → build: self-designed
96.6KB prompt note, 70/83 entries verified (12 unverified + 1
contradicted dropped) → dev gate: object direct 26.93 vs bare 7.38
(+19.55 [+11.15,+28.47]), vs cheatsheet +9.00; tokens DOWN with the
note → test (1 rollout, deviation ledgered above).

| gpt-5.1 arm (gpt-5-4 judge, 30q, 1 rollout) | direct | k5 | k20 | k20f | E |
|---|---:|---:|---:|---:|---:|
| no-study | 11.43 @ 1.9k | 15.10 | 13.47 | 25.80 @ 8.4k | 18.31 |
| own cheatsheet (50-iter study) | 24.37 @ 1.5k | 31.13 | 28.03 | 35.63 @ 10.8k | 31.47 |
| **fold-back** | **38.70 @ 1.5k** | 30.43 | **38.10** | **42.70** @ 16.2k | **39.44** |

Audited claims (paired bootstrap, seed 20260715, fresh stream per
endpoint):

1. **Fold-back beats no-study decisively on both endpoints**: ΔE
   +21.14, CI [+10.56, +34.42]; closed-book direct +27.27, CI
   [+16.40, +38.73] (BCa/t/Wilcoxon/sign all agree). Unhedged.
2. **Cheatsheet also beats no-study**: ΔE +13.17, CI [+5.75, +22.21]
   — the first non-null cheatsheet effect in the project (null on
   Sonnet/mini/codex; qwen's was positive) and CI-clean at every
   anchor (018). On a stale-priors model, even single-pass studying
   corrects knowledge.
3. **Fold-back vs cheatsheet on E: numerically ahead, NOT separable**
   — +7.97, CI [−4.27, +21.29], P=0.904 (null stable under 4 seeds ×
   2 rep counts × percentile/BCa). Point estimate only.
4. **Fold-back vs cheatsheet on the pre-registered secondary endpoint
   (closed-book direct): +14.33, marginal at p≈0.05** — CI
   [+0.03, +28.17], but the lower bound is a Monte-Carlo knife-edge
   (other seeds: −0.10…+0.53; t p=0.059, Wilcoxon p=0.040, sign
   p=0.115). Never "clean exclusion of zero."
5. Not token gaming: direct arms token-matched (1482/1496/1859, all
   weight 1.0); fold-back's k20f inflation (16.2k) PENALIZES it —
   swapping in the cheatsheet's token profile would RAISE E to 39.81.
   8/8 audited direct wins are knowledge (API facts present/absent),
   not form; reverse control exists (fold-back loses 10-vs-100 on a
   retriever question).
6. No test leakage (0 shared 12-word shingles vs questions and golds;
   positive control fires on study golds; residual overlap ≤ the
   cheatsheet arm's own baseline).

Required framing (audit): the studier is handed the exam mechanics
(build.py mechanics block) and its note is mined from judge-graded
practice on same-generator questions — the method is **"studying
graded past exams," not pure repository distillation**, and the
cheatsheet control received no mechanics disclosure and less study
compute (33KB from one 24.5k-token pass). Not study-compute-matched.
One model/corpus/judge, 30q × 1 rollout — no generality claims.
Commits differ per arm (37e0f3f/416b85e/2af128e); harness code
verified byte-identical across them.

Publication queue from the audit: (i) protocol-stripped-note ablation
(separate knowledge from metric-aware protocol at direct); (ii) +2-4
rollouts on the direct head to settle the marginal cheatsheet claim;
(iii) sonnet-judge robustness on the direct head; (iv) mechanics
disclosure or compute parity for a fairer cheatsheet control; (v) 018
anchor sweep + closed-book endpoint reporting.

## Results

(qwen compact pending)
