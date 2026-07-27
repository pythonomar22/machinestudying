# 016 — Delivery ablation (mini), GPT-5.1 gate, Qwen fold-back v2

**Status: all three workstreams complete (2026-07-27); Qwen v2 test
graded — beats no-study and v1 separably, ties the cheatsheet.**

## A. Mini delivery ablation — behavior was the whole regression

Variants rendered from the 015 build's verified entries, drop-only
(`studying/foldback/variants.py`): `knowledge_only` (110 entries, 46.0k
chars — no protocol capsule, no store) and `both0_only` (33 entries,
15.7k chars). Dev-30 gate (gptmini, direct+k5, 2 rollouts, gpt judge):

| arm | direct | k5 |
|---|---:|---:|
| bare | 11.30 @ 920 | 45.60 @ 1591 |
| knowledge_only | **24.92** @ 518 | **47.58** @ 1300 |
| both0_only | 14.80 @ 531 | 45.05 @ 1371 |

The protocol/store, not the knowledge, caused 015's tool-budget
collapse; the full note beats the compact one (dilution not binding).

Test shot (`dspy-gptminikonly-20260727`, **3 rollouts**, 360/360 ok,
gpt judge): direct 30.79@863, k5 47.81@1716, k20 45.87@1734, k20f
56.61@3886 → **E = 54.60**. Paired vs no-study (1 rollout): direct
**+12.86 (21W/5L/4T)** — knowledge transfer confirmed at test scale;
k20f +0.24 (regression eliminated); k5 −2.3 / k20 −7.6 (null-to-lean);
ΔE = −1.08, CI [−8.74, +4.38] — **statistical parity**.

**Mini conclusion (now fully decomposed):** fold-back object with
behavior E 43.4 < knowledge-only E 54.6 ≈ no-study 55.7. The distilled
knowledge is real (+13 direct at 3 rollouts) but mini's expertise is
saturated by its native tool behavior — every budget that carries
weight is already ≥ the note-augmented direct. No prompt-side object
tested can beat this model's no-study E; its studying headroom lives
at tool budgets, which notes do not reach.

## B. GPT-5.1 gate — the fold-back-shaped GPT model

`dspy-gpt51-nostudy-20260727` (30q × 4 budgets × 1 rollout, gpt judge):

| budget | lenient | gen tokens |
|---|---:|---:|
| direct | 11.43 | 1,859 |
| k5 | 15.10 | 3,613 |
| k20 | 13.47 | 3,716 |
| k20f | 25.80 | 8,387 |

**E = 18.31** — vs mini's 55.68 on identical harness/judge/questions:
the blog's "equally capable, different expertise" gap reproduced on the
WAUC axis. Profile is exactly fold-back-shaped: direct crippled by
stale priors (the blog's deprecated-API failure mode), voluntary tool
budgets barely help (stops early on wrong answers), best-so-far ≤k20
(15.1) spans ~0.64 of the weight axis. A mini-sized direct transfer
(+13…19) projects E ≈ 31–35 — a near-doubling. **GPT-5.1 fold-back is
the next headline experiment.**

## C. Qwen fold-back v2 — dev gate passed; test shot in flight

Infra (26a4565 + follow-ups): full pipeline model-parameterized (qwen
studier via session vLLM; temp-0.2 study convention restored with
thinking_token_budget 6000 after two rumination `finish=length`
failures; studier persona parameterized after catching Qwen being
addressed as "gpt-5.4-mini" — analysis regenerated clean; gptmini
config hashes byte-preserved throughout). vLLM bring-up needed the
sbatch PATH env (ninja/nvcc).

Stage-0 (`dspy-qwenfoldback2-20260727`, 70q, gpt judge): direct
**2.21 @ 2,596** vs k20f **30.64 @ 18,612** — 139/140 ok. Bucket
structure: BOTH0 = 4,790/7,000 (68%!) vs FLIP+ 2,055 — Qwen's forced
search finds far less than mini's, so the golds carry the distillation.

Analysis: 341 lessons (340 reusable, **289 gold-sourced**), 100%
recoverable coverage. Build: Qwen committed to **prompt_note** (vs
mini's hybrid — sensible given its expensive k20f); 214 assembled →
**153 kept** (52 unverified + 8 contradicted + 1 uncited dropped — the
external drop-only verifier caught substantially more bad content than
it did for mini); 45.8k-char note.

Dev-30 gate (bare vs object, direct+k5, 2 rollouts, gpt judge):

| arm | direct | k5 |
|---|---:|---:|
| bare | 0.50 @ 3122 | 6.80 @ 6030 |
| object_v2 | **6.97 @ 2527** | **11.08 @ 5647** |

Both budgets up AND both token counts down — unlike mini, Qwen gains
at tool budgets (the note carries what its search can't find). Gate
passed → declared test shot launched: 30q × 4 budgets × **3 rollouts**
(matching the 008 anchors), gpt judge, targets 9.04 (no-study) and
15.90 (cheatsheet). One dev-only infra note: the GPT-5.4 judge
abstained (`needs_regrade`) on one dev answer; devval now retries once
and, only for dev signal, accepts flagged claim verdicts.

### Test result (360/360 after the standing 8-episode resample; retry
### executed from the recorded source_commit 3f59481)

| Qwen arm (same seeds/rollouts/judge/questions) | direct | k5 | k20 | k20f | E |
|---|---:|---:|---:|---:|---:|
| no-study (008) | 4.5 @ 3.0k | 8.2 @ 5.7k | 9.8 @ 7.7k | 24.0 @ 20.5k | 9.04 |
| + cheatsheet (008) | 10.4 @ 2.8k | 14.3 @ 5.4k | 19.5 @ 7.5k | 29.5 @ 24.1k | 15.90 |
| + fold-back v1 (010) | 9.2 | 13.1 | 14.4 | 20.4 | 11.02 |
| **+ fold-back v2** | 10.96 @ 3.0k | 15.82 @ 6.6k | **21.98** @ 7.5k | 26.02 @ 25.4k | **16.02** |

Paired (10k-rep bootstrap, seed 20260715):

- **v2 > no-study: ΔE +6.98, CI [+3.08, +10.79]** — separable.
- **v2 > fold-back v1: ΔE +5.01, CI [+1.86, +7.08]** — the automated
  pipeline (gold-sourcing, claim targeting, verification,
  self-designed object) is a real improvement over 010.
- **v2 ≈ cheatsheet: ΔE +0.13, CI [−3.84, +3.80]** — a statistical
  tie, claimed as a tie. v2 leads at direct/k5/k20; the cheatsheet
  leads at k20f (−3.5).

Adversarial audit (wf_bcd12668-c55; all 1,440 grades recomputed exact,
provenance 0 mismatches, splits and note leakage-clean with zero test
shingles, resample lean n.s. — zeroing all 8 retried episodes still
leaves E=15.24 > no-study). **Audited wording:**

1. Claimable as stated: v2 beats no-study (+6.98, CI [+3.08, +10.79];
   survives ×3 correction, perm p=0.0006).
2. Reframed: v1 was the *teacher-mined* object (gpt-5.4-mini mining
   codex-teacher trajectories, 66KB, student-mismatched) — so +5.01
   (CI [+1.86, +7.08]) is an end-to-end pipeline comparison
   (self-studied v2 vs teacher-taught v1), not a single-factor one.
3. Softened: the cheatsheet comparison is a *failure to detect a
   difference* (no pre-registered equivalence margin; CI admits ±3.8),
   not demonstrated equivalence. The founding "beat 15.90" target is
   matched, not beaten.
4. Mechanism note: the k20f deficit (−3.5, itself n.s.) tracks a
   **note-length tax on forced search**, monotone across arms
   (cheatsheet 13.5KB→29.5, v2 45.9KB→26.0, v1 66KB→20.4; no-note
   24.0); protocol-suppression is NOT supported (finish attempts
   correlate positively with score within arms). Qwen's self-written
   protocol is metric-aware (targets ~2.8k tokens at direct — the 3k
   anchor) — disclosed as part of the method; realized direct tokens
   matched the anchors (3.0k), so no distortion materialized.
5. Note quality: 8/8 sampled entries API-real; 6 fully correct, 2
   minor imprecisions; 4 duplicate bullets (cosmetic).

Next lever suggested by (4): a size-disciplined v2 render (compress
toward ~15-20KB, drop-only) could reclaim the k20f tax — the one
experiment between "tie" and "beat."
