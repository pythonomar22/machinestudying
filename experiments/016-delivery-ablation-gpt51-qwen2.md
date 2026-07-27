# 016 — Delivery ablation (mini), GPT-5.1 gate, Qwen fold-back v2

**Status: mini ablation and GPT-5.1 gate complete (2026-07-27);
Qwen v2 stage-0 in flight.**

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

## C. Qwen fold-back v2 — in flight

Infra landed (26a4565): stage0/analyze/build fully model-parameterized
(qwen studier via session vLLM, temp-0.2 study convention restored;
gptmini config hashes byte-preserved). Two TP=2 vLLM servers on GPUs
0–3 (ninja/nvcc PATH fix after one failed bring-up). Stage-0
(`dspy-qwenfoldback2-20260727`: 70q direct+k20f, gpt judge) running;
analyze → build → dev-30 → test (target: beat cheatsheet 15.90) next.
