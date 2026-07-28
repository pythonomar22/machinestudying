# 018 — The anchor's censored band: what the 3k-token anchor can and cannot see

**Status: analysis of already-graded runs (no new episodes), 2026-07-27.
Prompted by the mini saturation result (016) during 017. Adversarially
audited (wf_ac2c41ff-8ce, three lenses, all verdicts sound-with-fixes);
every number below was recomputed from per-episode grade files by the
audit, and this revision incorporates its corrections — the original
draft's errors are listed at the bottom so they cannot silently
reappear.**

## The mechanics

Appendix-C expertise is a best-so-far area over the four budget points
with weight `w = min(T0/t, 1)`, `T0 = 3000` generated tokens. Above the
anchor this is 1/t decay; below it is FLAT. A point contributes E only
through the weight-width between it and the next-more-expensive point
(the most expensive point owns all remaining measure down to w=0 — a
tail convention in the same design family as the clip).

Measured no-study direct-segment widths (= dE per point of closed-book
accuracy), with realized direct-segment share of E in parentheses:

| model (no-study) | direct tokens | next point | width | share of E |
|---|---:|---:|---:|---:|
| Sonnet 4.5 | 1.6k | 9.7k (k5) | 0.690 | 44.9% |
| Qwen3.5-9B | 3.05k (just above anchor) | 5.7k | 0.463 | — |
| GPT-5.1 | 1.9k | 3.6k | 0.170 | — |
| gpt-5.4-mini | 1.0k | 1.9k | **0.000** | **0%** |

Mini's k5 width is ALSO 0.000 (direct 1.0k, k5 1.9k, k20 2.1k — all
under the anchor): mini's E depends only on its k20/k20f accuracies
plus censored maxima.

**The precise statement is censoring, not blindness.** Because E is
best-so-far, closed-book accuracy re-enters the area the moment it
exceeds the best tool-assisted score. For mini knowledge-only:
E(3000)=54.60 is identical for any direct accuracy in [0, 47.8] (the
k5 ceiling), and would move above it (audit-verified: direct 60 →
E 60.00). Mini's +12.86 direct gain (17.9 → 30.8) fell entirely inside
the censored band — priced at exactly zero — and ~17 more points would
too. So: **under the anchor, the metric credits sub-anchor knowledge
only where it beats search outright.**

Two framings, both true, and the paper needs both:

1. *Metric working as intended:* under the stated cost model
   (generated tokens only), mini's search at 1.9k tokens is nearly
   free, so "studying buys mini nothing it cannot get at negligible
   cost" is a legitimate positive finding about mini — transfer
   happened, economic value did not.
2. *Instrument power:* ΔE has ~zero power to detect knowledge transfer
   for sub-anchor models. The transfer construct therefore needs its
   own pre-committed endpoint — closed-book direct accuracy — where
   mini's +12.86 (CI [+8.6, +29.1] from 016/015) already exists.
   **Pre-registered here for the pending 017 gpt51 test: primary
   endpoint E(3000); secondary endpoint closed-book direct accuracy.**

## Anchor sweep (all gpt-5-4-graded arms; E(T0) from grades)

| arm | T0=250 | 500 | 1000 | 2000 | 3000 | 6000 |
|---|---:|---:|---:|---:|---:|---:|
| qwen no-study | 0.75 | 1.51 | 3.01 | 6.03 | 9.04 | 13.59 |
| qwen cheatsheet | 1.39 | 2.78 | 5.56 | 11.13 | 15.90 | 20.94 |
| qwen foldback v2 | 1.34 | 2.67 | 5.34 | 10.68 | 16.02 | 21.21 |
| mini no-study (1r) | 9.17 | 18.34 | 36.67 | 54.83 | 55.68 | 56.37 |
| mini cheatsheet (1r) | 10.14 | 20.29 | 37.73 | 49.52 | 50.21 | 50.67 |
| mini knowledge-only (3r) | 11.96 | 23.93 | 42.97 | 52.34 | 54.60 | 56.61 |
| gpt51 no-study | 2.11 | 4.22 | 8.44 | 16.02 | 18.31 | 22.76 |
| gpt51 cheatsheet | 4.66 | 9.33 | 18.65 | 29.10 | 31.47 | 33.63 |
| sonnet no-study | 4.24 | 8.49 | 16.98 | 29.11 | 32.94 | 44.46 |
| sonnet cheatsheet | 4.04 | 8.09 | 16.18 | 31.74 | 34.36 | 42.21 |

Reading discipline for this table (audit-imposed): for T0 below an
arm-pair's cheapest mean-token point, E(T0) is exactly linear in T0 —
the low columns are ONE statistic rescaled (and equal to the unclipped
1/t ranking); the informative range is T0 ≈ 1000–6000 where the clip
crosses observed token costs. All columns share all sampling noise —
sign-invariance across T0 is not replication. "At every anchor" ≈ 2–3
effective tests, not 6.

With paired per-question bootstrap CIs (10k reps, seed 20260715), the
claims that survive:

1. **gpt51 cheatsheet > no-study: CI-clean at EVERY anchor**
   (T0=3000: +13.17 [+5.75, +22.21]; worst-anchor P(>0)=0.997). The
   one sweep claim publishable exactly as written.
2. **qwen foldback v2 vs cheatsheet: indistinguishable within ±3.8 E
   at T0=3000** (sign flips across anchors) — an equivalence-bounded
   null, not a demonstrated tie.
3. **sonnet cheatsheet: null within [−4.26, +7.30]** at 3000, all
   anchors consistent.
4. **mini: point estimates favor konly over cheatsheet at every
   anchor (+1.8…+5.9), but the CI excludes zero only in the
   degenerate linear regime and knife-edge (T0=250: [+0.04, +3.20]);
   at the primary anchor it is a null (+4.40 [−4.46, +9.79]).** The
   CI-clean strict-anchor result is **konly > no-study at T0=500:
   +5.59 [+2.04, +9.05], P=0.999** (null at 3000: [−8.74, +4.38]) —
   and this is the pre-existing 016 contrast, so no forking-paths
   objection. Three-way ordering language is not claimable.
5. ΔE(3000) = −1.08 for mini konly decomposes as: +12.86 direct priced
   at zero (censored band) AND a real k20-segment best-so-far
   regression (53.43 → 47.81, ≈ −1.3 E) that the metric correctly
   prices. The dead zone explains the missing credit, not the negative
   sign.

Design debt the audit flagged: the mini contrasts mix 3-rollout
(konly) against 1-rollout arms (cheatsheet, no-study) — this would
fail report.py's own check_pair() and adds a second-order Jensen bias
favoring the noisier arm. **Before any mini number enters the paper:
rerun mini cheatsheet + no-study at 3 rollouts (480 episodes,
API-side, cheap) or drop all mini superiority language.** Queued.

## Should we switch to 1/t? (Omar's question)

`min(3000/t, 1)` IS 1/t decay wherever it is not clipped — the design
choice is the clip. Removing it entirely is measurably degenerate on
our own data: uncapped mini konly E = 143.6 (unbounded scale), and a
degenerate 50-token direct answer at accuracy 5 would score E = 381.6
— the uncapped metric rewards terseness without bound (an incentive
statement about metric-aware arms; measured arms currently show no
such gaming).

But keep-the-cap vs no-cap is not the whole option space
(audit-corrected): a log-token-window AUC (Dodge et al. "Show Your
Work"; Dolan–Moré performance profiles are the classic answer to
one-anchor-on-heterogeneous-solvers; cf. DAWNBench/MLPerf
threshold-sensitivity history) would fix the dead zone with neither
pathology — its real costs are a construct change and loss of
Appendix-C comparability. So the honest justification for keeping
E(3000) primary is **preregistration + replication continuity**, the
clinical-trial primary-endpoint norm — not that no alternative exists.
(grade_teacher.py already plots on a log10(tokens/3000) axis
internally.)

Recommendation, unchanged in substance, corrected in grounds:

1. Keep Appendix-C E(3000) primary for every claim.
2. Publish the raw per-arm four-point (mean tokens, score) curves as
   the headline figure — the sufficient statistic from which any
   E(T0) is derivable; future-proofs against token-efficiency drift.
3. Report E(T0) sweeps with pointwise bootstrap bands over the
   informative range only, linear-regime annotated; sweep tooling
   belongs in studybench/report.py, not ad-hoc scripts (queued; the
   audit left a working prototype at scratchpad/anchor_ci.py).
4. Closed-book direct accuracy is the pre-committed secondary endpoint
   for transfer claims (mini: +12.86 [+8.6, +29.1]; gpt51 test:
   pre-registered above).
5. Scope paragraph on the cost model: generated-tokens-only pricing is
   the standard scoping, but it structurally subsidizes study arms
   (prompt-side notes are free). Follow-up sensitivity: recharge
   headline deltas with prompt tokens at 0.1×/0.25× output rate —
   strengthens mini's null; must not be load-bearing for the gpt51
   cheatsheet win.

## Corrections from the audit (so they cannot regress)

- "69% of E is closed-book knowledge" (draft) → 0.690 is the
  sensitivity/width; the realized share is 44.9%.
- Sonnet table cell "5.0k" → 9.7k (width 0.690 was computed from the
  correct 9,665; the cell was a transcription error).
- Mini table cells were cheatsheet-arm tokens; no-study is 1.0k/1.9k.
- "Mathematically insensitive to closed-book accuracy" → one-sided
  censoring below the tool ceiling (47.8 for konly).
- "konly > cheatsheet at every anchor, anchor-robustly" → point-sign
  consistency only; null at the primary anchor.
- "The 1/t question dissolves" → it narrows to the clip only against
  the no-cap alternative; the log-window family required engagement.
