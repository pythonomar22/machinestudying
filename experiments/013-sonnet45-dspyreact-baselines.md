# 013 — Claude Sonnet 4.5 baselines in the paper's DSPy ReAct harness

**Status: complete — both runs graded and adversarially audited
(2026-07-25, workflow wf_ad1427f5-e12).**

## Objective

011/012 measured codex gpt-5.4-mini, but the codex harness is a black box:
its system prompt forces ~10k generated tokens even at direct, flooring
~72% of the WAUC weight axis, and it exposes no seed or sampling control.
Decision (Omar, 2026-07-25): retire codex entirely and run a strong API
model through the *same* dspy.ReAct harness Qwen uses, so the only changed
variable vs the 008 anchors is the model. The lab provides an Anthropic
key (no OpenAI), so the model is **Claude Sonnet 4.5** and the judge moves
to Sonnet 4.5 under the identical paper grading contract.

## Protocol

- Code: `studybench/react.py --model sonnet45` (commit e5b0365) —
  `anthropic/claude-sonnet-4-5`, temperature 1.0, max_tokens 32,768,
  hosted Anthropic API. No sampling seed exists on this API; the ledgered
  per-episode seeds are recorded but inert (single rollout anyway).
- Same everything else: 30-question `data/dspy.jsonl`, pinned corpus
  checkout, grep/glob/read_file tools with paper caps, budgets
  direct/k5/k20/k20f, NOTE_PREFIX, master seed 20260715.
- **One rollout** (API cost; Omar's directive), vs 3 in the Qwen anchors.
- Conditions: `baseline` (`runs/dspy-sonnet45-nostudy-20260725`) and
  `cheatsheet` (`runs/dspy-sonnet45-cheatsheet-20260725`, Sonnet studies
  its own 50-forced-iteration cheatsheet, paper protocol).
- Judge: `--judge sonnet` (grade-id `sonnet-4-5`) — claude-sonnet-4-5,
  paper contract (same prompt, strict claim schema, weighted-claim-sum
  lenient score), via litellm's response_format→forced-tool conversion.
  Smoke-validated: finish_reason=stop, strict JSON, usage ledger intact.

## Judge-change caveat (standing)

GPT-5.4 (`--judge gpt`) was the paper judge for every prior number
(Qwen anchors 9.04/15.90, codex 16.78/18.28). Without an OpenAI key it
cannot run. Sonnet-judged numbers are **not directly comparable** to any
GPT-5.4-judged number; within-013 comparisons (nostudy vs cheatsheet,
budget curves) are clean. Sonnet judging Sonnet answers also carries a
self-preference risk — flagged for the audit. If cross-model claims are
needed later, regrade the 008 Qwen anchor runs with `--judge sonnet`
(720 verdicts) to put all rows on one judge.

## Smoke evidence (deleted per convention after passing)

1-question smokes of both conditions passed: forced k20f ran exactly 20
iterations (finish-catches consumed), token ledger self-consistent,
grading round-trip validated. Sonnet direct cost ~1.6k gen tokens — the
codex 10k direct floor is gone in this harness, so the 3k-anchor weight
at direct is ~1.0. Note: the 2-iteration smoke study extracted a
degenerate 67-char cheatsheet (artifact of the absurd smoke budget);
the full 50-iteration note is manually inspected before the eval spends.

## Study-extraction failure and fix (ledgered)

The first full cheatsheet study (commit e5b0365) spent 50 forced
iterations / 302k gen tokens, then extracted a 75-char
continuation-thought ("Let me complete my 50th iteration…") as the
"cheatsheet" — the same signature as the 2-iteration smoke, i.e. 2/2
systematic: dspy's extract step never tells the model studying is over,
and Sonnet keeps role-playing the loop (Qwen never did this). The
non-empty guard let the junk note through; the eval fan-out was killed
after 11 episodes. Failed artifacts preserved under
`runs/failed-artifacts-013/cheatsheet-degenerate-study/`.

Fix (commit d230438): `ForcedReAct(closing_observation=…)` appends
"Study complete … write the complete cheatsheet" to the final study
observation only — k20f eval episodes byte-identical to before, so the
already-graded baseline stays valid — plus a 500-char degenerate-note
floor that fails loudly (`study.failed.json`) instead of spending on a
garbage-note eval. Re-smoke produced a real 5.3k-char note. Consequence:
the two arms' manifests differ in `source_commit` (baseline e5b0365,
cheatsheet 85beacb which contains the d230438 fix); audit-verified via
`git diff e5b0365..85beacb`, the diff touches only the study path — the
eval call site never passes `closing_observation` — so eval episodes
run identical code on both commits.

## Results

No-study (`runs/dspy-sonnet45-nostudy-20260725`, sonnet-4-5 judge,
120 verdicts, zero failed episodes):

| budget | mean lenient | mean gen tokens |
|---|---:|---:|
| direct | 27.83 | 1.6k |
| k5 | 55.80 | 9.7k |
| k20 | 62.63 | 16.8k |
| k20f | 66.47 | 25.2k |

**Expertise (4-point WAUC): 38.19.**

The leniency-vs-compute climb replicates in our own harness on a hosted
frontier model (+38.6 lenient from direct to k20f), and the direct point
costs 1.6k tokens — inside the 3k anchor, weight ≈ 1.0 — which is what
the codex harness's 10k-token floor could never show. Cross-judge
caveat: 38.19 is a sonnet-judged number; do not place it in the same
table as GPT-5.4-judged 9.04/15.90/16.78/18.28 without regrading.

Cheatsheet (`runs/dspy-sonnet45-cheatsheet-20260725`: 50-forced-iteration
self-study, 13,162-char note, 287,177 study gen tokens; 120 verdicts; two
k5 episodes errored and were resampled per the standing rerun protocol —
the live session log showed `AdapterParseError: Adapter JSONAdapter
failed to parse the LM response` with Sonnet wrapping its JSON in a stray
`"$PARAMETER_NAME"`-style key; the rerun overwrites the errored episode
files, so this quote is the only surviving record. Retries condition
those 2 answers on parseability and only the cheatsheet arm needed them
(0 baseline) — a disclosed, minor asymmetry. Final population 120/120 ok):

| budget | mean lenient | mean gen tokens |
|---|---:|---:|
| direct | 31.57 | 2.0k |
| k5 | 54.30 | 11.0k |
| k20 | 60.40 | 14.9k |
| k20f | 63.20 | 24.7k |

**Expertise (4-point WAUC): 39.35.**

## The within-013 picture (same questions, same sonnet-4-5 judge, 1 rollout)

| condition | direct | k5 | k20 | k20f | E |
|---|---:|---:|---:|---:|---:|
| Sonnet 4.5 no-study | 27.8 | 55.8 | 62.6 | 66.5 | 38.19 |
| Sonnet 4.5 + own cheatsheet | 31.6 | 54.3 | 60.4 | 63.2 | 39.35 |

Paired stats (n=30 questions, `scripts/paired_stats.py`, 10k reps,
seed 20260715 — reproduces exactly):

- ΔE = +1.16, 95% CI **[−5.34, +7.96]**, P(Δ>0) = 0.63 — **not
  separable from noise at one rollout.**
- Direct-budget paired delta: +3.73 lenient (10 wins / 4 losses /
  16 ties), itself non-significant (sign test p ≈ 0.18, bootstrap CI
  [−4.5, +12.0]); k5/k20/k20f nominally move −1.5/−2.2/−3.3.

## Adversarial audit (wf_ad1427f5-e12, 2026-07-25)

Five independent auditors + verification pass. Everything reproduces:
all 240 lenient scores re-derived from rubric weights (0 mismatches),
WAUC recomputed from scratch to full precision, provenance hashes
(episode↔grade↔manifest, seeds, note-prefix question hashes) verified
120/120 per arm, budget semantics exact (direct 0 tools, k20f 20 forced
iterations everywhere), and the note provably present only in the
cheatsheet arm's prompts.

Judge-noise findings (the important ones):

- The judge's self-reported `question_score` disagrees with the
  authoritative claim-weighted sum on 99/240 grades (41%, mean |diff|
  ~4–7 points, no systematic direction). Sensitivity: rescoring with the
  raw question_score gives ΔE = **−0.43** (vs +1.16).
- Exhaustive claim-level review (1,200 verdicts) found ~9 score-vs-
  rationale contradictions (0.75%), in BOTH directions (7 lenient — net
  favoring the cheatsheet arm as-graded, incl. 3 in cheatsheet/direct —
  and 2 harsh w=45 cases in no-study k20f). Symmetric correction moves
  ΔE to **−0.77** (CI [−6.95, +5.57]) and the direct delta to
  +1.7…+3.1 depending on variant.
- Verdict: the budget→leniency climb is monotone and survives every
  correction variant; **the cheatsheet effect has no stable sign** —
  +1.16/−0.43/−0.77 across scoring choices — so the printed 39.35 >
  38.19 ordering is not claimable, only the null is.

## Interpretation (audited)

1. **The budget→leniency curve replicates cleanly in our own harness**
   (+38.6 lenient from direct to k20f, monotone, robust to all judge
   corrections) — the 011 codex observation was not a codex-harness
   artifact. The judge-independent contrast with codex: codex's direct
   point cost 10.8k gen tokens (3k-anchor weight 0.28) while Sonnet's
   costs 1.6–2.0k (weight 1.0), which is why Sonnet's E lands near ~38
   while codex-mini was capped at ~17–18 (E values cross judges,
   directional only).
2. **No detectable cheatsheet effect for Sonnet 4.5 at this sample
   size.** ΔE is +1.16 nominal with CI [−5.34, +7.96] and its sign
   flips under judge-noise corrections; the CI still admits effects
   larger than codex-mini's +1.5. What is consistent across models is
   the *shape*: any nominal gain sits at direct, nothing at tool
   budgets. If a studying method is to beat this, it must compress
   answer-relevant knowledge into the direct budget (the fold-back
   thesis) — navigation hints are what a strong searcher re-derives in
   its first tool calls.
3. Sonnet attempts `finish` early under k20f forcing (mean 5.4×/episode
   no-study, 4.9× cheatsheet); forced continuation coincides with +3.8
   (no-study) / +2.8 (cheatsheet) lenient over k20, not significance-
   tested.

Caveats: one rollout; no sampling seed (Anthropic API); sonnet-judged
numbers are not comparable to any GPT-5.4-judged anchor without a
regrade; Sonnet-judging-Sonnet self-preference — the audit found no
hallucinated-credit cases in 12 manually read grades, but the 0.75%
contradiction rate nets in the cheatsheet arm's favor as-graded. If
cross-model tables are ever needed, regrade 008 with `--judge sonnet`.

Next: judge hardening (post-grade lint for rationale/score
contradictions), then the fold-back pipeline on Sonnet direct.
