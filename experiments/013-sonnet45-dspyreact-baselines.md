# 013 — Claude Sonnet 4.5 baselines in the paper's DSPy ReAct harness

**Status: runs in flight (2026-07-25).**

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
the two arms' manifests differ in `source_commit` (e5b0365 vs d230438);
the eval code path is identical, documented here for the audit.

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

Cheatsheet arm: in flight.

## Interpretation

TBD after adversarial audit.
