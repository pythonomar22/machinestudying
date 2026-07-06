# 003 — Faithfulness audit (three-agent adversarial; 2026-07-06)

Three adversarial auditors (harness / grading / study+data lanes) prosecuted the
replication against docs/paper.md, docs/blog.md, docs/jacob.md (including the
author's 2026-07-06 replies), and the pinned dspy source. 50 register entries.
Full structured register: session workflow wf_8afe65cb-551.

## The verdict in one paragraph

The dataset and scoring substrate are faithful (byte-exact questions/rubrics/spans,
A.5 judge prompt with the confirmed 0/1 change, gpt-5.4 judge, verified WAUC
formula, study tokens off-axis, judge sees the original un-prefixed question). The
harness is not: the paper ran **dspy.ReAct** — a text-field loop where every step
is a fresh stateless LM call (fresh ~1.5k-token thinking each turn), with a
`finish` tool for voluntary stopping and a ChainOfThought **extract** step
producing the graded answer — while we built a native tool-calling chat loop that
is materially stronger and ~5x cheaper in tokens. With the author's correction
that **Table 1 lenient = pure weighted sum**, our numbers are 2-4x above the paper
on every cell and excluded by our own CIs; decomposition shows the token axis
(harness shape) accounts for roughly half the WAUC gap (26.4→15.3 DSPy, 20.2→13.2
OpenClaw when substituting the paper's token means) and residual accuracy
inflation (stronger harness and/or judge severity) for the rest.

## Corrections applied

- report.py: `lenient` column/CI is now the pure weighted sum (Table 1 metric);
  the core-conjunctive gate is relabeled `len-cc` (strict-adjacent only).
- experiments/001 and 002: correction sections appended; the earlier
  "statistically consistent" (001) and "WAUC inside CI" (002) verdicts are
  superseded — they quoted the contradicted metric.

## Where every Table-1 discrepancy now stands

| gap | explanation | status |
|---|---|---|
| tokens 4-6x low everywhere | dspy.ReAct fresh-per-step thinking + extract step vs our chat loop | author-confirmed mechanism |
| k20 ≈ k20f in ours; k20 ≈ k5 in paper | no `finish` affordance in our loop → our agent never stops early | structural, author-implied |
| accuracy 1.5-4x high | stronger harness (nudges, cap-notices, better tools?, message-history memory) + unknown judge-severity share | needs the dspy.ReAct rerun + a cross-grading test to attribute |
| cheatsheet deltas | judge-confounded (base gpt-5.4 vs cheatsheet fugu) + same harness caveats | needs same-judge pass |

## Consolidated questions for Jacob (deduped, ranked)

**P0 — block interpreting our numbers against Table 1**
1. *Harness config:* which dspy version/commit ran the harness, with what ReAct
   signature (string + any instructions) and adapter (ChatAdapter default? JSON
   fallback?)? Was dspy.LM pointed at vLLM (which provider prefix), and did the §B
   sampling params (top_k/min_p/repetition_penalty) actually reach the server
   through litellm?
2. *Forced-20 mechanism:* how was "no early stopping" implemented on top of
   dspy.ReAct (finish tool removed, finish ignored, re-prompting?) — and did the
   prompt tell the model upfront it must use exactly 20 iterations? Same question
   for the study loop's "at least 50".
3. *Direct budget:* concretely what ran — dspy.Predict, ChainOfThought, or
   ReAct(max_iters=0) (which falls through to the extract step)?
4. *Tools:* the three tool functions' signatures and output behavior — grep
   flags/max matches, glob semantics, read_file interface (whole file vs line
   ranges), any observation-length cap — and the readable file set (dspy/+tests/
   and src/+extensions/? extension-filtered? were in-root markdown/config files
   readable? root README?).
5. *Cross-grading:* would he run 5-10 of our raw answers through his actual
   grader, or share one fully-graded transcript + grader config? This single
   exchange attributes the residual accuracy gap between judge severity and
   answer quality.

**P1 — material**
6. Voluntary stopping: is finish the only stop channel, and roughly how many
   iterations did the base model actually use at k=20?
7. Judge config: reasoning effort; the exact 0/1 prompt wording (what replaced the
   0.5 line); structured-output schema vs free JSON; evidence = whole numbered
   files or the released excerpts?
8. Hallucinated-API check: a separate deterministic pass (symbol existence against
   the pinned repo)? What ran for TypeScript (tsc? node?) and what were the
   Python sandbox semantics (exit-0 vs compile-only; how the code block was
   selected from a multi-block answer)?
9. Which prediction field(s) went to the grader and the sandbox — the extract
   step's answer field only?
10. Did the ReAct signature/system text disclose anything about the repository
    (name, layout, code roots), or did the agent start blind?
11. Study loop: the exact study instruction (does it hint the downstream questions
    are coding tasks / that the note gets prepended / that tools remain
    available?); one cheatsheet per corpus with no best-of-N?; did the cheatsheet
    come from the extract step; ~how many tokens did one study run generate?
12. Cheatsheet prepend: inside the question field or the system/instructions, and
    with any framing header?

**P2 — nice to pin**
13. Context length served for Qwen3.5-9B; did dspy's trajectory truncation fire in
    forced-20 runs?
14. Token axis composition: sum of completion tokens (thinking included) over all
    LM calls including extract and adapter retries, averaged over all 3×N
    episodes?
15. Strict composition exactly: compile ∧ API-check ∧ all-cores=1 → sum, else 0?
16. needs_regrade handling when the judge sets it.
17. Could he share one generated cheatsheet (or characterize length/style)?

## The accepted-divergence register

What a byte-for-byte replication would require changing, item by item. We are
ACCEPTING these divergences (decision record: experiments/004) on the strength of
the author's "I think native tool call should work"; this register is the complete
path back if we ever want absolute Table-1 numbers. Status legend:
**[known]** = we know what to build; **[blocked-on-Qn]** = requires the answer to
question n in the list above.

Harness (all: rip out studybench/rollout.py's chat loop, rebuild on dspy.ReAct):
1. [blocked-on-Q1] dspy.ReAct as the loop: exact dspy version/commit, signature
   string + instructions, adapter, dspy.LM provider prefix, and verification that
   §B sampling params survive litellm. Structural consequences we'd inherit:
   fresh stateless LM call per step (fresh ~1.5k-token thinking each turn — the
   paper's token axis), trajectory re-serialized into each prompt, JSON-parse
   retries, trajectory truncation on context overflow.
2. [known] `finish` tool as the only voluntary stop (their k20 ≈ k5 behavior);
   [blocked-on-Q2] the forced-N mechanism on top of it (finish removed / ignored /
   re-prompted; whether the model is told upfront).
3. [blocked-on-Q3] direct budget: Predict vs ChainOfThought vs ReAct(max_iters=0).
4. [known] answer channel: graded answer = the extract step's output field, not
   the last assistant message; [blocked-on-Q9] which fields exactly.
5. [blocked-on-Q4] tool implementations: grep/glob/read_file signatures, output
   caps, truncation messages, and the readable file set (extension filtering,
   in-root markdown, README). Ours are invented (GREP_MAX_MATCHES=50,
   READ_MAX_LINES=500, OBS_MAX_CHARS=25000).
6. [blocked-on-Q10] system-prompt disclosure: we hand the agent the repo name and
   code roots ("a free map"); theirs may start blind.
7. [known] remove our inventions, none of which exist in dspy.ReAct: cap-notice
   message, nudge machinery, tool-shaped-answer rejection, forced_short status,
   parallel_tool_calls=False, interleaved-thinking passback, per-episode seeds.

Grading:
8. [blocked-on-Q8] the hallucinated-API deterministic check (never built): likely
   symbol/import existence against the pinned repo; unknown TS equivalent. Affects
   strict/compile columns only (lenient skips automatic zeros).
9. [blocked-on-Q7] judge reasoning effort (our ablation: moves cells 10-30%
   relative), exact 0/1 wording, output schema vs free JSON, evidence = whole
   files vs excerpts (our ablation: ±1pt).
10. [blocked-on-Q8] sandbox semantics: exit-0 execution (ours) vs compile/import
    only (possibly theirs); code-block selection from multi-block answers.
11. [blocked-on-Q16] needs_regrade policy. [blocked-on-Q14] token-axis
    composition (extract call? adapter retries? ungradable episodes?).

Study (cheatsheet):
12. [blocked-on-Q11] study instruction text; whether "at least 50" means exactly
    50; single cheatsheet vs best-of-N; cheatsheet from the extract step; study
    sampling params.
13. [blocked-on-Q12] prepend placement (question field vs system) and framing
    text. Ours: user-message header "Reference notes on {lib} from your prior
    study...".

Serving:
14. [blocked-on-Q13] served context length (ours 262144) and whether trajectory
    truncation fired in their forced-20 runs.

## Register updates (2026-07-06, author round 3 — see experiments/006)

Answered and implemented: item 2 forced-N = catch finish, return keep-searching,
turn stays (Q2 closed); item 3 direct = dspy.Predict (Q3 closed); item 5 read_file
= line ranges capped at 200 (Q4 partially closed — grep/glob caps and observation
truncation still ours); item 1 partially (harness rebuilt on the pinned dspy
commit; his exact version still unknown). Still open: Q4 remainder, Q5
cross-grading, Q7 judge details, Q8 hallucinated-API check, Q10-12 study/prepend.

## Recommended next experiment (004)

Build the harness AS dspy.ReAct (the pinned library is in corpora/dspy; point
dspy.LM at our vLLM server), implement forced-N by the mechanism Jacob describes
(or ablate both if he doesn't reply), rerun the 600-episode base grid, grade with
gpt-5.4 under pure-sum lenient, and compare to Table 1. This is the decisive test:
it should simultaneously reproduce the token axis (34.6k k20f), the k20 early
stopping, and — if the residual is harness — the accuracy levels.
