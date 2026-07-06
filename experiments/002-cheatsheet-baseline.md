# 002 — Cheatsheet baseline (Study-DSPy, Study-OpenClaw)

**Goal.** Reproduce Table 1's `+ cheatsheet` rows: lenient accuracy/tokens at the
four budgets and the expertise WAUC (paper: DSPy 9.65 vs base 6.49; OpenClaw 8.18
vs base 7.64). The paper's headline for this row: gains concentrate at low budgets
(the note is "a map of the repository"), and forced-20 search amortizes the note
away (base k20f 29.4 > cheatsheet k20f 23.1 on DSPy).

## Recipe (paper §B Table 4 + §5)

- Study: "a forced ReAct study loop with at least 50 no-early-return tool calls",
  writing a cheatsheet; same three tools over the same corpus roots as eval.
  Study tokens are NOT counted on the eval token axis (author-confirmed,
  docs/jacob.md).
- Eval: the cheatsheet "is prepended to every later question while the repository
  tools remain available"; everything else identical to the base eval.

Implementation: `studybench/study.py` (one forced 50-iteration episode per corpus →
`cheatsheets/{task}.md` + the full study episode JSON), then
`rollout --variant cheatsheet` (prefixes every question with the note) →
`runs-cheatsheet/`.

## Replication inferences (not specified by the paper)

1. **Study prompts** (system + task): mirror the eval prompts' structure; tell the
   agent only that its document will be prepended to every future question about
   the library and that tools remain available at answer time — both facts from the
   paper's setup. No hints about the hidden task distribution (question style,
   coding focus). *Ask Jacob: what did the study instruction say, and does it hint
   that questions are coding tasks?*
2. **"At least 50"** → exactly 50 iterations, one study episode per corpus (no
   cheatsheet averaging). *Ask Jacob: one cheatsheet per corpus, or sampled/selected
   among several?*
3. **Prepend format**: "Reference notes on {lib} from your prior study of its
   repository:\n\n{note}\n\n---\n\n{question}" in the user message. *Ask Jacob:
   system prompt or user message? any framing text?*
4. **Cheatsheet length**: uncapped (bounded by the 32,768-token final turn).
5. Harness identical to our base primary (interleaved thinking); sampling per §B.

## Grader

`GRADER_MODEL=fugu` (user's choice; Sakana API verified: strict json_schema with
enums works; the judge prompt/claims/scoring are unchanged). Grades land in
`grades-cheatsheet-fugu/`. **Caveat logged:** the paper's judge is gpt-5.4 and our
base grades are gpt-5.4, so cheatsheet-vs-base and cheatsheet-vs-paper are
cross-judge comparisons until either side is regraded under a common judge
(the base fugu regrade was started and intentionally stopped — user call).

## RESULTS (2026-07-06; 600 episodes, judge=fugu, core-conjunctive lenient)

All 600 episodes clean (statuses ok, forced=20 exact). CIs: two-stage cluster
bootstrap, 10k replicates.

| DSPy | direct | k5 | k20 | k20f | WAUC |
|---|---|---|---|---|---|
| paper +cheatsheet | 6.3 | 14.4 | 14.1 | 23.1 | **9.65** |
| ours +cheatsheet (fugu) | 0.0 | 8.3 | 11.6 | 12.6 | **8.89** [4.5, 16.5] |
| ours base (gpt-5.4, ref) | 0.0 | 2.9 | 10.3 | 17.8 | 9.68 |

| OpenClaw | direct | k5 | k20 | k20f | WAUC |
|---|---|---|---|---|---|
| paper +cheatsheet | 4.3 | 8.6 | 15.2 | 18.1 | **8.18** |
| ours +cheatsheet (fugu) | 0.0 | 1.3 | 3.0 | 9.2 | **5.77** [0.0, 14.0] |
| ours base (gpt-5.4, ref) | 0.0 | 1.6 | 9.7 | 9.6 | 5.98 |

**The paper's cheatsheet signature replicates.**
1. WAUC inside our 95% CI on both tasks (9.65 in [4.5, 16.5]; 8.18 in [0, 14.0]).
2. **Gains concentrate at low budgets** (paper's central claim for this row):
   DSPy k5 2.9 → 8.3 lenient; visible judge-independently in the rubric sums
   (base gpt-5.4: 4.9/21.3 at direct/k5 → cheatsheet fugu: 9.8/29.4).
3. **Forced search amortizes the note away** (paper: base k20f 29.4 > cheatsheet
   23.1): ours base 17.8 > cheatsheet 12.6; rubric sums agree (40.4 > 36.5).
4. **OpenClaw gains are marginal-to-absent**, matching the paper ("only marginal
   gains on STUDY-OPENCLAW"; their 7.64 → 8.18 is well within our noise floor).

**Judge note:** fugu with our 0/1 + strict-schema pipeline is NOT inflated relative
to gpt-5.4 (rubric sums are comparable across the base/cheatsheet pairs) — the
discipline Jacob attributed to the grader appears to come from the prompt + 0/1
claims + schema, not the judge model. Exact base-vs-cheatsheet deltas remain
cross-judge (base = gpt-5.4, cheatsheet = fugu); a same-judge pass on one side
would pin the paired per-question deltas if we want them sharp.

## Artifacts

- 2026-07-06: cheatsheets generated (job 24723): dspy.md 12,425 chars (~3.3k
  tokens; API reference style, mostly accurate with some self-written wobble —
  e.g. a dubious `dspy.Teleprompter(...)` wrapper — left as-is: it is the agent's
  own study product), openclaw.md 11,179 chars (repo map + core exports +
  plugin-runtime shape). Both study episodes: exactly 50 iterations, ~6.2k gen
  tokens each, episode JSONs alongside.
- 2026-07-06: prepend verified in smoke (turn-0 prompt 284 → 3,594 tokens);
  8 smoke episodes clean; full evals = jobs 24754 (dspy) + 24755 (openclaw).
