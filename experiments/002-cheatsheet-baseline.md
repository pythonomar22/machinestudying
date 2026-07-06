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

## Artifacts

- 2026-07-06: cheatsheets generated (job 24723): dspy.md 12,425 chars (~3.3k
  tokens; API reference style, mostly accurate with some self-written wobble —
  e.g. a dubious `dspy.Teleprompter(...)` wrapper — left as-is: it is the agent's
  own study product), openclaw.md 11,179 chars (repo map + core exports +
  plugin-runtime shape). Both study episodes: exactly 50 iterations, ~6.2k gen
  tokens each, episode JSONs alongside.
- 2026-07-06: prepend verified in smoke (turn-0 prompt 284 → 3,594 tokens);
  8 smoke episodes clean; full evals = jobs 24754 (dspy) + 24755 (openclaw).
