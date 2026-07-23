# 011 — codex gpt-5.4-mini baselines on the real Study-DSPy test set

**Status: runs complete; no-study graded; cheatsheet grading blocked on
API credits (2026-07-23).**

## Objective

The 010 teacher curves were measured on our practice questions with no
study phase — not comparable to the Qwen anchors. This experiment runs
codex gpt-5.4-mini as a proper StudyBench condition pair on the real 30
test questions (`data/dspy.jsonl`), full paper protocol (4 budgets × 3
rollouts), so the only changed variables vs the 008 Qwen pair are the
model and harness. Code: `studying/codexbench.py`.

Budget semantics in codex terms: direct = 0 commands (closed-book);
k5/k20 = at most 5/20 shell commands, voluntary stop; k20f = exactly 20,
no early stop. Cheatsheet study = ≥50 forced exploration commands, note
prepended via the standard NOTE_PREFIX. Generated tokens = codex output
tokens including reasoning (event-stream ground truth). Rollouts are
independent resamples (codex exposes no sampling seed). Compliance,
sandbox outcomes, and self-reported tool logs (command/motivation/
discovery) recorded per case.

## Runs (both complete, zero session failures)

- `runs/dspy-codexmini-nostudy-20260723` — 360/360 records.
- `runs/dspy-codexmini-cheatsheet-20260723` — study (6 min, ~50 forced
  commands, 10,587-byte cheatsheet) + 360/360 records.

## Results

No-study (GPT-5.4 paper judge, 360 verdicts, one era):

| budget | mean lenient | mean gen tokens |
|---|---:|---:|
| direct | 25.82 | 10.8k |
| k5 | 64.91 | 13.5k |
| k20 | 67.10 | 20.0k |
| k20f | 71.06 | 20.3k |

**Expertise (official 4-point WAUC): 16.78.**

Reference (008, same questions/judge, Qwen3.5-9B + DSPy ReAct):
no-study 9.04, cheatsheet 15.90.

Interpretation so far: codex mini is 3–7× more accurate than Qwen base
at every budget, yet its expertise (16.78) barely clears Qwen's
cheatsheet (15.90) — its cheapest point costs 10.8k tokens, so ~72% of
the weight axis floors to zero. Model+harness capability without token
efficiency is exactly the paper's brute-force archetype. Note also
codex's test-set k5 (64.9) lands near its practice-set k5f (70.5,
experiments/010), supporting the rev-3 practice set's difficulty
calibration.

Cheatsheet run: **ungraded** — the OpenAI API key hit
`insufficient_quota` after 640 verdicts today (280 teacher + 360
no-study). `grades/dspy-codexmini-cheatsheet-20260723/.../grade_failures.json`
records all 360 as quota failures; grading is fully idempotent — rerun
`python -m studying.codexbench grade --run-id
dspy-codexmini-cheatsheet-20260723` once credits are restored.

## Caveats (standing)

One grading prompt in the no-study batch tripped OpenAI moderation
(`invalid_prompt`) and passed on retry; per-case isolation is now built
into the grader. Cross-model comparisons share questions and judge but
not harness/tools; the WAUC comparison is the metric working as
designed, not a same-agent ablation.
