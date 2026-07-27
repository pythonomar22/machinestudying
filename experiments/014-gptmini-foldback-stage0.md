# 014 — Fold-back stage 0: gpt-5.4-mini direct vs k20f on the study slice

**Status: complete (2026-07-26). Grades saved; stopping here per Omar's
scope — anatomy and mining are the next declared steps.**

## Objective

Stage 0 of the fold-back pipeline (the go/no-go gate from the 013
discussion): measure the studier's direct↔forced-k20 gap on the
70-question study slice of the rev-3 practice set, with bare questions,
so the gap (a) sizes the mining signal and (b) the k20f trajectories can
be reused directly as mining input. Studier switched from Sonnet 4.5 to
**gpt-5.4-mini in our dspy.ReAct harness** after the lab Anthropic key
ran out of credits (the 7e95733 Sonnet variant never executed); judge is
GPT-5.4, paper contract.

## Protocol

- Code: `studying/foldback/stage0.py --model gptmini --judge gpt`
  (commit f4fafb0); `studybench/react.py` gained the `gptmini` profile:
  `openai/gpt-5.4-mini`, temperature 1.0, `max_completion_tokens` 32,768
  (the gpt-5.x surface rejects `max_tokens`), seed passed,
  provider-default reasoning effort, reasoning tokens counted in
  `completion_tokens` (same gen-token axis as every other run).
- 70 study questions (stratified split, master seed 20260715, identical
  `split.json` to any future run at this seed), budgets **direct** and
  **k20f only**, **one rollout**, bare questions (no note).
- k20f episodes written to `study/attempts/{qid}.json` with `run.py`'s
  exact layout and seed derivation (`foldback-attempt`), so mining reuses
  them without re-running.
- Judge: GPT-5.4 paper contract per question (whole-file evidence),
  grades under `study/grades/{direct,k20f}/`, aggregate in
  `study/report.json`. Run: `runs/dspy-gptminifoldback-20260726`.

## Results (140/140 episodes ok, zero gave-ups; 140 verdicts)

| budget | mean lenient | mean gen tokens |
|---|---:|---:|
| direct | 15.93 | 859 |
| k20f | 59.50 | 3,517 |

Integrity: all 70 k20f episodes ran exactly 20 forced iterations
(median 9 repository calls + finish catches), answers median 3.1k chars;
direct episodes all 0 tool turns.

## Reading (stage-0 gate)

1. **The gap is +43.6 lenient points — a large mining signal.** The
   gate to proceed with mining is comfortably passed on size; the
   *anatomy* (is the flipped claim-weight reusable knowledge?) is the
   declared next step before mining.
2. **The codex token floor was indeed a harness artifact.** In our
   harness the same model's direct costs 859 gen tokens (codex: 10.8k)
   and even forced-k20 averages 3.5k (codex k20f: 20.3k). gpt-5.4-mini
   is drastically more token-efficient here than codex ever showed —
   its direct point now sits deep inside the 3k anchor.
3. Practice-slice difficulty reads consistent with 011's test-set
   curves (k20f 59.5 here vs codex-harness test k20f 71.1; different
   budgets semantics/harness, directional only).

Caveats: single rollout, no CI; practice questions are our own rev-3
set, not the paper benchmark; these are internal study-signal numbers —
they never mix with test-set tables.
