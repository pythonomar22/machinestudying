# 005 — Validation-set construction v2 (StudyBench pipeline replication)

**Status: complete (2026-07-21).** Deliverables:
`data_collection/artifacts/7_optimize_bundles/7_fulldspy_validation.jsonl`
(32 questions) and `7_smalldspy_validation.jsonl` (7 questions, derived).

## What this is

A full replication of the StudyBench coding-suite construction pipeline
(paper Appendix A.1, stages 1–5) with one declared substitution — public
stanfordnlp/dspy GitHub issues as the session source — producing a
validation set for the self-quizzing studying method. Stage files
`data_collection/1_...py` through `7_...py`, run in order; every prompt,
codex event stream, and raw output archived under
`data_collection/artifacts/<stage>/`. The complete fidelity ledger
(verbatim-copied vs inferred, every deliberate divergence, every
correction) is `data_collection/FIDELITY.md` — read it before touching
the pipeline.

## Final numbers

- 1,634 issues -> 302 filtered seed sessions -> 195 corpus-groundable ->
  4 capability topics (42/35/32/22 members, 64 noise).
- Stage 3a: 20 candidates x 4 topics (GPT-5.4 xhigh in codex, verbatim
  A.2, 10 nearest-centroid seeds); Stage 3b critic keeps 8/20 (paper's
  5/12 ratio); Stage 4 rubrics (verbatim A.4); Stage 5: syntax check +
  sandbox run + A.5 self-grade to 100, <=3 revision rounds.
- Result: 32/32 bundles pass with zero drops (3 needed revision). Gold
  answers are runnable offline programs (DummyLM), sandbox-verified,
  self-scored exactly 100.

## The big lesson (register incident)

The published A.2/A.3 templates never mention runnable code; the demand
lives in the paper's UNPUBLISHED placeholder values (evidence: 27/30
released questions say "runnable", 10/30 name DummyLM). Our first
reconstruction stated the contract descriptively, no validator enforced
it, and the critic's concision license silently stripped every program —
discovered only when stage 7's sandbox found nothing to execute. Fix:
imperative deliverable contract in the placeholder values + deterministic
validation (one fenced compiling program per answer; >=400-char,
>=2-paragraph questions) at every rewrite point. Full retrospective in
FIDELITY.md Principles ("form-register exception").

## Open items

1. Decontamination-by-dropping vs the 30 Study-DSPy test questions (the
   one remaining allowed test-set interaction) before the set is used.
2. Derived smalldspy set is lean (7 questions, no evaluation_metrics);
   fallback: dedicated critic pass over the 21 in-scope candidates.
3. One-time full-corpus no-study + cheatsheet baselines for the
   self-quizzing line (ledgered decision from the scope note).
4. No human review anywhere (declared gap, no substitute).
