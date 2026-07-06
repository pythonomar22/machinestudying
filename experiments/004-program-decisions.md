# 004 — Program decisions for the studying-methods phase (2026-07-06)

Decision record. We are moving from replication to our own study procedures
(self-quizzing, meta-learned cheatsheets). Four standing rules, adopted after the
faithfulness audit (experiments/003) and the author's "I think native tool call
should work":

## 1. Internal-comparison frame (Table 1 absolutes retired)

We keep our native tool-calling harness and stop targeting the paper's absolute
numbers. Expertise is defined per agent: a study procedure π works if
E(π(Σ,D); D) > E(Σ; D) for OUR Σ. All claims from here are internally controlled:
same harness, same judge, same questions, base curve as control, cheatsheet as the
reference studying method. Our Σ is stronger than the paper's (audit: ~2-4x
accuracy, ~5x cheaper tokens), which makes any studying gain we find a HARDER
result, not an easier one. The full path back to byte-for-byte absolute
replication is documented in experiments/003's accepted-divergence register.

## 2. Harness frozen

The harness that produced runs/ (base), runs-cheatsheet/, and cheatsheets/ is
frozen at git tag `harness-v1`. No changes to rollout semantics, tools, prompts,
budgets, or serving flags without re-running every baseline. Study procedures may
add NEW study-phase code (studybench/study.py siblings) and new prepended/context
artifacts, but the eval loop itself is immutable.

## 3. One judge for everything: fugu

Working grader = fugu (GRADER_MODEL in .env), grades in *-fugu/ trees. Reasons:
free credits for the iteration loop; our calibration evidence (same prompt +
0/1 claims + strict schema) shows it is not inflated relative to gpt-5.4.
Consequently the base runs are being regraded with fugu (grades-fugu/) so that
base, cheatsheet, and every future method share one yardstick; the gpt-5.4 base
grades (grades/) remain for reference. Headline results at milestones get an
optional gpt-5.4 confirmation pass. Never quote a cross-judge delta.

## 4. Overfitting discipline (30+20 public questions)

The benchmark is tiny and public; iterating procedures against real test scores
would manufacture fake expertise through researcher iteration. Rules:
- Methods are developed and tuned against SELF-GENERATED dev exams (the agent
  writes quiz questions from the corpus — required machinery for self-quizzing
  anyway). The real 50 questions are touched only at milestone evaluations.
- Every milestone evaluation is logged in experiments/ (including failures);
  no silent retries against the test set.
- Comparisons are paired by question (per-question deltas vs base under the same
  judge) with the two-stage bootstrap (report.py --ci); DSPy (n=30) is the primary
  battleground, OpenClaw (n=20) is reported but underpowered for paper-sized
  effects.

## 5. (Instrumentation) Study compute is always recorded

Study-phase tokens stay off the eval token axis (author-confirmed) but are logged
for every procedure (study episode JSONs record gen_tokens). Target artifact for
the program: the expertise-vs-study-compute curve — the paper's Appendix-D
"studying intelligence", which they left unmeasured.
