# 005 — Self-quizzing: design (draft v1, 2026-07-06)

**Intent.** A studying procedure π_quiz that converts corpus D + study compute into
a better agent, grounded in the human practice the user described: a student with
a textbook does the practice questions at each chapter's end — or writes their own
— because *retrieval practice* beats re-reading (the testing effect; Roediger &
Karpicke 2006). We instantiate quizzing as the **learning signal for the note**:
the agent quizzes itself, and what it gets wrong is exactly what its note must
teach. Success = E(π_quiz(Σ,D); D) > E(Σ; D) and > E(π_cheatsheet(Σ,D); D) under
the frozen harness (experiments/004 rules).

## 1. The central design idea: error-driven note distillation

The cheatsheet baseline writes down whatever the agent noticed in 50 steps of
reading. Most of that is wasted: the note re-states things the model already knows
(it "knows" DSPy from pretraining — the problem is its knowledge is *stale and
partly wrong*, e.g. it invents `GeoTeleprompter`). The information-optimal note
encodes the **delta between the model's priors and the corpus truth** — and the
only way to find that delta without test access is to make the model commit to
answers from its priors and check them against the corpus. That is self-quizzing.

Equivalently: this is meta-learning of the note. Each round, quiz misses *with the
current note in context* identify where the note is still insufficient; the
distill step is the update. (This unifies the user's two original ideas —
"meta-learn a better cheatsheet" and "quiz yourself during studying" — into one
procedure: quizzing IS the gradient signal for the cheatsheet.)

The final artifact is identical in kind to the cheatsheet — one markdown note
prepended to every eval question — so the eval mechanics, the frozen harness, and
the cheatsheet comparison are exactly apples-to-apples: **the studying algorithm
is the only variable.**

## 2. The loop

Per corpus; all calls are Qwen3.5-9B (the agent studies alone — no bigger model,
no external judge at study time; see §6). One **round** r:

```
COVER   pick the next K modules by a fixed corpus-derived order (top-level
        packages/dirs weighted by code size; "chapters" of the textbook).
        Deterministic; no model choice → guarantees breadth, prevents the
        agent from re-quizzing its favorite APIs.

QUIZ    for each module: a ReAct episode (tools, capped iterations) that skims
        the module and writes M quiz questions, each with (a) the question,
        (b) the files/symbols whose reading motivated it (evidence anchors),
        (c) a difficulty/type tag. Question types are prescribed for
        diversity: usage ("write code that ..."), behavior ("what happens
        when ..."), location ("where/how is ... implemented"), pitfall
        ("what breaks if ..."). Types derive from the corpus being a code
        library, not from the benchmark (see §5 leakage analysis).

ATTEMPT closed book: a fresh context containing ONLY note_{r-1} + the question
        (no tools). The model commits to an answer from its priors + note.
        This is retrieval practice; it is also exactly the eval-time "direct"
        condition, which is where WAUC weight concentrates.

VERIFY  open book: a fresh ReAct episode (tools; for DSPy also run_python in
        the pinned sandbox) that checks the attempt against the corpus and
        returns verdict ∈ {correct, partial, wrong}, the corrected answer,
        and file:line evidence. Code answers are executed — execution is the
        ungameable part of the feedback signal (available for DSPy; OpenClaw
        verification is corpus-reading only, which becomes a built-in
        ablation of execution grounding).

DISTILL for wrong/partial items only: update the note with delta-encoded
        entries ("prior belief: X. Actually: Y — dspy/clients/base_lm.py:96").
        Correct items add NOTHING (the model already knows them; note space
        is for the delta). At round end, one rewrite pass merges/organizes
        the note (soft cap ~4k tokens: comparable to the cheatsheet's 3.3k;
        logged, ablatable).
```

Snapshot note_r after every round; log study tokens (all completion tokens across
every study call). Milestone evals at r ∈ {1, 2, 4, 8} give the
**expertise-vs-study-compute curve** — the paper's Appendix-D "studying
intelligence", which they left unmeasured. The cheatsheet baseline is one point
(~6k study tokens); our r=1 lands near it, r=8 an order of magnitude above.

## 3. Quiz artifacts are kept, always

Every round emits `study-quiz/{task}/r{r}/` JSONLs:
`{question, type, module, evidence_anchors, attempt, verdict, correction,
evidence, tokens}`. Three reasons: (a) debuggability (CLAUDE.md: read everything);
(b) the held-out **dev exam** (§4); (c) this is precisely the dataset a future
weights track (SFT/RL on verified self-QA) would train on — designed once,
reusable.

## 4. Overfitting discipline, operationalized (rule 4 of experiments/004)

- **Dev exam**: a random ~20% of each round's questions are held out — never used
  for distillation, never seen with the note in the ATTEMPT step. Procedure
  iteration (prompt tweaks, K/M choices, cap tuning) is driven ONLY by dev-exam
  closed-book accuracy (self-verified) + note-quality reads. StudyBench's 50
  questions are touched only at pre-registered milestones, each logged in
  experiments/ regardless of outcome.
- **Pre-registration**: before the first milestone eval we freeze and write down:
  procedure config, study budgets, the exact comparisons (paired per-question
  deltas vs base and vs cheatsheet, fugu judge, two-stage bootstrap), and the
  success criterion (WAUC delta > 0 with CI excluding 0 on DSPy).
- **Leakage checklist** (checked per experiment, in the log):
  1. No study-time tool can read `data/`, `grades*/`, `runs*/`, `experiments/`
     (RepoTools only mounts corpus roots — verified; keep it that way).
  2. Study prompts contain no benchmark-derived phrasing (no "self-contained
     runnable program", no rubric vocabulary, no topic-label names).
  3. No procedure change motivated by real-test score movements between
     milestones (changes must cite dev-exam or artifact-inspection evidence).
  4. The QUIZ prompt's question-type list is justified from the corpus alone
     (§5), and we never show StudyBench questions to any study-time call.

## 5. The leakage question we must be honest about

Is prescribing "usage" questions (write code that uses the library) already
test-set knowledge? Our position: no — the corpus IS a code library; practicing
usage is what any student of a codebase does (the paper's own framing: the target
expert is "a developer who can take a user's question and address it correctly").
But style-matching further than that (e.g., "questions should describe a user
scenario with observed symptoms") WOULD be fitting StudyBench's house style. The
prescribed types stay at the granularity a textbook chapter quiz would have
(usage/behavior/location/pitfall), and the QUIZ prompt never mentions answer
format, length, or grading. This is recorded as an accepted judgment call —
flagged for stress-testing (red-team audit, §9).

## 6. What the agent may NOT use while studying

- No larger/other model anywhere in the loop (the paper called their own
  DeepSeek-generated QA "cheating"; our claim is *self*-studying).
- No StudyBench artifacts (checklist above).
- The sandbox (run_python against the pinned corpus install) IS allowed at study
  time: executing code against the corpus is studying-with-the-environment, the
  same category as reading it (humans run examples). It is not available at test
  time — the frozen harness has only grep/glob/read_file — which is fine: study
  and test conditions legitimately differ (the paper's own study loop differs
  from eval too).

## 7. Evaluation

- Artifact: `notes/selfquiz-r{r}/{task}.md` prepended exactly like the cheatsheet
  (same prefix framing, same `--variant` machinery generalized to arbitrary note
  dirs).
- Milestone eval: full 4-budget × 3-rollout grid per snapshot, fugu judge,
  pure-sum lenient headline.
- Comparisons (all paired per question, same judge):
  1. selfquiz-r* vs base — does studying help at all?
  2. selfquiz-r* vs cheatsheet — does retrieval practice beat re-reading at
     matched artifact type? (THE claim.)
  3. E vs study-compute curve across r — the intelligence frame.
- Primary: DSPy (n=30, execution-grounded verification). OpenClaw: reported,
  underpowered, no-execution ablation.
- Expected signature if the hypothesis is right: gains concentrated at direct/k5
  (where the note substitutes for search), larger than the cheatsheet's because
  the note spends its budget on the model's actual errors; same-or-smaller
  giveback at k20f.

## 8. Failure modes, pre-registered

1. **Self-confirming quizzes** (model asks what it already knows) → mitigated by
   COVER's forced breadth + closed-book ATTEMPT exposing real gaps; measured by
   dev-exam error rate per round (healthy: 30-70%, logged).
2. **Verification poisoning** (wrong "correction" enters the note) → mitigated by
   execution grounding (DSPy) and mandatory file:line evidence per entry; audited
   by manual reads of every note snapshot (they're small).
3. **Note bloat/distraction** → soft cap + round-end rewrite; ablate cap if
   needed.
4. **Study-compute explosion** → per-round token budget, logged, hard stop.
5. **Quiz-quality collapse at scale** (later rounds produce junk questions on
   obscure modules) → dev-exam trend + manual reads; acceptable if curve
   plateaus honestly.
6. **The delta note helps closed-book but confuses tool use** → visible as
   direct-gain-without-k5-gain; would itself be a finding.

## 9. Open design choices (to red-team before implementing)

- ATTEMPT with note_{r-1} in context (targets note-insufficiency; my choice) vs
  bare closed-book (targets model-prior gaps only)?
- K modules × M questions per round: start K=4, M=5 (≈20 questions/round,
  ~25-40k study tokens/round)?
- Dev-exam self-verified scoring: is self-verification reliable enough to steer
  iteration (spot-audit its verdicts manually in round 1)?
- Note rewrite each round vs append-only with periodic compaction?
- Should VERIFY see the attempt (risk: anchoring on the model's wrong answer) or
  re-derive the answer independently then compare (costlier, cleaner)?

## 10. Prior art (positioning; to be verified/extended by the research sweep)

- Testing effect / retrieval practice: Roediger & Karpicke 2006; elaborative
  interrogation (Dunlosky et al. 2013 — cited by the paper itself).
- Cartridges self-study (Eyuboglu et al. 2025, cited in paper): self-generated
  QA about a corpus — but as training data for KV compression (the
  "internalize-the-corpus" objective the paper argues is misaimed). Ours is
  weight-free and error-driven.
- The paper's own SFT baseline: bigger-model synthetic QA + weight training →
  cramming, expertise DOWN. Ours differs on every axis they flagged: self-
  generated (no cheating), verified against the corpus, distilled into context
  (not weights), error-targeted (not coverage-targeted).
- Self-Instruct / instruction backtranslation: self-generated training data,
  different objective (instruction-following, weights).
- SEAL-style self-edits, Reflexion/self-refine: related self-improvement loops
  but tied to task feedback at inference time, not corpus study before any task.
- Vogel et al. 2026 (cited in paper): deterministic tree-sitter knowledge-graph
  notes — hand-engineered, not learned from the model's own errors.

Gap we occupy: retrieval-practice-driven, execution-verified, **error-delta**
note construction, evaluated on cost-weighted expertise with a
study-compute-scaling curve.
