# 005 — Self-quizzing: design (v1.1, 2026-07-06; v1 revised per the
# learning-science review lane — see "Review round 1" at the bottom)

**Intent.** A studying procedure π_quiz that converts corpus D + study compute into
a better agent, grounded in the human practice the user described: a student with
a textbook does the practice questions at each chapter's end — or writes their own.
Framing (corrected in review): with frozen weights, retrieval practice cannot
strengthen the model's memory, so the testing-effect literature is grounds for the
*diagnostic* function, not a promise of the outcome — testing exposes real gaps
that re-reading hides (Roediger & Karpicke 2006 Exp 2: re-study inflates judged
learning while testing reveals it), and errorful attempts followed by corrective
feedback are potent (Kornell, Hays & Bjork 2009; Richland et al. 2009; Rowland
2014 meta: feedback amplifies). Self-quizzing here is **formative assessment of
the model+note system**, and the quiz misses are the **learning signal for the
note**. Whether error-distilled notes beat read-and-summarize notes is a genuinely
novel empirical question the human literature does not settle — that is THE claim
under test. Success = E(π_quiz(Σ,D); D) > E(Σ; D) and > E(π_cheatsheet(Σ,D); D)
under the frozen harness (experiments/004 rules).

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

VERIFY  two-phase, independence enforced (review round 1's strongest fix —
        self-preference and confirmation bias make grading-your-own-attempt
        the maximal-risk configuration; Panickssery 2024, Zheng 2023
        reference-guided judging, Xie 2024, Huang 2024):
        Phase A (derive): a fresh ReAct episode that receives ONLY the
        question — not the attempt, NOT the note — and derives the answer
        open-book with file:line evidence; for DSPy it also executes checks
        in the pinned sandbox, with check scripts constructed from corpus
        evidence (prefer running the repo's own tests) rather than from the
        attempt. Each derivation is tagged with an evidence class:
        executed-repo-test > executed-self-written > file-quote-only.
        Phase B (diff): a short call diffs Phase A's answer against the
        attempt → verdict ∈ {correct, partial, wrong, unresolved}, plus the
        delta. "unresolved" (both plausible, disagreeing) is excluded from
        distillation. Cost note: this reorders work rather than adding it —
        the corrected answer was needed for DISTILL anyway.

DISTILL for wrong/partial items only: append delta-encoded entries
        ("prior belief: X. Actually: Y — dspy/clients/base_lm.py:96") with
        provenance fields (round, evidence anchor, evidence class, verdict).
        Correct items add NOTHING (note space is for the delta).
        MECHANICAL INTEGRITY GATE (model-free): every entry must carry a
        verbatim quoted snippet + file:line; the harness string-matches the
        quote against the actual corpus file and REJECTS the entry on
        mismatch (catches hallucinated anchors automatically; the rejection
        rate is a free verifier-quality metric, logged per round).
        Note maintenance is append-only, with a compaction pass ONLY when
        the ~4k-token soft cap is exceeded; after any compaction, re-run
        ATTEMPT on a sample of the affected entries' questions and revert
        the compaction if the re-test regresses. The rewrite pass may also
        maintain a compact structural-overview section (~500 tokens,
        synthesized from accumulated evidence anchors) so the error-only
        note does not lose the cheatsheet's "repo map" function at k5.

RETEST  (r ≥ 2) ~20% of each round's quiz slots re-test previously-distilled
        entries and previously-"correct" items (successive relearning;
        also the regression test for note edits). An item counts as "known"
        (and its entry compactable) only after repeated success; ATTEMPT
        "correct" verdicts that gate a skip require agreement across 2
        samples.
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
  for distillation, never seen with the note in the ATTEMPT step. The pool
  ACCUMULATES across rounds (a 4-item exam would be all noise). Procedure
  iteration (prompt tweaks, K/M choices, cap tuning) is driven ONLY by dev-exam
  closed-book accuracy (self-verified) + note-quality reads. StudyBench's 50
  questions are touched only at pre-registered milestones, each logged in
  experiments/ regardless of outcome.
- **Dev-verdict audit protocol** (a 9B self-judge is not trustworthy by default —
  JudgeBench: 7-8B judges near chance on hard pairs): (a) round-1 human spot-audit
  of ~30 dev verdicts, requiring ≥80% agreement before the metric may drive any
  decision; re-audit after any VERIFY prompt change; (b) dev verdicts use 2-3
  independent Phase-A derivations with majority vote; (c) no variant is accepted
  whose dev delta is within the audited judge-noise band; (d) OpenClaw dev scores
  are lower-trust (no execution grounding) — audited disproportionately.
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
- Pre-registered effect ordering: the selfquiz-vs-cheatsheet delta should be
  LARGER on DSPy (stale-prior correction + execution grounding) than on OpenClaw
  (no priors to correct, no execution). If OpenClaw shows the larger delta, that
  falsifies the error-delta mechanism story and gets reported as such.
- Run-health diagnostic (watched every round, CLAUDE.md): for dev-exam items,
  also run a bare closed-book attempt (no note); the with-note minus bare gap is
  the note's marginal closed-book lift. Healthy dev error rate 30-70%; if it
  falls below 30%, escalate question difficulty (pitfall/behavior types on
  less-covered symbols) before concluding the corpus is exhausted.

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

## 9. Design choices — resolved in review round 1

- ATTEMPT with note_{r-1}: KEEP (response congruency is among the strongest
  transfer moderators, Pan & Rickard 2018 meta; matches the eval-time direct
  condition; bare attempts would re-flag gaps the note already fixed). Bare
  attempts retained only as the per-round diagnostic (§7).
- K=4 × M=5 per round: adopted as the starting point; question quality/difficulty
  control dominates the choice of K/M (see run-health rules, §7).
- Dev-exam self-scoring: usable ONLY under the audit protocol (§4).
- Note maintenance: append-only + cap-triggered compaction with regression
  re-test (not per-round rewrite).
- VERIFY: independent re-derivation then diff (two-phase, §2). Never grades the
  attempt directly; never sees the note.
- COVER ordering: size × import-degree weighting (pure size would starve small
  load-bearing modules like config/entry points); every top-level package
  touched by round 4 so the r=4 milestone reflects full-corpus breadth.

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

- (Added in review) Dynamic Cheatsheet and Active Reading: closest recent LLM
  work on maintained notes / self-generated study — to be differentiated
  precisely once the prior-art lane completes (pending, rate-limited).

## Operational spec (v1.2 clarifications, 2026-07-06 — from design review with Omar)

**Axiom 0 — corpus-agnostic.** The procedure's inputs are exactly: a repository,
the three tools, and a sandbox when the language runs. Nothing benchmark-derived
anywhere. DSPy/OpenClaw are testbeds, not targets; the mechanism must run on any
codebase via one command.

**Syllabus (COVER), corrected role.** Chapters = top-level modules, ordered once
by lines of code (descending; import-degree weighting is an ablation, not a
foundation — it requires per-language parsing and fights Axiom 0). The ordering
does NOT mean earlier chapters are studied *better* — every chapter gets identical
treatment when reached. Ordering matters solely under **budget truncation**: at a
round-r snapshot, only the first 4r chapters exist in the note, so the prefix
should be the most load-bearing one. (Incidental second-order effect, not a
mechanism: earlier chapters accrue more RETEST passes over the program's life.)

**QUIZ, verbatim shape.** One agent episode per chapter. Input: module path +
file list + instruction: "You are studying <module> to become an expert on this
repository. Explore it with your tools. Write M quiz questions that test whether
someone who has not just read this code could use it correctly — usage, behavior,
location, or pitfall questions. Each must be answerable from the repository alone,
must not contain its own answer, and must cite the files that motivated it."
Schema-forced JSON out: {question, type, anchors[], writer_sketch}. The sketch is
audit metadata, never ground truth. Gates (code, not model): anchor files must
exist; near-duplicate questions vs all prior rounds dropped; 1 of M held out to
the dev exam. The type menu is an anti-trivia heuristic, ablatable.

**VERIFY, exact I/O.**
- Phase A (derive): fresh agent episode; input = question ONLY (never the
  attempt, never the note). Tools + run_python (runnable corpora). Output
  (schema): derived_answer; evidence [{file, line, quote}]; probe + captured
  output when the question concerns executable behavior. Evidence class tag:
  executed-repo-test > executed-self-written > quote-only.
- Phase B (adjudicate): toolless call; input = question + Phase A output +
  attempt; output = verdict {correct, partial, wrong, unresolved} + delta.
  Compares claims, not wording. unresolved => dropped, never distilled.
- Firewall rationale: never ask "is this right?"; ask "what is right?" then
  compare mechanically (self-preference/anchoring/confirmation-bias evidence,
  review round 1).
- Ensembling: 2-3 independent Phase-A runs with majority agreement for all
  dev-exam verdicts and for OpenClaw distillation (no execution grounding there).

**DISTILL, exact record.** Per wrong/partial item, one call: input = question +
attempt + Phase A output; output (schema) = {belief, correction, quote, file,
line}. Model-free integrity gate: the harness string-matches `quote` at
`file:line` (±2 lines tolerance) and rejects on mismatch; rejection rate logged
as a verifier-quality metric. Entries append to a JSONL sidecar (with round,
verdict, evidence class); the eval-time note is its markdown rendering (grouped
by chapter, ~4k-token soft cap, plus the ~500-token repo-map section). Correct
items write nothing — the note stores only the verified diff between the model's
priors and the repo.

**Harness instantiation.** Everything runs on the faithful react harness
(experiments/006): ATTEMPT = dspy.Predict closed-book; QUIZ and VERIFY Phase A =
react episodes; eval = the standard react grid with the note prepended
(identical mechanics to the react-cheatsheet row, so the studying algorithm is
the only variable vs that baseline).

## Review round 1 (2026-07-06): learning-science lane

14 findings folded into this v1.1 (frozen-model reframing; two-phase independent
VERIFY; note excluded from VERIFY; mechanical quote-check gate; dev-verdict audit
protocol + accumulating dev pool; RETEST slots; append-only note + compaction
regression test; structural-overview section; evidence-class trust tiers;
DSPy>OpenClaw pre-registered ordering; COVER weighting; bare-attempt diagnostic;
health bands). Key claim-honesty fix: the testing effect's causal mechanism
(retrieval strengthening memory) does not exist in a frozen model — the design is
formative assessment of the model+note system, and whether error-distilled notes
beat read-and-summarize notes is exactly the novel question under test.
Pending: prior-art lane + 3 red-team lanes (leakage, rigor, mechanism) —
rate-limited, to be resumed.
