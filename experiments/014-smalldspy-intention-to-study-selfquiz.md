# 014 — SmallDSPy intention-to-study semantic self-quizzing

## Objective

Measure the exploratory SmallDSPy performance of a four-round semantic
self-quizzing procedure without conditioning evaluation on every stochastic
reference derivation succeeding. The procedure must never turn an unresolved
question into a correction, but an honest abstention must not erase the entire
study treatment.

This is a prospectively identified `semantic-selfquiz-v4` development screen.
It follows the terminal v3 construction failure documented in experiment 013
and is neither a rerun nor a continuation of that immutable tree. It is not a
confirmatory or publication-ready experiment.

## Motivation from the frozen v3 failure

V3 stopped after R1 because five of twelve train items and two of three dev
references were unresolved. All 319 provider calls were complete and audited;
the failures occurred inside reference construction. Across the seven
unresolved items, fewer than two derivations survived exact citation and
semantic-support checks, so reciprocal consensus never began.

The audit found two distinct causes:

1. Qwen often placed several physical source lines in one `Evidence.quote`.
   The exact-one-line verifier correctly rejected those citations, but this
   serialization mismatch discarded otherwise useful answer derivations.
2. Some generated questions contained ambiguous or false presuppositions.
   Those questions should remain unresolved and must never contribute a note
   correction.

Requiring every question to resolve before any study treatment exists creates
an additional selection problem: only stochastic constructions with no
abstentions can reach evaluation. V4 instead treats unresolved verification as
an observed study-procedure outcome.

## Prospective v4 contract

V4 preserves the v3 corpus, prompts except where stated below, Qwen model and
sampling, master seed, two independent reference derivations, corpus tools,
attempt protocol, reciprocal reference consensus, attempt adjudication,
correction support, note rendering, freshness rules, and four-round schedule.
The fresh study identity changes realized per-call seeds because identity is
part of the stochastic namespace.

Two changes are frozen before launch:

1. **Uniform answer-frozen citation pass.** After every successful independent
   answer derivation, exactly one additional ReAct pass receives only that
   derivation's question and frozen answer, plus repository tools. It may return
   evidence but may not change the answer. Every evidence object must quote one
   exact nonempty physical source line. The pass is run uniformly rather than
   being triggered by an observed support verdict. It cannot see the learner's
   attempt, note, writer sketch, sibling derivation, or later support judgment.
   Exact quote validation, semantic support, and reciprocal agreement still
   run afterward. All initial and citation-pass outputs and calls are retained.
2. **Intention-to-study abstention.** An unresolved train item admits no
   correction; an unresolved dev reference produces no paired dev verdict. Both
   remain immutable, counted, and visible, but are valid terminal treatment
   states. They do not prevent the next round. No question is regenerated or
   replaced, and no study namespace or seed is retried until it passes.

Automated treatment readiness is separate from automated claim readiness.
Treatment readiness requires exact expected artifacts in valid terminal states,
provenance, freshness, tool trajectories, usage, adapter audit, model identity,
lineage, and safety of every admitted evidence/correction. Claim readiness
retains the stricter requirement that train and dev records resolve. Thus an R4
note may be eligible for this diagnostic evaluation while remaining ineligible
for human-audit promotion or a confirmatory claim.

An abstention is valid only when every attempted provider phase completed
without an execution error. Derivation, citation, semantic-support, reciprocal
consensus, attempt-adjudication, distillation, correction-support, and paired
dev-adjudication errors fail treatment readiness even if a parent record would
otherwise summarize them as `unresolved`. Evidence-invalid answers and
error-free semantic disagreement may abstain. Adapter audit groups are checked
order-independently as either one accepted/error primary call or exactly one
rejected primary plus one accepted/error strict repair; no other call multiset
is accepted.

## Frozen identities and compute

| Object | Value |
|---|---|
| v4 construction | `smalldspy-semantic-react-r4-20260714c` |
| task | `smalldspy` |
| study seed | `43001` |
| attempt access | `react-corpus` |
| local model | `Qwen/Qwen3.5-9B` |
| model revision | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| treatment evaluation | `smalldspy-local-selfquiz-20260714c` |
| evaluation seed group | `smalldspy-local-cheatsheet-screen-20260713` |
| evaluation seed | `44001` |
| GPT sensitivity screen | `smalldspy-selfquiz-gpt54-whole-high-20260714c` |

The construction will use three authenticated TP=2 replicas over six L40S
GPUs. The seventh allocated GPU remains outside that homogeneous topology.
Every non-smoke launch must begin from a clean pushed source commit and a fresh
namespace.

## Conditional evaluation plan

Only an R4 artifact that passes every automated treatment-readiness gate may be
evaluated. Its exact note bytes and construction manifest will be bound to the
evaluation. The frozen SmallDSPy screen uses the same five questions,
`direct,k5,k20,k20f`, three rollouts, and paired seed group as experiment 012,
for 60 intention-to-treat cells. The same GPT-5.4 whole-evidence judging
protocol will then produce a diagnostic WAUC.

Baseline WAUC `5.220664728639385` and cheatsheet WAUC
`18.06684522893066` are the descriptive comparators. The five evaluation
questions have already been adaptively reused, v4 was designed after observing
v3, study compute and note size are unmatched, and no blinded human study audit
is planned. Any resulting number is therefore an exploratory implementation
screen, not a robust causal finding about self-quizzing.

## Status

Prospectively specified and implemented. Before launch, 70 focused integrity
tests passed, the pinned citation JSON schema compiled, and both runtime and
pure archive adapter-audit validators accepted the retained 319-call v3 ledger
after the order-independent audit fix. Execution is pending.
