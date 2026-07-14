# 010 — Audit 009 disposition and self-quizzing review (2026-07-13)

This is an append-only response to
[009 — Independent full-tree audit](009-independent-full-tree-audit.md). It
does not rewrite that audit or the earlier chronological logs. Its purposes are
to (1) disposition every substantive part of 009, (2) separate reproduced
measurements from interpretations, (3) record the additional future-run
safeguards added after 009, and (4) reassess the first self-quizzing method from
first principles.

No model, API, GPU, Slurm, checker, container, or benchmark run was performed
for this response. Historical quantities below were recomputed previously from
the checked-out artifacts; the present hardening was tested only with offline
unit, syntax, compilation, and dataset checks. A passing offline suite is
evidence about the contracts it exercises, not proof that a live provider,
cluster, checker artifact, or scientific hypothesis works.

## 1. Executive correction

Audit 009 is useful and mostly accurate about artifact counts and point
estimates, but its bottom-line confidence is too high. It conflates five things
that must remain separate:

1. a chronological log being candid;
2. a point estimate being reproducible from the current mutable artifact tree;
3. a confidence interval failing to exclude zero;
4. equivalence or parity; and
5. a causal or procedure-level conclusion.

Items 1–3 can be descriptive facts when tied to exact artifacts and an exact
analysis. Here the first two are substantially supported and most reported
intervals do contain zero, but one exact Audit 009 interval does not reproduce
under the documented bootstrap (§4.2). Items 4–5 are unsupported inferences. A
95% interval containing zero does **not** establish a tie, parity, the null, no
effect, or equivalence. It says the available estimate does not distinguish an
effect from zero at that procedure's resolution. The compatible effect range
includes both benefit and harm. Equivalence requires a pre-specified
practically meaningful margin and an equivalence/non-inferiority analysis
designed around it.

The defensible historical conclusion is therefore:

> Under the historical harness, judge, public question set, and known artifact
> defects, the executed pre-registered selfquiz milestones R1/R2/R4 did not meet
> their superiority criterion. The planned R8 milestone was not run after the
> earlier outcomes were inspected, so the complete registered sequence was not
> executed. Later static-note arms were adaptive and likewise supplied no
> robust superiority evidence. This is a failure to demonstrate the proposed
> advantage, not evidence that the methods are equivalent or that
> self-quizzing cannot work.

Integrity defects do not have a one-way effect. Leakage, stale grades,
incomplete populations, mutable notes, weak verification, non-fresh curricula,
and protocol deviations can create false positives **or false negatives**.
Consequently, 009's statement that the defects “only preclude positive claims”
is incorrect.

## 2. Scope and reproducibility of Audit 009 itself

| Audit 009 statement | Disposition |
|---|---|
| A separate session performed the audit. | A separate session is useful adversarial review, but it is internal review, not external independent replication. No reviewer identity, frozen audit environment, transcript, or executable audit package establishes stronger independence. |
| Every non-corpus/non-raw-run file was read. | This is an auditor declaration. The repository contains no machine-verifiable read log, so later readers cannot independently establish completeness. |
| Historical numbers were re-derived with independent scripts. | The reported point estimates are mostly reproducible, but those scripts, inputs, environment, and outputs were not preserved. The exact recomputation path cannot be rerun from a fresh clone. |
| The audit made no edits other than its report. | Plausible for that session, but another session changed the same uncommitted tree concurrently. The audit candidly discloses this. |
| The final audited state was quiescent. | Quiescence was observed, not content-addressed. The state was a large uncommitted diff on HEAD `48328be`; there is no commit identifying the exact bytes audited. Later commit `f59b681` is the preserved hardening baseline, but it is not identical merely by assertion. |
| Tests, compiles, shell parsing, and dataset validation passed. | Accepted as a historical execution report. No retained transcript or CI artifact proves it independently. Later offline reruns provide fresh evidence for the current tree only. |
| No workflow, GPU, API, or model call was made. | Consistent with the report and repository changes, but absence of external execution is not provable from Git alone. |
| Files changed concurrently were reread and every noted concern was closed. | File rereading is an auditor declaration, and the completeness claim is refuted by later-discovered concrete gaps. |
| Four follow-up gaps were fixed. | Internally inconsistent: §3 enumerates seven contemporaneous fixes and never identifies a four-item subset. The fixes themselves are dispositioned in §5.1. |

Raw `runs/` and `grades/` artifacts are ignored, locally mutable, and not
archived as a content-addressed release. Historical numbers that depend on them
are therefore reproducible from this workspace snapshot, not reproducible from
the Git repository alone. A publication package still needs immutable raw
artifacts plus the exact recomputation program and environment.

## 3. Disposition of §1, “Bottom line”

| Item | Disposition |
|---|---|
| 1. “The existing experimental record is honest ... Nothing is overclaimed.” | **Qualified/rejected in part.** The logs candidly preserve failures and later corrections, which is a real strength. However, 007 calls several results “robust findings,” says breadth and precision “compose without interference,” calls fresh OpenClaw parity “reproduced,” and gives a “procedure-level conclusion.” Those statements exceed what noisy, adaptively reused, defect-bearing data show. Honest intent does not guarantee valid inference. |
| 2. The 008 defect register reproduced. | **Supported for the enumerated checks.** The listed stale-grade, overlap, leakage, token-accounting, and overwritten-note counts reproduce. This does not prove every statement in 008, nor does it repair the affected evidence. |
| 3. The hardening code was correct and no blocker remained. | **Too strong.** Passing tests supported only the exercised contracts, and the state was moving and uncommitted. Later review found defects in the then-current implementation: fabricated selfquiz usage totals, validation not enforced at the persistence boundary, and inadequate TypeScript checker semantics/bundle closure. Local grading/runtime identity, the bounded live-smoke contract, and paired local screening are subsequent capabilities chosen for the new development policy; their earlier absence was not evidence that the old confirmatory-only architecture was internally incorrect. All current resolutions remain subject to the limits below. |
| 4. Casual benchmark numbers were deliberately impossible. | **Mechanically true at that state, scientifically incomplete.** Fail-closed confirmation protects claims, but a total absence of diagnostic grading makes method development impractical and pushes iteration onto a same-model internal dev signal. A separate local-Qwen diagnostic path has now been added. It remains permanently non-confirmatory. |

The correct readiness statement is split by purpose:

| Purpose | Current status |
|---|---|
| Offline method/code development | Ready after the current diff is reviewed and committed. |
| Local diagnostic screening on the public StudyBench questions | Code path implemented; requires a pinned local Qwen runtime and a real live smoke on a compute allocation. Its artifacts explicitly set all claim-readiness fields false. Repeated use makes the public set an adaptive development benchmark. |
| Strict checker-based local metrics | Requires the pinned Python SIF/Apptainer pair for DSPy and a complete, audited TypeScript compiler bundle for OpenClaw. Without the checker, only lenient scores are interpretable. |
| External confirmatory evaluation | Not ready. It requires a genuinely held-out evaluation population, a frozen design with adequate power, independent process checks where claimed, external grader credentials/runtime, and a fresh preregistered run. |
| Publication-level procedure claim | Not ready. Historical artifacts cannot be promoted retroactively. |

## 4. Disposition of §2, historical verification

### 4.1 Formula and population statements

The expertise implementation and the basic population/status counts checked in
009 are consistent with the current artifacts: React has 594 `ok` + 6
`no_answer` episodes and 600 grades; React-cheatsheet has 1,185 `ok` + 15
`no_answer` episodes and 1,200 grades; hybrid2 has 385/720 DSPy and 480/480
OpenClaw grades; hybrid3 has 1,182 `ok`, 9 `no_answer`, 9 `error`, and no grades;
and no React-family episode marked `ok` has an empty answer. This establishes
arithmetic on the available mutable files, not the validity or fresh-clone
reproducibility of the population that those files represent. In particular,
stale/missing grades, retries, unequal failures, mutable artifacts, adaptive
arm selection, and public-set reuse affect the scientific estimand even when
the arithmetic is exact.

### 4.2 Every headline row

| Historical row | Recomputed current-artifact result | Correct interpretation |
|---|---:|---|
| React base | DSPy 12.3104; OpenClaw 8.4545 | Point estimates reproduce. This does not make them byte-exact paper replications. |
| Selfquiz r1 | 13.6336; 10.1737 | Point estimates reproduce. |
| Selfquiz r2 | 11.2038; 10.6439 | Point estimates reproduce. |
| Selfquiz r4 | 11.7566; 9.3584 | Point estimates reproduce. |
| Native GPT base | 26.4025; 20.1993 | Point estimates reproduce for the local harness/artifacts. |
| Native Fugu base | 27.9910; 18.1582 | Point estimates reproduce for the local harness/artifacts. |
| Native cheatsheet Fugu | 28.4895; 17.9624 | These are the present-artifact values. Their difference from the earlier log is consistent with later retries/stale-grade repair, but the absent recomputation history prevents treating that causal explanation as independently reproduced. |
| Select cells, usage k20f, and hybrid direct | 4.4/19.8/22.3/28.7; 34.3; and 9.2, at displayed precision | Descriptive maxima/cells reproduce. “Best ever” is selection over adaptively inspected arms, not multiplicity-adjusted evidence. |
| Cheatsheet − base, three rollouts | +2.8733; +2.1362 | Descriptive paired gains reproduce. Intervals containing zero do not establish benefit, no benefit, or parity. |
| Hybrid − cheatsheet, six versus six | −0.7051; −0.7349 | The available data did not distinguish superiority. Calling this “parity” or “dead even” is unsupported without an equivalence margin/test. |
| Hybrid2 − cheatsheet, OpenClaw | +0.09953 | The point estimate reproduces approximately. Audit 009's interval `[-3.17, +3.25]` did **not** reproduce under the documented shared question/rollout bootstrap: seed 0 with 10,000 resamples gives approximately `[-2.664, +2.793]`. Both intervals contain zero, so the qualitative statement is only “inconclusive at this precision.” |

The fact that the hybrid2 interval changes with the implementation/seed is not
itself alarming for a Monte Carlo bootstrap, but it demonstrates why the exact
analysis program, seed, sampling unit, and artifact hashes must be retained.

### 4.3 Every defect-register spot check

| Audit check | Disposition |
|---|---|
| 39 stale grades: 3 + 2 + 4 + 8 + 7 + 5 + 10 | Reproduced from the present historical artifacts. Staleness can change estimates in either direction. |
| Hybrid3 DSPy repeats 86/86 normalized records; OpenClaw 0/92 | Reproduced for the records and normalization used by the audit. It establishes non-freshness under that rule. It does not prove vLLM's default seed was the sole root cause; deterministic sequential generation is a plausible mechanism, not conclusively isolated. |
| Dev-origin retests and 7/46, 4/57, 6/45, 5/50 retest-derived entry counts | Reproduced. This is train/dev contamination within the study construction. It invalidates the claimed dev isolation and weakens causal interpretation; it is distinct from direct StudyBench-question leakage. |
| Corrected cumulative token sums 920,709 / 1,832,447 and 935,111 / 1,747,223 | Reproduced from available round summaries. Historical accounting remains non-uniform, and absent provider usage must never be silently represented as zero. |
| Root cheatsheets overwritten; current 39,467 / 21,044 versus historical 64,363 / 25,183 | Reproduced in the current workspace/reachable history. This demonstrates why notes must be copied and hashed before use. |
| Dataset hashes/counts/excerpts | Strict validation supports the checked-in 30 + 20 records against the two locally available pinned corpus commits. This proves the validator's byte relation for this checkout, not the benchmark's external validity or exhaustiveness. |
| “`.env` was never committed” | Too absolute. No tracked `.env` was found in the reachable history examined. That does not establish absence from unreachable objects, other remotes/clones, logs, process environments, or past credential use. Deleting a key from `.env` does not revoke it; rotate credentials that may have been exposed. |

### 4.4 Assessment language

The experiment logs are unusually candid, but “faithful record” should mean a
chronological record with disclosed corrections, not a clean inferential
dataset. The following replacements apply wherever 007–009 use stronger
language:

| Earlier wording | Defensible wording |
|---|---|
| “DSPy > OpenClaw was falsified” | The pre-registered descriptive ordering failed and reversed in these artifacts. The proposed mechanism was not isolated, so its general falsification was not established. |
| “Hybrid ties/is at parity with cheatsheet” | No superiority was distinguished at the available precision; material benefit and harm remained compatible with the interval. |
| “Parity reproduced” | A second OpenClaw point estimate was near zero, but the construction was not a clean independent replication and equivalence was not tested. |
| “Corrections reliably add value with tools” | Several adaptively inspected cells were higher than base; robustness after multiplicity, artifact defects, and a fresh held-out replication was not established. |
| “Breadth and precision compose without interference” | The observed cell pattern was compatible with composition, but an interaction/no-interference hypothesis was not directly estimated with a factorial design. |
| “Optimal note size emerged” | In these adaptive arms, larger notes correlated with worse low-token-budget scores; note size was confounded with content, coverage, and construction stage. |
| “Negative/parity finding” | The superiority criterion was not met at the executed R1/R2/R4 milestones; planned R8 was not run, and effects remained unresolved within the reported interval and historical validity limits. |

## 5. Disposition of §3, code verification and architecture

The future-run architecture has strong fail-closed properties, but several
descriptions in 009 are proofs of byte/configuration consistency, not proofs of
scientific validity:

| Architecture claim | Disposition and boundary |
|---|---|
| Pinned corpora verified blob-by-blob | Strong within the locally available pinned Git trees and allowed file types. It does not attest an authoritative upstream release or semantic completeness. |
| Suffix-allowlisted tools | Strong containment for exposed repository files. It is a protocol choice, not proof that the model sees all evidence needed by a question. |
| “Immutable” manifests/artifacts | The code uses create-if-absent, hashes, and dependency revalidation. The underlying user-writable filesystem is not physically immutable; integrity is detected/rejected under the stated non-adversarial/same-user limits. |
| Source/env/note/episode binding | Strong accidental-drift detection. Broad source freezing also couples non-computational docs to execution and can be operationally costly (§6.3). |
| Provider ledgers | Stronger after missing usage became explicit `null`/incomplete. A provider can still omit or misreport usage/identity; the repository must disclose rather than infer it. |
| Grading revalidation | Strong for the contract and checker configuration it can rerun. Judge correctness, reference truth, and provider behavior are not proved by schema validity. |
| Strict reports/comparisons rebuild inputs | Strong arithmetic/provenance defense. A content-addressed report can still estimate the wrong scientific target if the population/design is biased. |
| Exact quote and anchor checks | Prove that cited bytes/path/line exist. They do not prove that a correction is semantically entailed, complete, representative, or non-misleading. |
| Same-model derivation/adjudication | Structurally separated prompts reduce direct anchoring, but references and adjudication still come from the same model family/runtime. This is automated diagnostic evidence, not independent ground truth. |
| Deterministic seeds | Make requested sampling contracts reproducible where the runtime honors them. They do not guarantee bitwise-deterministic vLLM/CUDA execution under concurrency/hardware/runtime changes. |
| Curriculum freshness gate | Detects exact/normalized/implemented lexical overlap among discoverable accepted curricula. It does not prove semantic novelty, independence from external corpora, or novelty relative to an artifact absent from local storage. |
| Human-audit validation | Proves declared schemas, populations, hashes, and decisions. Software cannot prove that a person was actually independent, blinded, diligent, or free of communication. |
| Paired bootstrap | Correctly shares sampled question and rollout indices for the implemented finite benchmark analysis. Its interval captures only those resampling dimensions. It omits judge/systematic error, same-model bias, topic/dependency clustering not represented in the resampling unit, design adaptivity, and public-set reuse. |

### 5.1 Audit 009's seven contemporaneous fixes

1. React missing usage no longer becomes zero. The same principle is now
   applied to selfquiz summaries: incomplete provider usage is represented as
   incomplete, with known subtotals separate from unknown totals.
2. Final episode validation now occurs at the shared persistence boundary, not
   only in the current producers. A future producer cannot write a final
   episode through that primitive without supplying a validator.
3. The vLLM API key is not put on a command line. It remains necessarily
   available to the serving process and recorded only where the authenticated
   topology contract requires; this reduces accidental exposure but is not a
   general same-user secret boundary.
4. Hidden corpus index flags are rejected, and exact blob reads still provide
   the substantive corpus check.
5. Hidden source index state and the tests are included in source validation.
6. Human-audit records use a closed schema. This prevents unsupported clauses
   from looking enforced; it does not automate the human process.
7. Model-cache hashing/launch checks close common path/symlink/race failures
   and bracket loading. As 009 says, a same-user adversarial
   write-and-restore remains outside the guarantee without a read-only
   content-addressed mount.

### 5.2 Missed defects and subsequent capabilities after Audit 009

These should not be conflated. A missed defect weakens 009's claim that no
correctness/integrity gap remained. A subsequently chosen capability answers a
new development-policy need; its earlier absence does not make the old
confirmatory-only architecture incorrect.

| Category | Gap or need | Future-run resolution | Remaining limit |
|---|---|---|---|
| Missed defect | Selfquiz usage summaries invented zeros for absent usage | Missing values are now explicit; completeness and known subtotals are recorded. | Historical totals are not retroactively repaired. |
| Missed defect | Producer validation was not enforced by the storage primitive | Final-status persistence now requires a caller-supplied full validator. | Correctness still depends on the validator's coverage. |
| New capability | No affordable exploratory benchmark grader | Added a separate local Qwen/vLLM diagnostic grade/report path. | It is permanently `claim_ready: false`, uses the same pinned model family as generation, and can overfit the public set. |
| Capability-integrity requirement | Local server transport versus identity was ambiguous | Loopback authentication, stable endpoint identity, actual transport disclosure, runtime attestation, and substantive resume checks are bound into artifacts. Port/job transport nuisance may vary; substantive runtime/model drift may not. | A live smoke is still required on a compute node. |
| New capability | A one-record local smoke needed a bounded contract | Local smoke now requires a purpose-`smoke` generation artifact and exactly one answered cell, stored under the smoke namespaces. | It validates plumbing, not full-grid reporting or scientific performance. |
| Missed defect | A TypeScript “compiler hash” could point to an exit-zero wrapper or incomplete files | A canonical content-pinned bundle plus semantic calibration must accept a valid program/relative TSX import and reject a real type error. | Bundle completeness and non-adversarial suitability remain a human/auditor responsibility; no real bundle ships with the repository. |
| New capability | No valid local paired estimator | A separate nonclaim screening comparison revalidates complete local reports/populations, pairs exact question/rollout cells, and uses shared resampling. | It remains diagnostic; interval inclusion of zero is explicitly inconclusive. |
| Capability-integrity defect | Standalone post-hoc screening recreated a live local-judge runtime | The screen now reuses the exact content-bound local-server attestation stored in each report, while freshly reattesting the current grading Python/package bytes, source, checker, grade specifications, and full population. Direct tests prove the live local-runtime collector is not called and stale/tampered cross-bindings fail closed. | A stored server/GPU attestation is not a fresh observation that the old server still exists; this path validates completed artifacts and intentionally does not claim live-runtime availability. |
| Comparison-integrity defect | Arm-level generation fingerprint sets could hide per-cell swaps | Reports now persist exact generation response-model, available-fingerprint, provider-call, missing-fingerprint, and harness identities for every final manifest episode. Strict and local comparisons require response models and available fingerprints to match within each paired question/rollout cell; adversarial tests reject swapped cells even when arm-level sets are identical. They also reject any arm containing more than one distinct available generation fingerprint, because matched `{A,B}` sets can still hide different revision multiplicities or order. | Provider-call counts, missing fingerprints, variable turn counts, and no-answer outcomes are disclosed per cell but are not forced equal. A missing generation fingerprint is nonfatal because the exact model revision, cache, launcher runtime, and environment remain independently bound; a second distinct available fingerprint is fatal. |
| Presentation defect | Local strict zeros could be mistaken for model failures when checkers are unavailable | Reports bind checker readiness and label interpretation as all-metrics or lenient-only. | Strict metrics are unavailable until real checker artifacts are configured. |
| Missed defect | Contradictory claim-readiness booleans could coexist | Grade validation rejects a positive subordinate readiness flag when the top-level claim flag is false. | Schema consistency is not scientific validity. |
| Documentation defect | Preregistration docs described an ancestor more loosely than code | Documentation now states the exact single-parent direct-child/add-only rule. | The rule is intentionally conservative and stricter than scientifically necessary. |

### 5.3 Documentation nits

- The stray `minimalist hi` token and spelling error in the root instructions
  are fixed identically in `CLAUDE.md` and `AGENTS.md`; the files remain
  byte-for-byte identical.
- The repository map now says that run/report/comparison/preregistration
  directories are created on first use.
- Experiment 007 remains unchanged as a historical log. Corrections live in
  008 and this document rather than silently rewriting history.

## 6. Disposition of §4, implications for the next method

### 6.1 Exploratory measurement

The confirmatory-only grader described by 009 is no longer the only path. The
new local diagnostic lane is intentionally separate:

- local Qwen/vLLM grades and reports cannot become confirmatory;
- complete full-grid reports are required for method ranking;
- a one-cell purpose-smoke checks only plumbing and is not reportable;
- local paired screening compares matched arms with a shared bootstrap;
- same-model/self-preference and public-set adaptivity are mandatory caveats;
- when a real checker is unavailable, only lenient scores may be interpreted;
- no local/external “agreement” is inferred from different populations.

This makes iteration affordable, but repeated local scoring converts the
public 50 questions into a development benchmark. A later publication claim
needs a genuinely held-out population. Assigning a fresh run ID or
preregistration to the same adaptively inspected questions does not make them
held out.

Internal selfquiz dev exams are useful construction diagnostics, not external
evidence: they are generated, answered, referenced, and adjudicated by the same
model family and target a different question distribution.

Calling the old confirmatory-only path the “strongest possible
anti-overfitting stance” was also too strong and subjective. A genuinely hidden
evaluation set is a stronger defense, while exclusive reliance on a generated
same-model dev proxy introduces a different overfitting/mismeasurement risk.

### 6.2 External prerequisites are stage-specific

Audit 009 bundled all prerequisites together. The accurate split is:

- local lenient screening: pinned Qwen model/revision, attested vLLM runtime,
  authenticated loopback server, and a live compute-node smoke;
- local strict DSPy metrics: additionally a pinned Apptainer executable and
  audited Python SIF;
- local strict OpenClaw metrics: additionally a complete pinned TypeScript
  checker bundle that passes semantic calibration;
- external confirmation: the relevant external judge credential/endpoint and
  all confirmation protocol requirements.

No API credential is needed for offline development or local grading.
Audit 009 did not inspect `.env` contents, so it could not establish that no
credential existed; it could only identify the required external dependency.

### 6.3 Frozen-source chain

The broad freeze is mechanically honest: changing experiment documents, tests,
or analysis code between stages invalidates the chain. It prevents convenient
after-the-fact code changes from inheriting an old provenance claim. It is also
more conservative than necessary: prose that does not affect execution is
treated like executable analysis, and “finish the whole chain before writing
up” conflicts with continuous experiment logging.

The implementation is retained for now because changing provenance scope is a
separate protocol decision. A future clean design could freeze stage-specific
executable/configuration inputs while binding append-only observations as
separate artifacts. No such chain-bound observation channel exists today.
Until one is implemented and tested, complete the computational chain before
editing frozen scoped files. Any contemporaneous note recorded elsewhere is
provisional and unbound; it must not inherit the chain's provenance claim.

### 6.4 Direct-child preregistration rule

The code intentionally requires the execution commit to be the single-parent
direct child of the implementation source commit and to add only the exact
preregistration files. This is stronger than is statistically necessary, but
it creates a simple auditable chronology. Documentation now matches it. Do not
weaken or work around it mid-study. Git ancestry proves repository ordering,
not that researchers had no earlier external knowledge, private experiments,
or exposure to outcomes; real preregistration is also a process commitment.

### 6.5 Legacy hybrid2/hybrid3

- Hybrid2 DSPy is incomplete and cannot be made into a clean confirmation by
  filling missing legacy grades. Old code could technically be reconstructed
  to produce more scores, so “can never be completed” is too absolute; doing so
  would not repair provenance. At most, the existing subset remains a
  historical diagnostic.
- Hybrid3 DSPy used a non-fresh curriculum under the recorded freshness rule
  and must not be described as a fresh confirmation.
- Neither arm should be retrofitted into the new manifest/claim system.

### 6.6 Selfquiz brittleness

Requiring exactly `M` admitted questions per chapter with a deterministic seed
turns a recoverable generation failure into a permanently failed study ID.
That is fail-closed, but it is not automatically a good research design.
Future methods should preregister bounded repair attempts and their seeds,
retain every rejected candidate/reason, and distinguish:

- **within-comparison control:** use the same frozen curriculum across arms
  when the causal contrast is the learning/update rule; and
- **procedure replication:** generate a new curriculum under a frozen policy
  to test generalization beyond one draw.

Progressively banning every stored curriculum may also exhaust a small corpus
without improving scientific validity. Freshness policy should define the
population it intends to generalize over and test semantic as well as lexical
overlap where feasible.

## 7. Disposition of §5, pre-flight checklist

| Audit item | Current disposition |
|---|---|
| 1. Commit the hardening pass | The earlier baseline was committed as `f59b681`. The present local-grader/audit-response diff must receive the same review and commit discipline before any non-smoke run. This document does not authorize a run, commit, or push. |
| 2. Choose dev-only versus diagnostic grading | Resolved in favor of a permanently nonclaim local-Qwen diagnostic lane plus later external confirmation. The local lane's public-set use must be logged as adaptive development. |
| 3. Build checkers and provision judge key | Split by stage as in §6.2. Checkers are required only for strict metrics; external keys are required only for external grading. No checker bundle/image is currently supplied. |
| 4. Six GPUs leave two free; smoke first | “Leaves two” assumes an eight-GPU node, shareable scheduler policy, and compatible allocation topology. It is contingent, not guaranteed by `cluster.md`. The local grader requests two GPUs. Any live job must first use the bounded purpose-smoke and inspect real output, but no live run was authorized here. |
| 5. New study ID, fresh curriculum, audit, two-commit preregistration | Appropriate only for a study intended to become confirmatory. Exploratory construction should remain explicitly nonclaim. A strong confirmation should use a genuinely held-out question population; the adaptively reused public 50 cannot become held out through new IDs. Within a paired mechanism experiment, sharing the curriculum may be preferable to generating confounded curricula. |
| 6. Rotate deleted/used credentials | Correct security advice. Deleting a value is not revocation. Contents of `.env` were not inspected. |

A future confirmation also needs an explicit target effect, power/precision
analysis, multiplicity plan, failure estimand, and (if “parity” matters) an
equivalence margin. More rollouts narrow rollout noise; they do not repair
question-set reuse, judge bias, or construction confounding.

## 8. First-principles reconstruction of the first self-quizzing attempt

The intended historical procedure was:

1. **Choose chapters.** Order corpus modules and expose the next fixed group.
2. **Generate quizzes.** The same Qwen model explores each chapter and writes
   usage/behavior/location/pitfall questions with anchors.
3. **Split a dev item.** One item per chapter is intended for an accumulating
   held-out construction exam.
4. **Attempt closed-book.** A fresh context answers using only the previous
   note and the question.
5. **Derive a reference open-book.** The same model family gets repository
   tools, and for DSPy may execute probes, without seeing the attempt/note.
6. **Adjudicate.** Another same-family call compares the derivation and attempt
   into correct/partial/wrong/unresolved plus a delta.
7. **Distill misses.** Wrong/partial differences become cited note entries;
   correct/unresolved items do not.
8. **Maintain/retest.** Later rounds append, select, compact, and retest prior
   entries while expanding coverage.
9. **Snapshot a static note.** The note is prepended to every evaluation
   question; model weights never change.
10. **Evaluate repeatedly.** Public StudyBench questions are answered over
    budgets/rollouts, externally graded, and used to choose later note variants.

The executed version departed from that intent in important ways documented by
008: dev identity/accumulation and retest isolation failed; evidence and
ensemble agreement were weaker than described; execution was optional; anchors
could be malformed; study seeding/freshness was incomplete; notes and eval
provenance were mutable; compaction changed over time; only 8 rather than about
30 dev verdicts were manually audited; and public evaluation was reused
adaptively.

## 9. What the method actually is

It is not learning in the usual parameter-update or human-memory-consolidation
sense. The frozen model does not retain a retrieval attempt across calls. The
persistent state is an external markdown note assembled from model-generated
errors and model-generated references, then inserted at evaluation time.

A precise name is:

> same-model, error-conditioned external-memory construction with static
> prompt-time retrieval.

“Self-quizzing” accurately describes how candidate note content is elicited,
but it can mislead if read as evidence that the model learned through retrieval
practice. The causal object is the **model + construction procedure + note +
evaluation harness**, not a changed model.

### 9.1 Similarities to human studying

- It forces an answer before feedback rather than only rereading.
- It uses misses to allocate future study attention.
- It supplies corrective feedback and revisits selected material.
- It accumulates a compact external study aid.

### 9.2 Important differences from human learning

- A new model context has no episodic memory of having attempted the question.
- No synaptic/parameter update or durable retrieval strengthening occurs.
- The same model family writes questions, attempts them, constructs references,
  judges errors, and writes the note, creating correlated blind spots.
- The final benefit can come entirely from extra tokens injected at inference,
  not from learned recall.
- Unlike the design's own motivating textbook analogy, this implementation's
  questions, feedback, adjudication, and study aid are generated by the same
  model family.
- The evaluation repeatedly measures a small public coding benchmark rather
  than long-term retention or transfer to a genuinely new domain.

The method is therefore human-inspired in control structure, but not close
enough to human retrieval practice to import the testing effect as a mechanism.

### 9.3 Prior-art and mechanism boundary

A bounded primary-source search was performed on 2026-07-13 to test the analogy
and novelty assumptions. It was not a systematic review, did not search every
database or citation graph, and supports no novelty claim.

Human-learning studies report that retrieval practice can improve delayed
retention relative to restudy ([Roediger and Karpicke,
2006](https://pubmed.ncbi.nlm.nih.gov/16507066/)), can outperform elaborative
concept mapping on later tests ([Karpicke and Blunt,
2011](https://pubmed.ncbi.nlm.nih.gov/21252317/)), and can support transfer to
new inference questions ([Butler,
2010](https://pubmed.ncbi.nlm.nih.gov/20804289/)). Corrective feedback matters
to the account and can improve later retention of corrected responses
([Butler, Karpicke, and Roediger,
2008](https://pubmed.ncbi.nlm.nih.gov/18605878/)). Evidence specifically about
student-generated questions is not uniformly positive; at least one classroom
study reports mixed outcome patterns ([Aflalo,
2018](https://journals.sagepub.com/doi/10.1177/1469787418769120)). Recent work
has also studied guided question generation for human learning rather than
assuming any generated question helps ([Cui et al.,
2024](https://aclanthology.org/2024.acl-long.632/)).

Those findings do not supply this system's mechanism. The human learner retains
state across practice and test; a fresh frozen-model context does not. Here the
attempt can matter later only through persistent bytes or a selection decision.
The closest computational prior-art family is therefore external verbal memory
and iterative self-feedback: [Reflexion](https://arxiv.org/abs/2303.11366),
[Self-Refine](https://arxiv.org/abs/2303.17651), and
[ExpeL](https://arxiv.org/abs/2308.10144) all precede this work. The present
study is a specific StudyBench comparison of error-conditioned static-note
construction. Experiment 005's “genuinely novel empirical question” wording is
withdrawn: this method is not novel by default, and any later novelty statement
requires a systematic comparison against these and adjacent methods.

## 10. Assumptions challenged one by one

| Assumption | Why it need not hold | Required test/control |
|---|---|---|
| Quiz misses identify the most valuable corpus knowledge | Misses can reflect sampling, ambiguous questions, poor question wording, or adjudicator error rather than useful gaps. | Independent question-quality audit; repeated attempts; externally authored curriculum; estimate miss reliability. |
| A miss measures a gap in the base model's prior | Every attempt receives `note_{r-1}`. From round 2 onward, a miss is a failure of the model-plus-current-note system, not a clean observation of the model's unaided prior. | State the conditional estimand; include no-note attempts or a crossed prior-note condition if base-prior gaps are the target. |
| A self-generated curriculum covers evaluation-relevant expertise | The model may ask easy symbol/trivia questions or share the same blind spots as its answers. | Measure synthesis depth/topic coverage and transfer to genuinely held-out user scenarios. |
| LOC-ordered modules are textbook chapters and equal `M` is fair allocation | Lines of code are not semantic importance, dependency centrality, difficulty, or user frequency. The implementation groups top-level/first-two-level paths, excludes root test/spec directories as chapters, and gives heterogeneous units the same question count. | Compare preregistered syllabus policies; report module size/dependency/topic coverage; allocate or stratify questions under an explicit target population. |
| Blind derivation is ground truth | Hiding the attempt reduces anchoring but does not remove same-model hallucination or correlated prior error. | Independent references/checkers/human adjudication on a blinded sample; persist probes and all evidence. |
| Exact quote existence verifies a correction | A real quote can be irrelevant, partial, or interpreted incorrectly. | Semantic claim/evidence audit and executable checks when the claim is executable. |
| Wrong/partial verdicts are stable learning signals | One stochastic attempt can be a transient sample; ordinal labels may be noisy. | Multiple paired attempts/references or a reliability model; predeclare how uncertainty affects distillation. |
| Dropping `unresolved` items is conservative without selection cost | It avoids encoding an unsupported answer, but may systematically discard the hardest, ambiguous, or most valuable uncertainties and makes the retained note a selected subset. | Report unresolved rates/content; independently adjudicate a blinded sample; include unresolved outcomes in the failure/selection estimand. |
| Correct items need no note content | A single correct draw need not mean robust knowledge; correct facts may provide useful structural breadth. | Repeated-attempt reliability and an arm retaining concise high-value correct/overview content. |
| Error-only notes are information-optimal | They omit orientation, dependencies, positive examples, and concepts the model never thought to quiz. | Compare against equal-length/equal-compute summary, coverage, and error-conditioned controls. |
| Appending entries monotonically helps | Every prepended token consumes attention/context and can distract or conflict. | Fixed note-token budget; factorial size/content experiment; retrieval-routed versus static injection. |
| Fixed rounds/K/M represent study compute fairly | Admission failures, variable trajectory lengths, and different corpora make calls/tokens/wall time unequal. | Predeclare the resource estimand; report calls, known/unknown tokens, GPU time, and inference-time context cost. |
| Retesting practices memory as it does for a person | Each retest is a fresh context. It cannot strengthen a durable memory of the prior attempt; it can only sample the current model-plus-note again and influence later persisted content/selection. | Describe retesting as repeated measurement/selection; test a persistent-memory or weight-update intervention separately if retrieval strengthening is the claim. |
| Same artifact type makes comparison apples-to-apples | Notes differ in length, study compute, content, inference tokens, and construction access. | Match or explicitly model construction compute, note bytes/tokens, and inference cost; include Pareto curves. |
| The forced-50 cheatsheet is unambiguously “cheap” | It is cheaper in the reported completion-token accounting, but prompt/prefill tokens, CPU/sandbox work, wall time, hardware, and later inference-context cost were not comprehensively matched. | Define and measure the resource vector before making an efficiency claim. |
| With-note improvement isolates retrieval practice | It can be caused by more/better corpus text, error targeting, verification, selection, or formatting. | Component ablations and a factorial design. |
| Dev accuracy predicts StudyBench performance | The dev set is same-model-generated and distributionally different; historically it was contaminated. | Validate correlation on development-only data, then freeze; do not call it external evidence. |
| Public benchmark reuse is harmless if each arm is declared first | Human/model choices adapt to observed score shapes even without prompt edits, causing selection bias. | Maintain an arm ledger; use nested development/validation/hidden-test populations. |
| “Success at any milestone” is one ordinary 95% test | Testing R1/R2/R4/R8 without a valid sequential or multiplicity plan inflates the chance of a positive crossing. R8 was then omitted after earlier outcomes were inspected, so the registered decision sequence was not completed. | Predeclare one primary time point or a valid group-sequential/multiplicity rule, including stopping and the treatment of every planned milestone. |
| More rollouts solve uncertainty | They reduce rollout sampling noise only. | Add new questions/domains/judges and model systematic uncertainty; perform power analysis at the intended sampling unit. |
| Similar scores mean methods tie | Non-significance is not equivalence. | Predeclare an equivalence margin and power an equivalence/non-inferiority test. |
| WAUC is a universal learning target | The 3,000-token anchor, zero-valued unmeasured 3k-to-first-budget region, held best-so-far tail, and steep low-token weighting encode a particular utility judgment. They can favor concise prompt artifacts and obscure different retention/transfer objectives. | Treat WAUC as one declared utility; report cells/resource vectors and sensitivity to defensible anchors, weights, and transfer outcomes without choosing them post hoc. |
| “Same Fugu judge” means one stable judge | Historical records bind the requested provider/model incompletely; an immutable accepted provider revision/fingerprint is absent for some calls. | Disclose missing fingerprints and use a pinned/attested judge where the provider permits. |
| The binary rubric is literally the paper rubric | The binary rule follows the author's clarification in `docs/jacob.md`; the published appendix contains older `0/0.5/1` wording. | Disclose the clarification whenever calling the local calculation paper-faithful. |

## 11. What the first experiment did and did not establish

### Established descriptively

- The implemented selfquiz notes could be constructed and used as static
  prompt-time memory.
- The pre-registered DSPy superiority criterion was not met at the executed
  R1/R2/R4 milestones; planned R8 was not run after those outcomes were seen.
- The observed effect ordering was opposite the pre-registered direction.
- In the inspected adaptive arms, note size/content and budget-cell behavior
  varied substantially; the forced-50 cheatsheet was a strong baseline.
- Historical integrity incidents were real and later documented rather than
  hidden.

### Not established

- that retrieval practice changed or improved the frozen model;
- that error-conditioned construction causes better expertise than rereading;
- that hybrid and cheatsheet are equivalent;
- that corrections improve tool use robustly;
- that larger notes causally hurt independent of content;
- that DSPy/OpenClaw differences identify the stale-prior mechanism;
- that a fresh procedure-level replication succeeded;
- that the complete pre-registered R1/R2/R4/R8 decision sequence was executed;
- that negative conclusions generalize beyond these models, corpora, questions,
  judges, budgets, or static-note methods.

High-level verdict: the method was coherent as a hypothesis about
error-conditioned external memory. Its comparison to human learning was
properly weakened in the design document, but later result language became too
strong. The executed milestones and later adaptive arms failed to demonstrate
the intended advantage; omission of planned R8 and the other defects/confounds
prevent the experiment from cleanly deciding the mechanism.

## 12. A sound next mechanism experiment

The causal graph comes first. In a reset frozen model, a discarded quiz attempt
has no path to a later evaluation. Any effect must be mediated by (a) which
material the attempt causes the procedure to select or (b) bytes retained in
the final note. The historical intervention therefore tests an external-memory
construction policy, not retrieval-induced strengthening.

The intuitive arm list is an **ablation family**, not a factorial design. In
particular, “error-conditioned note without attempts” needs a donor/pilot or
yoked error list—without some attempt there are no observed errors—and “quiz
without feedback” has no persistent intervention unless quiz material is itself
stored. If it is stored, the treatment is those stored bytes, not the transient
act of answering.

Keep these as benchmark anchors outside the crossed mechanism design:

| Anchor | Purpose |
|---|---|
| No note | Measures the frozen harness baseline. |
| Forced-50 cheatsheet | Preserves the strongest simple static-note baseline. |
| Construction-resource-, final-token-, and inference-context-matched direct summary | Tests a simple corpus-to-note policy under the same declared resource envelope. |

A clean crossed, content-yoked family can then vary two persistent components:

| Factor | Level 1 | Level 2 |
|---|---|---|
| Gap/selector signal | Target model's preregistered attempt-derived misses | Coverage-balanced independently/pilot-derived or shuffled/yoked control, matched on item count and declared topic/difficulty strata |
| Note synthesis | Attempt–reference delta | Direct question + independently verified reference summary, with the same selected items/evidence and final token budget |

The synthesis contrast must use identical selected questions and verified
evidence within each selector level. The selector contrast needs a frozen
matching/yoking algorithm, not researcher-picked “comparable” items. A separate
feedback-quality study may compare independently verified references with raw
same-model references, but crossing too many factors before powering the core
contrast would make the small benchmark less interpretable.

Design requirements:

1. Freeze the full candidate curriculum across mechanism arms; use exact
   content-yoking within synthesis contrasts and a frozen matching rule within
   selector contrasts; replicate later with a novel curriculum.
2. Match or model construction calls/tokens, final note tokens, and
   evaluation-time context cost.
3. Use fixed note budgets and a deterministic admission/selection rule.
4. Preserve all candidates, rejections, references, probes, usage, and lineage.
5. Pair generation seeds/rollout indices where the contract permits.
6. Use local Qwen grading only for adaptive screening; keep every screened arm
   in the multiplicity ledger.
7. Select at most one frozen candidate using a declared rule.
8. Evaluate that candidate once on a genuinely held-out question set with an
   independent/external grader and blinded human audit appropriate to the
   claim.
9. Choose the smallest effect worth caring about, conduct a power/precision
   analysis, and predeclare multiplicity, failures, stopping, and exclusions.
10. If the claim is “not meaningfully worse” or “equivalent,” specify the
    margin before seeing results and use the corresponding analysis.

This design can estimate the value of attempt-conditioned **selection** and
attempt–reference **rendering** beyond matched corpus text/resources. It still
does not estimate human retrieval strengthening or prove that the transient act
of quizzing is beneficial.

## 13. Revised verdict on Audit 009

Audit 009 was a valuable internal adversarial pass, not a final certificate.
Its reproduced ledgers and many code observations deserve to stand. Its
scientific wording needs the qualifications in this response:

- historical logs are candid but not uniformly inference-safe;
- current-artifact point estimates mostly reproduce, one documented bootstrap
  interval does not;
- non-significance is not parity/equivalence;
- validity defects can hide real effects as well as create them;
- internal independent review is not external replication;
- offline tests establish bounded contract evidence, not live readiness;
- local exploratory screening is now supported but deliberately nonclaim;
- strong future claims require a genuinely held-out evaluation, not another
  label placed on the adaptively reused public 50.

Its statements that the repository “earns an unusual degree of trust” and
documents incidents better than “most published papers” are subjective,
unmeasured comparisons and should not be treated as audit findings. A narrower
statement is supported: the repository preserves unusually detailed internal
disclosures and has substantial mechanical integrity checks. Likewise, “every
checkable claim reproduced” is too broad: the audit documents spot checks, one
reported interval fails reproduction, and the recomputation scripts are
absent. “Three prerequisites” also understates the actual gates, which include
clean committed code, stage-specific checkers/runtime, model cache/setup, a
design and preregistration, fresh identities/material, a real human process
where claimed, an external judge for confirmation, and a held-out population.

The appropriate project status is **go for offline design and, after a real
compute-node smoke, local diagnostic exploration; no-go for confirmatory or
publication claims until the unresolved prerequisites below are satisfied.**

## 14. Unresolved prerequisites and limits

- No live local-Qwen generation/grading smoke has been run for the new path.
- No audited Python SIF/Apptainer pair or TypeScript checker bundle is supplied.
- The local judge uses the same pinned Qwen model/revision as generation;
  same-model/self-preference bias is uncalibrated.
- There is no valid same-population protocol yet for estimating local-versus-
  external grader agreement; different exploratory/confirmatory populations
  cannot be used to claim calibration.
- The public 50 questions have extensive adaptive reuse; a genuinely held-out
  evaluation population is still needed.
- Historical raw artifacts and recomputation scripts are not packaged for
  fresh-clone reproduction.
- Human independence/blinding remains a real process, not a software fact.
- No power analysis or equivalence margin exists for the next method.
- The literature search in §9.3 was bounded rather than systematic, so no
  novelty claim is ready.
- Broad docs/source freezing has the operational tradeoff described in §6.3.
- Future selfquiz-like methods still need a preregistered curriculum-sharing,
  novelty, and bounded-repair policy.
- Credentials previously used or exposed must be revoked/rotated outside this
  repository; deletion alone is insufficient.
