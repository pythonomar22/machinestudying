# 008 — Repository and artifact audit (2026-07-12)

**Scope.** This began as a read-only research audit of the paper,
implementation, experiment notes, generated study material, runs, and grades
as they existed on 2026-07-12. After that inventory, the same session hardened
the implementation as documented in §9. It records no new evaluation result.
Historical numbers below are quoted from experiments/001–007 and are explicitly
separated from what the live artifact tree can reproduce now. No API, model,
GPU, benchmark, Slurm, container, or network research run was made; validation
was limited to offline static and unit checks.

**Bottom line.** The repository contains a substantial and useful empirical
program, but its strongest honest conclusion remains negative: under the local
faithful ReAct harness, no self-quizzing/static-note variant beat the cheap
cheatsheet produced by 50 forced ReAct study steps with a 95% paired confidence
interval excluding zero. The pre-registered DSPy success criterion was not met
and the predicted
DSPy-over-OpenClaw ordering was falsified. The current tree does **not** support
a fresh procedure-level replication claim: hybrid2 is half-fresh and only
partly graded, while hybrid3 is ungraded and its DSPy quiz curriculum repeated
the first pipeline exactly. Historical holdout leakage, missing provenance, and
stale/incomplete grades cannot be repaired retroactively. The future-run code
now fails closed on the mechanical identity, completeness, and lineage defects
it can verify. It cannot prove semantic freshness or human independence, and a
new confirmation still needs a clean committed tree, genuinely fresh study
material, an independent human audit, contained checkers, and preferably a new
hidden question split.

## 1. What the paper establishes

Machine studying starts with an agentic system
`Σ = (model, context, harness, assets, neural auxiliaries, tools)` and a
declarative corpus `D` that contains no task labels, rewards, or revealed test
distribution. A study procedure may change any part of the system before a
hidden downstream evaluation. The evaluation remains open-book: `D` is still
available at inference time. The target is therefore not mere corpus
memorization, but better conversion of inference compute into accuracy.

The paper's expertise metric is a weighted area under the best-so-far
performance curve over log generated-inference-tokens. It anchors at 3,000
tokens, assigns zero below the first measured point, carries the last score into
the tail, and uses a weight that halves with each doubling of inference compute.
This makes low-cost performance especially important. The decay and 3k anchor
are defensible choices, not natural constants; changing them can change method
rankings. Appendix D analogously defines studying intelligence as area under
expertise versus study compute, but the paper does not measure it.

The paper evaluates three study families:

- continual pre-training on code or documentation;
- synthetic SFT followed by on-policy self-distillation recovery; and
- a model-written cheatsheet produced by 50 forced ReAct study steps.

In the local faithful implementation, a caught `finish` selection remains one
recorded ReAct step and receives a continue-searching observation. Thus 50
steps does not necessarily mean 50 repository-tool executions.

Published lenient-WAUC results are:

| Method | Study-DSPy | Study-OpenClaw |
|---|---:|---:|
| Qwen3.5-9B base | 6.49 | 7.64 |
| CPT(code) | 3.71 | 7.82 |
| CPT(doc) | 3.92 | not reported |
| SFT + OPSD | 3.29 | not reported |
| self-written cheatsheet | 9.65 | 8.18 |

The paper therefore supports a narrow result: under its recipes, weight updates
did not improve expertise, while the cheatsheet improved DSPy mainly at cheap
budgets and offered only a small OpenClaw gain. It does **not** establish that
weight updates cannot work in general. Study compute was not matched, only three
rollouts were used, the coding benchmark has 50 public questions total, and the
expertise weighting is normative.

The Study-Literature result is also narrower than “retrieval is insufficient”
in general. GPT-5.1 and GPT-5.5 reached roughly the same share of must-cite
papers, but GPT-5.5 retained 11–20 points more of the recent must-cite papers in
its final top 100. That is evidence that recognition/selection differed even
when retrieval reach was similar; it is not a controlled intervention on study.

One textual discrepancy matters for grading: Appendix A.5 still describes
`0/0.5/1` claim scores, while the author later confirmed that the reported
coding runs used `0/1` and that Table 1 lenient is the pure weighted claim sum.
The local pipeline follows the author's correction.

## 2. Dataset and corpus inventory

### External benchmark tasks

| Paper task | Local benchmark data | Local corpus | Test-time scope | Local status |
|---|---|---|---|---|
| Study-DSPy | 30 questions; 6 topics; 143 rubric claims; 183 evidence spans | `corpora/dspy` at `9cdb0aac28b2a04b064e40697ccd301872cf6a43` | `dspy/`, `tests/`; Python answers | present and exercised |
| Study-OpenClaw | 20 questions; 4 topics; 100 rubric claims; 111 evidence spans | `corpora/openclaw` at `da228660306b55a9cce3b973946f3aacfc515848` | `src/`, `extensions/`; TypeScript answers | present and exercised |
| Study-Literature | none | none of the approximately 50k-paper corpus | BM25 search, select 100 papers | paper/docs only; no local implementation or run |

The 50 local coding questions, rubrics, golds, and excerpts are benchmark data.
The self-quiz files are not an external evaluation dataset; they are generated
study/development material. Each of the archived and fresh pipelines contains
86 DSPy item records and 92 OpenClaw item records across rounds 1–4, including
15 and 16 dev-marked records respectively. Across both pipelines that is 172
DSPy and 184 OpenClaw item records. Their labels are model-produced and subject
to the verification defects in §6, so they must not be described as ground
truth training data without further validation.

At the initial audit checkpoint, the repository reader was root-restricted but
not literally code-extension-only: it loaded every non-binary file no larger
than 5 MB under the configured roots. That historical scope included 2 DSPy and
75 OpenClaw Markdown/MDX/text-like files. It did not expose benchmark files,
which live outside the mounted corpora, but it contradicted the stronger
“code-only” description and may have differed from the paper harness. The
future-run reader is now suffix-allowlisted and rejects oversize/invalid allowed
code rather than silently omitting it; see §9. This does not change the scope of
already-recorded episodes.

### Locally absent study methods

CPT(code), CPT(doc), SFT+OPSD, and Study-Literature appear in the paper and
documentation but are not implemented or reproduced in this repository.
Retrieval-routed notes, self-quiz-derived SFT/RL, and the proposed R3/R8 follow-up
directions are proposals, not completed evaluations. Round 3 study artifacts
exist, but there was no R3 milestone evaluation; R8 was explicitly descoped.

## 3. Complete experiment ledger

“Recorded” means the number appears in the dated experiment notes. “Live” means
the corresponding episode/grade files currently exist. A live run is not fully
reproducible when its note content, hash, seed, or matching grade is missing.

| Effort | Episodes / judge | Recorded result | Live evidentiary status on 2026-07-12 |
|---|---|---|---|
| Native base, `runs/base` | 600; GPT-5.4 and Fugu grades | GPT WAUC 26.40 / 20.20 (DSPy/OpenClaw); Fugu 27.99 / 18.16 | all 600 episodes `ok`; both 600-grade sets present; materially stronger/cheaper than paper harness, so not an absolute replication |
| Native no-think-history ablation | 600 | 26.37 / 22.53 | run and grade artifacts deleted; result survives only in experiment notes |
| Native whole-file / judge-effort ablations | regrades of base | whole-files 27.04 / 20.05; OpenClaw xhigh 17.95 | ablation grade directories deleted; regenerable from base episodes if grader/config remain available |
| Native cheatsheet, `runs/cheatsheet` | 600; Fugu | 28.45 / 17.91; paired delta +0.84 [−4.88, 6.39] / −0.98 [−6.06, 3.97] | all episodes and grades present; no significant effect on the stronger native harness |
| Faithful base, `runs/react` | 600; Fugu | 12.31 [9.0, 16.4] / 8.45 [5.1, 13.0] | 594 `ok`, 6 `no_answer`; 600 grades; 3 run files are newer than grades |
| Faithful cheatsheet, `runs/react-cheatsheet` | 1,200 after extension to 6 rollouts; Fugu | initial 3-rollout WAUC 15.18 / 10.59; paired vs base +2.88 [−0.84, 7.10] / +2.36 [−0.57, 5.66] | 1,185 `ok`, 15 `no_answer`; 1,200 grades; 2 stale grades; evaluated note is not embedded/hashed in episodes and root cheatsheets were later overwritten |
| Selfquiz R1 | 600; Fugu | 13.63 / 10.17; vs cheatsheet −1.66 / −0.68, both nonsignificant | 596 `ok`; 600 grades; 4 stale |
| Selfquiz R2 | 600; Fugu | 11.20 / 10.64; DSPy worsened as note grew | 595 `ok`; 600 grades; 8 stale |
| Selfquiz R3 study | no milestone grid | study artifacts only | archived and fresh round artifacts present; no R3 performance result exists |
| Selfquiz R4 | 600; Fugu | 11.76 / 9.36; vs cheatsheet −3.29 / −1.51 | 593 `ok`; 600 grades; 7 stale; pre-registered success failed |
| Select-12 | 600; Fugu | DSPy cells 4.4/19.8/22.3/28.7, −2.45 vs cheatsheet; OpenClaw +1.43 vs base | 590 `ok`; 600 grades; 5 stale |
| Usage snippets | 600; Fugu | DSPy direct unchanged; k20f 34.3 (highest recorded cell); OpenClaw admitted 0/12 snippets | 591 `ok`, 9 non-ok; all 600 grades current; “execution-grounded selection” is overstated by the implementation |
| Hybrid breadth + corrections | 1,200; Fugu | final 6v6 delta vs cheatsheet −0.61 [−2.76, 1.52] / −0.84 [−3.62, 2.12] | 1,177 `ok`, 23 non-ok; 1,200 current grades; supports parity, not superiority |
| Studied summary | 1,200; Fugu | vs cheatsheet −3.12 [−6.99, 0.74] / −1.82; large 40.8k/45.8k-character notes hurt direct | 1,185 `ok`, 15 non-ok; 1,200 current grades |
| Hybrid2 “fresh” replication | 1,200 episodes; Fugu incomplete | OpenClaw −0.08 [−2.63, 2.70] vs cheatsheet; DSPy result incomplete | 1,187 `ok`; 865 grades total (OpenClaw 480/480, DSPy 385/720), 335 missing and 10 stale; used old deterministic cheatsheet plus fresh corrections, so it is half-fresh |
| Hybrid3 corrected replication | 1,200 episodes; no grades | no valid performance result | 1,182 `ok`; zero grades. Fresh cheatsheet seed and fresh corrections, but all 75/75 newly generated DSPy quiz questions exactly repeat run 1; OpenClaw overlap is 0/80 |
| R8 | none | descoped after declining R1/R2/R4 curve | no run should be inferred |

For hybrid3, the detailed non-ok breakdown is DSPy 708 `ok`, 5
`no_answer`, 7 `error`; OpenClaw 474 `ok`, 4 `no_answer`, 2 `error`.
Hybrid2 has 715/720 DSPy and 472/480 OpenClaw `ok` episodes. The grader
historically converted empty/non-ok answers to zero and reports included them.
The hardened policy distinguishes model `no_answer` (ITT zero) from
infrastructure/forced-short failures (immutable failed attempts that must be
retried under the identical contract before strict reporting). Historical
tables retain their original policy and are not silently recomputed.

## 4. What the replication program actually learned

The first native tool-calling harness was not faithful to the paper. It retained
interleaved reasoning, lacked the paper's `finish` affordance, searched longer
at voluntary budgets, generated far fewer tokens, and scored much higher. Those
runs were still useful: they showed that harness strength changes both the
absolute frontier and the apparent marginal value of studying.

The dspy.ReAct redo reproduced the relevant regime. OpenClaw is a good full-row
replication: local WAUC 8.45 versus paper 7.64, with all cells close. DSPy direct
and forced-20 endpoints match almost exactly (3.6 versus 3.3; 29.0 versus 29.4),
but local k5/k20 cells remain approximately twice the published values. Calling
the entire DSPy row an exact replication would therefore be false. Remaining
divergences include Fugu rather than GPT-5.4, evidence excerpts rather than
confirmed whole-file context, unknown original DSPy version/tool caps, and a
lighter forced-search token profile.

The first genuinely local studying method is the self-quizzing/error-delta note:
the model generates corpus-derived quizzes, attempts them closed-book, derives
an open-book answer, adjudicates the gap, and writes selected corrections into
a prepended note. Conceptually this is formative assessment of a frozen
model-plus-note system, not the human testing effect: retrieval practice cannot
update frozen model weights by itself.

Across the historical local measurements:

- The pre-registered success criterion was not met at R1, R2, or R4.
- The predicted larger DSPy effect was falsified; OpenClaw was more durable.
- Small, broad notes were more useful at direct than large comprehensive notes.
- Error-delta corrections often improved tool-budget point estimates relative
  to base, and execution-gated DSPy snippets produced the best recorded k20f
  cell, but these are adaptive, multiple-arm observations with confidence
  intervals that include zero—not confirmatory findings.
- The 6-rollout hybrid tied, rather than beat, the cheatsheet. This is the most
  reliable local summary of the static-prepend design space explored so far.
- On reported generated-study-tokens, self-quizzing was much less efficient
  than the one-pass cheatsheet.

## 5. Accounting and provenance corrections

1. The experiment-007 cumulative selfquiz token table does not match the live
   round summaries. Archived run-1 summaries sum to **920,709 DSPy** and
   **1,832,447 OpenClaw**, while the table states approximately 943k and 1,656k.
   The fresh summaries sum to **935,111 DSPy** and **1,747,223 OpenClaw**.
2. The cheatsheet was not a matched “64k per corpus” study point. The evaluated
   faithful study episodes used 64,363 generated tokens for DSPy and 25,183 for
   OpenClaw. The experiment table's 64k OpenClaw entry is wrong. Current root
   cheatsheet episode files are later hybrid3 products (39,467 DSPy and 21,044
   OpenClaw generated tokens), not the evaluated original pair.
3. Root cheatsheets and current selfquiz notes are mutable paths. Evaluation
   episodes store neither the note text nor a content hash, git revision, corpus
   revision, prompt/config digest, nor the sampling seed. Older runs therefore
   cannot be tied byte-for-byte to their study artifact from episode files alone.
4. Round-1 summary files were written before the documented recovery pass, so
   some historical admitted-entry counts differ from final recovered notes.
5. Study and inference “tokens” count generated/completion tokens. Prompt and
   note-prefill tokens, sandbox/CPU work, and wall-clock compute are unpriced.
   This follows the local/paper convention but is not a full FLOP or cost measure.
6. The historical pipeline persisted no machine-readable report snapshots.
   Its Markdown tables remain historical records; re-running hardened
   `report.py` over the mutable legacy tree is not a retroactive certification.
7. At the audit checkpoint, retried episode files were newer than 39
   corresponding grades across the React base, cheatsheet, R1/R2/R4, select,
   and hybrid2 arms. Hardened strict reporting now rejects this condition for
   new manifests, but the old grades remain stale.

## 6. Integrity and correctness register at the initial audit checkpoint

This register preserves what was wrong in the historical implementation and
artifacts. “Required fix” records the scientific requirement; §9 states the
current status of the future-run safeguards. A code fix does not upgrade old
evidence.

| ID | Severity | Finding and evidence | Consequence / required fix |
|---|---|---|---|
| I1 | critical | Prior dev items enter `prev_items` (`selfquiz.py:364–374`); retest copies omit `dev` (`393–399`); distillation excludes only `item.get("dev")` (`306–317`). Actual leaked prior-dev retests: archived DSPy 2/OpenClaw 1; fresh DSPy 2/OpenClaw 1. | The holdout trained the note. Preserve dev identity and categorically prohibit dev-derived distillation, including retests. |
| I2 | critical | The promised accumulating dev exam is not implemented. Each round summarizes only its current records (`435–457`); prior dev items are used only through the retest pool. | No valid cumulative dev curve exists. Maintain a separate immutable dev manifest and score it without updating on it. |
| I3 | high | Retests themselves can add entries because only `dev` is excluded. Retest-derived entries contributed archived DSPy 7/46 and OpenClaw 4/57 entries; fresh DSPy 6/45 and OpenClaw 5/50. | “Retest” is partly additional training, not an independent retention measure. Split training refresh from diagnostic retest. |
| I4 | critical | A reference derivation is selected by evidence-list length, without validating evidence before adjudication (`261–275`). The final quote gate requires only one six-character substring near one line (`184–199`) and never proves that the correction follows from it. | “Verified correction” is too strong. Validate every cited path/line/quote and require a semantic entailment/claim check before admission. |
| I5 | high | OpenClaw ensemble “agreement” only checks that another derivation also labels the attempt wrong/partial (`288–304`); it need not agree on the correct answer or correction. | Require substantive agreement over normalized claims/evidence, otherwise mark unresolved. |
| I6 | high | `run_python` is available for DSPy but need not be called; probe code/output and evidence class are not required or auditable. Selection claims execution priority, but scoring is only wrong=2/partial=1 (`551–570`). | Persist tool trajectory/probe output and make execution a mechanically checked field before calling an item execution-grounded. |
| I7 | high | The specified anchor-existence gate is absent before quiz admission. Manual audit found malformed/nonexistent anchors: archived DSPy 5/OpenClaw 14; fresh DSPy 5/OpenClaw 1. | Implement root containment, path existence, and optional line/symbol validation before ATTEMPT. |
| I8 | critical | `fresh_lm` and the selfquiz CLI expose no study seed. The alleged fresh DSPy curriculum repeats all 75/75 newly generated questions; OpenClaw repeats 0/80. | Add and persist per-episode seeds; fail a fresh-run gate on excessive exact/near overlap. Regenerate DSPy before any fresh claim. |
| I9 | medium | `RepoTools` loads all text-like files ≤5 MB under roots, with no extension filter (`tools.py:115–139`). | Either describe the scope honestly or add an explicit, tested file allowlist matching the benchmark protocol. |
| I10 | critical | Eval episodes omit note text/hash and seed; mutable root notes/cheatsheets have been overwritten by later arms. | Write an immutable manifest per variant and copy/hash the exact note before the first episode. Refuse resume if hashes differ. |
| I11 | high | The judge schema constrains claim IDs to an enum but cannot enforce uniqueness. After the second duplicate/missing-ID attempt, `grade_episode` still writes the malformed verdict (`grade.py:172–193`). Four stored Fugu grades contain duplicate/omitted claims. | Validate ordered unique IDs and exact cardinality after every attempt; fail without writing a grade if still malformed. |
| I12 | critical | `report.py` warns about count mismatches but aggregates any graded subset and does not compare mtimes (`57–83`). Hybrid2 can produce cells from as few as 1/180 or 24/180 grades and still return a four-point WAUC. | Default to fail-closed on missing, stale, duplicate, or unexpected populations; permit exploratory partial reports only with an explicit flag and conspicuous labels. |
| I13 | medium | DSPy code is executed; OpenClaw code gets only a tree-sitter syntax parse (`sandbox.py:43–97`). | Keep task conclusions separate. Do not interpret cross-task differences as execution-grounding effects without an executable TS checker. |
| I14 | high | The fresh `.venv-dspy` lacked tree-sitter packages that the old selfquiz launcher required even though the quote-only selfquiz implementation never used them. | The future launcher now requires only its actual DSPy/Pydantic runtime; TypeScript parsing remains isolated to the grading environment. |
| I15 | medium | Current faithful comparisons use Fugu and excerpts; the paper used GPT-5.4 and may have used whole evidence files. | Internal paired comparisons remain useful, but label them local-harness results, not byte-exact paper reproduction. |
| I16 | high | `ForcedReAct` catches `ValueError`, breaks, extracts, and leaves the episode status `ok` (`react.py:80–106`). One R4 OpenClaw forced episode has 12/20 actions; one hybrid3 DSPy episode has 16/20. | Record early termination and fail/retry forced episodes unless all 20 iterations are represented under the declared finish semantics. |
| I17 | high | Infra/no-answer episodes remain as zeros in uneven counts across arms; many retries have stale grades. | Freeze a pre-declared failure policy and regrade every retry before paired analysis. Report both intention-to-run and valid-episode sensitivity analyses. |
| I18 | high | Only 8/8 dev verdicts were human-audited, versus the pre-registered approximately 30 and ≥80% gate. | Dev scores never cleared their own steering gate. Finish a blinded stratified audit before using dev labels to alter the method. |
| I19 | medium | Qualitative inspection finds many selfquiz questions closer to symbol/trivia recall than the benchmark's cross-file user-scenario synthesis; no formal distribution audit was run. | Measure question type, synthesis depth, anchor validity, and benchmark-nearest-neighbor overlap before claiming representative practice. |
| I20 | medium | Cap-triggered compaction was deferred after OpenClaw R2 exceeded the cap; a later routine/guard was added and one merge rejected. The executed curve therefore did not follow a single fully frozen maintenance protocol. | Treat compaction as an explicit arm with immutable inputs and a pre-declared guard, not a silent implementation detail. |
| I21 | medium | Generated snippets that fail the usage gate are not persisted. | Store candidate, parser/runtime output, environment manifest, and rejection reason for every entry. |
| I22 | medium | No report JSON, note manifest, or end-to-end artifact audit is emitted by batch completion. | Make these mandatory run products and have finalization fail if they are absent. |

## 7. Claims this repository can and cannot support

### Supported, with the stated scope

- The native tool-calling harness is a much stronger and cheaper search regime
  than the paper's dspy.ReAct harness; harness choice materially changes scores.
- The faithful local harness reproduces OpenClaw's published base row well and
  DSPy's direct/forced endpoints, while DSPy voluntary-search cells remain hot.
- In same-harness, same-judge historical comparisons, the cheap cheatsheet is a
  strong baseline and no tested selfquiz/static-note arm significantly beats it.
- The pre-registered selfquiz criterion failed and the proposed DSPy-over-
  OpenClaw ordering was falsified.
- The 6-rollout original hybrid is statistically consistent with cheatsheet
  parity. This is artifact-level evidence, not a fresh procedure replication.
- Larger static notes correlate with worse cheap-budget performance in these
  adaptive arms; compact breadth is the better current engineering default.

### Not supported

- A clean procedure-level fresh replication of hybrid or self-quizzing.
- A claim that dev holdout results guided development without leakage.
- A claim that all admitted corrections are source-verified or
  execution-verified.
- A causal claim that retrieval practice improves a frozen model's memory.
- A general claim that self-quizzing corrections improve expertise, or that the
  best k20f point is robust after multiplicity correction.
- A locally measured studying-intelligence score with consistent study-cost
  accounting.
- Local conclusions about Study-Literature, CPT, SFT+OPSD, retrieval-routed
  notes, or weights trained from selfquiz artifacts.
- A broad conclusion that studying methods or weight updates cannot work. This
  program tested a small model, two local coding corpora, one public 50-question
  benchmark, and a highly specific metric/harness.

## 8. Remediation order before spending more evaluation compute

1. **Freeze the current evidence.** Copy exact notes into immutable variant
   directories and write a manifest containing SHA-256 hashes, episode/study
   seeds, git revision, corpus commits, model/server/environment versions,
   prompts/config, run start time, and judge configuration.
2. **Repair study validity.** Separate immutable accumulating dev data from
   training/retest data; block every dev-to-distill path; validate anchors;
   require correction/evidence agreement; persist execution probes; implement
   meaningful ensemble agreement; add deterministic seed control and overlap
   gates.
3. **Repair evaluation validity.** Make forced iteration failures explicit;
   make malformed judge output fatal; make reports reject partial/stale
   populations; pin real execution/compile checkers; decide and freeze the
   non-ok-episode policy.
4. **Reconcile existing artifacts.** Regrade the 39 stale React episodes,
   complete or formally abandon hybrid2, and do not grade hybrid3 as a fresh
   DSPy confirmation because its study curriculum is not fresh. Persist report
   JSON alongside any corrected table.
5. **Run one clean confirmation, not another adaptive arm.** Pre-register one
   method, use genuinely new seeded study artifacts, preserve all provenance,
   and preferably evaluate on a new hidden question split. Six or more paired
   rollouts improve precision but cannot cure reuse of the same 50 public
   questions.
6. **Only then expand the method class.** Query-routed retrieval over compact
   verified notes is the most direct next hypothesis because static prepending
   exposes a clear size/distraction frontier. A weights track using audited
   self-QA can follow, but current generated labels are not clean enough for a
   publishable SFT/RL comparison.

The future-run safeguards in steps 1–3 are substantially implemented. This
hardening diff has passed offline tests and a final adversarial review; commit
it as a clean baseline, then new method design and offline implementation can
continue. No historical result needs to be "fixed" first. A claim-ready run is
a separate gate: the pinned runner
environments need a fresh sync check, the canonical two-arm preregistration
must be introduced in its add-only commit, and the external checker artifacts
and independent human process described below remain unavailable. Step 4 was
deliberately not executed: no
API/model/GPU run was authorized, and regrading cannot restore missing
historical note bytes or erase leakage. Steps 5–6 remain future research.
Additional GPU-heavy evaluation should wait until the enforced prerequisites
pass and a genuinely fresh confirmation is pre-registered.

## 9. Post-audit hardening status (updated 2026-07-13)

The findings above describe the historical artifacts and the code at audit
time. The same work session subsequently implemented the following safeguards.
They prevent recurrence in new runs; they do not retroactively repair or
upgrade any historical result in §3.

| Area | Implemented safeguard | Current operational status |
|---|---|---|
| Corpus identity | DSPy and OpenClaw commits are exact constants. Validation enumerates the pinned Git tree, rejects `assume-unchanged` and `skip-worktree` state, permits only ordinary or executable code blobs, and verifies the exact Git blob object and mode again at every exposed read. | Both live corpora pass at the recorded commits. A post-validation worktree edit cannot be served as pinned corpus content. |
| Source freeze | Generation records the clean Git commit plus a byte inventory covering runtime code, scripts, data, registrations, locks, instructions, paper, README, experiment documents, and the complete test tree (excluding only untracked interpreter/test-runner caches). It independently compares scoped HEAD-tree entries, index entries, live Git-blob bytes, and executable modes; hidden index flags and ignored/untracked scoped files force `dirty: true` even when porcelain status is silent. Claim-ready grading, reporting, and comparison require the current inventory to equal that launch record exactly. | Judge prompts, score code, analysis code, tests, fixtures, or protocols cannot be changed after observing generation outcomes and reused under the same run. A change requires a new baseline, preregistration, and run IDs. |
| Dataset identity | The two JSONL files are pinned by SHA-256 and expected count; validation rejects duplicate/non-finite JSON, schema drift, duplicate IDs, bad weights/references, non-code evidence paths, invalid UTF-8, and any excerpt not byte-equivalent after deterministic line numbering and an exact pinned-blob read. | All 30 DSPy and 20 OpenClaw records pass strict offline validation. |
| Tool scope | Only `.py` is exposed for DSPy and only `.ts`, `.tsx`, `.js`, `.mjs`, `.cjs` for OpenClaw. Root escapes, escaping symlinks, invalid UTF-8/NUL content, malformed paths, non-integer line arguments, duplicate/invalid runner selections, unsupported Git modes, and bytes that drift from the pinned blob fail closed. Each tool instance preloads one exact snapshot. | No allowed code is silently size-filtered; the 9.82 MB OpenClaw viewer runtime is included. |
| Python checks | Host execution was removed. A valid answer runs only through a content-pinned absolute Apptainer SIF and content-pinned Apptainer executable, with `containall`, clean environment, no home, no network, one work bind, process-group timeout, and POSIX resource limits. | No image is bundled or assumed. Without both configured hashes, Python `compile_ok` is intentionally false and no generated code runs. |
| TypeScript checks | Tree-sitter is labelled syntax-only and records both grammar and core parser binary versions/hashes. Strict success requires an absolute executable compiler plus matching SHA-256. | No real compiler is configured by default, so syntax can be reported but `compile_ok` remains false. |
| Checker provenance | `sandbox.configuration_record(language)` returns the immutable image/runtime or compiler/parser identities without evaluating generated code. Each check binds the exact configuration hash and verifies the configuration both before and after execution; strict downstream validation independently reruns the deterministic check. | Mixed, changed, or self-consistently rewritten checker outcomes are rejected. A real strict report therefore still requires the pinned checker to be available. |
| Python environments | Setup pins the main interpreter to 3.14.6 and auxiliary DSPy/vLLM interpreters to 3.12.11. Root and DSPy projects have frozen locks, and each runner now syncs and checks its applicable lock before use; Optuna and the selfquiz parser additions were removed because those paths do not import them. The TypeScript parser remains pinned in the main grading environment. | Refresh with setup before a future run. The current DSPy environment is drifted from its lock and is not ready for a claim run until that refresh succeeds. Selfquiz preflights only its actual DSPy/Pydantic imports. |
| vLLM environment | A versioned 189-package lock is checked exactly. Startup records the full sorted inventory and hashes every file declared by every installed distribution's `RECORD` (including each `RECORD`), exact Python/vLLM entrypoint, CUDA home, `nvcc` bytes/version, Torch/CUDA versions, effective TP, and model revision. Model-cache inventory accepts Hugging Face's normal logical file links only after binding the stable link identity to a non-symlink regular blob inside the exact cache root; symlinked directories, escapes, nested blob links, special files, mid-hash mutation, and path replacement fail. The complete canonical inventory is regenerated immediately before server launch and after authenticated readiness, before episodes begin. Resume and paired-arm checks reject identical package versions with different attested installed bytes. | Model/runtime/local-cache identity is manifest-bound and startup is offline after explicit cache population. Failed post-readiness equality tears down the topology. This brackets model loading against accidental or persistent concurrent drift but does not make a same-user cache immutable or exclude an adversarial write-and-restore entirely between checks; use a read-only content-addressed mount when that threat matters. Successful loading shows that the local snapshot is usable, not that it matches an authoritative upstream file inventory. Original wheel archives, files not declared by a distribution, the standard library, driver binaries, and system libraries are not byte-archived by this repository. |
| GPU allocation and server identity | Serving requires a Slurm job ID and explicit unique `CUDA_VISIBLE_DEVICES`; maps each logical CUDA device to its runtime UUID before UUID-scoped inspection; and records each CUDA identifier, UUID, model, memory, driver, host, and Slurm binding. The server and inventory subprocesses receive a minimal allowlisted environment with ambient proxies, Python paths, and dynamic-loader injection variables cleared. Loopback endpoints are canonicalized; ports are preflighted; the ephemeral API key is delivered only in the final server process environment rather than an argument; authenticated readiness is bound to the launcher/job/host/process set. Current runner headers request six GPUs on the `matx` partition, which is divisible by the L40S TP=2 policy and leaves two devices free on an eight-GPU node. | There is no fallback that inspects unallocated node GPUs or accepts a stale server. No Slurm/GPU/server command was launched during this audit. |
| Preregistration | One strict canonical two-arm document freezes the hypothesis, exact note hashes, future run IDs, generation settings, failure policy, grading policy, primary estimand, paired bootstrap, multiplicity choice, and stopping rule. A two-commit design proves the implementation predates the registration; generation, grading, reporting, and comparison all revalidate the exact snapshot. | No confirmatory preregistration exists yet. After the reviewed implementation baseline is committed, introduce and commit only a new direct `preregistrations/*.json` file before either arm runs. Exploratory artifacts cannot be promoted later. |
| Run identity | Eval requires run ID, master seed, paired seed group, task-bound note manifest, and exact model revision; study requires study ID and seed. The first launch remains the immutable substantive environment baseline, while every launch writes a content-addressed exact environment snapshot and every generated artifact binds the snapshot that produced it. Cross-allocation resume permits only declared Slurm/allocation/transport nuisances. | New claim-ready runs cannot silently reuse mutable namespaces, unpaired sampling seeds, or substantively drifted retry environments. |
| Artifact primitives | Canonical JSON rejects duplicate keys and non-finite values. POSIX `dir_fd`/openat traversal with `O_NOFOLLOW` anchors reads, hashes, parent creation, mutable writes, hard-link-based immutable create-if-absent writes, and retained owner-only locks against symlink-swap races. Readers reject special files and mid-read mutation. | Existing identical bytes may be resumed only after their complete dependency chain is revalidated. Drift requires a new namespace. This deliberately fails closed without the required POSIX and hard-link primitives. |
| Provider attempts | SDK-internal retries are disabled. Every generation and judge transport attempt records a request hash, attempt ordinal, provider response identity, available usage, and failure status. Received malformed responses are retained once and are fatal; missing usage is `null`/unavailable, never fabricated as zero. | Infrastructure failures remain outside the successful ITT population until an identical-contract retry succeeds; all retained failures are report-visible. |
| Native and DSPy generation | Native responses may execute exactly one tool call per iteration and cannot call tools outside the declared budget. Faithful `dspy.ReAct` and the native runner remain distinct harnesses. Before immutable write or resume, both producers run the same full downstream episode validation over model/harness/provider identity, response IDs, attempt and usage ledgers, counters, budgets, failures, answer, and note binding. Missing or malformed DSPy usage is rejected rather than converted to zero. Completed forced-50 studies additionally revalidate intent, exact answer-note bytes, and construction inventory. | No historical episode is upgraded. A new faithful run requires the fully attested local server environment. |
| Selfquiz construction | Train, cumulative dev, and retest lineage are disjoint; dev/retest records cannot distill. Completed artifacts must match the exact phase call graph, strict integer seeds, planned chapters/episodes/ordinals, two distinct reference derivations, reciprocal answer/correction consensus, recomputed dev verdicts/deltas, source citations, raw evidence, trajectories, calls, usage, launch environment, dependencies, and failures. Same-study prior rounds and cross-study curricula enter the freshness audit. | Automated same-model verification is diagnostic only and always emits `claim_ready: false`. Lexical freshness and exact state-machine validity are not proof against semantic overlap. |
| Human audit | A complete-population, blinded, independent audit protocol with the exact versioned decision rule `all-required-reviews-pass-v1` must be snapshotted before round 1. Offline promotion strictly parses the protocol/result, requires one exact auditor identity throughout, re-hashes every construction dependency, re-derives the exact record/entry population, and writes a separate audited manifest without editing the automated record. Boolean review fields must be actual JSON booleans. A valid failure is archived and permanently blocks later promotion of the same construction. Evaluation and grading repeat the population, auditor, and decision validation. | Software verifies bytes, completeness, and declarations; it cannot prove that a reviewer was actually independent or blinded. |
| Grading | The author-confirmed binary rubric and pure weighted lenient sum are enforced. Before any provider request, grading revalidates the exact preregistered grader/evidence/effort policy, frozen source, environment and human-audit dependencies, episode/note bytes, prompt, checker configuration, and explicit canonical provider endpoint. The accepted raw response is retained and its bytes/hash are reparsed against stored claims and scores. Tool counters and observations reconcile exactly; at most two explicit judge attempts are allowed. | OpenAI grading is explicitly pinned to `https://api.openai.com/v1` rather than SDK ambient configuration. Whole-file evidence is paper-faithful; excerpts are a local diagnostic. Missing/mutable provider fingerprints are disclosed rather than invented as stable revisions. |
| Strict reporting | Reports require the complete manifest grid; revalidate preregistration, frozen source, every episode launch snapshot, note/audit dependencies, every grade, and an independent deterministic checker rerun; retain model non-answers as zero; reject stale/duplicate/unexpected artifacts; and disclose failures/fingerprints. The writer then reloads the underlying population and independently recomputes the aggregate, bootstrap, audit, and any contextual paper record before writing canonical content-addressed JSON. The expertise implementation follows the paper's 3k anchor, best-so-far envelope, log-token integration, weights, and held tail; regression tests cover Appendix C's 10.8 worked example and the DSPy Table 1 base value 6.49. | Partial historical reporting remains visibly diagnostic and cannot become claim-ready. Table 1 is contextual because the exact original environment is unavailable. |
| Paired comparison | Both content-addressed reports and their underlying populations are revalidated. Opposite roles must share the exact committed preregistration, intervention text, grading policy, and analysis/bootstrap contract. The note is the only intervention. Separate jobs may differ only in disclosed allocation/transport nuisances; substantive model/runtime/hardware/seeding fields must match. The bootstrap shares sampled questions and rollout indices. The writer reloads both reports and independently rebuilds every pairing, estimate, interval, source record, and hash. | Comparison is ITT and immutable. Missing generation fingerprints or incomplete/mismatched accepted judge fingerprints make it diagnostic rather than claim-ready; incomplete provider revision identity is never called exact. |
| Secret file | Shell preflight checks only metadata. If Python needs secrets, the loader parses a current-user-owned, regular, non-symlink mode-0600 `.env` as literal `KEY=VALUE` records, rejects ambiguity before mutating the environment, and never evaluates shell syntax. | The live `.env` satisfies the metadata check. Its contents were never printed or inspected during this audit; any formerly valid key should still be rotated. |

Offline verification performed after the changes:

- all 50 checked-in benchmark records validated against exact corpus excerpts;
- all 165 main-environment tests passed across environment loading, race-safe
  artifact primitives, preregistration, provenance, datasets/tools,
  sandboxing, scripts, generation, grading, strict reporting, paired
  comparison, and human-audit population/decision validation;
- all 48 DSPy-environment tests passed across faithful ReAct study reuse and
  selfquiz citation, exact state/call/seed lineage, split isolation, freshness,
  human-audit promotion, trajectory, environment, usage, and concurrency
  contracts (213 unique offline tests total);
- all ten shell/setup/Slurm files passed individual `bash -n` checks; and
- the edited Python files compiled successfully without model, API, GPU,
  container, network, or Slurm execution.

The most important remaining containment/reproducibility limits are explicit:
Apptainer shares the host kernel; no Python SIF or TypeScript compiler artifact
has yet been built and pinned; installed vLLM distribution bytes are attested
from `RECORD` but original wheel archives, undeclared files, and system runtime
bytes are not preserved; cache equality brackets model loading but is not an
immutable-filesystem guarantee against same-user write-and-restore; serving
providers may omit mutable runtime fingerprints (which is now
disclosed rather than fabricated); human independence/blinding remains a real
process rather than a software proof; the 50 public questions have been reused
adaptively; and all old runs retain the provenance, leakage, staleness, and
multiple-comparison limitations documented above. The hardening changes must
also be committed before `source.dirty: false` can support a new claim-ready
run; the offline tests and final adversarial code review reported here found no
remaining concrete integrity blocker.
