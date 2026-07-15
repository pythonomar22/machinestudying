# 013 — SmallDSPy semantic self-quizzing screen

## Objective and status

Attempt the existing `semantic-selfquiz-v2` algorithm on the configured
SmallDSPy corpus, then evaluate any valid final note on the same five SmallDSPy
questions used by experiment 012. Vanilla v2 failed during construction and
produced no treatment. A separately identified `semantic-selfquiz-v3` adaptive
transport variant was then tested as the exploratory target. This was a
development screen, not a confirmatory or publication-ready experiment.

Final status: two v2 smokes, one v2 full construction, and the adaptive v3 full
construction all failed before producing an eligible treatment; a third v2
launcher was stopped before API readiness. V3 completed and archived R1, but
failed three frozen automated-readiness gates and stopped before R2. No R4 study
note, treatment-evaluation answer, GPT judgment, or selfquiz score exists. The
failures are outcomes of their frozen identities and are not overwritten or
rerun.

## Frozen scope and method

- Corpus: the exact 59-file `smalldspy` corpus at DSPy commit
  `9cdb0aac28b2a04b064e40697ccd301872cf6a43`, restricted to
  `dspy/adapters`, `dspy/predict`, `dspy/primitives`, and `tests/predict`.
- Production-chapter syllabus: `dspy/adapters`, `dspy/predict`, and
  `dspy/primitives`; `tests/predict` remains readable supporting evidence but
  is not a chapter.
- Method: four rounds of the existing semantic loop. Each round visits all
  three production chapters and requests five questions per chapter. One
  question per chapter is deterministic dev; four are train. Therefore the
  full original population is 48 train and 12 dev questions. R2–R4 may each add
  up to three deterministic retests from resolved prior original train items.
- ATTEMPT: `react-corpus`, receiving the previous-round note, question, and the
  exact `grep`, `glob`, and 200-line `read_file` tools for at most five ReAct
  iterations.
- Reference and correction policy: two blind corpus-backed derivations, exact
  source-line evidence, semantic support checks, reciprocal reference
  agreement, attempt adjudication against both references, and correction
  admission only when both references support it.
- The initial v2 port changed no method component. After its terminal failures,
  v3 keeps the prompts, model, sampling, master seed and seed-derivation policy,
  tools, iteration caps, ensembles, train/dev/retest rules, distillation, note
  rendering, and semantic readiness gates, but changes and records the adapter
  policy described below.
- Because SmallDSPy is a subset of DSPy, the existing question-freshness rule
  compares new SmallDSPy questions with both prior SmallDSPy curricula and the
  archived full-DSPy selfquiz curricula; a new task label must not make an old
  DSPy question appear fresh.
- Model: `Qwen/Qwen3.5-9B` revision
  `c202236235762e1c871ad0ccb60c8ee5ba337b9a` with the repository's frozen
  sampling policy and no provider retries.
- Study seed: `43001`.

The only persistent learned state is the cumulative Markdown note. Model
weights do not change. Same-model question writing, reference construction,
adjudication, and distillation remain a central limitation.

## Immutable namespaces

| Object | Identifier |
|---|---|
| failed v2 construction smoke | `smoke-smalldspy-semantic-react-20260714c` |
| failed v2 construction smoke | `smoke-smalldspy-semantic-react-20260714d` |
| stopped pre-API launcher | prefix `smoke-e` |
| failed v2 full construction | `smalldspy-semantic-react-r4-20260714a` |
| failed v3 full construction | `smalldspy-semantic-react-r4-20260714b` |
| unattempted v3 treatment evaluation | `smalldspy-local-selfquiz-20260714b` |
| paired evaluation seed group | `smalldspy-local-cheatsheet-screen-20260713` |
| evaluation seed | `44001` |
| unattempted v3 GPT sensitivity screen | `smalldspy-selfquiz-gpt54-whole-high-20260714b` |

Any failed smoke or full namespace is retained and retired; it is not retried
under the same identity. A full construction begins only from clean, pushed
source. The v3 full construction is not represented as a rerun of v2.

## Adaptive v3 transport amendment

The v2 failures exposed two independent DSPy/Qwen serialization modes. Stock
`ChatAdapter` falls back to `JSONAdapter` after a parse failure. For local Qwen,
DSPy reports response-schema support as unavailable and the ReAct
`next_tool_args: dict[str, Any]` field is open-ended, so that fallback requests
only a generic JSON object. Qwen returned both an unexpected leading-dot field
name and an object with `>json`/`>reasoning` fields. Separately, Qwen produced a
syntactically valid `finish` action with nonempty arguments, violating the
frozen `finish {}` contract.

`semantic-selfquiz-v3` makes the following prospective, immutable change:

1. The normal DSPy ChatAdapter-formatted request remains primary.
2. A genuine adapter parse failure, or a parsed ReAct action that violates the
   frozen tool-name/argument contract, receives exactly one JSON-formatted
   repair request constrained by an explicit strict JSON schema. ReAct uses a
   four-branch `oneOf` schema pairing each tool name with only its legal
   arguments; `finish` requires exactly `{}`. Closed output signatures receive
   signature-derived closed schemas.
3. Transport/runtime failures are not retried. Responses with any provider
   finish reason other than `stop` fail rather than being repaired or accepted.
   The strict repair parser uses standards-compliant JSON, rejects duplicate,
   missing, or extra top-level keys, revalidates output types and the tool
   contract, and does not invoke `json_repair`.
4. Every provider call records the adapter mode and outcome, logical-call
   grouping, finish reasons, output fields, exact provider outputs, exact
   repair response format and hash, selected-output hash, and hashed error
   identity. Adapter-audit completeness is an automated readiness gate. The
   tool contract is checked before the first QUIZ model contact.

This is not a vanilla-v2 rerun. In particular, repairing a validly parsed but
illegal tool action gives the model a constrained second serialization attempt
instead of executing that illegal action. That can affect the learned note and
is part of the v3 treatment bundle. Moreover, the study identity participates
in stochastic namespaces, so the mandatory fresh v3 identity produces different
realized per-call seeds even though master seed `43001` and the derivation policy
are unchanged. V3 is therefore neither an isolated adapter ablation nor a
continuation of the failed v2 draw. The task/note and internal artifact schema
is bumped from 4 to 5, the adapter policy is frozen in the task manifest, and a
fresh construction identity is used. All realized output schemas compile under
the exact pinned vLLM xgrammar backend; primary success, repair, malformed
repair rejection, truncation rejection, and call-audit behavior were exercised
offline before launch.

## Evaluation plan

The preregistered conditional plan was to bind any eligible final R4 note
through its immutable construction manifest and prepend it to every SmallDSPy
question. Had construction succeeded, evaluation would have used the exact five
questions, `direct,k5,k20,k20f`, three rollouts, and the same seed group as the
frozen `h` baseline and cheatsheet populations, producing 60
intention-to-treat cells. The same paper-style GPT-5.4 whole-evidence protocol
used in experiment 012 would then have graded the answers. Comparisons to
baseline WAUC `5.220664728639385` and cheatsheet WAUC `18.06684522893066` would
have been descriptive grader-protocol sensitivity results only. Because no
eligible R4 note exists, none of this evaluation plan was executed.

Study compute, final note size, and inference-context cost are not matched to
forced-50. The five public questions are adaptively reused, and the GPT judge
has known contestable treatment-favoring decisions. Consequently, even a large
score change cannot establish a robust studying-method effect.

## Execution log and results

The initial smoke identity `smoke-smalldspy-semantic-react-20260714a` was
attempted once from source commit `dffd9ed`. Slurm restored the parent
interactive allocation's `/sailhome/omarah` working directory, so the shell
failed to open `scripts/run_args.sh` and exited before setup, model-server
launch, provider contact, or study-artifact creation. The identity, launcher
prefix, and port base are retired. Smoke `b` adds only an explicit Slurm
working directory and fresh infrastructure identifiers.

Smoke `b` was attempted once from source commit `102fd73`. It stopped before
model-server launch, provider contact, or study-artifact creation because the
isolated vLLM environment-inventory writer imported the full
`studybench.provenance` module. That transitively imported `json_repair`, which
is correctly absent from the exact vLLM server lock. Only the GPU inventory was
written. The `b` identity, launcher prefix, and port base are retired. The
installed-distribution byte-inventory implementation is now dependency-light
and shared by the main provenance module and isolated server setup; no package,
model, prompt, sampling, seed, or study-method contract changed. Smoke `c` uses
fresh identifiers.

Smoke `c` (`smoke-smalldspy-semantic-react-20260714c`) ran v2 once from clean
source commit `35fd2fa86e26a613a74b0656cf858658ce485281` on three TP=2 replicas
(six L40S GPUs). Six authenticated model-discovery GETs succeeded, then replica
0 served eleven local Qwen chat-completion requests, all HTTP 200; replicas 1
and 2 served no generation. The retained episode extracted three questions and
reports 99,478 prompt plus 3,494 generated tokens, but its final `finish` action
supplied `chapter`, `num_questions`, and `questions` instead of `{}`. Tool
execution raised `ValueError: Arg questions is not in the tool's args`; the
observation was an execution-error traceback, not `Completed.`. The exact
trajectory gate therefore rejected the episode. No aggregate questions,
training item, note, evaluation, or score was produced.

Smoke `d` used the same clean v2 source with a new identity and derived seed.
After six successful model-discovery GETs, replica 0 served eleven local Qwen
chat-completion requests, all HTTP 200; the other replicas served no generation.
The final extraction began with `{">json": {">reasoning": ...}}` rather than
the declared output fields. Stock JSONAdapter rejected it. The error episode has
zero questions, 88,946 prompt plus 6,216 generated tokens, and no downstream
construction artifact beyond the error episode.

Launcher `e` wrote the pre-readiness GPU/package/runtime/model-cache inventory
and topology files and began loading three TP=2 replicas on ports 36440--36442.
All three logs end immediately after initial profiling/warmup completed, before
`Application startup complete`, model discovery, or any chat request. The
retained evidence does not establish why the launcher stopped, so it is recorded
only as a pre-API interruption, not a model attempt. Its retained launcher files
also include two temporary GPU TSVs; no study tree, selfquiz log, or lock exists.

Full v2 identity `smalldspy-semantic-react-r4-20260714a` then ran once from the
same clean commit on ports 36500--36502. All 22 local Qwen chat-completion
requests returned HTTP 200 after six successful model-discovery GETs. Generation
was distributed 16/6/0 requests across replicas 0/1/2. The adapters and predict
chapters produced five questions each (6 and 10 calls), while primitives failed
after 6 calls when stock JSONAdapter received `{".next_thought": ...}`. Total
retained usage is 110,227 prompt plus 8,301 generated tokens. Because one chapter
failed, question aggregation terminated; no training, dev phase, completed
round, note, evaluation, or score exists. This identity remains a failed
construction rather than a research outcome.
It was launched even though neither prior smoke completed end to end and launcher
`e` never reached API readiness, contrary to the project's smoke-before-full
policy. That procedural deviation is retained rather than rationalized away.

Smoke `c` and `d` each retain a task manifest, R1 manifest, environment snapshot,
one episode, launcher inventories/logs, selfquiz log, and serialization lock.
Full `a` retains the corresponding files and three episode files. None retains
`questions.jsonl`, freshness, training/dev artifacts, usage ledgers, a note,
summary, evaluation, or score. Later vLLM `EngineDeadError` messages followed
cleanup SIGTERM and are shutdown artifacts, not the initiating failures; every
generation POST above had already returned HTTP 200.

### Adaptive v3 full construction

V3 identity `smalldspy-semantic-react-r4-20260714b` ran once from clean, pushed
source commit `0594dda880f344dd826dd9c569f5a28226b5ff1b`. Three authenticated
TP=2 replicas served the pinned Qwen model on ports 36510--36512 using six L40S
GPUs; the seventh allocated card remained outside the homogeneous TP=2 topology.
The 319 chat-completion requests were distributed 147/57/115 across the three
replicas. Every request returned HTTP 200, every provider finish reason was
`stop`, all response IDs were unique, and every recorded response model was
`Qwen/Qwen3.5-9B`.

The quiz phase produced exactly 15 fresh questions: five per production
chapter, split into 12 train and 3 dev with all four declared question types.
There were zero exact or near overlaps against the 172-question frozen
comparison bundle. All three quiz trajectories ended in legal `finish {}` /
`Completed.` turns. Two primary quiz actions violated the finish-argument
contract and were successfully handled by the preregistered v3 repair policy.

Across the complete R1 archive there are 319 provider calls: 305 accepted
primary ChatAdapter calls, 7 rejected primary calls, and 7 accepted strict-schema
repair calls. All 319 calls retain exact provider outputs and pass the adapter
audit. Provider-reported usage is complete: 862,568 prompt tokens, 284,579
generated tokens, and 1,147,147 total tokens. Construction-only usage is 271
calls and 995,483 total tokens; dev evaluation accounts for the remainder.

The 12 training items all completed at the execution level, but the substantive
verdict distribution was 6 `correct`, 1 `wrong`, and 5 `unresolved`. Each
unresolved item lacked the method's required pair of independently
source-supported references. Only the one wrong item yielded an admitted
correction. Its citation was manually checked against
`dspy/primitives/module.py:262` (`Defaults to 1.`), and the resulting R1 note is
584 characters with SHA-256
`186dec4837ee296f56e7a55c3091955fcfa304dd6630fe96f12c617cc9930445`.
That note is retained failure-state evidence, not an eligible treatment. The
citation check was a post-run diagnostic spot check; the registered full human
audit was not performed after automated readiness had already failed.

Of three dev references, one was `ok` and two were `unresolved`. Consequently,
one dev pair completed (`correct` with and without the note), while two dev exam
records were `reference_unresolved`. The final frozen readiness record passed
provenance, launch-environment binding, question freshness, quiz completeness,
lineage, evidence safety, usage, adapter audit, and model-identity checks. It
failed exactly `training_complete`, `dev_references_complete`, and
`dev_exam_complete`. `automated_claim_ready`, `claim_ready`, publication
readiness, and confirmatory readiness are all false. The launcher therefore
exited after R1 and did not begin R2--R4.

The 40-file construction inventory and its hashes revalidate with the
constructor's structural artifact checker. The general semantic archive replay
does not accept this failed tree: it requires completed dev records, and the
constructor replay requires prior-round readiness. That is a limitation of the
failed-run audit tooling and another reason not to present the R1 note as a valid
study archive; it did not cause the three substantive readiness failures above.

The treatment evaluation and GPT screen were not launched. Pointing either at
the partial R1 note would change the registered four-round intervention after
observing its failure. Therefore this experiment has no selfquiz evaluation
number and supports no baseline/cheatsheet performance comparison. Its honest
result is a construction-feasibility failure under this strict semantic
self-quizzing protocol and this Qwen model, not evidence that self-quizzing
improves or harms downstream StudyBench performance.
