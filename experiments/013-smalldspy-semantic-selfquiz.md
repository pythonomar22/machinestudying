# 013 — SmallDSPy semantic self-quizzing screen

## Objective and status

Attempt the existing `semantic-selfquiz-v2` algorithm on the configured
SmallDSPy corpus, then evaluate any valid final note on the same five SmallDSPy
questions used by experiment 012. Vanilla v2 failed during construction and
produced no treatment. A separately identified `semantic-selfquiz-v3` adaptive
transport variant is therefore the current exploratory target. This is a
development screen, not a confirmatory or publication-ready experiment.

Current status: two v2 smokes and one v2 full construction contacted the local
Qwen server but failed before producing a completed round; a third launcher was
stopped before API readiness. No R4 study note, treatment-evaluation answer, or
selfquiz score exists yet. The failures are outcomes of their frozen identities
and are not overwritten or rerun.

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
  tools, iteration caps,
  ensembles, train/dev/retest rules, distillation, note rendering, and semantic
  readiness gates, but changes and records the adapter policy described below.
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
| v3 full four-round study | `smalldspy-semantic-react-r4-20260714b` |
| v3 treatment evaluation | `smalldspy-local-selfquiz-20260714b` |
| paired evaluation seed group | `smalldspy-local-cheatsheet-screen-20260713` |
| evaluation seed | `44001` |
| v3 GPT sensitivity screen | `smalldspy-selfquiz-gpt54-whole-high-20260714b` |

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
   identity. Adapter-audit
   completeness is an automated readiness gate. The tool contract is checked
   before the first QUIZ model contact.

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

The final R4 note will be bound through its immutable construction manifest and
prepended to every SmallDSPy question. Evaluation uses the exact five questions,
`direct,k5,k20,k20f`, three rollouts, and the same seed group as the frozen `h`
baseline and cheatsheet populations, producing 60 intention-to-treat cells.
The same paper-style GPT-5.4 whole-evidence protocol used in experiment 012 will
grade the answers. Comparisons to baseline WAUC `5.220664728639385` and
cheatsheet WAUC `18.06684522893066` are descriptive grader-protocol sensitivity
results only.

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
and 2 served no generation. The retained
episode extracted three questions and reports 99,478 prompt plus 3,494 generated
tokens, but its final `finish` action supplied `chapter`, `num_questions`, and
`questions` instead of `{}`. Tool execution raised `ValueError: Arg questions
is not in the tool's args`; the observation was an execution-error traceback,
not `Completed.`. The exact trajectory gate therefore rejected the episode.
No aggregate questions, training item, note, evaluation, or score was produced.

Smoke `d` used the same clean v2 source with a new identity and derived seed.
After six successful model-discovery GETs, replica 0 served eleven local Qwen
chat-completion requests, all HTTP 200; the other replicas served no generation.
The final extraction
began with `{">json": {">reasoning": ...}}` rather than the declared output
fields. Stock JSONAdapter rejected it. The error episode has zero questions,
88,946 prompt plus 6,216 generated tokens, and no downstream construction
artifact beyond the error episode.

Launcher `e` wrote the pre-readiness GPU/package/runtime/model-cache inventory
and topology files and began loading three TP=2 replicas on ports 36440--36442.
All three logs end immediately after initial profiling/warmup completed, before
`Application startup complete`, model discovery, or any chat request. The
retained evidence does not establish why the launcher stopped, so it is recorded
only as a pre-API interruption, not a model attempt. Its retained launcher files
also include two temporary GPU TSVs; no study tree, selfquiz log, or lock exists.

Full v2 identity `smalldspy-semantic-react-r4-20260714a` then ran once from the
same clean commit on ports 36500--36502. All 22 local Qwen chat-completion
requests returned
HTTP 200 after six successful model-discovery GETs. Generation was distributed
16/6/0 requests across replicas 0/1/2. The adapters and predict chapters produced
five questions each (6 and 10 calls), while primitives failed after 6 calls when
stock JSONAdapter received
`{".next_thought": ...}`. Total retained usage is 110,227 prompt plus 8,301
generated tokens. Because one chapter failed, question aggregation terminated;
no training, dev phase, completed round, note, evaluation, or score exists.
This identity remains a failed construction rather than a research outcome.
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

No semantic study outcome has yet been produced. The next eligible construction
is the freshly versioned v3 `...20260714b` identity above.
