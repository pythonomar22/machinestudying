# 011 — Full-DSPy self-quizzing screen

Status: implementation and offline protocol preflight complete; execution is
pending. No model output or evaluation score exists for this experiment. This
is an exploratory screen on the public StudyBench DSPy questions, not a
confirmatory experiment.

## Decision history

The first draft restricted study to `dspy/adapters` for faster iteration while
retaining all 30 DSPy questions as evaluation. That design was stopped before
any model, API, grader, or GPU call. Only eight of the 30 questions have any
registered evidence under `dspy/adapters`, and only one is supported exclusively
by adapter files. A poor full-suite score would therefore conflate a bad study
method with a deliberately incomplete syllabus.

We considered authoring an adapters-only StudyBench-style evaluation, then
rejected that plan before authoring or scoring it. A researcher-written set
would introduce new question-quality, grading, leakage, and representativeness
problems. The revised experiment restores the original domain alignment:

- study corpus: the complete pinned DSPy production source, with its registered
  test tree available as supporting evidence;
- evaluation: all 30 public DSPy StudyBench questions;
- new interventions: self-quizzing only.

The abandoned adapter implementation remains useful as offline calibration for
the deterministic resolver. It is not an empirical arm and its five questions
are not an evaluation set.

## Research questions

The authorized screen asks two practical questions:

1. What downstream behavior follows from a semantic selfquiz note when the
   study-time question answerer may inspect the pinned corpus through bounded
   ReAct before committing its answer?
2. What downstream behavior follows when generated, hard-to-verify semantic
   questions are replaced by exact static call-neighborhood questions whose
   answers and corrections are scored mechanically?

This execution does not estimate a clean causal "corpus access effect."
Historical ATTEMPT was one closed-book `Predict` call, whereas the new semantic
ATTEMPT is bounded corpus-backed ReAct. Access, scaffold, number of calls, and
study compute all change together. A current closed-book implementation switch
is retained and tested for future work, but no fresh S-closed arm is part of the
authorized first screen. Historical results are context only.

The semantic-versus-graph comparison is also a method-bundle comparison:
question construction, answer distribution, verification, correction
rendering, and study cost all change together. It must not be called a
verifier-only causal ablation.

## Historical closed-book results already available

These are the requested prior results for self-quizzing without corpus access
during ATTEMPT. They are copied from experiments 007 and 008. Accuracy is
lenient weighted rubric score; expertise is lenient WAUC. The judge was Sakana
`fugu`, not the local Qwen judge used by this experiment.

| Historical DSPy arm | Corpus access during ATTEMPT | Chapters covered | Cumulative completion tokens | direct | k5 | k20 | k20f | WAUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| faithful base | not applicable | 0/15 | 0 | 3.57 | 16.20 | 19.60 | 29.00 | 12.31 |
| forced-50 cheatsheet | not applicable | full-repository study prompt | 64,363 | 9.90 | 17.93 | 17.66 | 27.47 | 15.18 |
| semantic selfquiz R1 | **no** | 4/15 | 278,848 | 5.36 | 17.42 | 24.26 | 29.02 | 13.63 |
| semantic selfquiz R2 | **no** | 8/15 | 506,831 | 3.67 | 18.30 | 19.86 | 25.22 | 11.20 |
| semantic selfquiz R4 | **no** | 15/15 | 920,709 plus uncounted compaction | 4.66 | 21.36 | 22.40 | 26.46 | 11.76 |

All reported confidence intervals included zero for the selfquiz comparisons.
The pre-registered historical criterion—selfquiz beating the cheatsheet on
DSPy with a paired 95% interval excluding zero—was not met.

These rows have material validity limitations and cannot serve as current
controls:

- the base tree has three grades that are stale relative to overwritten
  episodes;
- neither generation tree binds an episode seed, seed group, model revision,
  corpus/source snapshot, runtime, provider ledger, or content-addressed report;
- the cheatsheet bytes were not bound to its evaluation episodes and the root
  note was later overwritten;
- the historical generator allowed provider retries and sent no explicit
  per-cell seed;
- grading used Fugu with incomplete provider identity, whereas the new screen
  uses the pinned local Qwen model;
- historical selfquiz had the dev/retest, verification, freshness, and note
  provenance defects documented in experiments 008–010; and
- the same public evaluation has informed many adaptive choices.

The table is chronological evidence about what was tried. It is not silently
upgraded by current validators. B0 below is the authorized fresh no-note local
screen control. No fresh forced-50 or S-closed study control is included.

## Frozen screen matrix

The current execution set is:

| Arm | Study artifact | ATTEMPT access | Reference/verifier | Role |
|---|---|---|---|---|
| B0 | none | not applicable | not applicable | fresh matched no-note evaluation control |
| S-open | generated semantic note | bounded corpus-backed `dspy.ReAct` | blind corpus-backed references, same-model adjudication, source gates | requested semantic policy |
| G-open | deterministic static call-neighborhood note | bounded corpus-backed `dspy.ReAct` | exact AST-derived set comparison and model-free corrections | requested objective policy |

All three downstream arms use fresh, paired evaluation seeds under the same
current generator and local judge. B0 is not a study-compute-matched control;
it estimates the effect of supplying each final note to the current answering
system, not the effect of self-quizzing conditional on equal study resources.

A fresh S-closed arm would be required for a contemporaneous closed/open policy
comparison, and a closed-book graph arm would be required to study an
access-by-question-family interaction. Neither may be added after inspecting
this screen and then described as preregistered.

Frozen namespaces and seeds:

| Object | Identifier |
|---|---|
| S-open study | `dspy-semantic-react-r4-20260713` |
| G-open study | `dspy-callgraph-react-r1-20260713` |
| B0 evaluation | `dspy-local-base-20260713` |
| S-open evaluation | `dspy-local-semantic-react-r4-20260713` |
| G-open evaluation | `dspy-local-callgraph-r1-20260713` |
| paired evaluation seed group | `dspy-local-selfquiz-screen-20260713` |
| study seed | `43001` |
| evaluation seed | `44001` |
| local-report bootstrap seed | `45001` |
| local-report bootstrap replicates | `10000` |
| B0 grade | `qwen-local-base-20260713` |
| S-open grade | `qwen-local-semantic-react-r4-20260713` |
| G-open grade | `qwen-local-callgraph-20260713` |

Smoke artifacts use separate names containing `smoke` and cannot be promoted
or substituted for these full namespaces.

The full method is one-shot at these frozen namespaces. Construction may resume
only the same immutable study ID, source, protocol, topology, deterministic
seeds, and validated artifact checkpoints. Evaluation writes a durable
per-cell attempt intent before provider contact: a final result, recorded
failure, orphaned intent, or partial intent write can never be retried in the
screen namespace. Grading applies the same pre-contact intent across every
judge tier and permits at most one frozen two-attempt judging session per
answered cell. Each new
server launch must nevertheless use a fresh append-only launcher prefix and a
collision-free port base; launcher logs are not the stochastic artifact
namespace and the launcher rejects an existing prefix. A recorded
model/parse/reference failure is a method outcome, not permission to try fresh
IDs until one succeeds. The semantic
pipeline intentionally refuses to advance to the next round unless the prior
round passes every automated completeness/readiness gate; the graph pipeline
likewise requires every frozen train and dev record. Therefore a substantive
construction failure yields no treatment result and will be reported as such.
It will not be silently dropped, repaired post hoc, or replaced by a luckier
replication.

The existing matx3 allocation exposes seven L40S GPUs, but the frozen 262k
context launcher requires tensor parallelism two on this 46 GiB GPU class.
Seven is not divisible by two. Every full construction and downstream
generation arm therefore uses the same six-GPU subset as three TP=2 servers;
the seventh GPU is deliberately unexposed to the process rather than changing
topology or launching an unsupported TP=1 server. It remains reserved by the
seven-GPU Slurm allocation and is therefore not available to another user.
Local grading uses its separately pinned
single TP=2 server on two GPUs. GPU count, server count, visible devices, and
all launcher identities are recorded in every environment artifact.

### Exact existing-allocation launch recipe

Job `16142825` was opened from `/sailhome/omarah`, so every direct runner must
override `SLURM_SUBMIT_DIR`. `srun` supplies the six-device view; the commands
assert TP=2 and use a unique immutable launcher prefix and nonoverlapping port
base every time. `#SBATCH` directives are intentionally ignored inside the
already-running allocation. Run these from the repository root, sequentially,
only after the source commit is pushed and the tree is clean:

```bash
set -euo pipefail
SIX=(srun --jobid=16142825 --overlap --nodes=1 --ntasks=1 \
  --cpus-per-task=60 --cpu-bind=none --gres=gpu:l40s:6)
TWO=(srun --jobid=16142825 --overlap --nodes=1 --ntasks=1 \
  --cpus-per-task=20 --cpu-bind=none --gres=gpu:l40s:2)

# Construction and generation smokes; never reuse these namespaces or prefixes.
"${SIX[@]}" env SLURM_SUBMIT_DIR="$PWD" SB_TP=2 \
  SB_VLLM_LOG_PREFIX=logs/vllm-16142825-smoke-semantic-b SB_PORT_BASE=30010 \
  SB_STUDY_ID=smoke-dspy-semantic-react-20260713b SB_STUDY_SEED=43001 \
  SB_ATTEMPT_ACCESS=react-corpus SB_SMOKE=1 SB_DEBUG=1 \
  bash scripts/selfquiz.sbatch
"${SIX[@]}" env SLURM_SUBMIT_DIR="$PWD" SB_TP=2 \
  SB_VLLM_LOG_PREFIX=logs/vllm-16142825-smoke-graph-a SB_PORT_BASE=30100 \
  SB_STUDY_ID=smoke-dspy-callgraph-react-20260713a SB_STUDY_SEED=43001 \
  SB_SMOKE=1 SB_DEBUG=1 bash scripts/graph_study.sbatch
"${SIX[@]}" env SLURM_SUBMIT_DIR="$PWD" SB_TP=2 \
  SB_VLLM_LOG_PREFIX=logs/vllm-16142825-smoke-base-a SB_PORT_BASE=30200 \
  SB_TASKS=dspy SB_RUN_ID=smoke-dspy-base-20260713a SB_RUN_SEED=44001 \
  SB_SEED_GROUP=smoke-dspy-selfquiz-screen-20260713a SB_ROLLOUTS=1 \
  SB_BUDGETS=direct SB_SMOKE=1 SB_LIMIT=1 SB_EXPLORATORY=0 SB_STUDY=0 \
  SB_ALLOW_DIRTY=0 SB_NOTE_PATH= SB_NOTE_MANIFEST= bash scripts/react.sbatch
"${TWO[@]}" env SLURM_SUBMIT_DIR="$PWD" SB_TP=2 \
  SB_EXPECTED_LOCAL_SERVERS=1 \
  SB_VLLM_LOG_PREFIX=logs/vllm-16142825-smoke-grade-a SB_PORT_BASE=30300 \
  SB_TASK=dspy SB_RUN_ID=smoke-dspy-base-20260713a \
  SB_GRADE_ID=qwen-local-smoke-base-20260713a SB_LOCAL_SMOKE=1 \
  SB_DEBUG=1 SB_GRADE_CONCURRENCY=1 SB_EVIDENCE_MODE=excerpt_evidence \
  bash scripts/grade_local.sbatch
```

Execution amendment, 2026-07-13: the original semantic smoke namespace
`smoke-dspy-semantic-react-20260713a`, launcher prefix
`vllm-16142825-smoke-semantic-a`, and port base `30000` were attempted once
from source commit `93caa53`. The launcher exited during GPU-identity preflight
before writing a topology, starting a model server, contacting a provider, or
creating a study directory. PyTorch 2.11 returned a typed `_CUuuid` object
where the launcher had required `str`; commit `6e4b137` added strict canonical
conversion and a live six-GPU validation. The failed `-a` identifiers remain
retired. The replacement above changes only the diagnostic smoke identifiers
and port to `-b`/`30010`; the construction protocol, seeds, and all full-run
identifiers remain frozen.

The replacement `-b` launcher subsequently reached model startup but exited
before authenticated readiness or any provider request: Hugging Face's
repository-ID path rejected the otherwise attested offline snapshot because
three non-inference metadata files were absent. That namespace and its logs are
also retired. The launcher was changed to serve the exact byte-attested local
snapshot path while retaining `Qwen/Qwen3.5-9B` as the served response name;
the separate SmallDSPy infrastructure smoke recorded in experiment 012 then
started and authenticated all three TP=2 servers successfully. The superseded
full-DSPy objective was paused before any semantic study outcome was produced.

Stop here. Inspect every smoke artifact before running anything below: confirm
the served/response model identities, exact six-GPU/three-server construction
topology, complete nontruncated ReAct trajectories and actual corpus calls,
graph gold and deterministic score reconstruction, provider-usage ledgers,
note/protocol binding, local-grade output, and healthy GPUs. A smoke failure
requires preserving that namespace, fixing and committing the defect, and
using a new smoke ID, launcher suffix, and port. It is not permission to proceed
to a full ID.

Only after all four smoke inspections pass, run the following as a separate
phase:

```bash
set -euo pipefail
shopt -s nullglob
SIX=(srun --jobid=16142825 --overlap --nodes=1 --ntasks=1 \
  --cpus-per-task=60 --cpu-bind=none --gres=gpu:l40s:6)
TWO=(srun --jobid=16142825 --overlap --nodes=1 --ntasks=1 \
  --cpus-per-task=20 --cpu-bind=none --gres=gpu:l40s:2)

# Full construction. Each invocation sees six GPUs / three TP=2 servers.
"${SIX[@]}" env SLURM_SUBMIT_DIR="$PWD" SB_TP=2 \
  SB_VLLM_LOG_PREFIX=logs/vllm-16142825-full-semantic-a SB_PORT_BASE=31000 \
  SB_STUDY_ID=dspy-semantic-react-r4-20260713 SB_STUDY_SEED=43001 \
  SB_ATTEMPT_ACCESS=react-corpus SB_SMOKE=0 SB_DEBUG=0 SB_AUDIT_PROTOCOL= \
  bash scripts/selfquiz.sbatch
"${SIX[@]}" env SLURM_SUBMIT_DIR="$PWD" SB_TP=2 \
  SB_VLLM_LOG_PREFIX=logs/vllm-16142825-full-graph-a SB_PORT_BASE=31100 \
  SB_STUDY_ID=dspy-callgraph-react-r1-20260713 SB_STUDY_SEED=43001 \
  SB_SMOKE=0 SB_DEBUG=0 bash scripts/graph_study.sbatch

S_MANIFEST=study-selfquiz/studies/dspy-semantic-react-r4-20260713/dspy/notes/note-r4.manifest.json
G_MANIFEST=study-graph/studies/dspy-callgraph-react-r1-20260713/dspy/notes/note-r1.manifest.json
S_NOTE=$(.venv/bin/python -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); print(p.parent / json.loads(p.read_text())["note_path"])' "$S_MANIFEST")
G_NOTE=$(.venv/bin/python -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); print(p.parent / json.loads(p.read_text())["note_path"])' "$G_MANIFEST")
test -f "$S_NOTE"
test -f "$G_NOTE"

# Bounded treatment-plumbing preflights use the completed full notes. They
# exercise deep manifest/protocol/final-round validation and prompt binding.
"${SIX[@]}" env SLURM_SUBMIT_DIR="$PWD" SB_TP=2 \
  SB_VLLM_LOG_PREFIX=logs/vllm-16142825-smoke-eval-semantic-note-a \
  SB_PORT_BASE=31200 SB_TASKS=dspy \
  SB_RUN_ID=smoke-dspy-semantic-note-r4-20260713a SB_RUN_SEED=44001 \
  SB_SEED_GROUP=smoke-dspy-treatment-note-plumbing-20260713a \
  SB_ROLLOUTS=1 SB_BUDGETS=direct SB_SMOKE=1 SB_LIMIT=1 \
  SB_EXPLORATORY=0 SB_STUDY=0 SB_ALLOW_DIRTY=0 \
  SB_NOTE_PATH="$S_NOTE" SB_NOTE_MANIFEST="$S_MANIFEST" \
  bash scripts/react.sbatch
"${TWO[@]}" env SLURM_SUBMIT_DIR="$PWD" SB_TP=2 \
  SB_VLLM_LOG_PREFIX=logs/vllm-16142825-smoke-grade-semantic-note-a \
  SB_PORT_BASE=31300 SB_TASK=dspy \
  SB_RUN_ID=smoke-dspy-semantic-note-r4-20260713a \
  SB_GRADE_ID=qwen-local-smoke-semantic-note-r4-20260713a \
  SB_LOCAL_SMOKE=1 SB_DEBUG=1 SB_GRADE_CONCURRENCY=1 \
  SB_EVIDENCE_MODE=excerpt_evidence bash scripts/grade_local.sbatch
"${SIX[@]}" env SLURM_SUBMIT_DIR="$PWD" SB_TP=2 \
  SB_VLLM_LOG_PREFIX=logs/vllm-16142825-smoke-eval-graph-note-a \
  SB_PORT_BASE=31400 SB_TASKS=dspy \
  SB_RUN_ID=smoke-dspy-callgraph-note-r1-20260713a SB_RUN_SEED=44001 \
  SB_SEED_GROUP=smoke-dspy-treatment-note-plumbing-20260713a \
  SB_ROLLOUTS=1 SB_BUDGETS=direct SB_SMOKE=1 SB_LIMIT=1 \
  SB_EXPLORATORY=0 SB_STUDY=0 SB_ALLOW_DIRTY=0 \
  SB_NOTE_PATH="$G_NOTE" SB_NOTE_MANIFEST="$G_MANIFEST" \
  bash scripts/react.sbatch
"${TWO[@]}" env SLURM_SUBMIT_DIR="$PWD" SB_TP=2 \
  SB_VLLM_LOG_PREFIX=logs/vllm-16142825-smoke-grade-graph-note-a \
  SB_PORT_BASE=31500 SB_TASK=dspy \
  SB_RUN_ID=smoke-dspy-callgraph-note-r1-20260713a \
  SB_GRADE_ID=qwen-local-smoke-callgraph-note-r1-20260713a \
  SB_LOCAL_SMOKE=1 SB_DEBUG=1 SB_GRADE_CONCURRENCY=1 \
  SB_EVIDENCE_MODE=excerpt_evidence bash scripts/grade_local.sbatch
```

Stop again. Inspect both treatment generation episodes and local grades. Confirm
that each run bundled the intended content-addressed note and full construction
inventory, re-derived the expected S-open or G-open protocol identity, accepted
only semantic R4 or graph R1 as appropriate, and actually prepended the note.
These bounded treatment preflights are diagnostic and never enter the full
screen population.

Only after both treatment preflights pass, run the complete grids and separate
local grades:

```bash
set -euo pipefail
shopt -s nullglob
SIX=(srun --jobid=16142825 --overlap --nodes=1 --ntasks=1 \
  --cpus-per-task=60 --cpu-bind=none --gres=gpu:l40s:6)
TWO=(srun --jobid=16142825 --overlap --nodes=1 --ntasks=1 \
  --cpus-per-task=20 --cpu-bind=none --gres=gpu:l40s:2)
S_MANIFEST=study-selfquiz/studies/dspy-semantic-react-r4-20260713/dspy/notes/note-r4.manifest.json
G_MANIFEST=study-graph/studies/dspy-callgraph-react-r1-20260713/dspy/notes/note-r1.manifest.json
S_NOTE=$(.venv/bin/python -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); print(p.parent / json.loads(p.read_text())["note_path"])' "$S_MANIFEST")
G_NOTE=$(.venv/bin/python -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); print(p.parent / json.loads(p.read_text())["note_path"])' "$G_MANIFEST")
test -f "$S_NOTE"
test -f "$G_NOTE"

# Complete paired generation grids, sequentially on the identical topology.
"${SIX[@]}" env SLURM_SUBMIT_DIR="$PWD" SB_TP=2 \
  SB_VLLM_LOG_PREFIX=logs/vllm-16142825-eval-base-a SB_PORT_BASE=32000 \
  SB_TASKS=dspy SB_RUN_ID=dspy-local-base-20260713 SB_RUN_SEED=44001 \
  SB_SEED_GROUP=dspy-local-selfquiz-screen-20260713 SB_ROLLOUTS=3 \
  SB_BUDGETS=direct,k5,k20,k20f SB_EXPLORATORY=1 SB_SMOKE=0 SB_LIMIT=0 \
  SB_STUDY=0 SB_ALLOW_DIRTY=0 SB_NOTE_PATH= SB_NOTE_MANIFEST= \
  bash scripts/react.sbatch
"${SIX[@]}" env SLURM_SUBMIT_DIR="$PWD" SB_TP=2 \
  SB_VLLM_LOG_PREFIX=logs/vllm-16142825-eval-semantic-a SB_PORT_BASE=32100 \
  SB_TASKS=dspy SB_RUN_ID=dspy-local-semantic-react-r4-20260713 \
  SB_RUN_SEED=44001 SB_SEED_GROUP=dspy-local-selfquiz-screen-20260713 \
  SB_ROLLOUTS=3 SB_BUDGETS=direct,k5,k20,k20f SB_EXPLORATORY=1 \
  SB_SMOKE=0 SB_LIMIT=0 SB_STUDY=0 SB_ALLOW_DIRTY=0 \
  SB_NOTE_PATH="$S_NOTE" SB_NOTE_MANIFEST="$S_MANIFEST" bash scripts/react.sbatch
"${SIX[@]}" env SLURM_SUBMIT_DIR="$PWD" SB_TP=2 \
  SB_VLLM_LOG_PREFIX=logs/vllm-16142825-eval-graph-a SB_PORT_BASE=32200 \
  SB_TASKS=dspy SB_RUN_ID=dspy-local-callgraph-r1-20260713 \
  SB_RUN_SEED=44001 SB_SEED_GROUP=dspy-local-selfquiz-screen-20260713 \
  SB_ROLLOUTS=3 SB_BUDGETS=direct,k5,k20,k20f SB_EXPLORATORY=1 \
  SB_SMOKE=0 SB_LIMIT=0 SB_STUDY=0 SB_ALLOW_DIRTY=0 \
  SB_NOTE_PATH="$G_NOTE" SB_NOTE_MANIFEST="$G_MANIFEST" bash scripts/react.sbatch

# Historical local grading/reporting used one TP=2 server. The current generic
# runner defaults to three and therefore receives the explicit historical count.
"${TWO[@]}" env SLURM_SUBMIT_DIR="$PWD" SB_TP=2 \
  SB_EXPECTED_LOCAL_SERVERS=1 \
  SB_VLLM_LOG_PREFIX=logs/vllm-16142825-grade-base-a SB_PORT_BASE=33000 \
  SB_TASK=dspy SB_RUN_ID=dspy-local-base-20260713 \
  SB_GRADE_ID=qwen-local-base-20260713 SB_EVIDENCE_MODE=excerpt_evidence \
  SB_GRADE_CONCURRENCY=8 SB_LOCAL_SMOKE=0 SB_DEBUG=0 \
  SB_CI_REPLICATES=10000 SB_CI_SEED=45001 bash scripts/grade_local.sbatch
"${TWO[@]}" env SLURM_SUBMIT_DIR="$PWD" SB_TP=2 \
  SB_EXPECTED_LOCAL_SERVERS=1 \
  SB_VLLM_LOG_PREFIX=logs/vllm-16142825-grade-semantic-a SB_PORT_BASE=33100 \
  SB_TASK=dspy SB_RUN_ID=dspy-local-semantic-react-r4-20260713 \
  SB_GRADE_ID=qwen-local-semantic-react-r4-20260713 \
  SB_EVIDENCE_MODE=excerpt_evidence SB_GRADE_CONCURRENCY=8 \
  SB_LOCAL_SMOKE=0 SB_DEBUG=0 SB_CI_REPLICATES=10000 SB_CI_SEED=45001 \
  bash scripts/grade_local.sbatch
"${TWO[@]}" env SLURM_SUBMIT_DIR="$PWD" SB_TP=2 \
  SB_EXPECTED_LOCAL_SERVERS=1 \
  SB_VLLM_LOG_PREFIX=logs/vllm-16142825-grade-graph-a SB_PORT_BASE=33200 \
  SB_TASK=dspy SB_RUN_ID=dspy-local-callgraph-r1-20260713 \
  SB_GRADE_ID=qwen-local-callgraph-20260713 SB_EVIDENCE_MODE=excerpt_evidence \
  SB_GRADE_CONCURRENCY=8 SB_LOCAL_SMOKE=0 SB_DEBUG=0 \
  SB_CI_REPLICATES=10000 SB_CI_SEED=45001 bash scripts/grade_local.sbatch

# Resolve exactly one content-addressed report for each completed arm, then
# run the three frozen CPU-only paired screening comparisons.
B_REPORT=(reports/dspy-local-base-20260713/qwen-local-base-20260713/dspy/report-*.json)
S_REPORT=(reports/dspy-local-semantic-react-r4-20260713/qwen-local-semantic-react-r4-20260713/dspy/report-*.json)
G_REPORT=(reports/dspy-local-callgraph-r1-20260713/qwen-local-callgraph-20260713/dspy/report-*.json)
(( ${#B_REPORT[@]} == 1 && ${#S_REPORT[@]} == 1 && ${#G_REPORT[@]} == 1 ))
[[ -f "${B_REPORT[0]}" && -f "${S_REPORT[0]}" && -f "${G_REPORT[0]}" ]]
.venv/bin/python -m studybench.screen_compare \
  --control-report "${B_REPORT[0]}" --treatment-report "${S_REPORT[0]}" \
  --intervention-description "semantic selfquiz R4 note versus no note" \
  --bootstrap-replicates 10000 --bootstrap-seed 45001
.venv/bin/python -m studybench.screen_compare \
  --control-report "${B_REPORT[0]}" --treatment-report "${G_REPORT[0]}" \
  --intervention-description "deterministic call-neighborhood R1 note versus no note" \
  --bootstrap-replicates 10000 --bootstrap-seed 45001
.venv/bin/python -m studybench.screen_compare \
  --control-report "${S_REPORT[0]}" --treatment-report "${G_REPORT[0]}" \
  --intervention-description "deterministic call-neighborhood R1 note versus semantic selfquiz R4 note" \
  --bootstrap-replicates 10000 --bootstrap-seed 45001
```

Every prefix is append-only evidence. If a smoke exposes a code defect, keep
its artifacts, fix and recommit, and choose a new smoke suffix and prefix. An
infrastructure-only construction resume retains the exact frozen study ID and
immutable checkpoints but uses a fresh launcher suffix and collision-free
port. A generation or grading intent with no final artifact is terminally
ambiguous and must not be resumed. A
recorded substantive method failure terminates that frozen arm; do not restart
it under either the same or a luckier replacement ID.

## Semantic curriculum

S-open preserves the historical iterative scheduler and changes the requested
ATTEMPT policy rather than collapsing study into one empty-note batch:

- rounds: four sequential rounds under one study ID and seed;
- syllabus: the frozen ordered 15-chapter full-DSPy scheduler:
  `dspy/teleprompt`, `dspy/adapters`, `dspy/clients`, `dspy/predict`,
  `dspy/primitives`, `dspy/utils`, `dspy/dsp`, `dspy/signatures`,
  `dspy/datasets`, `dspy/retrievers`, `dspy/streaming`, `dspy/evaluate`,
  `dspy/propose`, `dspy`, and `dspy/experimental`;
- chapters: four slots per round; R4 covers the final three unseen chapters and
  wraps to the first, so the final note has covered all 15 chapters;
- questions: five per chapter slot, with one deterministic dev holdout per slot;
- effective original population: 64 train questions and 16 dev questions;
- R2-R4 each add up to four deterministic retest items sampled from eligible
  previously resolved originals (a correct verdict or an admitted correction);
  retests diagnose retention and cannot add note entries;
- each round after R1 supplies the exact prior-round note to ATTEMPT, while R1
  starts with the empty note;
- tests and all registered `dspy/**` source files remain visible through tools;
- study calls cannot access StudyBench questions, rubrics, grades, runs, or
  experiment files.

Every accepted generated question must have at least one exact anchor in its
assigned chapter; additional cross-chapter anchors are allowed. Final-R4
chapter coverage is still not exhaustive statement or behavior coverage.
Eighty generated questions are not a proxy for how much source the model read;
trajectories and tool coverage must be reported.

S-open ATTEMPT has at most five ordinary voluntary ReAct iterations using the
exact DSPy `grep`, `glob`, and 200-line `read_file` contracts. It commits an
answer before blind reference derivation and adjudication. Reference
construction, correction-support gates, dev isolation, freshness checks,
complete trajectories, provider usage, and immutable note binding follow the
hardened protocol. Indexed ReAct traces must begin at turn zero, remain
contiguous and complete, respect the phase-specific iteration cap, and use the
exact terminal DSPy `finish` record when present. A context-truncated suffix is
rejected during resume validation and cannot contribute to automated readiness.

S-open diagnoses gaps in a model-plus-prior-note-plus-tools policy. It is not
evidence for a human retrieval-practice or memory-strengthening mechanism.

## Deterministic static-neighborhood curriculum

G-open uses exactly 16 training and four dev questions. It does not match
final-R4 S-open's population, iterative schedule, or compute: it is a
deliberately sampled objective curriculum, so G-open versus S-open is a broad
method-bundle comparison.
The analyzer covers the 139 tracked UTF-8 production Python
files under `dspy/**/*.py` at DSPy commit
`9cdb0aac28b2a04b064e40697ccd301872cf6a43`. Tests remain available to ATTEMPT
as supporting evidence but are outside the scored static relation.

This is not a Python runtime call graph. The oracle is a conservative syntactic
relation over eligible direct module-body functions and calls resolved through
the explicitly frozen bare-name/import-alias rules. Dynamic dispatch, methods
as callees, callbacks, decorators, higher-order flow, reflection, rebinding,
wildcards, and unresolved attributes are excluded and inventoried.

Target selection is corpus-only and must not load the StudyBench questions,
answers, rubrics, topics, or evidence. Candidates have a nonempty neighborhood
of at most ten included edges and are stratified by the first package component
below `dspy`. One training target is chosen from every eligible stratum, then
four additional training targets and four dev targets are selected by a frozen
degree/symbol rule. Every dev gold `(path, line)` location must be disjoint from
the union of training gold locations and prior selected dev locations. The
final target list, edge counts, candidate inventory, selector trace, source
hash, analyzer hash, contract hash, question-bank hash, and excluded-candidate
hash are frozen below before source commit and before any model call.

The analyzer inventories 139 production Python files, 230 module-function
candidates, 187 eligible targets, and 5,776 explicitly retained excluded call
candidates. The 16 train questions contain 99 gold edges; the four dev
questions contain 21.

| Split | Target | Gold edges |
|---|---|---:|
| train | `dspy.adapters.utils.get_field_description_string` | 10 |
| train | `dspy.clients.lm._add_dspy_identifier_to_headers` | 6 |
| train | `dspy.datasets.gsm8k.gsm8k_metric` | 1 |
| train | `dspy.dsp.utils.utils.print_message` | 4 |
| train | `dspy.evaluate.metrics.normalize_text` | 9 |
| train | `dspy.predict.predict.serialize_object` | 4 |
| train | `dspy.primitives.python_interpreter._jsonrpc_notification` | 2 |
| train | `dspy.propose.dataset_summary_generator.create_dataset_summary` | 5 |
| train | `dspy.signatures.signature.ensure_signature` | 9 |
| train | `dspy.streaming.messages.sync_send_to_stream` | 6 |
| train | `dspy.teleprompt.utils.get_signature` | 9 |
| train | `dspy.utils.inspect_history._red` | 5 |
| train | `dspy.adapters.utils.translate_field_type` | 8 |
| train | `dspy.adapters.baml_adapter._render_type_str` | 7 |
| train | `dspy.adapters.types.image.encode_image` | 7 |
| train | `dspy.adapters.utils.format_field_value` | 7 |
| dev | `dspy.adapters.utils.parse_value` | 6 |
| dev | `dspy.signatures.signature._parse_signature` | 5 |
| dev | `dspy.teleprompt.bootstrap_finetune.all_predictors_have_lms` | 5 |
| dev | `dspy.teleprompt.utils.eval_candidate_program` | 5 |

Frozen identities:

| Identity | SHA-256 |
|---|---|
| resolver | `0a643e7952e214d5854837d64cebb01dc351a500fae7c9beaf03178d3ce8cc34` |
| relation contract | `999daa8224ecb80fbb7d2f9c24b87463b6d59d09db5ed1691fb0c1be9c2ac142` |
| production source | `375cf6fc8379ac2db817046eca8032cc83093bee207a4f4ba605cb72298932d5` |
| logical question bank | `1ec5df5730e920ed57decc8ff5dc03481da401c93011091ca7a88409afe4c20b` |
| canonical question-bank artifact | `48fc8e6093fd95547582a4e92964d09461e90cd4b93706bdc1bf3e92585a8063` |
| selection trace | `1f5de691f0864297c009f32e0c38c6371cece304540f61d4b53ea2773e287876` |
| eligible-candidate inventory | `0e9cd2f6d8ded34265f14b0e30494dfc3772442c37390228f4e28fa526b119cf` |
| excluded-call inventory | `42d386c2e22eaae3bf5f0b337d10332de1b5367d57af0b2b9cdf7536830778c6` |
| analyzer source | `eb1dce5ac180e3a9ddda3ef96215324c277cccd5cf1b2e135cd225a538b4b85a` |
| corpus-ReAct ATTEMPT protocol | `5e92fa1f1749c7b897bb179831270badc0a1398c3e377282da04d7dd0871e78f` |

Predictions are strict JSON edge sets. Scoring is model-free exact set
comparison with TP, FP, FN, precision, recall, F1, and exact-set rate. Invalid
schema scores zero. Training corrections contain only the exact deterministic
gold and source evidence; model-spurious edges are retained in the attempt
artifact but are not copied into the note. No dev question, gold, attempt,
score, or correction may enter the note.

The graph bank tests source navigation and structural reconstruction. It does
not directly test DSPy usage expertise. Exact line/path questions may reward
locator behavior, the narrow analyzer omits much of dynamic Python, and graph
facts may transfer poorly to StudyBench. A null downstream result would not
show that objective self-quizzing in general is ineffective.

## Evaluation and local grading contract

All current arms use the same downstream contract:

- population: all 30 DSPy StudyBench questions;
- corpus: DSPy commit `9cdb0aac28b2a04b064e40697ccd301872cf6a43`;
- budgets: `direct,k5,k20,k20f` in that order;
- rollouts: three per question and budget;
- generation model: `Qwen/Qwen3.5-9B` revision
  `c202236235762e1c871ad0ccb60c8ee5ba337b9a`;
- evaluation seed: `44001`, with one shared evaluation seed group;
- study seed: `43001`;
- harness: the same faithful `dspy.ReAct` evaluator; downstream tool contracts
  do not change across arms;
- grader: the same pinned local Qwen model through authenticated loopback vLLM,
  excerpt evidence, temperature zero, seed zero, and no reasoning effort;
- headline: exploratory local-Qwen lenient score cells and lenient WAUC;
- supporting output: no-answer/failure counts, generated inference tokens,
  paired question/rollout deltas, bootstrap intervals, study calls and complete
  prompt/completion/total usage, note bytes, and tool coverage.

The local judge avoids external API cost but is not free: GPU time, energy,
wall time, and allocation opportunity cost must be reported when measured. The
same model family studies, answers, and judges, creating self-preference and
correlated-error risk. Local scores are an adaptive ranking proxy, not paper
scores or independent ground truth.

WAUC prices generated inference tokens. It does not price study compute, note
prefill/context tokens, repository reads, grader compute, or prompt tokens.
Therefore no efficiency or studying-intelligence claim follows from WAUC alone.

If the deterministic code checker is unavailable, strict and compile-rate
aggregate fields must be serialized as explicit `null` values in reports.
Core-conjunctive rubric accuracy and its WAUC remain available because they do
not use compilation. Low-level fail-closed zeros are not observed model
failures.

## Interpretation and stopping rules

Declared comparisons:

1. S-open minus B0: exploratory downstream effect of supplying the semantic
   note to the current answering system.
2. G-open minus B0: exploratory downstream effect of supplying the deterministic
   call-neighborhood note.
3. G-open minus S-open: exploratory downstream difference between objective
   graph studying and semantic studying under the same ATTEMPT access.

The first two comparisons do not match study compute. The third changes the
entire curriculum/verifier/correction bundle. Historical S-closed, cheatsheet,
and base values use a different judge and weaker provenance; they are not
pooled into these local-Qwen comparisons.

Secondary diagnostics include study-time error/admission rates, note sizes,
per-budget downstream deltas, graph exact/F1, question/tool coverage, and
resource use. They are explanatory, not independent primary hypotheses. No
semantic shadow adjudicator is part of the graph protocol; deterministic
edge-set verification is the sole graph-training verdict.

No result from this screen is confirmatory: the 30 public questions have been
used repeatedly, the methods were motivated after inspecting historical
behavior, the local judge is correlated with the generator, and three rollouts
are noisy. Do not select another variant from these scores and describe its
same-dataset success as confirmation.

## Gates before any GPU-hour evaluation

Offline preflight completed on 2026-07-13: all 328 DSPy-environment tests and
all 11 sandbox-environment tests passed; both environments compiled the Python
tree; every shell launcher passed `bash -n`; `git diff --check` passed; and
`AGENTS.md` was byte-identical to `CLAUDE.md`. The frozen full and smoke
artifact namespaces and all 16 launcher prefixes were absent immediately
before the source freeze. Runtime health and port checks remain launch-time
gates on matx3.

1. Finish implementation and every offline test.
2. Freeze exact study/run IDs, graph targets, selection trace, and hashes in
   this document.
3. Read every changed line, run the complete main and DSPy test suites, compile
   Python, syntax-check shell, run `git diff --check`, and byte-compare
   `AGENTS.md` with `CLAUDE.md`.
4. Commit and push one clean source tree before any non-smoke construction or
   evaluation.
5. On the existing allocation, run one bounded S-open construction smoke, one
   G-open construction smoke, and one bounded B0 evaluation/local-grade smoke.
   Inspect model identity, trajectories (including the pinned built-in `finish`
   action), actual corpus-tool calls, deterministic gold, provider usage, note
   binding, local grade, and GPU health.
6. If either smoke fails, preserve it only in a smoke namespace, fix and
   recommit, and choose fresh immutable run IDs.
7. Generate complete study artifacts, then run one bounded note-bearing
   generation/local-grade preflight for each final S-open and G-open note before
   either full evaluation grid. Do not edit source or protocol documents between
   construction, preflight, generation, grading, reporting, and comparison.
8. Recompute aggregates from immutable artifacts, audit missing/extra/stale
   records and non-answer policy, and adversarially review every prospective
   interpretation before adding results here.

## Results

Pending. No current-protocol performance value has been observed.
