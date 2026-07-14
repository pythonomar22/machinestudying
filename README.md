# Machine Studying replication and method research

This repository studies whether a model can use a declarative code corpus to
become more effective on later coding questions. The corpus remains available
at inference time. Studying therefore means changing the model system before
evaluation—such as by constructing a note—not hiding the source and testing
memorization.

The current honest result is narrower than a successful new studying method.
Under the local `dspy.ReAct` harness matching the author-confirmed ReAct
mechanics, the executed pre-registered self-quizzing milestones (R1/R2/R4) did
not demonstrate superiority over the forced-50 cheatsheet by the paired 95%
confidence-interval criterion. That cheatsheet had lower reported study
completion-token cost; total compute and inference-context cost were not
matched. The planned R8 milestone was not run after the earlier outcomes were
inspected, so the complete pre-registered sequence was not executed. Later
static-note variants were adaptive and likewise supplied no robust superiority
evidence. Hybrid2 is incomplete, and hybrid3 is not a fresh DSPy replication
because all 75 generated DSPy quiz questions repeated the first pipeline.
These are useful diagnostic results showing a failure to demonstrate
superiority, not evidence of equivalence, no effect, or a general failure of
studying or weight updates.

The detailed paper interpretation, dataset inventory, experiment ledger, and
initial defect register are in
[experiments/008-repository-and-artifact-audit.md](experiments/008-repository-and-artifact-audit.md).
[experiments/010-audit-009-disposition-and-selfquiz-review.md](experiments/010-audit-009-disposition-and-selfquiz-review.md)
is the current project handoff: it challenges the later independent audit,
corrects parity/equivalence language, and reviews self-quizzing from first
principles.

## What is measured

The paper defines expertise as a weighted area under the best-so-far accuracy
curve over log generated inference tokens. The curve is anchored at 3,000
tokens, is zero before its first observation, and holds its final observation
through the tail. The weighting favors useful answers at low inference cost.
It is a chosen utility function, not a universal measure of learning.

The paper also proposes studying intelligence—expertise as a function of study
compute—but does not measure it. This repository records generated study
tokens, prompt tokens where available, and other usage metadata; it does not
yet claim a studying-intelligence result or full compute/FLOP accounting.

The local faithful runner uses `dspy.ReAct`. Its forced-50 study loop records
exactly 50 ReAct iterations. A `finish` selection that is caught and told to
continue remains one recorded iteration, so this is 50 ReAct steps, not
necessarily 50 repository-tool executions. The native tool-calling runner is a
separate, stronger harness and must not be pooled with or presented as an
absolute replication of the paper. New grading also distinguishes:

- `--whole-files`: the paper-faithful evidence-context variant; and
- `--excerpt-evidence`: a local diagnostic variant using benchmark excerpts.

The corrected coding rubric is binary per claim (`0` or `1`) and the lenient
question score is the pure weighted claim sum. This follows the author's
clarification; the paper appendix's older `0/0.5/1` text is inconsistent with
the reported Table 1 calculation.

## Data and research efforts

| Dataset or material | Local contents | What has been attempted | Current status |
|---|---:|---|---|
| Study-DSPy | 30 public questions, 143 claims, 183 evidence spans; DSPy at `9cdb0aac28b2a04b064e40697ccd301872cf6a43` | native and faithful base/cheatsheet runs; executed selfquiz milestones R1/R2/R4; select, usage, hybrid, summary, hybrid2, hybrid3 arms | useful historical evidence; the superiority criterion was not met at any executed pre-registered milestone, planned R8 was not run, and hybrid3 DSPy is not fresh |
| SmallDSPy iteration slice | five public Study-DSPy `react_agents_and_tools` questions and 59 Python files under four pinned DSPy source roots | paired no-note versus newly generated forced-50 cheatsheet screen, documented in experiment 012 | adaptive public subset for fast plumbing/method iteration only; it is not an independent held-out benchmark and cannot support a general DSPy claim |
| Study-OpenClaw | 20 public questions, 100 claims, 111 evidence spans; OpenClaw at `da228660306b55a9cce3b973946f3aacfc515848` | the same local static-note families | faithful base row tracks the paper reasonably well; small sample and no contained TypeScript execution currently limit claims |
| Generated selfquiz material | archived and fresh round artifacts, including internal train/dev records | error-delta note construction and cumulative internal dev exams | generated study/development material, not external ground truth; historical material cannot be promoted retroactively, while newly generated material intended for confirmation requires the complete pre-registered human audit |
| Study-Literature | not present | none | paper discussion only; no local result |
| CPT(code), CPT(doc), SFT+OPSD | no local implementation | none locally | paper baselines only; not reproduced here |

The complete per-arm episode, grade, result, and reproducibility table is in
§3 of the repository audit. Historical Markdown numbers are preserved as
historical records; the hardening work does not retroactively certify them.

## Mechanically claim-ready artifact lifecycle

A new confirmatory comparison satisfies the repository's mechanical contract
only if every stage below succeeds. The implementation fails closed when an
identity or completeness check fails. This is necessary, not sufficient, for a
scientific or publication claim: the design must also use an appropriate
held-out population, estimand, controls, precision/power, and real independent
processes where those are claimed.

1. **Pre-register.** Freeze the hypothesis, intervention, datasets, harness,
   budgets, rollout count, failure policy, evidence mode, grader, master seed,
   paired seed group, and analysis. Give every study, run, grade, report, and
   comparison a unique immutable namespace. Confirmatory evaluation uses the
   exact committed two-arm contract in
   [docs/preregistration.md](docs/preregistration.md); exploratory artifacts
   cannot be promoted after outcomes are known.
2. **Pin inputs.** Use the exact clean corpus commits and a clean source tree.
   Setup uses frozen `uv` locks and exact interpreter versions. A non-smoke run
   may not opt into a dirty tree. Claim-ready grading, reporting, and
   comparison must still run from the identical Git commit and byte-level
   research-source inventory recorded by generation; outcomes cannot be used
   to revise judge prompts, scoring, or analysis code in place.
3. **Study without benchmark access.** Study tools expose only the pinned code
   roots. They never expose `data/`, grades, runs, experiment notes, rewards, or
   benchmark questions. Every model call, seed, response identity, trajectory,
   usage record, rejection, and exact note dependency is retained.
4. **Audit generated labels.** A selfquiz audit protocol must be snapshotted
   before round 1. Automated gates can never promote a selfquiz note by
   themselves. A separate, complete, blinded and independent human audit must
   cover every cumulative train/dev record and every admitted entry; see
   [docs/human-audit.md](docs/human-audit.md).
5. **Evaluate paired arms.** Baseline and treatment use different run IDs but
   the same paired seed group and master seed. Their question/rollout seeds,
   task, grid, model, harness, sampling, corpora, environment, and inference
   effort must match. Separate Slurm jobs may differ only in recorded
   allocation/transport identities such as job ID, host, GPU UUID, inventory
   path, and ephemeral server key; GPU class/count/memory/driver, model-cache
   bytes, software/CUDA runtime, topology, and every other substantive field
   must still match. Every launch has a content-addressed environment snapshot,
   and every episode binds the snapshot that actually produced it.
6. **Grade explicitly.** Grading binds the exact episode and note snapshots,
   rubric, judge, canonical provider endpoint, prompt, evidence mode, and
   checker configuration. The accepted raw response is retained and its hash,
   parsed verdict, claims, and score must agree. Malformed judge output is a
   failed attempt, never a partial grade. A durable judge-attempt intent is
   written before the first request, so there are at most two judge requests
   for that cell across all processes and resumes. An orphaned intent is
   terminal and ambiguous; it is never permission to judge again.
7. **Report the full intention-to-run population.** Missing answers remain
   zero. Reports reject missing, stale, duplicate, or unexpected episodes and
   grades. Failed generation and judge attempts are disclosed. Before writing,
   the report reloads the population and recomputes its aggregate and bootstrap;
   strict report JSON is immutable and content-addressed.
8. **Compare paired reports.** Comparison revalidates both underlying
   populations, separates the declared note intervention from explicitly
   disclosed allocation-only nuisance differences, and rejects substantive
   drift. Its two-stage bootstrap samples the same questions and rollout
   indices in both arms. Before writing, the complete comparison is independently
   rebuilt from both reports. Generation model and available fingerprint sets
   must match within every paired question/rollout cell; call counts and
   unavailable fingerprints are disclosed without requiring equal turn counts.
   More than one available generation fingerprint in either arm is invalid.
   Missing generation fingerprints remain disclosed but do not gate readiness,
   because the exact model revision, cache, launcher runtime, and environment
   are independently bound. A missing accepted-judge fingerprint makes the
   comparison diagnostic rather than claim-ready. Exploratory
   observations remain labelled exploratory and do not become confirmatory
   through post-hoc reporting.

Never replace or edit a completed claim-ready artifact. Use a new ID for a new
study, run, regrade, report, or comparison. A retained lock file is coordination
metadata, not a result; an active lock prevents duplicate concurrent work.

## Future command templates

These templates document the intended handoff; none was executed during the
hardening pass. Run setup, serving, generation, selfquiz, grading, reporting,
and comparison from an appropriate Slurm compute allocation under the
repository's cluster policy. Generation, selfquiz, and exploratory local
grading use authenticated local vLLM servers. Confirmatory grading instead
contacts the selected external judge API and requires its provider key.
`scripts/setup.sh` may clone corpora and download Python packages.

First refresh the pinned environments, then generate and locally grade one
purpose-smoke record before executing a full population:

```bash
scripts/setup.sh
.venv-vllm/bin/hf download Qwen/Qwen3.5-9B \
  --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a

SB_TASKS=dspy SB_RUN_ID=smoke-dspy-001 SB_RUN_SEED=41001 \
SB_SEED_GROUP=smoke-pair-001 SB_SMOKE=1 SB_LIMIT=1 SB_ROLLOUTS=1 \
SB_BUDGETS=direct sbatch scripts/react.sbatch

SB_TASK=dspy SB_RUN_ID=smoke-dspy-001 \
SB_GRADE_ID=local-smoke-dspy-001 SB_LOCAL_SMOKE=1 \
SB_EVIDENCE_MODE=excerpt_evidence sbatch scripts/grade_local.sbatch
```

The generation smoke writes under `runs/smoke/`; the matching grading smoke
writes under `grades/smoke/`, grades exactly one answered cell, and exits
without a report. It exercises generation persistence, local judge structured
output, grading persistence, runtime binding, and checker plumbing. It is
deliberately partial and unreportable, so a passing smoke says nothing about
full-grid completion or method quality.

The explicit download is a cache-population step and uses the network. The
claim-ready launcher itself runs Hugging Face in offline mode and inventories
every logical model entry plus its resolved cache blob through stable,
no-follow file descriptors. It rejects symlinked directories, escaping or
nested storage links, special files, and mutation or path replacement during
hashing. A complete equality check runs immediately before vLLM starts and
again after authenticated readiness, before any episode begins; failure tears
down the server topology. Successful vLLM loading and readiness establish that
the bracketed local snapshot is usable. The repository does not retain an authoritative
upstream file inventory, so it does not claim that this proves the remote
snapshot is complete. It also hashes every file declared by every installed
Python distribution's `RECORD`; that attests the installed package bytes used
by a run, but it does not archive original wheels, undeclared files, the Python
standard library, driver binaries, or system libraries. The cache checks detect
accidental and persistent concurrent drift; they do not make a same-user cache
filesystem immutable or rule out an adversarial write-and-restore wholly
between checks. A read-only content-addressed cache mount remains the stronger
deployment when that threat is in scope.

### Full-DSPy self-quizzing screen

Forced-50 cheatsheet, semantic selfquiz, and deterministic static-neighborhood
study are distinct methods and require distinct immutable IDs. The current
semantic protocol runs four sequential rounds of four chapter slots, covering
all 15 production chapters (the fourth round wraps once), with five generated
questions per slot. Later rounds consume the prior note and include retention
retests. Its ATTEMPT is one at-most-five-turn `dspy.ReAct` policy
with the complete pinned DSPy corpus available through `grep`, `glob`, and
200-line `read_file`. DSPy also supplies the pinned terminal `finish` action.
The graph protocol instead uses a frozen sample of 16 train and four held-out
development targets, strict JSON edge sets, exact model-free scoring, and
model-free gold corrections. Its relation is conservative static syntax, not a
complete Python runtime call graph.

Construction is deliberately separate from downstream evaluation and grading.
These two commands only construct and finalize their content-addressed notes:

```bash
SB_STUDY_ID=dspy-semantic-react-r4-20260713 SB_STUDY_SEED=43001 \
SB_ATTEMPT_ACCESS=react-corpus SB_SMOKE=0 SB_DEBUG=0 SB_AUDIT_PROTOCOL= \
sbatch scripts/selfquiz.sbatch

SB_STUDY_ID=dspy-callgraph-react-r1-20260713 SB_STUDY_SEED=43001 \
SB_SMOKE=0 SB_DEBUG=0 sbatch scripts/graph_study.sbatch
```

After both constructions finish, resolve the exact content-addressed note paths
from the immutable manifests:

```bash
S_MANIFEST=study-selfquiz/studies/dspy-semantic-react-r4-20260713/dspy/notes/note-r4.manifest.json
G_MANIFEST=study-graph/studies/dspy-callgraph-react-r1-20260713/dspy/notes/note-r1.manifest.json
S_NOTE=$(.venv/bin/python -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); print(p.parent / json.loads(p.read_text())["note_path"])' "$S_MANIFEST")
G_NOTE=$(.venv/bin/python -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); print(p.parent / json.loads(p.read_text())["note_path"])' "$G_MANIFEST")
test -f "$S_NOTE" -a -f "$G_NOTE"
```

Launch the fresh no-note control and the two note treatments with the same
evaluation seed and seed group:

```bash
SB_TASKS=dspy SB_RUN_ID=dspy-local-base-20260713 SB_RUN_SEED=44001 \
SB_SEED_GROUP=dspy-local-selfquiz-screen-20260713 SB_ROLLOUTS=3 \
SB_BUDGETS=direct,k5,k20,k20f SB_EXPLORATORY=1 \
SB_STUDY=0 SB_SMOKE=0 SB_LIMIT=0 SB_NOTE_PATH= SB_NOTE_MANIFEST= \
sbatch scripts/react.sbatch

SB_TASKS=dspy SB_RUN_ID=dspy-local-semantic-react-r4-20260713 \
SB_RUN_SEED=44001 SB_SEED_GROUP=dspy-local-selfquiz-screen-20260713 \
SB_ROLLOUTS=3 SB_BUDGETS=direct,k5,k20,k20f SB_EXPLORATORY=1 \
SB_STUDY=0 SB_SMOKE=0 SB_LIMIT=0 \
SB_NOTE_PATH="$S_NOTE" SB_NOTE_MANIFEST="$S_MANIFEST" \
sbatch scripts/react.sbatch

SB_TASKS=dspy SB_RUN_ID=dspy-local-callgraph-r1-20260713 \
SB_RUN_SEED=44001 SB_SEED_GROUP=dspy-local-selfquiz-screen-20260713 \
SB_ROLLOUTS=3 SB_BUDGETS=direct,k5,k20,k20f SB_EXPLORATORY=1 \
SB_STUDY=0 SB_SMOKE=0 SB_LIMIT=0 \
SB_NOTE_PATH="$G_NOTE" SB_NOTE_MANIFEST="$G_MANIFEST" \
sbatch scripts/react.sbatch
```

Each full run contains 30 questions × four budgets × three rollouts. The
following separate local-only jobs can produce individual triage reports. Each
launcher derives the sole allowed one-shot qualification path from its fresh
authenticated server-launch ID:

```bash
SB_TASK=dspy SB_RUN_ID=dspy-local-base-20260713 \
SB_GRADE_ID=qwen-local-base-20260713 SB_EVIDENCE_MODE=excerpt_evidence \
SB_GRADE_CONCURRENCY=8 SB_LOCAL_SMOKE=0 SB_DEBUG=0 \
SB_CI_REPLICATES=10000 SB_CI_SEED=45001 sbatch scripts/grade_local.sbatch

SB_TASK=dspy SB_RUN_ID=dspy-local-semantic-react-r4-20260713 \
SB_GRADE_ID=qwen-local-semantic-react-r4-20260713 \
SB_EVIDENCE_MODE=excerpt_evidence SB_GRADE_CONCURRENCY=8 \
SB_LOCAL_SMOKE=0 SB_DEBUG=0 SB_CI_REPLICATES=10000 SB_CI_SEED=45001 \
sbatch scripts/grade_local.sbatch

SB_TASK=dspy SB_RUN_ID=dspy-local-callgraph-r1-20260713 \
SB_GRADE_ID=qwen-local-callgraph-20260713 \
SB_EVIDENCE_MODE=excerpt_evidence SB_GRADE_CONCURRENCY=8 \
SB_LOCAL_SMOKE=0 SB_DEBUG=0 SB_CI_REPLICATES=10000 SB_CI_SEED=45001 \
sbatch scripts/grade_local.sbatch
```

The semantic note remains exploratory and requires a genuinely independent,
pre-registered audit before any separate audited manifest could be created. A
syntactically valid audit declaration is not an audit. The current local-Qwen
screen is adaptive and cannot be promoted into a confirmatory claim. The exact
design, historical results, frozen graph bank, and interpretation limits are in
[`experiments/011-full-dspy-selfquiz-ablations.md`](experiments/011-full-dspy-selfquiz-ablations.md).

Do not feed those separately launched reports into a paired screen: their
qualification and judge-launch identities necessarily differ. A paired
control/treatment screen must qualify once, grade both frozen arms, and build
both reports inside one uninterrupted server lifecycle. Experiment 012 records
the exact multi-arm recipe used for the current SmallDSPy comparison.

### Exploratory local-Qwen screening

During method development, evaluate complete exploratory control and treatment
runs before using an external judge. Use distinct run IDs, the same master seed
and seed group, and the exact immutable note and construction manifest emitted
by each candidate method. The commands above are the frozen current screen.

`grade_local.sbatch` defaults to six GPUs and launches three homogeneous TP=2
servers for the exact pinned `Qwen/Qwen3.5-9B` model and revision used by the
generator. Set `SB_EXPECTED_LOCAL_SERVERS` and override Slurm's GPU request
together only when the target run manifest has a different immutable server
count. The grader orders authenticated loopback endpoints by numeric port and
routes every cell through its manifest-bound server slot; there is no endpoint
fallback or remapping. The script runs grading and reporting within the same
authenticated launcher lifecycle, sets `GRADER_MODEL=local`, and neither
requires nor uses an OpenAI or Sakana credential.

Local judge requests use the fixed temperature-zero, seed-zero, 256-token,
no-thinking score-only contract recorded by the grader. An answer-centered
system message prevents facts in the gold/evidence from being credited unless
the candidate answer itself correctly asserts them; the candidate is supplied
last in a JSON-escaped user payload. Constrained JSON contains every exact
rubric claim ID mapped directly to binary `0/1`, plus `needs_regrade`; it
requests no rationale and no model-generated total. The harness computes
weighted scores deterministically. `disable_any_whitespace` prevents arbitrary
grammar whitespace but permits xgrammar's fixed JSON separators, so raw
compactness is never a validity condition. Grades and reports bind the
local judge's model cache, software, CUDA, homogeneous GPU/topology, complete
endpoint count and slot map, request contract, and high-entropy non-secret
launcher ID. A loopback port remains disclosed transport, but a relaunch is a
different substantive grading lifecycle and cannot be spliced into an existing
population. The generic sbatch runner does not replace an experiment-specific
predeclared calibration gate. `studybench.local_judge_qualification` provides
experiment 012's balanced 20-case, 44-label, all-replica synthetic gate; it
contains no StudyBench content. It writes an immutable complete 60-request
pre-contact intent before its first judge call, then writes the terminal audit
before returning pass or failure. Local grading and reporting refuse to run
without revalidating and binding that exact same-launch passing audit. The
audit path is canonical and keyed only by the authenticated server-launch ID,
so a failed or orphaned intent cannot be bypassed by choosing another filename.
`excerpt_evidence` is the lower-context screening mode;
`SB_EVIDENCE_MODE=whole_files` supplies the full evidence files but still does
not make Qwen grading paper-faithful. Give every regrade a fresh
`SB_GRADE_ID`.

Local scores are a local adaptive ranking proxy, not ground truth. Their GPU,
wall-time, energy, and opportunity costs have not yet been measured. The exact
same pinned Qwen model is both generator and local judge, so
self-preference and correlated blind spots, factual errors, and stylistic
preferences are a central limitation. Inspect paired control/treatment changes
rather than comparing local absolute scores with Table 1. There is currently
no valid same-population local-versus-external calibration protocol: local and
external lanes intentionally accept different exploratory/confirmatory
populations, so agreement must not be inferred by comparing their scores. A
future calibration study needs its own frozen population, independent labels,
and protocol. Local grades and reports remain exploratory, non-claim-ready,
non-promotable, and unavailable for paper comparison or the strict confirmatory
comparison path.

When the deterministic checker configuration is ready, a local report may show
strict and compile-aware metrics as secondary diagnostics. When it is not
ready, the report explicitly labels the interpretation
`lenient-and-core-conjunctive-checker-unavailable`. Core-conjunctive rubric
accuracy and its WAUC remain reportable because they do not use compilation.
Low-level grade records retain zero-valued strict/compile fields as fail-closed
sentinels, but full and slice report JSON projects only unavailable strict and
compile aggregate fields to explicit `null` values. Neither sentinel is
measured strict performance or a model failure.

After complete local reports exist, compare matched arms with the separate
screening command. For experiment 011, the three frozen comparisons and their
exact intervention descriptions are recorded in
[`experiments/011-full-dspy-selfquiz-ablations.md`](experiments/011-full-dspy-selfquiz-ablations.md).
The generic form is:

```bash
.venv/bin/python -m studybench.screen_compare \
  --control-report reports/CONTROL_RUN/LOCAL_GRADE/dspy/report-SHA256.json \
  --treatment-report reports/TREATMENT_RUN/LOCAL_GRADE/dspy/report-SHA256.json \
  --intervention-description "frozen candidate note versus no note" \
  --bootstrap-replicates 10000 --bootstrap-seed 45001
```

Use the content-addressed report paths actually emitted by local reporting,
not the placeholders. No live Qwen server or GPU allocation is needed after
both reports exist: the command reuses the reports' content-bound local-server
attestations while independently requiring the current grading Python,
installed package bytes, repository source, and checker configuration to match.
It rejects stale or tampered runtime cross-bindings, smoke/partial grids, and
substantive generation, seed, runtime, grading, or checker differences outside
the note intervention. Generation model and available fingerprint identities
must match in every paired episode; per-episode provider-call and missing-
fingerprint counts, including no-answer cells and variable-turn methods, are
recorded but are not equality gates. Its shared two-stage bootstrap resamples
only questions and rollout indices. It does not include local-judge bias,
systematic grader error, topic/dependency clustering beyond that sampling unit,
adaptive public-set reuse, or other design uncertainty. A 95% interval
containing zero is inconclusive—not parity or equivalence—and even an interval
excluding zero is only a screening signal.

Choosing a method after seeing local StudyBench scores is adaptive development.
Once the method is frozen, confirmation requires a new preregistration, fresh
control/treatment run IDs, and a genuinely held-out question population. The
public 50 have already been reused adaptively; new run IDs do not make them
held out. Do not externally regrade the exploratory population and call it
confirmatory.

### External confirmation template (held-out population still required)

The commands below document the external lane, but the checked-in 30 + 20
public questions have already been reused adaptively and cannot supply the
genuinely held-out population required after local screening. Before treating
this as scientific confirmation, add and freeze a new evaluation population
through a reviewed dataset/protocol change. The existing public loaders can
produce a mechanically validated replication artifact, not erase prior
exposure.

For a paired control/treatment evaluation, first commit the reviewed
implementation baseline, then add and commit only one canonical two-arm
preregistration whose `source_commit` names that baseline. It must already
contain the future run IDs, exact note hashes, hypothesis, intervention wording,
generation and grading settings, bootstrap configuration, and stopping rule.
The file is introduced once and never edited. See
[docs/preregistration.md](docs/preregistration.md) for the full schema and
two-commit procedure.

Use distinct run IDs but one master seed and one paired seed group. Replace the
note paths with the exact immutable note and audited or forced-50 construction
manifest selected before registration and evaluation:

```bash
SB_TASKS=dspy SB_RUN_ID=control-dspy-001 SB_RUN_SEED=44001 \
SB_SEED_GROUP=paired-dspy-001 SB_ROLLOUTS=6 \
SB_PREREGISTRATION=preregistrations/paired-dspy-001.json \
SB_PREREGISTRATION_ROLE=control \
sbatch scripts/react.sbatch

SB_TASKS=dspy SB_RUN_ID=treatment-dspy-001 SB_RUN_SEED=44001 \
SB_SEED_GROUP=paired-dspy-001 SB_ROLLOUTS=6 \
SB_PREREGISTRATION=preregistrations/paired-dspy-001.json \
SB_PREREGISTRATION_ROLE=treatment \
SB_NOTE_PATH=study-selfquiz/studies/selfquiz-dspy-001/dspy/notes/by-sha256/HASH.md \
SB_NOTE_MANIFEST=study-selfquiz/studies/selfquiz-dspy-001/dspy/notes/note-r1.audited.manifest.json \
sbatch scripts/react.sbatch
```

Grade and report both fresh arms with the same explicit evidence mode, judge
effort, grader selection, and fresh grade ID. `GRADER_MODEL=openai` selects
GPT-5.4 and requires `OPENAI_API_KEY`; `GRADER_MODEL=fugu` selects the configured
Fugu provider and key. The matching report flag is `--grader openai` or
`--grader fugu`:

```bash
GRADER_MODEL=openai .venv/bin/python -m studybench.grade \
  --task dspy --run-id control-dspy-001 --grade-id grade-openai-whole-001 \
  --whole-files --judge-effort high
GRADER_MODEL=openai .venv/bin/python -m studybench.grade \
  --task dspy --run-id treatment-dspy-001 --grade-id grade-openai-whole-001 \
  --whole-files --judge-effort high

.venv/bin/python -m studybench.report \
  --tasks dspy --run-id control-dspy-001 --grader openai \
  --grade-id grade-openai-whole-001 --whole-files --judge-effort high \
  --ci 10000 --ci-seed 45001
.venv/bin/python -m studybench.report \
  --tasks dspy --run-id treatment-dspy-001 --grader openai \
  --grade-id grade-openai-whole-001 --whole-files --judge-effort high \
  --ci 10000 --ci-seed 45001

.venv/bin/python -m studybench.compare \
  --control-report reports/control-dspy-001/grade-openai-whole-001/dspy/report-SHA256.json \
  --treatment-report reports/treatment-dspy-001/grade-openai-whole-001/dspy/report-SHA256.json \
  --intervention study-note \
  --intervention-description "EXACT TEXT FROM preregistration.intervention" \
  --bootstrap-replicates 10000 --bootstrap-seed 45001
```

Do not type the literal `HASH` or `SHA256`: select the content-addressed files
actually emitted by the preceding stage. Python grading is deliberately blocked
until the configured checker image is pinned; OpenClaw grading is blocked until
an attested TypeScript checker bundle passes semantic calibration. A changed
method, population, seed, evidence mode, judge configuration, or note requires
a new namespace.
The comparison description, bootstrap count, and bootstrap seed must equal the
preregistered values exactly; they are not free post-hoc reporting choices.

TypeScript compile credit requires all four
`STUDYBENCH_TYPESCRIPT_CHECKER`,
`STUDYBENCH_TYPESCRIPT_CHECKER_SHA256`,
`STUDYBENCH_TYPESCRIPT_CHECKER_BUNDLE`, and
`STUDYBENCH_TYPESCRIPT_CHECKER_BUNDLE_SHA256` variables. The checker is an
absolute executable implementing `CHECKER SOURCE_PATH typescript|tsx`. The
bundle is canonical, compact, sorted-key JSON with a trailing newline and this
schema (the artifact array itself is sorted by absolute path):

```json
{"artifacts":[{"path":"/absolute/checker","roles":["checker"],"sha256":"CHECKER_SHA256"},{"path":"/absolute/typescript-compiler","roles":["compiler"],"sha256":"COMPILER_SHA256"}],"calibration_protocol":"studybench-typescript-checker-v1","schema_version":1}
```

Each artifact is a canonical, non-group/world-writable regular file; artifacts
with a `checker` or `runtime` role must be executable. Roles are sorted unique
subsets of `checker`, `compiler`, `configuration`, `dependency`, `library`, and
`runtime`. Exactly one checker artifact must match the configured executable,
and at least one compiler artifact is required. Include every semantically
relevant wrapper, runtime, compiler, configuration, standard-library, and
dependency file, then pin the exact manifest bytes. Neither `scripts/setup.sh`
nor `scripts/setup_grading.sh` constructs or certifies this external bundle.

Readiness runs three source-bound cases through the configured command: a
well-typed program must pass, a real type error must fail, and a TSX-mode program
with a relative TypeScript import must pass. This prevents an unconditional
exit-zero wrapper from receiving compile credit. It does not prove that a
manifest is complete, that a checker is non-adversarial, or that its policy is
appropriate for the target repository; those remain explicit human-audit
responsibilities.

## Failure and stopping policy

- A genuine model non-answer is an intention-to-run score of zero.
- Confirmatory generation follows its committed preregistration: infrastructure
  errors and forced-search shortfalls are retained outside the successful
  population and retried only under the identical immutable run contract.
- Smoke and exploratory generation are stricter one-shot screens. A durable
  per-cell attempt intent is written before provider contact. Any persisted
  nonfinal attempt, orphaned intent, or partial intent write is terminal for
  that cell; rerunning it could select a luckier completion. Only an
  interruption before intent writing begins is resumable.
- Judge requests use the same pre-contact principle for every grading tier.
  Once a judge intent exists, that cell can produce either its one final grade
  or one terminal failed-judge audit, never a new judging session after a crash.
- Serialized tool syntax after exhausted format repair is a non-answer, not an
  invented natural-language answer.
- Invalid or stale grades are fatal. They are never silently overwritten or
  aggregated.
- A partial report is available only as a conspicuously labelled legacy
  diagnostic and cannot be compared to paper results or promoted to
  claim-ready status.
- Study, inference, and judge failures do not justify changing the hypothesis,
  population, or analysis after looking at outcomes.

## Current prerequisites and deliberate blockers

No API, model, GPU, benchmark, Slurm, container, or external research run was
performed during the integrity hardening pass. Offline unit/static validation
was used only to check code and checked-in artifacts.

The current code mechanically enforces these prerequisites for the relevant
claim-ready path:

- commit the reviewed hardening work; non-smoke research paths require a clean
  recorded source tree;
- create and commit the canonical two-arm preregistration after that baseline
  commit and before either confirmatory arm runs;
- refresh the pinned environments with `scripts/setup.sh` on an appropriate
  compute node;
- configure and hash an absolute Apptainer executable and Python SIF before
  Python answers can receive contained-execution credit;
- configure and hash the complete TypeScript checker bundle described above,
  and pass its built-in semantic calibration, before TypeScript answers can
  receive compile credit (tree-sitter is syntax-only);
- pre-populate the exact pinned model revision in the local cache; the launcher
  then performs stable-descriptor inventory plus pre-load/post-readiness cache
  equality checks and records the vLLM runtime, CUDA toolkit, allocated
  GPUs/driver, server count, and tensor parallelism automatically;
- provide the selected external judge credential for confirmatory grading; the
  grader refuses to make a request when its key is unavailable, pins OpenAI
  grading to `https://api.openai.com/v1`, and does not honor an ambient endpoint
  override;
  and
- create and snapshot the independent human-audit protocol before selfquiz
  round 1.

These are research and reconstruction recommendations, not current software
gates:

- archive hash-pinned distribution artifacts when byte-identical Python
  package reconstruction is required; the vLLM environment attests the
  installed `RECORD`-declared bytes but does not preserve the original wheel
  archives or every host/runtime byte; and
- use a genuinely new study curriculum and a genuinely held-out question
  population for confirmation after adaptive method selection; the reused
  public questions can support only an explicitly labeled replication or
  diagnostic analysis, not a fresh confirmatory result.

Without the mechanically enforced prerequisites, the relevant strict stage is
expected to stop rather than produce a persuasive-looking but invalid number.
Following the recommendations is still necessary for the strongest
reproducibility and publication claim even when software checks pass.

`.env`, if present, must be a regular, non-symlink file owned by the current
user with mode `0600`. The loader accepts only simple `KEY=VALUE` records and
does not execute shell syntax. Because this repository's `.env` was previously
more broadly readable, rotate any key that was ever valid there before future
use.

## Repository map

| Path | Purpose |
|---|---|
| `docs/paper.md` | source paper; scientific target and metric definitions |
| `docs/preregistration.md` | exact confirmatory contract, schema, and two-commit procedure |
| `data/` | public benchmark questions, gold answers, rubrics, and evidence |
| `corpora/` | exact upstream source snapshots used for study and evidence validation |
| `studybench/` | runners, study method, grading, reporting, paired comparison, provenance, and safety checks |
| `scripts/` | pinned setup and Slurm entry points with strict argument/allocation validation |
| `experiments/` | chronological protocol, result, interpretation, and audit record |
| `studies/` | namespaced forced-50 cheatsheet studies |
| `runs/`, `grades/`, `reports/`, `comparisons/`, `screen-comparisons/` | evaluation, grading, strict comparison, and nonclaim local-screen artifacts |
| `study-selfquiz/studies/` | namespaced selfquiz construction and human-audit artifacts |
| `study-graph/studies/` | namespaced deterministic static-neighborhood study artifacts |
| `tests/` | offline research-integrity and boundary tests |

Artifact roots such as `studies/`, `study-graph/studies/`, `reports/`,
`comparisons/`, `screen-comparisons/`, and `preregistrations/` are created on
first use and may be absent in a fresh checkout.

`CLAUDE.md` and root `AGENTS.md` are intentionally byte-for-byte identical and
state the repository's operating and research-integrity rules.

## Interpretation rules

Use paired, same-harness comparisons whenever possible. Report estimates,
uncertainty, populations, failure counts, evidence mode, judge, and all adaptive
decisions. Do not turn a high cell, a point estimate, or a confidence interval
that includes zero into a finding. The 50 questions and their rubrics are
public, repeated across many adaptive arms, and therefore cannot substitute for
a fresh hidden confirmatory set. Failed success criteria, inconclusive results,
and invalidated predictions are first-class outcomes and should remain visible;
do not strengthen them into proof of no effect, parity, or a general negative
claim.
