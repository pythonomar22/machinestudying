# Machine Studying replication and method research

This repository studies whether a model can use a declarative code corpus to
become more effective on later coding questions. The corpus remains available
at inference time. Studying therefore means changing the model system before
evaluation—such as by constructing a note—not hiding the source and testing
memorization.

The current honest result is narrower than a successful new studying method.
Under the local `dspy.ReAct` harness matching the author-confirmed ReAct
mechanics, none of the tested self-quizzing or static-note variants beat the
cheap cheatsheet produced by 50 forced ReAct study steps with a paired 95%
confidence interval excluding zero. The pre-registered self-quizzing success
criterion failed. Hybrid2 is incomplete, and hybrid3 is not a fresh DSPy
replication because all 75 generated DSPy quiz questions repeated the first
pipeline. These are useful negative and diagnostic results, not evidence that
studying or weight updates cannot work in general.

The detailed paper interpretation, dataset inventory, experiment ledger,
artifact audit, supported claims, and defects are in
[experiments/008-repository-and-artifact-audit.md](experiments/008-repository-and-artifact-audit.md).
That audit is the authoritative project handoff.

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
| Study-DSPy | 30 public questions, 143 claims, 183 evidence spans; DSPy at `9cdb0aac28b2a04b064e40697ccd301872cf6a43` | native and faithful base/cheatsheet runs; four selfquiz rounds; select, usage, hybrid, summary, hybrid2, hybrid3 arms | useful historical evidence; no new method has a robust positive paired effect; hybrid3 DSPy study set is not fresh |
| Study-OpenClaw | 20 public questions, 100 claims, 111 evidence spans; OpenClaw at `da228660306b55a9cce3b973946f3aacfc515848` | the same local static-note families | faithful base row tracks the paper reasonably well; small sample and no contained TypeScript execution currently limit claims |
| Generated selfquiz material | archived and fresh round artifacts, including internal train/dev records | error-delta note construction and cumulative internal dev exams | generated study/development material, not external ground truth; future claim-ready use requires the complete pre-registered human audit |
| Study-Literature | not present | none | paper discussion only; no local result |
| CPT(code), CPT(doc), SFT+OPSD | no local implementation | none locally | paper baselines only; not reproduced here |

The complete per-arm episode, grade, result, and reproducibility table is in
§3 of the repository audit. Historical Markdown numbers are preserved as
historical records; the hardening work does not retroactively certify them.

## Claim-ready artifact lifecycle

A new confirmatory comparison is valid only if every stage below succeeds.
The implementation fails closed when an identity or completeness check fails.

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
   failed attempt, never a partial grade. There are at most two judge attempts.
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
   rebuilt from both reports. Missing generation or accepted-judge provider
   fingerprints make it diagnostic rather than claim-ready. Exploratory
   observations remain labelled exploratory and do not become confirmatory
   through post-hoc reporting.

Never replace or edit a completed claim-ready artifact. Use a new ID for a new
study, run, regrade, report, or comparison. A retained lock file is coordination
metadata, not a result; an active lock prevents duplicate concurrent work.

## Future command templates

These templates document the intended handoff; none was executed during the
hardening pass. Run setup, serving, generation, selfquiz, external-API grading,
reporting, and comparison from an appropriate Slurm compute allocation under
the repository's cluster policy. Generation and selfquiz use local vLLM
servers. Grading does not use that local server, but it still contacts the
selected external judge API and requires its provider key. `scripts/setup.sh`
may clone corpora and download Python packages.

First refresh the pinned environments, then perform a one-question diagnostic
before spending on a full population:

```bash
scripts/setup.sh
.venv-vllm/bin/hf download Qwen/Qwen3.5-9B \
  --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a

SB_TASKS=dspy SB_RUN_ID=smoke-dspy-001 SB_RUN_SEED=41001 \
SB_SEED_GROUP=smoke-pair-001 SB_SMOKE=1 SB_LIMIT=1 SB_ROLLOUTS=1 \
SB_BUDGETS=direct sbatch scripts/react.sbatch
```

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

A forced-50 cheatsheet study and a selfquiz study are distinct methods. Give
each a fresh ID. The selfquiz protocol must exist before round 1:

```bash
SB_TASKS=dspy SB_STUDY=1 SB_STUDY_ID=cheatsheet-dspy-001 \
SB_RUN_SEED=42001 sbatch scripts/react.sbatch

SB_TASKS=dspy SB_ROUND=1 SB_STUDY_ID=selfquiz-dspy-001 \
SB_STUDY_SEED=43001 SB_AUDIT_PROTOCOL=protocols/blind-audit-001.json \
sbatch scripts/selfquiz.sbatch

.venv-dspy/bin/python -m studybench.selfquiz \
  --task dspy --round 1 --study-id selfquiz-dspy-001 --seed 43001 \
  --promote-human-audit audits/selfquiz-dspy-001-r1.json
```

Only the final promotion command is offline. It must use a genuinely completed
independent audit; creating a syntactically passing declaration is not an
audit. Subsequent selfquiz rounds reuse the same study ID and seed, increment
`SB_ROUND`, and do not accept a new protocol.

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

Grade and report both arms with the same explicit evidence mode, judge effort,
grader selection, and fresh grade ID. `GRADER_MODEL=openai` selects GPT-5.4 and
requires `OPENAI_API_KEY`; `GRADER_MODEL=fugu` selects the configured Fugu
provider and key. The matching report flag is `--grader openai` or
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
the real TypeScript compiler is pinned. A changed method, population, seed,
evidence mode, judge configuration, or note requires a new namespace.
The comparison description, bootstrap count, and bootstrap seed must equal the
preregistered values exactly; they are not free post-hoc reporting choices.

## Failure and stopping policy

- A genuine model non-answer is an intention-to-run score of zero.
- Infrastructure errors and forced-search shortfalls are failed attempts. They
  are preserved outside the successful population and must be retried under
  the identical immutable run contract before strict reporting.
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
- configure and hash a real TypeScript compiler before TypeScript answers can
  receive compile credit (tree-sitter is syntax-only);
- pre-populate the exact pinned model revision in the local cache; the launcher
  then performs stable-descriptor inventory plus pre-load/post-readiness cache
  equality checks and records the vLLM runtime, CUDA toolkit, allocated
  GPUs/driver, server count, and tensor parallelism automatically;
- provide the selected external judge credential for grading; the grader
  refuses to make a request when its key is unavailable, pins OpenAI grading to
  `https://api.openai.com/v1`, and does not honor an ambient endpoint override;
  and
- create and snapshot the independent human-audit protocol before selfquiz
  round 1.

These are research and reconstruction recommendations, not current software
gates:

- archive hash-pinned distribution artifacts when byte-identical Python
  package reconstruction is required; the vLLM environment attests the
  installed `RECORD`-declared bytes but does not preserve the original wheel
  archives or every host/runtime byte; and
- use a genuinely new study curriculum and, preferably, a new hidden question
  split for the next confirmation, because repeated public questions cannot
  establish a fresh confirmatory result.

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
| `runs/`, `grades/`, `reports/`, `comparisons/` | evaluation, grading, reporting, and paired-comparison artifacts |
| `study-selfquiz/studies/` | namespaced selfquiz construction and human-audit artifacts |
| `tests/` | offline research-integrity and boundary tests |

`CLAUDE.md` and root `AGENTS.md` are intentionally byte-for-byte identical and
state the repository's operating and research-integrity rules.

## Interpretation rules

Use paired, same-harness comparisons whenever possible. Report estimates,
uncertainty, populations, failure counts, evidence mode, judge, and all adaptive
decisions. Do not turn a high cell, a point estimate, or a confidence interval
that includes zero into a finding. The 50 questions and their rubrics are
public, repeated across many adaptive arms, and therefore cannot substitute for
a fresh hidden confirmatory set. Negative results and invalidated hypotheses
are first-class outcomes and should remain visible.
