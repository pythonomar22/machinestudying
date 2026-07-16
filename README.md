# StudyBench baseline and cheatsheet evaluation

This repository does one thing: reproduce the two Qwen3.5-9B conditions from
Table 1 of the Machine Studying paper on Study-DSPy and Study-OpenClaw:

- `baseline`: no study phase;
- `cheatsheet`: 50 forced DSPy ReAct study iterations, followed by evaluation
  with the resulting note prepended to every held-out question.

Both conditions use the same model, repositories, tools, inference budgets,
sampling settings, questions, rollouts, and deterministic evaluation seeds.
Study tokens are recorded but excluded from the evaluation token axis.

## Paper targets

| Condition | DSPy expertise | OpenClaw expertise |
|---|---:|---:|
| baseline | 6.49 | 7.64 |
| cheatsheet | 9.65 | 8.18 |

Each expertise value is computed from four points: direct, voluntary ReAct with
at most 5 or 20 iterations, and exactly 20 ReAct iterations with no early exit.
Each point averages three rollouts over all 30 DSPy or 20 OpenClaw questions.

## Scripts

- `scripts/setup.sh` clones the two exact source snapshots and creates the
  locked grading, DSPy, and vLLM environments with `uv`.
- `scripts/nostudying.sbatch` uses all four allocated L40S GPUs to run the
  no-studying baseline on `data/smalldspy.jsonl` against `corpora/smalldspy`.
- `scripts/grading.sbatch` uses all four allocated L40S GPUs to grade one
  completed run with the pinned local Qwen judge.

Everything else is Python for the three actual stages: study/rollout, grade,
and report.

## Setup

Run setup inside a Slurm compute allocation, not on the login node:

```bash
scripts/setup.sh
.venv-vllm/bin/hf download Qwen/Qwen3.5-9B \
  --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a
```

The source snapshots are fixed to:

- DSPy `9cdb0aac28b2a04b064e40697ccd301872cf6a43`;
- OpenClaw `da228660306b55a9cce3b973946f3aacfc515848`.

Setup refuses to alter a checkout that is at the wrong commit or dirty.

## Smoke tests

Always test the complete path before a full run. A cheatsheet smoke uses two
study iterations rather than 50, so it tests plumbing without pretending to be
a result.

```bash
SB_RUN_ID=smoke-base SB_SEED=20260715 \
SB_BUDGETS=direct SB_ROLLOUTS=1 \
SB_SMOKE=1 SB_LIMIT=1 sbatch scripts/nostudying.sbatch
```

Inspect `logs/slurm/` and `logs/vllm-<job>-*.log` after startup and while the
job is running. `SB_DEBUG=1` retains each model call's prompt, output, and usage
inside the episode JSON.

## Full evaluation

Commit the reviewed code first. Full runs refuse a dirty working tree. Use the
same seed for the two conditions and run them on the same GPU class:

```bash
SB_RUN_ID=smalldspy-nostudy-20260715 SB_SEED=20260715 \
sbatch scripts/nostudying.sbatch
```

The cheatsheet job studies each repository before loading its StudyBench
questions. It saves the complete study trajectory and exact note under
`runs/table1-cheat/<task>/`; evaluation still has access to the same three
repository tools.

## Grade and report

Table 1 uses lenient grading. Following Jacob Li's clarification, every rubric
claim is binary (`0` or `1`) and the question score is the pure weighted claim
sum. The compilation and core-conjunctive zero gates apply only to strict
grading, so they are deliberately absent here. GPT-5.4 and Fugu share the paper
prompt, whole-file evidence, validation, and artifact format:

```bash
export OPENAI_API_KEY=...
uv run --frozen python -m studybench.grade runs/table1-base --judge gpt

export SAKANA_API_KEY=...
uv run --frozen python -m studybench.grade runs/table1-base --judge fugu
```

For offline diagnostic grading, launch the pinned Qwen3.5-9B judge on two TP=2
replicas. Thinking is enabled with a hard 10,000-token reasoning budget and a
five-minute timeout per request:

```bash
sbatch scripts/grading.sbatch runs/smalldspy-nostudy-20260715
```

Grades and the per-run report are written under
`grades/<run-id>/<grade-id>/<task>/`. Local Qwen and Fugu are diagnostic
proxies; paper-comparable reporting requires GPT-5.4. Calibrate local scores
manually before using them for exact method comparisons.

```bash
uv run --frozen python -m studybench.report \
  --baseline-run table1-base --cheatsheet-run table1-cheat
```

The reporter rejects partial or stale populations and refuses to compare arms
whose model, runtime, GPU class, sampling, tools, questions, rollouts, or seeds
differ.

## Exact local interpretation

- Direct is `dspy.Predict`; all other budgets use DSPy's text-field `ReAct`.
- Forced runs catch `finish`, keep that iteration in the trajectory, return a
  continue-searching observation, and continue to the exact iteration count.
- Tools are case-sensitive regex `grep`, recursive `glob`, and ranged
  `read_file`; reads are capped at the author-confirmed 200 lines.
- Local, previously established caps are 50 grep matches, 200 glob paths, and
  25,000 observation characters. These caps were not specified in the paper.
- Sampling is temperature 1.0, top-p 0.95, top-k 20, min-p 0, presence penalty
  1.5, repetition penalty 1.0, and at most 32,768 generated tokens per model call.
- Expertise is Appendix C's best-so-far weighted area: a 3,000-token anchor,
  zero before the first point, and the last point held through the tail.

The study prompt and note-prefix wording were not provided by the paper's
authors; both are fixed and recorded by this implementation. The public release
contains the questions and rubrics, so this is a reproducible open evaluation,
not a secret held-out benchmark. See `docs/results.md` for the historical local
measurements and their limitations.

## Repository map

```text
data/          StudyBench questions, rubrics, and evidence
docs/          paper, author correspondence, and concise result record
experiments/   reproducible experiment records and interpretations
scripts/       setup.sh, nostudying.sbatch, grading.sbatch, and the vLLM lock
studybench/    study/rollout, repository tools, grading, and reporting
```

Generated `runs/`, `grades/`, and `logs/` are ignored. Keep only complete runs
that support a result worth reporting.
