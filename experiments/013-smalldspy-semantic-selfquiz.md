# 013 — SmallDSPy semantic self-quizzing screen

## Objective and status

Apply the existing `semantic-selfquiz-v2` algorithm unchanged to the configured
SmallDSPy corpus, then evaluate its final note on the same five SmallDSPy
questions used by experiment 012. This is an adaptive exploratory development
screen, not a confirmatory or publication-ready experiment.

Status before execution: task binding implemented and offline-validated; no
semantic SmallDSPy model call, study artifact, evaluation answer, or score has
yet been produced.

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
- No method component changes for this task port: prompts, sampling, tool
  contracts, seeds, iteration caps, ensembles, train/dev/retest rules,
  distillation, note rendering, readiness gates, and failure policy remain the
  existing `semantic-selfquiz-v2` contracts.
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
| construction smoke | `smoke-smalldspy-semantic-react-20260714b` |
| full four-round study | `smalldspy-semantic-react-r4-20260714a` |
| treatment evaluation | `smalldspy-local-selfquiz-20260714a` |
| paired evaluation seed group | `smalldspy-local-cheatsheet-screen-20260713` |
| evaluation seed | `44001` |
| GPT sensitivity screen | `smalldspy-selfquiz-gpt54-whole-high-20260714a` |

Any failed smoke or full namespace is retained and retired; it is not retried
under the same identity. A full construction begins only from clean, pushed
source after the smoke artifact is manually inspected.

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

No semantic study outcome has yet been produced.
