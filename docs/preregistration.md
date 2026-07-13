# Confirmatory preregistration contract

Claim-ready evaluation is deliberately stricter than merely writing down a
hypothesis. One immutable JSON document binds both arms and freezes generation,
grading, analysis, failure, and stopping choices before either evaluation arm
runs. Smoke and explicitly exploratory runs do not need this contract, but
their artifacts cannot become claim-ready later.

## Required sequence

1. Finish exploratory method development and, for a note intervention, finish
   construction and any required independent human audit. Select the exact
   content-addressed note bytes.
2. Review the implementation and protocol documentation. Commit that state.
   This commit is the preregistration document's `source_commit`.
3. Create exactly one canonical `preregistrations/<preregistration_id>.json`
   file. Its two arms must name the future control and treatment run IDs and
   their exact note hashes (`null` for an arm with no note).
4. Commit only the new direct `preregistrations/*.json` file or files. Do not
   edit or rename a preregistration after its introducing commit.
5. Run both arms from that clean commit, passing the same preregistration path
   and the appropriate `control` or `treatment` role.

The validator requires `source_commit` to be an ancestor of the execution
commit and permits only direct `preregistrations/*.json` changes between them.
This two-commit design avoids asking a commit to contain its own hash while
still proving that code and protocol files did not change after registration.
Any implementation, dataset, script, README, paper, human-audit protocol,
experiment note, or other research-document change requires a new baseline
commit and a new preregistration ID.

## Exact schema

The supported schema is intentionally narrow. The JSON object has exactly the
fields illustrated below; values shown in angle brackets must be replaced.
The actual file must use the repository's canonical JSON encoding: UTF-8,
sorted object keys, no insignificant whitespace, and one final newline.

```json
{
  "analysis_policy": {
    "bootstrap_replicates": 10000,
    "bootstrap_seed": 45001,
    "confidence_interval": "paired_two_stage_question_then_rollout_percentile_95",
    "multiplicity_policy": "single_preregistered_primary_no_adjustment",
    "primary_estimand": "treatment_minus_control",
    "primary_metric": "expertise_lenient"
  },
  "arms": {
    "control": {
      "note_sha256": null,
      "run_id": "control-dspy-001"
    },
    "treatment": {
      "note_sha256": "<64-lowercase-hex-note-hash>",
      "run_id": "treatment-dspy-001"
    }
  },
  "corpus_commit": "9cdb0aac28b2a04b064e40697ccd301872cf6a43",
  "evaluation": {
    "budgets": [
      "direct",
      "k5",
      "k20",
      "k20f"
    ],
    "harness": "dspy.ReAct",
    "master_seed": 44001,
    "model": "openai/Qwen/Qwen3.5-9B",
    "model_revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    "rollouts": 6,
    "sampling": {
      "extra_body": {
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "top_k": 20
      },
      "max_tokens": 32768,
      "presence_penalty": 1.5,
      "temperature": 1.0,
      "top_p": 0.95
    },
    "seed_group": "paired-dspy-001",
    "seed_namespace": "dspy-react"
  },
  "failure_policy": {
    "forced_short": "invalid_until_retried",
    "infrastructure_error": "invalid_until_retried",
    "model_no_answer": "intention-to-run_zero"
  },
  "grading_policy": {
    "claim_scoring": "binary_0_1",
    "evidence_mode": "whole_files",
    "grader": "openai",
    "judge_effort": "high",
    "judge_model": "gpt-5.4",
    "question_scoring": "weighted_claim_sum"
  },
  "hypothesis": "<directional hypothesis fixed before evaluation>",
  "intervention": "<exact treatment-minus-control description used by compare>",
  "preregistration_id": "paired-dspy-001",
  "question_bundle_sha256": "<64-lowercase-hex-question-bundle-hash>",
  "schema_version": 1,
  "source_commit": "<pre-preregistration-implementation-commit>",
  "stopping_policy": {
    "interim_looks": 0,
    "population": "complete_manifest_grid",
    "stopping_rule": "no_outcome_dependent_stopping"
  },
  "task": "dspy"
}
```

Use `studybench.integrity.canonical_json_bytes` to serialize the completed
Python object rather than manually minifying it. The filename must equal the
`preregistration_id`, and the file must be introduced once and remain unchanged
in Git history.

The selected grader determines the judge model (`openai` means `gpt-5.4`;
`fugu` means `fugu`). Evidence mode is either `whole_files` or
`excerpt_evidence`. The implemented confirmatory analysis currently supports
one primary estimand only: treatment minus control on `expertise_lenient`, with
the exact paired two-stage percentile interval declared above and no interim or
outcome-dependent stopping.

## Execution and downstream enforcement

Both launch commands must pass the same file:

```bash
SB_PREREGISTRATION=preregistrations/paired-dspy-001.json \
SB_PREREGISTRATION_ROLE=control ... sbatch scripts/react.sbatch

SB_PREREGISTRATION=preregistrations/paired-dspy-001.json \
SB_PREREGISTRATION_ROLE=treatment ... sbatch scripts/react.sbatch
```

Generation snapshots the exact preregistration bytes into each run and checks
the arm's run ID, note hash, task, corpus, question bundle, harness, model and
revision, sampling, seed policy, budgets, rollouts, and failure policy. Grading
revalidates that snapshot and refuses any grader, evidence mode, effort, or
judge mismatch before contacting a provider. Reporting repeats the check.
Comparison requires the two reports to carry the same document with opposite
roles, requires its intervention description and bootstrap settings to match
the preregistration exactly, and includes the hypothesis and frozen policies in
the immutable comparison artifact. Claim-ready grading, reporting, and
comparison additionally require the current clean Git commit and complete
research-source byte inventory to equal the source record captured by the run;
changing judge, score, report, or comparison code after seeing generation
outcomes requires a new preregistration and new runs.

Cross-allocation retries do not rewrite this contract. Each launch gets its own
content-addressed environment snapshot; a retry is accepted only when the new
environment differs from the baseline in declared allocation/transport
nuisances and each resulting episode binds the snapshot that actually produced
it.
