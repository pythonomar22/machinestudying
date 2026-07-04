---
license: cc-by-4.0
language:
- en
tags:
- code
- question-answering
- llm-evaluation
- rubric-grading
- agents
task_categories:
- question-answering
- text-generation
size_categories:
- n<1K
configs:
- config_name: dspy
  data_files: dspy.jsonl
- config_name: openclaw
  data_files: openclaw.jsonl
---

# studybench

**studybench** is a small, high-effort benchmark of **expert-level coding questions** about real
open-source codebases, each paired with a **gold answer** and a **weighted, source-grounded grading
rubric**. The questions ask a model to produce working code that uses a specific library/framework
correctly; the rubric decomposes a correct answer into discrete, checkable claims, each tied to exact
lines of the upstream source.

This release publishes **both the questions and the full rubrics** (nothing is held back), so the
evaluation is fully transparent and reproducible.

## Configs

Pick a subset with the second argument of `load_dataset`:

```python
from datasets import load_dataset

dspy     = load_dataset("jacobli/studybench", "dspy")        # 30 questions
openclaw = load_dataset("jacobli/studybench", "openclaw")    # 20 questions
```

| config | questions | topics | codebase |
|---|---:|---:|---|
| `dspy` | 30 | 6 | [DSPy](https://github.com/stanfordnlp/dspy) |
| `openclaw` | 20 | 4 | [OpenClaw](https://github.com/openclaw/openclaw) |

## Schema

Each row has six fields:

| field | type | description |
|---|---|---|
| `id` | string | stable opaque identifier |
| `topic` | string | coarse category (see below) |
| `question` | string | the task prompt — asks for a self-contained, runnable solution |
| `gold_answer` | string | a reference solution (code) |
| `rubric` | list | weighted claims that define a correct answer |
| `evidence` | list | source excerpts that ground the rubric |

**`rubric`** — a list of claims; weights sum to **100** per question:

```json
{
  "claim_id": "c1",
  "claim_type": "core",          // "core" = essential; "supporting" = secondary
  "weight": 52,                   // integer; the rubric's weights sum to 100
  "statement": "…what must be true of a correct answer…",
  "span_ids": ["s4", "s8"]        // evidence spans grounding this claim
}
```

**`evidence`** — the source excerpts the grader is shown; every `span_ids` value in `rubric`
resolves to one of these `span_id`s:

```json
{
  "span_id": "s4",
  "path": "dspy/teleprompt/gepa/gepa.py",   // path within the upstream repo
  "start_line": 330,
  "end_line": 365,
  "excerpt": "0330:     def __init__(\n0331:         self,\n…"   // line-number-prefixed source
}
```

Excerpts are byte-exact copies of the upstream source at the pinned commits below (each line is
prefixed with its 1-indexed line number, e.g. `0330: `).

### Topics
- **dspy:** `gepa_optimizer_usage`, `prompt_optimization_workflows`, `rag_and_retrieval_pipelines`, `react_agents_and_tools`, `signature_schema_and_pydantic_types`, `evaluation_metrics_and_custom_eval`
- **openclaw:** `model_fallback_and_failover_logic`, `cross_session_channel_context_and_session_behavior_requests`, `memory_core_dreaming_and_promotion_pipeline`, `new_plugin_provider_and_channel_integration_requests`

## How the rubric is used for grading

A judge model is shown the **question**, the **candidate answer**, the **`gold_answer`**, the
**`rubric`**, and the **`evidence`** spans. It scores each claim independently (does the answer
satisfy the claim?), and the question score is the **weight-weighted fraction of satisfied claims**
(0–100). `claim_type` lets you apply an optional **conjunctive gate**: require every `core` claim to
be satisfied or the answer scores 0. The `evidence` excerpts are the *only* code context the judge
needs — grading does not require checking out the repositories.

## Source code & attribution

The `evidence` excerpts and `path` values reference these repositories at fixed commits:

| codebase | repo | commit | license |
|---|---|---|---|
| DSPy | `stanfordnlp/dspy` | `9cdb0aac28b2a04b064e40697ccd301872cf6a43` | MIT |
| OpenClaw | `openclaw/openclaw` | `da228660306b55a9cce3b973946f3aacfc515848` | MIT |

To inspect or extend the evidence, check out the corresponding repo at the pinned commit and open
the listed `path` at the given line range.

## Licensing

- **Questions, gold answers, and rubrics** (the original contributions of this dataset) are released
  under **CC-BY-4.0**.
- **Embedded source `excerpt`s** are derived from DSPy and OpenClaw and remain under their respective
  **MIT** licenses; attribution is provided above.

## Notes & limitations

- This is a deliberately small, expert-curated set (50 questions total), not a large-scale benchmark.
- Because both questions and rubrics are public, treat results as an **open** (non-held-out)
  evaluation; models may be trained on this content.
- The benchmark is grounded in specific repository snapshots; answers reflect the APIs at the pinned
  commits.
