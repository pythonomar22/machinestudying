# data_collection: StudyBench-style question construction for SmallDSPy

Replicates the StudyBench coding-suite construction pipeline (Machine
Studying paper, Appendix A) to produce our own set of user-anchored DSPy
questions for the self-quizzing studying method. The output,
`data/smalldspy_ourvalidationset.jsonl`, has exactly the released dataset's
fields (`id`, `topic`, `question`, `gold_answer`, `rubric`, `evidence`) and
targets the held-out SmallDSPy topic `react_agents_and_tools`.

## Pipeline

| Stage | Script | Artifacts |
|---|---|---|
| collect | `s0_collect_issues.py` | `artifacts/issues_raw.jsonl`, `collect_manifest.json` |
| Stage 1 filter | `s1_filter.py` | `artifacts/seed_pool.jsonl`, `filter_report.json` |
| Stage 2 embed | `s2a_embed.py` (GPU) | `artifacts/embeddings.npy`, `embeddings_index.json` |
| Stage 2 cluster | `s2b_cluster.py` | `artifacts/clusters.json`, `clusters.png` |
| Stage 3a/3b | `s3_questions.py` | `artifacts/seeds_sampled.json`, `candidates.json`, `finalists.json` |
| Stage 4 rubrics | `s4_rubrics.py` | `artifacts/bundles.json` |
| Stage 5 validate | `s5_validate.py` | `artifacts/validation_report.json`, `review/questions.md`, **`data/smalldspy_ourvalidationset.jsonl`** |

Every OpenAI request/response is appended to `logs/api/<stage>.jsonl`.
Prompt templates live in `prompts.py`: A.2/A.3/A.4 are transcribed verbatim
from the paper; everything else there is marked as our reconstruction.

## How to run

```bash
uv run --frozen python data_collection/s0_collect_issues.py
uv run data_collection/s1_filter.py
srun --overlap --jobid=<JOBID> .venv-vllm/bin/python data_collection/s2a_embed.py
uv run data_collection/s2b_cluster.py
uv run --frozen python data_collection/s3_questions.py
uv run --frozen python data_collection/s4_rubrics.py
uv run --frozen python data_collection/s5_validate.py
```

Requires `GITHUB_PAT` and `OPENAI_API_KEY` in `.env`, and a GPU allocation
for s2a (one L40S, ~2 minutes).

## Fidelity table

| Paper (Appendix A) | Ours | Divergence and reason |
|---|---|---|
| **Sources**: DSPy community QA sessions; OpenClaw closed GitHub issues | stanfordnlp/dspy **closed GitHub issues** (full 1,634-issue snapshot retained; filtering explicit) | DSPy community sessions are not public; we adopt the paper's OpenClaw recipe for DSPy. Added an issue-creation cutoff at the pinned corpus commit date so seeds cannot postdate the study corpus. |
| **Stage 1 filters**: length, English-only, question form ("first substantive turn begins with an interrogative or imperative, such as how, what, why, can, does, explain, show, or help"); exact dedup; MinHash `num_perm=128`, Jaccard 0.7 over question shingles | Same, via `s1_filter.py` | Unpublished parameters fixed by us and recorded in `filter_report.json`: 30–20,000-char window, `langdetect` (seed 0), an expanded interrogative/imperative list, short salutation lines skipped as non-substantive, word 3-shingles, earliest duplicate kept. |
| **Stage 2**: Qwen3-Embedding-8B + domain-aware prefix; UMAP to 10 dims (`n_neighbors=15`); HDBSCAN (`min_cluster_size=30`, `min_samples=5`); GPT-5.4 labels each cluster from 30 representative sessions | Same model (revision `1d8ad4c`), UMAP cosine + fixed seed; **both** paper-literal and pool-scaled (`min_cluster_size=10`) HDBSCAN recorded; GPT-5.4 labels the scaled clusters from the 30 nearest-centroid members | Instruct prefix, UMAP metric/seed, and representative-selection method are unpublished; ours are recorded in the artifacts. The paper's cluster size assumes thousands of sessions; our pool is 302 seeds. |
| **Cluster selection**: six DSPy clusters chosen, incl. `react_agents_and_tools` | Automatic: prefer the cluster whose GPT label matches the target topic. **Measured fact: no react/tools/agents cluster exists in the DSPy issue pool** (see `clusters.json` diagnostics), so selection falls back to label-conditioned keyword seeds (25 issues) | The issue distribution differs from the community-session distribution — ReAct/tools Q&A mass is too small to cluster. The fallback preserves the paper's intent (seeds anchoring the target label) and is fully recorded. |
| **Stage 3a**: GPT-5.4 in Codex at xhigh, full repo + docs (privileged), 20 seed sessions, 12 candidates (template A.2) | GPT-5.4 (`gpt-5.4-2026-03-05`) at xhigh via the Responses API, agentic loop with the studier's exact three tools (`studybench.tools.RepoTools`) over the pinned 66-file SmallDSPy corpus, plus the repo's `docs/` tree at the same commit (privileged); 20 seeds sampled with a fixed seed; template A.2 verbatim | Codex isn't scriptable here; the tool surface is identical to the answering agent's. Scope is the SmallDSPy corpus, not the full repo, because our questions must be answerable from the corpus the studier sees. Loop caps: 80 tool calls, 15k-char observations. Library-specific placeholder values are our reconstructions (marked in `prompts.py`). |
| **Stage 3b**: same model/harness as critic selects 5 finalists (template A.3) | Same harness, template A.3 verbatim, ≤5 finalists | — |
| **Stage 4**: GPT-5.4 turns gold answers into 2–8 atomic claims, core/supporting, weights sum to 100, 1–3 spans/claim, exact lines from numbered dumps, ≤300-line spans (template A.4) | Same, template A.4 verbatim + deterministic validation with up to 3 corrective retries; excerpts materialized byte-exact from the pinned corpus in the released `NNNN: ` format | Response schema is ours (the paper doesn't publish schemas). |
| **Stage 5**: deterministic syntax checker + sandbox; question/answer/rubric optimized jointly until the reference answer scores 100 under its own rubric; human expert review | Sandbox = `.venv-dspy` (dspy at the pinned corpus commit; the 5 released golds must pass as a control); GPT-5.4 self-grade with the exact paper judge contract from `studybench.grade` must reach 100; failures loop through `REVISE_TEMPLATE` (ours) ≤3 rounds; human-review file generated at `artifacts/review/questions.md` | Revision prompt is ours. **Added decontamination** (not needed in the paper, whose output *is* the test set): exact 3-word-shingle Jaccard vs every held-out test question must stay < 0.7, plus cross-bundle near-dedup; the review file lists each question's nearest test question. |

## Honest limitations

- GitHub issues are not the paper's DSPy seed source; the react/tools topic
  had to be selected by keyword fallback, and the 25 selected seeds include
  a few keyword false positives (the A.2 prompt is built for noisy seeds and
  the critic rejects off-label items, but the anchor distribution is
  thinner than the paper's).
- All `{placeholder}` values in the A.2/A.3/A.4 templates (library
  description, ok/not-ok naming bullets, examples, label description) are
  our reconstructions; the offline-DummyLM requirement is inferred from the
  released dataset's uniform style.
- Lenient-grading self-verification (score 100 under own rubric) uses
  GPT-5.4 as judge; judge noise means a bundle could pass with a borderline
  rubric. The review file exists precisely for the human pass.
