# Fidelity log: what is the paper's, what is ours

We replicate the StudyBench coding-suite construction pipeline (Machine
Studying paper, Appendix A.1) with one declared substitution: the source.
This file is the running ledger, per stage, of (a) what we copied exactly
from the paper and (b) every inference we had to make because the paper
does not specify it — each inference is a potential source of discrepancy
with the authors' actual pipeline. It grows one section per stage file.

## Source (file `1_scrape_sessions.py`)

**Theirs:** "a snapshot of real user-question sessions for each library" —
for DSPy, private community QA conversations; for OpenClaw, GitHub closed
issues.

**Ours (declared substitution):** all 1,634 public stanfordnlp/dspy GitHub
issues, one session per issue (opening question + up to 20 comments).

Known consequences of the substitution, measured in the v1 attempt (git
`47c3363`): the issue genre is bug reports / feature requests / setup,
while their community sessions are usage Q&A — their six exam topics do
not emerge as clusters from issue data. Also a role difference: their
sessions' answers came from "a weaker assistant"; our comments are human
(often maintainers). The generator prompt treats both as untrusted hints.

## Stage 1 — filter + dedup (file `2_filter_sessions.py`)

**Copied exactly from the paper:**
- The three filter categories: length, language (English only), question
  form ("the first substantive turn must begin with an interrogative or
  imperative").
- Exact deduplication by question text.
- MinHash near-deduplication with `num_perm=128` and Jaccard threshold
  **0.7** over question shingles.

**Our inferences (each a potential discrepancy):**

| # | Paper says | We had to decide | Our choice |
|---|---|---|---|
| 1 | source is sessions; OpenClaw used *closed* issues | which issue states qualify | closed only, mirroring their OpenClaw source (drops 303 open) |
| 2 | (nothing — their snapshot is contemporaneous with their corpus pin) | whether sessions may postdate the pinned study corpus | drop sessions created after the corpus commit date (2026-03-31); aligns the session epoch with the corpus epoch |
| 3 | "filtered by length" | thresholds | 30–20,000 chars of the cleaned question text |
| 4 | "language (English only)" | detector | `langdetect` 1.0.9, seed 0, on the first 1,500 chars |
| 5 | "first substantive turn" | what that means for an issue | the title, or the first substantive line of the markdown-cleaned body; code fences collapse to `[code]`; headers/quotes/tables/HTML stripped; salutation lines ≤60 chars skipped |
| 6 | "an interrogative or imperative, such as how, what, why, can, does, explain, show, or help" — explicitly open-ended | the full word list | 29 interrogatives + 24 imperatives (listed in the script and echoed into the output manifest); including request verbs (add/support/implement/fix) admits feature requests, consistent with OpenClaw's `_requests` topics |
| 7 | "deduplicate questions by text" | normalization | lowercase + whitespace collapse before comparing |
| 8 | "question shingles" | shingle definition | word 3-grams (character shingles would also be defensible) |
| 9 | (nothing) | which duplicate survives | the earliest-created session of each group |
| 10 | (nothing) | what one embedded/compared "question" is | cleaned `title + "\n\n" + body` |

Every parameter above is also recorded in
`artifacts/2_filter_sessions/2_seed_sessions.json` under
`inferred_parameters`, and every dropped session number is listed per
filter so the funnel is fully auditable.

## Stage 2 — embed, cluster, label (file `3_label_topics.py`)

**Copied exactly from the paper:**
- Embedding model: Qwen3-Embedding-8B, one vector per session's first
  substantive user question.
- UMAP to 10 dimensions.
- HDBSCAN as the clusterer.
- GPT-5.4 assigns a behavioral label per cluster.

**Deliberately re-tuned (not copied), per first principles:** the paper's
UMAP `n_neighbors=15` and HDBSCAN `(min_cluster_size=30, min_samples=5)`
were tuned to their session pool (plausibly thousands of community
sessions); our pool is 302 issues, on which the literal values collapse to
3 coarse clusters (DBCV 0.143 — kept on record in the sweep grid). We
selected parameters with a reproducible sweep (`sweep` phase, grid and
result in `artifacts/3_label_topics/3_cluster_sweep.json`) over n_neighbors x
min_cluster_size x min_samples, scored by HDBSCAN's DBCV
(`relative_validity_`) subject to constraints derived from what the
clusters are FOR:
- every kept cluster must have >= 15 members, so a topic can (nearly)
  supply the paper's 20 sampled seed sessions for generation;
- noise <= 45% of the pool;
- >= 4 clusters (enough topical diversity to be worth labeling).

Winner: `n_neighbors=25, min_cluster_size=10, min_samples=3` (DBCV 0.225,
6 clusters, 27.8% noise, sizes 72/55/41/19/16/15). The identical partition
arises for min_cluster_size in {10, 12, 15} — a stability plateau, so the
choice is not knife-edge. The unconstrained DBCV optimum (0.276) produces
clusters of size 8-9, too small to anchor generation, and n_neighbors=10
degenerates to one blob; both are visible in the grid.

The paper's 30-representative cap for labeling is also dropped, by the
same logic: our clusters hold only 218 sessions in total (largest 72), so
GPT-5.4 reads EVERY member of a cluster before naming it — the
"representative selection" approximation (and the inference of how
representatives would be chosen) disappears entirely.

**Our inferences (each a potential discrepancy):**

| # | Paper says | We had to decide | Our choice |
|---|---|---|---|
| 1 | "a domain-aware prefix prompt" | its wording | the Instruct prefix recorded in `3_embeddings_index.json` |
| 2 | (nothing) | embedded text + truncation | cleaned `question_text`, first 4,000 chars |
| 3 | (nothing) | UMAP metric / seed | cosine; `random_state=20260716` (paper unseeded, so exact reproduction of their partition is impossible in principle) |
| 4 | (nothing) | per-question truncation shown to the labeler | first 1,200 chars of each member's `question_text` |
| 5 | (no labeling prompt published) | the prompt | ours, embedded in the script; asks for snake_case label + description + a coherence flag, with the paper's Table-3 labels shown as style examples |
| 6 | six clusters selected for DSPy | what noise points get | `topic: null` (84 sessions); no forced assignment |

**Result (2026-07-17, whole-cluster labeling):** six topics —
optimizer_metrics_and_async_scaling (15),
setup_runtime_and_integration_troubleshooting (72),
prompt_customization_and_generation_controls (16),
documentation_and_repo_maintenance_contributions (19),
support_for_new_integrations_and_features (55),
custom_backend_integration_and_serving (41); 84 noise. Reading whole
clusters (vs the core-30) broadened some names to match full contents.
Two of six correspond to paper topics (optimizer/eval metrics; prompt
customization) — the tuned, finer clustering recovers structure the
paper-literal parameters missed. A react_agents_and_tools cluster still
does not emerge from issue data (ReAct-related bug reports sit inside the
troubleshooting cluster), consistent with the v1 finding (git `47c3363`).
