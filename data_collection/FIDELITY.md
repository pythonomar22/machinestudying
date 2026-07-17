# Fidelity log: what is the paper's, what is ours

We replicate the StudyBench coding-suite construction pipeline (Machine
Studying paper, Appendix A.1) with one declared substitution: the source.
This file is the running ledger, per stage, of (a) what we copied exactly
from the paper and (b) every inference we had to make because the paper
does not specify it — each inference is a potential source of discrepancy
with the authors' actual pipeline. It grows one section per stage file.

## Principles: what the pipeline may know

The studying paradigm evaluates on a hidden task, so this pipeline must be
buildable without knowledge of StudyBench. Two rules:

1. **Priors must be general.** Every design choice must be defensible as a
   claim about studying *any* codebase (e.g., "user questions are good
   study anchors", "expertise organizes around library capabilities, not
   the genre in which friction is reported", "study questions must be
   groundable in the study corpus — the corpus is the studier's whole
   world at study and answer time"), never as knowledge of this test set
   (its topics, labels, register, or questions).
2. **The test set may remove or measure, never add or steer.** Allowed:
   final transfer evaluation; dropping generated questions that are too
   similar to test questions (decontamination); post-hoc similarity audits
   that are reported, not optimized against. Forbidden: test labels or
   questions appearing in any construction prompt, topic selection by test
   taxonomy, register-matching against test questions as a design target.

Violation found and fixed on 2026-07-17: the first version of the Stage-2
labeling prompt showed StudyBench's Table-3 DSPy labels as style examples.
It was rewritten to encode the general capability-not-genre prior with
style examples from unrelated software domains, and the stage was rerun
(the leak could only have influenced label *naming*, not cluster
membership, but the relabeled artifacts replace the old ones entirely).

## Scope note: SmallDSPy vs the session source (2026-07-17)

SmallDSPy was built for cheap iteration on studying baselines: the subset
of Study-DSPy test questions (5 of 30) answerable from the smallest code
span, giving a 66-file study corpus — `dspy/predict`, `dspy/adapters`,
`dspy/primitives`, `tests/predict` only. Our seed sessions, however, come
from issues about the WHOLE codebase: measured against the 66-file scope,
roughly 60-70% of seed mass concerns code that is not in it
(`clients/` backends: 70 sessions; optimizers/`teleprompt`: 16; docs
site: 23; plus parts of other topics). Self-quizzing a SmallDSPy-corpus
studier on such questions is not merely off-distribution - the studier
cannot even read the relevant files.

Decision: candidate generation runs corpus-general - all topics, with the
generator given the FULL repository + docs at the pinned commit (which is
also the paper's literal Stage-3a setup, and the paper's actual Study-DSPy
corpus is the full `dspy/` + `tests/`). Consequently the self-quizzing
experiment line moves to the full-DSPy setting (study corpus = full repo,
test = all 30 Study-DSPy questions), which requires one-time no-study and
cheatsheet baselines on the full corpus; the existing SmallDSPy baselines
remain valid for the cheatsheet replication but are not paired with this
line. Each generated triple is additionally tagged with whether its
code_evidence lies inside the 66-file SmallDSPy scope, preserving the
option of a fast small-scope loop if enough in-scope questions accumulate.

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

Winner (mechanism-oriented embeddings, 2026-07-17):
`n_neighbors=10, min_cluster_size=15, min_samples=5` (DBCV 0.192,
7 clusters, 29.8% noise, sizes 70/36/35/23/17/16/15). The only
higher-DBCV config is a degenerate 290/12 two-way split, excluded by the
constraints. DBCV runs lower than under the earlier intent-oriented
instruct (0.225) — expected: genre has strong lexical cues ("[Bug]...")
that make clusters artificially separable, and the mechanism axis we
actually want is subtler.

The paper's 30-representative cap for labeling is also dropped, by the
same logic: our clusters hold only 218 sessions in total (largest 72), so
GPT-5.4 reads EVERY member of a cluster before naming it — the
"representative selection" approximation (and the inference of how
representatives would be chosen) disappears entirely.

**Our inferences (each a potential discrepancy):**

| # | Paper says | We had to decide | Our choice |
|---|---|---|---|
| 1 | "a domain-aware prefix prompt" | its wording | ours, encoding the capability-not-genre prior: represent each question by the library capability it concerns, not the kind of report it is (recorded in `3_embeddings_index.json`) |
| 2 | (nothing) | embedded text + truncation | cleaned `question_text`, first 4,000 chars |
| 3 | (nothing) | UMAP metric / seed | cosine; `random_state=20260716` (paper unseeded, so exact reproduction of their partition is impossible in principle) |
| 4 | (nothing) | per-question truncation shown to the labeler | first 1,200 chars of each member's `question_text` |
| 5 | (no labeling prompt published — Stage 2's prompt is NOT in the appendix, unlike A.2/A.3/A.4/A.5) | the prompt | ours: mechanism-anchored behavioral labels; genre labels ("bug reports") explicitly disqualified; snake_case style examples drawn from unrelated software domains; coherent=false when a cluster is united only by genre. The first version leaked StudyBench Table-3 labels as examples — see Principles; fixed and rerun |
| 6 | six clusters selected for DSPy | what noise points get | `topic: null` (90 sessions); no forced assignment |

**Corpus-groundability screen (added 2026-07-17, applied per session
BEFORE clustering).** The paper enforces, at generation and critic time,
that questions be answerable from the code roots alone (its "no
documentation-only questions" hard bans exist because the test-taker's
world is source + tests). The same constraint binds our studier for a
corpus-conditioned reason — during self-quizzing it must answer these
questions inside the corpus — so we state it as a general prior
(Principles #1) and apply it as a per-session screen ahead of clustering:
GPT-5.4 (effort `low`, prompt in the script, every verdict + rationale in
`3_groundability.json` / `3_screening_log.jsonl`) judges whether each
session's friction is resolvable from source + tests. **195 of 302
sessions pass; 107 screened out** (dependency/env breakage, external-API
feature requests, docs-website content, repo process, hosted services,
research-methodology questions). Clustering, labeling, and all downstream
seeding use the groundable subset only. Applying the prior per-session
rather than as a post-hoc cluster flag lets formerly grab-bag-bound
sessions re-cluster by capability. The paper's own late enforcement is
still inherited verbatim in Stage 3a/3b. The labeler retains a
cluster-level `corpus_groundable` flag as a redundant sanity check.

**Result (2026-07-17, screen + mechanism instruct + whole-cluster
labeling):** sweep winner on the 195-session subset is
`n_neighbors=25, min_cluster_size=15, min_samples=3` (DBCV 0.219, stable
across mcs 15-20; top-DBCV rows are degenerate 182/13 splits rejected by
the constraints). Four topics, all coherent and groundable —
module_composition_and_runtime_interoperability (42: Signatures,
Predictors, ReAct agents, typed fields, save/load state),
lm_provider_integration_and_configuration (35),
prompt_compilation_customization_and_inspection (32),
evaluation_metrics_and_parallelism (22); 64 noise. The old grab-bag
dissolved as intended: its runtime-mechanism members re-clustered into
module_composition. Still no dedicated agents/tools cluster (v1 finding,
git `47c3363`, persists); such sessions now sit inside module_composition.
Label wording shifts across reruns (the labeler is not seeded); the
partition itself is deterministic given the screen verdicts (which are a
one-time LLM judgment, archived).
