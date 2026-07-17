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

Every parameter above is also recorded in `artifacts/2_seed_sessions.json`
under `inferred_parameters`, and every dropped session number is listed per
filter so the funnel is fully auditable.
