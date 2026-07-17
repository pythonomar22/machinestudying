# /// script
# requires-python = ">=3.12"
# dependencies = ["datasketch==1.6.5", "langdetect==1.0.9"]
# ///
"""s1: paper Stage 1 — filter the issue snapshot into the seed pool.

Paper (Appendix A.1, Stage 1): conversations are filtered by length, language
(English only), and question form (first substantive turn begins with an
interrogative or imperative such as how, what, why, can, does, explain, show,
or help), then exact-deduplicated by text and near-deduplicated with MinHash
(num_perm=128, Jaccard threshold 0.7 over question shingles).

Unpublished parameters we had to fix ourselves (see README fidelity table):
closed issues only (the paper's OpenClaw recipe), a created-at cutoff at the
pinned corpus commit date, the 30..20000-char length window, the expanded
interrogative/imperative word list, and word 3-shingles for MinHash.

Usage: uv run data_collection/s1_filter.py
"""

from __future__ import annotations

import re
import subprocess

from datasketch import MinHash, MinHashLSH
from langdetect import DetectorFactory, LangDetectException, detect

from common import ARTIFACTS, ROOT, normalize, read_jsonl, shingles, write_json, write_jsonl

DetectorFactory.seed = 0

MIN_CHARS, MAX_CHARS = 30, 20_000
NUM_PERM, JACCARD_THRESHOLD = 128, 0.7
MAX_COMMENTS, MAX_COMMENT_CHARS = 5, 2_500

# Paper's list is open-ended ("such as how, what, why, can, does, explain,
# show, or help"); this is our documented expansion of it.
INTERROGATIVES = {
    "how", "what", "why", "when", "where", "which", "who", "whose", "whom",
    "can", "could", "does", "do", "did", "is", "are", "was", "were", "will",
    "would", "should", "shall", "has", "have", "had", "am", "may", "might",
}
IMPERATIVES = {
    "explain", "show", "help", "describe", "tell", "give", "provide",
    "write", "make", "create", "add", "support", "allow", "implement",
    "fix", "clarify", "document", "consider", "suggest", "share", "expose",
    "enable", "let", "please",
}
LEAD_WORDS = INTERROGATIVES | IMPERATIVES
# A short salutation line is not a "substantive" turn opener; skip it.
GREETINGS = {"hi", "hello", "hey", "greetings", "thanks", "thank", "dear"}
GREETING_MAX_CHARS = 60


def corpus_commit_date() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT / "corpora/smalldspy"), "show", "-s", "--format=%cI", "HEAD"],
        check=True, capture_output=True, text=True, timeout=30,
    ).stdout.strip()


def clean(markdown: str) -> str:
    """Markdown/issue-template noise -> prose (code fences become [code])."""
    text = re.sub(r"<!--.*?-->", " ", markdown or "", flags=re.DOTALL)
    text = re.sub(r"```.*?(```|\Z)", " [code] ", text, flags=re.DOTALL)
    text = re.sub(r"<[^>\n]{1,80}>", " ", text)
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith(("#", ">", "|"))
    ]
    return re.sub(r"[ \t]+", " ", "\n".join(lines)).strip()


def first_word(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip().strip("*_`").lstrip("[](). ")
        if not stripped or stripped == "[code]":
            continue
        match = re.match(r"[A-Za-z']+", stripped)
        word = match.group().lower() if match else ""
        if word in GREETINGS and len(stripped) <= GREETING_MAX_CHARS:
            continue
        return word
    return ""


def minhash(shingle_set: set[str]) -> MinHash:
    signature = MinHash(num_perm=NUM_PERM)
    for shingle in shingle_set:
        signature.update(shingle.encode("utf-8"))
    return signature


def main() -> None:
    raw = read_jsonl(ARTIFACTS / "issues_raw.jsonl")
    cutoff = corpus_commit_date()
    funnel = {"raw": len(raw)}
    dropped: dict[str, list[int]] = {}

    def drop(stage: str, number: int) -> None:
        dropped.setdefault(stage, []).append(number)

    survivors = []
    for issue in sorted(raw, key=lambda row: row["createdAt"]):
        number = issue["number"]
        if issue["state"] != "CLOSED":
            drop("state_not_closed", number)
            continue
        if issue["createdAt"] > cutoff:
            drop("after_corpus_commit", number)
            continue
        body = clean(issue["body"])
        question_text = f"{issue['title'].strip()}\n\n{body}".strip()
        if not MIN_CHARS <= len(question_text) <= MAX_CHARS:
            drop("length", number)
            continue
        try:
            language = detect(question_text[:1500])
        except LangDetectException:
            language = "unknown"
        if language != "en":
            drop("not_english", number)
            continue
        if first_word(issue["title"]) not in LEAD_WORDS and first_word(body) not in LEAD_WORDS:
            drop("question_form", number)
            continue
        survivors.append(
            {
                "number": number,
                "title": issue["title"],
                "question_text": question_text,
                "created_at": issue["createdAt"],
                "closed_at": issue["closedAt"],
                "url": issue["url"],
                "labels": issue["labels"],
                "author": issue["author"],
                "comments": [
                    {"author": comment["author"], "body": comment["body"][:MAX_COMMENT_CHARS]}
                    for comment in issue["comments"]
                    if comment["author_type"] == "User" and comment["author"] != issue["author"]
                ][:MAX_COMMENTS],
            }
        )
    funnel["closed_dated_length_english_form"] = len(survivors)

    seen_exact: set[str] = set()
    exact_unique = []
    for row in survivors:
        key = normalize(row["question_text"])
        if key in seen_exact:
            drop("exact_duplicate", row["number"])
            continue
        seen_exact.add(key)
        exact_unique.append(row)
    funnel["exact_unique"] = len(exact_unique)

    lsh = MinHashLSH(threshold=JACCARD_THRESHOLD, num_perm=NUM_PERM)
    near_pairs = []
    seed_pool = []
    for row in exact_unique:  # already oldest-first; earliest of a group wins
        signature = minhash(shingles(row["question_text"]))
        matches = lsh.query(signature)
        if matches:
            near_pairs.append({"kept": matches[0], "dropped": row["number"]})
            drop("near_duplicate", row["number"])
            continue
        lsh.insert(row["number"], signature)
        seed_pool.append(row)
    funnel["seed_pool"] = len(seed_pool)

    write_jsonl(ARTIFACTS / "seed_pool.jsonl", seed_pool)
    write_json(
        ARTIFACTS / "filter_report.json",
        {
            "corpus_commit_cutoff": cutoff,
            "parameters": {
                "min_chars": MIN_CHARS,
                "max_chars": MAX_CHARS,
                "minhash_num_perm": NUM_PERM,
                "jaccard_threshold": JACCARD_THRESHOLD,
                "shingle_words": 3,
                "lead_words": sorted(LEAD_WORDS),
            },
            "funnel": funnel,
            "dropped_counts": {stage: len(numbers) for stage, numbers in dropped.items()},
            "dropped_issue_numbers": dropped,
            "near_duplicate_pairs": near_pairs,
        },
    )
    print(f"funnel: {funnel}")
    print(f"wrote {len(seed_pool)} seeds to {ARTIFACTS / 'seed_pool.jsonl'}")


if __name__ == "__main__":
    main()
