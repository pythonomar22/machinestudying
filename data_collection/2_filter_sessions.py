# /// script
# requires-python = ">=3.12"
# dependencies = ["datasketch==1.6.5", "langdetect==1.0.9"]
# ///
"""Stage 1 (paper): filter and deduplicate the session snapshot.

Paper, verbatim: "Conversations are filtered by length, language (English
only), and question form: the first substantive turn must begin with an
interrogative or imperative, such as how, what, why, can, does, explain,
show, or help. We then deduplicate questions by text, and near-deduplicate
using MinHash with num_perm = 128 and a Jaccard threshold of 0.7 over
question shingles."

Everything the paper leaves unspecified is an inference of ours; each one
is listed in FIDELITY.md and echoed in this file's output manifest.

Usage:
    uv run data_collection/2_filter_sessions.py

Reads artifacts/1_sessions.json; writes artifacts/2_seed_sessions.json
(manifest with the full filter funnel + the surviving sessions, each with
its cleaned `question_text`).
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from datasketch import MinHash, MinHashLSH
from langdetect import DetectorFactory, LangDetectException, detect

DetectorFactory.seed = 0  # langdetect is otherwise nondeterministic

DC = Path(__file__).resolve().parent
ROOT = DC.parent
INPUT = DC / "artifacts" / "1_sessions.json"
OUTPUT = DC / "artifacts" / "2_seed_sessions.json"

# --- Paper-specified parameters (exact) -----------------------------------
NUM_PERM = 128
JACCARD_THRESHOLD = 0.7

# --- Our inferences (see FIDELITY.md) --------------------------------------
MIN_CHARS, MAX_CHARS = 30, 20_000       # "filtered by length": thresholds ours
SHINGLE_WORDS = 3                        # "question shingles": word 3-grams ours
GREETINGS = {"hi", "hello", "hey", "greetings", "thanks", "thank", "dear"}
GREETING_MAX_CHARS = 60                  # short salutations are not substantive

# The paper's list is explicitly open-ended ("such as how, what, why, can,
# does, explain, show, or help"); this is our documented expansion.
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


def corpus_commit_date() -> str:
    """Pin the session epoch to the study corpus commit (our inference)."""
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
    """First word of the first substantive line (skips short salutations)."""
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


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def shingles(text: str) -> set[str]:
    words = normalize(text).split()
    if len(words) < SHINGLE_WORDS:
        return {" ".join(words)} if words else set()
    return {
        " ".join(words[i : i + SHINGLE_WORDS])
        for i in range(len(words) - SHINGLE_WORDS + 1)
    }


def minhash(shingle_set: set[str]) -> MinHash:
    signature = MinHash(num_perm=NUM_PERM)
    for shingle in shingle_set:
        signature.update(shingle.encode("utf-8"))
    return signature


def main() -> None:
    snapshot = json.loads(INPUT.read_text(encoding="utf-8"))
    cutoff = corpus_commit_date()
    dropped: dict[str, list[int]] = {}

    def drop(stage: str, number: int) -> None:
        dropped.setdefault(stage, []).append(number)

    survivors = []
    for row in sorted(snapshot["sessions"], key=lambda item: item["created_at"]):
        number = row["number"]
        if row["state"] != "CLOSED":
            drop("state_not_closed", number)
            continue
        if row["created_at"] > cutoff:
            drop("after_corpus_commit", number)
            continue
        question_text = f"{row['title'].strip()}\n\n{clean(row['body'])}".strip()
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
        if (
            first_word(row["title"]) not in LEAD_WORDS
            and first_word(clean(row["body"])) not in LEAD_WORDS
        ):
            drop("question_form", number)
            continue
        survivors.append({**row, "question_text": question_text})

    seen_exact: set[str] = set()
    exact_unique = []
    for row in survivors:
        key = normalize(row["question_text"])
        if key in seen_exact:
            drop("exact_duplicate", row["number"])
            continue
        seen_exact.add(key)
        exact_unique.append(row)

    lsh = MinHashLSH(threshold=JACCARD_THRESHOLD, num_perm=NUM_PERM)
    near_pairs = []
    seed_sessions = []
    for row in exact_unique:  # oldest-first: the earliest of a group survives
        signature = minhash(shingles(row["question_text"]))
        matches = lsh.query(signature)
        if matches:
            near_pairs.append({"kept": matches[0], "dropped": row["number"]})
            drop("near_duplicate", row["number"])
            continue
        lsh.insert(row["number"], signature)
        seed_sessions.append(row)

    funnel = {
        "snapshot": len(snapshot["sessions"]),
        "after_filters": len(survivors),
        "after_exact_dedup": len(exact_unique),
        "seed_sessions": len(seed_sessions),
    }
    OUTPUT.write_text(
        json.dumps(
            {
                "source": str(INPUT.name),
                "paper_parameters": {
                    "minhash_num_perm": NUM_PERM,
                    "jaccard_threshold": JACCARD_THRESHOLD,
                },
                "inferred_parameters": {
                    "closed_only": True,
                    "created_at_cutoff": cutoff,
                    "min_chars": MIN_CHARS,
                    "max_chars": MAX_CHARS,
                    "language_detector": "langdetect-1.0.9, seed 0",
                    "shingle_words": SHINGLE_WORDS,
                    "greeting_skip_max_chars": GREETING_MAX_CHARS,
                    "lead_words": sorted(LEAD_WORDS),
                },
                "funnel": funnel,
                "dropped_counts": {k: len(v) for k, v in dropped.items()},
                "dropped_numbers": dropped,
                "near_duplicate_pairs": near_pairs,
                "sessions": seed_sessions,
            },
            ensure_ascii=False,
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"funnel: {funnel}")
    print(f"dropped: { {k: len(v) for k, v in dropped.items()} }")
    print(f"wrote {len(seed_sessions)} seed sessions to {OUTPUT}")


if __name__ == "__main__":
    main()
