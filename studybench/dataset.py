"""Pinned StudyBench questions and source repositories."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent
NOTE_PREFIX = (
    "Reference notes on {library} from your prior study of its repository:\n\n"
    "{note}\n\n---\n\n"
)


@dataclass(frozen=True)
class Corpus:
    name: str
    display: str
    repo: Path
    roots: tuple[str, ...]
    commit: str
    dataset_sha256: str
    question_count: int
    file_count: int | None = None
    snapshot_sha256: str | None = None


CORPORA = {
    "smalldspy": Corpus(
        "smalldspy",
        "SmallDSPy",
        ROOT / "corpora/smalldspy",
        ("dspy", "tests"),
        "9cdb0aac28b2a04b064e40697ccd301872cf6a43",
        "b152153a9ec159dc99f89d9a1ca085a88d04be818b348e58cebf620513b2c75d",
        5,
        66,
        "edfd5e412afa87ff13e24c1515157c71199fa3e92c1a95e06bde4372ff450b5a",
    ),
    "dspy": Corpus(
        "dspy",
        "DSPy",
        ROOT / "corpora/dspy",
        ("dspy", "tests"),
        "9cdb0aac28b2a04b064e40697ccd301872cf6a43",
        "c814c8da2d49aa892930a9d4408f087707720d9e6f84511de7479d0854580325",
        30,
    ),
    "openclaw": Corpus(
        "openclaw",
        "OpenClaw",
        ROOT / "corpora/openclaw",
        ("src", "extensions"),
        "da228660306b55a9cce3b973946f3aacfc515848",
        "d08f953f9480623a54762fae9fa8b35a9538b375692c4d058895aeba5a1dc50f",
        20,
    ),
}


def _git(repo: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError(f"cannot inspect corpus {repo}: {error}") from error
    return result.stdout.strip()


def verify_corpus(corpus: Corpus) -> None:
    if _git(corpus.repo, "rev-parse", "HEAD") != corpus.commit:
        raise ValueError(f"{corpus.display} is not at pinned commit {corpus.commit}")
    if _git(corpus.repo, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError(f"{corpus.display} checkout is dirty: {corpus.repo}")


@lru_cache(maxsize=None)
def read_corpus_file(corpus: Corpus, relative: str) -> str:
    logical = PurePosixPath(relative)
    if logical.is_absolute() or ".." in logical.parts or not logical.parts:
        raise ValueError(f"invalid corpus path: {relative}")
    if logical.parts[0] not in corpus.roots:
        raise ValueError(f"path is outside the exposed code roots: {relative}")
    path = corpus.repo.joinpath(*logical.parts)
    root = corpus.repo.resolve()
    if path.is_symlink() or not path.is_file() or root not in path.resolve().parents:
        raise ValueError(f"path is not a regular corpus file: {relative}")
    return path.read_text(encoding="utf-8", errors="replace")


@lru_cache(maxsize=None)
def load_questions(task: str) -> tuple[dict, ...]:
    corpus = CORPORA[task]
    path = ROOT / "data" / f"{task}.jsonl"
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != corpus.dataset_sha256:
        raise ValueError(f"StudyBench data hash mismatch: {path}")
    rows = tuple(json.loads(line) for line in payload.splitlines() if line.strip())
    if len(rows) != corpus.question_count:
        raise ValueError(f"expected {corpus.question_count} {task} questions, found {len(rows)}")
    ids: set[str] = set()
    for row in rows:
        if set(row) != {"id", "topic", "question", "gold_answer", "rubric", "evidence"}:
            raise ValueError(f"unexpected fields in StudyBench row: {row.get('id')}")
        if not isinstance(row["id"], str) or row["id"] in ids:
            raise ValueError(f"invalid or duplicate question id: {row.get('id')}")
        ids.add(row["id"])
        evidence_ids = {span["span_id"] for span in row["evidence"]}
        claim_ids = [claim["claim_id"] for claim in row["rubric"]]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError(f"duplicate rubric claim id: {row['id']}")
        if sum(claim["weight"] for claim in row["rubric"]) != 100:
            raise ValueError(f"rubric weights do not sum to 100: {row['id']}")
        for claim in row["rubric"]:
            if claim["claim_type"] not in {"core", "supporting"}:
                raise ValueError(f"invalid claim type: {row['id']}/{claim['claim_id']}")
            if not set(claim["span_ids"]).issubset(evidence_ids):
                raise ValueError(f"unknown evidence span: {row['id']}/{claim['claim_id']}")
        for span in row["evidence"]:
            if span["start_line"] < 1 or span["end_line"] < span["start_line"]:
                raise ValueError(f"invalid evidence range: {row['id']}/{span['span_id']}")
            logical = PurePosixPath(span["path"])
            if not logical.parts or logical.parts[0] not in corpus.roots or ".." in logical.parts:
                raise ValueError(f"evidence path escapes corpus roots: {span['path']}")
    return rows
