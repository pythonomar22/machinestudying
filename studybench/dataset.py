"""Pinned StudyBench corpora and strict validation of their question bundles."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Any

from .integrity import read_artifact_bytes_with_mode

ROOT = Path(__file__).resolve().parent.parent
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
ROW_FIELDS = {"id", "topic", "question", "gold_answer", "rubric", "evidence"}
EVIDENCE_FIELDS = {"span_id", "path", "start_line", "end_line", "excerpt"}
RUBRIC_FIELDS = {"claim_id", "claim_type", "weight", "statement", "span_ids"}


@dataclass(frozen=True)
class Corpus:
    name: str
    display: str
    repo: Path
    roots: tuple[str, ...]
    language: str
    commit: str
    code_suffixes: tuple[str, ...]
    dataset_sha256: str
    question_count: int


CORPORA = {
    "dspy": Corpus(
        name="dspy",
        display="DSPy",
        repo=ROOT / "corpora/dspy",
        roots=("dspy", "tests"),
        language="python",
        commit="9cdb0aac28b2a04b064e40697ccd301872cf6a43",
        code_suffixes=(".py",),
        dataset_sha256="c814c8da2d49aa892930a9d4408f087707720d9e6f84511de7479d0854580325",
        question_count=30,
    ),
    "openclaw": Corpus(
        name="openclaw",
        display="OpenClaw",
        repo=ROOT / "corpora/openclaw",
        roots=("src", "extensions"),
        language="typescript",
        commit="da228660306b55a9cce3b973946f3aacfc515848",
        code_suffixes=(".ts", ".tsx", ".js", ".mjs", ".cjs"),
        dataset_sha256="d08f953f9480623a54762fae9fa8b35a9538b375692c4d058895aeba5a1dc50f",
        question_count=20,
    ),
}


def _git(repo: Path, *args: str) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(f"cannot inspect pinned corpus {repo}: {exc}") from exc
    return proc.stdout.strip()


def _git_bytes(repo: Path, *args: str) -> bytes:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(f"cannot inspect pinned corpus {repo}: {exc}") from exc
    return proc.stdout


@lru_cache(maxsize=None)
def _pinned_code_index(corpus: Corpus) -> tuple[str, dict[str, tuple[str, str]]]:
    """Return the pinned tree's code paths as ``path -> (mode, blob oid)``."""

    algorithm = _git(corpus.repo, "rev-parse", "--show-object-format")
    if algorithm not in hashlib.algorithms_available:
        raise ValueError(f"unsupported Git object format for {corpus.name}: {algorithm}")
    raw = _git_bytes(
        corpus.repo,
        "ls-tree",
        "-r",
        "-z",
        "--full-tree",
        corpus.commit,
        "--",
        *corpus.roots,
    )
    entries: dict[str, tuple[str, str]] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        try:
            metadata, encoded_path = record.split(b"\t", 1)
            mode, kind, object_id = metadata.decode("ascii").split(" ", 2)
            relative = encoded_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as exc:
            raise ValueError(f"invalid pinned Git tree entry in {corpus.name}") from exc
        logical = PurePosixPath(relative)
        if logical.suffix.lower() not in corpus.code_suffixes:
            continue
        if kind != "blob" or mode not in {"100644", "100755"}:
            raise ValueError(f"path escapes configured corpus root: {relative}")
        if (
            logical.is_absolute()
            or logical.as_posix() != relative
            or any(part in ("", ".", "..") for part in logical.parts)
            or not logical.parts
            or logical.parts[0] not in corpus.roots
            or relative in entries
        ):
            raise ValueError(f"unsafe pinned code entry in {corpus.name}: {relative!r}")
        entries[relative] = (mode, object_id)
    if not entries:
        raise ValueError(f"pinned corpus has no allowed code files: {corpus.name}")
    return algorithm, entries


def read_pinned_code_bytes(corpus: Corpus, relative: str) -> bytes:
    """Race-safely read a code file and require its exact committed blob bytes."""

    path = resolve_code_path(corpus, relative)
    algorithm, entries = _pinned_code_index(corpus)
    expected = entries.get(relative)
    if expected is None:
        raise ValueError(f"code path is not tracked at pinned commit: {relative}")
    mode, object_id = expected
    data, permissions = read_artifact_bytes_with_mode(path.absolute())
    digest = hashlib.new(
        algorithm,
        b"blob " + str(len(data)).encode("ascii") + b"\0" + data,
    ).hexdigest()
    if digest != object_id:
        raise ValueError(f"code file differs from pinned commit: {relative}")
    executable = bool(permissions & 0o111)
    if executable != (mode == "100755"):
        raise ValueError(f"code file mode differs from pinned commit: {relative}")
    return data


def validate_corpus_snapshot(corpus: Corpus) -> None:
    """Fail unless *corpus* is the expected clean source snapshot."""

    if not corpus.repo.is_dir() or not (corpus.repo / ".git").is_dir():
        raise ValueError(f"missing git checkout for {corpus.name}: {corpus.repo}")
    head = _git(corpus.repo, "rev-parse", "HEAD")
    if head != corpus.commit:
        raise ValueError(
            f"{corpus.name} is at {head}, expected pinned commit {corpus.commit}"
        )
    dirty = _git(corpus.repo, "status", "--porcelain=v1", "--untracked-files=all")
    if dirty:
        first = dirty.splitlines()[0]
        raise ValueError(f"{corpus.name} checkout is dirty ({first}); refusing mixed provenance")
    indexed = _git_bytes(corpus.repo, "ls-files", "-v", "-z")
    flagged = [
        record for record in indexed.split(b"\0")
        if record and record[:2] != b"H "
    ]
    if flagged:
        raise ValueError(
            f"{corpus.name} checkout uses hidden index flags; refusing mixed provenance"
        )

    repo = corpus.repo.resolve(strict=True)
    for root_name in corpus.roots:
        root_path = corpus.repo / root_name
        try:
            resolved = root_path.resolve(strict=True)
        except OSError as exc:
            raise ValueError(f"missing corpus root {root_name!r} in {corpus.name}") from exc
        if root_path.is_symlink() or not resolved.is_dir() or not resolved.is_relative_to(repo):
            raise ValueError(f"unsafe corpus root {root_name!r} in {corpus.name}")


def resolve_code_path(corpus: Corpus, relative: str) -> Path:
    """Resolve a dataset/tool path and enforce the corpus's code-only boundary."""

    if (
        not isinstance(relative, str)
        or not relative
        or "\\" in relative
        or "\x00" in relative
    ):
        raise ValueError(f"invalid repository path: {relative!r}")
    logical = PurePosixPath(relative)
    if (
        logical.is_absolute()
        or logical.as_posix() != relative
        or any(part in ("", ".", "..") for part in logical.parts)
    ):
        raise ValueError(f"repository path must be normalized and relative: {relative!r}")
    if logical.suffix.lower() not in corpus.code_suffixes:
        raise ValueError(f"non-code path is outside the {corpus.name} scope: {relative}")
    if logical.parts[0] not in corpus.roots:
        raise ValueError(f"path is outside configured roots for {corpus.name}: {relative}")

    candidate = corpus.repo.joinpath(*logical.parts)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"missing corpus code file: {relative}") from exc
    root = (corpus.repo / logical.parts[0]).resolve(strict=True)
    if candidate.is_symlink() or not resolved.is_file() or not resolved.is_relative_to(root):
        raise ValueError(f"path escapes configured corpus root: {relative}")
    return candidate


def tracked_code_paths(corpus: Corpus) -> list[Path]:
    """Return every tracked file in the corpus's explicit code allowlist."""

    _, entries = _pinned_code_index(corpus)
    return [resolve_code_path(corpus, relative) for relative in sorted(entries)]


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        value[key] = item
    return value


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _identifier(value: object, label: str) -> str:
    text = _nonempty_string(value, label)
    if not IDENTIFIER.fullmatch(text):
        raise ValueError(f"{label} contains unsafe characters: {text!r}")
    return text


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _strict_utf8(corpus: Corpus, relative: str) -> str:
    try:
        return read_pinned_code_bytes(corpus, relative).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"file is not valid UTF-8: {relative}") from exc


def validate_questions(
    corpus: Corpus,
    data_path: Path,
    *,
    expected_sha256: str | None,
    expected_count: int | None,
) -> list[dict[str, Any]]:
    """Validate a complete benchmark bundle, including source-excerpt identity."""

    raw = data_path.read_bytes()
    if expected_sha256 is not None:
        observed = hashlib.sha256(raw).hexdigest()
        if observed != expected_sha256:
            raise ValueError(
                f"dataset hash mismatch for {corpus.name}: {observed}, expected {expected_sha256}"
            )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"dataset is not valid UTF-8: {data_path}") from exc
    if not text or not text.endswith("\n"):
        raise ValueError(f"dataset must be non-empty and newline-terminated: {data_path}")

    rows: list[dict[str, Any]] = []
    seen_qids: set[str] = set()
    source_cache: dict[Path, list[str]] = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank dataset record at {data_path}:{line_number}")
        try:
            row = json.loads(
                line,
                object_pairs_hook=_object_without_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"invalid JSON at {data_path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"dataset record {line_number} is not an object")
        if set(row) != ROW_FIELDS:
            raise ValueError(
                f"record {line_number} fields are {sorted(row)}, expected {sorted(ROW_FIELDS)}"
            )

        qid = _identifier(row.get("id"), f"record {line_number} id")
        if qid in seen_qids:
            raise ValueError(f"duplicate question id: {qid}")
        seen_qids.add(qid)
        for field in ("topic", "question", "gold_answer"):
            _nonempty_string(row.get(field), f"{qid}.{field}")

        evidence = row.get("evidence")
        rubric = row.get("rubric")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError(f"{qid}.evidence must be a non-empty list")
        if not isinstance(rubric, list) or not rubric:
            raise ValueError(f"{qid}.rubric must be a non-empty list")

        span_ids: set[str] = set()
        for index, span in enumerate(evidence):
            if not isinstance(span, dict):
                raise ValueError(f"{qid}.evidence[{index}] must be an object")
            if set(span) != EVIDENCE_FIELDS:
                raise ValueError(f"{qid}.evidence[{index}] has unexpected fields")
            span_id = _identifier(span.get("span_id"), f"{qid}.evidence[{index}].span_id")
            if span_id in span_ids:
                raise ValueError(f"duplicate evidence span id in {qid}: {span_id}")
            span_ids.add(span_id)
            relative = _nonempty_string(span.get("path"), f"{qid}.{span_id}.path")
            source = resolve_code_path(corpus, relative)
            start, end = span.get("start_line"), span.get("end_line")
            if (
                isinstance(start, bool)
                or isinstance(end, bool)
                or not isinstance(start, int)
                or not isinstance(end, int)
                or start < 1
                or end < start
            ):
                raise ValueError(f"invalid line range for {qid}.{span_id}: {start}-{end}")
            excerpt = _nonempty_string(span.get("excerpt"), f"{qid}.{span_id}.excerpt")
            if source not in source_cache:
                source_cache[source] = _strict_utf8(corpus, relative).splitlines()
            lines = source_cache[source]
            if end > len(lines):
                raise ValueError(
                    f"evidence range exceeds {relative}: {start}-{end} of {len(lines)}"
                )
            expected = "\n".join(
                f"{number:04d}: {lines[number - 1]}" for number in range(start, end + 1)
            )
            if excerpt != expected:
                raise ValueError(f"evidence excerpt does not match {relative}:{start}-{end}")

        claim_ids: set[str] = set()
        weight_total = 0
        has_core_claim = False
        for index, claim in enumerate(rubric):
            if not isinstance(claim, dict):
                raise ValueError(f"{qid}.rubric[{index}] must be an object")
            if set(claim) != RUBRIC_FIELDS:
                raise ValueError(f"{qid}.rubric[{index}] has unexpected fields")
            claim_id = _identifier(claim.get("claim_id"), f"{qid}.rubric[{index}].claim_id")
            if claim_id in claim_ids:
                raise ValueError(f"duplicate rubric claim id in {qid}: {claim_id}")
            claim_ids.add(claim_id)
            if claim.get("claim_type") not in ("core", "supporting"):
                raise ValueError(f"invalid claim type for {qid}.{claim_id}")
            has_core_claim = has_core_claim or claim["claim_type"] == "core"
            _nonempty_string(claim.get("statement"), f"{qid}.{claim_id}.statement")
            weight = claim.get("weight")
            if isinstance(weight, bool) or not isinstance(weight, int) or weight <= 0:
                raise ValueError(f"invalid weight for {qid}.{claim_id}: {weight!r}")
            weight_total += weight
            refs = claim.get("span_ids")
            if (
                not isinstance(refs, list)
                or not refs
                or any(not isinstance(ref, str) or ref not in span_ids for ref in refs)
            ):
                raise ValueError(f"invalid evidence references for {qid}.{claim_id}")
            if len(refs) != len(set(refs)):
                raise ValueError(f"duplicate evidence references for {qid}.{claim_id}")
        if weight_total != 100:
            raise ValueError(f"rubric weights for {qid} sum to {weight_total}, expected 100")
        if not has_core_claim:
            raise ValueError(f"rubric for {qid} must contain at least one core claim")
        rows.append(row)

    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(
            f"{corpus.name} has {len(rows)} questions, expected {expected_count}"
        )
    return rows


def load_questions(task: str) -> list[dict[str, Any]]:
    if task not in CORPORA:
        raise ValueError(f"unknown StudyBench task: {task!r}")
    corpus = CORPORA[task]
    validate_corpus_snapshot(corpus)
    return validate_questions(
        corpus,
        ROOT / "data" / f"{corpus.name}.jsonl",
        expected_sha256=corpus.dataset_sha256,
        expected_count=corpus.question_count,
    )
