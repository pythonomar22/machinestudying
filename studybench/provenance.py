"""Fail-closed manifests for study and evaluation artifacts.

Historical runs predate these contracts and remain readable as legacy evidence.
New runs must live under a caller-chosen ID and bind every episode to one exact
dataset, corpus, prompt, note, seed policy, and source tree.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from fnmatch import fnmatchcase
from functools import lru_cache
from importlib import metadata
import hashlib
import importlib.util
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import socket
import stat
import subprocess
import sys
from typing import Any, Callable
from urllib.parse import urlsplit

from .dataset import ROOT, Corpus, validate_corpus_snapshot
from .human_audit import (
    HumanAuditError,
    validate_human_audit_protocol,
    validate_human_audit_result,
)
from .integrity import (
    canonical_json_bytes,
    load_json_artifact,
    read_artifact_bytes,
    read_artifact_bytes_with_mode,
    sha256_bytes,
    sha256_file,
    sha256_json,
    sha256_text,
    stable_seed,
    strict_json_loads,
    write_immutable_bytes,
    write_immutable_json,
)
from .model_cache import ATTESTATION_POLICY as MODEL_CACHE_ATTESTATION_POLICY
from .preregistration import (
    PREREGISTRATION_SCHEMA_VERSION,
    RUN_FAILURE_POLICY,
    SCREEN_FAILURE_POLICY,
    bind_preregistration,
    revalidate_run_preregistration,
)
from .study_protocol import (
    HUMAN_AUDITED_NOTE_MANIFEST_TYPE,
    SEMANTIC_SELFQUIZ_NOTE_MANIFEST_TYPE,
    STATIC_GRAPH_NOTE_MANIFEST_TYPE,
    StudyProtocolError,
    validate_construction_protocol,
    validate_forced50_config,
    validate_forced50_episode,
    validate_study_note_archive,
)


SCHEMA_VERSION = 1
VLLM_VERSION = "0.24.0"
VLLM_PYTHON_VERSION = "3.12.11"
MAIN_PYTHON_VERSION = "3.14.6"
DSPY_COMMIT = "9cdb0aac28b2a04b064e40697ccd301872cf6a43"
MODEL_ID = "Qwen/Qwen3.5-9B"
MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
ENVIRONMENT_COMPATIBILITY_POLICY = "allocation-and-transport-nuisances-v1"
GRADING_RUNTIME_ATTESTATION_POLICY = (
    "python-executable-uv-lock-installed-distributions-sha256-v1"
)
LOCAL_JUDGE_RUNTIME_ATTESTATION_POLICY = "normalized-launch-environment-sha256-v1"
_ID = re.compile(r"[a-z0-9][a-z0-9._-]{2,79}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PACKAGE = re.compile(r"[a-z0-9][a-z0-9._-]*==[^\s=]+\Z")
_SOURCE_GLOBS = (
    "studybench/*.py",
    "scripts/*",
    "data/*.jsonl",
    "preregistrations/*.json",
    "pyproject.toml",
    "uv.lock",
    ".python-version",
    "README.md",
    "AGENTS.md",
    "CLAUDE.md",
    "cluster.md",
    "docs/*.md",
    "experiments/*.md",
)
_SOURCE_TREE_SCOPES = ("tests",)
_SOURCE_PATHSPECS = (*_SOURCE_GLOBS, *_SOURCE_TREE_SCOPES)
_TRANSIENT_TEST_DIRECTORIES = frozenset(
    {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
)
_TRANSIENT_TEST_SUFFIXES = frozenset({".pyc", ".pyo"})


@dataclass(frozen=True)
class RunContext:
    root: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    note: str
    prompt_prefix: str
    launch_environment: dict[str, object] | None = None
    launch_environment_record: dict[str, object] | None = None

    @property
    def note_sha256(self) -> str | None:
        note = self.manifest["spec"].get("note")
        return note["sha256"] if note else None


def validate_id(value: str, label: str = "run ID") -> str:
    if not _ID.fullmatch(value):
        raise ValueError(
            f"{label} must be 3-80 lowercase letters, digits, '.', '_' or '-': {value!r}"
        )
    return value


def validate_local_server_urls(raw: str, *, expected_count: int | None = None) -> list[str]:
    """Validate the loopback-only OpenAI-compatible endpoints used in research.

    Endpoint ports are deliberately not part of run identity because Slurm
    assigns a fresh collision-free port range on retry.  The server count and
    all model/environment identities remain manifest-bound.
    """

    if not isinstance(raw, str) or not raw:
        raise ValueError("at least one local model server URL is required")
    urls = raw.split(",")
    canonical: list[str] = []
    for url in urls:
        if not url or url.strip() != url:
            raise ValueError("model server URLs must be nonempty and contain no outer whitespace")
        try:
            parsed = urlsplit(url)
            port = parsed.port
        except ValueError as exc:
            raise ValueError(f"invalid model server URL: {url!r}") from exc
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
            or port is None
            or not 1 <= port <= 65535
            or parsed.path.rstrip("/") != "/v1"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "research model endpoints must be explicit loopback HTTP URLs ending in /v1"
            )
        canonical.append(f"http://localhost:{port}/v1")
    if len(canonical) != len(set(canonical)):
        raise ValueError(
            "model server URLs must have unique normalized loopback host/port identities"
        )
    if expected_count is not None and len(canonical) != expected_count:
        raise ValueError(
            f"received {len(canonical)} model server URL(s), environment declares "
            f"{expected_count}"
        )
    return canonical


def _git(*args: str, cwd: Path | None = None) -> str:
    cwd = ROOT if cwd is None else cwd
    proc = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False
    )
    if proc.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def _git_bytes(*args: str, cwd: Path | None = None) -> bytes:
    cwd = ROOT if cwd is None else cwd
    proc = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, check=False
    )
    if proc.returncode:
        detail = os.fsdecode(proc.stderr).strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return proc.stdout


def _source_git_path(raw: bytes) -> str:
    relative = os.fsdecode(raw)
    logical = PurePosixPath(relative)
    if (
        not relative
        or logical.is_absolute()
        or logical.as_posix() != relative
        or any(part in ("", ".", "..") for part in logical.parts)
    ):
        raise ValueError(f"unsafe research source path from Git: {relative!r}")
    return relative


def _source_path_in_scope(relative: str) -> bool:
    logical = PurePosixPath(relative)
    return (
        bool(logical.parts) and logical.parts[0] in _SOURCE_TREE_SCOPES
    ) or any(
        relative.count("/") == pattern.count("/")
        and fnmatchcase(relative, pattern)
        for pattern in _SOURCE_GLOBS
    )


def _head_source_entries() -> dict[str, tuple[str, str, str]]:
    entries: dict[str, tuple[str, str, str]] = {}
    raw_entries = _git_bytes("ls-tree", "-r", "-z", "--full-tree", "HEAD")
    for raw_entry in raw_entries.split(b"\0"):
        if not raw_entry:
            continue
        try:
            header, raw_path = raw_entry.split(b"\t", 1)
            raw_mode, raw_kind, raw_oid = header.split(b" ", 2)
            mode = raw_mode.decode("ascii")
            kind = raw_kind.decode("ascii")
            oid = raw_oid.decode("ascii")
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError("malformed Git HEAD source entry") from error
        relative = _source_git_path(raw_path)
        if not _source_path_in_scope(relative):
            continue
        if relative in entries:
            raise ValueError(f"duplicate Git HEAD source entry: {relative}")
        entries[relative] = (mode, kind, oid)
    return entries


def _index_source_entries() -> set[tuple[str, str, str, str]]:
    entries: set[tuple[str, str, str, str]] = set()
    raw_entries = _git_bytes("ls-files", "--stage", "-z")
    for raw_entry in raw_entries.split(b"\0"):
        if not raw_entry:
            continue
        try:
            header, raw_path = raw_entry.split(b"\t", 1)
            raw_mode, raw_oid, raw_stage = header.split(b" ", 2)
            mode = raw_mode.decode("ascii")
            oid = raw_oid.decode("ascii")
            stage = raw_stage.decode("ascii")
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError("malformed Git index source entry") from error
        relative = _source_git_path(raw_path)
        if _source_path_in_scope(relative):
            entries.add((relative, mode, oid, stage))
    return entries


def _source_has_hidden_index_state() -> bool:
    for record in _git_bytes("ls-files", "-v", "-z").split(b"\0"):
        if not record:
            continue
        if len(record) < 3 or record[1:2] != b" ":
            return True
        relative = _source_git_path(record[2:])
        if _source_path_in_scope(relative) and record[:2] != b"H ":
            return True
    return False


def _git_blob_oid(data: bytes, object_format: str) -> str:
    if object_format not in {"sha1", "sha256"}:
        raise ValueError(f"unsupported Git object format: {object_format!r}")
    digest = hashlib.new(object_format)
    digest.update(f"blob {len(data)}\0".encode("ascii"))
    digest.update(data)
    return digest.hexdigest()


def corpus_record(corpus: Corpus) -> dict[str, object]:
    validate_corpus_snapshot(corpus)
    return {
        "name": corpus.name,
        "commit": corpus.commit,
        "dirty": False,
        "roots": list(corpus.roots),
        "language": corpus.language,
        "suffixes": sorted(corpus.code_suffixes),
    }


def source_record() -> dict[str, object]:
    head_entries = _head_source_entries()
    index_entries = _index_source_entries()
    candidate_paths = {
        ROOT.joinpath(*PurePosixPath(relative).parts)
        for relative in head_entries
    }
    candidate_paths.update(
        ROOT.joinpath(*PurePosixPath(relative).parts)
        for relative, _mode, _oid, _stage in index_entries
    )
    for pattern in _SOURCE_GLOBS:
        for path in ROOT.glob(pattern):
            candidate_paths.add(path)
    # Freeze the complete test tree, including ignored and untracked fixtures.
    # Interpreter/test-runner caches are generated execution residue rather
    # than research source; tracked cache files remain covered by the Git tree.
    test_root = ROOT / "tests"
    try:
        test_root_metadata = test_root.lstat()
    except FileNotFoundError:
        test_root_metadata = None
    except OSError as error:
        raise ValueError(f"cannot inspect research source path: {test_root}") from error
    if test_root_metadata is not None:
        candidate_paths.add(test_root)
        if stat.S_ISDIR(test_root_metadata.st_mode):
            for path in test_root.rglob("*"):
                relative = path.relative_to(test_root)
                if (
                    any(part in _TRANSIENT_TEST_DIRECTORIES for part in relative.parts)
                    or path.suffix in _TRANSIENT_TEST_SUFFIXES
                ):
                    continue
                candidate_paths.add(path)

    files: dict[str, dict[str, object]] = {}
    live: dict[str, tuple[bytes, int]] = {}
    missing_path = False
    for path in sorted(candidate_paths):
        try:
            metadata_record = path.lstat()
        except FileNotFoundError:
            missing_path = True
            continue
        except OSError as error:
            raise ValueError(f"cannot inspect research source path: {path}") from error
        if stat.S_ISLNK(metadata_record.st_mode):
            raise ValueError(f"research source path must not be a symlink: {path}")
        if not stat.S_ISREG(metadata_record.st_mode):
            continue
        data, mode = read_artifact_bytes_with_mode(path)
        relative = str(path.relative_to(ROOT))
        live[relative] = (data, mode)
        files[relative] = {
            "sha256": sha256_bytes(data),
            "bytes": len(data),
        }

    expected_index = {
        (relative, mode, oid, "0")
        for relative, (mode, _kind, oid) in head_entries.items()
    }
    hidden_index_state = _source_has_hidden_index_state()
    object_format = _git("rev-parse", "--show-object-format")
    live_matches_head = set(live) == set(head_entries)
    if live_matches_head:
        for relative, (expected_mode, kind, expected_oid) in head_entries.items():
            data, filesystem_mode = live[relative]
            live_mode = "100755" if filesystem_mode & 0o111 else "100644"
            if (
                kind != "blob"
                or expected_mode not in {"100644", "100755"}
                or live_mode != expected_mode
                or _git_blob_oid(data, object_format) != expected_oid
            ):
                live_matches_head = False
                break
    # Porcelain remains useful for untracked files and path-level changes, but
    # correctness does not rely on it: HEAD, index, live bytes, and live modes
    # are compared independently so Git's hidden flags and core.filemode cannot
    # conceal source drift.
    status = _git_bytes(
        "status", "--porcelain", "-z", "--untracked-files=all", "--", *_SOURCE_PATHSPECS
    )
    dirty = bool(
        status
        or missing_path
        or hidden_index_state
        or index_entries != expected_index
        or not live_matches_head
    )
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "dirty": dirty,
        "files": files,
        "tree_sha256": sha256_json(files),
    }


def validate_current_source(expected: object) -> dict[str, object]:
    """Require downstream analysis to use the exact source frozen at launch.

    A preregistration is not meaningful if the judge prompt, score code, or
    analysis can change after outcomes are generated.  Claim-ready grading,
    reporting, and comparison all pass through this check and therefore use
    the same clean Git commit and byte inventory as the run manifest.
    """

    current = source_record()
    try:
        matches = canonical_json_bytes(current) == canonical_json_bytes(expected)
    except (TypeError, ValueError):
        matches = False
    if not matches:
        raise ValueError(
            "current research source differs from the run's frozen source record"
        )
    return current


def _canonical_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _installed_file_identity(path: Path) -> tuple[int, str]:
    """Stream one large installed file while rejecting mutation during hashing."""

    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"installed distribution path is not a regular file: {path}")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise ValueError(f"installed distribution file changed while hashing: {path}")
    current = path.stat(follow_symlinks=False)
    if (
        not stat.S_ISREG(current.st_mode)
        or (current.st_dev, current.st_ino) != (after.st_dev, after.st_ino)
    ):
        raise ValueError(f"installed distribution file changed while hashing: {path}")
    return after.st_size, digest.hexdigest()


def _build_installed_distribution_inventory(
    distributions: object,
    *,
    prefix: Path,
    python_version: str,
) -> dict[str, object]:
    """Hash every installed file declared by every distribution ``RECORD``."""

    resolved_prefix = Path(prefix).resolve(strict=True)
    if not resolved_prefix.is_dir():
        raise ValueError("installed-distribution prefix is not a directory")
    if not isinstance(python_version, str) or not re.fullmatch(
        r"[0-9]+\.[0-9]+\.[0-9]+", python_version
    ):
        raise ValueError("installed-distribution Python version is invalid")

    rows: list[dict[str, object]] = []
    names: set[str] = set()
    try:
        candidates = list(distributions)
    except TypeError as error:
        raise ValueError("installed distributions are not iterable") from error
    for distribution in candidates:
        distribution_metadata = getattr(distribution, "metadata", None)
        raw_name = (
            distribution_metadata.get("Name")
            if hasattr(distribution_metadata, "get")
            else None
        )
        version = getattr(distribution, "version", None)
        if not isinstance(raw_name, str) or not raw_name:
            raise ValueError("installed distribution has no package name")
        name = _canonical_package_name(raw_name)
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name):
            raise ValueError(f"installed distribution has an invalid package name: {name!r}")
        if (
            not isinstance(version, str)
            or not version
            or any(character.isspace() for character in version)
        ):
            raise ValueError(f"installed distribution {name!r} has an invalid version")
        if name in names:
            raise ValueError(f"duplicate installed distribution: {name}")
        names.add(name)

        declared_files = getattr(distribution, "files", None)
        if not declared_files:
            raise ValueError(f"installed distribution {name} has no RECORD file list")
        try:
            record_text = distribution.read_text("RECORD")
        except (OSError, UnicodeError) as error:
            raise ValueError(f"cannot read installed distribution RECORD: {name}") from error
        if not isinstance(record_text, str) or not record_text:
            raise ValueError(f"installed distribution {name} has no readable RECORD")
        expected_record_sha256 = sha256_text(record_text)
        files: list[dict[str, object]] = []
        seen_paths: set[str] = set()
        for declared in declared_files:
            try:
                located = Path(distribution.locate_file(declared))
                if located.is_symlink():
                    raise ValueError("installed file is a symlink")
                resolved = located.resolve(strict=True)
                relative = resolved.relative_to(resolved_prefix).as_posix()
            except (OSError, RuntimeError, ValueError) as error:
                raise ValueError(
                    f"installed distribution {name} has an unsafe or missing file: "
                    f"{declared}"
                ) from error
            logical = PurePosixPath(relative)
            if (
                logical.is_absolute()
                or not logical.parts
                or any(part in ("", ".", "..") for part in logical.parts)
                or "\\" in relative
                or any(ord(character) < 32 or ord(character) == 127 for character in relative)
            ):
                raise ValueError(
                    f"installed distribution {name} has an unsafe file path: {relative!r}"
                )
            if relative in seen_paths:
                raise ValueError(
                    f"installed distribution {name} lists a file more than once: {relative}"
                )
            seen_paths.add(relative)
            try:
                size, digest = _installed_file_identity(resolved)
            except (OSError, ValueError) as error:
                raise ValueError(
                    f"cannot read installed distribution file: {relative}"
                ) from error
            files.append({
                "path": relative,
                "bytes": size,
                "sha256": digest,
            })
        files.sort(key=lambda file: str(file["path"]))
        record_files = [
            file
            for file in files
            if PurePosixPath(str(file["path"])).name == "RECORD"
            and PurePosixPath(str(file["path"])).parent.name.endswith(".dist-info")
            and file["sha256"] == expected_record_sha256
        ]
        if len(record_files) != 1:
            raise ValueError(
                f"installed distribution {name} has no unique owning dist-info/RECORD"
            )
        rows.append({
            "name": name,
            "version": version,
            "record_path": record_files[0]["path"],
            "record_sha256": record_files[0]["sha256"],
            "file_count": len(files),
            "total_bytes": sum(int(file["bytes"]) for file in files),
            "files": files,
            "tree_sha256": sha256_json(files),
        })
    rows.sort(key=lambda row: (str(row["name"]), str(row["version"])))
    if not rows:
        raise ValueError("installed-distribution inventory is empty")
    return {
        "schema_version": 1,
        "python_version": python_version,
        "prefix": str(resolved_prefix),
        "distribution_count": len(rows),
        "file_count": sum(int(row["file_count"]) for row in rows),
        "total_bytes": sum(int(row["total_bytes"]) for row in rows),
        "distributions": rows,
        "tree_sha256": sha256_json(rows),
    }


def installed_distribution_inventory() -> dict[str, object]:
    """Return the exact installed-code identity of the running environment."""

    return _build_installed_distribution_inventory(
        metadata.distributions(),
        prefix=Path(sys.prefix),
        python_version=platform.python_version(),
    )


def _validate_installed_distribution_inventory(inventory: object) -> None:
    """Fail unless *inventory* is a complete, self-consistent byte inventory."""

    if not isinstance(inventory, dict) or set(inventory) != {
        "schema_version",
        "python_version",
        "prefix",
        "distribution_count",
        "file_count",
        "total_bytes",
        "distributions",
        "tree_sha256",
    }:
        raise ValueError("installed-distribution inventory fields are invalid")
    python_version = inventory.get("python_version")
    prefix = inventory.get("prefix")
    distributions = inventory.get("distributions")
    if (
        type(inventory.get("schema_version")) is not int
        or inventory["schema_version"] != 1
        or not isinstance(python_version, str)
        or not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", python_version)
        or not isinstance(prefix, str)
        or not Path(prefix).is_absolute()
        or not isinstance(distributions, list)
        or not distributions
    ):
        raise ValueError("installed-distribution inventory header is invalid")
    if distributions != sorted(
        distributions,
        key=lambda row: (
            str(row.get("name", "")) if isinstance(row, dict) else "",
            str(row.get("version", "")) if isinstance(row, dict) else "",
        ),
    ):
        raise ValueError("installed distributions are not deterministically ordered")

    names: list[str] = []
    total_files = 0
    total_bytes = 0
    for distribution in distributions:
        if not isinstance(distribution, dict) or set(distribution) != {
            "name",
            "version",
            "record_path",
            "record_sha256",
            "file_count",
            "total_bytes",
            "files",
            "tree_sha256",
        }:
            raise ValueError("installed-distribution record fields are invalid")
        name = distribution.get("name")
        version = distribution.get("version")
        files = distribution.get("files")
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name)
            or not isinstance(version, str)
            or not version
            or any(character.isspace() for character in version)
            or not isinstance(files, list)
            or not files
            or files
            != sorted(
                files,
                key=lambda file: (
                    str(file.get("path", "")) if isinstance(file, dict) else ""
                ),
            )
        ):
            raise ValueError("installed-distribution record is invalid")
        names.append(name)

        paths: list[str] = []
        observed_bytes = 0
        file_by_path: dict[str, dict[str, object]] = {}
        for file in files:
            if not isinstance(file, dict) or set(file) != {"path", "bytes", "sha256"}:
                raise ValueError("installed-distribution file record is invalid")
            path = file.get("path")
            logical = PurePosixPath(str(path))
            if (
                not isinstance(path, str)
                or logical.is_absolute()
                or not logical.parts
                or logical.as_posix() != path
                or any(part in ("", ".", "..") for part in logical.parts)
                or "\\" in path
                or any(ord(character) < 32 or ord(character) == 127 for character in path)
                or type(file.get("bytes")) is not int
                or file["bytes"] < 0
                or not _SHA256.fullmatch(str(file.get("sha256", "")))
            ):
                raise ValueError("installed-distribution file identity is invalid")
            paths.append(path)
            observed_bytes += file["bytes"]
            file_by_path[path] = file
        if len(paths) != len(set(paths)):
            raise ValueError("installed-distribution file paths are not unique")
        record_path = distribution.get("record_path")
        record = file_by_path.get(record_path) if isinstance(record_path, str) else None
        record_logical = PurePosixPath(record_path) if isinstance(record_path, str) else None
        if (
            record is None
            or record_logical is None
            or record_logical.name != "RECORD"
            or not record_logical.parent.name.endswith(".dist-info")
            or distribution.get("record_sha256") != record.get("sha256")
            or type(distribution.get("file_count")) is not int
            or distribution["file_count"] != len(files)
            or type(distribution.get("total_bytes")) is not int
            or distribution["total_bytes"] != observed_bytes
            or distribution.get("tree_sha256") != sha256_json(files)
        ):
            raise ValueError("installed-distribution RECORD or aggregate is invalid")
        total_files += len(files)
        total_bytes += observed_bytes
    if (
        len(names) != len(set(names))
        or type(inventory.get("distribution_count")) is not int
        or inventory["distribution_count"] != len(distributions)
        or type(inventory.get("file_count")) is not int
        or inventory["file_count"] != total_files
        or type(inventory.get("total_bytes")) is not int
        or inventory["total_bytes"] != total_bytes
        or inventory.get("tree_sha256") != sha256_json(distributions)
    ):
        raise ValueError("installed-distribution inventory aggregate is invalid")


def _runner_environment_record() -> dict[str, object]:
    """Snapshot the complete package set and Python runtime of this process."""

    packages = []
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name")
        version = distribution.version
        if not isinstance(name, str) or not name or not isinstance(version, str) or not version:
            raise ValueError("runner contains a distribution without a name/version identity")
        packages.append({"name": _canonical_package_name(name), "version": version})
    packages.sort(key=lambda row: (row["name"], row["version"]))

    executable = Path(sys.executable)
    resolved_executable = executable.resolve(strict=True)
    if not resolved_executable.is_file():
        raise ValueError("runner Python executable does not resolve to a regular file")
    pyvenv_path = Path(sys.prefix) / "pyvenv.cfg"
    pyvenv: dict[str, object] | None = None
    if pyvenv_path.is_file() and not pyvenv_path.is_symlink():
        pyvenv_bytes = read_artifact_bytes(pyvenv_path)
        pyvenv = {
            "path": str(pyvenv_path),
            "sha256": sha256_bytes(pyvenv_bytes),
            "bytes": len(pyvenv_bytes),
            "text": pyvenv_bytes.decode("utf-8"),
        }
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": str(executable),
            "resolved_executable": str(resolved_executable),
            "executable_sha256": sha256_file(resolved_executable),
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
            "pyvenv_cfg": pyvenv,
        },
        "packages": packages,
        "packages_sha256": sha256_json(packages),
    }


def _source_file_record(path: Path) -> dict[str, object]:
    data = read_artifact_bytes(path)
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256_bytes(data),
        "bytes": len(data),
    }


def _runner_lock_attestation(runner: dict[str, object]) -> dict[str, object]:
    """Prove that this interpreter is synchronized to its applicable uv lock."""

    packages = runner.get("packages")
    python_identity = runner.get("python")
    if not isinstance(packages, list) or not isinstance(python_identity, dict):
        raise ValueError("runner identity is unavailable for lock attestation")
    versions = {
        row.get("name"): row.get("version")
        for row in packages
        if isinstance(row, dict)
    }
    is_dspy = isinstance(versions.get("dspy"), str) and bool(versions["dspy"])
    kind = "dspy" if is_dspy else "main"
    expected_python = VLLM_PYTHON_VERSION if is_dspy else MAIN_PYTHON_VERSION
    if python_identity.get("version") != expected_python:
        raise ValueError(
            f"{kind} runner uses Python {python_identity.get('version')}, "
            f"expected {expected_python}"
        )
    project = ROOT / "corpora" / "dspy" if is_dspy else ROOT
    lock_path = project / "uv.lock"
    project_path = project / "pyproject.toml"
    lock = _source_file_record(lock_path)
    project_file = _source_file_record(project_path)

    uv = shutil.which("uv")
    if not uv:
        raise ValueError("uv is unavailable for frozen-environment verification")
    uv_path = Path(uv).resolve(strict=True)
    if not uv_path.is_file():
        raise ValueError("uv does not resolve to a regular executable")
    version_process = subprocess.run(
        [str(uv_path), "--version"], capture_output=True, text=True, check=False
    )
    if version_process.returncode or not version_process.stdout.strip():
        raise ValueError("cannot record the uv runtime identity")
    command = [
        str(uv_path),
        "sync",
        "--project",
        str(project),
        "--frozen",
    ]
    if is_dspy:
        command.append("--no-dev")
    command.append("--check")
    check_environment = os.environ.copy()
    check_environment.update({
        "UV_PROJECT_ENVIRONMENT": sys.prefix,
        "UV_NO_PROGRESS": "1",
    })
    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        env=check_environment,
    )
    if process.returncode:
        detail = (process.stderr or process.stdout).strip().splitlines()
        raise ValueError(
            f"{kind} runner differs from its frozen uv lock"
            + (f": {detail[-1]}" if detail else "")
        )

    dspy_corpus = None
    dspy_import = None
    if is_dspy:
        commit = _git("rev-parse", "HEAD", cwd=project)
        dirty = bool(
            _git("status", "--porcelain", "--untracked-files=all", cwd=project)
        )
        if commit != DSPY_COMMIT or dirty:
            raise ValueError("DSPy harness source is dirty or not at its pinned commit")
        spec = importlib.util.find_spec("dspy")
        if spec is None or not isinstance(spec.origin, str):
            raise ValueError("cannot identify the imported DSPy package")
        origin = Path(spec.origin).resolve(strict=True)
        prefix = Path(sys.prefix).resolve(strict=True)
        if not origin.is_file() or not origin.is_relative_to(prefix):
            raise ValueError("DSPy is not imported from the synchronized environment")
        dspy_corpus = {"commit": commit, "dirty": dirty}
        dspy_import = {
            "version": versions["dspy"],
            "origin": str(origin),
            "origin_sha256": sha256_file(origin),
        }
    return {
        "schema_version": 1,
        "kind": kind,
        "python_version": expected_python,
        "lock": lock,
        "project": project_file,
        "uv": {
            "path": str(uv_path),
            "sha256": sha256_file(uv_path),
            "version": version_process.stdout.strip(),
        },
        "sync_check": {
            "status": "synchronized",
            "arguments": command[1:],
        },
        "dspy_corpus": dspy_corpus,
        "dspy_import": dspy_import,
    }


def _secure_inventory_bytes(path_variable: str, hash_variable: str) -> tuple[bytes, dict[str, object]]:
    """Read one launcher artifact, constrained to an owner-only file in logs/."""

    raw_path = os.environ.get(path_variable)
    expected_hash = os.environ.get(hash_variable)
    if not raw_path or not expected_hash or not _SHA256.fullmatch(expected_hash):
        raise ValueError(f"missing or invalid {path_variable}/{hash_variable}")
    if "\\" in raw_path:
        raise ValueError(f"unsafe {path_variable}")
    relative = PurePosixPath(raw_path)
    if (
        relative.is_absolute()
        or len(relative.parts) != 2
        or relative.parts[0] != "logs"
        or not re.fullmatch(r"[A-Za-z0-9._-]+", relative.parts[1])
    ):
        raise ValueError(f"unsafe {path_variable}")
    path = ROOT.joinpath(*relative.parts)
    for component in (ROOT / "logs", path):
        if component.is_symlink():
            raise ValueError(f"{path_variable} traverses a symlink")
    try:
        file_stat = path.stat(follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"missing {path_variable}") from error
    if (
        not stat.S_ISREG(file_stat.st_mode)
        or file_stat.st_uid != os.getuid()
        or stat.S_IMODE(file_stat.st_mode) != 0o600
    ):
        raise ValueError(f"{path_variable} is not an owner-only regular file")
    data = read_artifact_bytes(path)
    if sha256_bytes(data) != expected_hash:
        raise ValueError(f"{path_variable} changed after server launch")
    return data, {"path": raw_path, "sha256": expected_hash, "bytes": len(data)}


def _vllm_package_snapshot() -> dict[str, object]:
    data, record = _secure_inventory_bytes(
        "SB_VLLM_ENV_INVENTORY", "SB_VLLM_ENV_SHA256"
    )
    try:
        inventory = strict_json_loads(data, label="vLLM installed-code inventory")
    except ValueError as error:
        raise ValueError("vLLM installed-code inventory is invalid JSON") from error
    if canonical_json_bytes(inventory) != data:
        raise ValueError("vLLM installed-code inventory is not canonical JSON")
    _validate_installed_distribution_inventory(inventory)
    return {**record, "inventory": inventory}


def _vllm_lock_record() -> tuple[list[str], str]:
    lock_bytes = read_artifact_bytes(ROOT / "scripts" / "vllm-requirements.lock")
    try:
        lock_text = lock_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("vLLM lock is not UTF-8") from error
    lines = [
        line.strip()
        for line in lock_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if (
        not lines
        or any(not _PACKAGE.fullmatch(line) for line in lines)
        or lines != sorted(lines, key=lambda line: tuple(line.split("==", 1)))
        or len(lines) != len(set(lines))
    ):
        raise ValueError("checked-in vLLM lock has an invalid format")
    return lines, sha256_bytes(lock_bytes)


def _json_inventory_snapshot(
    path_variable: str, hash_variable: str, *, label: str
) -> dict[str, object]:
    data, record = _secure_inventory_bytes(path_variable, hash_variable)
    inventory = strict_json_loads(data, label=label)
    if not isinstance(inventory, dict) or canonical_json_bytes(inventory) != data:
        raise ValueError(f"{label} is not canonical JSON")
    return {**record, "inventory": inventory}


def environment_record() -> dict[str, object]:
    """Record exact, secret-free runner/server/allocation identities.

    Hardware is read only from the launcher's allocated-GPU inventory.  This
    function intentionally never invokes ``nvidia-smi`` because an unscoped
    query would silently include GPUs belonging to other Slurm jobs.
    """

    errors: dict[str, str] = {}
    unsafe_runner_variables = sorted(
        name for name in (
            "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
            "http_proxy", "https_proxy", "all_proxy",
            "PYTHONHOME", "PYTHONPATH", "LD_PRELOAD", "LD_AUDIT",
        )
        if os.environ.get(name)
    )
    if unsafe_runner_variables:
        errors["runner_process_environment"] = (
            "claim-ready runner inherits unsafe process variables: "
            + ", ".join(unsafe_runner_variables)
        )

    def capture(label: str, function):
        try:
            return function()
        except (OSError, UnicodeError, ValueError) as error:
            errors[label] = str(error)
            return None

    runner = capture("runner", _runner_environment_record)
    runner_lock = capture(
        "runner_lock",
        lambda: _runner_lock_attestation(runner)
        if isinstance(runner, dict)
        else (_ for _ in ()).throw(ValueError("runner identity is unavailable")),
    )
    vllm_environment = capture("vllm_environment", _vllm_package_snapshot)
    vllm_runtime = capture(
        "vllm_runtime",
        lambda: _json_inventory_snapshot(
            "SB_VLLM_RUNTIME_INVENTORY",
            "SB_VLLM_RUNTIME_SHA256",
            label="vLLM runtime inventory",
        ),
    )
    model_cache = capture(
        "model_cache",
        lambda: _json_inventory_snapshot(
            "SB_MODEL_CACHE_INVENTORY",
            "SB_MODEL_CACHE_SHA256",
            label="model-cache inventory",
        ),
    )
    allocation = capture(
        "allocation",
        lambda: _json_inventory_snapshot(
            "SB_GPU_INVENTORY",
            "SB_GPU_INVENTORY_SHA256",
            label="allocated-GPU inventory",
        ),
    )
    api_key = os.environ.get("SB_VLLM_API_KEY")
    api_key_sha256 = sha256_text(api_key) if api_key else None
    if (
        api_key_sha256 is None
        or os.environ.get("SB_VLLM_API_KEY_SHA256") != api_key_sha256
        or os.environ.get("SB_SERVER_LAUNCH_ID") != api_key_sha256
    ):
        errors["server_identity"] = "ephemeral vLLM API identity is missing or inconsistent"
    live_allocation = {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS") or None,
        "slurm_step_gpus": os.environ.get("SLURM_STEP_GPUS") or None,
        "slurm_job_nodelist": os.environ.get("SLURM_JOB_NODELIST") or None,
        "slurm_node_id": os.environ.get("SLURM_NODEID") or None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "hostname": socket.gethostname(),
    }
    gpu_rows = []
    if isinstance(allocation, dict):
        candidate = allocation.get("inventory", {}).get("gpus", [])
        if isinstance(candidate, list):
            gpu_rows = [row for row in candidate if isinstance(row, dict)]
        allocation_inventory = allocation.get("inventory")
        slurm_inventory = (
            allocation_inventory.get("slurm")
            if isinstance(allocation_inventory, dict)
            else None
        )
        if isinstance(allocation_inventory, dict) and isinstance(slurm_inventory, dict):
            expected_live_allocation = {
                "slurm_job_id": slurm_inventory.get("job_id"),
                "slurm_job_gpus": slurm_inventory.get("job_gpus"),
                "slurm_step_gpus": slurm_inventory.get("step_gpus"),
                "slurm_job_nodelist": slurm_inventory.get("job_nodelist"),
                "slurm_node_id": slurm_inventory.get("node_id"),
                "cuda_visible_devices": allocation_inventory.get(
                    "cuda_visible_devices"
                ),
                "hostname": allocation_inventory.get("hostname"),
            }
            if live_allocation != expected_live_allocation:
                errors["runner_allocation"] = (
                    "live runner allocation does not match the launcher inventory"
                )
            if os.environ.get("SB_SERVER_HOSTNAME") != live_allocation["hostname"]:
                errors["runner_hostname"] = (
                    "live runner hostname does not match the authenticated launcher"
                )
    gpu_models = sorted(
        {row["name"] for row in gpu_rows if isinstance(row.get("name"), str)}
    )
    drivers = sorted(
        {
            row["driver_version"]
            for row in gpu_rows
            if isinstance(row.get("driver_version"), str)
        }
    )
    runner_packages = runner.get("packages", []) if isinstance(runner, dict) else []
    package_versions = {
        row["name"]: row["version"]
        for row in runner_packages
        if isinstance(row, dict)
        and isinstance(row.get("name"), str)
        and isinstance(row.get("version"), str)
    }
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "packages": {
            name: package_versions.get(name)
            for name in ("dspy", "openai", "pydantic")
        },
        "runner": runner,
        "runner_lock": runner_lock,
        "gpu_models": gpu_models or None,
        "nvidia_driver": drivers or None,
        "allocation": allocation,
        "vllm_version": os.environ.get("SB_VLLM_VERSION"),
        "vllm_environment_sha256": os.environ.get("SB_VLLM_ENV_SHA256"),
        "vllm_environment": vllm_environment,
        "vllm_runtime": vllm_runtime,
        "model_cache": model_cache,
        "model_id": os.environ.get("SB_MODEL_ID"),
        "model_revision": os.environ.get("SB_MODEL_REVISION"),
        "tensor_parallel_size": os.environ.get("SB_TP_EFFECTIVE"),
        "visible_gpu_count": os.environ.get("SB_NGPU"),
        "server_count": os.environ.get("SB_NSERVE"),
        "cuda_visible_devices": os.environ.get("SB_CUDA_VISIBLE_DEVICES"),
        "slurm_job_id": os.environ.get("SB_SLURM_JOB_ID"),
        "runner_allocation": live_allocation,
        "server_launch_id": os.environ.get("SB_SERVER_LAUNCH_ID"),
        "vllm_api_key_sha256": api_key_sha256,
        "inventory_errors": errors,
    }


def _json_snapshot_is_valid(snapshot: object) -> bool:
    if not isinstance(snapshot, dict) or set(snapshot) != {
        "path", "sha256", "bytes", "inventory"
    }:
        return False
    fingerprint = snapshot.get("sha256")
    size = snapshot.get("bytes")
    inventory = snapshot.get("inventory")
    return (
        isinstance(snapshot.get("path"), str)
        and bool(snapshot["path"])
        and isinstance(fingerprint, str)
        and bool(_SHA256.fullmatch(fingerprint))
        and type(size) is int
        and size == len(canonical_json_bytes(inventory))
        and sha256_json(inventory) == fingerprint
    )


def _runner_lock_is_valid(
    attestation: object, runner: dict[str, object]
) -> bool:
    if not isinstance(attestation, dict) or set(attestation) != {
        "schema_version",
        "kind",
        "python_version",
        "lock",
        "project",
        "uv",
        "sync_check",
        "dspy_corpus",
        "dspy_import",
    }:
        return False
    packages = runner.get("packages")
    python_identity = runner.get("python")
    if not isinstance(packages, list) or not isinstance(python_identity, dict):
        return False
    versions = {
        row.get("name"): row.get("version")
        for row in packages
        if isinstance(row, dict)
    }
    is_dspy = isinstance(versions.get("dspy"), str) and bool(versions["dspy"])
    expected_kind = "dspy" if is_dspy else "main"
    expected_python = VLLM_PYTHON_VERSION if is_dspy else MAIN_PYTHON_VERSION
    project_root = ROOT / "corpora" / "dspy" if is_dspy else ROOT
    expected_paths = {
        "lock": (project_root / "uv.lock").relative_to(ROOT).as_posix(),
        "project": (project_root / "pyproject.toml").relative_to(ROOT).as_posix(),
    }
    if (
        type(attestation.get("schema_version")) is not int
        or attestation["schema_version"] != 1
        or attestation.get("kind") != expected_kind
        or attestation.get("python_version") != expected_python
        or python_identity.get("version") != expected_python
    ):
        return False
    for field, relative in expected_paths.items():
        record = attestation.get(field)
        try:
            data = read_artifact_bytes(ROOT / relative)
        except (OSError, ValueError):
            return False
        if (
            not isinstance(record, dict)
            or set(record) != {"path", "sha256", "bytes"}
            or record.get("path") != relative
            or record.get("sha256") != sha256_bytes(data)
            or type(record.get("bytes")) is not int
            or record.get("bytes") != len(data)
        ):
            return False
    uv = attestation.get("uv")
    sync_check = attestation.get("sync_check")
    expected_arguments = [
        "sync",
        "--project",
        str(project_root),
        "--frozen",
    ]
    if is_dspy:
        expected_arguments.append("--no-dev")
    expected_arguments.append("--check")
    if (
        not isinstance(uv, dict)
        or set(uv) != {"path", "sha256", "version"}
        or not isinstance(uv.get("path"), str)
        or not Path(uv["path"]).is_absolute()
        or not _SHA256.fullmatch(str(uv.get("sha256", "")))
        or not isinstance(uv.get("version"), str)
        or not uv["version"]
        or not isinstance(sync_check, dict)
        or sync_check
        != {"status": "synchronized", "arguments": expected_arguments}
    ):
        return False
    dspy_corpus = attestation.get("dspy_corpus")
    dspy_import = attestation.get("dspy_import")
    if is_dspy:
        return bool(
            dspy_corpus == {"commit": DSPY_COMMIT, "dirty": False}
            and isinstance(dspy_import, dict)
            and set(dspy_import) == {"version", "origin", "origin_sha256"}
            and dspy_import.get("version") == versions["dspy"]
            and isinstance(dspy_import.get("origin"), str)
            and Path(dspy_import["origin"]).is_absolute()
            and _SHA256.fullmatch(str(dspy_import.get("origin_sha256", "")))
        )
    return dspy_corpus is None and dspy_import is None


def _validate_grading_runtime_record(record: object) -> None:
    """Validate the compact grading-process identity stored in reports.

    The installed-code tree digest commits to the full RECORD-derived byte
    inventory.  Keeping only its aggregate here makes the attestation small
    enough for report provenance without weakening that commitment.
    """

    if not isinstance(record, dict) or set(record) != {
        "schema_version",
        "attestation_policy",
        "python",
        "packages",
        "packages_sha256",
        "runner_lock",
        "installed_code",
    }:
        raise ValueError("grading-runtime attestation fields are invalid")
    python_identity = record.get("python")
    packages = record.get("packages")
    installed_code = record.get("installed_code")
    if (
        type(record.get("schema_version")) is not int
        or record["schema_version"] != 1
        or record.get("attestation_policy") != GRADING_RUNTIME_ATTESTATION_POLICY
        or not isinstance(python_identity, dict)
        or set(python_identity) != {
            "version",
            "implementation",
            "executable",
            "resolved_executable",
            "executable_sha256",
            "prefix",
            "base_prefix",
            "pyvenv_cfg",
        }
        or not isinstance(packages, list)
        or not packages
        or packages
        != sorted(
            packages,
            key=lambda row: (
                str(row.get("name", "")) if isinstance(row, dict) else "",
                str(row.get("version", "")) if isinstance(row, dict) else "",
            ),
        )
        or record.get("packages_sha256") != sha256_json(packages)
        or not isinstance(record.get("runner_lock"), dict)
        or not isinstance(installed_code, dict)
        or set(installed_code) != {
            "schema_version",
            "python_version",
            "prefix",
            "distribution_count",
            "file_count",
            "total_bytes",
            "tree_sha256",
        }
    ):
        raise ValueError("grading-runtime attestation header is invalid")

    names = []
    for package in packages:
        if (
            not isinstance(package, dict)
            or set(package) != {"name", "version"}
            or not isinstance(package.get("name"), str)
            or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", package["name"])
            or not isinstance(package.get("version"), str)
            or not package["version"]
            or any(character.isspace() for character in package["version"])
        ):
            raise ValueError("grading-runtime package identity is invalid")
        names.append(package["name"])
    if len(names) != len(set(names)):
        raise ValueError("grading-runtime package names are not unique")

    pyvenv = python_identity.get("pyvenv_cfg")
    if (
        not isinstance(python_identity.get("version"), str)
        or not re.fullmatch(
            r"[0-9]+\.[0-9]+\.[0-9]+", python_identity["version"]
        )
        or not isinstance(python_identity.get("implementation"), str)
        or not python_identity["implementation"]
        or any(
            not isinstance(python_identity.get(field), str)
            or not Path(python_identity[field]).is_absolute()
            for field in (
                "executable", "resolved_executable", "prefix", "base_prefix"
            )
        )
        or not _SHA256.fullmatch(
            str(python_identity.get("executable_sha256", ""))
        )
        or not isinstance(pyvenv, dict)
        or set(pyvenv) != {"path", "sha256", "bytes", "text"}
        or not isinstance(pyvenv.get("path"), str)
        or not Path(pyvenv["path"]).is_absolute()
        or not isinstance(pyvenv.get("text"), str)
        or type(pyvenv.get("bytes")) is not int
        or pyvenv["bytes"] != len(pyvenv["text"].encode("utf-8"))
        or pyvenv.get("sha256") != sha256_text(pyvenv["text"])
    ):
        raise ValueError("grading-runtime Python identity is invalid")

    if (
        type(installed_code.get("schema_version")) is not int
        or installed_code["schema_version"] != 1
        or installed_code.get("python_version") != python_identity["version"]
        or not isinstance(installed_code.get("prefix"), str)
        or not Path(installed_code["prefix"]).is_absolute()
        or Path(installed_code["prefix"]).resolve()
        != Path(python_identity["prefix"]).resolve()
        or type(installed_code.get("distribution_count")) is not int
        or installed_code["distribution_count"] != len(packages)
        or type(installed_code.get("file_count")) is not int
        or installed_code["file_count"] <= 0
        or type(installed_code.get("total_bytes")) is not int
        or installed_code["total_bytes"] <= 0
        or not _SHA256.fullmatch(str(installed_code.get("tree_sha256", "")))
    ):
        raise ValueError("grading-runtime installed-code summary is invalid")
    canonical_json_bytes(record)


@lru_cache(maxsize=1)
def _grading_runtime_record_bytes() -> bytes:
    """Build and cache one immutable current-process grading attestation."""

    runner = _runner_environment_record()
    runner_lock = _runner_lock_attestation(runner)
    if not _runner_lock_is_valid(runner_lock, runner):
        raise ValueError("grading runner lock attestation is invalid")
    inventory = installed_distribution_inventory()
    _validate_installed_distribution_inventory(inventory)
    installed_packages = [
        {
            "name": distribution["name"],
            "version": distribution["version"],
        }
        for distribution in inventory["distributions"]
    ]
    if runner["packages"] != installed_packages:
        raise ValueError(
            "grading runner package identities disagree with installed-code inventory"
        )
    python_identity = runner["python"]
    if (
        inventory["python_version"] != python_identity["version"]
        or Path(inventory["prefix"]).resolve()
        != Path(python_identity["prefix"]).resolve()
    ):
        raise ValueError(
            "grading runner Python identity disagrees with installed-code inventory"
        )
    record = {
        "schema_version": 1,
        "attestation_policy": GRADING_RUNTIME_ATTESTATION_POLICY,
        "python": python_identity,
        "packages": runner["packages"],
        "packages_sha256": runner["packages_sha256"],
        "runner_lock": runner_lock,
        "installed_code": {
            field: inventory[field]
            for field in (
                "schema_version",
                "python_version",
                "prefix",
                "distribution_count",
                "file_count",
                "total_bytes",
                "tree_sha256",
            )
        },
    }
    _validate_grading_runtime_record(record)
    return canonical_json_bytes(record)


def grading_runtime_record() -> dict[str, object]:
    """Return a mutation-safe compact identity for the live grading process."""

    record = strict_json_loads(
        _grading_runtime_record_bytes(), label="grading-runtime attestation"
    )
    if not isinstance(record, dict):  # pragma: no cover - guarded by construction
        raise ValueError("grading-runtime attestation is not an object")
    return record


def grading_runtime_sha256(record: object | None = None) -> str:
    """Hash a compact grading-runtime record after structural validation."""

    value = grading_runtime_record() if record is None else record
    _validate_grading_runtime_record(value)
    return sha256_json(value)


def validate_grading_runtime_record(
    record: object, *, require_current: bool = True
) -> None:
    """Validate a report record and, by default, reattest the live runtime."""

    _validate_grading_runtime_record(record)
    if require_current and canonical_json_bytes(record) != _grading_runtime_record_bytes():
        raise ValueError("grading runtime differs from the recorded attestation")


def environment_is_claim_ready(environment: dict[str, object]) -> bool:
    """Validate exact identities emitted by the pinned authenticated launcher."""

    if not isinstance(environment, dict) or environment.get("inventory_errors") != {}:
        return False
    runner = environment.get("runner")
    if not isinstance(runner, dict) or set(runner) != {
        "python", "packages", "packages_sha256"
    }:
        return False
    python_identity = runner.get("python")
    packages = runner.get("packages")
    if (
        not isinstance(python_identity, dict)
        or not isinstance(packages, list)
        or not packages
        or not isinstance(runner.get("packages_sha256"), str)
        or sha256_json(packages) != runner["packages_sha256"]
        or packages
        != sorted(packages, key=lambda row: (row.get("name", ""), row.get("version", "")))
    ):
        return False
    names = []
    for package in packages:
        if (
            not isinstance(package, dict)
            or set(package) != {"name", "version"}
            or not isinstance(package.get("name"), str)
            or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", package["name"])
            or not isinstance(package.get("version"), str)
            or not package["version"]
            or any(character.isspace() for character in package["version"])
        ):
            return False
        names.append(package["name"])
    if len(names) != len(set(names)):
        return False
    if not _runner_lock_is_valid(environment.get("runner_lock"), runner):
        return False
    selected_packages = environment.get("packages")
    package_versions = {
        package["name"]: package["version"] for package in packages
    }
    if (
        not isinstance(selected_packages, dict)
        or selected_packages
        != {
            name: package_versions.get(name)
            for name in ("dspy", "openai", "pydantic")
        }
        or not all(selected_packages.get(name) for name in ("openai", "pydantic"))
    ):
        return False
    pyvenv = python_identity.get("pyvenv_cfg")
    if (
        not isinstance(python_identity.get("version"), str)
        or not isinstance(python_identity.get("implementation"), str)
        or environment.get("python") != python_identity.get("version")
        or environment.get("implementation") != python_identity.get("implementation")
        or not isinstance(python_identity.get("executable"), str)
        or not isinstance(python_identity.get("resolved_executable"), str)
        or not isinstance(python_identity.get("executable_sha256"), str)
        or not _SHA256.fullmatch(python_identity["executable_sha256"])
        or not isinstance(python_identity.get("prefix"), str)
        or not isinstance(python_identity.get("base_prefix"), str)
        or python_identity["prefix"] == python_identity["base_prefix"]
        or not isinstance(pyvenv, dict)
        or set(pyvenv) != {"path", "sha256", "bytes", "text"}
        or not isinstance(pyvenv.get("path"), str)
        or not isinstance(pyvenv.get("text"), str)
        or type(pyvenv.get("bytes")) is not int
        or pyvenv["bytes"] != len(pyvenv["text"].encode("utf-8"))
        or pyvenv.get("sha256") != sha256_text(pyvenv["text"])
    ):
        return False

    vllm_environment = environment.get("vllm_environment")
    if not _json_snapshot_is_valid(vllm_environment):
        return False
    installed_code = vllm_environment["inventory"]
    try:
        _validate_installed_distribution_inventory(installed_code)
        locked_packages, lock_sha256 = _vllm_lock_record()
    except (OSError, UnicodeError, ValueError):
        return False
    installed_versions = [
        f"{distribution['name']}=={distribution['version']}"
        for distribution in installed_code["distributions"]
    ]
    if (
        installed_code.get("python_version") != VLLM_PYTHON_VERSION
        or installed_versions != locked_packages
        or f"vllm=={VLLM_VERSION}" not in installed_versions
    ):
        return False

    vllm_runtime = environment.get("vllm_runtime")
    model_cache = environment.get("model_cache")
    allocation = environment.get("allocation")
    if not all(
        _json_snapshot_is_valid(snapshot)
        for snapshot in (vllm_runtime, model_cache, allocation)
    ):
        return False
    runtime_inventory = vllm_runtime["inventory"]
    cuda_toolkit = runtime_inventory.get("cuda_toolkit")
    nvcc = cuda_toolkit.get("nvcc") if isinstance(cuda_toolkit, dict) else None
    torch_identity = runtime_inventory.get("torch")
    if (
        type(runtime_inventory.get("schema_version")) is not int
        or runtime_inventory["schema_version"] != 1
        or runtime_inventory.get("package_inventory_sha256")
        != vllm_environment.get("sha256")
        or runtime_inventory.get("lock_sha256") != lock_sha256
        or runtime_inventory.get("python", {}).get("version")
        != VLLM_PYTHON_VERSION
        or runtime_inventory.get("python", {}).get("prefix")
        != installed_code.get("prefix")
        or not _SHA256.fullmatch(
            str(runtime_inventory.get("python", {}).get("executable_sha256", ""))
        )
        or not _SHA256.fullmatch(
            str(runtime_inventory.get("vllm_entrypoint", {}).get("sha256", ""))
        )
        or not isinstance(cuda_toolkit, dict)
        or not isinstance(cuda_toolkit.get("cuda_home"), str)
        or not Path(cuda_toolkit["cuda_home"]).is_absolute()
        or not isinstance(nvcc, dict)
        or not isinstance(nvcc.get("path"), str)
        or not Path(nvcc["path"]).is_absolute()
        or not isinstance(nvcc.get("resolved_path"), str)
        or not Path(nvcc["resolved_path"]).is_absolute()
        or not _SHA256.fullmatch(str(nvcc.get("sha256", "")))
        or not isinstance(nvcc.get("version_text"), str)
        or not nvcc["version_text"]
        or nvcc.get("version_sha256")
        != sha256_text(nvcc["version_text"])
        or not isinstance(torch_identity, dict)
        or not isinstance(torch_identity.get("version"), str)
        or not torch_identity["version"]
        or not isinstance(torch_identity.get("cuda_version"), str)
        or not torch_identity["cuda_version"]
    ):
        return False

    model_inventory = model_cache["inventory"]
    model_files = model_inventory.get("files")
    if (
        type(model_inventory.get("schema_version")) is not int
        or model_inventory["schema_version"] != 1
        or model_inventory.get("attestation_policy")
        != MODEL_CACHE_ATTESTATION_POLICY
        or model_inventory.get("model") != MODEL_ID
        or model_inventory.get("revision") != MODEL_REVISION
        or not isinstance(model_files, list)
        or not model_files
        or type(model_inventory.get("file_count")) is not int
        or model_inventory.get("file_count") != len(model_files)
        or model_inventory.get("tree_sha256") != sha256_json(model_files)
        or type(model_inventory.get("total_bytes")) is not int
        or model_inventory.get("total_bytes")
        != sum(
            row.get("bytes", -1) if isinstance(row, dict) else -1
            for row in model_files
        )
    ):
        return False
    model_paths = []
    for row in model_files:
        if not isinstance(row, dict) or set(row) != {
            "path", "storage_path", "bytes", "sha256"
        }:
            return False
        logical = PurePosixPath(str(row.get("path", "")))
        storage = PurePosixPath(str(row.get("storage_path", "")))
        if (
            logical.is_absolute()
            or storage.is_absolute()
            or not logical.parts
            or not storage.parts
            or any(part in ("", ".", "..") for part in logical.parts)
            or any(part in ("", ".", "..") for part in storage.parts)
            or type(row.get("bytes")) is not int
            or row["bytes"] < 0
            or not _SHA256.fullmatch(str(row.get("sha256", "")))
        ):
            return False
        model_paths.append(str(logical))
    if model_paths != sorted(model_paths) or len(model_paths) != len(set(model_paths)):
        return False

    allocation_inventory = allocation["inventory"]
    gpus = allocation_inventory.get("gpus")
    slurm = allocation_inventory.get("slurm")
    if (
        type(allocation_inventory.get("schema_version")) is not int
        or allocation_inventory["schema_version"] != 1
        or not isinstance(gpus, list)
        or not gpus
        or type(allocation_inventory.get("gpu_count")) is not int
        or allocation_inventory.get("gpu_count") != len(gpus)
        or not isinstance(slurm, dict)
        or slurm.get("job_id") != environment.get("slurm_job_id")
        or allocation_inventory.get("cuda_visible_devices")
        != environment.get("cuda_visible_devices")
    ):
        return False
    runner_allocation = environment.get("runner_allocation")
    expected_runner_allocation = {
        "slurm_job_id": slurm.get("job_id"),
        "slurm_job_gpus": slurm.get("job_gpus"),
        "slurm_step_gpus": slurm.get("step_gpus"),
        "slurm_job_nodelist": slurm.get("job_nodelist"),
        "slurm_node_id": slurm.get("node_id"),
        "cuda_visible_devices": allocation_inventory.get("cuda_visible_devices"),
        "hostname": allocation_inventory.get("hostname"),
    }
    if runner_allocation != expected_runner_allocation:
        return False
    cuda_identifiers = str(environment.get("cuda_visible_devices", "")).split(",")
    if len(cuda_identifiers) != len(gpus):
        return False
    for expected_identifier, row in zip(cuda_identifiers, gpus, strict=True):
        if (
            not isinstance(row, dict)
            or set(row) != {
                "cuda_identifier", "uuid", "name", "memory_mib", "driver_version"
            }
            or row.get("cuda_identifier") != expected_identifier
            or not isinstance(row.get("uuid"), str)
            or not row["uuid"]
            or not isinstance(row.get("name"), str)
            or not row["name"]
            or type(row.get("memory_mib")) is not int
            or row["memory_mib"] <= 0
            or not isinstance(row.get("driver_version"), str)
            or not row["driver_version"]
        ):
            return False
    expected_models = sorted({row["name"] for row in gpus})
    expected_drivers = sorted({row["driver_version"] for row in gpus})
    if (
        environment.get("gpu_models") != expected_models
        or environment.get("nvidia_driver") != expected_drivers
        or environment.get("vllm_version") != VLLM_VERSION
        or environment.get("vllm_environment_sha256")
        != vllm_environment.get("sha256")
        or environment.get("model_id") != MODEL_ID
        or environment.get("model_revision") != MODEL_REVISION
        or environment.get("server_launch_id")
        != environment.get("vllm_api_key_sha256")
        or not _SHA256.fullmatch(str(environment.get("server_launch_id", "")))
    ):
        return False
    raw_counts = [
        environment.get("tensor_parallel_size"),
        environment.get("visible_gpu_count"),
        environment.get("server_count"),
    ]
    if any(
        not isinstance(value, str) or not re.fullmatch(r"[1-9][0-9]*", value)
        for value in raw_counts
    ):
        return False
    tp, gpu_count, server_count = (int(value) for value in raw_counts)
    return (
        tp > 0
        and gpu_count == len(gpus)
        and server_count > 0
        and tp * server_count == gpu_count
    )


def normalized_environment(environment: object) -> dict[str, Any]:
    """Return the stable research identity of one exact launch environment.

    Slurm allocation IDs, hostnames, physical GPU UUIDs, CUDA ordinals, launcher
    inventory paths, and the ephemeral API identity necessarily change when a
    run resumes in a new allocation.  Everything else remains substantive:
    packages, executables, model bytes/revision, CUDA runtime, driver, GPU class
    and memory, tensor-parallel topology, and server count are all preserved.

    The caller must retain the unnormalized record as the audit artifact.  This
    function is only the explicit compatibility policy for cross-allocation
    resumes and paired comparisons.
    """

    if not isinstance(environment, dict):
        raise ValueError("environment must be a JSON object")
    value = deepcopy(environment)
    for field in (
        "slurm_job_id",
        "server_launch_id",
        "vllm_api_key_sha256",
        "cuda_visible_devices",
    ):
        if field in value:
            value[field] = "<ALLOCATION-TRANSPORT-IDENTITY>"

    for field in ("vllm_environment", "vllm_runtime", "model_cache"):
        snapshot = value.get(field)
        if isinstance(snapshot, dict) and "path" in snapshot:
            snapshot["path"] = "<LAUNCHER-INVENTORY-PATH>"

    allocation = value.get("allocation")
    if isinstance(allocation, dict):
        for field in ("path", "sha256", "bytes"):
            if field in allocation:
                allocation[field] = "<ALLOCATION-SNAPSHOT-IDENTITY>"
        inventory = allocation.get("inventory")
        if isinstance(inventory, dict):
            for field in ("hostname", "cuda_visible_devices"):
                if field in inventory:
                    inventory[field] = "<ALLOCATION-IDENTITY>"
            slurm = inventory.get("slurm")
            if isinstance(slurm, dict):
                for field in slurm:
                    slurm[field] = "<SLURM-ALLOCATION-IDENTITY>"
            gpus = inventory.get("gpus")
            if isinstance(gpus, list):
                for gpu in gpus:
                    if isinstance(gpu, dict):
                        for field in ("cuda_identifier", "uuid"):
                            if field in gpu:
                                gpu[field] = "<ALLOCATED-GPU-IDENTITY>"
    runner_allocation = value.get("runner_allocation")
    if isinstance(runner_allocation, dict):
        for field in runner_allocation:
            runner_allocation[field] = "<RUNNER-ALLOCATION-IDENTITY>"
    return value


def environments_compatible(baseline: object, candidate: object) -> bool:
    """Whether two exact environments differ only in declared launch nuisances."""

    try:
        return canonical_json_bytes(
            normalized_environment(baseline)
        ) == canonical_json_bytes(normalized_environment(candidate))
    except (TypeError, ValueError):
        return False


def environment_contract_record(environment: object) -> dict[str, object]:
    """Hash the normalized substantive identity stored beside an exact baseline."""

    normalized = normalized_environment(environment)
    return {
        "schema_version": 1,
        "policy": ENVIRONMENT_COMPATIBILITY_POLICY,
        "sha256": sha256_json(normalized),
    }


def environment_contract_is_valid(record: object, environment: object) -> bool:
    """Validate a stored normalized-contract fingerprint without bool/int coercion."""

    try:
        return (
            isinstance(record, dict)
            and canonical_json_bytes(record)
            == canonical_json_bytes(environment_contract_record(environment))
        )
    except (TypeError, ValueError):
        return False


def _validate_local_judge_runtime_record(record: object) -> None:
    """Validate the compact, allocation-normalized local judge identity."""

    if not isinstance(record, dict) or set(record) != {
        "schema_version",
        "attestation_policy",
        "environment_contract",
        "model",
        "server",
        "hardware",
    }:
        raise ValueError("local judge runtime fields are invalid")
    contract = record.get("environment_contract")
    model = record.get("model")
    server = record.get("server")
    hardware = record.get("hardware")
    if (
        type(record.get("schema_version")) is not int
        or record["schema_version"] != 1
        or record.get("attestation_policy")
        != LOCAL_JUDGE_RUNTIME_ATTESTATION_POLICY
        or not isinstance(contract, dict)
        or set(contract) != {"schema_version", "policy", "sha256"}
        or type(contract.get("schema_version")) is not int
        or contract["schema_version"] != 1
        or contract.get("policy") != ENVIRONMENT_COMPATIBILITY_POLICY
        or not _SHA256.fullmatch(str(contract.get("sha256", "")))
        or not isinstance(model, dict)
        or set(model) != {
            "id",
            "revision",
            "cache_inventory_sha256",
            "cache_file_count",
            "cache_total_bytes",
            "cache_tree_sha256",
        }
        or model.get("id") != MODEL_ID
        or model.get("revision") != MODEL_REVISION
        or not _SHA256.fullmatch(str(model.get("cache_inventory_sha256", "")))
        or type(model.get("cache_file_count")) is not int
        or model["cache_file_count"] <= 0
        or type(model.get("cache_total_bytes")) is not int
        or model["cache_total_bytes"] <= 0
        or not _SHA256.fullmatch(str(model.get("cache_tree_sha256", "")))
        or not isinstance(server, dict)
        or set(server) != {
            "vllm_version",
            "installed_inventory_sha256",
            "installed_distribution_count",
            "installed_file_count",
            "installed_total_bytes",
            "installed_tree_sha256",
            "runtime_inventory_sha256",
            "tensor_parallel_size",
            "visible_gpu_count",
            "server_count",
        }
        or server.get("vllm_version") != VLLM_VERSION
        or any(
            not _SHA256.fullmatch(str(server.get(field, "")))
            for field in (
                "installed_inventory_sha256",
                "installed_tree_sha256",
                "runtime_inventory_sha256",
            )
        )
        or any(
            type(server.get(field)) is not int or server[field] <= 0
            for field in (
                "installed_distribution_count",
                "installed_file_count",
                "installed_total_bytes",
                "tensor_parallel_size",
                "visible_gpu_count",
                "server_count",
            )
        )
        or server["tensor_parallel_size"] * server["server_count"]
        != server["visible_gpu_count"]
        or not isinstance(hardware, dict)
        or set(hardware) != {"gpu_models", "nvidia_driver", "gpu_profiles"}
        or not isinstance(hardware.get("gpu_models"), list)
        or not hardware["gpu_models"]
        or any(
            not isinstance(value, str) or not value
            for value in hardware["gpu_models"]
        )
        or hardware["gpu_models"] != sorted(set(hardware["gpu_models"]))
        or not isinstance(hardware.get("nvidia_driver"), list)
        or not hardware["nvidia_driver"]
        or any(
            not isinstance(value, str) or not value
            for value in hardware["nvidia_driver"]
        )
        or hardware["nvidia_driver"] != sorted(set(hardware["nvidia_driver"]))
        or not isinstance(hardware.get("gpu_profiles"), list)
        or not hardware["gpu_profiles"]
        or hardware["gpu_profiles"]
        != sorted(
            hardware["gpu_profiles"],
            key=lambda row: (
                str(row.get("name", "")) if isinstance(row, dict) else "",
                row.get("memory_mib", -1)
                if isinstance(row, dict)
                and type(row.get("memory_mib")) is int
                else -1,
                str(row.get("driver_version", ""))
                if isinstance(row, dict)
                else "",
            ),
        )
    ):
        raise ValueError("local judge runtime identity is invalid")
    for profile in hardware["gpu_profiles"]:
        if (
            not isinstance(profile, dict)
            or set(profile) != {"name", "memory_mib", "driver_version", "count"}
            or not isinstance(profile.get("name"), str)
            or not profile["name"]
            or type(profile.get("memory_mib")) is not int
            or profile["memory_mib"] <= 0
            or not isinstance(profile.get("driver_version"), str)
            or not profile["driver_version"]
            or type(profile.get("count")) is not int
            or profile["count"] <= 0
        ):
            raise ValueError("local judge GPU profile is invalid")
    if sum(profile["count"] for profile in hardware["gpu_profiles"]) != server[
        "visible_gpu_count"
    ]:
        raise ValueError("local judge GPU profile count is inconsistent")
    canonical_json_bytes(record)


@lru_cache(maxsize=1)
def _local_judge_runtime_record_bytes() -> bytes:
    """Capture one validated local judge launch and discard its large inventories."""

    environment = environment_record()
    if not environment_is_claim_ready(environment):
        errors = environment.get("inventory_errors")
        detail = f": {errors}" if errors else ""
        raise ValueError(f"local judge launch environment is not claim-ready{detail}")
    installed_code = environment["vllm_environment"]["inventory"]
    model_cache = environment["model_cache"]["inventory"]
    allocation = environment["allocation"]["inventory"]
    profiles: dict[tuple[str, int, str], int] = {}
    for gpu in allocation["gpus"]:
        key = (gpu["name"], gpu["memory_mib"], gpu["driver_version"])
        profiles[key] = profiles.get(key, 0) + 1
    record = {
        "schema_version": 1,
        "attestation_policy": LOCAL_JUDGE_RUNTIME_ATTESTATION_POLICY,
        # This commits to every validated substantive field in the full launch
        # record while excluding only the explicit allocation/transport nuisances.
        "environment_contract": environment_contract_record(environment),
        "model": {
            "id": environment["model_id"],
            "revision": environment["model_revision"],
            "cache_inventory_sha256": environment["model_cache"]["sha256"],
            "cache_file_count": model_cache["file_count"],
            "cache_total_bytes": model_cache["total_bytes"],
            "cache_tree_sha256": model_cache["tree_sha256"],
        },
        "server": {
            "vllm_version": environment["vllm_version"],
            "installed_inventory_sha256": environment["vllm_environment"]["sha256"],
            "installed_distribution_count": installed_code["distribution_count"],
            "installed_file_count": installed_code["file_count"],
            "installed_total_bytes": installed_code["total_bytes"],
            "installed_tree_sha256": installed_code["tree_sha256"],
            "runtime_inventory_sha256": environment["vllm_runtime"]["sha256"],
            "tensor_parallel_size": int(environment["tensor_parallel_size"]),
            "visible_gpu_count": int(environment["visible_gpu_count"]),
            "server_count": int(environment["server_count"]),
        },
        "hardware": {
            "gpu_models": environment["gpu_models"],
            "nvidia_driver": environment["nvidia_driver"],
            "gpu_profiles": [
                {
                    "name": name,
                    "memory_mib": memory_mib,
                    "driver_version": driver,
                    "count": count,
                }
                for (name, memory_mib, driver), count in sorted(profiles.items())
            ],
        },
    }
    _validate_local_judge_runtime_record(record)
    return canonical_json_bytes(record)


def local_judge_runtime_record() -> dict[str, object]:
    """Return the compact substantive identity of the live local judge launch."""

    record = strict_json_loads(
        _local_judge_runtime_record_bytes(), label="local judge runtime attestation"
    )
    if not isinstance(record, dict):  # pragma: no cover - guarded by construction
        raise ValueError("local judge runtime attestation is not an object")
    return record


def local_judge_runtime_sha256(record: object | None = None) -> str:
    """Hash a compact local judge identity after structural validation."""

    value = local_judge_runtime_record() if record is None else record
    _validate_local_judge_runtime_record(value)
    return sha256_json(value)


def validate_local_judge_runtime_record(
    record: object, *, require_current: bool = True
) -> None:
    """Validate a stored local judge identity and optionally reattest the launch."""

    _validate_local_judge_runtime_record(record)
    if require_current and canonical_json_bytes(record) != _local_judge_runtime_record_bytes():
        raise ValueError("local judge runtime differs from the recorded attestation")


def write_environment_snapshot(
    root: Path,
    relative_directory: PurePosixPath,
    environment: dict[str, object],
) -> dict[str, object]:
    """Write an exact content-addressed launch record below an artifact root."""

    if (
        not isinstance(relative_directory, PurePosixPath)
        or relative_directory.is_absolute()
        or not relative_directory.parts
        or any(part in ("", ".", "..") for part in relative_directory.parts)
        or "\\" in str(relative_directory)
    ):
        raise ValueError("environment snapshot directory must be a safe relative path")
    try:
        data = canonical_json_bytes(environment)
    except (TypeError, ValueError) as error:
        raise ValueError("environment snapshot is not canonical JSON data") from error
    digest = sha256_bytes(data)
    relative = relative_directory / f"environment-{digest}.json"
    write_immutable_bytes(root.joinpath(*relative.parts), data)
    return {
        "schema_version": 1,
        "sha256": digest,
        "bytes": len(data),
        "snapshot": str(relative),
    }


def validate_environment_snapshot(
    root: Path,
    record: object,
    *,
    baseline: object,
    require_claim_ready: bool,
) -> dict[str, object]:
    """Revalidate an exact launch snapshot and its stable baseline contract."""

    if type(require_claim_ready) is not bool:
        raise ValueError("require_claim_ready must be a boolean")
    if (
        not isinstance(record, dict)
        or set(record) != {"schema_version", "sha256", "bytes", "snapshot"}
        or type(record.get("schema_version")) is not int
        or record["schema_version"] != 1
        or not isinstance(record.get("sha256"), str)
        or not _SHA256.fullmatch(record["sha256"])
        or type(record.get("bytes")) is not int
        or record["bytes"] < 0
        or not isinstance(record.get("snapshot"), str)
        or not record["snapshot"]
        or "\\" in record["snapshot"]
    ):
        raise ValueError("environment snapshot record is invalid")
    relative = PurePosixPath(record["snapshot"])
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in ("", ".", "..") for part in relative.parts)
        or str(relative) != record["snapshot"]
    ):
        raise ValueError("environment snapshot path is unsafe")
    path = root.joinpath(*relative.parts)
    try:
        data = read_artifact_bytes(path)
        environment = strict_json_loads(data, label="launch environment snapshot")
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError("environment snapshot is missing or invalid") from error
    if (
        len(data) != record["bytes"]
        or sha256_bytes(data) != record["sha256"]
        or not isinstance(environment, dict)
        or canonical_json_bytes(environment) != data
    ):
        raise ValueError("environment snapshot bytes do not match their record")
    if not environments_compatible(baseline, environment):
        raise ValueError("launch environment has substantive drift from its baseline")
    if require_claim_ready and not environment_is_claim_ready(environment):
        raise ValueError("launch environment snapshot is not claim-ready")
    return environment


def _load_note(
    run_root: Path,
    note_path: Path | None,
    note_manifest_path: Path | None,
    *,
    require_manifest: bool,
    require_claim_ready: bool | None = None,
    allow_smoke: bool = False,
    expected_task: str | None = None,
    expected_model: str | None = None,
    expected_model_revision: str | None = None,
    expected_response_model: str | None = None,
    expected_sampling: dict[str, object] | None = None,
    expected_corpus_commit: str | None = None,
    expected_corpus: dict[str, object] | None = None,
    expected_source: dict[str, object] | None = None,
    expected_environment: dict[str, object] | None = None,
    expected_corpus_display: str | None = None,
    expected_note_sha256: str | None = None,
    expected_note_manifest_sha256: str | None = None,
) -> tuple[str, dict[str, object] | None]:
    if type(require_manifest) is not bool or type(allow_smoke) is not bool:
        raise ValueError("require_manifest and allow_smoke must be boolean")
    if require_claim_ready is None:
        # Preserve the historical direct-call contract while allowing
        # prepare_run to distinguish full exploratory runs from confirmation.
        require_claim_ready = require_manifest
    if type(require_claim_ready) is not bool:
        raise ValueError("require_claim_ready must be boolean")
    if require_claim_ready and not require_manifest:
        raise ValueError("claim-ready notes require a construction manifest")
    protocol_expectations_supplied = any(
        value is not None
        for value in (
            expected_model,
            expected_model_revision,
            expected_response_model,
            expected_sampling,
            expected_corpus,
            expected_source,
            expected_environment,
            expected_corpus_display,
        )
    )

    def contained_regular_file(parent: Path, relative: Path, label: str) -> Path:
        """Validate a manifest-relative file without resolving through symlinks."""

        if (
            relative.is_absolute()
            or not relative.parts
            or any(part in ("", ".", "..") for part in relative.parts)
        ):
            raise ValueError(f"note manifest has unsafe {label}")
        candidate = parent.absolute() / relative
        try:
            read_artifact_bytes(candidate)
        except (OSError, ValueError) as error:
            raise ValueError(f"note manifest is missing or has unsafe {label}") from error
        return candidate

    if note_path is None:
        if note_manifest_path is not None:
            raise ValueError("--note-manifest requires --note")
        return "", None
    note_path = note_path.absolute()
    note_bytes = read_artifact_bytes(note_path)
    try:
        note = note_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"note is not valid UTF-8: {note_path}") from error
    if not note.strip():
        raise ValueError(f"note is empty: {note_path}")
    note_hash = sha256_bytes(note_bytes)
    if expected_note_sha256 is not None and note_hash != expected_note_sha256:
        raise ValueError("note bytes differ from the existing run manifest")

    auxiliary_artifacts: dict[str, tuple[Path, bytes]] = {}
    construction_dependencies: dict[str, tuple[dict[str, object], bytes]] = {}
    construction_inventory_sha256: str | None = None
    protocol_summary: dict[str, object] | None = None
    forced50_protocol: dict[str, object] | None = None
    relative_note: Path | None = None
    if note_manifest_path is None:
        if require_manifest:
            raise ValueError("research runs with --note also require --note-manifest")
    else:
        note_manifest_path = note_manifest_path.absolute()
        manifest_bytes = read_artifact_bytes(note_manifest_path)
        manifest_hash = sha256_bytes(manifest_bytes)
        if (expected_note_manifest_sha256 is not None
                and manifest_hash != expected_note_manifest_sha256):
            raise ValueError("note construction manifest differs from the existing run")
        try:
            construction = strict_json_loads(
                manifest_bytes, label="note construction manifest"
            )
        except ValueError as error:
            raise ValueError(f"invalid note manifest: {note_manifest_path}") from error
        if not isinstance(construction, dict) or construction.get("note_sha256") != note_hash:
            raise ValueError("note bytes do not match note_manifest.note_sha256")
        if require_manifest:
            if construction.get("task") != expected_task:
                raise ValueError("note manifest task does not match the evaluation task")
            if construction.get("corpus_commit") != expected_corpus_commit:
                raise ValueError("note manifest corpus commit does not match the evaluation corpus")
            try:
                validate_id(construction.get("study_id"), "study ID")
            except (TypeError, ValueError) as error:
                raise ValueError("note manifest has no valid study_id") from error
            claim_ready = construction.get("claim_ready")
            if claim_ready is None and isinstance(construction.get("config"), dict):
                claim_ready = construction["config"].get("claim_ready")
            if require_claim_ready and claim_ready is not True:
                raise ValueError("note construction manifest is not claim-ready")
            if claim_ready not in (True, False) or type(claim_ready) is not bool:
                raise ValueError("note construction manifest has invalid claim readiness")
            raw_note_path = construction.get("note_path")
            relative_note = Path(str(raw_note_path))
            if (not isinstance(raw_note_path, str) or not raw_note_path
                    or "\\" in raw_note_path or relative_note.is_absolute()
                    or any(part in ("", ".", "..") for part in relative_note.parts)
                    or relative_note.as_posix() != raw_note_path):
                raise ValueError("note manifest contains an unsafe note_path")
            recorded_note = contained_regular_file(
                note_manifest_path.parent, relative_note, "recorded note artifact")
            if sha256_bytes(read_artifact_bytes(recorded_note)) != note_hash:
                raise ValueError("note manifest's recorded note artifact is missing or changed")

            manifest_type = construction.get("manifest_type")
            if claim_ready is False:
                readiness = construction.get("automated_readiness")
                inventory = construction.get("construction_artifacts")
                construction_inventory_sha256 = construction.get(
                    "construction_artifacts_sha256")
                contradictory_readiness = any(
                    field in construction and construction.get(field) is not False
                    for field in (
                        "publication_claim_ready",
                        "confirmatory_claim_ready",
                    )
                )
                readiness_shape_valid = (
                    isinstance(readiness, dict)
                    and bool(readiness)
                    and all(type(value) is bool for value in readiness.values())
                )
                full_automated_gate = (
                    readiness_shape_valid
                    and construction.get("automated_claim_ready") is True
                    and all(readiness.values())
                )
                smoke_automated_gate = (
                    allow_smoke
                    and readiness_shape_valid
                    and construction.get("automated_claim_ready") is False
                    and readiness.get("non_smoke") is False
                )
                if (type(construction.get("schema_version")) is not int
                        or construction["schema_version"] <= 0
                        or contradictory_readiness
                        or not (full_automated_gate or smoke_automated_gate)
                        or not isinstance(inventory, dict) or not inventory
                        or not isinstance(construction_inventory_sha256, str)
                        or not re.fullmatch(
                            r"[0-9a-f]{64}", construction_inventory_sha256)
                        or sha256_json(inventory) != construction_inventory_sha256):
                    raise ValueError(
                        "exploratory note did not pass its automated construction gates")
                study_root = note_manifest_path.parent.parent.absolute()
                for raw_relative, artifact_record in inventory.items():
                    if (not isinstance(raw_relative, str) or not raw_relative
                            or "\\" in raw_relative):
                        raise ValueError(
                            "construction inventory contains an unsafe path")
                    relative = PurePosixPath(raw_relative)
                    if (relative.is_absolute() or ".." in relative.parts
                            or str(relative) != raw_relative or not relative.parts):
                        raise ValueError(
                            "construction inventory contains an unsafe path")
                    if (not isinstance(artifact_record, dict)
                            or set(artifact_record) != {"sha256", "bytes"}
                            or not isinstance(artifact_record.get("sha256"), str)
                            or not re.fullmatch(
                                r"[0-9a-f]{64}", artifact_record["sha256"])
                            or type(artifact_record.get("bytes")) is not int
                            or artifact_record["bytes"] < 0):
                        raise ValueError(
                            f"construction inventory metadata is invalid: {raw_relative}")
                    artifact = study_root.joinpath(*relative.parts)
                    try:
                        data = read_artifact_bytes(artifact)
                    except (OSError, ValueError) as error:
                        raise ValueError(
                            f"construction dependency is missing: {raw_relative}") from error
                    if (len(data) != artifact_record["bytes"]
                            or sha256_bytes(data) != artifact_record["sha256"]):
                        raise ValueError(
                            f"construction dependency changed: {raw_relative}")
                    construction_dependencies[raw_relative] = (artifact_record, data)
                if manifest_type not in {
                    SEMANTIC_SELFQUIZ_NOTE_MANIFEST_TYPE,
                    STATIC_GRAPH_NOTE_MANIFEST_TYPE,
                }:
                    raise ValueError(
                        "exploratory note type does not match a recognized protocol"
                    )
                try:
                    protocol_summary = validate_study_note_archive(
                        construction,
                        {
                            path: data
                            for path, (_, data) in construction_dependencies.items()
                        },
                        note_bytes,
                        expected_task=expected_task,
                        expected_model=expected_model,
                        expected_model_revision=expected_model_revision,
                        expected_sampling=expected_sampling,
                        expected_corpus_commit=expected_corpus_commit,
                        expected_corpus=expected_corpus,
                        expected_source=expected_source,
                        expected_environment=expected_environment,
                        expected_environment_contract=(
                            environment_contract_record(expected_environment)
                            if expected_environment is not None else None
                        ),
                        environments_compatible=environments_compatible,
                        require_final_semantic=True,
                        allow_smoke=allow_smoke,
                    )
                except StudyProtocolError as error:
                    raise ValueError(
                        f"exploratory note protocol binding is invalid: {error}"
                    ) from error
            elif manifest_type == HUMAN_AUDITED_NOTE_MANIFEST_TYPE:
                if "automated_readiness" not in construction:
                    raise ValueError("human-audited note has no automated readiness record")
                human = construction.get("human_audit")
                base_record = construction.get("construction_manifest")
                if (not isinstance(human, dict) or human.get("status") != "passed"
                        or not isinstance(base_record, dict)):
                    raise ValueError("human-audited note has no passing audit chain")

                def load_relative(record: dict[str, object], path_key: str,
                                  hash_key: str, label: str) -> tuple[Path, bytes]:
                    relative = Path(str(record.get(path_key, "")))
                    if (
                        relative.is_absolute()
                        or not relative.parts
                        or any(part in ("", ".", "..") for part in relative.parts)
                    ):
                        raise ValueError(f"human-audited note has unsafe {label} path")
                    artifact = contained_regular_file(
                        note_manifest_path.parent, relative, label)
                    try:
                        data = read_artifact_bytes(artifact)
                    except (OSError, ValueError) as error:
                        raise ValueError(f"human-audited note is missing {label}") from error
                    if sha256_bytes(data) != record.get(hash_key):
                        raise ValueError(f"human-audited note {label} hash mismatch")
                    auxiliary_artifacts[label] = (relative, data)
                    return relative, data

                _, base_bytes = load_relative(
                    base_record, "path", "sha256", "construction_manifest")
                _, audit_bytes = load_relative(
                    human, "result_path", "result_sha256", "human_audit_result")
                _, protocol_bytes = load_relative(
                    human, "protocol_path", "protocol_sha256", "human_audit_protocol")
                try:
                    base = strict_json_loads(
                        base_bytes, label="human-audit construction manifest"
                    )
                    audit_result = strict_json_loads(
                        audit_bytes, label="human-audit result"
                    )
                    audit_protocol = strict_json_loads(
                        protocol_bytes, label="human-audit protocol"
                    )
                except ValueError as error:
                    raise ValueError("human-audit chain contains invalid JSON") from error
                shared = ("study_id", "task", "round", "corpus_commit", "note_sha256",
                          "note_path", "entry_ids", "entries", "usage",
                          "method", "protocol_summary",
                          "automated_claim_ready", "automated_readiness",
                          "construction_artifacts", "construction_artifacts_sha256")
                if (not isinstance(base, dict) or base.get("claim_ready") is not False
                        or base.get("automated_claim_ready") is not True
                        or type(base.get("round")) is not int
                        or type(construction.get("round")) is not int
                        or any(base.get(key) != construction.get(key) for key in shared)):
                    raise ValueError("audited note drifted from its automated construction manifest")
                inventory = base.get("construction_artifacts")
                construction_inventory_sha256 = base.get("construction_artifacts_sha256")
                if (not isinstance(inventory, dict) or not inventory
                        or not isinstance(construction_inventory_sha256, str)
                        or not re.fullmatch(r"[0-9a-f]{64}", construction_inventory_sha256)
                        or sha256_json(inventory) != construction_inventory_sha256):
                    raise ValueError("self-study construction inventory is missing or invalid")
                study_root = note_manifest_path.parent.parent.absolute()
                for raw_relative, artifact_record in inventory.items():
                    if (not isinstance(raw_relative, str) or not raw_relative
                            or "\\" in raw_relative):
                        raise ValueError("construction inventory contains an unsafe path")
                    relative = PurePosixPath(raw_relative)
                    if (relative.is_absolute() or ".." in relative.parts
                            or str(relative) != raw_relative or not relative.parts):
                        raise ValueError("construction inventory contains an unsafe path")
                    if (not isinstance(artifact_record, dict)
                            or set(artifact_record) != {"sha256", "bytes"}
                            or not isinstance(artifact_record.get("sha256"), str)
                            or not re.fullmatch(r"[0-9a-f]{64}", artifact_record["sha256"])
                            or type(artifact_record.get("bytes")) is not int
                            or artifact_record["bytes"] < 0):
                        raise ValueError(
                            f"construction inventory metadata is invalid: {raw_relative}"
                        )
                    artifact = study_root.joinpath(*relative.parts)
                    try:
                        data = read_artifact_bytes(artifact)
                    except (OSError, ValueError) as error:
                        raise ValueError(
                            f"construction dependency is missing: {raw_relative}"
                        ) from error
                    if (len(data) != artifact_record["bytes"]
                            or sha256_bytes(data) != artifact_record["sha256"]):
                        raise ValueError(
                            f"construction dependency changed: {raw_relative}"
                        )
                    construction_dependencies[raw_relative] = (artifact_record, data)
                if protocol_expectations_supplied or "protocol_summary" in construction:
                    if base.get("manifest_type") not in {
                        SEMANTIC_SELFQUIZ_NOTE_MANIFEST_TYPE,
                        STATIC_GRAPH_NOTE_MANIFEST_TYPE,
                    }:
                        raise ValueError(
                            "human-audited base note type is not a recognized protocol"
                        )
                    dependency_bytes = {
                        path: data
                        for path, (_, data) in construction_dependencies.items()
                    }
                    try:
                        base_summary = validate_study_note_archive(
                            base,
                            dependency_bytes,
                            note_bytes,
                            expected_task=expected_task,
                            expected_model=expected_model,
                            expected_model_revision=expected_model_revision,
                            expected_sampling=expected_sampling,
                            expected_corpus_commit=expected_corpus_commit,
                            expected_corpus=expected_corpus,
                            expected_source=expected_source,
                            expected_environment=expected_environment,
                            expected_environment_contract=(
                                environment_contract_record(expected_environment)
                                if expected_environment is not None else None
                            ),
                            environments_compatible=environments_compatible,
                            require_final_semantic=True,
                            allow_smoke=allow_smoke,
                        )
                        protocol_summary = validate_construction_protocol(
                            construction,
                            dependency_bytes,
                            expected_task=expected_task,
                            expected_model=expected_model,
                            expected_model_revision=expected_model_revision,
                            expected_sampling=expected_sampling,
                            expected_corpus_commit=expected_corpus_commit,
                            expected_corpus=expected_corpus,
                            expected_source=expected_source,
                            expected_environment=expected_environment,
                            expected_environment_contract=(
                                environment_contract_record(expected_environment)
                                if expected_environment is not None else None
                            ),
                            environments_compatible=environments_compatible,
                            allow_human_audited=True,
                            require_final_semantic=True,
                        )
                    except StudyProtocolError as error:
                        raise ValueError(
                            f"human-audited note protocol binding is invalid: {error}"
                        ) from error
                    if protocol_summary != base_summary:
                        raise ValueError(
                            "human-audited note protocol differs from its base construction"
                        )
                try:
                    audit_validation = validate_human_audit_result(
                        audit_result,
                        base,
                        {
                            path: data
                            for path, (_, data) in construction_dependencies.items()
                        },
                    )
                except HumanAuditError as error:
                    raise ValueError(
                        f"human-audit population or decision is invalid: {error}"
                    ) from error
                if not audit_validation.passed:
                    raise ValueError("human-audited note does not contain a passing audit")
                try:
                    auditor_id = validate_id(audit_result.get("auditor_id"), "auditor ID")
                except (TypeError, ValueError) as error:
                    raise ValueError("human-audit result has an invalid auditor ID") from error
                audit_expected = {
                    "schema_version": 1,
                    "study_id": construction["study_id"],
                    "task": construction["task"],
                    "round": construction["round"],
                    "construction_manifest_sha256": base_record["sha256"],
                    "note_sha256": note_hash,
                    "protocol_sha256": human["protocol_sha256"],
                    "auditor_id": auditor_id,
                    "blinding_preserved": True,
                    "reviewer_independent": True,
                    "decision": "pass",
                }
                if (not isinstance(audit_result, dict)
                        or human.get("auditor_id") != auditor_id
                        or type(audit_result.get("schema_version")) is not int
                        or type(audit_result.get("round")) is not int
                        or type(construction.get("round")) is not int
                        or audit_result.get("blinding_preserved") is not True
                        or audit_result.get("reviewer_independent") is not True
                        or any(audit_result.get(key) != value
                               for key, value in audit_expected.items())):
                    raise ValueError("human-audit result does not bind the promoted note")
                try:
                    validate_human_audit_protocol(audit_protocol)
                except HumanAuditError as error:
                    raise ValueError(
                        "human-audit protocol is not the required blinded protocol"
                    ) from error
            elif manifest_type == "forced-50-cheatsheet":
                config = construction.get("config")
                inventory = construction.get("construction_artifacts")
                construction_inventory_sha256 = construction.get(
                    "construction_artifacts_sha256")
                if (type(construction.get("manifest_schema")) is not int
                        or construction["manifest_schema"] != 1
                        or not isinstance(config, dict)
                        or config.get("method") != "forced-50-cheatsheet"
                        or config.get("claim_ready") is not True
                        or config.get("study_id") != construction.get("study_id")
                        or config.get("task") != construction.get("task")
                        or not isinstance(config.get("corpus"), dict)
                        or config["corpus"].get("commit")
                        != construction.get("corpus_commit")
                        or not isinstance(inventory, dict)
                        or set(inventory) != {"intent.json", "episode.json"}
                        or construction_inventory_sha256 != sha256_json(inventory)):
                    raise ValueError("forced-50 construction manifest is incomplete")
                if expected_corpus_display is not None:
                    try:
                        forced50_protocol = validate_forced50_config(
                            config,
                            corpus_display=expected_corpus_display,
                            expected_task=expected_task,
                            expected_model=expected_model,
                            expected_model_revision=expected_model_revision,
                            expected_response_model=expected_response_model,
                            expected_sampling=expected_sampling,
                            expected_corpus=expected_corpus,
                            expected_source=expected_source,
                            expected_environment=expected_environment,
                            environments_compatible=environments_compatible,
                        )
                    except StudyProtocolError as error:
                        raise ValueError(
                            f"forced-50 protocol binding is invalid: {error}"
                        ) from error
                elif protocol_expectations_supplied:
                    raise ValueError(
                        "forced-50 preflight requires the evaluation corpus display name"
                    )
                loaded_dependencies = {}
                for raw_relative, artifact_record in inventory.items():
                    if (not isinstance(artifact_record, dict)
                            or set(artifact_record) != {"sha256", "bytes"}
                            or not isinstance(artifact_record.get("sha256"), str)
                            or not re.fullmatch(r"[0-9a-f]{64}", artifact_record["sha256"])
                            or type(artifact_record.get("bytes")) is not int
                            or artifact_record["bytes"] < 0):
                        raise ValueError(
                            f"forced-50 construction metadata is invalid: {raw_relative}"
                        )
                    relative = Path(raw_relative)
                    artifact = contained_regular_file(
                        note_manifest_path.parent, relative,
                        f"forced-50 dependency {raw_relative}")
                    data = read_artifact_bytes(artifact)
                    if (len(data) != artifact_record["bytes"]
                            or sha256_bytes(data) != artifact_record["sha256"]):
                        raise ValueError(
                            f"forced-50 construction dependency changed: {raw_relative}"
                        )
                    try:
                        loaded_dependencies[raw_relative] = strict_json_loads(
                            data, label=f"forced-50 dependency {raw_relative}"
                        )
                    except ValueError as error:
                        raise ValueError(
                            f"forced-50 dependency is invalid JSON: {raw_relative}"
                        ) from error
                    construction_dependencies[raw_relative] = (artifact_record, data)
                intent = loaded_dependencies["intent.json"]
                episode = loaded_dependencies["episode.json"]
                if canonical_json_bytes(intent) != construction_dependencies[
                    "intent.json"
                ][1]:
                    raise ValueError("forced-50 intent is not canonically encoded")
                if forced50_protocol is not None:
                    try:
                        validate_forced50_episode(
                            construction_dependencies["episode.json"][1],
                            config=config,
                            expected_note_sha256=note_hash,
                        )
                    except StudyProtocolError as error:
                        raise ValueError(
                            f"forced-50 study episode is invalid: {error}"
                        ) from error
                intent_hash = construction.get("intent_sha256")
                episode_hash = construction.get("episode_sha256")
                integer_identities = (
                    config.get("episode_seed"),
                    episode.get("seed") if isinstance(episode, dict) else None,
                    episode.get("completion_tokens") if isinstance(episode, dict) else None,
                    episode.get("prompt_tokens") if isinstance(episode, dict) else None,
                    episode.get("total_tokens") if isinstance(episode, dict) else None,
                    construction.get("study_generated_tokens"),
                    construction.get("study_prompt_tokens"),
                    construction.get("study_total_tokens"),
                )
                if (any(type(value) is not int for value in integer_identities)
                        or not isinstance(intent, dict) or intent != config
                        or config.get("study_id") != construction.get("study_id")
                        or config.get("task") != construction.get("task")
                        or not isinstance(config.get("corpus"), dict)
                        or config["corpus"].get("commit")
                        != construction.get("corpus_commit")
                        or intent_hash != sha256_json(intent)
                        or not isinstance(episode, dict)
                        or episode_hash != sha256_json(episode)
                        or episode.get("study_intent_sha256") != intent_hash
                        or episode.get("question_sha256")
                        != config.get("study_question_sha256")
                        or episode.get("status") != "ok"
                        or not isinstance(episode.get("answer"), str)
                        or sha256_text(episode["answer"]) != note_hash
                        or episode.get("model") != config.get("model")
                        or episode.get("model_revision") != config.get("model_revision")
                        or episode.get("seed") != config.get("episode_seed")
                        or construction.get("study_generated_tokens")
                        != episode.get("completion_tokens")
                        or construction.get("study_prompt_tokens")
                        != episode.get("prompt_tokens")
                        or construction.get("study_total_tokens")
                        != episode.get("total_tokens")):
                    raise ValueError("forced-50 intent, episode, and note do not bind exactly")
            else:
                raise ValueError(
                    "unknown claim-ready note manifest type; implement an explicit validator"
                )

    snapshot = Path("inputs") / f"note-{note_hash}.md"
    write_immutable_bytes(run_root / snapshot, note_bytes)
    record: dict[str, object] = {
        "sha256": note_hash,
        "bytes": len(note_bytes),
        "snapshot": str(snapshot),
        "source_name": note_path.name,
    }
    if protocol_summary is not None:
        record["protocol_summary"] = protocol_summary
    if forced50_protocol is not None:
        record["forced50_protocol"] = forced50_protocol
    if note_manifest_path is not None:
        manifest_snapshot = Path("inputs") / f"note-manifest-{manifest_hash}.json"
        write_immutable_bytes(run_root / manifest_snapshot, manifest_bytes)
        record["construction_manifest"] = {
            "sha256": manifest_hash,
            "snapshot": str(manifest_snapshot),
        }
        if auxiliary_artifacts or construction_dependencies:
            bundle_root = Path("inputs") / f"note-provenance-{manifest_hash}"
            bundled_manifest = bundle_root / note_manifest_path.name
            write_immutable_bytes(run_root / bundled_manifest, manifest_bytes)
            if relative_note is None:
                raise ValueError("audited note manifest has no safe note_path")
            write_immutable_bytes(run_root / bundle_root / relative_note, note_bytes)
            bundled = {}
            for label, (relative, data) in auxiliary_artifacts.items():
                destination = bundle_root / relative
                write_immutable_bytes(run_root / destination, data)
                bundled[label] = {
                    "sha256": sha256_bytes(data),
                    "snapshot": str(destination),
                }
            record["provenance_bundle"] = {
                "root": str(bundle_root),
                "manifest_snapshot": str(bundled_manifest),
                "note_snapshot": str(bundle_root / relative_note),
            }
            if bundled:
                record["provenance_bundle"]["artifacts"] = bundled
            if construction_dependencies:
                construction_root = bundle_root / "construction"
                construction_snapshots = {}
                for raw_relative, (artifact_record, data) in sorted(
                        construction_dependencies.items()):
                    destination = construction_root.joinpath(
                        *PurePosixPath(raw_relative).parts)
                    write_immutable_bytes(run_root / destination, data)
                    construction_snapshots[raw_relative] = {
                        **artifact_record,
                        "snapshot": str(destination),
                    }
                record["provenance_bundle"]["construction_artifacts"] = {
                    "root": str(construction_root),
                    "inventory_sha256": construction_inventory_sha256,
                    "artifacts": construction_snapshots,
                }
    return note, record


def prepare_run(
    *,
    run_id: str,
    task: str,
    corpus: Corpus,
    questions: list[dict[str, Any]],
    budgets: list[str],
    rollouts: int,
    harness: str,
    model: str,
    model_revision: str,
    sampling: dict[str, object],
    master_seed: int,
    seed_namespace: str,
    seed_group: str,
    note_path: Path | None,
    note_manifest_path: Path | None,
    note_prefix_template: str | None,
    smoke: bool,
    exploratory: bool,
    allow_dirty: bool,
    preregistration_path: Path | None,
    preregistration_role: str | None,
    extra: dict[str, object],
) -> RunContext:
    """Create or validate a task manifest and snapshot its exact note bytes."""

    validate_id(run_id)
    validate_id(seed_group, "seed group")
    if any(type(value) is not bool for value in (smoke, exploratory, allow_dirty)):
        raise ValueError("smoke, exploratory, and allow_dirty must be booleans")
    if smoke and exploratory:
        raise ValueError("a run cannot be both smoke and exploratory")
    if allow_dirty and not smoke:
        raise ValueError("--allow-dirty is restricted to isolated smoke runs")
    if (preregistration_path is None) != (preregistration_role is None):
        raise ValueError("--preregistration and --preregistration-role must be provided together")
    if preregistration_path is None and not (smoke or exploratory):
        raise ValueError(
            "confirmatory runs require a committed preregistration; use --exploratory "
            "for an explicitly non-claim-ready run"
        )
    if not seed_namespace:
        raise ValueError("seed namespace must not be empty")
    if type(master_seed) is not int:
        raise ValueError("master seed must be a JSON integer")
    if type(rollouts) is not int:
        raise ValueError("rollouts must be a JSON integer")
    if (
        not isinstance(harness, str)
        or not harness
        or not isinstance(model, str)
        or not model
        or not isinstance(model_revision, str)
        or not model_revision
        or not isinstance(sampling, dict)
        or not sampling
    ):
        raise ValueError("harness, model, model revision, and sampling must be explicit")
    if not isinstance(extra, dict):
        raise ValueError("extra run configuration must be an object")
    extra_record = dict(extra)
    if extra_record.get("model_revision") != model_revision:
        raise ValueError("extra.model_revision differs from the canonical model revision")
    if (
        not isinstance(extra_record.get("expected_response_model"), str)
        or not extra_record["expected_response_model"]
    ):
        raise ValueError("an explicit expected response-model identity is required")
    if not isinstance(questions, list) or not questions or any(
        not isinstance(question, dict) for question in questions
    ):
        raise ValueError("a run must contain at least one question")
    if (
        rollouts <= 0
        or not isinstance(budgets, list)
        or not budgets
        or any(not isinstance(budget, str) for budget in budgets)
        or len(set(budgets)) != len(budgets)
    ):
        raise ValueError("rollouts must be positive and budgets must be unique/nonempty")
    question_ids = [q.get("id") for q in questions]
    if any(not isinstance(qid, str) or not qid for qid in question_ids):
        raise ValueError("every question must have a nonempty string ID")
    if any(
        not isinstance(question.get("question"), str) or not question["question"]
        for question in questions
    ):
        raise ValueError("every question must have nonempty question text")
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("question IDs must be unique")
    unsafe_components = [
        value for value in [*question_ids, *budgets]
        if not isinstance(value, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value)
    ]
    if unsafe_components:
        raise ValueError(f"question IDs and budgets must be safe path components: {unsafe_components}")
    question_bundle_sha256 = sha256_json(questions)
    requested_note_sha256 = (
        sha256_bytes(read_artifact_bytes(Path(note_path).absolute()))
        if note_path is not None
        else None
    )

    corpus_info = corpus_record(corpus)
    if corpus_info["dirty"]:
        raise ValueError(f"corpus checkout is dirty: {corpus.repo}")
    if corpus_info["commit"] != corpus.commit:
        raise ValueError(
            f"{corpus.name} is at {corpus_info['commit']}, expected pinned {corpus.commit}"
        )
    source = source_record()
    if source["dirty"] and not (allow_dirty or smoke):
        raise ValueError("research source files are dirty; commit them or use a new diagnostic smoke run")
    preregistration = None
    if preregistration_path is not None:
        preregistration = bind_preregistration(
            preregistration_path,
            role=preregistration_role,
            run_id=run_id,
            task=task,
            corpus_commit=corpus_info["commit"],
            source_head_commit=source["git_commit"],
            question_bundle_sha256=question_bundle_sha256,
            harness=harness,
            model=model,
            model_revision=model_revision,
            sampling=sampling,
            master_seed=master_seed,
            seed_namespace=seed_namespace,
            seed_group=seed_group,
            budgets=budgets,
            rollouts=rollouts,
            failure_policy=RUN_FAILURE_POLICY,
            note_sha256=requested_note_sha256,
            root=ROOT,
        )
    environment = environment_record()
    environment_ready = environment_is_claim_ready(environment)
    if harness == "dspy.ReAct":
        packages = environment.get("packages")
        environment_ready = bool(
            environment_ready
            and isinstance(packages, dict)
            and packages.get("dspy")
        )
    if not environment_ready and not (allow_dirty or smoke):
        raise ValueError(
            "research environment is incomplete; run through the pinned server launcher "
            "or use a diagnostic smoke run"
        )

    base = ROOT / "runs" / "smoke" if smoke else ROOT / "runs"
    run_root = base / run_id / task
    manifest_path = run_root / "manifest.json"
    existing_manifest = None
    if manifest_path.exists():
        existing_manifest = load_json_artifact(manifest_path)
    elif run_root.exists() and any(run_root.iterdir()):
        raise ValueError(
            f"refusing to add a manifest to a nonempty legacy/partial run: {run_root}"
        )
    run_root.mkdir(parents=True, exist_ok=True)
    baseline_environment = environment
    if existing_manifest is not None:
        existing_environment = (
            existing_manifest.get("spec", {}).get("environment")
            if isinstance(existing_manifest, dict)
            and isinstance(existing_manifest.get("spec"), dict)
            else None
        )
        if not isinstance(existing_environment, dict):
            raise ValueError("existing run manifest has no valid environment baseline")
        if not environments_compatible(existing_environment, environment):
            raise ValueError(
                "run environment has substantive drift; choose a new --run-id"
            )
        baseline_environment = existing_environment
    launch_environment_record = write_environment_snapshot(
        run_root, PurePosixPath("inputs/environments"), environment
    )
    existing_note = (
        existing_manifest.get("spec", {}).get("note")
        if isinstance(existing_manifest, dict) else None
    )
    if existing_manifest is not None and bool(existing_note) != bool(note_path):
        raise ValueError("note presence differs from the existing run manifest")
    existing_construction = (
        existing_note.get("construction_manifest")
        if isinstance(existing_note, dict) else None
    )
    note, note_record = _load_note(
        run_root,
        note_path,
        note_manifest_path,
        # A bounded diagnostic may omit a construction manifest only when no
        # manifest was supplied. If one is supplied, exercise the exact same
        # note/protocol/final-round validation as the full evaluation path.
        require_manifest=(not smoke or note_manifest_path is not None),
        require_claim_ready=not (smoke or exploratory),
        allow_smoke=smoke,
        expected_task=task,
        expected_model=model,
        expected_model_revision=model_revision,
        expected_response_model=extra_record["expected_response_model"],
        expected_sampling=sampling,
        expected_corpus_commit=corpus.commit,
        expected_corpus=corpus_info,
        expected_source=source,
        expected_environment=environment,
        expected_corpus_display=(
            getattr(corpus, "display", None) if note_path is not None else None
        ),
        expected_note_sha256=(existing_note or {}).get("sha256"),
        expected_note_manifest_sha256=(existing_construction or {}).get("sha256"),
    )
    if preregistration is not None:
        preregistration_snapshot = (
            Path("inputs") / f"preregistration-{preregistration.sha256}.json"
        )
        write_immutable_bytes(
            run_root / preregistration_snapshot, preregistration.data
        )
        preregistration_record: dict[str, object] = {
            "schema_version": PREREGISTRATION_SCHEMA_VERSION,
            "status": "bound",
            "role": preregistration_role,
            "source_path": preregistration.relative_path,
            "sha256": preregistration.sha256,
            "bytes": len(preregistration.data),
            "snapshot": preregistration_snapshot.as_posix(),
            "executed_source_commit": preregistration.head_commit,
            "document": preregistration.document,
        }
    else:
        preregistration_record = {
            "schema_version": PREREGISTRATION_SCHEMA_VERSION,
            "status": "not_provided",
            "reason": "smoke" if smoke else "exploratory",
        }
    if note:
        if not note_prefix_template or note_prefix_template.count("{note}") != 1:
            raise ValueError("a note requires a prompt template containing exactly one {note}")
        prompt_prefix = note_prefix_template.format(note=note)
    else:
        prompt_prefix = ""

    question_records = [
        {
            "id": q["id"],
            "sha256": sha256_json(q),
            "question_text_sha256": sha256_bytes(q["question"].encode("utf-8")),
        }
        for q in questions
    ]
    expected: list[str] = []
    episode_seeds: dict[str, int] = {}
    for budget in budgets:
        for rollout in range(rollouts):
            for qid in question_ids:
                relative = f"{budget}/r{rollout}/{qid}.json"
                expected.append(relative)
                episode_seeds[relative] = stable_seed(
                    master_seed, seed_namespace, seed_group, task, qid, budget, rollout
                )
    presented_prompt_hashes = {
        q["id"]: sha256_bytes((prompt_prefix + q["question"]).encode("utf-8"))
        for q in questions
    }
    spec: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "task": task,
        "purpose": "smoke" if smoke else ("exploratory" if exploratory else "confirmatory"),
        "claim_ready": bool(
            not smoke
            and not exploratory
            and preregistration is not None
            and not source["dirty"]
            and environment_ready
        ),
        "harness": harness,
        "model": model,
        "model_revision": model_revision,
        "sampling": sampling,
        "master_seed": master_seed,
        "seed_policy": {
            "algorithm": "sha256-canonical-json-mod-2147483647",
            "namespace": seed_namespace,
            "seed_group": seed_group,
            "ordered_parts": [
                "master_seed", "namespace", "seed_group", "task", "qid", "budget", "rollout"
            ],
            "episode_seeds": episode_seeds,
        },
        "budgets": budgets,
        "rollouts": rollouts,
        "questions": question_records,
        "question_bundle_sha256": question_bundle_sha256,
        "prompt_policy": {
            "note_prefix_template": note_prefix_template if note else None,
            "presented_prompt_sha256": presented_prompt_hashes,
        },
        "expected_episodes": expected,
        "failure_policy": (
            SCREEN_FAILURE_POLICY if smoke or exploratory else RUN_FAILURE_POLICY
        ),
        "preregistration": preregistration_record,
        "corpus": corpus_info,
        "source": source,
        # The first launch is the immutable substantive baseline.  Every
        # invocation is separately snapshotted and every new episode binds its
        # exact launch record; compatible Slurm allocation churn therefore does
        # not rewrite this manifest.
        "environment": baseline_environment,
        "environment_contract": environment_contract_record(
            baseline_environment
        ),
        "note": note_record,
        "extra": extra_record,
    }
    if preregistration is not None:
        revalidate_run_preregistration(spec, run_root)
    if existing_manifest is not None:
        if (
            not isinstance(existing_manifest, dict)
            or set(existing_manifest) != {"manifest_schema", "spec"}
            or type(existing_manifest.get("manifest_schema")) is not int
            or existing_manifest["manifest_schema"] != SCHEMA_VERSION
            or not isinstance(existing_manifest.get("spec"), dict)
            or canonical_json_bytes(existing_manifest["spec"])
            != canonical_json_bytes(spec)
        ):
            raise ValueError(f"run manifest drift; choose a new --run-id: {manifest_path}")
        manifest = existing_manifest
    else:
        manifest = {"manifest_schema": SCHEMA_VERSION, "spec": spec}
        write_immutable_json(manifest_path, manifest)
    return RunContext(
        run_root,
        manifest,
        sha256_file(manifest_path),
        note,
        prompt_prefix,
        environment,
        launch_environment_record,
    )


def episode_identity(
    context: RunContext,
    *,
    q: dict[str, Any],
    prompt: str,
    budget: str,
    rollout: int,
    seed: int,
) -> dict[str, object]:
    if type(rollout) is not int or rollout < 0 or type(seed) is not int:
        raise ValueError("episode rollout and seed must be JSON integers")
    identity = {
        "manifest_sha256": context.manifest_sha256,
        "question_sha256": sha256_json(q),
        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        "note_sha256": context.note_sha256,
        "seed": seed,
        "task": context.manifest["spec"]["task"],
        "qid": q["id"],
        "budget": budget,
        "rollout": rollout,
    }
    if context.launch_environment_record is None:
        raise ValueError("run context has no exact launch-environment snapshot")
    identity["environment_snapshot"] = context.launch_environment_record
    spec = context.manifest["spec"]
    relative = f"{budget}/r{rollout}/{q['id']}.json"
    expected_seed = spec["seed_policy"]["episode_seeds"].get(relative)
    expected_prompt = spec["prompt_policy"]["presented_prompt_sha256"].get(q["id"])
    if expected_seed != seed or expected_prompt != identity["prompt_sha256"]:
        raise ValueError("episode seed or presented prompt does not match the run manifest")
    return identity


def validate_resumable_episode(
    path: Path,
    identity: dict[str, object],
    *,
    context: RunContext | None = None,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        episode = load_json_artifact(path)
    except (OSError, ValueError) as error:
        raise ValueError(f"invalid existing episode: {path}") from error
    if not isinstance(episode, dict):
        raise ValueError(f"existing episode is not an object: {path}")
    for key, expected in identity.items():
        if key == "environment_snapshot":
            continue
        observed = episode.get(key)
        if (
            key in {"seed", "rollout"}
            and type(observed) is not int
            or type(observed) is not type(expected)
            or observed != expected
        ):
            raise ValueError(f"existing episode provenance mismatch for {key}: {path}")
    if "environment_snapshot" in identity:
        if context is None:
            raise ValueError("run context is required to validate episode environment")
        spec = context.manifest.get("spec")
        if not isinstance(spec, dict) or not isinstance(spec.get("environment"), dict):
            raise ValueError("run manifest has no environment baseline")
        if not environment_contract_is_valid(
            spec.get("environment_contract"), spec["environment"]
        ):
            raise ValueError("run manifest environment contract is invalid")
        try:
            validate_environment_snapshot(
                context.root,
                episode.get("environment_snapshot"),
                baseline=spec["environment"],
                require_claim_ready=spec.get("purpose") != "smoke",
            )
        except ValueError as error:
            raise ValueError(f"existing episode environment is invalid: {path}") from error
    return episode


SCREEN_ATTEMPT_INTENT_SCHEMA_VERSION = 1
SCREEN_ATTEMPT_POLICY = "screen-one-shot-attempt-intent-v1"
_PRODUCER_CONTRACT_KEYS = {
    "expected_model",
    "expected_model_revision",
    "expected_harness",
    "expected_response_model",
}


def _screen_attempt_paths(
    context: RunContext, expected_path: Path
) -> tuple[Path, Path, PurePosixPath]:
    try:
        relative_path = expected_path.relative_to(context.root)
    except ValueError as error:
        raise ValueError("expected episode is outside its run root") from error
    relative = PurePosixPath(relative_path.as_posix())
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in ("", ".", "..") for part in relative.parts)
        or relative.suffix != ".json"
    ):
        raise ValueError("expected episode path is unsafe")
    intent = context.root.joinpath("attempt-intents", *relative.parts)
    failures = context.root.joinpath(
        "failed-attempts", *relative.with_suffix("").parts
    )
    return intent, failures, relative


def _validate_screen_policy(context: RunContext) -> bool:
    spec = context.manifest.get("spec")
    if not isinstance(spec, dict):
        raise ValueError("run manifest has no spec")
    purpose = spec.get("purpose")
    if purpose == "confirmatory":
        if spec.get("failure_policy") != RUN_FAILURE_POLICY:
            raise ValueError("confirmatory run failure policy is invalid")
        return False
    if purpose not in {"smoke", "exploratory"}:
        raise ValueError("run purpose is invalid")
    if spec.get("failure_policy") != SCREEN_FAILURE_POLICY:
        raise ValueError("screen run failure policy is invalid")
    return True


def _validate_producer_contract(contract: object) -> dict[str, str]:
    if (
        not isinstance(contract, dict)
        or set(contract) != _PRODUCER_CONTRACT_KEYS
        or any(not isinstance(value, str) or not value for value in contract.values())
    ):
        raise ValueError("episode producer contract is invalid")
    return contract


def _validate_screen_failed_episode(
    episode: object,
    identity: dict[str, object],
    contract: dict[str, str],
    *,
    expected_episode: str,
    attempt: int,
) -> None:
    """Validate the common exact producer contract for a failed screen cell."""

    if not isinstance(episode, dict):
        raise ValueError("screen failed episode is not an object")
    if (
        episode.get("status") not in {"error", "forced_short"}
        or not isinstance(episode.get("error"), str)
        or not episode["error"]
        or episode.get("failure_attempt") != attempt
        or episode.get("expected_episode") != expected_episode
        or episode.get("model") != contract["expected_model"]
        or episode.get("model_revision") != contract["expected_model_revision"]
        or episode.get("harness") != contract["expected_harness"]
        or not isinstance(episode.get("started"), str)
        or not episode["started"]
        or not isinstance(episode.get("finished"), str)
        or not episode["finished"]
        or not isinstance(episode.get("answer"), str)
        or not isinstance(episode.get("turns"), list)
    ):
        raise ValueError("screen failed episode lifecycle or producer is invalid")
    try:
        started = datetime.fromisoformat(episode["started"])
        finished = datetime.fromisoformat(episode["finished"])
    except ValueError as error:
        raise ValueError("screen failed episode timestamps are invalid") from error
    if (
        started.tzinfo is None
        or finished.tzinfo is None
        or finished < started
    ):
        raise ValueError("screen failed episode timestamp order is invalid")
    for key, expected in identity.items():
        if canonical_json_bytes(episode.get(key)) != canonical_json_bytes(expected):
            raise ValueError(f"screen failed episode identity mismatch for {key}")
    counters = [
        episode.get(field)
        for field in (
            "prompt_tokens", "completion_tokens", "total_tokens", "gen_tokens"
        )
    ]
    if (
        any(type(value) is not int or value < 0 for value in counters)
        or counters[2] != counters[0] + counters[1]
        or counters[3] != counters[1]
    ):
        raise ValueError("screen failed episode token accounting is invalid")
    for field in ("n_tool_iters",):
        if type(episode.get(field)) is not int or episode[field] < 0:
            raise ValueError("screen failed episode iteration count is invalid")
    if contract["expected_harness"] == "dspy.ReAct":
        if (
            type(episode.get("n_react_iters")) is not int
            or episode["n_react_iters"] < 0
            or type(episode.get("finish_catches")) is not int
            or episode["finish_catches"] < 0
            or type(episode.get("n_lm_calls")) is not int
            or episode["n_lm_calls"] < 0
            or not isinstance(episode.get("usage_ledger"), list)
        ):
            raise ValueError("DSPy failed episode audit is invalid")
        if (
            episode["n_react_iters"] != len(episode["turns"])
            or episode["n_tool_iters"] + episode["finish_catches"]
            != episode["n_react_iters"]
        ):
            raise ValueError("DSPy failed episode trajectory counts are invalid")
    elif not isinstance(episode.get("request_attempts"), list):
        raise ValueError("native failed episode request audit is invalid")
    expected_usage = _screen_failure_usage_audit(episode, contract)
    if episode.get("failure_usage") != expected_usage:
        raise ValueError("screen failed episode usage audit is invalid")


def _screen_failure_usage_audit(
    episode: dict[str, object], contract: dict[str, str]
) -> dict[str, object]:
    """Validate known provider ledgers and label unobserved failure usage."""

    prompt = episode.get("prompt_tokens")
    completion = episode.get("completion_tokens")
    total = episode.get("total_tokens")
    if contract["expected_harness"] == "dspy.ReAct":
        ledger = episode.get("usage_ledger")
        calls = episode.get("n_lm_calls")
        if not isinstance(ledger, list) or type(calls) is not int or calls < len(ledger):
            raise ValueError("DSPy failed usage ledger length is invalid")
        ledger_prompt = ledger_completion = ledger_total = 0
        required = {
            "call", "response_model", "response_id", "system_fingerprint",
            "request_messages_sha256", "outputs_sha256", "provider_usage",
            "prompt_tokens", "completion_tokens", "total_tokens",
        }
        for index, record in enumerate(ledger):
            if (
                not isinstance(record, dict)
                or set(record) != required
                or record.get("call") != index
                or type(record.get("call")) is not int
                or not isinstance(record.get("request_messages_sha256"), str)
                or not _SHA256.fullmatch(record["request_messages_sha256"])
                or not isinstance(record.get("outputs_sha256"), str)
                or not _SHA256.fullmatch(record["outputs_sha256"])
            ):
                raise ValueError("DSPy failed usage call record is invalid")
            values = [record.get(field) for field in (
                "prompt_tokens", "completion_tokens", "total_tokens"
            )]
            usage = record.get("provider_usage")
            if (
                any(type(value) is not int or value < 0 for value in values)
                or values[2] != values[0] + values[1]
                or not isinstance(usage, dict)
                or any(usage.get(field) != value for field, value in zip(
                    ("prompt_tokens", "completion_tokens", "total_tokens"),
                    values,
                ))
            ):
                raise ValueError("DSPy failed provider usage is invalid")
            for field in ("response_model", "response_id", "system_fingerprint"):
                value = record.get(field)
                if value is not None and (not isinstance(value, str) or not value):
                    raise ValueError(f"DSPy failed call has invalid {field}")
            ledger_prompt += values[0]
            ledger_completion += values[1]
            ledger_total += values[2]
        if [ledger_prompt, ledger_completion, ledger_total] != [
            prompt, completion, total
        ]:
            raise ValueError("DSPy failed usage prefix disagrees with counters")
        unknown = calls - len(ledger)
    else:
        turns = episode.get("turns")
        attempts = episode.get("request_attempts")
        if not isinstance(turns, list) or not isinstance(attempts, list):
            raise ValueError("native failed provider audit is invalid")
        turn_prompt = turn_completion = turn_total = 0
        response_attempts = []
        grouped: dict[int, list[int]] = {}
        for record in attempts:
            if not isinstance(record, dict):
                raise ValueError("native failed request attempt is not an object")
            logical = record.get("logical_call")
            attempt = record.get("attempt")
            status = record.get("status")
            if (
                type(logical) is not int
                or logical < 0
                or type(attempt) is not int
                or not 1 <= attempt <= 4
                or not isinstance(record.get("request_sha256"), str)
                or not _SHA256.fullmatch(record["request_sha256"])
                or status not in {"response", "transport_error"}
            ):
                raise ValueError("native failed request attempt identity is invalid")
            grouped.setdefault(logical, []).append(attempt)
            if status == "response":
                if not {
                    "logical_call", "attempt", "status", "request_sha256",
                    "response_id", "response_model",
                }.issubset(record):
                    raise ValueError("native failed response attempt is incomplete")
                response_attempts.append(record)
            elif (
                not {
                    "logical_call", "attempt", "status", "request_sha256",
                    "error_type", "error", "usage",
                }.issubset(record)
                or record.get("usage") != "unknown"
                or not isinstance(record.get("error_type"), str)
                or not isinstance(record.get("error"), str)
            ):
                raise ValueError("native failed transport attempt is incomplete")
        if grouped and sorted(grouped) != list(range(max(grouped) + 1)):
            raise ValueError("native failed request logical calls are not contiguous")
        if any(values != list(range(1, len(values) + 1)) for values in grouped.values()):
            raise ValueError("native failed request retry sequence is invalid")
        required_turn = {
            "response_id", "response_model", "system_fingerprint", "reasoning",
            "content", "tool_calls", "observations", "finish_reason",
            "prompt_tokens", "completion_tokens", "total_tokens",
        }
        for index, turn in enumerate(turns):
            if not isinstance(turn, dict) or not required_turn.issubset(turn):
                raise ValueError("native failed completed turn is incomplete")
            values = [turn.get(field) for field in (
                "prompt_tokens", "completion_tokens", "total_tokens"
            )]
            if (
                any(type(value) is not int or value < 0 for value in values)
                or values[2] != values[0] + values[1]
                or index >= len(response_attempts)
                or response_attempts[index].get("logical_call") != index
                or response_attempts[index].get("response_id")
                != turn.get("response_id")
                or response_attempts[index].get("response_model")
                != turn.get("response_model")
            ):
                raise ValueError("native failed completed turn audit is invalid")
            turn_prompt += values[0]
            turn_completion += values[1]
            turn_total += values[2]
        if [turn_prompt, turn_completion, turn_total] != [prompt, completion, total]:
            raise ValueError("native failed known usage disagrees with counters")
        if len(response_attempts) not in {len(turns), len(turns) + 1}:
            raise ValueError("native failed response/turn mapping is invalid")
        if episode.get("n_tool_iters") > len(turns):
            raise ValueError("native failed tool count exceeds completed turns")
        unknown = len(attempts) - len(turns)
    return {
        "status": "complete" if unknown == 0 else "partial-known-prefix",
        "known_prompt_tokens": prompt,
        "known_completion_tokens": completion,
        "known_total_tokens": total,
        "unknown_provider_attempts": unknown,
    }


def validate_screen_failure_state(
    context: RunContext,
    expected_path: Path,
    identity: dict[str, object],
    producer_contract: dict[str, str],
    *,
    require_current_environment: bool = False,
) -> dict[str, object]:
    """Validate and return terminal failed attempts for one screen cell.

    Smoke and exploratory runs are one-shot at the cell level.  The existence
    of any persisted nonfinal attempt therefore blocks every subsequent model
    request for that cell.  We validate the complete append-only directory so
    malformed state cannot be silently ignored or used to reset the policy.
    Confirmatory runs retain their separately preregistered retry policy.
    """

    contract = _validate_producer_contract(producer_contract)
    if type(require_current_environment) is not bool:
        raise ValueError("screen environment-binding flag must be a boolean")
    if not isinstance(identity, dict) or not identity:
        raise ValueError("episode identity is invalid")
    screen = _validate_screen_policy(context)
    intent_path, failure_dir, relative = _screen_attempt_paths(
        context, expected_path
    )
    if not screen:
        return {"intent": None, "failed_attempts": []}

    intent = None
    if intent_path.exists():
        try:
            intent_bytes = read_artifact_bytes(intent_path)
            intent = strict_json_loads(intent_bytes, label=f"attempt intent {intent_path}")
        except (OSError, ValueError) as error:
            raise ValueError(f"invalid screen attempt intent: {intent_path}") from error
        if (
            not isinstance(intent, dict)
            or set(intent) != {
                "schema_version", "policy", "expected_episode", "identity",
                "producer_contract",
            }
            or intent_bytes != canonical_json_bytes(intent)
            or intent.get("schema_version") != SCREEN_ATTEMPT_INTENT_SCHEMA_VERSION
            or type(intent.get("schema_version")) is not int
            or intent.get("policy") != SCREEN_ATTEMPT_POLICY
            or intent.get("expected_episode") != str(relative)
            or intent.get("producer_contract") != contract
            or not isinstance(intent.get("identity"), dict)
            or set(intent["identity"]) != set(identity)
        ):
            raise ValueError(f"screen attempt intent is invalid: {intent_path}")
        for key, expected in identity.items():
            if key == "environment_snapshot":
                continue
            observed = intent["identity"].get(key)
            if (
                key in {"seed", "rollout"}
                and type(observed) is not int
                or type(observed) is not type(expected)
                or canonical_json_bytes(observed) != canonical_json_bytes(expected)
            ):
                raise ValueError(
                    f"screen attempt intent identity mismatch for {key}: {intent_path}"
                )
        spec = context.manifest["spec"]
        try:
            validate_environment_snapshot(
                context.root,
                intent["identity"].get("environment_snapshot"),
                baseline=spec["environment"],
                require_claim_ready=spec.get("purpose") != "smoke",
            )
        except (KeyError, ValueError) as error:
            raise ValueError(
                f"screen attempt intent environment is invalid: {intent_path}"
            ) from error
        if (
            require_current_environment
            and canonical_json_bytes(intent["identity"].get("environment_snapshot"))
            != canonical_json_bytes(identity.get("environment_snapshot"))
        ):
            raise ValueError(
                f"screen attempt intent is bound to another launch: {intent_path}"
            )
    elif intent_path.is_symlink():
        raise ValueError(f"screen attempt intent is a symlink: {intent_path}")

    if not failure_dir.exists():
        return {"intent": intent_path if intent is not None else None,
                "failed_attempts": []}
    if failure_dir.is_symlink() or not failure_dir.is_dir():
        raise ValueError(f"failed-attempt path is not a safe directory: {failure_dir}")

    entries = sorted(failure_dir.iterdir(), key=lambda path: path.name)
    indexed: dict[int, Path] = {}
    for path in entries:
        match = re.fullmatch(r"attempt-([1-9][0-9]*)\.json", path.name)
        if (
            match is None
            or path.is_symlink()
            or not path.is_file()
            or int(match.group(1)) in indexed
        ):
            raise ValueError(f"invalid failed-attempt artifact: {path}")
        indexed[int(match.group(1))] = path
    if sorted(indexed) != [1]:
        raise ValueError(
            f"screen failed-attempt directory must contain exactly attempt 1: {failure_dir}"
        )
    if intent is None:
        raise ValueError(f"screen failed attempt has no prior intent: {failure_dir}")

    validated = []
    expected_relative = str(relative)
    for index in sorted(indexed):
        path = indexed[index]
        try:
            failure_bytes = read_artifact_bytes(path)
            parsed_failure = strict_json_loads(
                failure_bytes, label=f"failed attempt {path}"
            )
        except (OSError, ValueError) as error:
            raise ValueError(f"failed-attempt artifact is invalid: {path}") from error
        if (
            not isinstance(parsed_failure, dict)
            or failure_bytes != canonical_json_bytes(parsed_failure)
        ):
            raise ValueError(f"failed-attempt artifact is not canonical: {path}")
        episode = validate_resumable_episode(
            path, intent["identity"], context=context
        )
        if episode is None or episode != parsed_failure:
            raise ValueError(f"failed-attempt artifact is invalid: {path}")
        try:
            _validate_screen_failed_episode(
                episode,
                intent["identity"],
                contract,
                expected_episode=expected_relative,
                attempt=index,
            )
        except ValueError as error:
            raise ValueError(f"failed-attempt artifact is invalid: {path}") from error
        validated.append(path)
    return {"intent": intent_path, "failed_attempts": validated}


def write_screen_attempt_intent(
    context: RunContext,
    expected_path: Path,
    identity: dict[str, object],
    producer_contract: dict[str, str],
) -> Path | None:
    """Durably mark a screen cell before the first provider request."""

    state = validate_screen_failure_state(
        context, expected_path, identity, producer_contract
    )
    if not _validate_screen_policy(context):
        return None
    if state["intent"] is not None or state["failed_attempts"]:
        raise ValueError("screen cell already has an attempt intent")
    intent_path, _, relative = _screen_attempt_paths(context, expected_path)
    write_immutable_json(intent_path, {
        "schema_version": SCREEN_ATTEMPT_INTENT_SCHEMA_VERSION,
        "policy": SCREEN_ATTEMPT_POLICY,
        "expected_episode": str(relative),
        "identity": identity,
        "producer_contract": _validate_producer_contract(producer_contract),
    })
    observed = validate_screen_failure_state(
        context, expected_path, identity, producer_contract
    )
    if observed["intent"] != intent_path or observed["failed_attempts"]:
        raise ValueError("screen attempt intent failed post-write validation")
    return intent_path


def validate_screen_attempt_tree(
    context: RunContext,
    cells: list[tuple[Path, dict[str, object]]],
    producer_contract: dict[str, str],
) -> dict[str, dict[str, object]]:
    """Globally validate the closed screen attempt/failure namespace.

    This must run while the caller holds the run-level generation lock and
    before it starts any worker.  Unknown, malformed, or out-of-grid artifacts
    then fail the whole invocation before provider contact.
    """

    contract = _validate_producer_contract(producer_contract)
    if not _validate_screen_policy(context):
        return {}
    expected: dict[PurePosixPath, tuple[Path, dict[str, object]]] = {}
    intent_files: set[PurePosixPath] = set()
    failure_files: set[PurePosixPath] = set()
    for path, identity in cells:
        intent_path, _, relative = _screen_attempt_paths(context, path)
        if relative in expected:
            raise ValueError(f"duplicate expected screen cell: {relative}")
        expected[relative] = (path, identity)
        intent_files.add(PurePosixPath(intent_path.relative_to(
            context.root / "attempt-intents").as_posix()))
        failure_files.add(relative.with_suffix("") / "attempt-1.json")

    def validate_closed_tree(
        root: Path, allowed_files: set[PurePosixPath], label: str
    ) -> None:
        if root.is_symlink():
            raise ValueError(f"{label} root is a symlink: {root}")
        if not root.exists():
            return
        if not root.is_dir():
            raise ValueError(f"{label} root is not a safe directory: {root}")
        allowed_dirs = {
            PurePosixPath(*relative.parts[:depth])
            for relative in allowed_files
            for depth in range(1, len(relative.parts))
        }
        for candidate in root.rglob("*"):
            relative = PurePosixPath(candidate.relative_to(root).as_posix())
            if candidate.is_symlink():
                raise ValueError(f"{label} tree contains a symlink: {candidate}")
            if candidate.is_dir():
                if relative not in allowed_dirs:
                    raise ValueError(f"{label} tree contains an unknown directory: {candidate}")
            elif candidate.is_file():
                if relative not in allowed_files:
                    raise ValueError(f"{label} tree contains an unknown file: {candidate}")
            else:
                raise ValueError(f"{label} tree contains a special file: {candidate}")

    validate_closed_tree(
        context.root / "attempt-intents", intent_files, "attempt-intent"
    )
    validate_closed_tree(
        context.root / "failed-attempts", failure_files, "failed-attempt"
    )
    budgets = {relative.parts[0] for relative in expected}
    for budget in budgets:
        budget_files = {
            PurePosixPath(*relative.parts[1:])
            for relative in expected
            if relative.parts[0] == budget
        }
        validate_closed_tree(
            context.root / budget, budget_files, f"result budget {budget}"
        )
    for candidate in context.root.glob("*/r*/*"):
        relative = PurePosixPath(candidate.relative_to(context.root).as_posix())
        if relative.parts[0] not in budgets:
            raise ValueError(
                f"result tree contains an unknown budget artifact: {candidate}"
            )
    states = {}
    for relative, (path, identity) in expected.items():
        state = validate_screen_failure_state(
            context, path, identity, contract
        )
        if path.exists() and state["failed_attempts"]:
            raise ValueError(
                f"screen cell has both a final result and a failed attempt: {relative}"
            )
        if path.exists():
            if state["intent"] is None:
                raise ValueError(
                    f"screen final result has no attempt intent: {relative}"
                )
            try:
                result_bytes = read_artifact_bytes(path)
                result = strict_json_loads(
                    result_bytes, label=f"screen result {path}"
                )
                intent_bytes = read_artifact_bytes(state["intent"])
                intent = strict_json_loads(
                    intent_bytes, label=f"attempt intent {state['intent']}"
                )
            except (OSError, ValueError) as error:
                raise ValueError(
                    f"screen final result binding is invalid: {relative}"
                ) from error
            if (
                not isinstance(result, dict)
                or not isinstance(intent, dict)
                or result_bytes != canonical_json_bytes(result)
                or result.get("status") not in {"ok", "no_answer"}
                or any(
                    canonical_json_bytes(result.get(key))
                    != canonical_json_bytes(value)
                    for key, value in intent["identity"].items()
                )
                or result.get("model") != contract["expected_model"]
                or result.get("model_revision")
                != contract["expected_model_revision"]
                or result.get("harness") != contract["expected_harness"]
            ):
                raise ValueError(
                    f"screen final result does not bind its attempt intent: {relative}"
                )
        states[str(relative)] = state
    return states


def validate_persisted_screen_attempt_tree(
    run_task_root: Path,
    manifest: dict[str, object],
    *,
    require_complete: bool = True,
) -> list[dict[str, object]]:
    """Reconstruct and validate a persisted screen's one-shot attempt ledger.

    Grading and reporting call this independently of the generation process.
    The only environment candidate is read from each immutable intent, then
    re-attested against the manifest baseline and required to equal the result.
    """

    if type(require_complete) is not bool:
        raise ValueError("screen completeness flag must be a boolean")
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"manifest_schema", "spec"}
        or type(manifest.get("manifest_schema")) is not int
        or manifest["manifest_schema"] != SCHEMA_VERSION
        or not isinstance(manifest.get("spec"), dict)
    ):
        raise ValueError("run manifest is invalid")
    spec = manifest["spec"]
    if spec.get("purpose") == "confirmatory":
        if spec.get("failure_policy") != RUN_FAILURE_POLICY:
            raise ValueError("confirmatory run failure policy is invalid")
        return []
    if spec.get("purpose") not in {"smoke", "exploratory"}:
        raise ValueError("run purpose is invalid")
    extra = spec.get("extra")
    expected = spec.get("expected_episodes")
    questions = spec.get("questions")
    seeds = spec.get("seed_policy", {}).get("episode_seeds") \
        if isinstance(spec.get("seed_policy"), dict) else None
    prompts = spec.get("prompt_policy", {}).get("presented_prompt_sha256") \
        if isinstance(spec.get("prompt_policy"), dict) else None
    note = spec.get("note")
    if (
        not isinstance(extra, dict)
        or not isinstance(expected, list)
        or not expected
        or not isinstance(questions, list)
        or not isinstance(seeds, dict)
        or not isinstance(prompts, dict)
        or (note is not None and not isinstance(note, dict))
    ):
        raise ValueError("screen manifest ledger inputs are invalid")
    contract = _validate_producer_contract({
        "expected_model": spec.get("model"),
        "expected_model_revision": spec.get("model_revision"),
        "expected_harness": spec.get("harness"),
        "expected_response_model": extra.get("expected_response_model"),
    })
    question_hashes = {}
    for question in questions:
        if (
            not isinstance(question, dict)
            or set(question) != {"id", "sha256", "question_text_sha256"}
            or not isinstance(question.get("id"), str)
            or question["id"] in question_hashes
            or not isinstance(question.get("sha256"), str)
        ):
            raise ValueError("screen question manifest is invalid")
        question_hashes[question["id"]] = question["sha256"]
    try:
        manifest_bytes = read_artifact_bytes(run_task_root / "manifest.json")
    except (OSError, ValueError) as error:
        raise ValueError("screen manifest artifact is unavailable") from error
    if (
        manifest_bytes != canonical_json_bytes(manifest)
        or strict_json_loads(manifest_bytes, label="screen manifest") != manifest
    ):
        raise ValueError("screen manifest artifact is not canonical or current")
    context = RunContext(
        Path(run_task_root), manifest, sha256_bytes(manifest_bytes), "", ""
    )
    cells = []
    for raw_relative in expected:
        if not isinstance(raw_relative, str) or "\\" in raw_relative:
            raise ValueError("screen expected episode path is invalid")
        relative = PurePosixPath(raw_relative)
        if (
            relative.is_absolute()
            or len(relative.parts) != 3
            or str(relative) != raw_relative
            or relative.suffix != ".json"
            or not relative.parts[1].startswith("r")
            or not relative.parts[1][1:].isdecimal()
        ):
            raise ValueError("screen expected episode path is invalid")
        budget, rollout_dir, filename = relative.parts
        qid = filename.removesuffix(".json")
        if qid not in question_hashes:
            raise ValueError("screen expected episode has an unknown question")
        intent_path = run_task_root.joinpath("attempt-intents", *relative.parts)
        intent_environment = None
        if intent_path.exists() or intent_path.is_symlink():
            try:
                candidate = strict_json_loads(
                    read_artifact_bytes(intent_path), label=f"attempt intent {intent_path}"
                )
            except (OSError, ValueError):
                candidate = None
            if isinstance(candidate, dict) and isinstance(
                candidate.get("identity"), dict
            ):
                intent_environment = candidate["identity"].get(
                    "environment_snapshot"
                )
        identity = {
            "manifest_sha256": context.manifest_sha256,
            "question_sha256": question_hashes[qid],
            "prompt_sha256": prompts.get(qid),
            "note_sha256": note.get("sha256") if isinstance(note, dict) else None,
            "seed": seeds.get(raw_relative),
            "task": spec.get("task"),
            "qid": qid,
            "budget": budget,
            "rollout": int(rollout_dir[1:]),
            "environment_snapshot": intent_environment,
        }
        cells.append((run_task_root.joinpath(*relative.parts), identity))
    states = validate_screen_attempt_tree(context, cells, contract)
    records = []
    for path, _ in cells:
        relative = path.relative_to(run_task_root).as_posix()
        state = states[relative]
        if require_complete and not path.is_file():
            raise ValueError(
                f"screen cell is terminally incomplete or unattempted: {relative}"
            )
        if state["intent"] is None:
            if require_complete:
                raise ValueError(f"screen cell has no attempt intent: {relative}")
            continue
        intent_bytes = read_artifact_bytes(state["intent"])
        records.append({
            "expected_episode": relative,
            "path": state["intent"].relative_to(run_task_root).as_posix(),
            "sha256": sha256_bytes(intent_bytes),
            "bytes": len(intent_bytes),
            "outcome": (
                "final" if path.is_file()
                else ("failed" if state["failed_attempts"] else "ambiguous")
            ),
        })
    records.sort(key=lambda record: record["expected_episode"])
    return records


def write_episode_result(
    context: RunContext,
    expected_path: Path,
    episode: dict[str, Any],
    *,
    validate_final: Callable[[dict[str, Any]], None] | None = None,
    screen_identity: dict[str, object] | None = None,
    producer_contract: dict[str, str] | None = None,
) -> Path:
    """Write a validated final outcome or retain a failed attempt.

    Harness-specific producer validation is mandatory at this persistence
    boundary. That keeps a future runner from accidentally making an alleged
    ``ok``/``no_answer`` artifact durable merely by calling this shared helper.
    """

    if context.launch_environment_record is not None:
        if episode.get("environment_snapshot") != context.launch_environment_record:
            raise ValueError("new episode does not bind the current launch environment")
        spec = context.manifest.get("spec")
        if not isinstance(spec, dict) or not isinstance(spec.get("environment"), dict):
            raise ValueError("run manifest has no environment baseline")
        if not environment_contract_is_valid(
            spec.get("environment_contract"), spec["environment"]
        ):
            raise ValueError("run manifest environment contract is invalid")
        validate_environment_snapshot(
            context.root,
            context.launch_environment_record,
            baseline=spec["environment"],
            require_claim_ready=spec.get("purpose") != "smoke",
        )
    status = episode.get("status")
    spec = context.manifest.get("spec")
    screen = isinstance(spec, dict) and spec.get("purpose") in {
        "smoke", "exploratory"
    }
    if screen:
        if not isinstance(screen_identity, dict) or not isinstance(
            producer_contract, dict
        ):
            raise ValueError(
                "screen persistence requires exact identity and producer contract"
            )
        screen_state = validate_screen_failure_state(
            context,
            expected_path,
            screen_identity,
            producer_contract,
            require_current_environment=True,
        )
        if screen_state["intent"] is None:
            raise ValueError(
                "screen result has no unused valid pre-contact attempt intent"
            )
        if screen_state["failed_attempts"]:
            raise ValueError("screen cell already has a terminal failed attempt")
    if status in ("ok", "no_answer"):
        if not callable(validate_final):
            raise ValueError(
                "final episode persistence requires a producer validator"
            )
        validate_final(episode)
        write_immutable_json(expected_path, episode)
        if screen:
            observed = validate_screen_failure_state(
                context,
                expected_path,
                screen_identity,
                producer_contract,
                require_current_environment=True,
            )
            if observed["intent"] is None or observed["failed_attempts"]:
                raise ValueError("persisted screen final failed post-write validation")
        return expected_path

    relative = expected_path.relative_to(context.root)
    failure_dir = context.root / "failed-attempts" / relative.with_suffix("")
    index = 1
    if screen and failure_dir.exists():
        raise ValueError("screen cell already has a terminal failed attempt")
    if screen and (expected_path.exists() or expected_path.is_symlink()):
        raise ValueError("screen cell already has a final result")
    while not screen and (failure_dir / f"attempt-{index}.json").exists():
        index += 1
    while True:
        failure_path = failure_dir / f"attempt-{index}.json"
        stored_episode = {
            **episode,
            "failure_attempt": index,
            "expected_episode": str(relative),
        }
        if screen:
            validated_contract = _validate_producer_contract(producer_contract)
            stored_episode["failure_usage"] = _screen_failure_usage_audit(
                stored_episode, validated_contract
            )
            _validate_screen_failed_episode(
                stored_episode,
                screen_identity,
                validated_contract,
                expected_episode=str(relative),
                attempt=index,
            )
        try:
            write_immutable_json(failure_path, stored_episode)
            if screen:
                observed = validate_screen_failure_state(
                    context,
                    expected_path,
                    screen_identity,
                    producer_contract,
                    require_current_environment=True,
                )
                if observed["failed_attempts"] != [failure_path]:
                    raise ValueError(
                        "persisted screen failure failed post-write validation"
                    )
            return failure_path
        except FileExistsError:
            if screen:
                raise
            # A concurrent retry claimed this index with different bytes.
            index += 1
