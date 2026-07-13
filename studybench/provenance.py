"""Fail-closed manifests for study and evaluation artifacts.

Historical runs predate these contracts and remain readable as legacy evidence.
New runs must live under a caller-chosen ID and bind every episode to one exact
dataset, corpus, prompt, note, seed policy, and source tree.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from fnmatch import fnmatchcase
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
from typing import Any
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
    bind_preregistration,
    revalidate_run_preregistration,
)


SCHEMA_VERSION = 1
VLLM_VERSION = "0.24.0"
VLLM_PYTHON_VERSION = "3.12.11"
MAIN_PYTHON_VERSION = "3.14.6"
DSPY_COMMIT = "9cdb0aac28b2a04b064e40697ccd301872cf6a43"
MODEL_ID = "Qwen/Qwen3.5-9B"
MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
ENVIRONMENT_COMPATIBILITY_POLICY = "allocation-and-transport-nuisances-v1"
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
    expected_task: str | None = None,
    expected_corpus_commit: str | None = None,
    expected_note_sha256: str | None = None,
    expected_note_manifest_sha256: str | None = None,
) -> tuple[str, dict[str, object] | None]:
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
            if not isinstance(construction.get("study_id"), str):
                raise ValueError("note manifest has no study_id")
            claim_ready = construction.get("claim_ready")
            if claim_ready is None and isinstance(construction.get("config"), dict):
                claim_ready = construction["config"].get("claim_ready")
            if claim_ready is not True:
                raise ValueError("note construction manifest is not claim-ready")
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
            if manifest_type == "human-audited-note":
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
                if (not isinstance(config, dict)
                        or config.get("method") != "forced-50-cheatsheet"
                        or config.get("claim_ready") is not True
                        or not isinstance(inventory, dict)
                        or set(inventory) != {"intent.json", "episode.json"}
                        or construction_inventory_sha256 != sha256_json(inventory)):
                    raise ValueError("forced-50 construction manifest is incomplete")
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
        require_manifest=not smoke,
        expected_task=task,
        expected_corpus_commit=corpus.commit,
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
        "failure_policy": RUN_FAILURE_POLICY,
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


def write_episode_result(
    context: RunContext,
    expected_path: Path,
    episode: dict[str, Any],
) -> Path:
    """Write a final outcome or retain a non-final attempt without overwriting."""

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
    if status in ("ok", "no_answer"):
        write_immutable_json(expected_path, episode)
        return expected_path

    relative = expected_path.relative_to(context.root)
    failure_dir = context.root / "failed-attempts" / relative.with_suffix("")
    index = 1
    while (failure_dir / f"attempt-{index}.json").exists():
        index += 1
    while True:
        failure_path = failure_dir / f"attempt-{index}.json"
        try:
            write_immutable_json(
                failure_path,
                {**episode, "failure_attempt": index, "expected_episode": str(relative)},
            )
            return failure_path
        except FileExistsError:
            # A concurrent retry claimed this index with different bytes.
            index += 1
