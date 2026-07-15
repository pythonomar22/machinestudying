"""Dependency-light installed-environment identity for isolated runtimes."""

from __future__ import annotations

from importlib import metadata
import hashlib
import os
from pathlib import Path, PurePosixPath
import platform
import re
import stat
import sys

from .integrity import sha256_json, sha256_text


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


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
