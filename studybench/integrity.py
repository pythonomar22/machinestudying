"""Small, dependency-free primitives for reproducible research artifacts.

Artifact identity is defined over bytes, not mtimes or mutable path names.  The
immutable writers deliberately fail when a path already contains different
bytes: silently replacing an input to a completed run destroys provenance.

Path hardening relies on the POSIX ``openat`` family exposed by Python's
``dir_fd`` arguments and on ``O_NOFOLLOW``/``O_DIRECTORY``.  Artifact access
fails closed on platforms without those primitives rather than using a racy
check-then-open fallback.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
import errno
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
from typing import Any

import fcntl


def _directory_open_flags() -> int:
    """Flags for race-safe POSIX directory traversal.

    ``dir_fd`` plus ``O_NOFOLLOW`` is intentionally required.  Falling back to
    a check-then-open sequence would restore the symlink race these helpers are
    meant to exclude.
    """

    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise RuntimeError(
            "race-safe artifact access requires POSIX O_NOFOLLOW and O_DIRECTORY"
        )
    return (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )


def _path_parts(path: Path) -> tuple[str, tuple[str, ...], str]:
    """Split a path into an anchor, parent components, and a leaf name."""

    path = Path(path)
    parts = path.parts
    if path.is_absolute():
        anchor = path.anchor
        parts = parts[1:]
    else:
        anchor = "."
    if not parts:
        raise ValueError(f"artifact path has no file name: {path}")
    return anchor, tuple(parts[:-1]), parts[-1]


def _unsafe_component(path: Path, exc: OSError) -> None:
    if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
        raise ValueError(
            f"artifact path contains a symlink or non-directory component: {path}"
        ) from exc
    raise exc


def _open_directory_at(parent_fd: int, component: str, path: Path) -> int:
    try:
        return os.open(component, _directory_open_flags(), dir_fd=parent_fd)
    except OSError as exc:
        _unsafe_component(path, exc)
        raise AssertionError("unreachable")


@contextmanager
def _open_parent_directory(path: Path, *, create: bool):
    """Yield an anchored parent descriptor and leaf without following links."""

    path = Path(path)
    anchor, parent_parts, leaf = _path_parts(path)
    descriptor = os.open(anchor, _directory_open_flags())
    try:
        for component in parent_parts:
            try:
                child = _open_directory_at(descriptor, component, path)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(component, mode=0o777, dir_fd=descriptor)
                except FileExistsError:
                    # Another creator won.  Opening with O_NOFOLLOW below
                    # validates that the winner installed a directory, not a
                    # link or another filesystem object.
                    pass
                child = _open_directory_at(descriptor, component, path)
            os.close(descriptor)
            descriptor = child
        yield descriptor, leaf
    finally:
        os.close(descriptor)


def _read_regular_file_at(
    parent_fd: int, leaf: str, path: Path,
) -> tuple[bytes, int]:
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    # Opening a FIFO merely to reject it must not block indefinitely.
    flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(leaf, flags, dir_fd=parent_fd)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ValueError(f"artifact path contains a symlink: {path}") from exc
        raise

    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"artifact is not a regular file: {path}")

        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        data = b"".join(chunks)

        after = os.fstat(descriptor)
        stable_fields = (
            "st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns",
        )
        if (
            any(getattr(before, field) != getattr(after, field) for field in stable_fields)
            or len(data) != after.st_size
        ):
            raise ValueError(f"artifact changed while it was being read: {path}")

        try:
            current = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError as exc:
            raise ValueError(f"artifact changed while it was being read: {path}") from exc
        if (
            not stat.S_ISREG(current.st_mode)
            or (current.st_dev, current.st_ino, current.st_mode)
            != (after.st_dev, after.st_ino, after.st_mode)
        ):
            raise ValueError(f"artifact changed while it was being read: {path}")
        return data, stat.S_IMODE(after.st_mode)
    finally:
        os.close(descriptor)


def _read_regular_bytes_at(parent_fd: int, leaf: str, path: Path) -> bytes:
    return _read_regular_file_at(parent_fd, leaf, path)[0]


def _read_regular_file(path: Path) -> tuple[bytes, int]:
    path = Path(path)
    with _open_parent_directory(path, create=False) as (parent_fd, leaf):
        return _read_regular_file_at(parent_fd, leaf, path)


def _read_regular_bytes(path: Path) -> bytes:
    path = Path(path)
    with _open_parent_directory(path, create=False) as (parent_fd, leaf):
        return _read_regular_bytes_at(parent_fd, leaf, path)


def _leaf_metadata(parent_fd: int, leaf: str) -> os.stat_result | None:
    try:
        return os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _create_temporary(parent_fd: int) -> tuple[int, str]:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    for _ in range(128):
        name = f".artifact-{secrets.token_hex(16)}.tmp"
        try:
            descriptor = os.open(name, flags, 0o600, dir_fd=parent_fd)
        except FileExistsError:
            continue
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid():
            os.close(descriptor)
            try:
                os.unlink(name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
            raise RuntimeError("new artifact temporary is not a current-user regular file")
        return descriptor, name
    raise FileExistsError("could not allocate a unique artifact temporary")


def _write_and_sync(descriptor: int, data: bytes) -> None:
    # The caller keeps the descriptor open until the directory operation is
    # complete.  ``closefd=False`` also ensures a write/fsync exception is not
    # obscured by a second close in the caller's cleanup path.
    with os.fdopen(descriptor, "wb", closefd=False) as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def canonical_json_bytes(value: object) -> bytes:
    """Return the unique UTF-8 JSON encoding used for hashes and manifests."""

    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def canonical_json(value: object) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def strict_json_loads(data: str | bytes, *, label: str = "JSON") -> Any:
    """Parse standards-compliant JSON without duplicate keys or non-finite numbers."""

    def object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"{label} contains non-finite number {value}")

    try:
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        return json.loads(
            data,
            object_pairs_hook=object_without_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    """Hash a regular file without following any symlink component."""

    return sha256_bytes(_read_regular_bytes(Path(path)))


def sha256_json(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def stable_seed(master_seed: int, *parts: object) -> int:
    """Derive a provider-compatible seed without Python's randomized hash()."""

    payload = {"master_seed": master_seed, "parts": list(parts)}
    # Keep the value in the widely accepted signed 31-bit seed range.
    return int.from_bytes(hashlib.sha256(canonical_json_bytes(payload)).digest()[:8], "big") % (
        2**31 - 1
    )


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _atomic_write(path: Path, data: bytes) -> None:
    path = Path(path)
    with _open_parent_directory(path, create=True) as (parent_fd, leaf):
        metadata = _leaf_metadata(parent_fd, leaf)
        if metadata is not None and stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"refusing to replace artifact symlink: {path}")
        if metadata is not None and not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"refusing to replace non-regular artifact: {path}")

        descriptor, temporary = _create_temporary(parent_fd)
        try:
            _write_and_sync(descriptor, data)
            # renameat operates relative to the already-open directory.  If a
            # path component is concurrently renamed and replaced by a link,
            # the write remains anchored to the directory we validated.
            os.replace(
                temporary,
                leaf,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
        finally:
            os.close(descriptor)
            try:
                os.unlink(temporary, dir_fd=parent_fd)
            except FileNotFoundError:
                pass


def atomic_write_bytes(path: Path, data: bytes) -> None:
    _atomic_write(path, data)


def atomic_write_text(path: Path, text: str) -> None:
    _atomic_write(path, text.encode("utf-8"))


def atomic_write_json(path: Path, value: object) -> None:
    _atomic_write(path, canonical_json_bytes(value))


def _write_immutable(path: Path, data: bytes) -> None:
    path = Path(path)
    with _open_parent_directory(path, create=True) as (parent_fd, leaf):
        descriptor, temporary = _create_temporary(parent_fd)
        try:
            _write_and_sync(descriptor, data)
            # linkat gives create-if-absent semantics even with concurrent
            # writers.  Both names are relative to the validated parent fd, so
            # swapping an ancestor for a symlink cannot redirect this write.
            try:
                os.link(
                    temporary,
                    leaf,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                    follow_symlinks=False,
                )
            except FileExistsError:
                try:
                    existing = _read_regular_bytes_at(parent_fd, leaf, path)
                except (OSError, ValueError) as exc:
                    raise FileExistsError(
                        f"refusing unsafe immutable artifact: {path}"
                    ) from exc
                if existing != data:
                    raise FileExistsError(
                        f"refusing to replace immutable artifact: {path}"
                    )
        finally:
            os.close(descriptor)
            try:
                os.unlink(temporary, dir_fd=parent_fd)
            except FileNotFoundError:
                pass


def write_immutable_bytes(path: Path, data: bytes) -> None:
    _write_immutable(path, data)


def write_immutable_text(path: Path, text: str) -> None:
    _write_immutable(path, text.encode("utf-8"))


def write_immutable_json(path: Path, value: object) -> None:
    _write_immutable(path, canonical_json_bytes(value))


def file_record(path: Path) -> dict[str, Any]:
    """Describe the exact bytes at *path* for embedding in a manifest."""

    path = Path(path)
    data = _read_regular_bytes(path)
    return {"path": str(path), "sha256": sha256_bytes(data), "bytes": len(data)}


def assert_record(path: Path, record: Mapping[str, object]) -> None:
    """Fail if a file no longer matches a previously stored artifact record."""

    expected_hash = record.get("sha256")
    expected_bytes = record.get("bytes")
    data = _read_regular_bytes(Path(path))
    if expected_hash != sha256_bytes(data) or expected_bytes != len(data):
        raise ValueError(f"artifact does not match manifest record: {path}")


def read_artifact_bytes(path: Path) -> bytes:
    """Read exact bytes only through a regular path with no symlink components."""

    return _read_regular_bytes(Path(path))


def read_artifact_bytes_with_mode(path: Path) -> tuple[bytes, int]:
    """Read exact bytes and permission mode from one stable, anchored descriptor."""

    return _read_regular_file(Path(path))


def load_json_artifact(path: Path) -> Any:
    """Read a regular, non-symlink UTF-8 JSON artifact with strict semantics."""

    return strict_json_loads(_read_regular_bytes(Path(path)), label=str(path))


@contextmanager
def exclusive_process_lock(path: Path):
    """Hold a non-blocking, crash-releasing lock for one unit of work.

    The lock file is retained so unlink/recreate races cannot create two lock
    inodes for the same artifact.  Its contents are not research evidence.
    """

    path = Path(path)
    with _open_parent_directory(path, create=True) as (parent_fd, leaf):
        flags = (
            os.O_RDWR
            | os.O_CREAT
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
        )
        descriptor = os.open(leaf, flags, 0o600, dir_fd=parent_fd)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise RuntimeError(f"lock path is not a regular file: {path}")
            if metadata.st_uid != os.getuid():
                raise RuntimeError(f"lock file is not owned by the current user: {path}")
            if stat.S_IMODE(metadata.st_mode) != 0o600:
                raise RuntimeError(f"lock file must have mode 0600: {path}")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(
                    f"another process is already working on: {path}"
                ) from exc
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
