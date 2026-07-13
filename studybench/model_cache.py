"""Race-aware identity checks for an offline Hugging Face model snapshot.

Hugging Face snapshots normally contain logical file symlinks into the model's
``blobs/`` directory.  Those leaf links are accepted only when their exact link
identity resolves lexically inside the selected cache and the resolved target
can be opened as a non-symlink regular file.  Directories and storage files are
opened relative to already-validated descriptors with ``O_NOFOLLOW``.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from typing import Iterator


ATTESTATION_POLICY = "stable-openat-sha256-v1"
_MODEL_ID = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*\Z"
)
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_STABLE_FIELDS = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)


class ModelCacheIntegrityError(ValueError):
    """The selected cache cannot be bound to one stable byte inventory."""


def _directory_flags() -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if type(nofollow) is not int or not nofollow:
        raise ModelCacheIntegrityError("model-cache checks require POSIX O_NOFOLLOW")
    if type(directory) is not int or not directory:
        raise ModelCacheIntegrityError("model-cache checks require POSIX O_DIRECTORY")
    return os.O_RDONLY | nofollow | directory | getattr(os, "O_CLOEXEC", 0)


def _file_flags() -> int:
    return (
        os.O_RDONLY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return all(getattr(left, field) == getattr(right, field) for field in _STABLE_FIELDS)


def _validate_component(component: str) -> None:
    if (
        not component
        or component in {".", ".."}
        or "/" in component
        or "\0" in component
    ):
        raise ModelCacheIntegrityError(f"unsafe model-cache path component: {component!r}")


@contextmanager
def _directory_chain(
    root_fd: int,
    components: tuple[str, ...],
    *,
    display: Path,
) -> Iterator[int]:
    """Open a directory chain and prove every name still names the opened inode."""

    opened: list[tuple[int, str, int, os.stat_result]] = []
    parent_fd = root_fd
    try:
        for component in components:
            _validate_component(component)
            try:
                named = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
                child_fd = os.open(component, _directory_flags(), dir_fd=parent_fd)
            except OSError as error:
                raise ModelCacheIntegrityError(
                    f"model-cache directory is missing, special, or symlinked: {display}"
                ) from error
            opened_identity = os.fstat(child_fd)
            if not stat.S_ISDIR(named.st_mode) or not _same_identity(
                named, opened_identity
            ):
                os.close(child_fd)
                raise ModelCacheIntegrityError(
                    f"model-cache directory changed while opening: {display}"
                )
            opened.append((parent_fd, component, child_fd, opened_identity))
            parent_fd = child_fd
        yield parent_fd
        for parent, component, child, identity in opened:
            current = os.stat(component, dir_fd=parent, follow_symlinks=False)
            if not _same_identity(current, identity) or not _same_identity(
                os.fstat(child), identity
            ):
                raise ModelCacheIntegrityError(
                    f"model-cache directory changed while inspecting: {display}"
                )
    except OSError as error:
        raise ModelCacheIntegrityError(
            f"model-cache directory changed while inspecting: {display}"
        ) from error
    finally:
        for _, _, descriptor, _ in reversed(opened):
            os.close(descriptor)


def _hash_regular_at(
    parent_fd: int,
    leaf: str,
    *,
    display: Path,
    expected: os.stat_result | None = None,
) -> tuple[int, str]:
    """Hash one stable regular descriptor and prove its path was not replaced."""

    _validate_component(leaf)
    try:
        descriptor = os.open(leaf, _file_flags(), dir_fd=parent_fd)
    except OSError as error:
        raise ModelCacheIntegrityError(
            f"model-cache file is missing, special, or symlinked: {display}"
        ) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or (
            expected is not None and not _same_identity(expected, before)
        ):
            raise ModelCacheIntegrityError(
                f"model-cache file is not a stable regular file: {display}"
            )
        digest = hashlib.sha256()
        observed_bytes = 0
        while True:
            chunk = os.read(descriptor, 16 * 1024 * 1024)
            if not chunk:
                break
            observed_bytes += len(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (
            not _same_identity(before, after)
            or observed_bytes != after.st_size
            or not stat.S_ISREG(after.st_mode)
        ):
            raise ModelCacheIntegrityError(
                f"model-cache file changed while hashing: {display}"
            )
        try:
            current = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        except OSError as error:
            raise ModelCacheIntegrityError(
                f"model-cache file changed while hashing: {display}"
            ) from error
        if not _same_identity(current, after):
            raise ModelCacheIntegrityError(
                f"model-cache file was replaced while hashing: {display}"
            )
        return observed_bytes, digest.hexdigest()
    finally:
        os.close(descriptor)


def _hash_cache_file(
    hub_fd: int, hub: Path, components: tuple[str, ...]
) -> tuple[int, str]:
    if not components:
        raise ModelCacheIntegrityError("model-cache storage path has no file name")
    display = hub.joinpath(*components)
    with _directory_chain(hub_fd, components[:-1], display=display) as parent_fd:
        return _hash_regular_at(parent_fd, components[-1], display=display)


def _resolve_link_inside_cache(
    base: tuple[str, ...], target: str, *, display: Path
) -> tuple[str, ...]:
    target_path = PurePosixPath(target)
    if target_path.is_absolute() or not target_path.parts:
        raise ModelCacheIntegrityError(
            f"model snapshot link must be relative and nonempty: {display}"
        )
    resolved = list(base)
    for component in target_path.parts:
        if component == ".":
            continue
        if component == "..":
            if not resolved:
                raise ModelCacheIntegrityError(
                    f"model snapshot link escapes the cache: {display}"
                )
            resolved.pop()
            continue
        _validate_component(component)
        resolved.append(component)
    if not resolved:
        raise ModelCacheIntegrityError(
            f"model snapshot link does not name a cache file: {display}"
        )
    return tuple(resolved)


def _walk_snapshot(
    directory_fd: int,
    *,
    hub_fd: int,
    hub: Path,
    snapshot_components: tuple[str, ...],
    relative: tuple[str, ...] = (),
) -> list[dict[str, object]]:
    before_directory = os.fstat(directory_fd)
    if not stat.S_ISDIR(before_directory.st_mode):
        raise ModelCacheIntegrityError("model snapshot root is not a directory")
    try:
        names = sorted(os.listdir(directory_fd))
    except OSError as error:
        raise ModelCacheIntegrityError("cannot enumerate the pinned model snapshot") from error

    rows: list[dict[str, object]] = []
    for name in names:
        _validate_component(name)
        logical_parts = (*relative, name)
        logical_path = hub.joinpath(*snapshot_components, *logical_parts)
        try:
            identity = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError as error:
            raise ModelCacheIntegrityError(
                f"model snapshot entry changed during enumeration: {logical_path}"
            ) from error
        if stat.S_ISDIR(identity.st_mode):
            try:
                child_fd = os.open(name, _directory_flags(), dir_fd=directory_fd)
            except OSError as error:
                raise ModelCacheIntegrityError(
                    f"model snapshot directory is symlinked or special: {logical_path}"
                ) from error
            try:
                if not _same_identity(identity, os.fstat(child_fd)):
                    raise ModelCacheIntegrityError(
                        f"model snapshot directory changed while opening: {logical_path}"
                    )
                rows.extend(
                    _walk_snapshot(
                        child_fd,
                        hub_fd=hub_fd,
                        hub=hub,
                        snapshot_components=snapshot_components,
                        relative=logical_parts,
                    )
                )
                current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if not _same_identity(identity, current):
                    raise ModelCacheIntegrityError(
                        f"model snapshot directory changed during hashing: {logical_path}"
                    )
            finally:
                os.close(child_fd)
            continue

        if stat.S_ISREG(identity.st_mode):
            size, digest = _hash_regular_at(
                directory_fd, name, display=logical_path, expected=identity
            )
            storage_components = (*snapshot_components, *logical_parts)
        elif stat.S_ISLNK(identity.st_mode):
            try:
                target = os.readlink(name, dir_fd=directory_fd)
            except OSError as error:
                raise ModelCacheIntegrityError(
                    f"cannot read model snapshot link: {logical_path}"
                ) from error
            storage_components = _resolve_link_inside_cache(
                (*snapshot_components, *relative), target, display=logical_path
            )
            size, digest = _hash_cache_file(hub_fd, hub, storage_components)
            try:
                current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                current_target = os.readlink(name, dir_fd=directory_fd)
            except OSError as error:
                raise ModelCacheIntegrityError(
                    f"model snapshot link changed during hashing: {logical_path}"
                ) from error
            if not _same_identity(identity, current) or current_target != target:
                raise ModelCacheIntegrityError(
                    f"model snapshot link changed during hashing: {logical_path}"
                )
        else:
            raise ModelCacheIntegrityError(
                f"model snapshot contains a special file: {logical_path}"
            )

        rows.append(
            {
                "path": PurePosixPath(*logical_parts).as_posix(),
                "storage_path": PurePosixPath(*storage_components).as_posix(),
                "bytes": size,
                "sha256": digest,
            }
        )

    after_directory = os.fstat(directory_fd)
    if not _same_identity(before_directory, after_directory):
        raise ModelCacheIntegrityError(
            f"model snapshot directory changed during hashing: "
            f"{hub.joinpath(*snapshot_components, *relative)}"
        )
    return rows


def canonical_json_bytes(value: object) -> bytes:
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


def build_model_cache_inventory(
    model: str, revision: str, cache_root: Path
) -> dict[str, object]:
    """Return a stable inventory of one exact local model snapshot."""

    if not isinstance(model, str) or not _MODEL_ID.fullmatch(model):
        raise ModelCacheIntegrityError(f"invalid model ID: {model!r}")
    if not isinstance(revision, str) or not _REVISION.fullmatch(revision):
        raise ModelCacheIntegrityError(f"model revision must be a full commit: {revision!r}")
    raw_hub = Path(cache_root)
    try:
        hub = raw_hub.resolve(strict=True)
        named_hub = hub.stat(follow_symlinks=False)
        hub_fd = os.open(hub, _directory_flags())
    except OSError as error:
        raise ModelCacheIntegrityError(f"pinned Hugging Face cache is absent: {raw_hub}") from error
    try:
        opened_hub = os.fstat(hub_fd)
        if not stat.S_ISDIR(named_hub.st_mode) or not _same_identity(
            named_hub, opened_hub
        ):
            raise ModelCacheIntegrityError("Hugging Face cache root changed while opening")
        snapshot_components = (
            "models--" + model.replace("/", "--"),
            "snapshots",
            revision,
        )
        snapshot_path = hub.joinpath(*snapshot_components)
        with _directory_chain(
            hub_fd, snapshot_components, display=snapshot_path
        ) as snapshot_fd:
            files = _walk_snapshot(
                snapshot_fd,
                hub_fd=hub_fd,
                hub=hub,
                snapshot_components=snapshot_components,
            )
        current_hub = hub.stat(follow_symlinks=False)
        if not _same_identity(current_hub, opened_hub):
            raise ModelCacheIntegrityError("Hugging Face cache root changed while hashing")
    finally:
        os.close(hub_fd)

    logical_names = {str(row["path"]) for row in files}
    if "config.json" not in logical_names:
        raise ModelCacheIntegrityError("pinned model snapshot has no config.json")
    if not any(name.endswith((".safetensors", ".bin")) for name in logical_names):
        raise ModelCacheIntegrityError("pinned model snapshot has no weight files")
    files.sort(key=lambda row: str(row["path"]))
    return {
        "schema_version": 1,
        "attestation_policy": ATTESTATION_POLICY,
        "model": model,
        "revision": revision,
        "cache_root": str(hub),
        "snapshot": str(hub.joinpath(*snapshot_components)),
        "file_count": len(files),
        "total_bytes": sum(int(row["bytes"]) for row in files),
        "files": files,
        "tree_sha256": hashlib.sha256(canonical_json_bytes(files)).hexdigest(),
    }


def _strict_json(data: bytes, *, label: str) -> object:
    def no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ModelCacheIntegrityError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=no_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ModelCacheIntegrityError(f"{label} contains non-finite value {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelCacheIntegrityError(f"{label} is not canonical UTF-8 JSON") from error


def _read_stable_regular(path: Path) -> bytes:
    path = Path(path)
    try:
        parent = path.parent.resolve(strict=True)
        parent_fd = os.open(parent, _directory_flags())
    except OSError as error:
        raise ModelCacheIntegrityError(f"cannot open model-cache inventory: {path}") from error
    try:
        leaf = path.name
        _validate_component(leaf)
        descriptor = os.open(leaf, _file_flags(), dir_fd=parent_fd)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ModelCacheIntegrityError(
                    f"model-cache inventory is not a regular file: {path}"
                )
            chunks: list[bytes] = []
            while chunk := os.read(descriptor, 1024 * 1024):
                chunks.append(chunk)
            after = os.fstat(descriptor)
            current = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
            data = b"".join(chunks)
            if (
                not _same_identity(before, after)
                or not _same_identity(after, current)
                or len(data) != after.st_size
            ):
                raise ModelCacheIntegrityError(
                    f"model-cache inventory changed while reading: {path}"
                )
            return data
        finally:
            os.close(descriptor)
    except OSError as error:
        raise ModelCacheIntegrityError(f"cannot read model-cache inventory: {path}") from error
    finally:
        os.close(parent_fd)


def verify_model_cache_inventory(
    model: str,
    revision: str,
    inventory_path: Path,
    expected_sha256: str,
) -> None:
    """Fail unless the live cache exactly matches a canonical prior inventory."""

    data = _read_stable_regular(Path(inventory_path))
    if (
        not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
        or hashlib.sha256(data).hexdigest() != expected_sha256
    ):
        raise ModelCacheIntegrityError(
            "model-cache inventory differs from its prelaunch fingerprint"
        )
    expected = _strict_json(data, label="model-cache inventory")
    if not isinstance(expected, dict) or canonical_json_bytes(expected) != data:
        raise ModelCacheIntegrityError("model-cache inventory is not canonical JSON")
    if (
        expected.get("attestation_policy") != ATTESTATION_POLICY
        or expected.get("model") != model
        or expected.get("revision") != revision
        or not isinstance(expected.get("cache_root"), str)
        or not Path(expected["cache_root"]).is_absolute()
    ):
        raise ModelCacheIntegrityError(
            "model-cache inventory does not bind the requested model and cache"
        )
    observed = build_model_cache_inventory(model, revision, Path(expected["cache_root"]))
    if canonical_json_bytes(observed) != data:
        raise ModelCacheIntegrityError(
            "model cache differs from its prelaunch canonical inventory"
        )


def _write_new_inventory(path: Path, data: bytes) -> None:
    path = Path(path)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.fchmod(descriptor, 0o600)
    finally:
        os.close(descriptor)


def main(arguments: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    try:
        if len(arguments) == 5 and arguments[0] == "create":
            _, model, revision, cache_root, destination = arguments
            inventory = build_model_cache_inventory(model, revision, Path(cache_root))
            _write_new_inventory(Path(destination), canonical_json_bytes(inventory))
            return 0
        if len(arguments) == 5 and arguments[0] == "verify":
            _, model, revision, inventory, expected_sha256 = arguments
            verify_model_cache_inventory(
                model, revision, Path(inventory), expected_sha256
            )
            return 0
        raise ModelCacheIntegrityError(
            "usage: model_cache.py create MODEL REVISION CACHE_ROOT DESTINATION; "
            "or model_cache.py verify MODEL REVISION INVENTORY EXPECTED_SHA256"
        )
    except (OSError, ModelCacheIntegrityError) as error:
        raise SystemExit(f"model-cache attestation failed: {error}") from error


if __name__ == "__main__":
    raise SystemExit(main())
