"""Minimal, fail-closed loading for the repository's private ``.env`` file.

The loader never prints values, never follows a symlink, and refuses files that
are not owned by the current user with mode 0600.  Research commands must be as
safe when invoked directly as they are through the Slurm wrappers.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import stat


_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def _required_open_flag(name: str) -> int:
    """Return one nonzero OS safety flag or fail before touching the file."""

    value = getattr(os, name, None)
    if type(value) is not int or value == 0:
        raise RuntimeError(
            f"secure environment loading requires the POSIX {name} primitive"
        )
    return value


def load_private_env(path: Path) -> None:
    """Add unset ``KEY=VALUE`` entries from *path* to :mod:`os.environ`.

    Missing files are allowed.  Existing environment variables take priority.
    Quoting, interpolation, ``export``, and shell syntax are intentionally not
    supported: accepting a shell language here would expand the attack surface
    and make the effective configuration less transparent.
    """

    path = Path(path)
    nofollow = _required_open_flag("O_NOFOLLOW")
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | nofollow
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0),
        )
    except FileNotFoundError:
        return
    except OSError as exc:
        raise RuntimeError(f"refusing unsafe environment file {path}: {exc}") from exc

    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"environment file is not a regular file: {path}")
        if stat.S_IMODE(metadata.st_mode) != 0o600:
            raise RuntimeError(f"environment file must have mode 0600: {path}")
        if hasattr(os, "getuid") and metadata.st_uid != os.getuid():
            raise RuntimeError(f"environment file is not owned by the current user: {path}")
        with os.fdopen(descriptor, "r", encoding="utf-8", errors="strict") as handle:
            descriptor = -1
            lines = handle.read().splitlines()
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"environment file is not valid UTF-8: {path}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    parsed: list[tuple[str, str]] = []
    seen: set[str] = set()
    for number, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise RuntimeError(f"invalid environment entry at {path}:{number}")
        key, value = line.split("=", 1)
        key = key.strip()
        if not _KEY.fullmatch(key):
            raise RuntimeError(f"invalid environment key at {path}:{number}")
        if key in seen:
            raise RuntimeError(f"duplicate environment key at {path}:{number}")
        value = value.strip()
        if "\0" in value:
            raise RuntimeError(f"NUL byte in environment value at {path}:{number}")
        parsed.append((key, value))
        seen.add(key)

    for key, value in parsed:
        os.environ.setdefault(key, value)
