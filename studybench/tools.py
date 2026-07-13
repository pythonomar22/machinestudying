"""The three ReAct tools (grep, glob, read_file) over a pinned corpus checkout.

File contents are preloaded into memory (the corpora are ~100MB; NFS reads per call
would be far too slow). Output caps are replication inferences (the paper does not
specify them); they are sized so a forced 20-iteration episode stays well inside the
model's 262k context.
"""

import bisect
import fnmatch
import glob as globlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import PurePosixPath

import regex  # not re: its match timeout stops catastrophic backtracking

from .dataset import (
    Corpus,
    read_pinned_code_bytes,
    tracked_code_paths,
    validate_corpus_snapshot,
)

GREP_MAX_MATCHES = 50
GREP_MAX_LINE_CHARS = 240
GREP_TIME_BUDGET = 10.0  # seconds; guards against catastrophic regex backtracking
GLOB_MAX_PATHS = 200
READ_MAX_LINES = 500
OBS_MAX_CHARS = 25_000
MAX_PATTERN_CHARS = 1_000
MAX_LINE_NUMBER = 10_000_000


def _object_without_duplicate_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate tool argument: {key!r}")
        value[key] = item
    return value


def _reject_json_constant(value):
    raise ValueError(f"non-finite tool argument is forbidden: {value}")

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search the repository code for a regular expression (case-sensitive). "
            "Returns matching lines as path:line_number:line.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regular expression to search for.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Optional file or directory path to restrict the search to.",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "List repository files matching a glob pattern, e.g. 'dspy/**/*.py'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern (supports **)."}
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the repository, optionally a specific line range. "
            f"At most {READ_MAX_LINES} lines are returned per call.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the repository root.",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "1-indexed first line to read.",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "1-indexed last line to read.",
                    },
                },
                "required": ["path"],
            },
        },
    },
]


def _glob_to_regex(pat: str) -> str:
    """glob -> regex with ** support, matching Python 3.13's glob.translate
    (recursive=True, include_hidden=True) semantics."""
    i, n, out = 0, len(pat), []
    while i < n:
        c = pat[i]
        if c == "*":
            if pat[i:i + 3] == "**/":
                out.append("(?:[^/]+/)*"); i += 3
            elif pat[i:i + 2] == "**":
                out.append(".*"); i += 2
            else:
                out.append("[^/]*"); i += 1
        elif c == "?":
            out.append("[^/]"); i += 1
        elif c == "[":
            j = i + 1
            if j < n and pat[j] in "!]":
                j += 1
            while j < n and pat[j] != "]":
                j += 1
            if j < n:
                cls = pat[i + 1:j]
                out.append("[" + ("^" + cls[1:] if cls.startswith("!") else cls) + "]")
                i = j + 1
            else:
                out.append(re.escape(c)); i += 1
        else:
            out.append(re.escape(c)); i += 1
    return "".join(out) + r"\Z"


class RepoTools:
    def __init__(self, corpus: Corpus, read_max_lines: int = READ_MAX_LINES):
        validate_corpus_snapshot(corpus)
        self.corpus = corpus
        self.read_max_lines = read_max_lines
        if type(read_max_lines) is not int or read_max_lines < 1:
            raise ValueError("read_max_lines must be a positive integer")

        # Git enumeration excludes ignored/generated files while the resolver
        # rejects broken or root-escaping tracked symlinks.
        paths = tracked_code_paths(corpus)
        with ThreadPoolExecutor(16) as pool:
            texts = pool.map(self._load, paths)
        self.text: dict[str, str] = {
            p.relative_to(corpus.repo).as_posix(): text for p, text in zip(paths, texts)
        }
        self.files = list(self.text)
        # Character offset of each line start, for match -> line-number lookup in grep.
        self._starts = {
            f: [0] + [m.end() for m in re.finditer("\n", t)] for f, t in self.text.items()
        }

    def _load(self, path) -> str:
        relative = path.relative_to(self.corpus.repo).as_posix()
        data = read_pinned_code_bytes(self.corpus, relative)
        if b"\x00" in data:
            raise ValueError(f"code file contains NUL bytes: {path}")
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"code file is not valid UTF-8: {path}") from exc

    @staticmethod
    def _relative_path(value: str, *, allow_empty: bool = False) -> str:
        if not isinstance(value, str) or "\\" in value or "\x00" in value:
            raise ValueError("path must be a POSIX string relative to the repository")
        if not value:
            if allow_empty:
                return ""
            raise ValueError("path must not be empty")
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or path.as_posix() != value
            or any(part in ("", ".", "..") for part in path.parts)
        ):
            raise ValueError(f"path must be normalized and relative: {value!r}")
        return path.as_posix()

    def dispatch(self, name: str, arguments: str) -> str:
        """Run one tool call; always returns an observation string, never raises."""
        try:
            args = (
                json.loads(
                    arguments,
                    object_pairs_hook=_object_without_duplicate_keys,
                    parse_constant=_reject_json_constant,
                )
                if arguments else {}
            )
            if not isinstance(args, dict):
                return f"Error: tool arguments must be a JSON object, got: {arguments[:200]}"
            handlers = {"grep": self._grep, "glob": self._glob, "read_file": self._read_file}
            if name not in handlers:
                raise ValueError(f"unknown tool: {name!r}")
            obs = handlers[name](**args)
        except Exception as e:  # bad tool name, bad args, bad regex, ...
            obs = f"Error: {type(e).__name__}: {e}"
        if len(obs) > OBS_MAX_CHARS:
            obs = obs[:OBS_MAX_CHARS] + "\n... (output truncated)"
        return obs

    def _grep(self, pattern: str, path: str | None = None) -> str:
        if not isinstance(pattern, str) or len(pattern) > MAX_PATTERN_CHARS:
            return f"Error: grep pattern must be at most {MAX_PATTERN_CHARS} characters."
        try:
            rx = regex.compile(pattern)
        except regex.error:
            rx = regex.compile(regex.escape(pattern))  # fall back to a literal search
        path = self._relative_path(path or "", allow_empty=True)
        candidates = [f for f in self.files if not path or f == path or f.startswith(path + "/")]
        if path and not candidates:
            return f"Error: no files under '{path}'."
        matches, truncated = [], False
        deadline = time.monotonic() + GREP_TIME_BUDGET
        for f in candidates:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                truncated = True
                break
            starts, seen_line = self._starts[f], -1
            try:
                for m in rx.finditer(self.text[f], timeout=max(0.001, remaining)):
                    line = bisect.bisect_right(starts, m.start())  # 1-indexed
                    if line == seen_line:
                        continue  # one report per line
                    seen_line = line
                    text = self.text[f][starts[line - 1]:].split("\n", 1)[0]
                    matches.append(f"{f}:{line}:{text[:GREP_MAX_LINE_CHARS]}")
                    if len(matches) >= GREP_MAX_MATCHES:
                        truncated = True
                        break
            except TimeoutError:  # pathological backtracking; report what we have
                truncated = True
            if truncated or time.monotonic() > deadline:
                truncated = True
                break
        if not matches:
            return f"No matches for /{pattern}/."
        out = "\n".join(matches)
        if truncated:
            out += "\n... (more matches may exist; narrow the pattern or use path)"
        return out

    def _glob(self, pattern: str) -> str:
        if not isinstance(pattern, str) or not pattern or len(pattern) > MAX_PATTERN_CHARS:
            return (
                "Error: glob pattern must be a non-empty string of at most "
                f"{MAX_PATTERN_CHARS} characters."
            )
        p = self._relative_path(pattern)
        if hasattr(globlib, "translate"):  # Python >= 3.13
            rx = re.compile(globlib.translate(p, recursive=True, include_hidden=True))
        else:  # 3.12 (the dspy.ReAct runner venv): equivalent translation
            rx = re.compile(_glob_to_regex(p))
        hits = [f for f in self.files if rx.match(f)]
        if not hits:
            # a bare filename pattern like '*.py' is a common mistake; match basenames too
            hits = [f for f in self.files if fnmatch.fnmatch(f.rsplit("/", 1)[-1], pattern)]
        if not hits:
            return f"No files match '{pattern}'."
        out = "\n".join(hits[:GLOB_MAX_PATHS])
        if len(hits) > GLOB_MAX_PATHS:
            out += f"\n... ({len(hits) - GLOB_MAX_PATHS} more files not shown)"
        return out

    def _read_file(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> str:
        path = self._relative_path(path)
        if path not in self.text:
            return f"Error: '{path}' is not a readable file in this repository."
        lines = self.text[path].splitlines()
        n = len(lines)
        if start_line is not None and type(start_line) is not int:
            return "Error: start_line must be a positive, 1-indexed integer."
        if end_line is not None and type(end_line) is not int:
            return "Error: end_line must be a positive, 1-indexed integer."
        start = start_line if start_line is not None else 1
        requested_end = end_line if end_line is not None else n
        if (
            start < 1
            or requested_end < 1
            or start > MAX_LINE_NUMBER
            or requested_end > MAX_LINE_NUMBER
        ):
            return (
                "Error: start_line and end_line must be positive, 1-indexed "
                f"integers no greater than {MAX_LINE_NUMBER}."
            )
        end = min(n, requested_end)
        if start > n or end < start:
            return f"Error: '{path}' has {n} lines; requested lines {start}-{end}."
        if end - start + 1 > self.read_max_lines:
            end = start + self.read_max_lines - 1
        body = "\n".join(f"{i}: {lines[i - 1][:500]}" for i in range(start, end + 1))
        header = f"{path} (lines {start}-{end} of {n})"
        if end < n:
            header += " — use start_line/end_line to read more"
        return f"{header}\n{body}"
