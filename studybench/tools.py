"""The paper's three repository tools: grep, glob, and ranged file reads."""

from __future__ import annotations

import bisect
import fnmatch
import glob as globlib
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import regex

from .dataset import Corpus, read_corpus_file, verify_corpus

GREP_MAX_MATCHES = 50
GLOB_MAX_PATHS = 200
READ_MAX_LINES = 200
OBSERVATION_MAX_CHARS = 25_000
MAX_FILE_BYTES = 5_000_000


def _glob_to_regex(pattern: str) -> str:
    """Translate globs with ``**`` for the Python 3.12 runner."""

    index, out = 0, []
    while index < len(pattern):
        char = pattern[index]
        if pattern[index : index + 3] == "**/":
            out.append("(?:[^/]+/)*")
            index += 3
        elif pattern[index : index + 2] == "**":
            out.append(".*")
            index += 2
        elif char == "*":
            out.append("[^/]*")
            index += 1
        elif char == "?":
            out.append("[^/]")
            index += 1
        elif char == "[":
            end = index + 1
            if end < len(pattern) and pattern[end] in "!]":
                end += 1
            while end < len(pattern) and pattern[end] != "]":
                end += 1
            if end == len(pattern):
                out.append(r"\[")
                index += 1
            else:
                character_class = pattern[index + 1 : end]
                if character_class.startswith("!"):
                    character_class = "^" + character_class[1:]
                out.append(f"[{character_class}]")
                index = end + 1
        else:
            out.append(re.escape(char))
            index += 1
    return "".join(out) + r"\Z"


class RepoTools:
    """An immutable in-memory view of one clean, pinned corpus checkout."""

    def __init__(self, corpus: Corpus):
        verify_corpus(corpus)
        result = subprocess.run(
            ["git", "-C", str(corpus.repo), "ls-files", "-z", "--", *corpus.roots],
            check=True,
            capture_output=True,
            timeout=30,
        )
        relatives = sorted(path.decode() for path in result.stdout.split(b"\0") if path)

        def load(relative: str) -> tuple[str, str] | None:
            path = corpus.repo / relative
            if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
                return None
            data = path.read_bytes()
            if b"\0" in data[:8192]:
                return None
            return relative, read_corpus_file(corpus, relative)

        with ThreadPoolExecutor(max_workers=16) as pool:
            loaded = pool.map(load, relatives)
        self.text = dict(item for item in loaded if item is not None)
        self.files = tuple(self.text)
        self._line_starts = {
            path: [0, *(match.end() for match in re.finditer("\n", text))]
            for path, text in self.text.items()
        }

    @staticmethod
    def _cap(observation: str) -> str:
        if len(observation) <= OBSERVATION_MAX_CHARS:
            return observation
        return observation[:OBSERVATION_MAX_CHARS] + "\n... (output truncated)"

    def grep(self, pattern: str, path: str = "") -> str:
        try:
            expression = regex.compile(pattern)
        except regex.error:
            expression = regex.compile(regex.escape(pattern))
        path = path.strip("/")
        candidates = [
            name for name in self.files
            if not path or name == path or name.startswith(path + "/")
        ]
        if path and not candidates:
            return f"Error: no files under '{path}'."
        matches: list[str] = []
        deadline = time.monotonic() + 10
        truncated = False
        for name in candidates:
            starts, previous_line = self._line_starts[name], -1
            try:
                for match in expression.finditer(
                    self.text[name], timeout=max(0.05, deadline - time.monotonic())
                ):
                    line = bisect.bisect_right(starts, match.start())
                    if line == previous_line:
                        continue
                    previous_line = line
                    content = self.text[name][starts[line - 1] :].split("\n", 1)[0]
                    matches.append(f"{name}:{line}:{content[:500]}")
                    if len(matches) == GREP_MAX_MATCHES:
                        truncated = True
                        break
            except TimeoutError:
                truncated = True
            if truncated or time.monotonic() >= deadline:
                break
        if not matches:
            return f"No matches for /{pattern}/."
        output = "\n".join(matches)
        if truncated:
            output += "\n... (more matches may exist; narrow the search)"
        return self._cap(output)

    def glob(self, pattern: str) -> str:
        pattern = pattern.strip("/")
        if hasattr(globlib, "translate"):
            expression = re.compile(globlib.translate(pattern, recursive=True, include_hidden=True))
        else:
            expression = re.compile(_glob_to_regex(pattern))
        matches = [path for path in self.files if expression.match(path)]
        if not matches and "/" not in pattern:
            matches = [path for path in self.files if fnmatch.fnmatch(path.rsplit("/", 1)[-1], pattern)]
        if not matches:
            return f"No files match '{pattern}'."
        output = "\n".join(matches[:GLOB_MAX_PATHS])
        if len(matches) > GLOB_MAX_PATHS:
            output += f"\n... ({len(matches) - GLOB_MAX_PATHS} more files not shown)"
        return self._cap(output)

    def read_file(self, path: str, start_line: int = 1, end_line: int = 0) -> str:
        path = path.strip("/")
        if path not in self.text:
            return f"Error: '{path}' is not a readable file in this repository."
        lines = self.text[path].splitlines()
        start = max(1, int(start_line))
        requested_end = int(end_line) if end_line else len(lines)
        end = min(len(lines), requested_end, start + READ_MAX_LINES - 1)
        if start > len(lines) or end < start:
            return f"Error: '{path}' has {len(lines)} lines; requested {start}-{requested_end}."
        body = "\n".join(f"{number}: {lines[number - 1][:500]}" for number in range(start, end + 1))
        header = f"{path} (lines {start}-{end} of {len(lines)})"
        if end < len(lines):
            header += " — use start_line/end_line to read more"
        return self._cap(f"{header}\n{body}")
