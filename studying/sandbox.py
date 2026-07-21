"""Run one candidate program offline in the pinned DSPy sandbox."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from studybench.dataset import ROOT

VENV_PY = ROOT / ".venv-dspy" / "bin" / "python"
RUN_TIMEOUT = 180
TAIL_CHARS = 3_000
PROGRAM_FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)
_BLOCKED = ("API_KEY", "TOKEN", "SECRET", "_PAT")


def extract_program(text: str) -> tuple[str | None, int]:
    """The longest fenced python block, plus how many blocks the text holds."""

    blocks = PROGRAM_FENCE.findall(text)
    return (max(blocks, key=len) if blocks else None), len(blocks)


def run_program(program: str, run_dir: Path) -> dict:
    """Execute one program in the pinned sandbox; archive and return the outcome."""

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "program.py").write_text(program, encoding="utf-8")
    try:
        compile(program, "program.py", "exec")
    except SyntaxError as error:
        result = {"compiled": False, "returncode": None, "timeout": False,
                  "stdout": "", "stderr": f"SyntaxError: {error}"}
        (run_dir / "verdict.txt").write_text("syntax error\n", encoding="utf-8")
        return result
    env = {key: value for key, value in os.environ.items()
           if not any(word in key.upper() for word in _BLOCKED)}
    env["DSPY_CACHEDIR"] = str(run_dir / "cache")
    try:
        completed = subprocess.run(
            [str(VENV_PY), "program.py"], cwd=run_dir, env=env,
            capture_output=True, text=True, timeout=RUN_TIMEOUT,
        )
        result = {"compiled": True, "returncode": completed.returncode,
                  "timeout": False, "stdout": completed.stdout[-TAIL_CHARS:],
                  "stderr": completed.stderr[-TAIL_CHARS:]}
    except subprocess.TimeoutExpired as error:
        result = {"compiled": True, "returncode": None, "timeout": True,
                  "stdout": str(error.stdout or "")[-TAIL_CHARS:],
                  "stderr": str(error.stderr or "")[-TAIL_CHARS:]}
    (run_dir / "stdout.txt").write_text(result["stdout"], encoding="utf-8")
    (run_dir / "stderr.txt").write_text(result["stderr"], encoding="utf-8")
    (run_dir / "verdict.txt").write_text(
        ("timeout" if result["timeout"] else f"returncode {result['returncode']}") + "\n",
        encoding="utf-8",
    )
    return result


def passed(result: dict) -> bool:
    return result["compiled"] and not result["timeout"] and result["returncode"] == 0
