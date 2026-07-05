"""Step 1 of grading (per the first author): run the generated code in a sandbox to
check it is syntactically valid / compiles. Python answers (DSPy) are additionally
executed against the pinned DSPy install, which also catches hallucinated APIs
(the questions demand self-contained offline programs using DummyLM). TypeScript
answers (OpenClaw) import repo internals and cannot run standalone, so they get a
syntax check via tree-sitter.
"""

import re
import subprocess
import tempfile
from pathlib import Path

from .dataset import ROOT

PYTHON_BIN = ROOT / ".venv-dspy/bin/python"  # created by scripts/setup_grading.sh
RUN_TIMEOUT = 240

FENCE = re.compile(r"```[ \t]*([\w+-]*)[^\n]*\n(.*?)```", re.DOTALL)
LANG_TAGS = {
    "python": {"python", "python3", "py", ""},
    "typescript": {"typescript", "ts", "tsx", "javascript", "js", ""},
}


def extract_code(answer: str, language: str) -> list[tuple[str, str]]:
    """(tag, body) fenced code blocks in the answer matching the task language."""
    return [
        (tag.lower(), body) for tag, body in FENCE.findall(answer)
        if tag.lower() in LANG_TAGS[language] and body.strip()
    ]


def check(answer: str, language: str) -> dict:
    """Compilation check for one answer. Returns a dict with 'compile_ok' plus detail."""
    blocks = extract_code(answer, language)
    if not blocks:
        return {"compile_ok": False, "detail": "no code block found in answer"}
    # the main program = largest explicitly-tagged block; untagged fences (often
    # sample output or config) are considered only when no tagged block exists
    tagged = [b for b in blocks if b[0]]
    tag, program = max(tagged or blocks, key=lambda b: len(b[1]))
    if language == "python":
        return _check_python(program)
    return _check_typescript(program, tsx=tag == "tsx")


def _check_python(program: str) -> dict:
    try:
        compile(program, "<answer>", "exec")
    except SyntaxError as e:
        return {"compile_ok": False, "syntax_ok": False, "detail": f"SyntaxError: {e}"}
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "answer.py"
        path.write_text(program)
        try:
            # scrubbed env: no API keys from .env / the user shell reach generated code
            proc = subprocess.run(
                [str(PYTHON_BIN), "-I", str(path)], cwd=td,
                env={"PATH": "/usr/bin:/bin", "HOME": td, "TMPDIR": td},
                capture_output=True, text=True, timeout=RUN_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return {"compile_ok": False, "syntax_ok": True, "run_ok": False,
                    "detail": f"timeout after {RUN_TIMEOUT}s"}
    return {
        "compile_ok": proc.returncode == 0,
        "syntax_ok": True,
        "run_ok": proc.returncode == 0,
        "detail": (proc.stderr or proc.stdout)[-2000:],
    }


_TS_PARSERS: dict[bool, object] = {}


def _check_typescript(program: str, tsx: bool = False) -> dict:
    if tsx not in _TS_PARSERS:
        import tree_sitter_typescript as tst
        from tree_sitter import Language, Parser
        lang = tst.language_tsx() if tsx else tst.language_typescript()
        _TS_PARSERS[tsx] = Parser(Language(lang))
    tree = _TS_PARSERS[tsx].parse(program.encode())
    errors = []

    def walk(node):
        if node.type == "ERROR" or node.is_missing:
            errors.append(f"line {node.start_point[0] + 1}: {node.type}")
        elif node.has_error:  # only descend where an error hides
            for child in node.children:
                walk(child)

    walk(tree.root_node)
    return {
        "compile_ok": not errors,
        "syntax_ok": not errors,
        "detail": "; ".join(errors[:5]) if errors else "parses cleanly",
    }
