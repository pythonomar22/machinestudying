"""StudyBench questions and the pinned corpora they are asked about."""

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Corpus:
    name: str  # dataset config name ("dspy" / "openclaw")
    display: str  # name used in prompts ("DSPy" / "OpenClaw")
    repo: Path  # local checkout at the pinned commit
    roots: tuple[str, ...]  # top-level dirs the agent may access (no docs at test time)
    language: str  # language of the expected answer code: "python" | "typescript"


CORPORA = {
    "dspy": Corpus("dspy", "DSPy", ROOT / "corpora/dspy", ("dspy", "tests"), "python"),
    "openclaw": Corpus(
        "openclaw", "OpenClaw", ROOT / "corpora/openclaw", ("src", "extensions"), "typescript"
    ),
}


def load_questions(task: str) -> list[dict]:
    with open(ROOT / f"data/{task}.jsonl") as f:
        return [json.loads(line) for line in f]
