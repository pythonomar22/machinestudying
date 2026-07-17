"""Shared plumbing for the question-collection pipeline.

Every OpenAI call made by any stage goes through `chat()` so the complete
request and response are appended to `logs/api/<stage>.jsonl`.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

DC = Path(__file__).resolve().parent
ROOT = DC.parent
ARTIFACTS = DC / "artifacts"
LOGS = DC / "logs"
API_LOGS = LOGS / "api"

GITHUB_OWNER = "stanfordnlp"
GITHUB_REPO = "dspy"
OPENAI_MODEL = "gpt-5.4"          # same model id the paper-tier judge uses
REASONING_EFFORT = "xhigh"        # paper: "GPT-5.4 in Codex at xhigh effort"
MASTER_SEED = 20260716            # fixes seed-session sampling and dedup order


def load_env() -> None:
    """Load KEY=VALUE lines from ROOT/.env without overriding the environment."""
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


SHINGLE_WORDS = 3


def normalize(text: str) -> str:
    import re

    return re.sub(r"\s+", " ", text.lower()).strip()


def shingles(text: str) -> set[str]:
    """Word n-gram shingles used for near-dedup and decontamination."""
    words = normalize(text).split()
    if len(words) < SHINGLE_WORDS:
        return {" ".join(words)} if words else set()
    return {
        " ".join(words[i : i + SHINGLE_WORDS])
        for i in range(len(words) - SHINGLE_WORDS + 1)
    }


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _log(stage: str, started: float, request: dict, response) -> None:
    API_LOGS.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": started,
        "elapsed_seconds": round(time.time() - started, 3),
        "request": request,
        "response": response.model_dump(exclude_none=True),
    }
    with open(API_LOGS / f"{stage}.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def chat(client, stage: str, request: dict):
    """One logged chat.completions call; returns the OpenAI response object."""
    started = time.time()
    response = client.chat.completions.create(**request)
    _log(stage, started, request, response)
    return response


def respond(client, stage: str, request: dict):
    """One logged Responses API call (needed for tools + reasoning effort)."""
    started = time.time()
    response = client.responses.create(**request)
    _log(stage, started, request, response)
    return response
