"""Teacher trajectories: codex gpt-5.4-mini answers practice questions at
forced exploration budgets.

Diagnostic collection for the fold-back method: a stronger model explores
the pinned fulldspy corpus under exactly-k exploration budgets so we can
study WHAT large search buys (locations, connections, verification habits)
and distill it into k=0 study objects. Uses the ChatGPT-account codex CLI —
never the OpenAI API key. Nothing is graded here; the raw event stream is
the ground truth for tool calls, and the model self-reports per-step
motivation and discovery in `tool_log`.

Usage:
    .venv-dspy/bin/python -m studying.foldback.teacher --run-id RUN \
        --seed SEED [--budgets 5,20,50] [--workers 8] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from studybench.artifacts import read_json, sha256_text, write_json, write_text
from studybench.dataset import CORPORA, ROOT
from studybench.tools import RepoTools

from ..sandbox import extract_program, passed, run_program
from .data import load_practice_questions, practice_dataset_sha256, split_practice

MODEL = "gpt-5.4-mini"
EFFORT = "medium"
CODEX_TIMEOUT = 5_400
CORPUS_REPO = ROOT / "corpora" / "dspy"
DEFAULT_BUDGETS = (5, 20, 50)
DEFAULT_WORKERS = 8
log = logging.getLogger("studying.foldback.teacher")

PROMPT = """You are an expert answering one question about the repository in your working directory (a pinned DSPy checkout: `dspy/` source and `tests/`; treat `docs/` as absent at answer time — your answer must be recoverable from code and tests alone).

## Exploration budget (hard requirement)
Run EXACTLY {k} repository-exploration shell commands — no fewer, no more. One command per step (a single grep/find/ls/sed/cat/head invocation counts as one step). If you feel confident before spending the budget, KEEP EXPLORING: use the remaining steps to verify your answer against the source, check edge cases, and confirm every API you cite exists in THIS checkout. Do not run python; explore by reading source only.

## Motivation discipline (for later analysis)
Before every command, decide exactly what you want to learn from it. You will report every step afterwards in `tool_log`: the command you ran, WHY you ran it (`motivation`), and what it actually revealed (`discovery`, one or two sentences, concrete — file paths, symbols, behaviors). The log must match the commands you really executed, in order.

## Question
{question}

## Answer contract
The `answer` field must END with your complete deliverable: exactly ONE fenced ```python block containing a small, self-contained, runnable program (offline, no API key; whenever a language model is needed use the offline stub this library ships for its own tests, `dspy.utils.dummies.DummyLM`), ending with the prints/assertions the question demands. Any explanation goes in short prose BEFORE the fence or in inline comments. Every API you use must be one you verified in this checkout during exploration.

Return JSON matching the provided schema and nothing else."""

DIRECT_PROMPT = """You are an expert answering one question about the repository in your working directory (a pinned DSPy checkout of the DSPy library). This is a CLOSED-BOOK answer.

## Exploration budget (hard requirement)
Run EXACTLY 0 shell commands. Do not explore, list, read, or search anything — no commands of any kind. Answer purely from what you already know about this library. `tool_log` must be an empty array.

## Question
{question}

## Answer contract
The `answer` field must END with your complete deliverable: exactly ONE fenced ```python block containing a small, self-contained, runnable program (offline, no API key; whenever a language model is needed use the offline stub this library ships for its own tests, `dspy.utils.dummies.DummyLM`), ending with the prints/assertions the question demands. Any explanation goes in short prose BEFORE the fence or in inline comments. Use only APIs you are confident exist in this library.

Return JSON matching the provided schema and nothing else."""

SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "tool_log": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "step": {"type": "integer"},
                    "command": {"type": "string"},
                    "motivation": {"type": "string"},
                    "discovery": {"type": "string"},
                },
                "required": ["step", "command", "motivation", "discovery"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["answer", "tool_log"],
    "additionalProperties": False,
}

ALLOWED_PATH_PREFIXES = ("/bin", "/usr", "/dev", "/proc", "/etc", "/lib", "/sbin")
ABS_PATH = re.compile(r"(?:^|[\s'\"=(:;])(/[A-Za-z0-9_.@/-]+)")
ESCAPE_TOKENS = re.compile(r"(?:(?<![\w.])\.\./|(?:^|[\s'\"=(:;])~(?=[/\s'\"]|$))")


def command_executions(events_path: Path) -> list[str]:
    commands = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        item = event.get("item") or {}
        if event.get("type") == "item.completed" and item.get("type") == "command_execution":
            commands.append(item.get("command") or "")
    return commands


def escape_violations(commands: list[str]) -> list[str]:
    corpus = str(CORPUS_REPO)
    violations = []
    for command in commands:
        bad = [
            token for token in ABS_PATH.findall(command)
            if not token.startswith(corpus) and not token.startswith(ALLOWED_PATH_PREFIXES)
        ]
        if ESCAPE_TOKENS.search(command):
            bad.append("~ or ../ escape")
        if bad:
            violations.append(f"{command[:160]} -> {sorted(set(bad))[:4]}")
    return violations


def budget_dir(k: int) -> str:
    return "direct" if k == 0 else f"k{k}f"


def attempt_case(row: dict, k: int, out_dir: Path) -> dict:
    """One codex attempt; idempotent via record.json."""

    record_path = out_dir / "record.json"
    if record_path.exists():
        return read_json(record_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt = (DIRECT_PROMPT.format(question=row["question"]) if k == 0
              else PROMPT.format(k=k, question=row["question"]))
    write_text(out_dir / "prompt.txt", prompt)
    schema_path = out_dir / "schema.json"
    schema_path.write_text(json.dumps(SCHEMA, indent=1), encoding="utf-8")
    last_message = out_dir / "last_message.json"
    events_path = out_dir / "events.jsonl"
    command = [
        "codex", "exec",
        "-m", MODEL,
        "-c", f"model_reasoning_effort={EFFORT}",
        "-s", "read-only",
        "-C", str(CORPUS_REPO),
        "--output-schema", str(schema_path),
        "-o", str(last_message),
        "--json",
        "-",
    ]
    with open(events_path, "w", encoding="utf-8") as events:
        result = subprocess.run(
            command, input=prompt, stdout=events, stderr=subprocess.PIPE,
            text=True, timeout=CODEX_TIMEOUT,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"codex exec failed for {row['id']} k={k} ({result.returncode}): "
            f"{result.stderr[-2000:]}"
        )
    payload = json.loads(last_message.read_text(encoding="utf-8"))
    answer = payload["answer"]
    write_text(out_dir / "answer.md", answer)

    commands = command_executions(events_path)
    program, fences = extract_program(answer)
    sandbox = run_program(program, out_dir / "sandbox") if program is not None else None
    record = {
        "qid": row["id"],
        "topic": row["topic"],
        "budget_k": k,
        "model": MODEL,
        "effort": EFFORT,
        "answer_chars": len(answer),
        "fenced_python_blocks": fences,
        "reported_steps": len(payload["tool_log"]),
        "executed_commands": len(commands),
        "budget_compliant": len(commands) == k,
        "escape_violations": escape_violations(commands),
        "sandbox": None if sandbox is None else {
            "compiled": sandbox["compiled"],
            "returncode": sandbox["returncode"],
            "timeout": sandbox["timeout"],
            "passed": passed(sandbox),
        },
    }
    write_json(record_path, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", required=True, type=int,
                        help="master seed; selects the same 70/30 split as the study loop")
    parser.add_argument("--budgets", default=",".join(str(k) for k in DEFAULT_BUDGETS))
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--limit", type=int, default=0,
                        help="smoke: only the first N study questions")
    args = parser.parse_args()
    if not args.run_id.replace("-", "").replace("_", "").isalnum():
        parser.error("--run-id must contain only letters, digits, '-' and '_'")
    budgets = [int(k) for k in args.budgets.split(",")]
    if not budgets or any(k < 0 for k in budgets):
        parser.error("--budgets must be non-negative integers (0 = direct, closed-book)")

    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(ROOT / "logs" / f"{args.run_id}-teacher.log"),
        ],
    )

    corpus = CORPORA["dspy"]
    repository = RepoTools(corpus)  # verifies the checkout and pins the snapshot
    rows = load_practice_questions()
    by_id = {row["id"]: row for row in rows}
    split = split_practice(rows, args.seed)
    study_ids = split["study_ids"][: args.limit or None]

    run_root = ROOT / "runs" / args.run_id / "dspy"
    run_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "kind": "foldback-teacher",
        "run_id": args.run_id,
        "task": "dspy",
        "model": MODEL,
        "effort": EFFORT,
        "harness": "codex exec (codex-cli), read-only, corpus checkout; ChatGPT-account auth",
        "corpus_commit": corpus.commit,
        "corpus_snapshot_sha256": repository.snapshot_sha256,
        "practice_dataset_sha256": practice_dataset_sha256(),
        "master_seed": args.seed,
        "split": "study slice of the seeded 70/30 stratified split (dev untouched)",
        "budgets_forced_commands": budgets,
        "prompt_sha256": sha256_text(PROMPT),
        "workers": args.workers,
        "limit": args.limit,
        "questions": list(study_ids),
        "grading": "deliberately absent — runs only; sandbox execution is deterministic logging",
    }
    # The stored manifest is authoritative; adding NEW budgets to an existing
    # run is allowed (recorded as a union), everything else must match.
    STABLE = ("kind", "task", "model", "effort", "harness", "corpus_commit",
              "corpus_snapshot_sha256", "practice_dataset_sha256", "master_seed",
              "split", "questions", "grading", "limit", "prompt_sha256")
    manifest["direct_prompt_sha256"] = sha256_text(DIRECT_PROMPT)
    manifest_path = run_root / "teacher.json"
    if manifest_path.exists():
        existing = read_json(manifest_path)
        mismatch = [key for key in STABLE if existing.get(key) != manifest.get(key)]
        if mismatch:
            raise SystemExit(
                f"teacher configuration changed ({mismatch}); use a new --run-id: {manifest_path}"
            )
        existing["budgets_forced_commands"] = sorted(
            set(existing["budgets_forced_commands"]) | set(budgets)
        )
        existing["workers"] = args.workers
        existing["direct_prompt_sha256"] = manifest["direct_prompt_sha256"]
        write_json(manifest_path, existing)
        manifest = existing
    else:
        write_json(manifest_path, manifest)

    cases = [(by_id[qid], k) for k in budgets for qid in study_ids]
    log.info("%d cases: %d questions x budgets %s, %d workers",
             len(cases), len(study_ids), budgets, args.workers)

    failures = []

    def run_case(case) -> None:
        row, k = case
        try:
            record = attempt_case(row, k, run_root / budget_dir(k) / row["id"])
            log.info("%s %s: commands=%d compliant=%s fences=%d sandbox=%s",
                     budget_dir(k), row["id"], record["executed_commands"],
                     record["budget_compliant"], record["fenced_python_blocks"],
                     (record["sandbox"] or {}).get("passed"))
        except Exception as error:
            failures.append((row["id"], k, str(error)))
            log.error("k%df %s FAILED: %s", k, row["id"], str(error)[:300])

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run_case, cases))
    if failures:
        raise SystemExit(
            f"{len(failures)} case(s) failed — rerun the same command to retry them: "
            f"{[(qid, k) for qid, k, _ in failures][:10]}"
        )

    summary = {"budgets": {}}
    for k in manifest["budgets_forced_commands"]:
        records = [read_json(run_root / budget_dir(k) / qid / "record.json") for qid in study_ids]
        with_program = [r for r in records if r["sandbox"] is not None]
        summary["budgets"][budget_dir(k)] = {
            "cases": len(records),
            "answers_with_single_fence": sum(r["fenced_python_blocks"] == 1 for r in records),
            "budget_compliant": sum(r["budget_compliant"] for r in records),
            "mean_executed_commands": round(
                sum(r["executed_commands"] for r in records) / len(records), 2),
            "sandbox_passed": sum(r["sandbox"]["passed"] for r in with_program),
            "escape_violation_cases": sum(bool(r["escape_violations"]) for r in records),
            "mean_answer_chars": round(sum(r["answer_chars"] for r in records) / len(records)),
        }
    write_json(run_root / "summary.json", summary)
    log.info("teacher runs complete: %s", json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
