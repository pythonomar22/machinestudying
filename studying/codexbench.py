"""StudyBench conditions for codex gpt-5.4-mini on the real test questions.

The paper protocol transplanted onto a codex agent: 30 Study-DSPy test
questions x four budgets x three rollouts, conditions baseline (no study)
and cheatsheet (50 forced exploration commands -> note prepended to every
question). Budgets mirror the Qwen semantics in codex terms:

  direct  closed-book, zero commands
  k5      AT MOST 5 exploration commands, voluntary early stop
  k20     AT MOST 20, voluntary early stop
  k20f    EXACTLY 20, no early stop

Generated tokens = codex output tokens (reasoning included), parsed from
the event stream; rollouts are independent resamples (codex exposes no
sampling seed). Grading uses the paper-contract GPT-5.4 judge and the
official four-point WAUC.

Usage:
    .venv-dspy/bin/python -m studying.codexbench run --run-id RUN \
        --condition baseline|cheatsheet [--rollouts 3] [--workers 10] [--limit N]
    .venv-dspy/bin/python -m studying.codexbench grade --run-id RUN
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from studybench.artifacts import read_json, sha256_text, write_json, write_text
from studybench.dataset import CORPORA, NOTE_PREFIX, ROOT, load_questions
from studybench.tools import RepoTools

from .sandbox import extract_program, passed, run_program

MODEL = "gpt-5.4-mini"
EFFORT = "medium"
CODEX_TIMEOUT = 5_400
CORPUS_REPO = ROOT / "corpora" / "dspy"
BUDGETS = ("direct", "k5", "k20", "k20f")
STUDY_COMMANDS = 50
log = logging.getLogger("studying.codexbench")

CONTRACT = """## Answer contract
The `answer` field must END with your complete deliverable: exactly ONE fenced ```python block containing a small, self-contained, runnable program (offline, no API key; whenever a language model is needed use the offline stub this library ships for its own tests, `dspy.utils.dummies.DummyLM`), ending with the prints/assertions the question demands. Any explanation goes in short prose BEFORE the fence or in inline comments.

Return JSON matching the provided schema and nothing else."""

HEADER = """You are an expert answering one question about the repository in your working directory (a pinned DSPy checkout: `dspy/` source and `tests/`; treat `docs/` as absent at answer time)."""

BUDGET_SECTIONS = {
    "direct": """## Exploration budget (hard requirement)
Run EXACTLY 0 shell commands. Do not explore, list, read, or search anything. Answer purely from what you already know. `tool_log` must be an empty array.""",
    "k5": """## Exploration budget (hard requirement)
Run AT MOST 5 repository-exploration shell commands (one command per step). Stop exploring as soon as you are confident — you do not need to use the full budget. Do not run python; explore by reading source only. Report every command you actually ran in `tool_log` (command, motivation, discovery), in order.""",
    "k20": """## Exploration budget (hard requirement)
Run AT MOST 20 repository-exploration shell commands (one command per step). Stop exploring as soon as you are confident — you do not need to use the full budget. Do not run python; explore by reading source only. Report every command you actually ran in `tool_log` (command, motivation, discovery), in order.""",
    "k20f": """## Exploration budget (hard requirement)
Run EXACTLY 20 repository-exploration shell commands — no fewer, no more (one command per step). If you feel confident before spending the budget, KEEP EXPLORING: verify your answer against the source, check edge cases, and confirm every API you cite exists in THIS checkout. Do not run python; explore by reading source only. Report every command in `tool_log` (command, motivation, discovery), in order.""",
}
BUDGET_TARGET = {"direct": 0, "k5": 5, "k20": 20, "k20f": 20}

STUDY_PROMPT = """You are studying the DSPy repository in your working directory (`dspy/` source and `tests/`; treat `docs/` as absent). Write yourself a cheatsheet: a reference document that will be prepended to every future question you answer about DSPy. You will not see those questions in advance, and repository tools will remain available when you answer them.

Explore with EXACTLY {n} repository-exploration shell commands — no fewer, no more (one command per step; no python). If you feel confident before spending the budget, keep exploring and verifying. Report every command in `tool_log` (command, motivation, discovery). Then return the complete cheatsheet in the `cheatsheet` field.

Return JSON matching the provided schema and nothing else."""

TOOL_LOG_SCHEMA = {
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
}
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}, "tool_log": TOOL_LOG_SCHEMA},
    "required": ["answer", "tool_log"],
    "additionalProperties": False,
}
STUDY_SCHEMA = {
    "type": "object",
    "properties": {"cheatsheet": {"type": "string"}, "tool_log": TOOL_LOG_SCHEMA},
    "required": ["cheatsheet", "tool_log"],
    "additionalProperties": False,
}


def run_codex(prompt: str, out_dir: Path, schema: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_text(out_dir / "prompt.txt", prompt)
    schema_path = out_dir / "schema.json"
    schema_path.write_text(json.dumps(schema, indent=1), encoding="utf-8")
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
        raise RuntimeError(f"codex exec failed ({result.returncode}): {result.stderr[-2000:]}")
    return json.loads(last_message.read_text(encoding="utf-8"))


def session_stats(events_path: Path) -> dict:
    commands, gen_tokens, reasoning = [], 0, 0
    for line in events_path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        item = event.get("item") or {}
        if event.get("type") == "item.completed" and item.get("type") == "command_execution":
            commands.append(item.get("command") or "")
        if event.get("type") == "turn.completed":
            usage = event.get("usage") or {}
            gen_tokens += int(usage.get("output_tokens") or 0)
            reasoning += int(usage.get("reasoning_output_tokens") or 0)
    return {"commands": commands, "gen_tokens": gen_tokens, "reasoning_tokens": reasoning}


def eval_case(row: dict, budget: str, rollout: int, prefix: str, out_dir: Path) -> dict:
    record_path = out_dir / "record.json"
    if record_path.exists():
        return read_json(record_path)
    prompt = "\n\n".join([
        HEADER,
        BUDGET_SECTIONS[budget],
        "## Question\n" + prefix + row["question"],
        CONTRACT,
    ])
    payload = run_codex(prompt, out_dir, ANSWER_SCHEMA)
    answer = payload["answer"]
    write_text(out_dir / "answer.md", answer)
    stats = session_stats(out_dir / "events.jsonl")
    target = BUDGET_TARGET[budget]
    executed = len(stats["commands"])
    compliant = executed == 0 if budget == "direct" else (
        executed <= target if budget in ("k5", "k20") else executed == target
    )
    program, fences = extract_program(answer)
    sandbox = run_program(program, out_dir / "sandbox") if program is not None else None
    record = {
        "qid": row["id"],
        "budget": budget,
        "rollout": rollout,
        "model": MODEL,
        "effort": EFFORT,
        "answer_chars": len(answer),
        "fenced_python_blocks": fences,
        "reported_steps": len(payload["tool_log"]),
        "executed_commands": executed,
        "budget_compliant": compliant,
        "gen_tokens": stats["gen_tokens"],
        "reasoning_tokens": stats["reasoning_tokens"],
        "sandbox": None if sandbox is None else {
            "compiled": sandbox["compiled"], "returncode": sandbox["returncode"],
            "timeout": sandbox["timeout"], "passed": passed(sandbox),
        },
    }
    write_json(record_path, record)
    return record


def study_phase(run_root: Path) -> str:
    note_path = run_root / "cheatsheet.md"
    if note_path.exists():
        return note_path.read_text(encoding="utf-8")
    out_dir = run_root / "study"
    payload = run_codex(STUDY_PROMPT.format(n=STUDY_COMMANDS), out_dir, STUDY_SCHEMA)
    note = payload["cheatsheet"]
    if not note.strip():
        raise RuntimeError("study session produced an empty cheatsheet")
    stats = session_stats(out_dir / "events.jsonl")
    write_json(out_dir / "study.json", {
        "kind": "codex-cheatsheet",
        "commands_target": STUDY_COMMANDS,
        "executed_commands": len(stats["commands"]),
        "gen_tokens": stats["gen_tokens"],
        "reasoning_tokens": stats["reasoning_tokens"],
        "reported_steps": len(payload["tool_log"]),
        "cheatsheet_sha256": sha256_text(note),
        "cheatsheet_chars": len(note),
    })
    write_text(note_path, note)
    return note


def cmd_run(args) -> None:
    rows = list(load_questions("dspy"))[: args.limit or None]
    repository = RepoTools(CORPORA["dspy"])  # verifies pinned checkout
    run_root = ROOT / "runs" / args.run_id / "dspy"
    run_root.mkdir(parents=True, exist_ok=True)

    note = study_phase(run_root) if args.condition == "cheatsheet" else ""
    prefix = NOTE_PREFIX.format(library="DSPy", note=note) if note else ""

    manifest = {
        "schema_version": 1,
        "kind": "codexbench",
        "run_id": args.run_id,
        "task": "dspy",
        "condition": args.condition,
        "model": MODEL,
        "effort": EFFORT,
        "harness": "codex exec (codex-cli), read-only shell at the corpus checkout; ChatGPT-account auth",
        "corpus_commit": CORPORA["dspy"].commit,
        "corpus_snapshot_sha256": repository.snapshot_sha256,
        "dataset_sha256": CORPORA["dspy"].dataset_sha256,
        "budgets": list(BUDGETS),
        "budget_semantics": "direct=0; k5<=5 voluntary; k20<=20 voluntary; k20f==20 forced (shell commands)",
        "rollouts": args.rollouts,
        "sampling": "provider default; rollouts are independent resamples (no seed control)",
        "gen_token_axis": "codex output tokens incl. reasoning, from the event stream",
        "note_sha256": sha256_text(note) if note else None,
        "question_ids": [row["id"] for row in rows],
        "limit": args.limit,
    }
    manifest_path = run_root / "codex.json"
    if manifest_path.exists():
        existing = read_json(manifest_path)
        if {k: v for k, v in existing.items() if k != "workers"} != manifest:
            raise SystemExit(f"run configuration changed; use a new --run-id: {manifest_path}")
    else:
        write_json(manifest_path, manifest)

    cases = [(row, budget, rollout)
             for budget in BUDGETS
             for rollout in range(args.rollouts)
             for row in rows]
    log.info("%s: %d cases (%d questions x %d budgets x %d rollouts), %d workers",
             args.condition, len(cases), len(rows), len(BUDGETS), args.rollouts, args.workers)

    failures = []

    def one(case):
        row, budget, rollout = case
        try:
            record = eval_case(row, budget, rollout, prefix,
                               run_root / budget / f"r{rollout}" / row["id"])
            log.info("%s/r%d/%s: cmds=%d ok=%s gen=%d sandbox=%s",
                     budget, rollout, row["id"], record["executed_commands"],
                     record["budget_compliant"], record["gen_tokens"],
                     (record["sandbox"] or {}).get("passed"))
        except Exception as error:
            failures.append((row["id"], budget, rollout, str(error)))
            log.error("%s/r%d/%s FAILED: %s", budget, rollout, row["id"], str(error)[:300])

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(one, cases))
    if failures:
        raise SystemExit(f"{len(failures)} case(s) failed — rerun the same command to retry: "
                         f"{[(q, b, r) for q, b, r, _ in failures][:10]}")

    summary = {"condition": args.condition, "budgets": {}}
    for budget in BUDGETS:
        records = [read_json(run_root / budget / f"r{r}" / row["id"] / "record.json")
                   for r in range(args.rollouts) for row in rows]
        with_program = [rec for rec in records if rec["sandbox"] is not None]
        summary["budgets"][budget] = {
            "cases": len(records),
            "mean_gen_tokens": round(sum(r["gen_tokens"] for r in records) / len(records)),
            "mean_executed_commands": round(
                sum(r["executed_commands"] for r in records) / len(records), 2),
            "budget_compliant": sum(r["budget_compliant"] for r in records),
            "single_fence": sum(r["fenced_python_blocks"] == 1 for r in records),
            "sandbox_passed": sum(r["sandbox"]["passed"] for r in with_program),
        }
    write_json(run_root / "summary.json", summary)
    log.info("run complete: %s", json.dumps(summary, indent=1))


def cmd_grade(args) -> None:
    import os

    from openai import OpenAI

    from studybench.grade import build_prompt, response_schema, score_verdict, weighted_auc

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required for the paper judge")
    run_root = ROOT / "runs" / args.run_id / "dspy"
    manifest = read_json(run_root / "codex.json")
    rows = {row["id"]: row for row in load_questions("dspy")}
    qids = manifest["question_ids"]
    rollouts = manifest["rollouts"]
    grade_root = ROOT / "grades" / args.run_id / "gpt-5-4" / "dspy"
    api = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=600, max_retries=2)

    def grade_case(case):
        budget, rollout, qid = case
        path = grade_root / budget / f"r{rollout}" / f"{qid}.json"
        if path.exists():
            return budget, read_json(path)
        case_dir = run_root / budget / f"r{rollout}" / qid
        answer = (case_dir / "answer.md").read_text(encoding="utf-8")
        record = read_json(case_dir / "record.json")
        if not answer.strip():
            grade = {"claims": [
                {"claim_id": c["claim_id"], "score": 0, "rationale": "No answer."}
                for c in rows[qid]["rubric"]], "lenient": 0, "judge_response": None}
        else:
            prompt = build_prompt("dspy", rows[qid], answer, "paper")
            response = api.chat.completions.create(
                model="gpt-5.4",
                messages=[{"role": "user", "content": prompt}],
                response_format=response_schema(rows[qid], "paper"),
            )
            verdict = json.loads(response.choices[0].message.content)
            claims, score, question_score = score_verdict(rows[qid], verdict, "paper")
            grade = {"claims": claims, "lenient": score,
                     "judge_question_score": question_score,
                     "judge_response": {
                         "id": response.id, "model": response.model,
                         "finish_reason": response.choices[0].finish_reason,
                         "usage": response.usage.model_dump(exclude_none=True)
                         if response.usage else None}}
        grade.update(qid=qid, budget=budget, rollout=rollout,
                     gen_tokens=record["gen_tokens"])
        write_json(path, grade)
        log.info("%s/r%d/%s lenient=%s", budget, rollout, qid, grade["lenient"])
        return budget, grade

    cases = [(budget, rollout, qid) for budget in BUDGETS
             for rollout in range(rollouts) for qid in qids]
    failures = []

    def safe_grade(case):
        for attempt in range(2):
            try:
                return grade_case(case)
            except Exception as error:  # per-case isolation; retried once
                last_error = error
        failures.append((case, str(last_error)[:300]))
        log.error("grade failed %s: %s", case, str(last_error)[:200])
        return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        graded = [g for g in pool.map(safe_grade, cases) if g is not None]
    if failures:
        write_json(grade_root / "grade_failures.json",
                   [{"case": list(case), "error": err} for case, err in failures])
        raise SystemExit(
            f"{len(failures)} case(s) failed to grade (see grade_failures.json); "
            "rerun to retry, or handle flagged cases deliberately"
        )

    report = {"run_id": args.run_id, "condition": manifest["condition"],
              "model": MODEL, "harness": manifest["harness"],
              "judge": {"model": "gpt-5.4", "contract": "paper"},
              "questions": len(qids), "rollouts": rollouts, "budgets": {}}
    points = []
    for budget in BUDGETS:
        population = [g for b, g in graded if b == budget]
        mean_lenient = sum(g["lenient"] for g in population) / len(population)
        mean_tokens = sum(g["gen_tokens"] for g in population) / len(population)
        report["budgets"][budget] = {
            "episodes": len(population),
            "mean_lenient": mean_lenient,
            "mean_generated_tokens": mean_tokens,
        }
        points.append((mean_tokens, mean_lenient))
    report["expertise_wauc"] = weighted_auc(points)
    write_json(grade_root / "report.json", report)
    for budget in BUDGETS:
        result = report["budgets"][budget]
        log.info("%s: mean_lenient=%.2f mean_gen_tokens=%.0f",
                 budget, result["mean_lenient"], result["mean_generated_tokens"])
    log.info("expertise (4-point WAUC): %.4f", report["expertise_wauc"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "grade"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--condition", choices=("baseline", "cheatsheet"), default="baseline")
    parser.add_argument("--rollouts", type=int, default=1)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if not args.run_id.replace("-", "").replace("_", "").isalnum():
        parser.error("--run-id must contain only letters, digits, '-' and '_'")
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(ROOT / "logs" / f"{args.run_id}-codexbench.log")],
    )
    if args.command == "run":
        cmd_run(args)
    else:
        cmd_grade(args)


if __name__ == "__main__":
    main()
