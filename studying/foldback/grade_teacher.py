"""Grade teacher trajectory runs against the rev-3 practice rubrics.

Paper-contract GPT-5.4 judge (same prompt/schema/scoring primitives as
studybench.grade) applied to each budget's answers; generated tokens per
session are parsed from the codex event stream (sum of output_tokens over
completed turns). Reports mean lenient + mean generated tokens per budget
and the WAUC over the measured points (same estimator as studybench).

Usage:
    set -a; source .env; set +a
    .venv-dspy/bin/python -m studying.foldback.grade_teacher --run-id RUN
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

import math

from studybench.artifacts import read_json, write_json
from studybench.dataset import ROOT

from .data import load_practice_questions
from .devval import _grade_one
from .teacher import budget_dir


def wauc(points: list[tuple[float, float]]) -> float:
    """Appendix-C WAUC over ANY number of measured points (studybench's
    weighted_auc guards the four-point paper protocol; this is the same
    integral: 3k anchor, weight halving per compute doubling, best-so-far
    step function, zero below the first point)."""

    xs = sorted((max(0.0, math.log10(tokens / 3000.0)), score) for tokens, score in points)
    total, best = 0.0, 0.0
    for index, (x, score) in enumerate(xs):
        best = max(best, score)
        upper = xs[index + 1][0] if index + 1 < len(xs) else None
        weight = 10.0 ** -x - (10.0 ** -upper if upper is not None else 0.0)
        total += weight * best
    return total

JUDGE_TIMEOUT = 600
WORKERS = 8
log = logging.getLogger("studying.foldback.grade_teacher")


def session_gen_tokens(events_path: Path) -> int:
    total = 0
    for line in events_path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "turn.completed":
            usage = event.get("usage") or {}
            total += int(usage.get("output_tokens") or 0)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    import os
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required for the paper judge")

    run_root = ROOT / "runs" / args.run_id / "dspy"
    manifest = read_json(run_root / "teacher.json")
    budgets = [budget_dir(k) for k in manifest["budgets_forced_commands"]]
    by_id = {row["id"]: row for row in load_practice_questions()}
    qids = manifest["questions"]
    api = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=JUDGE_TIMEOUT, max_retries=2)

    def grade_case(case) -> tuple[str, dict]:
        budget, qid = case
        path = run_root / "grades" / budget / f"{qid}.json"
        if path.exists():
            return budget, read_json(path)
        case_dir = run_root / budget / qid
        answer = (case_dir / "answer.md").read_text(encoding="utf-8")
        grade = _grade_one(api, by_id[qid], answer)
        grade.update(
            qid=qid,
            budget=budget,
            gen_tokens=session_gen_tokens(case_dir / "events.jsonl"),
        )
        write_json(path, grade)
        log.info("%s %s lenient=%s gen_tokens=%d", budget, qid,
                 grade["lenient"], grade["gen_tokens"])
        return budget, grade

    cases = [(budget, qid) for budget in budgets for qid in qids]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        graded = list(pool.map(grade_case, cases))

    report = {"run_id": args.run_id, "judge": {"model": "gpt-5.4", "contract": "paper"},
              "questions": len(qids), "budgets": {}}
    points = []
    for budget in budgets:
        population = [grade for b, grade in graded if b == budget]
        mean_lenient = sum(g["lenient"] for g in population) / len(population)
        mean_tokens = sum(g["gen_tokens"] for g in population) / len(population)
        report["budgets"][budget] = {
            "episodes": len(population),
            "mean_lenient": mean_lenient,
            "mean_generated_tokens": mean_tokens,
            "per_question": {g["qid"]: g["lenient"] for g in sorted(population, key=lambda g: g["qid"])},
        }
        points.append((mean_tokens, mean_lenient))
    report["wauc_measured_points"] = wauc(points)
    report["wauc_note"] = (
        "WAUC over the measured forced points only (no k=0 point: the axis "
        "head below the cheapest point scores zero; not comparable to "
        "test-set expertise numbers)"
    )
    write_json(run_root / "grades" / "report.json", report)
    for budget in budgets:
        result = report["budgets"][budget]
        log.info("%s: mean_lenient=%.2f mean_gen_tokens=%.0f",
                 budget, result["mean_lenient"], result["mean_generated_tokens"])
    log.info("wauc over measured points: %.4f", report["wauc_measured_points"])


if __name__ == "__main__":
    main()
