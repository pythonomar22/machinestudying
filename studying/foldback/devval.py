"""Dev-slice evaluation: score a note variant on the held-out 30 questions.

Direct + k5, paper-contract GPT-5.4 grading, idempotent per episode/grade.
This is the offline iteration signal; the dev slice never enters the study
object, and the held-out StudyBench test set is untouched until the one
declared test evaluation.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

from studybench.artifacts import read_json, stable_seed, write_json
from studybench.dataset import NOTE_PREFIX
from studybench.grade import build_prompt, response_schema, score_claims, score_verdict
from studybench.react import run_episode

JUDGE_MODEL = "gpt-5.4"
JUDGE_TIMEOUT = 600
BUDGET_ITERS = {"direct": 0, "k5": 5, "k20": 20}
log = logging.getLogger("studying.foldback.devval")


def _episode_ok(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        episode = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return episode.get("status") in {"ok", "no_answer", "gave_up"}


def _grade_one(api: OpenAI, row: dict, answer: str) -> dict:
    if not answer.strip():
        return {
            "claims": [
                {"claim_id": claim["claim_id"], "score": 0, "rationale": "No answer."}
                for claim in row["rubric"]
            ],
            "lenient": 0,
            "judge_response": None,
        }
    prompt = build_prompt("dspy", row, answer, "paper")
    abstained = False
    for attempt in range(2):
        response = api.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format=response_schema(row, "paper"),
        )
        verdict = json.loads(response.choices[0].message.content)
        try:
            claims, score, question_score = score_verdict(row, verdict, "paper")
            break
        except ValueError as error:
            if "regrading" not in str(error):
                raise
            if attempt == 0:
                continue  # one semantic retry on judge abstention
            # dev-only fallback: accept the claim verdicts, flag the abstention
            claims, score = score_claims(row, verdict["claims"])
            question_score = verdict.get("question_score")
            abstained = True
    return {
        "claims": claims,
        "lenient": score,
        "judge_abstained": abstained,
        "judge_question_score": question_score,
        "judge_response": {
            "id": response.id,
            "model": response.model,
            "finish_reason": response.choices[0].finish_reason,
            "usage": response.usage.model_dump(exclude_none=True) if response.usage else None,
        },
    }


def evaluate_variant(
    *,
    variant: str,
    note: str,
    rows: list[dict],
    corpus,
    tools: list,
    base_urls: list[str],
    api_key: str,
    master_seed: int,
    out_dir: Path,
    budgets: tuple[str, ...] = ("direct", "k5"),
    rollouts: int = 2,
    debug: bool = False,
    model: str | None = None,
    model_revision: str | None = None,
    sampling: dict | None = None,
) -> dict:
    """Evaluate one note variant on the dev slice; fully idempotent.

    ``model``/``model_revision``/``sampling`` override the harness default
    (local Qwen) — e.g. a hosted studier with ``base_urls=[None]``."""

    model_kwargs = (
        {"model": model, "model_revision": model_revision, "sampling": sampling}
        if model is not None else {}
    )

    report_path = out_dir / "report.json"
    if report_path.exists():
        return read_json(report_path)
    prefix = NOTE_PREFIX.format(library=corpus.display, note=note) if note else ""

    cases = [
        (row, budget, rollout)
        for budget in budgets
        for rollout in range(rollouts)
        for row in rows
    ]

    def run_case(index: int, case) -> None:
        row, budget, rollout = case
        path = out_dir / "episodes" / budget / f"r{rollout}" / f"{row['id']}.json"
        if _episode_ok(path):
            return
        episode = None
        for attempt in range(2):
            episode = run_episode(
                corpus=corpus,
                tools=tools,
                question={"id": row["id"], "question": prefix + row["question"]},
                condition="foldback-dev",
                budget=budget,
                rollout=rollout,
                seed=stable_seed(master_seed, "foldback-dev", variant, row["id"],
                                 budget, rollout, attempt),
                base_url=base_urls[index % len(base_urls)],
                max_iters=BUDGET_ITERS[budget],
                forced=False,
                debug=debug,
                **model_kwargs,
            )
            if episode["status"] in {"ok", "no_answer"}:
                break
        if episode["status"] not in {"ok", "no_answer"}:
            episode["status"] = "gave_up"
            log.warning("dev episode gave up: %s %s/r%d/%s (%s)",
                        variant, budget, rollout, row["id"], episode.get("error", ""))
        episode["variant"] = variant
        write_json(path, episode)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda item: run_case(*item), enumerate(cases)))

    api = OpenAI(api_key=api_key, timeout=JUDGE_TIMEOUT, max_retries=2)
    by_id = {row["id"]: row for row in rows}

    def grade_case(case) -> tuple[str, str, dict]:
        row, budget, rollout = case
        path = out_dir / "grades" / budget / f"r{rollout}" / f"{row['id']}.json"
        if path.exists():
            return budget, row["id"], read_json(path)
        episode = read_json(out_dir / "episodes" / budget / f"r{rollout}" / f"{row['id']}.json")
        grade = _grade_one(api, by_id[row["id"]], episode.get("answer", ""))
        grade.update(qid=row["id"], budget=budget, rollout=rollout, variant=variant,
                     episode_status=episode["status"],
                     gen_tokens=episode.get("gen_tokens", 0))
        write_json(path, grade)
        return budget, row["id"], grade

    with ThreadPoolExecutor(max_workers=4) as pool:
        graded = list(pool.map(grade_case, cases))

    report = {"variant": variant, "note_chars": len(note), "rollouts": rollouts,
              "budgets": {}}
    for budget in budgets:
        population = [grade for b, _, grade in graded if b == budget]
        per_question: dict[str, list] = {}
        for _, qid, grade in graded:
            if grade["budget"] == budget:
                per_question.setdefault(qid, []).append(grade["lenient"])
        report["budgets"][budget] = {
            "episodes": len(population),
            "mean_lenient": sum(g["lenient"] for g in population) / len(population),
            "mean_generated_tokens": sum(g["gen_tokens"] for g in population) / len(population),
            "per_question": {qid: sum(v) / len(v) for qid, v in sorted(per_question.items())},
        }
    write_json(report_path, report)
    for budget, result in report["budgets"].items():
        log.info("dev %s %s: mean_lenient=%.2f mean_tokens=%.0f",
                 variant, budget, result["mean_lenient"], result["mean_generated_tokens"])
    return report
