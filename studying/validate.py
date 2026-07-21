"""Fixed validation set: loading, per-round evaluation, paper-contract grading.

The validation set is our own StudyBench-pipeline replication
(data/smalldspy_validation.jsonl, provenance in data_collection/). It is
evaluated after every study round purely for reporting: validation results
never reach the quizmaster, the verifier, or the distiller.
"""

from __future__ import annotations

import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

from openai import OpenAI

from studybench.artifacts import read_json, stable_seed, write_json
from studybench.dataset import CORPORA, NOTE_PREFIX, ROOT
from studybench.grade import build_prompt, response_schema, score_verdict
from studybench.react import run_episode

VALIDATION_PATH = ROOT / "data" / "smalldspy_validation.jsonl"
JUDGE_MODEL = "gpt-5.4"
JUDGE_TIMEOUT = 600
BUDGET_ITERS = {"direct": 0, "k5": 5, "k20": 20}
log = logging.getLogger("studying.validate")


def validation_dataset_sha256() -> str:
    return hashlib.sha256(VALIDATION_PATH.read_bytes()).hexdigest()


def load_validation_questions() -> tuple[dict, ...]:
    corpus = CORPORA["smalldspy"]
    rows = tuple(
        json.loads(line)
        for line in VALIDATION_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    ids: set[str] = set()
    for row in rows:
        if set(row) != {"id", "topic", "question", "gold_answer", "rubric", "evidence"}:
            raise ValueError(f"unexpected fields in validation row: {row.get('id')}")
        if row["id"] in ids:
            raise ValueError(f"duplicate validation id: {row['id']}")
        ids.add(row["id"])
        if sum(claim["weight"] for claim in row["rubric"]) != 100:
            raise ValueError(f"validation rubric weights do not sum to 100: {row['id']}")
        span_ids = {span["span_id"] for span in row["evidence"]}
        for claim in row["rubric"]:
            if not set(claim["span_ids"]).issubset(span_ids):
                raise ValueError(f"unknown evidence span: {row['id']}/{claim['claim_id']}")
        for span in row["evidence"]:
            logical = PurePosixPath(span["path"])
            if not logical.parts or logical.parts[0] not in corpus.roots or ".." in logical.parts:
                raise ValueError(f"validation evidence escapes corpus roots: {span['path']}")
    if not rows:
        raise ValueError(f"empty validation set: {VALIDATION_PATH}")
    return rows


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
    prompt = build_prompt("smalldspy", row, answer, "paper")
    response = api.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format=response_schema(row, "paper"),
    )
    verdict = json.loads(response.choices[0].message.content)
    claims, score, question_score = score_verdict(row, verdict, "paper")
    return {
        "claims": claims,
        "lenient": score,
        "judge_question_score": question_score,
        "judge_response": {
            "id": response.id,
            "model": response.model,
            "finish_reason": response.choices[0].finish_reason,
            "usage": response.usage.model_dump(exclude_none=True) if response.usage else None,
        },
    }


def run_validation(
    *,
    rows: tuple[dict, ...],
    note: str,
    corpus,
    tools: list,
    base_urls: list[str],
    api_key: str,
    master_seed: int,
    round_no: int,
    out_dir: Path,
    budgets: tuple[str, ...],
    debug: bool,
) -> dict:
    """Evaluate the fixed validation set under the current note; report only."""

    report_path = out_dir / "report.json"
    if report_path.exists():
        return read_json(report_path)
    prefix = NOTE_PREFIX.format(library=corpus.display, note=note) if note else ""

    cases = [(row, budget) for budget in budgets for row in rows]

    def run_case(index: int, case) -> Path:
        row, budget = case
        path = out_dir / "episodes" / budget / f"{row['id']}.json"
        if _episode_ok(path):
            return path
        question = {"id": row["id"], "question": prefix + row["question"]}
        episode = None
        for attempt in range(2):
            episode = run_episode(
                corpus=corpus,
                tools=tools,
                question=question,
                condition="selfquiz-validation",
                budget=budget,
                rollout=0,
                seed=stable_seed(master_seed, "selfquiz-val", round_no, row["id"], budget, attempt),
                base_url=base_urls[index % len(base_urls)],
                max_iters=BUDGET_ITERS[budget],
                forced=False,
                debug=debug,
            )
            if episode["status"] in {"ok", "no_answer"}:
                break
        if episode["status"] not in {"ok", "no_answer"}:
            episode["status"] = "gave_up"
            log.warning("validation episode gave up: round %d %s/%s (%s)",
                        round_no, budget, row["id"], episode.get("error", ""))
        episode["round"] = round_no
        write_json(path, episode)
        return path

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda item: run_case(*item), enumerate(cases)))

    api = OpenAI(api_key=api_key, timeout=JUDGE_TIMEOUT, max_retries=2)
    by_id = {row["id"]: row for row in rows}

    def grade_case(case) -> tuple[str, str, dict]:
        row, budget = case
        path = out_dir / "grades" / budget / f"{row['id']}.json"
        episode = read_json(out_dir / "episodes" / budget / f"{row['id']}.json")
        if path.exists():
            return budget, row["id"], read_json(path)
        grade = _grade_one(api, by_id[row["id"]], episode.get("answer", ""))
        grade.update(qid=row["id"], budget=budget, round=round_no,
                     episode_status=episode["status"],
                     gen_tokens=episode.get("gen_tokens", 0))
        write_json(path, grade)
        return budget, row["id"], grade

    with ThreadPoolExecutor(max_workers=4) as pool:
        graded = list(pool.map(grade_case, cases))

    report = {"round": round_no, "note_chars": len(note), "budgets": {}}
    for budget in budgets:
        population = [grade for b, _, grade in graded if b == budget]
        report["budgets"][budget] = {
            "episodes": len(population),
            "mean_lenient": sum(g["lenient"] for g in population) / len(population),
            "mean_generated_tokens": sum(g["gen_tokens"] for g in population) / len(population),
            "per_question": {g["qid"]: g["lenient"] for g in population},
        }
    write_json(report_path, report)
    for budget, result in report["budgets"].items():
        log.info("round %d validation %s: mean_lenient=%.2f mean_tokens=%.0f",
                 round_no, budget, result["mean_lenient"], result["mean_generated_tokens"])
    return report
