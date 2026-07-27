"""Fold-back stage 0: a studier's direct + forced-k20 on the practice study slice.

Runs the selected hosted studier (bare questions, no note) at the two
ends of the compute axis on the 70-question study slice of the rev-3
practice set, then grades both budgets with the selected paper-contract
judge. The gap and its claim-level anatomy decide whether fold-back
mining proceeds; the k20f episodes double as the mining input (same
layout and seeds as run.py's attempts phase). The dev slice and the
StudyBench test set are untouched.

Usage:
    .venv-dspy/bin/python -m studying.foldback.stage0 --run-id RUN \
        --seed 20260715 --model gptmini --judge gpt [--smoke] [--debug]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import litellm
from openai import OpenAI

from studybench.artifacts import read_json, stable_seed, write_json
from studybench.dataset import CORPORA, ROOT
from studybench.grade import build_prompt, response_schema, score_verdict
from studybench.react import (
    CLAUDE_MODEL,
    CLAUDE_SAMPLING,
    GPT_MODEL,
    GPT_SAMPLING,
    TOOL_CONFIG,
    make_tools,
    run_episode,
)
from studybench.tools import RepoTools

from .data import load_practice_questions, practice_dataset_sha256, split_practice

MODELS = {
    "sonnet45": (CLAUDE_MODEL, CLAUDE_SAMPLING, "ANTHROPIC_API_KEY"),
    "gptmini": (GPT_MODEL, GPT_SAMPLING, "OPENAI_API_KEY"),
}
JUDGES = {
    "gpt": ("gpt-5.4", "OPENAI_API_KEY"),
    "sonnet": ("claude-sonnet-4-5", "ANTHROPIC_API_KEY"),
}
JUDGE_MAX_TOKENS = 8_192
JUDGE_TIMEOUT = 600
BUDGETS = {"direct": (0, False), "k20f": (20, True)}
log = logging.getLogger("studying.foldback.stage0")


def _source_state(smoke: bool) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    )
    if dirty and not smoke:
        raise SystemExit("full stage-0 runs require a clean committed source tree")
    return commit, dirty


def _episode_ok(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        episode = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return episode.get("status") in {"ok", "no_answer", "gave_up"}


def _grade_one(judge: str, row: dict, answer: str, api_key: str) -> dict:
    if not answer.strip():
        return {
            "claims": [
                {"claim_id": claim["claim_id"], "score": 0, "rationale": "No answer."}
                for claim in row["rubric"]
            ],
            "lenient": 0,
            "judge_response": None,
        }
    judge_model = JUDGES[judge][0]
    prompt = build_prompt("dspy", row, answer, "paper")
    if judge == "sonnet":
        response = litellm.completion(
            model=f"anthropic/{judge_model}",
            api_key=api_key,
            timeout=JUDGE_TIMEOUT,
            max_tokens=JUDGE_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
            response_format=response_schema(row, "paper"),
        )
    else:
        response = OpenAI(api_key=api_key, timeout=JUDGE_TIMEOUT, max_retries=2).chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            response_format=response_schema(row, "paper"),
        )
    if response.choices[0].finish_reason != "stop":
        raise RuntimeError(f"judge finish_reason={response.choices[0].finish_reason!r}")
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--judge", required=True, choices=sorted(JUDGES))
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if not args.run_id.replace("-", "").replace("_", "").isalnum():
        parser.error("--run-id must contain only letters, digits, '-' and '_'")
    model, sampling, model_key_env = MODELS[args.model]
    judge_model, judge_key_env = JUDGES[args.judge]
    for env in {model_key_env, judge_key_env}:
        if not os.environ.get(env):
            raise SystemExit(f"{env} is required for --model {args.model} / --judge {args.judge}")

    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(ROOT / "logs" / f"{args.run_id}-stage0.log"),
        ],
    )
    source_commit, source_dirty = _source_state(args.smoke)

    corpus = CORPORA["dspy"]
    repository = RepoTools(corpus)
    tools = make_tools(repository)
    rows = load_practice_questions()
    by_id = {row["id"]: row for row in rows}

    run_root = ROOT / "runs" / args.run_id / corpus.name
    study_root = run_root / "study"
    study_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": 1,
        "kind": "foldback-stage0",
        "run_id": args.run_id,
        "task": corpus.name,
        "smoke": args.smoke,
        "debug": args.debug,
        "source_commit": source_commit,
        "source_dirty": source_dirty,
        "corpus_commit": corpus.commit,
        "corpus_snapshot_sha256": repository.snapshot_sha256,
        "model": model,
        "model_revision": None,
        "harness": "dspy.ReAct",
        "sampling": sampling,
        "tools": {**TOOL_CONFIG, "corpus_roots": list(corpus.roots)},
        "master_seed": args.seed,
        "budgets": sorted(BUDGETS),
        "practice_dataset_sha256": practice_dataset_sha256(),
        "split": "stratified 70 study / 30 dev, seeded",
        "judge": {"model": judge_model, "contract": "paper",
                  "tier": "internal-study-signal", "max_tokens": JUDGE_MAX_TOKENS},
        "note": "bare questions; k20f episodes are reusable as mining attempts",
    }
    manifest_path = run_root / "stage0.json"
    volatile = {"source_commit", "source_dirty"}
    if manifest_path.exists():
        existing = read_json(manifest_path)
        if {k: v for k, v in existing.items() if k not in volatile} != {
            k: v for k, v in manifest.items() if k not in volatile
        }:
            raise SystemExit(f"stage-0 configuration changed; use a new --run-id: {manifest_path}")
    else:
        write_json(manifest_path, manifest)

    split_path = study_root / "split.json"
    if not split_path.exists():
        write_json(split_path, split_practice(rows, args.seed))
    split = read_json(split_path)
    study_ids = split["study_ids"][: 3 if args.smoke else None]
    log.info("study slice: %d questions", len(study_ids))

    def episode_path(budget: str, qid: str) -> Path:
        # k20f lands in attempts/ — the exact layout run.py's mining phase reads.
        return (study_root / "attempts" / f"{qid}.json" if budget == "k20f"
                else study_root / "direct" / f"{qid}.json")

    def run_case(index: int, case) -> None:
        budget, qid = case
        path = episode_path(budget, qid)
        if _episode_ok(path):
            return
        max_iters, forced = BUDGETS[budget]
        seed_tag = "foldback-attempt" if budget == "k20f" else "foldback-direct"
        episode = None
        for try_no in range(2):
            episode = run_episode(
                corpus=corpus,
                tools=tools,
                question={"id": qid, "question": by_id[qid]["question"]},
                condition="foldback-study",
                budget="k20f-attempt" if budget == "k20f" else "direct",
                rollout=0,
                seed=stable_seed(args.seed, seed_tag, qid, try_no),
                base_url=None,
                max_iters=2 if args.smoke and forced else max_iters,
                forced=forced,
                debug=args.debug,
                model=model,
                model_revision=None,
                sampling=sampling,
            )
            if episode["status"] in {"ok", "no_answer"}:
                break
        if episode["status"] not in {"ok", "no_answer"}:
            episode["status"] = "gave_up"
            log.warning("%s gave up: %s (%s)", budget, qid, episode.get("error", "")[:200])
        write_json(path, episode)
        log.info("%s/%s status=%s iters=%d gen_tokens=%d",
                 budget, qid, episode["status"], episode["react_iterations"],
                 episode["gen_tokens"])

    cases = [(budget, qid) for budget in sorted(BUDGETS) for qid in study_ids]
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        list(pool.map(lambda item: run_case(*item), enumerate(cases)))
    for budget in sorted(BUDGETS):
        statuses = [read_json(episode_path(budget, qid))["status"] for qid in study_ids]
        log.info("%s: %d ok, %d no_answer, %d gave_up", budget,
                 statuses.count("ok"), statuses.count("no_answer"),
                 statuses.count("gave_up"))

    judge_api_key = os.environ[judge_key_env]

    def grade_case(case) -> tuple[str, str, dict]:
        budget, qid = case
        path = study_root / "grades" / budget / f"{qid}.json"
        if path.exists():
            return budget, qid, read_json(path)
        episode = read_json(episode_path(budget, qid))
        grade = _grade_one(args.judge, by_id[qid], episode.get("answer", ""), judge_api_key)
        grade.update(qid=qid, budget=budget, judge=judge_model,
                     episode_status=episode["status"],
                     gen_tokens=episode.get("gen_tokens", 0))
        write_json(path, grade)
        log.info("grade %s/%s lenient=%s", budget, qid, grade["lenient"])
        return budget, qid, grade

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        graded = list(pool.map(grade_case, cases))

    report = {"kind": "foldback-stage0", "model": model, "judge": judge_model,
              "questions": len(study_ids), "budgets": {}}
    for budget in sorted(BUDGETS):
        population = [grade for b, _, grade in graded if b == budget]
        report["budgets"][budget] = {
            "episodes": len(population),
            "mean_lenient": sum(g["lenient"] for g in population) / len(population),
            "mean_generated_tokens": sum(g["gen_tokens"] for g in population) / len(population),
            "per_question": {qid: grade["lenient"] for b, qid, grade in graded if b == budget},
        }
    write_json(study_root / "report.json", report)
    for budget, result in report["budgets"].items():
        log.info("stage0 %s: mean_lenient=%.2f mean_gen_tokens=%.0f",
                 budget, result["mean_lenient"], result["mean_generated_tokens"])


if __name__ == "__main__":
    main()
