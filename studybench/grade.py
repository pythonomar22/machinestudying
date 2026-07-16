"""Grade a complete rollout with GPT-5.4 and the paper's lenient rubric."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
from functools import lru_cache
from pathlib import Path

from openai import AsyncOpenAI

from .artifacts import read_json, sha256_json, write_json
from .dataset import CORPORA, ROOT, load_questions, read_corpus_file, verify_corpus

JUDGE = "gpt-5.4"
OPENAI_URL = "https://api.openai.com/v1"
log = logging.getLogger("studybench.grade")
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}")

PROMPT = """You are grading one model answer for a private {library} expert QA benchmark.

## Scoring model
- The question gets one final continuous 0-100 score.
- Claims are only the internal rubric used to compute that question's score.
- Score each claim as:
  - `0` = wrong or missing
  - `1` = fully correct
- Do not give extra credit for material outside the rubric.
- If an answer is polished but misses essential content, score the missing claims low.
- Use the evidence spans and gold answer to resolve ambiguity.

## Output rules
- Score every rubric claim exactly once.
- `question_score` must equal the weighted sum of the claim scores.
- Set `needs_regrade` to `true` only if the rubric or evidence is genuinely insufficient to judge the answer confidently.
- Keep rationales concise and specific.

## Inputs
- Question ID: `{qid}`
- Label: `{topic}`
- Question: `{question}`
- Model answer:
{answer}

## Gold answer
{gold}

## Claim rubric
{rubric}

## Evidence spans
{evidence}

## Whole evidence files
{files}

Return JSON that matches the schema exactly."""


def artifact_path(root: Path, budget: str, rollout: int, qid: str) -> Path:
    return root / "episodes" / budget / f"r{rollout}" / f"{qid}.json"


def load_run(run_id: str, task: str) -> tuple[dict, dict[str, dict], dict[tuple, dict]]:
    if not RUN_ID.fullmatch(run_id):
        raise ValueError(f"invalid run ID: {run_id!r}")
    root = ROOT / "runs" / run_id / task
    manifest = read_json(root / "run.json")
    if (
        manifest.get("run_id") != run_id
        or manifest.get("task") != task
        or manifest.get("corpus_commit") != CORPORA[task].commit
        or manifest.get("dataset_sha256") != CORPORA[task].dataset_sha256
        or manifest.get("condition") not in {"baseline", "cheatsheet"}
    ):
        raise ValueError(f"invalid run manifest: {root / 'run.json'}")
    all_rows = {row["id"]: row for row in load_questions(task)}
    qids = manifest.get("question_ids")
    budgets = manifest.get("budgets")
    rollouts = manifest.get("rollouts")
    if (
        not isinstance(qids, list)
        or len(qids) != len(set(qids))
        or not set(qids).issubset(all_rows)
        or not isinstance(budgets, list)
        or not budgets
        or type(rollouts) is not int
        or rollouts < 1
    ):
        raise ValueError("invalid population in run manifest")

    expected = {
        artifact_path(root, budget, rollout, qid)
        for budget in budgets
        for rollout in range(rollouts)
        for qid in qids
    }
    episode_root = root / "episodes"
    actual = set(episode_root.rglob("*.json")) if episode_root.exists() else set()
    if actual != expected:
        raise ValueError(
            f"run population is incomplete: {len(expected - actual)} missing, "
            f"{len(actual - expected)} unexpected"
        )

    run_hash = sha256_json(manifest)
    episodes = {}
    for budget in budgets:
        for rollout in range(rollouts):
            for qid in qids:
                path = artifact_path(root, budget, rollout, qid)
                episode = read_json(path)
                identity = (task, qid, budget, rollout)
                if (
                    episode.get("run_config_sha256") != run_hash
                    or tuple(episode.get(field) for field in ("task", "qid", "budget", "rollout"))
                    != identity
                    or episode.get("condition") != manifest["condition"]
                    or episode.get("status") not in {"ok", "no_answer"}
                    or type(episode.get("gen_tokens")) is not int
                    or episode["gen_tokens"] < 0
                ):
                    raise ValueError(f"invalid episode: {path}")
                if episode["status"] == "ok" and not episode.get("answer", "").strip():
                    raise ValueError(f"successful episode has no answer: {path}")
                episodes[(budget, rollout, qid)] = episode
    return manifest, {qid: all_rows[qid] for qid in qids}, episodes


@lru_cache(maxsize=None)
def numbered_file(task: str, relative: str) -> str:
    source = read_corpus_file(CORPORA[task], relative)
    return "\n".join(
        f"{number:04d}: {line}" for number, line in enumerate(source.splitlines(), 1)
    )


def build_prompt(task: str, row: dict, answer: str) -> str:
    corpus = CORPORA[task]
    whole_files = [
        f"### {relative}\n{numbered_file(task, relative)}"
        for relative in dict.fromkeys(span["path"] for span in row["evidence"])
    ]
    return PROMPT.format(
        library=corpus.display,
        qid=row["id"],
        topic=row["topic"],
        question=row["question"],
        answer=answer,
        gold=row["gold_answer"],
        rubric=json.dumps(row["rubric"], ensure_ascii=False, indent=2),
        evidence=json.dumps(row["evidence"], ensure_ascii=False, indent=2),
        files="\n\n".join(whole_files),
    )


def response_schema(row: dict) -> dict:
    claim_ids = [claim["claim_id"] for claim in row["rubric"]]
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "studybench_grade",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "claims": {
                        "type": "array",
                        "minItems": len(claim_ids),
                        "maxItems": len(claim_ids),
                        "items": {
                            "type": "object",
                            "properties": {
                                "claim_id": {"type": "string", "enum": claim_ids},
                                "score": {"type": "integer", "enum": [0, 1]},
                                "rationale": {"type": "string"},
                            },
                            "required": ["claim_id", "score", "rationale"],
                            "additionalProperties": False,
                        },
                    },
                    "question_score": {"type": "number"},
                    "needs_regrade": {"type": "boolean"},
                },
                "required": ["claims", "question_score", "needs_regrade"],
                "additionalProperties": False,
            },
        },
    }


def score_verdict(row: dict, verdict: dict) -> tuple[list[dict], int]:
    claims = verdict.get("claims")
    if not isinstance(claims, list):
        raise ValueError("judge returned no claims")
    by_id = {claim.get("claim_id"): claim for claim in claims if isinstance(claim, dict)}
    rubric_ids = [claim["claim_id"] for claim in row["rubric"]]
    if len(claims) != len(by_id) or set(by_id) != set(rubric_ids):
        raise ValueError("judge claim IDs do not match the rubric")
    if any(
        type(by_id[claim_id].get("score")) is not int
        or by_id[claim_id]["score"] not in (0, 1)
        for claim_id in rubric_ids
    ):
        raise ValueError("judge returned a non-binary claim score")
    score = sum(
        claim["weight"] * by_id[claim["claim_id"]]["score"] for claim in row["rubric"]
    )
    if verdict.get("needs_regrade") is not False or verdict.get("question_score") != score:
        raise ValueError("judge requested regrading or returned inconsistent arithmetic")
    return [by_id[claim_id] for claim_id in rubric_ids], score


def expected_grade_manifest(manifest: dict) -> dict:
    return {
        "schema_version": 2,
        "run_config_sha256": sha256_json(manifest),
        "grader_contract_sha256": sha256_json(
            {
                "prompt": PROMPT,
                "response_schema": 1,
                "claim_scores": [0, 1],
                "score": "weighted_claim_sum",
            }
        ),
        "judge": JUDGE,
        "endpoint": OPENAI_URL,
        "evidence": "whole_files",
        "claim_scores": [0, 1],
        "score": "weighted_claim_sum",
    }


def validate_grade(row: dict, episode: dict, grade_record: dict) -> None:
    expected = {
        "episode_sha256": sha256_json(episode),
        "run_config_sha256": episode["run_config_sha256"],
        "task": episode["task"],
        "qid": episode["qid"],
        "budget": episode["budget"],
        "rollout": episode["rollout"],
        "seed": episode["seed"],
        "condition": episode["condition"],
        "model": episode["model"],
        "model_revision": episode["model_revision"],
        "judge": JUDGE,
        "gen_tokens": episode["gen_tokens"],
    }
    if any(grade_record.get(field) != value for field, value in expected.items()):
        raise ValueError(f"stale or mismatched grade: {episode['qid']}")
    _, score = score_verdict(
        row,
        {
            "claims": grade_record.get("claims"),
            "question_score": grade_record.get("judge_question_score"),
            "needs_regrade": grade_record.get("needs_regrade"),
        },
    )
    if grade_record.get("lenient") != score:
        raise ValueError(f"incorrect lenient score: {episode['qid']}")


async def grade(args) -> None:
    verify_corpus(CORPORA[args.task])
    manifest, rows, episodes = load_run(args.run_id, args.task)
    output = ROOT / "grades" / args.run_id / args.task
    grade_manifest = expected_grade_manifest(manifest)
    manifest_path = output / "grade.json"
    if manifest_path.exists() and read_json(manifest_path) != grade_manifest:
        raise ValueError(f"grade configuration changed: {manifest_path}")
    write_json(manifest_path, grade_manifest)

    pending = []
    for key, episode in episodes.items():
        path = artifact_path(output, *key)
        if path.exists():
            validate_grade(rows[key[2]], episode, read_json(path))
        else:
            pending.append((key, episode, path))
    log.info("%d/%d grades pending", len(pending), len(episodes))
    if not pending:
        return
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required")
    client = AsyncOpenAI(
        api_key=api_key, base_url=OPENAI_URL, timeout=600, max_retries=2
    )
    semaphore = asyncio.Semaphore(args.concurrency)

    async def one(key, episode, path):
        budget, rollout, qid = key
        row = rows[qid]
        if episode["status"] == "no_answer":
            claims = [
                {"claim_id": claim["claim_id"], "score": 0, "rationale": "No answer."}
                for claim in row["rubric"]
            ]
            score = 0
            question_score = 0
            needs_regrade = False
        else:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=JUDGE,
                    messages=[{"role": "user", "content": build_prompt(args.task, row, episode["answer"])}],
                    response_format=response_schema(row),
                )
            verdict = json.loads(response.choices[0].message.content)
            claims, score = score_verdict(row, verdict)
            question_score = verdict["question_score"]
            needs_regrade = verdict["needs_regrade"]
        grade_record = {
            "episode_sha256": sha256_json(episode),
            "run_config_sha256": episode["run_config_sha256"],
            "task": args.task,
            "qid": qid,
            "budget": budget,
            "rollout": rollout,
            "seed": episode["seed"],
            "condition": episode["condition"],
            "model": episode["model"],
            "model_revision": episode["model_revision"],
            "judge": JUDGE,
            "gen_tokens": episode["gen_tokens"],
            "claims": claims,
            "judge_question_score": question_score,
            "needs_regrade": needs_regrade,
            "lenient": score,
        }
        validate_grade(row, episode, grade_record)
        write_json(path, grade_record)
        log.info("%s/r%d/%s lenient=%d", budget, rollout, qid, score)
        if args.debug:
            log.debug("%s", json.dumps(claims, ensure_ascii=False, indent=2))

    try:
        await asyncio.gather(*(one(*item) for item in pending))
    finally:
        await client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task", required=True, choices=CORPORA)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        asyncio.run(grade(args))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"grading error: {error}") from error


if __name__ == "__main__":
    main()
