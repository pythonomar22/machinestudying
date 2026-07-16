"""Grade a complete rollout with the paper's lenient rubric."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from statistics import fmean

from openai import AsyncOpenAI, OpenAIError

from .artifacts import read_json, sha256_json, sha256_text, stable_seed, write_json
from .dataset import (
    CORPORA,
    NOTE_PREFIX,
    ROOT,
    load_questions,
    read_corpus_file,
    verify_corpus,
)

DEFAULT_JUDGE = "gpt-5.4"
DEFAULT_ENDPOINT = "https://api.openai.com/v1"
LOCAL_JUDGE = "Qwen/Qwen3.5-9B"
LOCAL_JUDGE_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
BUDGETS = ("direct", "k5", "k20", "k20f")
log = logging.getLogger("studybench.grade")
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}")

JUDGES = {
    "gpt": {
        "model": DEFAULT_JUDGE,
        "revision": None,
        "endpoint": DEFAULT_ENDPOINT,
        "key_env": "OPENAI_API_KEY",
        "grade_id": "gpt-5-4",
        "tier": "paper",
        "temperature": None,
        "seed": None,
        "max_tokens": None,
        "thinking_token_budget": None,
        "thinking": "provider-default",
        "contract": "paper",
        "timeout_seconds": 600,
        "runtime": None,
    },
    "fugu": {
        "model": "fugu",
        "revision": None,
        "endpoint": "https://api.sakana.ai/v1",
        "key_env": "SAKANA_API_KEY",
        "grade_id": "fugu",
        "tier": "diagnostic-external-proxy",
        "temperature": None,
        "seed": None,
        "max_tokens": None,
        "thinking_token_budget": None,
        "thinking": "provider-default",
        "contract": "paper",
        "timeout_seconds": 600,
        "runtime": None,
    },
    "local": {
        "model": LOCAL_JUDGE,
        "revision": LOCAL_JUDGE_REVISION,
        "endpoint": "local-vllm-0.24.0-tp2-xgrammar-compact",
        "key_env": "OPENAI_API_KEY",
        "grade_id": "qwen35-9b-thinking-local",
        "tier": "diagnostic-local-proxy",
        "temperature": 0,
        "seed": 0,
        "max_tokens": 65536,
        "thinking_token_budget": 4000,
        "thinking": "enabled",
        "contract": "local",
        "timeout_seconds": 300,
        "runtime": {
            "vllm": "0.24.0",
            "model_runner": "v1",
            "tensor_parallel": 2,
            "max_model_len": 262144,
            "max_num_seqs": 1,
            "reasoning_parser": "qwen3",
            "structured_outputs": {
                "backend": "xgrammar",
                "disable_any_whitespace": True,
                "enable_in_reasoning": False,
            },
        },
    },
    "local10k": {
        "model": LOCAL_JUDGE,
        "revision": LOCAL_JUDGE_REVISION,
        "endpoint": "local-vllm-0.24.0-tp2-xgrammar-compact",
        "key_env": "OPENAI_API_KEY",
        "grade_id": "qwen35-9b-thinking10k-local",
        "tier": "diagnostic-local-proxy",
        "temperature": 0,
        "seed": 0,
        "max_tokens": 65536,
        "thinking_token_budget": 10000,
        "thinking": "enabled",
        "contract": "local-forced",
        "timeout_seconds": 300,
        "runtime": {
            "vllm": "0.24.0",
            "model_runner": "v1",
            "tensor_parallel": 2,
            "max_model_len": 262144,
            "max_num_seqs": 1,
            "reasoning_parser": "qwen3",
            "structured_outputs": {
                "backend": "xgrammar",
                "disable_any_whitespace": True,
                "enable_in_reasoning": False,
            },
        },
    },
}

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
{output_rules}
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

LOCAL_FORCED_PROMPT = PROMPT.replace(
    "- Set `needs_regrade` to `true` only if the rubric or evidence is genuinely "
    "insufficient to judge the answer confidently.",
    "- The rubric and evidence bundle are complete. Always issue the best-supported "
    "binary verdict and set `needs_regrade` to `false`; do not abstain because an "
    "answer is ambiguous, contradictory, incomplete, or incorrect.",
)

PAPER_OUTPUT_RULES = """- Score every rubric claim exactly once.
- `question_score` must equal the weighted sum of the claim scores."""

LOCAL_OUTPUT_RULES = """- Return a `claims` object keyed by the rubric claim IDs.
- Include every rubric claim ID exactly once and no other keys.
- For each claim ID, return its binary `score` and concise `rationale`.
- Do not output `question_score`; the harness computes the weighted sum."""

LOCAL_CONTRACTS = {"local", "local-forced"}


def contract_prompt(contract: str) -> str:
    if contract in {"paper", "local"}:
        return PROMPT
    if contract == "local-forced":
        return LOCAL_FORCED_PROMPT
    raise ValueError(f"unknown grader contract: {contract}")


def contract_output_rules(contract: str) -> str:
    if contract == "paper":
        return PAPER_OUTPUT_RULES
    if contract in LOCAL_CONTRACTS:
        return LOCAL_OUTPUT_RULES
    raise ValueError(f"unknown grader contract: {contract}")


def artifact_path(root: Path, budget: str, rollout: int, qid: str) -> Path:
    return root / "episodes" / budget / f"r{rollout}" / f"{qid}.json"


def resolve_run_path(raw_path: str) -> list[tuple[str, str]]:
    root = ROOT.resolve()
    runs_root = (root / "runs").resolve()
    given = Path(raw_path)
    given = (given if given.is_absolute() else root / given).resolve()
    if (given / "run.json").is_file():
        task_dirs = [given]
    else:
        task_dirs = sorted(path.parent for path in given.glob("*/run.json"))
    if not task_dirs:
        raise ValueError("run path must be a task run or contain task runs")

    targets = []
    for task_dir in task_dirs:
        try:
            run_id, task = task_dir.relative_to(runs_root).parts
        except ValueError as error:
            raise ValueError(f"run path is outside {runs_root}: {task_dir}") from error
        if not SAFE_ID.fullmatch(run_id) or task not in CORPORA:
            raise ValueError(f"invalid run path: {task_dir}")
        targets.append((run_id, task))
    return targets


def load_run(run_id: str, task: str) -> tuple[dict, dict[str, dict], dict[tuple, dict]]:
    if not SAFE_ID.fullmatch(run_id):
        raise ValueError(f"invalid run ID: {run_id!r}")
    root = ROOT / "runs" / run_id / task
    manifest = read_json(root / "run.json")
    if (
        manifest.get("schema_version") not in {1, 2}
        or manifest.get("run_id") != run_id
        or manifest.get("task") != task
        or manifest.get("corpus_commit") != CORPORA[task].commit
        or manifest.get("dataset_sha256") != CORPORA[task].dataset_sha256
        or manifest.get("condition") not in {"baseline", "cheatsheet"}
    ):
        raise ValueError(f"invalid run manifest: {root / 'run.json'}")
    corpus = CORPORA[task]
    recorded_snapshot = (
        manifest.get("corpus_file_count"),
        manifest.get("corpus_snapshot_sha256"),
    )
    expected_snapshot = (corpus.file_count, corpus.snapshot_sha256)
    if manifest["schema_version"] == 1:
        valid_snapshot = recorded_snapshot == (None, None)
    elif corpus.file_count is None:
        count, snapshot = recorded_snapshot
        valid_snapshot = (
            type(count) is int
            and count > 0
            and isinstance(snapshot, str)
            and re.fullmatch(r"[0-9a-f]{64}", snapshot) is not None
        )
    else:
        valid_snapshot = recorded_snapshot == expected_snapshot
    if not valid_snapshot:
        raise ValueError(f"invalid corpus snapshot in run manifest: {root / 'run.json'}")
    if manifest["schema_version"] == 2 and manifest.get("corpus_display") != corpus.display:
        raise ValueError(f"invalid corpus display in run manifest: {root / 'run.json'}")
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
    note_prefix = ""
    if manifest["condition"] == "cheatsheet":
        note_path = root / "cheatsheet.md"
        if not note_path.is_file():
            raise ValueError(f"missing cheatsheet: {note_path}")
        note = note_path.read_text(encoding="utf-8")
        note_prefix = NOTE_PREFIX.format(
            library=CORPORA[task].display,
            note=note,
        )
        if (
            manifest.get("note_sha256") != sha256_text(note)
            or manifest.get("note_prefix_sha256") != sha256_text(note_prefix)
        ):
            raise ValueError(f"cheatsheet does not match run manifest: {note_path}")
        if manifest["schema_version"] == 2:
            study_path = root / "study.json"
            study = read_json(study_path)
            expected_iterations = 2 if manifest.get("smoke") else 50
            expected_study = {
                "iterations": expected_iterations,
                "seed": stable_seed(manifest["master_seed"], "study", task),
                "episode_sha256": sha256_json(study),
                "generated_tokens": study.get("gen_tokens"),
                "repository_tool_calls": study.get("repository_tool_calls"),
                "finish_catches": study.get("finish_catches"),
            }
            if (
                manifest.get("study") != expected_study
                or study.get("task") != task
                or study.get("qid") != "cheatsheet"
                or study.get("condition") != "cheatsheet"
                or study.get("budget") != "study"
                or study.get("rollout") != 0
                or study.get("seed") != expected_study["seed"]
                or study.get("model") != manifest["model"]
                or study.get("model_revision") != manifest["model_revision"]
                or study.get("status") != "ok"
                or study.get("react_iterations") != expected_iterations
                or study.get("answer") != note
            ):
                raise ValueError(f"invalid cheatsheet study artifact: {study_path}")
    elif manifest["schema_version"] == 2 and any(
        manifest.get(field) is not None
        for field in ("note_sha256", "note_prefix_sha256", "study")
    ):
        raise ValueError(f"baseline run has cheatsheet metadata: {root / 'run.json'}")
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
                    or episode.get("seed")
                    != stable_seed(
                        manifest["master_seed"],
                        "eval",
                        task,
                        qid,
                        budget,
                        rollout,
                    )
                    or episode.get("model") != manifest["model"]
                    or episode.get("model_revision") != manifest["model_revision"]
                    or episode.get("harness") != manifest["harness"]
                    or episode.get("question_sha256")
                    != sha256_text(note_prefix + all_rows[qid]["question"])
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


def build_prompt(task: str, row: dict, answer: str, contract: str = "paper") -> str:
    corpus = CORPORA[task]
    whole_files = [
        f"### {relative}\n{numbered_file(task, relative)}"
        for relative in dict.fromkeys(span["path"] for span in row["evidence"])
    ]
    return contract_prompt(contract).format(
        library=corpus.display,
        output_rules=contract_output_rules(contract),
        qid=row["id"],
        topic=row["topic"],
        question=row["question"],
        answer=answer,
        gold=row["gold_answer"],
        rubric=json.dumps(row["rubric"], ensure_ascii=False, indent=2),
        evidence=json.dumps(row["evidence"], ensure_ascii=False, indent=2),
        files="\n\n".join(whole_files),
    )


def response_schema(row: dict, contract: str = "paper") -> dict:
    claim_ids = [claim["claim_id"] for claim in row["rubric"]]
    if contract == "paper":
        claims = {
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
        }
        properties = {
            "claims": claims,
            "question_score": {"type": "number"},
            "needs_regrade": {"type": "boolean"},
        }
        required = ["claims", "question_score", "needs_regrade"]
    elif contract in LOCAL_CONTRACTS:
        claim = {
            "type": "object",
            "properties": {
                "score": {"type": "integer", "enum": [0, 1]},
                "rationale": {"type": "string"},
            },
            "required": ["score", "rationale"],
            "additionalProperties": False,
        }
        properties = {
            "claims": {
                "type": "object",
                "properties": {claim_id: claim for claim_id in claim_ids},
                "required": claim_ids,
                "additionalProperties": False,
            },
            "needs_regrade": (
                {"type": "boolean", "enum": [False]}
                if contract == "local-forced"
                else {"type": "boolean"}
            ),
        }
        required = ["claims", "needs_regrade"]
    else:
        raise ValueError(f"unknown grader contract: {contract}")
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "studybench_grade",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def score_claims(row: dict, claims: object) -> tuple[list[dict], int]:
    rubric_ids = [claim["claim_id"] for claim in row["rubric"]]
    if not isinstance(claims, list) or len(claims) != len(rubric_ids):
        raise ValueError("judge returned no claims")
    by_id = {}
    for claim in claims:
        if not isinstance(claim, dict) or set(claim) != {
            "claim_id",
            "score",
            "rationale",
        }:
            raise ValueError("judge claim has missing or unexpected fields")
        claim_id = claim["claim_id"]
        if claim_id in by_id:
            raise ValueError("judge returned duplicate claim IDs")
        by_id[claim_id] = claim
    if set(by_id) != set(rubric_ids):
        raise ValueError("judge claim IDs do not match the rubric")
    if any(
        type(by_id[claim_id].get("score")) is not int
        or by_id[claim_id]["score"] not in (0, 1)
        for claim_id in rubric_ids
    ):
        raise ValueError("judge returned a non-binary claim score")
    if any(
        not isinstance(by_id[claim_id].get("rationale"), str)
        for claim_id in rubric_ids
    ):
        raise ValueError("judge returned a non-string claim rationale")
    score = sum(
        claim["weight"] * by_id[claim["claim_id"]]["score"] for claim in row["rubric"]
    )
    return [by_id[claim_id] for claim_id in rubric_ids], score


def score_verdict(
    row: dict, verdict: dict, contract: str = "paper"
) -> tuple[list[dict], int, int | float | None]:
    if contract == "paper":
        expected_fields = {"claims", "question_score", "needs_regrade"}
    elif contract in LOCAL_CONTRACTS:
        expected_fields = {"claims", "needs_regrade"}
    else:
        raise ValueError(f"unknown grader contract: {contract}")
    if not isinstance(verdict, dict) or set(verdict) != expected_fields:
        raise ValueError("judge verdict has missing or unexpected fields")
    raw_claims = verdict["claims"]
    if contract in LOCAL_CONTRACTS:
        if not isinstance(raw_claims, dict):
            raise ValueError("judge returned no claims")
        normalized = []
        for claim_id, claim in raw_claims.items():
            if not isinstance(claim, dict) or set(claim) != {"score", "rationale"}:
                raise ValueError("judge claim has missing or unexpected fields")
            normalized.append({"claim_id": claim_id, **claim})
        raw_claims = normalized
    claims, score = score_claims(row, raw_claims)
    if verdict.get("needs_regrade") is not False:
        raise ValueError("judge requested regrading")
    question_score = verdict.get("question_score")
    if contract == "paper" and (
        isinstance(question_score, bool)
        or not isinstance(question_score, (int, float))
        or question_score != score
    ):
        raise ValueError("judge returned inconsistent arithmetic")
    return claims, score, question_score


def grader_contract_sha256(contract: str) -> str:
    return sha256_json(
        {
            "prompt": contract_prompt(contract),
            "output_rules": contract_output_rules(contract),
            "response_schema": f"{contract}-1",
            "claim_scores": [0, 1],
            "score": "weighted_claim_sum",
        }
    )


def expected_grade_manifest(manifest: dict, judge: dict) -> dict:
    return {
        "schema_version": 4,
        "run_config_sha256": sha256_json(manifest),
        "grader_contract_sha256": grader_contract_sha256(judge["contract"]),
        "judge": judge,
        "evidence": "whole_files",
        "claim_scores": [0, 1],
        "score": "weighted_claim_sum",
    }


def validate_grade(
    row: dict, episode: dict, grade_record: dict, grade_manifest: dict
) -> None:
    judge = grade_manifest["judge"]
    profile = JUDGES[judge["provider"]]
    expected = {
        "episode_sha256": sha256_json(episode),
        "run_config_sha256": episode["run_config_sha256"],
        "grade_config_sha256": sha256_json(grade_manifest),
        "task": episode["task"],
        "qid": episode["qid"],
        "budget": episode["budget"],
        "rollout": episode["rollout"],
        "seed": episode["seed"],
        "condition": episode["condition"],
        "model": episode["model"],
        "model_revision": episode["model_revision"],
        "judge": judge["model"],
        "judge_revision": judge["revision"],
        "judge_server_slot": server_slot(episode, judge["replicas"]),
        "judge_schema_sha256": sha256_json(
            response_schema(row, judge["contract"])
        ),
        "gen_tokens": episode["gen_tokens"],
    }
    result_fields = {
        "judge_attempts",
        "judge_prompt_sha256",
        "judge_response",
        "claims",
        "judge_question_score",
        "needs_regrade",
        "lenient",
    }
    if set(grade_record) != set(expected) | result_fields or any(
        grade_record.get(field) != value for field, value in expected.items()
    ):
        raise ValueError(f"stale or mismatched grade: {episode['qid']}")

    if episode["status"] == "no_answer":
        raw_response = None
        prompt_sha256 = None
        attempts = 0
        claims = [
            {"claim_id": claim["claim_id"], "score": 0, "rationale": "No answer."}
            for claim in row["rubric"]
        ]
        score = 0
        question_score = 0 if judge["contract"] == "paper" else None
        needs_regrade = False
    else:
        raw_response = grade_record["judge_response"]
        if not isinstance(raw_response, dict) or set(raw_response) != {
            "id",
            "created",
            "model",
            "system_fingerprint",
            "finish_reason",
            "usage",
            "content",
            "reasoning",
        }:
            raise ValueError(f"invalid raw judge response: {episode['qid']}")
        if raw_response["reasoning"] is not None and not isinstance(
            raw_response["reasoning"], str
        ):
            raise ValueError(f"invalid judge reasoning: {episode['qid']}")
        content = validate_response(raw_response, profile)
        try:
            verdict = json.loads(content)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid saved judge JSON: {episode['qid']}") from error
        claims, score, question_score = score_verdict(
            row, verdict, judge["contract"]
        )
        needs_regrade = verdict["needs_regrade"]
        prompt_sha256 = sha256_text(
            build_prompt(
                episode["task"],
                row,
                episode["answer"],
                judge["contract"],
            )
        )
        attempts = 1

    if (
        grade_record["judge_response"] != raw_response
        or grade_record["judge_prompt_sha256"] != prompt_sha256
        or grade_record["judge_attempts"] != attempts
        or grade_record["claims"] != claims
        or grade_record["judge_question_score"] != question_score
        or grade_record["needs_regrade"] != needs_regrade
        or grade_record["lenient"] != score
    ):
        raise ValueError(f"incorrect lenient score: {episode['qid']}")


def judge_config(provider: str, grade_id: str, replicas: int) -> dict:
    if provider not in JUDGES or replicas < 1:
        raise ValueError("invalid judge configuration")
    profile = JUDGES[provider]
    runtime = profile["runtime"]
    if profile["contract"] in LOCAL_CONTRACTS:
        runtime = {
            **runtime,
            "dependency_lock_sha256": sha256_text(
                (ROOT / "scripts" / "vllm-requirements.lock").read_text(
                    encoding="utf-8"
                )
            ),
        }
    return {
        "grade_id": grade_id,
        "provider": provider,
        "tier": profile["tier"],
        "model": profile["model"],
        "revision": profile["revision"],
        "endpoint": profile["endpoint"],
        "replicas": replicas,
        "sampling": {
            "temperature": profile["temperature"],
            "seed": profile["seed"],
            "max_tokens": profile["max_tokens"],
            "thinking_token_budget": profile["thinking_token_budget"],
        },
        "thinking": profile["thinking"],
        "contract": profile["contract"],
        "timeout_seconds": profile["timeout_seconds"],
        "semantic_retries": 0,
        "runtime": runtime,
    }


def validate_grade_manifest(
    manifest: dict, grade_manifest: dict, grade_id: str
) -> None:
    judge = grade_manifest.get("judge")
    if not isinstance(judge, dict) or judge.get("grade_id") != grade_id:
        raise ValueError("invalid grade manifest judge")
    replicas = judge.get("replicas")
    if type(replicas) is not int or replicas < 1:
        raise ValueError("invalid grade manifest replicas")
    expected_judge = judge_config(judge.get("provider"), grade_id, replicas)
    if judge != expected_judge or grade_manifest != expected_grade_manifest(
        manifest, expected_judge
    ):
        raise ValueError("grade manifest does not match its judge configuration")


def response_record(response) -> dict:
    message = response.choices[0].message
    return {
        "id": getattr(response, "id", None),
        "created": getattr(response, "created", None),
        "model": getattr(response, "model", None),
        "system_fingerprint": getattr(response, "system_fingerprint", None),
        "finish_reason": response.choices[0].finish_reason,
        "usage": response.usage.model_dump(exclude_none=True) if response.usage else None,
        "content": message.content,
        "reasoning": getattr(message, "reasoning", None)
        or getattr(message, "reasoning_content", None),
    }


def validate_response(raw: dict, profile: dict) -> str:
    if raw["finish_reason"] != "stop":
        raise ValueError(f"judge finish_reason={raw['finish_reason']!r}")
    if profile["contract"] in LOCAL_CONTRACTS and raw["model"] != profile["model"]:
        raise ValueError(f"local judge returned model {raw['model']!r}")
    usage = raw["usage"]
    if profile["contract"] in LOCAL_CONTRACTS:
        if not isinstance(usage, dict) or any(
            type(usage.get(field)) is not int or usage[field] < 0
            for field in ("prompt_tokens", "completion_tokens", "total_tokens")
        ):
            raise ValueError("judge returned incomplete usage")
        if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
            raise ValueError("judge usage total is inconsistent")
        if usage["completion_tokens"] >= profile["max_tokens"]:
            raise ValueError("judge exhausted its completion-token budget")
    content = raw["content"]
    if not isinstance(content, str) or not content.strip():
        raise ValueError("judge returned no final content")
    return content


def server_slot(episode: dict, replicas: int) -> int:
    identity = [
        episode["task"],
        episode["qid"],
        episode["budget"],
        episode["rollout"],
    ]
    return int(sha256_json(identity), 16) % replicas


def weighted_auc(points: list[tuple[float, float]]) -> float:
    """Appendix C: 3k anchor and best-so-far accuracy over generated tokens."""

    ordered = sorted(points)
    if len(ordered) != 4 or any(tokens <= 0 for tokens, _ in ordered):
        raise ValueError("expertise requires four positive-token budget points")
    area = best = 0.0
    for index, (tokens, accuracy) in enumerate(ordered):
        best = max(best, accuracy)
        weight = min(3000 / tokens, 1.0)
        next_weight = (
            min(3000 / ordered[index + 1][0], 1.0)
            if index + 1 < len(ordered)
            else 0.0
        )
        area += (weight - next_weight) * best
    return area


def write_report(
    output: Path,
    manifest: dict,
    rows: dict[str, dict],
    episodes: dict[tuple, dict],
    grade_manifest: dict,
) -> None:
    expected_paths = {artifact_path(output, *key) for key in episodes}
    episode_root = output / "episodes"
    actual_paths = (
        set(episode_root.rglob("*.json")) if episode_root.exists() else set()
    )
    if actual_paths != expected_paths:
        raise ValueError(
            f"grade population is incomplete: {len(expected_paths - actual_paths)} "
            f"missing, {len(actual_paths - expected_paths)} unexpected"
        )

    grades = {}
    for key, episode in episodes.items():
        path = artifact_path(output, *key)
        grade_record = read_json(path)
        validate_grade(rows[key[2]], episode, grade_record, grade_manifest)
        grades[key] = grade_record

    budgets = {}
    points = []
    for budget in manifest["budgets"]:
        population = [grade for key, grade in grades.items() if key[0] == budget]
        expected = len(manifest["question_ids"]) * manifest["rollouts"]
        if len(population) != expected:
            raise ValueError(f"incomplete grade population for {budget}")
        mean_tokens = fmean(grade["gen_tokens"] for grade in population)
        mean_score = fmean(grade["lenient"] for grade in population)
        budgets[budget] = {
            "episodes": len(population),
            "mean_lenient": mean_score,
            "mean_generated_tokens": mean_tokens,
        }
        points.append((mean_tokens, mean_score))

    expertise = weighted_auc(points) if manifest["budgets"] == list(BUDGETS) else None
    report = {
        "schema_version": 1,
        "run_id": manifest["run_id"],
        "task": manifest["task"],
        "condition": manifest["condition"],
        "grade_config_sha256": sha256_json(grade_manifest),
        "judge": grade_manifest["judge"],
        "episodes": len(grades),
        "budgets": budgets,
        "expertise": expertise,
    }
    write_json(output / "report.json", report)
    for budget, result in budgets.items():
        log.info(
            "%s mean_lenient=%.2f mean_generated_tokens=%.1f",
            budget,
            result["mean_lenient"],
            result["mean_generated_tokens"],
        )
    if expertise is not None:
        log.info("expertise=%.4f", expertise)


async def grade_task(args, profile: dict, base_urls: list[str], run_id: str, task: str) -> None:
    verify_corpus(CORPORA[task])
    manifest, rows, episodes = load_run(run_id, task)
    judge = judge_config(args.judge, args.grade_id, len(base_urls))
    output = ROOT / "grades" / run_id / args.grade_id / task
    grade_manifest = expected_grade_manifest(manifest, judge)
    manifest_path = output / "grade.json"
    if manifest_path.exists() and read_json(manifest_path) != grade_manifest:
        raise ValueError(f"grade configuration changed: {manifest_path}")
    (output / "report.json").unlink(missing_ok=True)

    expected_paths = {artifact_path(output, *key) for key in episodes}
    episode_root = output / "episodes"
    actual_paths = (
        set(episode_root.rglob("*.json")) if episode_root.exists() else set()
    )
    unexpected = actual_paths - expected_paths
    if unexpected:
        raise ValueError(f"grade population has {len(unexpected)} unexpected files")

    pending = []
    for key, episode in episodes.items():
        path = artifact_path(output, *key)
        if path.exists():
            validate_grade(rows[key[2]], episode, read_json(path), grade_manifest)
        else:
            pending.append((key, episode, path))
    log.info("%d/%d grades pending", len(pending), len(episodes))
    if not pending:
        if not manifest_path.is_file():
            raise ValueError(f"missing grade manifest: {manifest_path}")
        write_report(output, manifest, rows, episodes, grade_manifest)
        return

    selected = pending[: args.limit or None]
    api_key = os.environ.get(profile["key_env"])
    if not api_key:
        raise ValueError(f"{profile['key_env']} is required for {args.judge} grading")
    clients = [
        AsyncOpenAI(
            api_key=api_key,
            base_url=url,
            timeout=profile["timeout_seconds"],
            max_retries=0,
        )
        for url in base_urls
    ]
    if args.concurrency % len(clients):
        raise ValueError("--concurrency must divide evenly across judge replicas")
    per_server_concurrency = args.concurrency // len(clients)
    semaphores = [
        asyncio.Semaphore(per_server_concurrency) for _ in clients
    ]
    write_json(manifest_path, grade_manifest)

    async def one(key, episode, path):
        budget, rollout, qid = key
        row = rows[qid]
        slot = server_slot(episode, len(clients))
        raw_response = None
        prompt_sha256 = None
        if episode["status"] == "no_answer":
            claims = [
                {"claim_id": claim["claim_id"], "score": 0, "rationale": "No answer."}
                for claim in row["rubric"]
            ]
            score = 0
            question_score = 0 if profile["contract"] == "paper" else None
            needs_regrade = False
        else:
            prompt = build_prompt(
                task,
                row,
                episode["answer"],
                profile["contract"],
            )
            prompt_sha256 = sha256_text(prompt)
            request = {
                "model": profile["model"],
                "messages": [{"role": "user", "content": prompt}],
                "response_format": response_schema(row, profile["contract"]),
            }
            if profile["temperature"] is not None:
                request["temperature"] = profile["temperature"]
            if profile["max_tokens"] is not None:
                request["max_tokens"] = profile["max_tokens"]
            extra_body = {}
            if profile["thinking"] != "provider-default":
                extra_body["chat_template_kwargs"] = {
                    "enable_thinking": profile["thinking"] == "enabled"
                }
            if profile["thinking_token_budget"] is not None:
                extra_body["thinking_token_budget"] = profile["thinking_token_budget"]
            if extra_body:
                request["extra_body"] = extra_body

            if profile["seed"] is not None:
                request["seed"] = profile["seed"]
            async with semaphores[slot]:
                response = await clients[slot].chat.completions.create(**request)
            raw_response = response_record(response)
            try:
                content = validate_response(raw_response, profile)
                verdict = json.loads(content)
                claims, score, question_score = score_verdict(
                    row, verdict, profile["contract"]
                )
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(
                    f"invalid judge verdict: {budget}/r{rollout}/{qid}: {error}"
                ) from error
            needs_regrade = verdict["needs_regrade"]

        grade_record = {
            "episode_sha256": sha256_json(episode),
            "run_config_sha256": episode["run_config_sha256"],
            "grade_config_sha256": sha256_json(grade_manifest),
            "task": task,
            "qid": qid,
            "budget": budget,
            "rollout": rollout,
            "seed": episode["seed"],
            "condition": episode["condition"],
            "model": episode["model"],
            "model_revision": episode["model_revision"],
            "judge": profile["model"],
            "judge_revision": profile["revision"],
            "judge_server_slot": slot,
            "judge_attempts": 1 if raw_response else 0,
            "judge_prompt_sha256": prompt_sha256,
            "judge_schema_sha256": sha256_json(
                response_schema(row, profile["contract"])
            ),
            "judge_response": raw_response,
            "gen_tokens": episode["gen_tokens"],
            "claims": claims,
            "judge_question_score": question_score,
            "needs_regrade": needs_regrade,
            "lenient": score,
        }
        validate_grade(row, episode, grade_record, grade_manifest)
        write_json(path, grade_record)
        log.info("%s/r%d/%s lenient=%d", budget, rollout, qid, score)
        if args.debug:
            log.debug("%s", json.dumps(raw_response, ensure_ascii=False, indent=2))
        return path

    try:
        results = await asyncio.gather(
            *(one(*item) for item in selected),
            return_exceptions=True,
        )
    finally:
        await asyncio.gather(*(client.close() for client in clients))
    failures = [result for result in results if isinstance(result, BaseException)]
    if failures:
        raise ValueError(
            f"grading failed for {len(failures)}/{len(selected)} episodes: {failures[0]}"
        ) from failures[0]

    if len(selected) < len(pending):
        log.info("smoke grade complete; %d grades remain", len(pending) - len(selected))
        return
    write_report(output, manifest, rows, episodes, grade_manifest)


async def grade(args) -> None:
    profile = JUDGES[args.judge]
    if profile["contract"] in LOCAL_CONTRACTS:
        base_urls = [url.strip() for url in (args.base_urls or "").split(",") if url.strip()]
        if not base_urls or any(
            not re.fullmatch(r"http://(?:127\.0\.0\.1|localhost):[0-9]+/v1", url)
            for url in base_urls
        ):
            raise ValueError("--base-urls must contain local loopback vLLM endpoints")
    else:
        if args.base_urls:
            raise ValueError("--base-urls is only valid with a local judge")
        base_urls = [profile["endpoint"]]
    args.grade_id = args.grade_id or profile["grade_id"]
    if not SAFE_ID.fullmatch(args.grade_id):
        raise ValueError("grade ID must contain only letters, digits, '-' and '_'")
    for run_id, task in resolve_run_path(args.run):
        await grade_task(args, profile, base_urls, run_id, task)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="runs/RUN_ID or runs/RUN_ID/TASK")
    parser.add_argument("--judge", choices=JUDGES, required=True)
    parser.add_argument("--grade-id")
    parser.add_argument("--base-urls", help="comma-separated local vLLM endpoints")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    if args.limit < 0:
        parser.error("--limit must be non-negative")
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        asyncio.run(grade(args))
    except (OSError, ValueError, json.JSONDecodeError, OpenAIError) as error:
        raise SystemExit(f"grading error: {error}") from error


if __name__ == "__main__":
    main()
