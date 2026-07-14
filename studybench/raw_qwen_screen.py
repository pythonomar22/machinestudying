"""Run one explicitly unqualified raw-Qwen SmallDSPy grading screen.

This is a deliberately isolated escape hatch for inspecting the completed
SmallDSPy exploratory runs after the synthetic local-judge qualification
failed.  It does not write normal grades or reports and cannot produce a
claim-ready artifact.  Every answered episode receives exactly one raw judge
request; the treatment ``no_answer`` remains an intention-to-treat zero.
"""

from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from dataclasses import dataclass
import inspect
import os
from pathlib import Path
import subprocess
from statistics import mean
from typing import Any, Callable

from openai import AsyncOpenAI

from . import grade, provenance, report
from .dataset import CORPORA, ROOT, load_questions
from .integrity import (
    canonical_json_bytes,
    exclusive_process_lock,
    read_artifact_bytes,
    sha256_bytes,
    sha256_json,
    strict_json_loads,
    write_immutable_json,
)


RAW_SCREEN_SCHEMA_VERSION = 1
RAW_INTENT_SCHEMA_VERSION = 1
RAW_AUDIT_SCHEMA_VERSION = 1
TASK = "smalldspy"
BASE_ARM = "base"
TREATMENT_ARM = "treatment"
BASE_RUN_ID = "smalldspy-local-base-20260714h"
TREATMENT_RUN_ID = "smalldspy-local-cheatsheet-20260714h"
RUN_MANIFEST_SHA256 = {
    BASE_RUN_ID: "53acd1a4f5fca258a190fc01f8330806c166c64ed5aad0e95b8d33ac3149bacd",
    TREATMENT_RUN_ID: "06da43b3014a22637d51c13be4357ead6faf344dbd0b81cda17fea260a9b18c3",
}
FAILED_QUALIFICATION_PATH = (
    "logs/local-judge-qualification-"
    "38a7c81c25a19e6269fbb9b6daaafa27e0da0aac36451ddd96eb9f85a715058e.json"
)
FAILED_QUALIFICATION_SHA256 = (
    "cc3615daedbbbae96802e93369d7b21a97feaea91da374f51f5f8c4c9ce924c6"
)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 45_001
CONCURRENCY_PER_SERVER = 8
RAW_REQUEST_OPTIONS = {
    "temperature": 0,
    "seed": 0,
    "max_tokens": 256,
    "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
}
RAW_REQUEST_POLICY = "unqualified-qwen-no-thinking-binary-one-request-v1"
FAILED_QUALIFICATION_REQUEST_POLICY = (
    "qwen-answer-centered-system-json-binary-one-attempt-v4"
)
BANNER = (
    "UNQUALIFIED RAW LOCAL-QWEN SCREEN — SYNTHETIC QUALIFICATION FAILED; "
    "NOT CLAIM-READY; DO NOT USE FOR PAPER OR PUBLICATION CLAIMS"
)


class RawQwenScreenError(RuntimeError):
    """The raw screen cannot proceed or cannot produce a complete result."""


@dataclass(frozen=True)
class PreparedScreen:
    intent: dict[str, Any]
    cells: list[dict[str, Any]]
    rows: dict[str, dict[str, Any]]


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except (OSError, ValueError):
        return str(path.resolve())


def _read_canonical_object(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = read_artifact_bytes(path)
        value = strict_json_loads(raw, label=label)
    except (OSError, ValueError) as exc:
        raise RawQwenScreenError(f"{label} is missing or invalid: {path}") from exc
    if not isinstance(value, dict) or raw != canonical_json_bytes(value):
        raise RawQwenScreenError(f"{label} is not a canonical JSON object")
    return value, raw


def _clean_pushed_source() -> dict[str, Any]:
    """Require a clean source commit present in at least one remote-tracking ref."""

    try:
        source = provenance.source_record()
    except (OSError, RuntimeError, ValueError) as exc:
        raise RawQwenScreenError("cannot attest current screen source") from exc
    if source.get("dirty") is not False:
        raise RawQwenScreenError("raw screen requires a clean committed source tree")
    commit = source.get("git_commit")
    try:
        completed = subprocess.run(
            [
                "git", "for-each-ref", "--format=%(refname)",
                f"--contains={commit}", "refs/remotes",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RawQwenScreenError("cannot attest pushed screen source") from exc
    refs = sorted(
        line for line in completed.stdout.splitlines()
        if line.startswith("refs/remotes/") and not line.endswith("/HEAD")
    )
    if not refs:
        raise RawQwenScreenError(
            "raw screen source commit is not present in a remote-tracking ref"
        )
    return {
        "policy": "clean-head-contained-in-remote-tracking-ref-v1",
        "source": source,
        "source_sha256": sha256_json(source),
        "remote_tracking_refs": refs,
    }


def _failed_qualification(path: Path) -> dict[str, Any]:
    try:
        expected_path = (ROOT / FAILED_QUALIFICATION_PATH).resolve(strict=True)
        supplied_path = path.resolve(strict=True)
    except OSError as exc:
        raise RawQwenScreenError("the frozen failed qualification audit is missing") from exc
    if supplied_path != expected_path:
        raise RawQwenScreenError("raw screen requires the exact frozen failed audit")
    audit, raw = _read_canonical_object(path, label="failed qualification audit")
    requests = audit.get("requests")
    responses = audit.get("responses")
    if (
        audit.get("all_passed") is not False
        or audit.get("claim_ready") is not False
        or audit.get("source_stable") is not True
        or audit.get("judge_model") != grade.LOCAL_GRADER_MODEL
        or audit.get("judge_model_revision") != grade.LOCAL_GRADER_MODEL_REVISION
        or audit.get("judge_request_policy")
        != FAILED_QUALIFICATION_REQUEST_POLICY
        or audit.get("judge_request_options") != RAW_REQUEST_OPTIONS
        or audit.get("expected_chat_request_count") != 60
        or not isinstance(requests, list)
        or len(requests) != 60
        or not isinstance(responses, list)
        or len(responses) != 60
        or sha256_bytes(raw) != FAILED_QUALIFICATION_SHA256
    ):
        raise RawQwenScreenError(
            "the supplied audit is not the complete failed no-thinking qualification"
        )
    launch_id = audit.get("server_launch_id")
    if (
        not isinstance(launch_id, str)
        or len(launch_id) != 64
        or any(character not in "0123456789abcdef" for character in launch_id)
    ):
        raise RawQwenScreenError("failed qualification audit has no launch identity")
    return {
        "path": _display_path(path),
        "sha256": sha256_bytes(raw),
        "bytes": len(raw),
        "all_passed": False,
        "server_launch_id": launch_id,
    }


def _manifest_commit(run_root: Path) -> str:
    manifest, _ = _read_canonical_object(
        run_root / TASK / "manifest.json", label="run manifest"
    )
    commit = manifest.get("spec", {}).get("source", {}).get("git_commit")
    if not isinstance(commit, str) or len(commit) not in {40, 64}:
        raise RawQwenScreenError("run manifest has no historical source commit")
    return commit


def _load_arm(run_id: str, questions: list[dict[str, Any]]) -> tuple[dict, list[dict]]:
    try:
        run_id = provenance.validate_id(run_id)
    except (TypeError, ValueError) as exc:
        raise RawQwenScreenError(str(exc)) from exc
    run_root = ROOT / "runs" / run_id
    corpus = CORPORA[TASK]
    try:
        context = grade.load_claim_manifest(
            run_root / TASK,
            corpus,
            questions,
            require_claim_ready=False,
            historical_exploratory_source_commit=_manifest_commit(run_root),
        )
    except (OSError, TypeError, ValueError, grade.GradeIntegrityError) as exc:
        raise RawQwenScreenError(f"run {run_id!r} failed manifest validation") from exc
    if context["spec"].get("run_id") != run_id:
        raise RawQwenScreenError("run directory and manifest IDs differ")
    if context["manifest_sha256"] != RUN_MANIFEST_SHA256.get(run_id):
        raise RawQwenScreenError("run manifest is outside the frozen raw-screen scope")
    rows = {row["id"]: row for row in questions}
    episodes = []
    for relative in context["expected_episodes"]:
        path = run_root / TASK / relative
        episode, raw = _read_canonical_object(path, label="run episode")
        qid = Path(relative).stem
        try:
            grade.validate_episode(episode, rows[qid])
            grade.validate_manifest_episode(episode, rows[qid], context)
        except (KeyError, TypeError, ValueError, grade.GradeIntegrityError) as exc:
            raise RawQwenScreenError(f"invalid run episode {relative}") from exc
        if episode.get("status") not in {"ok", "no_answer"}:
            raise RawQwenScreenError(f"nonfinal run episode {relative}")
        episodes.append({
            "relative": relative,
            "path": _display_path(path),
            "sha256": sha256_bytes(raw),
            "episode": episode,
        })
    return context, episodes


def _paired_inputs(
    base: tuple[dict, list[dict]], treatment: tuple[dict, list[dict]]
) -> None:
    left, left_episodes = base
    right, right_episodes = treatment
    for field in ("task", "budgets", "rollouts", "questions", "expected_episodes"):
        if left["spec"].get(field) != right["spec"].get(field):
            raise RawQwenScreenError(f"screen arms differ in manifest {field}")
    if left["spec"].get("budgets") != report.BUDGET_ORDER:
        raise RawQwenScreenError("screen arms do not use the report WAUC budget grid")
    if len(left_episodes) != 60 or len(right_episodes) != 60:
        raise RawQwenScreenError("raw SmallDSPy screen requires 60 cells per arm")
    left_status = [item["episode"]["status"] for item in left_episodes]
    right_status = [item["episode"]["status"] for item in right_episodes]
    if left_status.count("ok") != 60 or left_status.count("no_answer") != 0:
        raise RawQwenScreenError("base arm is not the expected 60 answered cells")
    if right_status.count("ok") != 59 or right_status.count("no_answer") != 1:
        raise RawQwenScreenError(
            "treatment arm is not the expected 59 answered plus one no_answer cells"
        )


def prepare_screen(
    *,
    base_run_id: str,
    treatment_run_id: str,
    judge_base_urls: str,
    failed_qualification_audit: str | Path,
) -> PreparedScreen:
    """Validate all inputs and build the complete immutable request intent."""

    if base_run_id != BASE_RUN_ID or treatment_run_id != TREATMENT_RUN_ID:
        raise RawQwenScreenError(
            "raw screen is pinned to the completed base and treatment run IDs"
        )

    questions = load_questions(TASK)
    rows = {row["id"]: row for row in questions}
    if len(rows) != 5 or len(rows) != len(questions):
        raise RawQwenScreenError("SmallDSPy question bundle is not the expected five rows")
    for row in questions:
        grade.rubric_ids(row)
    base = _load_arm(base_run_id, questions)
    treatment = _load_arm(treatment_run_id, questions)
    _paired_inputs(base, treatment)

    try:
        urls = provenance.validate_local_server_urls(judge_base_urls)
        grade._validate_local_grader_environment(urls)
        grading_runtime = provenance.grading_runtime_record()
        local_runtime = provenance.local_judge_runtime_record()
    except (OSError, TypeError, ValueError, grade.GradeIntegrityError) as exc:
        raise RawQwenScreenError("local judge launch provenance is invalid") from exc
    expected_servers = base[0]["spec"]["server_assignment"]["server_count"]
    if (
        len(urls) != expected_servers
        or treatment[0]["spec"]["server_assignment"]["server_count"]
        != expected_servers
        or local_runtime["server"]["server_count"] != expected_servers
    ):
        raise RawQwenScreenError("judge topology does not match both run manifests")

    source = _clean_pushed_source()
    qualification = _failed_qualification(Path(failed_qualification_audit))
    if qualification["server_launch_id"] == local_runtime["server_launch_id"]:
        raise RawQwenScreenError(
            "raw screen requires a fresh launch distinct from the failed qualification"
        )

    cells: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    contexts = {BASE_ARM: base, TREATMENT_ARM: treatment}
    for arm, (context, episodes) in contexts.items():
        slots = context["episode_server_slots"]
        for item in episodes:
            episode = item["episode"]
            cell = {
                "arm": arm,
                "relative": item["relative"],
                "episode_path": item["path"],
                "episode_sha256": item["sha256"],
                "qid": episode["qid"],
                "budget": episode["budget"],
                "rollout": episode["rollout"],
                "status": episode["status"],
                "gen_tokens": episode["gen_tokens"],
                "server_slot": episode["server_slot"],
            }
            cells.append(cell)
            if episode["status"] == "no_answer":
                continue
            slot = slots[item["relative"]]
            messages = grade.build_judge_messages(
                CORPORA[TASK], rows[episode["qid"]], episode["answer"],
                False, grade.LOCAL_GRADER_MODEL,
            )
            payload = {
                "model": grade.LOCAL_GRADER_MODEL,
                "messages": messages,
                "response_format": grade.judge_schema(
                    rows[episode["qid"]], grade.LOCAL_GRADER_MODEL
                ),
                **deepcopy(RAW_REQUEST_OPTIONS),
            }
            requests.append({
                "request_index": len(requests),
                "arm": arm,
                "relative": item["relative"],
                "qid": episode["qid"],
                "server_slot": slot,
                "url": urls[slot],
                "payload": payload,
                "payload_sha256": sha256_json(payload),
            })
    if len(requests) != 119:
        raise RawQwenScreenError("raw screen did not resolve to exactly 119 requests")

    arm_sources = {
        arm: {
            "run_id": context["spec"]["run_id"],
            "manifest_path": _display_path(
                context["run_task_root"] / "manifest.json"
            ),
            "manifest_sha256": context["manifest_sha256"],
            "generation_source_sha256": sha256_json(context["spec"]["source"]),
            "population_cells": len(episodes),
            "answered_cells": sum(
                item["episode"]["status"] == "ok" for item in episodes
            ),
        }
        for arm, (context, episodes) in contexts.items()
    }
    intent = {
        "raw_intent_schema_version": RAW_INTENT_SCHEMA_VERSION,
        "claim_ready": False,
        "judge_qualified": False,
        "diagnostic_only": True,
        "banner": BANNER,
        "task": TASK,
        "arms": arm_sources,
        "failed_qualification_audit": qualification,
        "failed_qualification_audit_sha256": qualification["sha256"],
        "grader_source": source,
        "grading_runtime": grading_runtime,
        "grading_runtime_sha256": provenance.grading_runtime_sha256(
            grading_runtime
        ),
        "local_judge_runtime": local_runtime,
        "local_judge_runtime_sha256": provenance.local_judge_runtime_sha256(
            local_runtime
        ),
        "judge": {
            "model": grade.LOCAL_GRADER_MODEL,
            "model_revision": grade.LOCAL_GRADER_MODEL_REVISION,
            "endpoint_identity": grade.LOCAL_GRADER_ENDPOINT_IDENTITY,
            "ordered_urls": urls,
            "server_assignment": grade.LOCAL_GRADER_SERVER_ASSIGNMENT_POLICY,
            "request_policy": RAW_REQUEST_POLICY,
            "request_options": deepcopy(RAW_REQUEST_OPTIONS),
            "verdict_contract": grade.LOCAL_GRADER_VERDICT_CONTRACT,
            "attempts_per_answered_cell": 1,
            "retries": 0,
        },
        "estimand": {
            "population": "all_60_manifest_cells_per_arm_intention_to_treat",
            "no_answer": "zero",
            "score": "lenient_weighted_binary_claim_sum",
            "wauc": "studybench.report.expertise",
        },
        "bootstrap": {
            "method": "studybench.report.bootstrap_population",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
        },
        "cells": cells,
        "requests": requests,
        "requests_sha256": sha256_json(requests),
    }
    return PreparedScreen(intent=intent, cells=cells, rows=rows)


def _error(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


async def _one_request(
    request: dict[str, Any],
    row: dict[str, Any],
    client: Any,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        key: request[key]
        for key in (
            "request_index", "arm", "relative", "qid", "server_slot",
            "url", "payload_sha256",
        )
    }
    try:
        async with semaphore:
            response = await client.chat.completions.create(**request["payload"])
    except Exception as exc:
        return {**record, "accepted": False, "content": None,
                "response": None, "validation_error": _error(exc)}

    try:
        response_record, content, response_error, _ = grade._response_attempt(response, 1)
    except Exception as exc:  # defensive: retain a terminal audit even on inspector failure
        return {**record, "accepted": False, "content": None,
                "response": None, "validation_error": _error(exc)}
    error: BaseException | None = response_error
    claims = None
    scores = None
    if error is None and response_record.get("response_model") != grade.LOCAL_GRADER_MODEL:
        error = RawQwenScreenError("response model does not match pinned local Qwen")
    if error is None:
        try:
            verdict = grade.parse_json(content, label="raw Qwen verdict")
            claims, claim_scores = grade.validate_verdict(
                row, verdict, grade.LOCAL_GRADER_MODEL
            )
            scores = grade.score_from_claims(row, claim_scores, compile_ok=False)
        except (TypeError, ValueError, grade.GradeIntegrityError) as exc:
            error = exc
    response_record["accepted"] = error is None
    if error is None:
        try:
            grade.validate_judge_attempt_record(
                response_record, 1, accepted=True
            )
        except (TypeError, ValueError, grade.GradeIntegrityError) as exc:
            error = exc
            response_record["accepted"] = False
    if error is not None:
        response_record["validation_error"] = _error(error)
        if isinstance(content, str):
            response_record["invalid_content"] = content
    return {
        **record,
        "accepted": error is None,
        "content": content,
        "response": response_record,
        "validation_error": _error(error) if error is not None else None,
        "claims": claims,
        "lenient": scores["lenient"] if scores is not None else None,
        "cores_ok": scores["cores_ok"] if scores is not None else None,
    }


def _revalidate_provenance(intent: dict[str, Any]) -> None:
    try:
        source = intent["grader_source"]["source"]
        provenance.validate_current_source(source)
        if sha256_json(source) != intent["grader_source"]["source_sha256"]:
            raise ValueError("grader source hash differs from its attestation")
        provenance.validate_grading_runtime_record(
            intent["grading_runtime"], require_current=True
        )
        if provenance.grading_runtime_sha256(
            intent["grading_runtime"]
        ) != intent["grading_runtime_sha256"]:
            raise ValueError("grading runtime hash differs from its attestation")
        provenance.validate_local_judge_runtime_record(
            intent["local_judge_runtime"], require_current=True
        )
        if provenance.local_judge_runtime_sha256(
            intent["local_judge_runtime"]
        ) != intent["local_judge_runtime_sha256"]:
            raise ValueError("local runtime hash differs from its attestation")
        qualification = intent["failed_qualification_audit"]
        raw = read_artifact_bytes(ROOT / qualification["path"])
    except (OSError, TypeError, ValueError) as exc:
        raise RawQwenScreenError("screen provenance changed during model contact") from exc
    if (
        sha256_bytes(raw) != qualification["sha256"]
        or len(raw) != qualification["bytes"]
    ):
        raise RawQwenScreenError("failed qualification audit changed during contact")


def _validate_response_census(
    intent: dict[str, Any], responses: list[dict[str, Any]],
) -> None:
    """Require one exact terminal response identity for every intended request."""

    identity_fields = (
        "request_index", "arm", "relative", "qid", "server_slot", "url",
        "payload_sha256",
    )
    expected = [
        {field: request[field] for field in identity_fields}
        for request in intent["requests"]
    ]
    observed = [
        {field: response.get(field) for field in identity_fields}
        for response in responses
    ]
    if len(responses) != len(expected) or observed != expected:
        raise RawQwenScreenError(
            "raw response census does not exactly match the request intent"
        )
    indices = [response["request_index"] for response in responses]
    if indices != list(range(len(expected))) or len(indices) != len(set(indices)):
        raise RawQwenScreenError("raw response indices are missing or duplicated")
    for response in responses:
        if not response.get("accepted"):
            continue
        content = response.get("content")
        attempt = response.get("response")
        if not isinstance(content, str) or not isinstance(attempt, dict):
            raise RawQwenScreenError("accepted raw response has no exact content")
        encoded = content.encode("utf-8")
        if (
            attempt.get("content_sha256") != sha256_bytes(encoded)
            or attempt.get("content_bytes") != len(encoded)
        ):
            raise RawQwenScreenError(
                "accepted raw response content differs from its provider audit"
            )


def _arm_summary(
    arm: str, cells: list[dict[str, Any]], responses: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    by_request = {(item["arm"], item["relative"]): item for item in responses}
    population = {budget: [] for budget in report.BUDGET_ORDER}
    for cell in cells:
        if cell["arm"] != arm:
            continue
        response = by_request.get((arm, cell["relative"]))
        no_answer = cell["status"] == "no_answer"
        if no_answer != (response is None):
            raise RawQwenScreenError("response grid does not match ITT episode statuses")
        population[cell["budget"]].append({
            "qid": cell["qid"],
            "budget": cell["budget"],
            "rollout": cell["rollout"],
            "episode_status": cell["status"],
            "gen_tokens": cell["gen_tokens"],
            "lenient": 0 if no_answer else response["lenient"],
            "cores_ok": False if no_answer else response["cores_ok"],
        })
    budgets = {}
    points = []
    for budget in report.BUDGET_ORDER:
        values = population[budget]
        if len(values) != 15:
            raise RawQwenScreenError("arm budget does not contain 15 ITT cells")
        score = mean(item["lenient"] for item in values)
        tokens = mean(item["gen_tokens"] for item in values)
        budgets[budget] = {
            "n": 15,
            "answered": sum(item["episode_status"] == "ok" for item in values),
            "no_answer": sum(
                item["episode_status"] == "no_answer" for item in values
            ),
            "mean_lenient": score,
            "mean_gen_tokens": tokens,
        }
        points.append((tokens, score))
    bootstrap = report.bootstrap_population(
        population, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED
    )
    return {
        "budgets": budgets,
        "lenient_wauc": report.expertise(points),
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "budget_lenient": {
                budget: bootstrap[budget] for budget in report.BUDGET_ORDER
            },
            "lenient_wauc": bootstrap["wauc"],
        },
    }, population


async def run_prepared_screen(
    prepared: PreparedScreen,
    *,
    output_dir: Path,
    client_factory: Callable[[str], Any] | None = None,
    provenance_revalidator: Callable[[dict[str, Any]], None] = _revalidate_provenance,
) -> Path:
    """Persist intent, contact each answered cell once, and persist audit/result."""

    output_dir = Path(output_dir)
    intent_path = output_dir / "intent.json"
    audit_path = output_dir / "raw-audit.json"
    result_path = output_dir / "result.json"
    existing = [path for path in (intent_path, audit_path, result_path) if path.exists() or path.is_symlink()]
    if existing:
        raise RawQwenScreenError(
            "raw screen namespace is not fresh: " + ", ".join(map(str, existing))
        )
    write_immutable_json(intent_path, prepared.intent)
    intent_raw = read_artifact_bytes(intent_path)
    if intent_raw != canonical_json_bytes(prepared.intent):
        raise RawQwenScreenError("durable raw-screen intent failed verification")

    api_key = os.environ.get("SB_VLLM_API_KEY")
    if client_factory is None:
        if not api_key:
            raise RawQwenScreenError("SB_VLLM_API_KEY is missing after intent write")
        client_factory = lambda url: AsyncOpenAI(
            timeout=600, max_retries=0, base_url=url, api_key=api_key
        )
    urls = prepared.intent["judge"]["ordered_urls"]
    clients: dict[str, Any] = {}
    responses: list[dict[str, Any]] = []
    terminal_error = None
    try:
        provenance_revalidator(prepared.intent)
    except Exception as exc:
        terminal_error = _error(exc)
    if terminal_error is None:
        try:
            for url in urls:
                clients[url] = client_factory(url)
            semaphores = {
                url: asyncio.Semaphore(CONCURRENCY_PER_SERVER) for url in urls
            }
            responses = list(await asyncio.gather(*(
                _one_request(
                    request,
                    prepared.rows[request["qid"]],
                    clients[request["url"]],
                    semaphores[request["url"]],
                )
                for request in prepared.intent["requests"]
            )))
        except BaseException as exc:
            terminal_error = _error(exc)
        finally:
            for client in clients.values():
                close = getattr(client, "close", None)
                if not callable(close):
                    close = getattr(client, "aclose", None)
                if not callable(close):
                    continue
                try:
                    result = close()
                    if inspect.isawaitable(result):
                        await result
                except BaseException as exc:
                    if terminal_error is None:
                        terminal_error = _error(exc)

    try:
        provenance_revalidator(prepared.intent)
    except Exception as exc:
        if terminal_error is None:
            terminal_error = _error(exc)
    try:
        _validate_response_census(prepared.intent, responses)
    except Exception as exc:
        if terminal_error is None:
            terminal_error = _error(exc)
    rejected = [item["request_index"] for item in responses if not item["accepted"]]
    if rejected and terminal_error is None:
        terminal_error = {
            "type": "RejectedJudgeResponses",
            "message": f"{len(rejected)} response(s) failed closed validation",
        }
    arm_results: dict[str, Any] = {}
    if terminal_error is None:
        try:
            for arm in (BASE_ARM, TREATMENT_ARM):
                arm_results[arm], _ = _arm_summary(
                    arm, prepared.cells, responses
                )
            provenance_revalidator(prepared.intent)
        except Exception as exc:
            terminal_error = _error(exc)
    audit = {
        "raw_audit_schema_version": RAW_AUDIT_SCHEMA_VERSION,
        "claim_ready": False,
        "judge_qualified": False,
        "diagnostic_only": True,
        "banner": BANNER,
        "intent_path": _display_path(intent_path),
        "intent_sha256": sha256_bytes(intent_raw),
        "failed_qualification_audit_sha256": prepared.intent[
            "failed_qualification_audit_sha256"
        ],
        "expected_request_count": len(prepared.intent["requests"]),
        "request_count": len(responses),
        "accepted_count": sum(item["accepted"] for item in responses),
        "rejected_request_indices": rejected,
        "complete": terminal_error is None,
        "terminal_error": terminal_error,
        "responses": responses,
        "responses_sha256": sha256_json(responses),
    }
    write_immutable_json(audit_path, audit)
    audit_raw = read_artifact_bytes(audit_path)
    if audit_raw != canonical_json_bytes(audit):
        raise RawQwenScreenError("durable raw response audit failed verification")
    if terminal_error is not None:
        raise RawQwenScreenError(
            f"raw screen failed closed; terminal audit: {_display_path(audit_path)}"
        )
    result = {
        "raw_screen_schema_version": RAW_SCREEN_SCHEMA_VERSION,
        "claim_ready": False,
        "judge_qualified": False,
        "diagnostic_only": True,
        "banner": BANNER,
        "task": TASK,
        "estimand": prepared.intent["estimand"],
        "failed_qualification_audit_sha256": prepared.intent[
            "failed_qualification_audit_sha256"
        ],
        "intent": {
            "path": _display_path(intent_path),
            "sha256": sha256_bytes(intent_raw),
            "bytes": len(intent_raw),
        },
        "raw_audit": {
            "path": _display_path(audit_path),
            "sha256": sha256_bytes(audit_raw),
            "bytes": len(audit_raw),
        },
        "arms": arm_results,
    }
    write_immutable_json(result_path, result)
    result_raw = read_artifact_bytes(result_path)
    if result_raw != canonical_json_bytes(result):
        raise RawQwenScreenError("durable raw-screen result failed verification")
    return result_path


async def run_screen(
    *,
    base_run_id: str,
    treatment_run_id: str,
    judge_base_urls: str,
    failed_qualification_audit: str | Path,
    screen_id: str,
    output_root: str | Path = "raw-qwen-screens",
) -> Path:
    try:
        screen_id = provenance.validate_id(screen_id, "screen ID")
    except (TypeError, ValueError) as exc:
        raise RawQwenScreenError(str(exc)) from exc
    root = Path(output_root)
    if not root.is_absolute():
        root = ROOT / root
    output_dir = root / screen_id
    with exclusive_process_lock(output_dir / ".lock"):
        unexpected = [path for path in output_dir.iterdir() if path.name != ".lock"]
        if unexpected:
            raise RawQwenScreenError("raw screen output namespace is not fresh")
        prepared = prepare_screen(
            base_run_id=base_run_id,
            treatment_run_id=treatment_run_id,
            judge_base_urls=judge_base_urls,
            failed_qualification_audit=failed_qualification_audit,
        )
        return await run_prepared_screen(prepared, output_dir=output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run-id", required=True)
    parser.add_argument("--treatment-run-id", required=True)
    parser.add_argument("--judge-base-url", required=True)
    parser.add_argument("--failed-qualification-audit", required=True)
    parser.add_argument("--screen-id", required=True)
    parser.add_argument("--output-dir", default="raw-qwen-screens")
    args = parser.parse_args()
    try:
        path = asyncio.run(run_screen(
            base_run_id=args.base_run_id,
            treatment_run_id=args.treatment_run_id,
            judge_base_urls=args.judge_base_url,
            failed_qualification_audit=args.failed_qualification_audit,
            screen_id=args.screen_id,
            output_root=args.output_dir,
        ))
    except (OSError, TypeError, ValueError, RawQwenScreenError) as exc:
        raise SystemExit(f"INTEGRITY ERROR: {exc}") from exc
    print(BANNER)
    print(f"immutable unqualified result: {_display_path(path)}")


if __name__ == "__main__":
    main()
