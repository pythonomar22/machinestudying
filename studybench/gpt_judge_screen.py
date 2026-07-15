"""Regrade the frozen SmallDSPy screen with the paper-model GPT judge.

This module is deliberately separate from the confirmatory grading path.  The
two source runs are historical, adaptive, exploratory artifacts without a
preregistration, so an external regrade cannot make them claim-ready.  The
screen instead freezes one joint 120-cell intention-to-treat population, sends
each of the 119 stored answers to GPT-5.4 exactly once, and leaves the treatment
``no_answer`` as zero without contacting the judge.

The GPT protocol follows the repository's reconstruction of Appendix A.5:
whole numbered evidence files, binary claims per the author's correction,
concise rationales, and high reasoning effort per the repository convention.
It is not a controlled model-only comparison with the raw-Qwen screen, whose
prompt, evidence, and output contract differ.
"""

from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from dataclasses import dataclass
import inspect
import os
from pathlib import Path
import random
from statistics import mean
from typing import Any, Callable

from openai import AsyncOpenAI

from . import grade, provenance, raw_qwen_screen as raw_qwen, report
from .dataset import CORPORA, ROOT, load_questions
from .env import load_private_env
from .integrity import (
    canonical_json_bytes,
    exclusive_process_lock,
    read_artifact_bytes,
    sha256_bytes,
    sha256_json,
    strict_json_loads,
    write_immutable_json,
)


SCREEN_SCHEMA_VERSION = 1
INTENT_SCHEMA_VERSION = 1
AUDIT_SCHEMA_VERSION = 1
TASK = raw_qwen.TASK
BASE_ARM = raw_qwen.BASE_ARM
TREATMENT_ARM = raw_qwen.TREATMENT_ARM
BASE_RUN_ID = raw_qwen.BASE_RUN_ID
TREATMENT_RUN_ID = raw_qwen.TREATMENT_RUN_ID
JUDGE_MODEL = grade.GRADERS["openai"][0]
JUDGE_BASE_URL = grade.CANONICAL_OPENAI_BASE_URL
JUDGE_EFFORT = "high"
CONCURRENCY = 8
BOOTSTRAP_REPLICATES = raw_qwen.BOOTSTRAP_REPLICATES
BOOTSTRAP_SEED = raw_qwen.BOOTSTRAP_SEED
REQUEST_POLICY = "gpt54-a5-binary-whole-files-high-one-request-v1"
REQUEST_OPTIONS = {"reasoning_effort": JUDGE_EFFORT}
REQUEST_ORDER_POLICY = "paired-relative-base-then-treatment-v1"
QWEN_ROOT = (
    ROOT
    / "raw-qwen-screens"
    / "smalldspy-base-cheatsheet-raw-qwen-20260714n"
)
QWEN_ARTIFACT_SHA256 = {
    "intent.json": "85c0fc11ebd5e6f91264fb2aefc852ed1eab3751a5b1e2f7fecdbcc7b21601ab",
    "raw-audit.json": "59c1434044ee567aee090862b9ebe31bd362f048c4aee577e608a80af7674e42",
    "result.json": "d10ee48bc7e0ef85c696fddf195bdfe4348aeb1f85d288cc8995bc4f79631413",
}
BANNER = (
    "POST-HOC GPT-5.4 GRADER-PROTOCOL SENSITIVITY SCREEN — ADAPTIVE "
    "FIVE-QUESTION POPULATION; NOT CLAIM-READY; NOT A TABLE-1 REPLICATION"
)


class GPTJudgeScreenError(RuntimeError):
    """The GPT screen cannot proceed or cannot produce a complete result."""


@dataclass(frozen=True)
class PreparedScreen:
    intent: dict[str, Any]
    cells: list[dict[str, Any]]
    rows: dict[str, dict[str, Any]]


def _display_path(path: Path) -> str:
    return raw_qwen._display_path(path)


def _read_canonical_object(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = read_artifact_bytes(path)
        value = strict_json_loads(raw, label=label)
    except (OSError, ValueError) as exc:
        raise GPTJudgeScreenError(f"{label} is missing or invalid: {path}") from exc
    if not isinstance(value, dict) or raw != canonical_json_bytes(value):
        raise GPTJudgeScreenError(f"{label} is not a canonical JSON object")
    return value, raw


def _qwen_reference() -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate and bind the exact raw-Qwen screen being compared."""

    records: dict[str, Any] = {}
    values: dict[str, Any] = {}
    for name, expected_sha256 in QWEN_ARTIFACT_SHA256.items():
        path = QWEN_ROOT / name
        value, raw = _read_canonical_object(path, label=f"raw-Qwen {name}")
        digest = sha256_bytes(raw)
        if digest != expected_sha256:
            raise GPTJudgeScreenError(f"raw-Qwen {name} differs from its frozen hash")
        records[name] = {
            "path": _display_path(path),
            "sha256": digest,
            "bytes": len(raw),
        }
        values[name] = value
    intent = values["intent.json"]
    audit = values["raw-audit.json"]
    result = values["result.json"]
    if (
        intent.get("claim_ready") is not False
        or audit.get("complete") is not True
        or audit.get("accepted_count") != 119
        or result.get("claim_ready") is not False
        or result.get("judge_qualified") is not False
        or result.get("diagnostic_only") is not True
    ):
        raise GPTJudgeScreenError("raw-Qwen reference is not the frozen complete diagnostic")
    return records, values


def _cell_record(arm: str, item: dict[str, Any]) -> dict[str, Any]:
    episode = item["episode"]
    return {
        "arm": arm,
        "relative": item["relative"],
        "episode_path": item["path"],
        "episode_sha256": item["sha256"],
        "qid": episode["qid"],
        "budget": episode["budget"],
        "rollout": episode["rollout"],
        "status": episode["status"],
        "gen_tokens": episode["gen_tokens"],
        "generation_server_slot": episode["server_slot"],
    }


def _request_payload(row: dict[str, Any], answer: str) -> dict[str, Any]:
    """Build the exact paper-style GPT payload for one frozen answer."""

    return {
        "model": JUDGE_MODEL,
        "messages": grade.build_judge_messages(
            CORPORA[TASK], row, answer, True, JUDGE_MODEL
        ),
        "response_format": grade.judge_schema(row, JUDGE_MODEL),
        **deepcopy(REQUEST_OPTIONS),
    }


def prepare_screen(*, base_run_id: str, treatment_run_id: str) -> PreparedScreen:
    """Validate both frozen arms and build the complete pre-contact intent."""

    if base_run_id != BASE_RUN_ID or treatment_run_id != TREATMENT_RUN_ID:
        raise GPTJudgeScreenError(
            "GPT screen is pinned to the completed h base and treatment run IDs"
        )
    questions = load_questions(TASK)
    rows = {row["id"]: row for row in questions}
    if len(rows) != 5 or len(rows) != len(questions):
        raise GPTJudgeScreenError("SmallDSPy question bundle is not the expected five rows")
    for row in questions:
        grade.rubric_ids(row)

    try:
        base = raw_qwen._load_arm(base_run_id, questions)
        treatment = raw_qwen._load_arm(treatment_run_id, questions)
        raw_qwen._paired_inputs(base, treatment)
        source = raw_qwen._clean_pushed_source()
        grading_runtime = provenance.grading_runtime_record()
    except (
        OSError,
        TypeError,
        ValueError,
        grade.GradeIntegrityError,
        raw_qwen.RawQwenScreenError,
    ) as exc:
        raise GPTJudgeScreenError("frozen GPT screen preflight failed") from exc

    qwen_records, _ = _qwen_reference()
    contexts = {BASE_ARM: base, TREATMENT_ARM: treatment}
    episodes_by_arm = {
        arm: {item["relative"]: item for item in episodes}
        for arm, (_, episodes) in contexts.items()
    }
    relatives = [item["relative"] for item in base[1]]
    if set(relatives) != set(episodes_by_arm[TREATMENT_ARM]):
        raise GPTJudgeScreenError("paired arm episode grids differ")

    cells = [
        _cell_record(arm, item)
        for arm, (_, episodes) in contexts.items()
        for item in episodes
    ]
    requests: list[dict[str, Any]] = []
    for relative in relatives:
        for arm in (BASE_ARM, TREATMENT_ARM):
            item = episodes_by_arm[arm][relative]
            episode = item["episode"]
            if episode["status"] == "no_answer":
                continue
            payload = _request_payload(rows[episode["qid"]], episode["answer"])
            requests.append({
                "request_index": len(requests),
                "arm": arm,
                "relative": relative,
                "qid": episode["qid"],
                "url": JUDGE_BASE_URL,
                "payload": payload,
                "payload_sha256": sha256_json(payload),
            })
    if len(cells) != 120 or len(requests) != 119:
        raise GPTJudgeScreenError("GPT screen population is not exactly 120/119")

    arm_sources = {
        arm: {
            "run_id": context["spec"]["run_id"],
            "manifest_path": _display_path(context["run_task_root"] / "manifest.json"),
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
        "gpt_judge_intent_schema_version": INTENT_SCHEMA_VERSION,
        "claim_ready": False,
        "diagnostic_only": True,
        "paper_comparison_allowed": False,
        "banner": BANNER,
        "task": TASK,
        "arms": arm_sources,
        "raw_qwen_reference": qwen_records,
        "grader_source": source,
        "grading_runtime": grading_runtime,
        "grading_runtime_sha256": provenance.grading_runtime_sha256(
            grading_runtime
        ),
        "judge": {
            "provider": "openai",
            "requested_model": JUDGE_MODEL,
            "endpoint": JUDGE_BASE_URL,
            "evidence_mode": "whole_files",
            "prompt_contract": "studybench-grade-external-a5-reconstruction-v1",
            "claim_contract": "binary-0-or-1-with-concise-rationale-v1",
            "judge_effort": JUDGE_EFFORT,
            "effort_provenance": "repository-convention-not-author-confirmed",
            "request_policy": REQUEST_POLICY,
            "request_options": deepcopy(REQUEST_OPTIONS),
            "request_order_policy": REQUEST_ORDER_POLICY,
            "concurrency": CONCURRENCY,
            "attempts_per_answered_cell": 1,
            "sdk_retries": 0,
            "score_source": "harness-recomputed-from-exact-binary-claim-labels",
        },
        "estimand": {
            "population": "all_60_manifest_cells_per_arm_intention_to_treat",
            "no_answer": "zero_without_judge_contact",
            "score": "lenient_weighted_binary_claim_sum",
            "wauc": "studybench.report.expertise",
            "strict_and_compile": "not_measured_in_this_lenient-only-screen",
        },
        "bootstrap": {
            "method": "paired_question_then_shared-rollout-resampling",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "uncertainty_scope": (
                "question-and-rollout-sampling-only; judge-systematic-and-"
                "adaptive-reuse-uncertainty-omitted"
            ),
        },
        "cells": cells,
        "cells_sha256": sha256_json(cells),
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
    identity = {
        key: request[key]
        for key in (
            "request_index",
            "arm",
            "relative",
            "qid",
            "url",
            "payload_sha256",
        )
    }
    try:
        async with semaphore:
            response = await client.chat.completions.create(**request["payload"])
    except Exception as exc:
        return {
            **identity,
            "accepted": False,
            "content": None,
            "response": None,
            "validation_error": _error(exc),
        }

    try:
        response_record, content, response_error, _ = grade._response_attempt(response, 1)
    except Exception as exc:
        return {
            **identity,
            "accepted": False,
            "content": None,
            "response": None,
            "validation_error": _error(exc),
        }
    error: BaseException | None = response_error
    claims = None
    scores = None
    if error is None:
        try:
            verdict = grade.parse_json(content, label="GPT-5.4 verdict")
            claims, claim_scores = grade.validate_verdict(row, verdict, JUDGE_MODEL)
            scores = grade.score_from_claims(row, claim_scores, compile_ok=False)
        except (TypeError, ValueError, grade.GradeIntegrityError) as exc:
            error = exc
    response_record["accepted"] = error is None
    if error is None:
        try:
            grade.validate_judge_attempt_record(response_record, 1, accepted=True)
        except (TypeError, ValueError, grade.GradeIntegrityError) as exc:
            error = exc
            response_record["accepted"] = False
    if error is not None:
        response_record["validation_error"] = _error(error)
        if isinstance(content, str):
            response_record["invalid_content"] = content
    return {
        **identity,
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
        for cell in intent["cells"]:
            raw = read_artifact_bytes(ROOT / cell["episode_path"])
            if sha256_bytes(raw) != cell["episode_sha256"]:
                raise ValueError(f"episode changed: {cell['episode_path']}")
        qwen_records, _ = _qwen_reference()
        if qwen_records != intent["raw_qwen_reference"]:
            raise ValueError("raw-Qwen reference changed")
    except (OSError, TypeError, ValueError) as exc:
        raise GPTJudgeScreenError("GPT screen provenance changed during contact") from exc


def _validate_response_census(
    intent: dict[str, Any], responses: list[dict[str, Any]]
) -> None:
    identity_fields = (
        "request_index",
        "arm",
        "relative",
        "qid",
        "url",
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
        raise GPTJudgeScreenError("response census does not match the frozen request intent")
    for index, response in enumerate(responses):
        if response.get("request_index") != index:
            raise GPTJudgeScreenError("response indices are missing or reordered")
        if not response.get("accepted"):
            continue
        content = response.get("content")
        attempt = response.get("response")
        if not isinstance(content, str) or not isinstance(attempt, dict):
            raise GPTJudgeScreenError("accepted response has no exact provider content")
        encoded = content.encode("utf-8")
        if (
            attempt.get("content_sha256") != sha256_bytes(encoded)
            or attempt.get("content_bytes") != len(encoded)
        ):
            raise GPTJudgeScreenError("accepted response content differs from its audit")


def _paired_bootstrap(
    control: dict[str, list[dict[str, Any]]],
    treatment: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Bootstrap the paired lenient arm contrast without inventing strict scores."""

    def index(population: dict[str, list[dict[str, Any]]]) -> dict[tuple[str, int, str], dict]:
        return {
            (item["budget"], item["rollout"], item["qid"]): item
            for values in population.values()
            for item in values
        }

    left = index(control)
    right = index(treatment)
    if set(left) != set(right):
        raise GPTJudgeScreenError("paired bootstrap arm grids differ")
    qids = sorted({key[2] for key in left})
    rollouts = {
        (budget, qid): sorted(
            rollout for b, rollout, q in left if b == budget and q == qid
        )
        for budget in report.BUDGET_ORDER
        for qid in qids
    }
    if any(values != list(range(len(values))) for values in rollouts.values()):
        raise GPTJudgeScreenError("paired bootstrap rollout grid is incomplete")
    rng = random.Random(BOOTSTRAP_SEED)
    budget_deltas = {budget: [] for budget in report.BUDGET_ORDER}
    wauc_deltas: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled_qids = rng.choices(qids, k=len(qids))
        points = {BASE_ARM: [], TREATMENT_ARM: []}
        for budget in report.BUDGET_ORDER:
            values = {BASE_ARM: [], TREATMENT_ARM: []}
            tokens = {BASE_ARM: [], TREATMENT_ARM: []}
            for qid in sampled_qids:
                available = rollouts[(budget, qid)]
                for rollout in rng.choices(available, k=len(available)):
                    key = (budget, rollout, qid)
                    for arm, population in ((BASE_ARM, left), (TREATMENT_ARM, right)):
                        values[arm].append(population[key]["lenient"])
                        tokens[arm].append(population[key]["gen_tokens"])
            arm_means = {arm: mean(values[arm]) for arm in values}
            budget_deltas[budget].append(
                arm_means[TREATMENT_ARM] - arm_means[BASE_ARM]
            )
            for arm in (BASE_ARM, TREATMENT_ARM):
                points[arm].append((mean(tokens[arm]), arm_means[arm]))
        wauc_deltas.append(
            report.expertise(points[TREATMENT_ARM])
            - report.expertise(points[BASE_ARM])
        )

    def interval(values: list[float]) -> dict[str, float]:
        ordered = sorted(values)
        return {
            "mean": mean(values),
            "lower_95": ordered[round(0.025 * (len(ordered) - 1))],
            "upper_95": ordered[round(0.975 * (len(ordered) - 1))],
        }

    return {
        "method": "paired_question_then_shared-rollout_resampling",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "budget_lenient_treatment_minus_base": {
            budget: interval(values) for budget, values in budget_deltas.items()
        },
        "lenient_wauc_treatment_minus_base": interval(wauc_deltas),
        "uncertainty_scope": (
            "question-and-rollout-sampling-only; judge-systematic-and-adaptive-"
            "reuse-uncertainty-omitted"
        ),
    }


def _judge_protocol_comparison(
    gpt_results: dict[str, Any], gpt_responses: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compare frozen aggregate and claim labels without calling them model-only."""

    _, qwen_values = _qwen_reference()
    qwen_result = qwen_values["result.json"]
    qwen_audit = qwen_values["raw-audit.json"]
    qwen_by_cell = {
        (item["arm"], item["relative"]): item for item in qwen_audit["responses"]
    }
    confusion = {
        "qwen_0_gpt_0": 0,
        "qwen_0_gpt_1": 0,
        "qwen_1_gpt_0": 0,
        "qwen_1_gpt_1": 0,
    }
    arm_disagreements = {
        BASE_ARM: {"claims": 0, "disagreements": 0},
        TREATMENT_ARM: {"claims": 0, "disagreements": 0},
    }
    for response in gpt_responses:
        qwen = qwen_by_cell[(response["arm"], response["relative"])]
        qwen_claims = {claim["claim_id"]: claim["score"] for claim in qwen["claims"]}
        gpt_claims = {claim["claim_id"]: claim["score"] for claim in response["claims"]}
        if set(qwen_claims) != set(gpt_claims):
            raise GPTJudgeScreenError("GPT and Qwen claim grids differ")
        for claim_id in qwen_claims:
            left = qwen_claims[claim_id]
            right = gpt_claims[claim_id]
            confusion[f"qwen_{left}_gpt_{right}"] += 1
            arm_disagreements[response["arm"]]["claims"] += 1
            arm_disagreements[response["arm"]]["disagreements"] += left != right
    if sum(confusion.values()) != 619:
        raise GPTJudgeScreenError("GPT/Qwen claim comparison is not the 619-claim grid")

    arms = {}
    for arm in (BASE_ARM, TREATMENT_ARM):
        qwen_arm = qwen_result["arms"][arm]
        gpt_arm = gpt_results[arm]
        arms[arm] = {
            "qwen_raw_lenient_wauc": qwen_arm["lenient_wauc"],
            "gpt_paper_style_lenient_wauc": gpt_arm["lenient_wauc"],
            "gpt_minus_qwen_wauc": (
                gpt_arm["lenient_wauc"] - qwen_arm["lenient_wauc"]
            ),
            "budgets": {
                budget: {
                    "qwen_raw_mean_lenient": qwen_arm["budgets"][budget][
                        "mean_lenient"
                    ],
                    "gpt_paper_style_mean_lenient": gpt_arm["budgets"][budget][
                        "mean_lenient"
                    ],
                    "gpt_minus_qwen": (
                        gpt_arm["budgets"][budget]["mean_lenient"]
                        - qwen_arm["budgets"][budget]["mean_lenient"]
                    ),
                }
                for budget in report.BUDGET_ORDER
            },
            "claim_label_comparison": arm_disagreements[arm],
        }
    qwen_gap = (
        qwen_result["arms"][TREATMENT_ARM]["lenient_wauc"]
        - qwen_result["arms"][BASE_ARM]["lenient_wauc"]
    )
    gpt_gap = (
        gpt_results[TREATMENT_ARM]["lenient_wauc"]
        - gpt_results[BASE_ARM]["lenient_wauc"]
    )
    return {
        "interpretation": (
            "grader-protocol sensitivity, not a model-only judge comparison; "
            "evidence, prompt order, system prompt, rationale elicitation, and "
            "reasoning policy differ"
        ),
        "arms": arms,
        "claim_confusion": confusion,
        "raw_qwen_treatment_minus_base_wauc": qwen_gap,
        "gpt_paper_style_treatment_minus_base_wauc": gpt_gap,
        "change_in_arm_gap_gpt_minus_qwen": gpt_gap - qwen_gap,
    }


async def run_prepared_screen(
    prepared: PreparedScreen,
    *,
    output_dir: Path,
    client_factory: Callable[[], Any] | None = None,
    provenance_revalidator: Callable[[dict[str, Any]], None] = _revalidate_provenance,
) -> Path:
    """Write intent, make the frozen request census, and write audit/result."""

    output_dir = Path(output_dir)
    intent_path = output_dir / "intent.json"
    audit_path = output_dir / "raw-audit.json"
    result_path = output_dir / "result.json"
    existing = [
        path
        for path in (intent_path, audit_path, result_path)
        if path.exists() or path.is_symlink()
    ]
    if existing:
        raise GPTJudgeScreenError(
            "GPT screen namespace is not fresh: " + ", ".join(map(str, existing))
        )
    write_immutable_json(intent_path, prepared.intent)
    intent_raw = read_artifact_bytes(intent_path)
    if intent_raw != canonical_json_bytes(prepared.intent):
        raise GPTJudgeScreenError("durable GPT intent failed verification")

    api_key = os.environ.get("OPENAI_API_KEY")
    if client_factory is None:
        if not api_key:
            raise GPTJudgeScreenError("OPENAI_API_KEY is missing after intent write")
        client_factory = lambda: AsyncOpenAI(
            timeout=600,
            max_retries=0,
            base_url=JUDGE_BASE_URL,
            api_key=api_key,
        )
    client = None
    responses: list[dict[str, Any]] = []
    terminal_error = None
    try:
        provenance_revalidator(prepared.intent)
    except Exception as exc:
        terminal_error = _error(exc)
    if terminal_error is None:
        try:
            client = client_factory()
            semaphore = asyncio.Semaphore(CONCURRENCY)
            responses = list(
                await asyncio.gather(
                    *(
                        _one_request(
                            request,
                            prepared.rows[request["qid"]],
                            client,
                            semaphore,
                        )
                        for request in prepared.intent["requests"]
                    )
                )
            )
        except BaseException as exc:
            terminal_error = _error(exc)
        finally:
            if client is not None:
                close = getattr(client, "close", None)
                if not callable(close):
                    close = getattr(client, "aclose", None)
                if callable(close):
                    try:
                        closed = close()
                        if inspect.isawaitable(closed):
                            await closed
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

    observed_models = sorted({
        item["response"]["response_model"]
        for item in responses
        if item.get("accepted") and isinstance(item.get("response"), dict)
    })
    if terminal_error is None and len(observed_models) != 1:
        terminal_error = {
            "type": "HeterogeneousJudgeModels",
            "message": f"expected one returned model identity, got {observed_models}",
        }
    arm_results: dict[str, Any] = {}
    populations: dict[str, dict[str, list[dict[str, Any]]]] = {}
    paired_bootstrap = None
    comparison = None
    if terminal_error is None:
        try:
            for arm in (BASE_ARM, TREATMENT_ARM):
                arm_results[arm], populations[arm] = raw_qwen._arm_summary(
                    arm, prepared.cells, responses
                )
            paired_bootstrap = _paired_bootstrap(
                populations[BASE_ARM], populations[TREATMENT_ARM]
            )
            comparison = _judge_protocol_comparison(arm_results, responses)
            provenance_revalidator(prepared.intent)
        except Exception as exc:
            terminal_error = _error(exc)

    response_fingerprints = sorted({
        item["response"]["system_fingerprint"]
        for item in responses
        if item.get("accepted")
        and isinstance(item.get("response"), dict)
        and item["response"].get("system_fingerprint") is not None
    })
    missing_fingerprints = sum(
        item.get("accepted")
        and isinstance(item.get("response"), dict)
        and item["response"].get("system_fingerprint") is None
        for item in responses
    )
    usage = {
        key: sum(
            item["response"]["usage"][key]
            for item in responses
            if item.get("accepted")
        )
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    audit = {
        "gpt_judge_audit_schema_version": AUDIT_SCHEMA_VERSION,
        "claim_ready": False,
        "diagnostic_only": True,
        "paper_comparison_allowed": False,
        "banner": BANNER,
        "intent_path": _display_path(intent_path),
        "intent_sha256": sha256_bytes(intent_raw),
        "expected_request_count": len(prepared.intent["requests"]),
        "request_count": len(responses),
        "accepted_count": sum(item["accepted"] for item in responses),
        "rejected_request_indices": rejected,
        "observed_response_models": observed_models,
        "observed_system_fingerprints": response_fingerprints,
        "missing_system_fingerprint_responses": missing_fingerprints,
        "accepted_usage_total": usage,
        "complete": terminal_error is None,
        "terminal_error": terminal_error,
        "responses": responses,
        "responses_sha256": sha256_json(responses),
    }
    write_immutable_json(audit_path, audit)
    audit_raw = read_artifact_bytes(audit_path)
    if audit_raw != canonical_json_bytes(audit):
        raise GPTJudgeScreenError("durable GPT audit failed verification")
    if terminal_error is not None:
        raise GPTJudgeScreenError(
            f"GPT screen failed closed; terminal audit: {_display_path(audit_path)}"
        )

    result = {
        "gpt_judge_screen_schema_version": SCREEN_SCHEMA_VERSION,
        "claim_ready": False,
        "diagnostic_only": True,
        "paper_comparison_allowed": False,
        "banner": BANNER,
        "task": TASK,
        "judge": prepared.intent["judge"],
        "estimand": prepared.intent["estimand"],
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
        "paired_bootstrap": paired_bootstrap,
        "judge_protocol_comparison": comparison,
    }
    write_immutable_json(result_path, result)
    result_raw = read_artifact_bytes(result_path)
    if result_raw != canonical_json_bytes(result):
        raise GPTJudgeScreenError("durable GPT result failed verification")
    return result_path


async def run_screen(
    *,
    base_run_id: str,
    treatment_run_id: str,
    screen_id: str,
    output_root: str | Path = "gpt-judge-screens",
) -> Path:
    try:
        screen_id = provenance.validate_id(screen_id, "screen ID")
    except (TypeError, ValueError) as exc:
        raise GPTJudgeScreenError(str(exc)) from exc
    root = Path(output_root)
    if not root.is_absolute():
        root = ROOT / root
    output_dir = root / screen_id
    with exclusive_process_lock(output_dir / ".lock"):
        unexpected = [path for path in output_dir.iterdir() if path.name != ".lock"]
        if unexpected:
            raise GPTJudgeScreenError("GPT screen output namespace is not fresh")
        prepared = prepare_screen(
            base_run_id=base_run_id,
            treatment_run_id=treatment_run_id,
        )
        return await run_prepared_screen(prepared, output_dir=output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run-id", required=True)
    parser.add_argument("--treatment-run-id", required=True)
    parser.add_argument("--screen-id", required=True)
    parser.add_argument("--output-dir", default="gpt-judge-screens")
    args = parser.parse_args()
    load_private_env(ROOT / ".env")
    try:
        path = asyncio.run(
            run_screen(
                base_run_id=args.base_run_id,
                treatment_run_id=args.treatment_run_id,
                screen_id=args.screen_id,
                output_root=args.output_dir,
            )
        )
    except (OSError, TypeError, ValueError, GPTJudgeScreenError) as exc:
        raise SystemExit(f"INTEGRITY ERROR: {exc}") from exc
    print(BANNER)
    print(f"immutable GPT sensitivity result: {_display_path(path)}")


if __name__ == "__main__":
    main()
