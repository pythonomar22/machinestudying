"""Paired local-Qwen comparisons for exploratory method screening only.

This module is deliberately separate from :mod:`studybench.compare`.  It
accepts only complete, content-addressed local-proxy reports, reloads their
entire populations, and emits an immutable diagnostic artifact that can never
be claim-ready.  A promising screen still requires a fresh preregistered run
with an external grader before it can support a research claim.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import compare as strict_compare
from . import report
from . import sandbox
from .grade import (
    CURRENT_SMOKE_SOURCE_POLICY,
    GRADE_SCHEMA_VERSION,
    GradeIntegrityError,
    HISTORICAL_EXPLORATORY_SOURCE_POLICY,
    JUDGE_ATTEMPT_POLICY,
    LOCAL_GRADER_ENDPOINT_IDENTITY,
    LOCAL_GRADER_MODEL,
    LOCAL_GRADER_MODEL_REVISION,
    LOCAL_GRADER_RATIONALE_POLICY,
    LOCAL_GRADER_REQUEST_OPTIONS,
    LOCAL_GRADER_REQUEST_POLICY,
    LOCAL_GRADER_SERVER_ASSIGNMENT_POLICY,
    LOCAL_GRADER_VERDICT_CONTRACT,
    MAX_JUDGE_ATTEMPTS,
    file_sha256,
    parse_json,
    stable_sha256,
    validate_generation_source_validation,
)
from .integrity import (
    canonical_json_bytes,
    read_artifact_bytes,
    sha256_bytes,
    sha256_json,
    write_immutable_json,
)
from .provenance import (
    normalized_environment,
    validate_id,
    validate_local_server_urls,
    validate_server_assignment_record,
)


SCREEN_COMPARISON_SCHEMA_VERSION = 6
INTERVENTION_KIND = "study-note"
DIAGNOSTIC_BANNER = (
    "DIAGNOSTIC LOCAL-QWEN SCREEN ONLY — NOT CLAIM-READY; "
    "DO NOT USE FOR PAPER OR PUBLICATION CLAIMS"
)
LIMITATIONS = [
    "The local Qwen proxy is not an independent external grader; same-model or "
    "same-family self-preference and correlated errors can change method rankings.",
    "The percentile intervals cover only resampled benchmark questions and rollout "
    "indices; they omit judge uncertainty, grader calibration error, adaptive benchmark "
    "reuse, and every other systematic error.",
    "A 95% interval containing zero is inconclusive, not evidence of parity or "
    "equivalence. An interval excluding zero is still only a diagnostic screen.",
    "This public-benchmark screen is adaptive and diagnostic. A selected method needs "
    "a fresh preregistered paired run and external grading before any research claim.",
]
_REPORT_KEYS = {
    "report_schema_version",
    "claim_ready",
    "task",
    "run_id",
    "budget_order",
    "run_manifest",
    "generation_runtime",
    "note_provenance",
    "failed_attempts",
    "failed_judge_audits",
    "grading_manifest",
    "population",
    "population_sha256",
    "aggregate",
    "bootstrap",
    "paper_comparison",
    "report_source",
}
_REPORT_AUDIT_KEYS = (
    "run_manifest",
    "generation_runtime",
    "note_provenance",
    "failed_attempts",
    "failed_judge_audits",
    "grading_manifest",
    "population",
    "population_sha256",
)
_SCREEN_KEYS = {
    "screen_comparison_schema_version",
    "claim_ready",
    "diagnostic_only",
    "banner",
    "task",
    "direction",
    "estimand",
    "intervention",
    "sources",
    "pairing",
    "point_estimates",
    "bootstrap",
    "limitations",
    "screen_comparison_source",
}
_INTERVENTION_PATH_PREFIXES = (
    "/run_id",
    "/note",
    "/prompt_policy/note_prefix_template",
    "/prompt_policy/presented_prompt_sha256",
)
_SEED_ALGORITHM = "sha256-canonical-json-mod-2147483647"
_SEED_ORDERED_PARTS = [
    "master_seed",
    "namespace",
    "seed_group",
    "task",
    "qid",
    "budget",
    "rollout",
]


class ScreenComparisonIntegrityError(RuntimeError):
    """A local report pair cannot support even a diagnostic comparison."""


@dataclass(frozen=True)
class LoadedScreenArm:
    """One local report and its independently revalidated population."""

    report_path: Path
    report_sha256: str
    population: dict[str, list[dict[str, Any]]]
    audit: dict[str, Any]
    aggregate: dict[str, Any]
    checker_configuration: dict[str, Any]

    @property
    def spec(self) -> dict[str, Any]:
        return self.audit["run_manifest"]["spec"]

    @property
    def run_id(self) -> str:
        return self.spec["run_id"]

    @property
    def task(self) -> str:
        return self.spec["task"]


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(report.ROOT.resolve()).as_posix()
    except (OSError, ValueError):
        return str(path.resolve())


def _report_digest(path: Path, artifact: dict[str, Any], raw: bytes) -> str:
    digest = sha256_bytes(raw)
    if raw != canonical_json_bytes(artifact):
        raise ScreenComparisonIntegrityError("source report is not canonically encoded")
    if path.name != f"report-{digest}.json":
        raise ScreenComparisonIntegrityError("source report is not content-addressed")
    return digest


def _local_grading_config(artifact: dict[str, Any]) -> dict[str, Any]:
    grading = artifact.get("grading_manifest")
    config = grading.get("config") if isinstance(grading, dict) else None
    if (
        not isinstance(config, dict)
        or grading.get("sha256") != sha256_json(config)
    ):
        raise ScreenComparisonIntegrityError("local grading manifest is invalid")
    expected = {
        "claim_ready": False,
        "grading_tier": "diagnostic-local-proxy",
        "local_proxy": True,
        "judge_requested_model": LOCAL_GRADER_MODEL,
        "judge_attempt_policy": JUDGE_ATTEMPT_POLICY,
        "max_judge_attempts": MAX_JUDGE_ATTEMPTS,
        "judge_endpoint_identity": LOCAL_GRADER_ENDPOINT_IDENTITY,
        "judge_model_revision": LOCAL_GRADER_MODEL_REVISION,
        "judge_request_policy": LOCAL_GRADER_REQUEST_POLICY,
        "judge_verdict_contract": LOCAL_GRADER_VERDICT_CONTRACT,
        "judge_rationale_policy": LOCAL_GRADER_RATIONALE_POLICY,
        "judge_request_options": LOCAL_GRADER_REQUEST_OPTIONS,
        "judge_effort": "",
    }
    if any(config.get(field) != value for field, value in expected.items()):
        raise ScreenComparisonIntegrityError(
            "source report is not a pinned diagnostic local-Qwen report"
        )
    try:
        source_validation = validate_generation_source_validation(
            config.get("generation_source_validation")
        )
    except GradeIntegrityError as exc:
        raise ScreenComparisonIntegrityError(
            "source report has an invalid generation/grader source binding"
        ) from exc
    if source_validation["policy"] == CURRENT_SMOKE_SOURCE_POLICY:
        raise ScreenComparisonIntegrityError(
            "grading-smoke source binding is not valid for a complete local report"
        )
    if (
        source_validation["claim_ready"] is not False
        or source_validation["paper_comparison_allowed"] is not False
    ):
        raise ScreenComparisonIntegrityError(
            "local source-stage binding permits claim-ready or paper use"
        )
    if (
        config.get("grade_schema_version") != GRADE_SCHEMA_VERSION
        or type(config.get("whole_files")) is not bool
        or config.get("judge_system_fingerprint_scope")
        != "accepted_final_attempts_only"
        or not isinstance(config.get("judge_response_models"), list)
        or len(config["judge_response_models"]) > 1
        or not all(
            isinstance(value, str) and value
            for value in config["judge_response_models"]
        )
        or not isinstance(config.get("judge_system_fingerprints"), list)
        or not all(
            isinstance(value, str) and value
            for value in config["judge_system_fingerprints"]
        )
        or config["judge_system_fingerprints"]
        != sorted(set(config["judge_system_fingerprints"]))
        or not isinstance(
            config.get("accepted_judge_system_fingerprint_by_episode"), dict
        )
        or type(config.get("missing_judge_system_fingerprint_calls")) is not int
        or config["missing_judge_system_fingerprint_calls"] < 0
        or not isinstance(config.get("grading_spec_sha256_by_question"), dict)
    ):
        raise ScreenComparisonIntegrityError(
            "local report has an invalid grading runtime disclosure"
        )
    for field in ("grading_runtime", "local_judge_runtime"):
        record = config.get(field)
        try:
            valid = (
                isinstance(record, dict)
                and config.get(f"{field}_sha256") == sha256_json(record)
            )
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise ScreenComparisonIntegrityError(
                f"local report has invalid {field.replace('_', ' ')} provenance"
            )
    try:
        report._recorded_local_qualification_path(config)
    except report.ReportIntegrityError as exc:
        raise ScreenComparisonIntegrityError(
            "local report has invalid qualification provenance"
        ) from exc
    base_url = config.get("judge_base_url")
    validation_urls = config.get("judge_validation_urls")
    transport_urls = config.get("judge_transport_urls")
    contacted_urls = config.get("judge_contacted_urls")
    if (
        not isinstance(base_url, str)
        or not isinstance(validation_urls, list)
        or not validation_urls
        or not isinstance(transport_urls, list)
        or not transport_urls
        or not isinstance(contacted_urls, list)
        or not all(isinstance(value, str) and value for value in transport_urls)
        or not all(isinstance(value, str) and value for value in contacted_urls)
        or transport_urls != sorted(set(transport_urls))
        or contacted_urls != sorted(set(contacted_urls))
        or not set(contacted_urls).issubset(transport_urls)
    ):
        raise ScreenComparisonIntegrityError(
            "local report has an invalid loopback transport disclosure"
        )
    try:
        canonical_base_url = validate_local_server_urls(
            base_url, expected_count=1
        )[0]
        canonical_validation_urls = validate_local_server_urls(
            ",".join(validation_urls)
        )
        recorded = [
            validate_local_server_urls(value, expected_count=1)[0]
            for value in transport_urls
        ]
        contacted = [
            validate_local_server_urls(value, expected_count=1)[0]
            for value in contacted_urls
        ]
    except (TypeError, ValueError) as exc:
        raise ScreenComparisonIntegrityError(
            "local report contains a non-loopback judge transport"
        ) from exc
    if (
        canonical_base_url != base_url
        or canonical_validation_urls != validation_urls
        or recorded != transport_urls
        or contacted != contacted_urls
        or base_url != validation_urls[0]
    ):
        raise ScreenComparisonIntegrityError(
            "local report contains a non-canonical judge transport"
        )
    specification = artifact.get("run_manifest", {}).get("spec")
    expected_slots = (
        specification.get("server_assignment", {}).get("episode_slots")
        if isinstance(specification, dict) else None
    )
    run_server_count = (
        specification.get("server_assignment", {}).get("server_count")
        if isinstance(specification, dict) else None
    )
    transport_server_count = (
        specification.get("extra", {}).get("server_transport", {}).get(
            "server_count"
        )
        if isinstance(specification, dict) else None
    )
    runtime_server_count = config["local_judge_runtime"].get(
        "server", {}
    ).get("server_count")
    expected_assignment = {
        "policy": LOCAL_GRADER_SERVER_ASSIGNMENT_POLICY,
        "server_count": len(validation_urls),
        "source_field": "episode.server_slot",
        "server_slot_by_episode": expected_slots,
    }
    if (
        config.get("judge_server_assignment") != expected_assignment
        or runtime_server_count != len(validation_urls)
        or run_server_count != len(validation_urls)
        or transport_server_count != len(validation_urls)
    ):
        raise ScreenComparisonIntegrityError(
            "local report has an invalid manifest-bound judge assignment"
        )
    return config


def _validate_checker_binding(
    arm: LoadedScreenArm, *, expected_language: str | None = None
) -> bool:
    """Bind interpretation and every deterministic score to one checker."""

    configuration = arm.checker_configuration
    if (
        not isinstance(configuration, dict)
        or type(configuration.get("ready")) is not bool
    ):
        raise ScreenComparisonIntegrityError("checker configuration is invalid")
    config = _local_grading_config({
        "grading_manifest": arm.audit.get("grading_manifest"),
        "run_manifest": arm.audit.get("run_manifest"),
    })
    if config["generation_source_validation"]["generation_source"] != arm.spec.get(
        "source"
    ):
        raise ScreenComparisonIntegrityError(
            "source-stage record does not match the run manifest generation source"
        )
    interpretation = config.get("checker_interpretation")
    language = (
        expected_language
        if expected_language is not None
        else interpretation.get("language")
        if isinstance(interpretation, dict)
        else None
    )
    ready = configuration["ready"]
    expected = {
        "language": language,
        "sandbox_configuration_sha256": stable_sha256(configuration),
        "ready": ready,
        "score_interpretation": (
            "all-metrics"
            if ready
            else "lenient-and-core-conjunctive-checker-unavailable"
        ),
    }
    if (
        not isinstance(language, str)
        or not language
        or interpretation != expected
    ):
        raise ScreenComparisonIntegrityError(
            "local report checker interpretation is stale or invalid"
        )
    checker_sha256 = expected["sandbox_configuration_sha256"]
    if any(
        not isinstance(grade, dict)
        or not isinstance(grade.get("compile_check"), dict)
        or grade["compile_check"].get("configuration_sha256") != checker_sha256
        for grades in arm.population.values()
        for grade in grades
    ):
        raise ScreenComparisonIntegrityError(
            "source grades do not bind the declared checker configuration"
        )
    return ready


def _accepted_judge_runtime_map(
    arm: LoadedScreenArm,
) -> dict[tuple[str, int, str], dict[str, Any]]:
    """Validate and map only final accepted judge observations by cell."""

    config = _local_grading_config({
        "grading_manifest": arm.audit.get("grading_manifest"),
        "run_manifest": arm.audit.get("run_manifest"),
    })
    try:
        grades = strict_compare._grade_map(arm.population)
        accepted_fingerprints = (
            strict_compare._accepted_judge_fingerprint_map(arm)
        )
    except strict_compare.ComparisonIntegrityError as exc:
        raise ScreenComparisonIntegrityError(str(exc)) from exc
    response_model_values = [
        grade.get("judge_response_model")
        for grade in grades.values()
        if grade.get("episode_status") == "ok"
    ]
    if any(
        not isinstance(value, str) or not value
        for value in response_model_values
    ):
        raise ScreenComparisonIntegrityError(
            "local report has invalid accepted judge response models"
        )
    response_models = sorted(set(response_model_values))
    if (
        any(
            grade.get("episode_status") not in {"ok", "no_answer"}
            for grade in grades.values()
        )
        or any(
            grade.get("judge_response_model") is not None
            for grade in grades.values()
            if grade.get("episode_status") == "no_answer"
        )
        or response_models not in ([], [LOCAL_GRADER_MODEL])
        or config["judge_response_models"] != response_models
    ):
        raise ScreenComparisonIntegrityError(
            "local report has inconsistent accepted judge runtime provenance"
        )

    mapped = {}
    for key, grade in grades.items():
        if grade["episode_status"] == "no_answer":
            mapped[key] = {
                "status": "not_applicable_no_answer",
                "response_model": None,
                "system_fingerprint": None,
            }
            continue
        fingerprint = accepted_fingerprints[key]
        mapped[key] = {
            "status": "available" if fingerprint is not None else "unavailable",
            "response_model": grade["judge_response_model"],
            "system_fingerprint": fingerprint,
        }
    return mapped


def load_local_report(path: str | Path) -> LoadedScreenArm:
    """Load one immutable local report and revalidate every underlying artifact."""

    source = Path(path).absolute()
    try:
        raw = read_artifact_bytes(source)
        artifact = parse_json(raw, label=f"local source report {source}")
    except (OSError, ValueError, GradeIntegrityError) as exc:
        raise ScreenComparisonIntegrityError(
            f"cannot load local source report: {source}"
        ) from exc
    if not isinstance(artifact, dict) or set(artifact) != _REPORT_KEYS:
        raise ScreenComparisonIntegrityError(
            "local source report schema fields are incomplete or unknown"
        )
    if (
        artifact.get("report_schema_version") != report.REPORT_SCHEMA_VERSION
        or artifact.get("claim_ready") is not False
        or artifact.get("budget_order") != report.BUDGET_ORDER
        or artifact.get("paper_comparison") is not None
    ):
        raise ScreenComparisonIntegrityError(
            "source report is not diagnostic local-proxy output"
        )
    try:
        validate_id(artifact.get("run_id"))
    except (TypeError, ValueError) as exc:
        raise ScreenComparisonIntegrityError("source report has an invalid run ID") from exc
    task = artifact.get("task")
    if not isinstance(task, str) or task not in report.CORPORA:
        raise ScreenComparisonIntegrityError("source report has an invalid task")
    digest = _report_digest(source, artifact, raw)
    expected_source = {"studybench/report.py": file_sha256(Path(report.__file__).resolve())}
    if artifact.get("report_source") != expected_source:
        raise ScreenComparisonIntegrityError(
            "source report was produced by a different report implementation"
        )

    config = _local_grading_config(artifact)
    source_validation = config["generation_source_validation"]
    historical_source_commit = (
        source_validation["generation_source"]["git_commit"]
        if source_validation["policy"]
        == HISTORICAL_EXPLORATORY_SOURCE_POLICY
        else None
    )
    try:
        run_root, grade_root = strict_compare._population_roots(artifact)
    except strict_compare.ComparisonIntegrityError as exc:
        raise ScreenComparisonIntegrityError(str(exc)) from exc
    specification = artifact.get("run_manifest", {}).get("spec")
    if not isinstance(specification, dict):
        raise ScreenComparisonIntegrityError("source run specification is invalid")
    try:
        population, audit = report.revalidate_recorded_local_diagnostic_evaluation(
            task,
            grade_root,
            run_root,
            rollouts=specification.get("rollouts"),
            judge_base_url=",".join(config["judge_validation_urls"]),
            qualification_audit=report._recorded_local_qualification_path(
                config
            ),
            grading_runtime=config["grading_runtime"],
            local_judge_runtime=config["local_judge_runtime"],
            whole_files=config.get("whole_files", False),
            historical_exploratory_source_commit=historical_source_commit,
        )
    except (KeyError, TypeError, report.ReportIntegrityError) as exc:
        raise ScreenComparisonIntegrityError(
            f"underlying local population failed revalidation: {exc}"
        ) from exc
    for key in _REPORT_AUDIT_KEYS:
        if artifact.get(key) != audit.get(key):
            raise ScreenComparisonIntegrityError(
                f"source report {key} no longer matches the local population"
            )
    try:
        checker_configuration = sandbox.configuration_record(
            report.CORPORA[task].language
        )
    except (OSError, TypeError, ValueError) as exc:
        raise ScreenComparisonIntegrityError(
            "cannot reattest the deterministic checker configuration"
        ) from exc
    checker_ready = checker_configuration.get("ready") is True
    aggregate = report.aggregate_population(population)
    expected_report_aggregate = report.reportable_aggregate(
        aggregate, checker_ready=checker_ready
    )
    if artifact.get("aggregate") != expected_report_aggregate:
        raise ScreenComparisonIntegrityError(
            "source report aggregate no longer recomputes"
        )
    try:
        strict_compare._validate_source_bootstrap(
            artifact,
            population,
            checker_ready=checker_ready,
        )
    except strict_compare.ComparisonIntegrityError as exc:
        raise ScreenComparisonIntegrityError(str(exc)) from exc
    arm = LoadedScreenArm(
        report_path=source,
        report_sha256=digest,
        population=population,
        audit=audit,
        aggregate=aggregate,
        checker_configuration=checker_configuration,
    )
    _validate_checker_binding(
        arm, expected_language=report.CORPORA[task].language
    )
    _accepted_judge_runtime_map(arm)
    try:
        strict_compare._generation_identity_map(arm)
    except strict_compare.ComparisonIntegrityError as exc:
        raise ScreenComparisonIntegrityError(str(exc)) from exc
    _validate_exploratory_spec(arm)
    _validate_complete_grid(arm)
    return arm


def _path_is_or_contains(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(prefix + "/")


def _normalized_screen_spec(spec: dict[str, Any]) -> dict[str, Any]:
    value = deepcopy(spec)
    prompt_policy = value.get("prompt_policy")
    if not isinstance(prompt_policy, dict):
        raise ScreenComparisonIntegrityError("run prompt policy is invalid")
    try:
        value["environment"] = normalized_environment(value.get("environment"))
    except ValueError as exc:
        raise ScreenComparisonIntegrityError("run environment is invalid") from exc
    value["run_id"] = "<SCREEN-RUN-ID>"
    value["note"] = "<SCREEN-NOTE>"
    prompt_policy["note_prefix_template"] = "<SCREEN-NOTE-TEMPLATE>"
    prompt_policy["presented_prompt_sha256"] = "<SCREEN-PROMPTS>"
    return value


def _normalized_grading_contract(manifest: object) -> dict[str, Any]:
    if not isinstance(manifest, dict) or set(manifest) != {"sha256", "config"}:
        raise ScreenComparisonIntegrityError("grading manifest is invalid")
    config = manifest.get("config")
    if not isinstance(config, dict) or manifest.get("sha256") != sha256_json(config):
        raise ScreenComparisonIntegrityError("grading manifest hash is inconsistent")
    value = deepcopy(config)
    # Ports and endpoint lists are transport. Accepted response summaries are
    # outcomes: a no-answer arm never calls the judge. They are validated and
    # paired separately below. Requested model/revision/options, grader code,
    # runtime attestations, checker, and all other policy remain substantive.
    value["judge_base_url"] = "<LOOPBACK-TRANSPORT>"
    value["judge_validation_urls"] = "<LOOPBACK-VALIDATION-TRANSPORTS>"
    value["judge_transport_urls"] = "<LOOPBACK-TRANSPORTS>"
    value["judge_contacted_urls"] = "<LOOPBACK-CONTACTED-TRANSPORTS>"
    for field in (
        "judge_response_models",
        "judge_system_fingerprints",
        "accepted_judge_system_fingerprint_by_episode",
        "missing_judge_system_fingerprint_calls",
    ):
        value[field] = "<ACCEPTED-JUDGE-OUTCOMES>"
    intent_ledger = value.get("judge_attempt_intents")
    if (
        not isinstance(intent_ledger, dict)
        or set(intent_ledger) != {"policy", "count", "sha256", "artifacts"}
        or intent_ledger.get("policy") != JUDGE_ATTEMPT_POLICY
        or type(intent_ledger.get("count")) is not int
        or intent_ledger["count"] < 0
        or not isinstance(intent_ledger.get("artifacts"), list)
        or intent_ledger["count"] != len(intent_ledger["artifacts"])
        or intent_ledger.get("sha256") != sha256_json(intent_ledger["artifacts"])
    ):
        raise ScreenComparisonIntegrityError(
            "judge-attempt intent ledger is invalid")
    # The policy is substantive. Ledger paths, hashes, terminal outcomes, and
    # count are arm outcomes (including treatment-only no-answer cells) that
    # have already been fully revalidated while loading each report.
    value["judge_attempt_intents"] = {
        "policy": JUDGE_ATTEMPT_POLICY,
        "count": "<JUDGE-INTENT-OUTCOMES>",
        "sha256": "<JUDGE-INTENT-OUTCOMES>",
        "artifacts": "<JUDGE-INTENT-OUTCOMES>",
    }
    return value


def _validate_exploratory_spec(arm: LoadedScreenArm) -> None:
    spec = arm.spec
    preregistration = spec.get("preregistration")
    try:
        valid_run_id = validate_id(spec.get("run_id")) == spec.get("run_id")
    except (TypeError, ValueError):
        valid_run_id = False
    if (
        not valid_run_id
        or not isinstance(spec.get("task"), str)
        or not spec["task"]
        or spec.get("purpose") != "exploratory"
        or spec.get("claim_ready") is not False
        or not isinstance(preregistration, dict)
        or preregistration.get("status") != "not_provided"
        or preregistration.get("reason") != "exploratory"
        or spec.get("budgets") != report.BUDGET_ORDER
        or type(spec.get("master_seed")) is not int
        or type(spec.get("rollouts")) is not int
        or spec["rollouts"] <= 0
    ):
        raise ScreenComparisonIntegrityError(
            "local screen arm is smoke, partial, or not a complete exploratory run"
        )


def _validate_complete_grid(arm: LoadedScreenArm) -> None:
    """Reject a fabricated or partial in-memory arm before any statistic."""

    questions = arm.spec.get("questions")
    if (
        not isinstance(questions, list)
        or not questions
        or any(
            not isinstance(question, dict)
            or not isinstance(question.get("id"), str)
            or not question["id"]
            or not _is_sha256(question.get("sha256"))
            for question in questions
        )
    ):
        raise ScreenComparisonIntegrityError("local screen question grid is invalid")
    qids = [question["id"] for question in questions]
    if len(qids) != len(set(qids)):
        raise ScreenComparisonIntegrityError("local screen has duplicate question IDs")
    expected_paths = [
        f"{budget}/r{rollout}/{qid}.json"
        for budget in report.BUDGET_ORDER
        for rollout in range(arm.spec["rollouts"])
        for qid in qids
    ]
    extra = arm.spec.get("extra")
    server_transport = extra.get("server_transport") \
        if isinstance(extra, dict) else None
    server_count = server_transport.get("server_count") \
        if isinstance(server_transport, dict) else None
    try:
        validate_server_assignment_record(
            arm.spec.get("server_assignment"), expected_paths, server_count
        )
    except (TypeError, ValueError) as exc:
        raise ScreenComparisonIntegrityError(
            f"local screen server assignment is invalid: {exc}"
        ) from exc
    seed_policy = arm.spec.get("seed_policy")
    episode_seeds = (
        seed_policy.get("episode_seeds")
        if isinstance(seed_policy, dict)
        else None
    )
    if (
        arm.spec.get("expected_episodes") != expected_paths
        or not isinstance(seed_policy, dict)
        or seed_policy.get("algorithm") != _SEED_ALGORITHM
        or not isinstance(seed_policy.get("namespace"), str)
        or not seed_policy["namespace"]
        or not isinstance(seed_policy.get("seed_group"), str)
        or not seed_policy["seed_group"]
        or seed_policy.get("ordered_parts") != _SEED_ORDERED_PARTS
        or not isinstance(episode_seeds, dict)
        or set(episode_seeds) != set(expected_paths)
        or any(
            type(seed) is not int or seed < 0
            for seed in episode_seeds.values()
        )
    ):
        raise ScreenComparisonIntegrityError(
            "local screen manifest does not declare the complete benchmark grid"
        )
    expected_keys = {
        (budget, rollout, qid)
        for budget in report.BUDGET_ORDER
        for rollout in range(arm.spec["rollouts"])
        for qid in qids
    }
    try:
        observed_grades = set(strict_compare._grade_map(arm.population))
        observed_records = set(strict_compare._population_record_map(arm))
    except (KeyError, TypeError, strict_compare.ComparisonIntegrityError) as exc:
        raise ScreenComparisonIntegrityError(
            "local screen population record is invalid"
        ) from exc
    if observed_grades != expected_keys or observed_records != expected_keys:
        raise ScreenComparisonIntegrityError(
            "local screen population is partial or contains unexpected episodes"
        )
    if arm.aggregate != report.aggregate_population(arm.population):
        raise ScreenComparisonIntegrityError(
            "local screen aggregate does not recompute from its population"
        )


def _paired_judge_runtime(
    control: LoadedScreenArm, treatment: LoadedScreenArm
) -> dict[str, Any]:
    """Compare accepted judge identities only on jointly judged cells."""

    control_runtime = _accepted_judge_runtime_map(control)
    treatment_runtime = _accepted_judge_runtime_map(treatment)
    if set(control_runtime) != set(treatment_runtime):
        raise ScreenComparisonIntegrityError(
            "paired screens do not contain the same accepted-judge cell grid"
        )
    records = []
    model_mismatches = []
    fingerprint_mismatches = []
    jointly_judged = 0
    jointly_missing_fingerprint = 0
    one_arm_no_answer = 0
    for key in sorted(
        control_runtime,
        key=lambda value: (
            report.BUDGET_ORDER.index(value[0]),
            value[1],
            value[2],
        ),
    ):
        budget, rollout, qid = key
        left = control_runtime[key]
        right = treatment_runtime[key]
        record = {
            "budget": budget,
            "rollout": rollout,
            "qid": qid,
            "control": left,
            "treatment": right,
        }
        records.append(record)
        jointly_observed = (
            left["status"] != "not_applicable_no_answer"
            and right["status"] != "not_applicable_no_answer"
        )
        if not jointly_observed:
            one_arm_no_answer += (
                left["status"] == "not_applicable_no_answer"
            ) != (right["status"] == "not_applicable_no_answer")
            continue
        jointly_judged += 1
        if left["response_model"] != right["response_model"]:
            model_mismatches.append(record)
        if left["status"] != "available" or right["status"] != "available":
            jointly_missing_fingerprint += 1
        elif left["system_fingerprint"] != right["system_fingerprint"]:
            fingerprint_mismatches.append(record)

    def preview(rows: list[dict[str, Any]]) -> str:
        return ", ".join(
            f"{row['budget']}/r{row['rollout']}/{row['qid']}"
            for row in rows[:10]
        )

    if model_mismatches:
        raise ScreenComparisonIntegrityError(
            "paired accepted judge response models differ at "
            + preview(model_mismatches)
        )
    if fingerprint_mismatches:
        raise ScreenComparisonIntegrityError(
            "paired accepted judge fingerprints differ at "
            + preview(fingerprint_mismatches)
        )
    if jointly_judged == 0:
        verification = "not_applicable_no_jointly_judged_cells"
    elif jointly_missing_fingerprint:
        verification = "matched_models_fingerprint_incomplete_and_disclosed"
    else:
        verification = "matched_models_and_complete_fingerprints_by_joint_cell"
    return {
        "comparison_scope": "jointly_judged_cells_only",
        "verification": verification,
        "jointly_judged_cells": jointly_judged,
        "jointly_judged_cells_with_missing_fingerprint": (
            jointly_missing_fingerprint
        ),
        "cells_with_exactly_one_no_answer": one_arm_no_answer,
        "records": records,
        "sha256": sha256_json(records),
    }


def validate_pair(
    control: LoadedScreenArm,
    treatment: LoadedScreenArm,
    *,
    intervention_description: str,
) -> dict[str, Any]:
    """Require a paired exploratory note-only intervention."""

    if (
        not isinstance(intervention_description, str)
        or not intervention_description.strip()
        or intervention_description != intervention_description.strip()
        or len(intervention_description) > 1000
    ):
        raise ScreenComparisonIntegrityError(
            "intervention description must be trimmed and 1–1000 characters"
        )
    _validate_exploratory_spec(control)
    _validate_exploratory_spec(treatment)
    _validate_complete_grid(control)
    _validate_complete_grid(treatment)
    control_checker_ready = _validate_checker_binding(control)
    treatment_checker_ready = _validate_checker_binding(treatment)
    if control.task != treatment.task:
        raise ScreenComparisonIntegrityError("paired screens use different tasks")
    if control.run_id == treatment.run_id:
        raise ScreenComparisonIntegrityError("paired screens require distinct run IDs")
    judge_runtime_pairing = _paired_judge_runtime(control, treatment)

    control_seed = control.spec.get("seed_policy")
    treatment_seed = treatment.spec.get("seed_policy")
    if control.spec.get("master_seed") != treatment.spec.get("master_seed"):
        raise ScreenComparisonIntegrityError("paired screens use different master seeds")
    if not isinstance(control_seed, dict) or not isinstance(treatment_seed, dict):
        raise ScreenComparisonIntegrityError("paired seed policy is missing")
    for field in (
        "algorithm",
        "namespace",
        "seed_group",
        "ordered_parts",
        "episode_seeds",
    ):
        if control_seed.get(field) != treatment_seed.get(field):
            raise ScreenComparisonIntegrityError(
                f"paired screens use different seed-policy {field}"
            )

    normalized_control = _normalized_screen_spec(control.spec)
    normalized_treatment = _normalized_screen_spec(treatment.spec)
    if normalized_control != normalized_treatment:
        differences = strict_compare._leaf_differences(
            normalized_control, normalized_treatment
        )
        preview = ", ".join(differences[:10])
        raise ScreenComparisonIntegrityError(
            "paired run specifications differ outside the note intervention"
            + (f": {preview}" if preview else "")
        )

    control_grading = _local_grading_config({
        "grading_manifest": control.audit["grading_manifest"],
        "run_manifest": control.audit["run_manifest"],
    })
    treatment_grading = _local_grading_config({
        "grading_manifest": treatment.audit["grading_manifest"],
        "run_manifest": treatment.audit["run_manifest"],
    })
    if (
        control_grading["local_judge_qualification_sha256"]
        != treatment_grading["local_judge_qualification_sha256"]
        or control_grading["local_judge_qualification"]
        != treatment_grading["local_judge_qualification"]
    ):
        raise ScreenComparisonIntegrityError(
            "paired screens require the exact same local-judge qualification"
        )
    control_contract = _normalized_grading_contract(
        control.audit["grading_manifest"]
    )
    treatment_contract = _normalized_grading_contract(
        treatment.audit["grading_manifest"]
    )
    if control_contract != treatment_contract:
        differences = strict_compare._leaf_differences(
            control_contract, treatment_contract
        )
        preview = ", ".join(differences[:10])
        raise ScreenComparisonIntegrityError(
            "paired screens use different substantive grading contracts"
            + (f": {preview}" if preview else "")
        )

    try:
        generation_runtime_pairing = strict_compare._paired_generation_runtime(
            control, treatment
        )
    except strict_compare.ComparisonIntegrityError as exc:
        raise ScreenComparisonIntegrityError(str(exc)) from exc

    control_note = control.spec.get("note")
    treatment_note = treatment.spec.get("note")
    if control_note is not None and not isinstance(control_note, dict):
        raise ScreenComparisonIntegrityError("screen control has an invalid study note")
    if not isinstance(treatment_note, dict):
        raise ScreenComparisonIntegrityError("screen treatment has no study note")
    control_note_sha256 = (
        control_note.get("sha256") if isinstance(control_note, dict) else None
    )
    treatment_note_sha256 = treatment_note.get("sha256")
    invalid_control_note_hash = (
        control_note_sha256 is not None
        and not _is_sha256(control_note_sha256)
    )
    if (
        invalid_control_note_hash
        or not _is_sha256(treatment_note_sha256)
        or control_note_sha256 == treatment_note_sha256
    ):
        raise ScreenComparisonIntegrityError("screen intervention does not change note bytes")
    control_prompts = control.spec["prompt_policy"].get("presented_prompt_sha256")
    treatment_prompts = treatment.spec["prompt_policy"].get(
        "presented_prompt_sha256"
    )
    if (
        not isinstance(control_prompts, dict)
        or not isinstance(treatment_prompts, dict)
        or set(control_prompts) != set(treatment_prompts)
        or set(control_prompts)
        != {question["id"] for question in control.spec["questions"]}
        or not all(_is_sha256(value) for value in control_prompts.values())
        or not all(_is_sha256(value) for value in treatment_prompts.values())
        or any(
            control_prompts[qid] == treatment_prompts[qid]
            for qid in control_prompts
        )
    ):
        raise ScreenComparisonIntegrityError(
            "study-note intervention does not change every presented prompt"
        )

    observed = strict_compare._leaf_differences(control.spec, treatment.spec)
    observed_intervention = [
        path
        for path in observed
        if any(_path_is_or_contains(path, prefix) for prefix in _INTERVENTION_PATH_PREFIXES)
    ]
    observed_nuisance = [
        path for path in observed if strict_compare._is_environment_nuisance_path(path)
    ]
    if len(observed_intervention) + len(observed_nuisance) != len(observed):
        raise ScreenComparisonIntegrityError(
            "an undisclosed run-manifest difference escaped validation"
        )

    checker_control = control.checker_configuration
    checker_treatment = treatment.checker_configuration
    if (
        checker_control != checker_treatment
        or control_checker_ready != treatment_checker_ready
    ):
        raise ScreenComparisonIntegrityError(
            "paired screens use different deterministic checker configurations"
        )
    checker_ready = control_checker_ready
    return {
        "kind": INTERVENTION_KIND,
        "description": intervention_description,
        "direction": "treatment_minus_control",
        "allowed_intervention_manifest_paths": list(_INTERVENTION_PATH_PREFIXES),
        "matched_environment_nuisance_policy": list(
            strict_compare._ENVIRONMENT_NUISANCE_POLICY
        ),
        "observed_manifest_leaf_paths": observed,
        "observed_intervention_leaf_paths": observed_intervention,
        "observed_environment_nuisance_leaf_paths": observed_nuisance,
        "control": {
            "run_id": control.run_id,
            "note_sha256": control_note_sha256,
            "note_prefix_template": control.spec["prompt_policy"].get(
                "note_prefix_template"
            ),
            "presented_prompt_sha256": control_prompts,
        },
        "treatment": {
            "run_id": treatment.run_id,
            "note_sha256": treatment_note_sha256,
            "note_prefix_template": treatment.spec["prompt_policy"].get(
                "note_prefix_template"
            ),
            "presented_prompt_sha256": treatment_prompts,
        },
        "seed_pairing": {
            "master_seed": control.spec["master_seed"],
            "algorithm": control_seed["algorithm"],
            "namespace": control_seed["namespace"],
            "seed_group": control_seed["seed_group"],
            "episode_seeds_sha256": sha256_json(control_seed["episode_seeds"]),
        },
        "matched_run_specification_sha256": sha256_json(normalized_control),
        "matched_grading_contract_sha256": sha256_json(control_contract),
        "generation_runtime_pairing": generation_runtime_pairing,
        "accepted_judge_runtime_pairing": judge_runtime_pairing,
        "local_judge_qualification": {
            "policy": "exact-same-revalidated-passing-audit-v1",
            "audit_sha256": control_grading[
                "local_judge_qualification_sha256"
            ],
            "binding_sha256": control_grading[
                "local_judge_qualification"
            ]["binding_sha256"],
            "audit": control_grading["local_judge_qualification"]["audit"],
        },
        "grading_transport": {
            "policy": (
                "authenticated-loopback-validation-recorded-routes-and-contacts-"
                "are-transport-only"
            ),
            "control_validation_urls": control_grading[
                "judge_validation_urls"
            ],
            "control_recorded_route_urls": control_grading[
                "judge_transport_urls"
            ],
            "control_contacted_urls": control_grading[
                "judge_contacted_urls"
            ],
            "treatment_validation_urls": treatment_grading[
                "judge_validation_urls"
            ],
            "treatment_recorded_route_urls": treatment_grading[
                "judge_transport_urls"
            ],
            "treatment_contacted_urls": treatment_grading[
                "judge_contacted_urls"
            ],
            "server_assignment": control_grading["judge_server_assignment"],
            "matched_endpoint_identity": LOCAL_GRADER_ENDPOINT_IDENTITY,
        },
        "checker_interpretation": {
            "ready": checker_ready,
            "configuration_sha256": stable_sha256(checker_control),
            "configuration": checker_control,
            "score_policy": (
                "lenient_primary_with_strict_secondary"
                if checker_ready
                else "lenient_and_core_conjunctive_strict_and_compile_unavailable"
            ),
            "strict_and_compile_metrics": (
                "reported_as_secondary"
                if checker_ready
                else "omitted_checker_not_claim_ready"
            ),
        },
    }


def _source_record(arm: LoadedScreenArm) -> dict[str, Any]:
    grading = arm.audit["grading_manifest"]["config"]
    return {
        "run_id": arm.run_id,
        "report_path": _display_path(arm.report_path),
        "report_sha256": arm.report_sha256,
        "run_manifest_sha256": arm.audit["run_manifest"]["sha256"],
        "run_specification_sha256": arm.audit["run_manifest"]["spec_sha256"],
        "grading_manifest_sha256": arm.audit["grading_manifest"]["sha256"],
        "population_sha256": arm.audit["population_sha256"],
        "population_size": len(arm.audit["population"]),
        "generation_runtime": arm.audit["generation_runtime"],
        "source_stage": grading["generation_source_validation"],
        "judge_runtime": {
            "requested_model": grading["judge_requested_model"],
            "model_revision": grading["judge_model_revision"],
            "request_policy": grading["judge_request_policy"],
            "verdict_contract": grading["judge_verdict_contract"],
            "rationale_policy": grading["judge_rationale_policy"],
            "request_options": grading["judge_request_options"],
            "response_models": grading["judge_response_models"],
            "system_fingerprints": grading["judge_system_fingerprints"],
            "accepted_system_fingerprint_by_episode": grading[
                "accepted_judge_system_fingerprint_by_episode"
            ],
            "missing_system_fingerprint_calls": grading[
                "missing_judge_system_fingerprint_calls"
            ],
            "endpoint_identity": grading["judge_endpoint_identity"],
            "validation_urls": grading["judge_validation_urls"],
            "recorded_route_urls": grading["judge_transport_urls"],
            "contacted_urls": grading["judge_contacted_urls"],
            "server_assignment": grading["judge_server_assignment"],
            "grading_runtime_sha256": grading["grading_runtime_sha256"],
            "local_judge_runtime_sha256": grading[
                "local_judge_runtime_sha256"
            ],
            "local_judge_qualification_sha256": grading[
                "local_judge_qualification_sha256"
            ],
            "local_judge_qualification": grading[
                "local_judge_qualification"
            ],
            "checker_interpretation": grading["checker_interpretation"],
        },
        "note_provenance": arm.audit["note_provenance"],
        "failed_generation_attempts": arm.audit["failed_attempts"],
        "failed_judge_attempts": arm.audit["failed_judge_audits"],
    }


def _pairing_records(
    control: LoadedScreenArm, treatment: LoadedScreenArm
) -> list[dict[str, Any]]:
    control_grades = strict_compare._grade_map(control.population)
    treatment_grades = strict_compare._grade_map(treatment.population)
    control_records = strict_compare._population_record_map(control)
    treatment_records = strict_compare._population_record_map(treatment)
    seeds = control.spec["seed_policy"]["episode_seeds"]
    slots = control.spec["server_assignment"]["episode_slots"]
    question_hashes = {
        record["id"]: record["sha256"] for record in control.spec["questions"]
    }
    records = []
    for budget in report.BUDGET_ORDER:
        keys = sorted(
            (key for key in control_grades if key[0] == budget),
            key=lambda key: (key[1], key[2]),
        )
        for _, rollout, qid in keys:
            key = (budget, rollout, qid)
            relative = f"{budget}/r{rollout}/{qid}.json"
            left = control_grades[key]
            right = treatment_grades[key]
            records.append({
                "task": control.task,
                "qid": qid,
                "question_sha256": question_hashes[qid],
                "budget": budget,
                "rollout": rollout,
                "paired_seed": seeds[relative],
                "server_slot": slots[relative],
                "control": {
                    "episode_sha256": control_records[key]["episode_sha256"],
                    "grade_sha256": control_records[key]["grade_sha256"],
                    "status": left["episode_status"],
                    "lenient": left["lenient"],
                    "gen_tokens": left["gen_tokens"],
                },
                "treatment": {
                    "episode_sha256": treatment_records[key]["episode_sha256"],
                    "grade_sha256": treatment_records[key]["grade_sha256"],
                    "status": right["episode_status"],
                    "lenient": right["lenient"],
                    "gen_tokens": right["gen_tokens"],
                },
            })
    return records


def _project_point_estimates(
    full: dict[str, Any], *, checker_ready: bool
) -> dict[str, Any]:
    score_metrics = ["lenient", "len_cc"] + (["strict"] if checker_ready else [])
    operational_metrics = ["tokens", "no_answer_rate"]
    return {
        "budgets": {
            budget: {
                "n_per_arm": values["n_per_arm"],
                "control_no_answer": values["control_no_answer"],
                "treatment_no_answer": values["treatment_no_answer"],
                **{
                    metric: values[metric]
                    for metric in score_metrics + operational_metrics
                },
                **(
                    {"compile_rate": values["compile_rate"]}
                    if checker_ready
                    else {}
                ),
            }
            for budget, values in full["budgets"].items()
        },
        "expertise": {
            "expertise_lenient": full["expertise"]["expertise_lenient"],
            **(
                {"expertise_strict": full["expertise"]["expertise_strict"]}
                if checker_ready
                else {}
            ),
        },
    }


def _project_bootstrap(
    full: dict[str, Any], *, checker_ready: bool
) -> dict[str, Any]:
    score_metrics = ["lenient", "len_cc"] + (["strict"] if checker_ready else [])
    operational_metrics = ["tokens", "no_answer_rate"]
    return {
        "budgets": {
            budget: {
                metric: values[metric]
                for metric in score_metrics
                + operational_metrics
                + (["compile_rate"] if checker_ready else [])
            }
            for budget, values in full["budgets"].items()
        },
        "expertise": {
            "expertise_lenient": full["expertise"]["expertise_lenient"],
            **(
                {"expertise_strict": full["expertise"]["expertise_strict"]}
                if checker_ready
                else {}
            ),
        },
    }


def build_screen_comparison(
    control: LoadedScreenArm,
    treatment: LoadedScreenArm,
    *,
    intervention_description: str,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Build one deterministic, permanently non-claim-ready screen."""

    if type(bootstrap_replicates) is not int or bootstrap_replicates <= 0:
        raise ScreenComparisonIntegrityError(
            "bootstrap replicates must be a positive integer"
        )
    if type(bootstrap_seed) is not int:
        raise ScreenComparisonIntegrityError("bootstrap seed must be an integer")
    intervention = validate_pair(
        control,
        treatment,
        intervention_description=intervention_description,
    )
    checker_ready = intervention["checker_interpretation"]["ready"]
    pairing = _pairing_records(control, treatment)
    full_points = strict_compare.point_estimates(control, treatment)
    full_bootstrap = strict_compare.paired_bootstrap(
        control.population,
        treatment.population,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    return {
        "screen_comparison_schema_version": SCREEN_COMPARISON_SCHEMA_VERSION,
        "claim_ready": False,
        "diagnostic_only": True,
        "banner": DIAGNOSTIC_BANNER,
        "task": control.task,
        "direction": "treatment_minus_control",
        "estimand": {
            "population": "all_exploratory_manifest_episodes_intention_to_treat",
            "model_no_answer": "zero",
            "study_tokens_in_expertise": False,
            "primary_score": "lenient_local_qwen_proxy",
        },
        "intervention": intervention,
        "sources": {
            "control": _source_record(control),
            "treatment": _source_record(treatment),
        },
        "pairing": {
            "records": pairing,
            "sha256": sha256_json(pairing),
        },
        "point_estimates": _project_point_estimates(
            full_points, checker_ready=checker_ready
        ),
        "bootstrap": {
            "method": "paired_two_stage_question_then_rollout",
            "arm_pairing": "identical_sampled_rollout_indices",
            "confidence_interval": "percentile_95",
            "uncertainty_scope": (
                "question_and_rollout_resampling_only_judge_systematic_and_"
                "adaptive_reuse_uncertainty_omitted"
            ),
            "zero_in_interval_interpretation": (
                "inconclusive_not_parity_or_equivalence"
            ),
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "results": _project_bootstrap(
                full_bootstrap, checker_ready=checker_ready
            ),
        },
        "limitations": LIMITATIONS,
        "screen_comparison_source": {
            "studybench/screen_compare.py": file_sha256(Path(__file__).resolve()),
            "studybench/compare.py": file_sha256(
                Path(strict_compare.__file__).resolve()
            ),
            "studybench/report.py": file_sha256(Path(report.__file__).resolve()),
        },
    }


def write_screen_comparison(
    artifact: dict[str, Any], *, output_root: str | Path = "screen-comparisons"
) -> Path:
    """Recompute and immutably write a diagnostic comparison artifact."""

    if (
        not isinstance(artifact, dict)
        or set(artifact) != _SCREEN_KEYS
        or artifact.get("screen_comparison_schema_version")
        != SCREEN_COMPARISON_SCHEMA_VERSION
        or artifact.get("claim_ready") is not False
        or artifact.get("diagnostic_only") is not True
        or artifact.get("banner") != DIAGNOSTIC_BANNER
        or artifact.get("limitations") != LIMITATIONS
    ):
        raise ScreenComparisonIntegrityError(
            "refusing to write an invalid or claim-ready screen comparison"
        )
    try:
        sources = artifact["sources"]
        intervention = artifact["intervention"]
        bootstrap = artifact["bootstrap"]
        control = load_local_report(sources["control"]["report_path"])
        treatment = load_local_report(sources["treatment"]["report_path"])
        expected = build_screen_comparison(
            control,
            treatment,
            intervention_description=intervention["description"],
            bootstrap_replicates=bootstrap["replicates"],
            bootstrap_seed=bootstrap["seed"],
        )
    except (
        KeyError,
        TypeError,
        ValueError,
        GradeIntegrityError,
        ScreenComparisonIntegrityError,
    ) as exc:
        raise ScreenComparisonIntegrityError(
            "screen comparison cannot be independently recomputed"
        ) from exc
    if canonical_json_bytes(expected) != canonical_json_bytes(artifact):
        raise ScreenComparisonIntegrityError(
            "screen comparison differs from independent recomputation"
        )

    control_id = validate_id(artifact["sources"]["control"]["run_id"])
    treatment_id = validate_id(artifact["sources"]["treatment"]["run_id"])
    task = artifact.get("task")
    if (
        not isinstance(task, str)
        or not task
        or Path(task).name != task
        or "/" in task
        or "\\" in task
    ):
        raise ScreenComparisonIntegrityError("screen comparison task is invalid")
    root = Path(output_root)
    if not root.is_absolute():
        root = report.ROOT / root
    digest = sha256_json(artifact)
    path = (
        root
        / task
        / f"{control_id}--vs--{treatment_id}"
        / f"screen-comparison-{digest}.json"
    )
    write_immutable_json(path, artifact)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-report", required=True)
    parser.add_argument("--treatment-report", required=True)
    parser.add_argument(
        "--intervention-description",
        required=True,
        help="concise description of the treatment-minus-control note contrast",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--output-dir", default="screen-comparisons")
    args = parser.parse_args()
    try:
        control = load_local_report(args.control_report)
        treatment = load_local_report(args.treatment_report)
        artifact = build_screen_comparison(
            control,
            treatment,
            intervention_description=args.intervention_description,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
        )
        output = write_screen_comparison(artifact, output_root=args.output_dir)
    except (
        GradeIntegrityError,
        KeyError,
        ValueError,
        ScreenComparisonIntegrityError,
    ) as exc:
        raise SystemExit(f"INTEGRITY ERROR: {exc}") from exc

    print(DIAGNOSTIC_BANNER)
    print("Intervals omit judge and systematic error; same-model judge bias may dominate.")
    print("A 95% interval containing zero is inconclusive, not parity or equivalence.")
    for budget in report.BUDGET_ORDER:
        point = artifact["point_estimates"]["budgets"][budget]["lenient"]
        interval = artifact["bootstrap"]["results"]["budgets"][budget]["lenient"]
        print(
            f"  {budget:8} local lenient delta "
            f"{point['treatment_minus_control']:6.2f} "
            f"[{interval['lower_95']:6.2f}, {interval['upper_95']:6.2f}]"
        )
    point = artifact["point_estimates"]["expertise"]["expertise_lenient"]
    interval = artifact["bootstrap"]["results"]["expertise"][
        "expertise_lenient"
    ]
    print(
        f"  expertise local lenient delta {point['treatment_minus_control']:6.2f} "
        f"[{interval['lower_95']:6.2f}, {interval['upper_95']:6.2f}]"
    )
    print(f"immutable diagnostic comparison: {_display_path(output)}")


if __name__ == "__main__":
    main()
