"""Fail-closed paired comparisons of two finalized research reports.

This module is intentionally downstream of :mod:`studybench.report`.  It does
not provide a second path around strict reporting: each arm must already have a
canonical, content-addressed report, and the complete underlying run and grade
population is reloaded before any statistic is computed.

The only supported intervention is a study note.  Treatment-minus-control
deltas use the exact manifest grid (intention to treat); ``no_answer`` remains
zero.  The two-stage bootstrap resamples questions, then rollout indices, using
the same sampled rollout indices in both arms.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
import random
from statistics import mean
from typing import Any

from .dataset import ROOT
from .grade import GradeIntegrityError, file_sha256, parse_json
from .integrity import (
    canonical_json_bytes,
    read_artifact_bytes,
    sha256_bytes,
    sha256_json,
    write_immutable_json,
)
from .preregistration import PreregistrationError, validate_preregistration
from .provenance import normalized_environment, validate_id
from . import report


COMPARISON_SCHEMA_VERSION = 4
INTERVENTION_KIND = "study-note"
_CLAIM_READY_JUDGE_REVISION_STATUSES = {
    "matched_complete_accepted_fingerprints_by_paired_cell",
    "not_applicable_no_judged_answers",
}
_CLAIM_READY_GENERATION_REVISION_STATUSES = {
    "matched_complete_provider_fingerprint_set",
}
BOOTSTRAP_METRICS = (
    "lenient",
    "len_cc",
    "strict",
    "tokens",
    "compile_rate",
    "no_answer_rate",
)
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
_STUDY_USAGE_KEYS = (
    "study_prompt_tokens",
    "study_generated_tokens",
    "study_total_tokens",
    "usage",
    "round_usage",
    "cumulative_usage",
    "round_usage_by_phase",
    "cumulative_usage_by_phase",
    "round_construction_usage",
    "cumulative_construction_usage",
    "round_construction_usage_by_phase",
    "cumulative_construction_usage_by_phase",
)


class ComparisonIntegrityError(RuntimeError):
    """The reports cannot support the requested paired research claim."""


@dataclass(frozen=True)
class LoadedArm:
    """One report whose exact underlying population has been revalidated."""

    report_path: Path
    report_sha256: str
    population: dict[str, list[dict[str, Any]]]
    audit: dict[str, Any]
    aggregate: dict[str, Any]
    study_usage: dict[str, Any]
    generation_retries: list[dict[str, Any]]
    judge_retries: list[dict[str, Any]]

    @property
    def spec(self) -> dict[str, Any]:
        return self.audit["run_manifest"]["spec"]

    @property
    def run_id(self) -> str:
        return self.spec["run_id"]

    @property
    def task(self) -> str:
        return self.spec["task"]


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except (OSError, ValueError):
        return str(path.resolve())


def _recorded_file(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ComparisonIntegrityError(f"{label} path is missing")
    logical = Path(value)
    if not logical.is_absolute() and any(
        part in ("", ".", "..") for part in logical.parts
    ):
        raise ComparisonIntegrityError(f"{label} path is not normalized")
    candidate = logical if logical.is_absolute() else ROOT / logical
    try:
        read_artifact_bytes(candidate)
    except (OSError, ValueError) as exc:
        raise ComparisonIntegrityError(f"{label} is missing: {candidate}") from exc
    return candidate.absolute()


def _report_digest(path: Path, artifact: dict[str, Any], raw: bytes) -> str:
    digest = sha256_bytes(raw)
    if raw != canonical_json_bytes(artifact):
        raise ComparisonIntegrityError("source report is not canonically encoded")
    expected_name = f"report-{digest}.json"
    if path.name != expected_name:
        raise ComparisonIntegrityError(
            f"source report is not content-addressed: expected filename {expected_name}"
        )
    return digest


def _population_roots(artifact: dict[str, Any]) -> tuple[Path, Path]:
    run_manifest = artifact.get("run_manifest")
    if not isinstance(run_manifest, dict):
        raise ComparisonIntegrityError("source report has no run manifest record")
    manifest_path = _recorded_file(
        run_manifest.get("path"), label="source run manifest"
    )
    task = artifact.get("task")
    run_id = artifact.get("run_id")
    if (
        not isinstance(task, str)
        or not task
        or not isinstance(run_id, str)
        or not run_id
        or manifest_path.name != "manifest.json"
        or manifest_path.parent.name != task
        or manifest_path.parent.parent.name != run_id
    ):
        raise ComparisonIntegrityError("source run-manifest path disagrees with the report")
    run_root = manifest_path.parent.parent

    records = artifact.get("population")
    if not isinstance(records, list) or not records:
        raise ComparisonIntegrityError("source report has an empty or invalid population")
    grade_roots: set[Path] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ComparisonIntegrityError("source population record is not an object")
        try:
            budget = record["budget"]
            rollout = record["rollout"]
            qid = record["qid"]
        except KeyError as exc:
            raise ComparisonIntegrityError("source population identity is incomplete") from exc
        if (
            not isinstance(budget, str)
            or type(rollout) is not int
            or rollout < 0
            or not isinstance(qid, str)
            or not qid
        ):
            raise ComparisonIntegrityError("source population identity is invalid")
        expected_tail = (task, budget, f"r{rollout}", f"{qid}.json")
        episode_path = _recorded_file(
            record.get("episode_path"), label="source episode"
        )
        grade_path = _recorded_file(record.get("grade_path"), label="source grade")
        if tuple(episode_path.parts[-4:]) != expected_tail:
            raise ComparisonIntegrityError("source episode path disagrees with its identity")
        if tuple(grade_path.parts[-4:]) != expected_tail:
            raise ComparisonIntegrityError("source grade path disagrees with its identity")
        if episode_path.parents[3] != run_root:
            raise ComparisonIntegrityError("source population mixes run roots")
        grade_roots.add(grade_path.parents[3])
    if len(grade_roots) != 1:
        raise ComparisonIntegrityError("source population mixes grade roots")
    grade_root = grade_roots.pop()
    if grade_root.parent.name != run_id:
        raise ComparisonIntegrityError("source grade root is not scoped to its run ID")
    return run_root, grade_root


def _validate_source_bootstrap(
    artifact: dict[str, Any], population: dict[str, list[dict[str, Any]]]
) -> None:
    bootstrap = artifact.get("bootstrap")
    if not isinstance(bootstrap, dict) or set(bootstrap) != {
        "replicates",
        "seed",
        "results",
    }:
        raise ComparisonIntegrityError("source report bootstrap record is invalid")
    replicates = bootstrap["replicates"]
    seed = bootstrap["seed"]
    if type(replicates) is not int or replicates < 0 or type(seed) is not int:
        raise ComparisonIntegrityError("source report bootstrap configuration is invalid")
    if replicates == 0:
        expected = None
    else:
        expected = report.bootstrap_population(population, replicates, seed)
    if bootstrap["results"] != expected:
        raise ComparisonIntegrityError("source report bootstrap no longer recomputes")


def _validate_usage_total(value: object, *, label: str) -> None:
    if not isinstance(value, dict):
        raise ComparisonIntegrityError(f"{label} is not an object")
    required = ("calls", "prompt_tokens", "generated_tokens", "total_tokens")
    for field in required:
        token = value.get(field)
        if type(token) is not int or token < 0:
            raise ComparisonIntegrityError(f"{label} has invalid {field}")
    if value["total_tokens"] != value["prompt_tokens"] + value["generated_tokens"]:
        raise ComparisonIntegrityError(f"{label} token totals are inconsistent")


def _study_usage(spec: dict[str, Any], run_root: Path) -> dict[str, Any]:
    note = spec.get("note")
    if note is None:
        return {
            "available": False,
            "note_present": False,
            "accounting_status": "not_applicable",
            "metadata": {},
        }
    if not isinstance(note, dict):
        raise ComparisonIntegrityError("run note record is invalid")
    construction_record = note.get("construction_manifest")
    if not isinstance(construction_record, dict):
        raise ComparisonIntegrityError("run note has no construction manifest")
    construction_path = _recorded_file(
        str(run_root / str(construction_record.get("snapshot", ""))),
        label="snapshotted note construction manifest",
    )
    try:
        construction_bytes = read_artifact_bytes(construction_path)
        construction = parse_json(
            construction_bytes, label="snapshotted note construction manifest"
        )
    except (OSError, GradeIntegrityError) as exc:
        raise ComparisonIntegrityError("note construction manifest is invalid") from exc
    if (
        not isinstance(construction, dict)
        or sha256_bytes(construction_bytes) != construction_record.get("sha256")
        or construction.get("note_sha256") != note.get("sha256")
    ):
        raise ComparisonIntegrityError("note construction manifest binding changed")

    metadata = {key: construction[key] for key in _STUDY_USAGE_KEYS if key in construction}
    scalar_keys = {
        "study_prompt_tokens",
        "study_generated_tokens",
        "study_total_tokens",
    }
    present_scalars = scalar_keys.intersection(metadata)
    if present_scalars:
        if present_scalars != scalar_keys:
            raise ComparisonIntegrityError("study manifest has a partial token ledger")
        for key in scalar_keys:
            if type(metadata[key]) is not int or metadata[key] < 0:
                raise ComparisonIntegrityError(f"study manifest has invalid {key}")
        if metadata["study_total_tokens"] != (
            metadata["study_prompt_tokens"] + metadata["study_generated_tokens"]
        ):
            raise ComparisonIntegrityError("study manifest token totals are inconsistent")
    for key, value in metadata.items():
        if key in scalar_keys:
            continue
        if key.endswith("_by_phase"):
            if not isinstance(value, dict):
                raise ComparisonIntegrityError(f"study manifest has invalid {key}")
            for phase, phase_total in value.items():
                if not isinstance(phase, str) or not phase:
                    raise ComparisonIntegrityError(f"study manifest has invalid {key} phase")
                _validate_usage_total(phase_total, label=f"{key}.{phase}")
        else:
            _validate_usage_total(value, label=key)
    return {
        "available": bool(metadata),
        "note_present": True,
        "accounting_status": "validated_construction_manifest_metadata",
        "note_sha256": note["sha256"],
        "construction_manifest_sha256": construction_record["sha256"],
        "study_id": construction.get("study_id"),
        "method": construction.get(
            "method",
            construction.get("config", {}).get("method")
            if isinstance(construction.get("config"), dict)
            else None,
        ),
        "metadata": metadata,
        "metadata_sha256": sha256_json(metadata),
    }


def load_source_report(path: str | Path) -> LoadedArm:
    """Reload and verify one canonical strict report and its entire population."""

    path = Path(path)
    try:
        resolved = _recorded_file(str(path.absolute()), label="source report")
        raw = read_artifact_bytes(resolved)
        artifact = parse_json(raw, label=f"source report {resolved}")
    except (OSError, ValueError, GradeIntegrityError) as exc:
        raise ComparisonIntegrityError(f"cannot load source report: {path}") from exc
    if not isinstance(artifact, dict):
        raise ComparisonIntegrityError("source report is not an immutable JSON object")
    if set(artifact) != _REPORT_KEYS:
        raise ComparisonIntegrityError("source report schema fields are incomplete or unknown")
    if (
        artifact.get("report_schema_version") != report.REPORT_SCHEMA_VERSION
        or artifact.get("claim_ready") is not True
        or artifact.get("budget_order") != report.BUDGET_ORDER
    ):
        raise ComparisonIntegrityError("source report is not claim-ready strict output")
    if artifact.get("paper_comparison") is not None:
        raise ComparisonIntegrityError(
            "paired local-arm comparison requires reports without a Table 1 comparison"
        )
    run_id = artifact.get("run_id")
    task = artifact.get("task")
    try:
        validate_id(run_id)
    except (TypeError, ValueError) as exc:
        raise ComparisonIntegrityError("source report has an invalid run ID") from exc
    if not isinstance(task, str) or not task:
        raise ComparisonIntegrityError("source report has an invalid task")
    digest = _report_digest(resolved, artifact, raw)
    expected_source = {"studybench/report.py": file_sha256(Path(report.__file__).resolve())}
    if artifact.get("report_source") != expected_source:
        raise ComparisonIntegrityError(
            "source report was produced by a different report implementation"
        )

    run_root, grade_root = _population_roots(artifact)
    grading = artifact.get("grading_manifest")
    config = grading.get("config") if isinstance(grading, dict) else None
    if not isinstance(config, dict):
        raise ComparisonIntegrityError("source report grading manifest is invalid")
    judge_model = config.get("judge_requested_model")
    whole_files = config.get("whole_files")
    effort = config.get("judge_effort")
    if (
        not isinstance(judge_model, str)
        or not judge_model
        or type(whole_files) is not bool
        or not isinstance(effort, str)
    ):
        raise ComparisonIntegrityError("source report grading configuration is invalid")

    run_specification = artifact["run_manifest"].get("spec")
    if not isinstance(run_specification, dict):
        raise ComparisonIntegrityError("source report run specification is invalid")
    try:
        population, audit = report.load_complete_evaluation(
            task,
            grade_root,
            run_root,
            rollouts=run_specification.get("rollouts"),
            judge_model=judge_model,
            whole_files=whole_files,
            effort=effort,
        )
    except (KeyError, TypeError, report.ReportIntegrityError) as exc:
        raise ComparisonIntegrityError(
            f"underlying source population failed strict revalidation: {exc}"
        ) from exc
    for key in _REPORT_AUDIT_KEYS:
        if artifact.get(key) != audit.get(key):
            raise ComparisonIntegrityError(
                f"source report {key} no longer matches the validated population"
            )
    aggregate = report.aggregate_population(population)
    if artifact.get("aggregate") != aggregate:
        raise ComparisonIntegrityError("source report aggregate no longer recomputes")
    _validate_source_bootstrap(artifact, population)
    generation_retries, judge_retries = _successful_retry_disclosures(
        population, audit
    )
    return LoadedArm(
        report_path=resolved,
        report_sha256=digest,
        population=population,
        audit=audit,
        aggregate=aggregate,
        study_usage=_study_usage(audit["run_manifest"]["spec"], run_root),
        generation_retries=generation_retries,
        judge_retries=judge_retries,
    )


def _leaf_differences(left: object, right: object, path: str = "") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        differences = []
        for key in sorted(set(left) | set(right)):
            child = f"{path}/{key}"
            if key not in left or key not in right:
                differences.append(child)
            else:
                differences.extend(_leaf_differences(left[key], right[key], child))
        return differences
    if left != right:
        return [path or "/"]
    return []


_INTERVENTION_PATH_PREFIXES = (
    "/run_id",
    "/note",
    "/prompt_policy/note_prefix_template",
    "/prompt_policy/presented_prompt_sha256",
    "/preregistration/role",
)
_ENVIRONMENT_NUISANCE_EXACT_PATHS = {
    "/environment/slurm_job_id",
    "/environment/server_launch_id",
    "/environment/vllm_api_key_sha256",
    "/environment/cuda_visible_devices",
    "/environment/vllm_environment/path",
    "/environment/vllm_runtime/path",
    "/environment/model_cache/path",
    "/environment/allocation/path",
    "/environment/allocation/sha256",
    "/environment/allocation/bytes",
    "/environment/allocation/inventory/hostname",
    "/environment/allocation/inventory/cuda_visible_devices",
    # _leaf_differences treats lists atomically; normalized equality above has
    # already proved that only per-row CUDA identifiers/UUIDs differ.
    "/environment/allocation/inventory/gpus",
}
_ENVIRONMENT_NUISANCE_POLICY = (
    "/environment/{slurm_job_id,server_launch_id,vllm_api_key_sha256,cuda_visible_devices}",
    "/environment/{vllm_environment,vllm_runtime,model_cache}/path",
    "/environment/allocation/{path,sha256,bytes}",
    "/environment/allocation/inventory/{hostname,cuda_visible_devices,slurm/*}",
    "/environment/allocation/inventory/gpus/*/{cuda_identifier,uuid}",
    "/environment/runner_allocation/*",
)


def _normalized_environment(environment: object) -> dict[str, Any]:
    """Apply the shared, explicit cross-allocation nuisance policy."""

    try:
        return normalized_environment(environment)
    except ValueError as exc:
        raise ComparisonIntegrityError("run environment is invalid") from exc


def _is_environment_nuisance_path(path: str) -> bool:
    if path in _ENVIRONMENT_NUISANCE_EXACT_PATHS:
        return True
    if path.startswith("/environment/allocation/inventory/slurm/"):
        return True
    if path.startswith("/environment/runner_allocation/"):
        return True
    return bool(
        path.startswith("/environment/allocation/inventory/gpus/")
        and path.rsplit("/", 1)[-1] in {"cuda_identifier", "uuid"}
    )


def _normalized_intervention_spec(spec: dict[str, Any]) -> dict[str, Any]:
    value = deepcopy(spec)
    prompt_policy = value.get("prompt_policy")
    if not isinstance(prompt_policy, dict):
        raise ComparisonIntegrityError("run prompt policy is invalid")
    value["environment"] = _normalized_environment(value.get("environment"))
    value["run_id"] = "<INTERVENTION-RUN-ID>"
    value["note"] = "<INTERVENTION-NOTE>"
    prompt_policy["note_prefix_template"] = "<INTERVENTION-NOTE-TEMPLATE>"
    prompt_policy["presented_prompt_sha256"] = "<INTERVENTION-PROMPTS>"
    preregistration = value.get("preregistration")
    if not isinstance(preregistration, dict) or preregistration.get("status") != "bound":
        raise ComparisonIntegrityError("run preregistration binding is invalid")
    preregistration["role"] = "<PREREGISTERED-ARM-ROLE>"
    return value


def _normalized_grading_contract(manifest: object) -> dict[str, Any]:
    """Return the pre-run grader contract without observed provider metadata."""

    if not isinstance(manifest, dict) or set(manifest) != {"sha256", "config"}:
        raise ComparisonIntegrityError("grading manifest is invalid")
    config = manifest.get("config")
    if (
        not isinstance(config, dict)
        or manifest.get("sha256") != sha256_json(config)
    ):
        raise ComparisonIntegrityError("grading manifest hash is inconsistent")
    value = deepcopy(config)
    for field in (
        "judge_response_models",
        "judge_system_fingerprints",
        "accepted_judge_system_fingerprint_by_episode",
        "missing_judge_system_fingerprint_calls",
    ):
        if field not in value:
            raise ComparisonIntegrityError("grading runtime disclosure is incomplete")
        value[field] = "<OBSERVED-JUDGE-RUNTIME>"
    return value


def _grade_map(population: dict[str, list[dict[str, Any]]]) -> dict[tuple[str, int, str], dict]:
    if set(population) != set(report.BUDGET_ORDER):
        raise ComparisonIntegrityError("population budget set is incomplete or unknown")
    mapped = {}
    for budget in report.BUDGET_ORDER:
        grades = population.get(budget)
        if not isinstance(grades, list):
            raise ComparisonIntegrityError(f"population has no {budget} grades")
        for grade in grades:
            if not isinstance(grade, dict):
                raise ComparisonIntegrityError("population grade is not an object")
            if grade.get("budget") != budget:
                raise ComparisonIntegrityError("population grade is in the wrong budget cell")
            key = (budget, grade.get("rollout"), grade.get("qid"))
            if type(key[1]) is not int or not isinstance(key[2], str):
                raise ComparisonIntegrityError("population grade identity is invalid")
            if key in mapped:
                raise ComparisonIntegrityError(f"duplicate population cell: {key}")
            if grade.get("episode_status") == "no_answer" and any(
                grade.get(field) != expected
                for field, expected in (
                    ("lenient", 0),
                    ("strict", 0),
                    ("cores_ok", False),
                )
            ):
                raise ComparisonIntegrityError("no_answer did not remain an ITT zero")
            mapped[key] = grade
    return mapped


def _accepted_judge_fingerprint_map(
    arm: LoadedArm,
) -> dict[tuple[str, int, str], str | None]:
    """Validate accepted-attempt fingerprints against the exact judged grid."""

    grading = arm.audit.get("grading_manifest", {}).get("config")
    raw = (
        grading.get("accepted_judge_system_fingerprint_by_episode")
        if isinstance(grading, dict)
        else None
    )
    if not isinstance(raw, dict):
        raise ComparisonIntegrityError(
            "report has no per-episode accepted judge fingerprint record"
        )

    grades = _grade_map(arm.population)
    expected_paths: dict[str, tuple[str, int, str]] = {}
    for key, grade in grades.items():
        status = grade.get("episode_status")
        if status not in {"ok", "no_answer"}:
            raise ComparisonIntegrityError("population has an invalid episode status")
        if status == "ok":
            budget, rollout, qid = key
            expected_paths[f"{budget}/r{rollout}/{qid}.json"] = key
    if set(raw) != set(expected_paths):
        raise ComparisonIntegrityError(
            "accepted judge fingerprint grid does not match judged answers"
        )

    mapped: dict[tuple[str, int, str], str | None] = {}
    for relative, key in expected_paths.items():
        fingerprint = raw[relative]
        if fingerprint is not None and (
            not isinstance(fingerprint, str) or not fingerprint
        ):
            raise ComparisonIntegrityError(
                "accepted judge fingerprint must be a nonempty string or null"
            )
        mapped[key] = fingerprint

    fingerprints = sorted({value for value in mapped.values() if value is not None})
    missing = sum(value is None for value in mapped.values())
    if (
        grading.get("judge_system_fingerprints") != fingerprints
        or grading.get("missing_judge_system_fingerprint_calls") != missing
    ):
        raise ComparisonIntegrityError(
            "accepted judge fingerprint summary disagrees with its per-episode record"
        )
    return mapped


def _judge_fingerprint_observation(
    mapping: dict[tuple[str, int, str], str | None],
    key: tuple[str, int, str],
) -> dict[str, Any]:
    if key not in mapping:
        return {"status": "not_applicable_no_answer", "system_fingerprint": None}
    fingerprint = mapping[key]
    return {
        "status": "available" if fingerprint is not None else "unavailable",
        "system_fingerprint": fingerprint,
    }


def _population_record_map(arm: LoadedArm) -> dict[tuple[str, int, str], dict]:
    mapped = {}
    for record in arm.audit["population"]:
        key = (record["budget"], record["rollout"], record["qid"])
        if key in mapped:
            raise ComparisonIntegrityError(f"duplicate population audit cell: {key}")
        mapped[key] = record
    return mapped


def _runtime_models(arm: LoadedArm) -> tuple[list[str], list[str]]:
    generation = arm.audit.get("generation_runtime")
    grading = arm.audit.get("grading_manifest", {}).get("config")
    if not isinstance(generation, dict) or not isinstance(grading, dict):
        raise ComparisonIntegrityError("report runtime identity is invalid")
    generation_models = generation.get("response_models")
    judge_models = grading.get("judge_response_models")
    generation_fingerprints = generation.get("system_fingerprints")
    judge_fingerprints = grading.get("judge_system_fingerprints")
    judge_fingerprint_scope = grading.get("judge_system_fingerprint_scope")
    missing_generation = generation.get("missing_system_fingerprint_calls")
    missing_judge = grading.get("missing_judge_system_fingerprint_calls")
    if (
        not isinstance(generation_models, list)
        or len(generation_models) != 1
        or not all(isinstance(model, str) and model for model in generation_models)
        or not isinstance(judge_models, list)
        or len(judge_models) > 1
        or not all(isinstance(model, str) and model for model in judge_models)
        or not isinstance(generation_fingerprints, list)
        or generation_fingerprints != sorted(set(generation_fingerprints))
        or not all(isinstance(value, str) and value for value in generation_fingerprints)
        or not isinstance(judge_fingerprints, list)
        or judge_fingerprints != sorted(set(judge_fingerprints))
        or not all(isinstance(value, str) and value for value in judge_fingerprints)
        or judge_fingerprint_scope != "accepted_final_attempts_only"
        or type(missing_generation) is not int
        or missing_generation < 0
        or type(missing_judge) is not int
        or missing_judge < 0
    ):
        raise ComparisonIntegrityError("report contains invalid model/fingerprint identity")
    if any(
        grade["episode_status"] == "ok"
        for grades in arm.population.values()
        for grade in grades
    ) and len(judge_models) != 1:
        raise ComparisonIntegrityError("graded answers have no homogeneous judge response model")
    return generation_models, judge_models


_BOUND_PREREGISTRATION_KEYS = {
    "schema_version",
    "status",
    "role",
    "source_path",
    "sha256",
    "bytes",
    "snapshot",
    "executed_source_commit",
    "document",
}


def _bound_preregistration(
    arm: LoadedArm, *, expected_role: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the embedded arm binding already rechecked by strict reporting."""

    spec = arm.spec
    record = spec.get("preregistration")
    if (
        spec.get("purpose") != "confirmatory"
        or spec.get("claim_ready") is not True
        or not isinstance(record, dict)
        or set(record) != _BOUND_PREREGISTRATION_KEYS
        or type(record.get("schema_version")) is not int
        or record["schema_version"] != 1
        or record.get("status") != "bound"
        or record.get("role") != expected_role
    ):
        raise ComparisonIntegrityError(
            f"{expected_role} report has no valid bound preregistration"
        )
    try:
        document = validate_preregistration(record.get("document"))
    except PreregistrationError as exc:
        raise ComparisonIntegrityError(
            f"{expected_role} preregistration is invalid: {exc}"
        ) from exc
    data = canonical_json_bytes(document)
    digest = sha256_bytes(data)
    expected_source = f"preregistrations/{document['preregistration_id']}.json"
    expected_snapshot = f"inputs/preregistration-{digest}.json"
    source = spec.get("source")
    note = spec.get("note")
    note_sha256 = note.get("sha256") if isinstance(note, dict) else None
    arm_contract = document["arms"][expected_role]
    if (
        record.get("sha256") != digest
        or type(record.get("bytes")) is not int
        or record["bytes"] != len(data)
        or record.get("source_path") != expected_source
        or record.get("snapshot") != expected_snapshot
        or not isinstance(source, dict)
        or record.get("executed_source_commit") != source.get("git_commit")
        or arm_contract.get("run_id") != arm.run_id
        or arm_contract.get("note_sha256") != note_sha256
    ):
        raise ComparisonIntegrityError(
            f"{expected_role} preregistration does not bind its report arm"
        )

    grading = arm.audit.get("grading_manifest", {}).get("config")
    if not isinstance(grading, dict):
        raise ComparisonIntegrityError("source report grading manifest is invalid")
    policy = document["grading_policy"]
    expected_grader = {"gpt-5.4": "openai", "fugu": "fugu"}.get(
        grading.get("judge_requested_model")
    )
    actual_policy = {
        "grader": expected_grader,
        "judge_model": grading.get("judge_requested_model"),
        "evidence_mode": (
            "whole_files" if grading.get("whole_files") is True
            else "excerpt_evidence"
        ),
        "judge_effort": grading.get("judge_effort"),
        "claim_scoring": "binary_0_1",
        "question_scoring": "weighted_claim_sum",
    }
    if actual_policy != policy:
        raise ComparisonIntegrityError(
            f"{expected_role} grading differs from its preregistration"
        )
    return record, document


def validate_pair(
    control: LoadedArm,
    treatment: LoadedArm,
    *,
    intervention_description: str,
) -> dict[str, Any]:
    """Require exact pairing and return the disclosed intervention record."""

    if (
        not isinstance(intervention_description, str)
        or not intervention_description.strip()
        or intervention_description != intervention_description.strip()
    ):
        raise ComparisonIntegrityError("intervention description must be nonempty and trimmed")
    if len(intervention_description) > 1000:
        raise ComparisonIntegrityError("intervention description is too long")
    if control.task != treatment.task:
        raise ComparisonIntegrityError("paired arms use different tasks")
    if control.run_id == treatment.run_id:
        raise ComparisonIntegrityError("control and treatment must use distinct run IDs")

    control_preregistration, control_document = _bound_preregistration(
        control, expected_role="control"
    )
    treatment_preregistration, treatment_document = _bound_preregistration(
        treatment, expected_role="treatment"
    )
    if canonical_json_bytes(control_document) != canonical_json_bytes(
        treatment_document
    ):
        raise ComparisonIntegrityError("paired arms use different preregistrations")
    if (
        control_preregistration["sha256"] != treatment_preregistration["sha256"]
        or intervention_description != control_document["intervention"]
    ):
        raise ComparisonIntegrityError(
            "comparison intervention differs from the shared preregistration"
        )

    control_spec = control.spec
    treatment_spec = treatment.spec
    control_seed = control_spec.get("seed_policy")
    treatment_seed = treatment_spec.get("seed_policy")
    seed_fields = ("master_seed",)
    if any(control_spec.get(field) != treatment_spec.get(field) for field in seed_fields):
        raise ComparisonIntegrityError("paired arms use different master seeds")
    if not isinstance(control_seed, dict) or not isinstance(treatment_seed, dict):
        raise ComparisonIntegrityError("paired seed policy is missing")
    for field in ("algorithm", "namespace", "seed_group", "ordered_parts", "episode_seeds"):
        if control_seed.get(field) != treatment_seed.get(field):
            raise ComparisonIntegrityError(f"paired arms use different seed-policy {field}")

    normalized_control = _normalized_intervention_spec(control_spec)
    normalized_treatment = _normalized_intervention_spec(treatment_spec)
    if normalized_control != normalized_treatment:
        differences = _leaf_differences(normalized_control, normalized_treatment)
        preview = ", ".join(differences[:10])
        raise ComparisonIntegrityError(
            "paired run specifications differ outside the disclosed note intervention"
            + (f": {preview}" if preview else "")
        )
    control_generation, control_judge = _runtime_models(control)
    treatment_generation, treatment_judge = _runtime_models(treatment)
    if control_generation != treatment_generation or control_judge != treatment_judge:
        raise ComparisonIntegrityError("paired arms resolved to different provider models")
    control_fingerprints = control.audit["generation_runtime"].get("system_fingerprints")
    treatment_fingerprints = treatment.audit["generation_runtime"].get(
        "system_fingerprints"
    )
    if control_fingerprints != treatment_fingerprints:
        raise ComparisonIntegrityError("paired arms have different generation fingerprints")

    control_grading = control.audit["grading_manifest"]["config"]
    treatment_grading = treatment.audit["grading_manifest"]["config"]
    control_grading_contract = _normalized_grading_contract(
        control.audit["grading_manifest"]
    )
    treatment_grading_contract = _normalized_grading_contract(
        treatment.audit["grading_manifest"]
    )
    if control_grading_contract != treatment_grading_contract:
        raise ComparisonIntegrityError(
            "paired arms use different grader, evidence mode, effort, or grading source"
        )

    control_grades = _grade_map(control.population)
    treatment_grades = _grade_map(treatment.population)
    control_records = _population_record_map(control)
    treatment_records = _population_record_map(treatment)
    if (
        set(control_grades) != set(treatment_grades)
        or set(control_records) != set(treatment_records)
        or set(control_grades) != set(control_records)
    ):
        raise ComparisonIntegrityError("paired arms do not contain the exact same grid")

    control_accepted_fingerprints = _accepted_judge_fingerprint_map(control)
    treatment_accepted_fingerprints = _accepted_judge_fingerprint_map(treatment)
    fingerprint_pairing = []
    known_mismatches = []
    for key in sorted(
        set(control_grades),
        key=lambda value: (report.BUDGET_ORDER.index(value[0]), value[1], value[2]),
    ):
        budget, rollout, qid = key
        control_observation = _judge_fingerprint_observation(
            control_accepted_fingerprints, key
        )
        treatment_observation = _judge_fingerprint_observation(
            treatment_accepted_fingerprints, key
        )
        record = {
            "budget": budget,
            "rollout": rollout,
            "qid": qid,
            "control": control_observation,
            "treatment": treatment_observation,
        }
        fingerprint_pairing.append(record)
        if (
            control_observation["status"] == "available"
            and treatment_observation["status"] == "available"
            and control_observation["system_fingerprint"]
            != treatment_observation["system_fingerprint"]
        ):
            known_mismatches.append(record)
    if known_mismatches:
        preview = ", ".join(
            f"{record['budget']}/r{record['rollout']}/{record['qid']}"
            for record in known_mismatches[:10]
        )
        raise ComparisonIntegrityError(
            "paired accepted judge fingerprints differ at " + preview
        )

    control_note = control_spec.get("note")
    treatment_note = treatment_spec.get("note")
    if not isinstance(treatment_note, dict):
        raise ComparisonIntegrityError("study-note treatment has no note")
    control_note_sha256 = (
        control_note.get("sha256") if isinstance(control_note, dict) else None
    )
    treatment_note_sha256 = treatment_note.get("sha256")
    if control_note_sha256 == treatment_note_sha256:
        raise ComparisonIntegrityError("study-note intervention does not change note bytes")
    control_prompts = control_spec["prompt_policy"].get("presented_prompt_sha256")
    treatment_prompts = treatment_spec["prompt_policy"].get("presented_prompt_sha256")
    if (
        not isinstance(control_prompts, dict)
        or not isinstance(treatment_prompts, dict)
        or set(control_prompts) != set(treatment_prompts)
        or any(control_prompts[qid] == treatment_prompts[qid] for qid in control_prompts)
    ):
        raise ComparisonIntegrityError(
            "study-note intervention does not change every presented question prompt"
        )

    observed = _leaf_differences(control_spec, treatment_spec)
    observed_intervention = [
        path for path in observed if path.startswith(_INTERVENTION_PATH_PREFIXES)
    ]
    observed_nuisance = [
        path for path in observed if _is_environment_nuisance_path(path)
    ]
    if len(observed_intervention) + len(observed_nuisance) != len(observed):
        raise ComparisonIntegrityError("an undisclosed manifest difference escaped validation")
    control_missing_generation = control.audit["generation_runtime"][
        "missing_system_fingerprint_calls"
    ]
    treatment_missing_generation = treatment.audit["generation_runtime"][
        "missing_system_fingerprint_calls"
    ]
    control_missing_judge = control_grading[
        "missing_judge_system_fingerprint_calls"
    ]
    treatment_missing_judge = treatment_grading[
        "missing_judge_system_fingerprint_calls"
    ]
    judge_fingerprints_complete = (
        control_missing_judge == 0 and treatment_missing_judge == 0
    )
    if (
        judge_fingerprints_complete
        and control_grading["judge_system_fingerprints"]
        != treatment_grading["judge_system_fingerprints"]
    ):
        raise ComparisonIntegrityError("paired arms have different judge fingerprints")
    if not control_accepted_fingerprints and not treatment_accepted_fingerprints:
        judge_revision_verification = "not_applicable_no_judged_answers"
    elif judge_fingerprints_complete:
        judge_revision_verification = (
            "matched_complete_accepted_fingerprints_by_paired_cell"
        )
    else:
        judge_revision_verification = (
            "accepted_provider_fingerprint_incomplete_and_disclosed"
        )
    return {
        "kind": INTERVENTION_KIND,
        "description": intervention_description,
        "direction": "treatment_minus_control",
        "preregistration_verification": "matched_bound_two_arm_contract",
        "preregistration": {
            "preregistration_id": control_document["preregistration_id"],
            "sha256": control_preregistration["sha256"],
            "hypothesis": control_document["hypothesis"],
            "intervention": control_document["intervention"],
            "grading_policy": control_document["grading_policy"],
            "analysis_policy": control_document["analysis_policy"],
            "stopping_policy": control_document["stopping_policy"],
        },
        "allowed_intervention_manifest_paths": list(_INTERVENTION_PATH_PREFIXES),
        "matched_environment_nuisance_policy": list(_ENVIRONMENT_NUISANCE_POLICY),
        "observed_manifest_leaf_paths": observed,
        "observed_intervention_leaf_paths": observed_intervention,
        "observed_environment_nuisance_leaf_paths": observed_nuisance,
        "control": {
            "run_id": control.run_id,
            "note_sha256": control_note_sha256,
            "note_prefix_template": control_spec["prompt_policy"].get(
                "note_prefix_template"
            ),
            "presented_prompt_sha256": control_prompts,
        },
        "treatment": {
            "run_id": treatment.run_id,
            "note_sha256": treatment_note_sha256,
            "note_prefix_template": treatment_spec["prompt_policy"].get(
                "note_prefix_template"
            ),
            "presented_prompt_sha256": treatment_prompts,
        },
        "matched_run_specification_sha256": sha256_json(normalized_control),
        "matched_grading_contract_sha256": sha256_json(control_grading_contract),
        "generation_revision_verification": (
            "matched_complete_provider_fingerprint_set"
            if control_fingerprints
            and control_missing_generation == 0
            and treatment_missing_generation == 0
            else "provider_fingerprint_incomplete_and_disclosed"
        ),
        "judge_revision_verification": judge_revision_verification,
        "accepted_judge_fingerprint_pairing": {
            "records": fingerprint_pairing,
            "sha256": sha256_json(fingerprint_pairing),
        },
        "provider_fingerprint_disclosure": {
            "generation": {
                "matched_system_fingerprints": control_fingerprints,
                "control_missing_calls": control_missing_generation,
                "treatment_missing_calls": treatment_missing_generation,
            },
            "judge": {
                "control_system_fingerprints": control_grading[
                    "judge_system_fingerprints"
                ],
                "treatment_system_fingerprints": treatment_grading[
                    "judge_system_fingerprints"
                ],
                "control_missing_calls": control_missing_judge,
                "treatment_missing_calls": treatment_missing_judge,
            },
        },
        "seed_pairing": {
            "master_seed": control_spec["master_seed"],
            "algorithm": control_seed["algorithm"],
            "namespace": control_seed["namespace"],
            "seed_group": control_seed["seed_group"],
            "episode_seeds_sha256": sha256_json(control_seed["episode_seeds"]),
        },
    }


def _cell_value(grade: dict[str, Any], metric: str) -> float:
    if metric == "len_cc":
        return float(grade["lenient"] if grade["cores_ok"] else 0)
    if metric == "tokens":
        value = grade["gen_tokens"]
        if type(value) is not int or value < 0:
            raise ComparisonIntegrityError("invalid grade metric tokens")
        return float(value)
    if metric == "compile_rate":
        value = grade["compile_check"]["compile_ok"]
        if type(value) is not bool:
            raise ComparisonIntegrityError("invalid grade metric compile_rate")
        return float(value)
    if metric == "no_answer_rate":
        return float(grade["episode_status"] == "no_answer")
    value = grade[metric]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ComparisonIntegrityError(f"invalid grade metric {metric}")
    return float(value)


def point_estimates(control: LoadedArm, treatment: LoadedArm) -> dict[str, Any]:
    """Compute exact-grid ITT arm values and treatment-minus-control deltas."""

    budgets = {}
    for budget in report.BUDGET_ORDER:
        control_budget = control.aggregate["budgets"][budget]
        treatment_budget = treatment.aggregate["budgets"][budget]
        if control_budget["n"] != treatment_budget["n"]:
            raise ComparisonIntegrityError("paired budget population counts differ")
        metrics = {
            "lenient": (control_budget["lenient"], treatment_budget["lenient"]),
            "len_cc": (control_budget["len_cc"], treatment_budget["len_cc"]),
            "strict": (control_budget["strict"], treatment_budget["strict"]),
            "tokens": (control_budget["tokens"], treatment_budget["tokens"]),
            "compile_rate": (
                control_budget["compile_rate"],
                treatment_budget["compile_rate"],
            ),
            "no_answer_rate": (
                control_budget["no_answer"] / control_budget["n"],
                treatment_budget["no_answer"] / treatment_budget["n"],
            ),
        }
        budgets[budget] = {
            "n_per_arm": control_budget["n"],
            "control_no_answer": control_budget["no_answer"],
            "treatment_no_answer": treatment_budget["no_answer"],
            **{
                metric: {
                    "control": values[0],
                    "treatment": values[1],
                    "treatment_minus_control": values[1] - values[0],
                }
                for metric, values in metrics.items()
            },
        }
    expertise_values = {}
    for metric in ("expertise_lenient", "expertise_strict"):
        left = control.aggregate[metric]
        right = treatment.aggregate[metric]
        expertise_values[metric] = {
            "control": left,
            "treatment": right,
            "treatment_minus_control": right - left,
        }
    return {"budgets": budgets, "expertise": expertise_values}


def _percentile_interval(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    lower = ordered[round(0.025 * (len(ordered) - 1))]
    upper = ordered[round(0.975 * (len(ordered) - 1))]
    return {"mean": mean(values), "lower_95": lower, "upper_95": upper}


def paired_bootstrap(
    control_population: dict[str, list[dict[str, Any]]],
    treatment_population: dict[str, list[dict[str, Any]]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Paired question/rollout bootstrap with shared rollout draws across arms."""

    if type(replicates) is not int or replicates <= 0:
        raise ComparisonIntegrityError("bootstrap replicates must be a positive integer")
    if type(seed) is not int:
        raise ComparisonIntegrityError("bootstrap seed must be an integer")
    control = _grade_map(control_population)
    treatment = _grade_map(treatment_population)
    if set(control) != set(treatment):
        raise ComparisonIntegrityError("bootstrap arms do not have identical cells")
    qids = sorted({key[2] for key in control})
    if not qids:
        raise ComparisonIntegrityError("cannot bootstrap an empty population")
    rollout_ids: dict[tuple[str, str], list[int]] = {}
    for budget in report.BUDGET_ORDER:
        for qid in qids:
            ids = sorted(
                rollout for b, rollout, q in control if b == budget and q == qid
            )
            if not ids or ids != list(range(len(ids))):
                raise ComparisonIntegrityError(
                    f"bootstrap cell {budget}/{qid} has an incomplete rollout grid"
                )
            rollout_ids[(budget, qid)] = ids
    counts = {len(ids) for ids in rollout_ids.values()}
    if len(counts) != 1:
        raise ComparisonIntegrityError("bootstrap cells have unequal rollout counts")

    rng = random.Random(seed)
    budget_samples = {
        budget: {metric: [] for metric in BOOTSTRAP_METRICS}
        for budget in report.BUDGET_ORDER
    }
    expertise_samples = {"expertise_lenient": [], "expertise_strict": []}
    for _ in range(replicates):
        sampled_questions = rng.choices(qids, k=len(qids))
        arm_points = {
            "control": {"lenient": [], "strict": []},
            "treatment": {"lenient": [], "strict": []},
        }
        for budget in report.BUDGET_ORDER:
            values = {
                "control": {metric: [] for metric in BOOTSTRAP_METRICS},
                "treatment": {metric: [] for metric in BOOTSTRAP_METRICS},
            }
            for qid in sampled_questions:
                available = rollout_ids[(budget, qid)]
                # This draw is deliberately shared across the two arms.
                sampled_rollouts = rng.choices(available, k=len(available))
                for rollout in sampled_rollouts:
                    key = (budget, rollout, qid)
                    for arm_name, grades in (
                        ("control", control),
                        ("treatment", treatment),
                    ):
                        grade = grades[key]
                        for metric in BOOTSTRAP_METRICS:
                            values[arm_name][metric].append(_cell_value(grade, metric))
            arm_means = {
                arm_name: {
                    metric: mean(metric_values)
                    for metric, metric_values in arm_values.items()
                }
                for arm_name, arm_values in values.items()
            }
            for metric in BOOTSTRAP_METRICS:
                budget_samples[budget][metric].append(
                    arm_means["treatment"][metric] - arm_means["control"][metric]
                )
            for arm_name in ("control", "treatment"):
                arm_points[arm_name]["lenient"].append(
                    (arm_means[arm_name]["tokens"], arm_means[arm_name]["lenient"])
                )
                arm_points[arm_name]["strict"].append(
                    (arm_means[arm_name]["tokens"], arm_means[arm_name]["strict"])
                )
        for score_kind, output_key in (
            ("lenient", "expertise_lenient"),
            ("strict", "expertise_strict"),
        ):
            expertise_samples[output_key].append(
                report.expertise(arm_points["treatment"][score_kind])
                - report.expertise(arm_points["control"][score_kind])
            )
    return {
        "budgets": {
            budget: {
                metric: _percentile_interval(samples)
                for metric, samples in metrics.items()
            }
            for budget, metrics in budget_samples.items()
        },
        "expertise": {
            metric: _percentile_interval(samples)
            for metric, samples in expertise_samples.items()
        },
    }


def _source_record(arm: LoadedArm) -> dict[str, Any]:
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
        "judge_runtime": {
            "requested_model": grading["judge_requested_model"],
            "response_models": grading["judge_response_models"],
            "system_fingerprint_scope": grading["judge_system_fingerprint_scope"],
            "system_fingerprints": grading["judge_system_fingerprints"],
            "accepted_system_fingerprint_by_episode": grading[
                "accepted_judge_system_fingerprint_by_episode"
            ],
            "missing_system_fingerprint_calls": grading[
                "missing_judge_system_fingerprint_calls"
            ],
        },
        "note_provenance": arm.audit["note_provenance"],
        "study_usage": arm.study_usage,
        "failed_generation_attempts": arm.audit["failed_attempts"],
        "failed_judge_attempts": arm.audit["failed_judge_audits"],
        "generation_transport_retries_before_final_episode": {
            "count": len(arm.generation_retries),
            "sha256": sha256_json(arm.generation_retries),
            "attempts": arm.generation_retries,
        },
        "rejected_judge_attempts_before_valid_grade": {
            "count": len(arm.judge_retries),
            "sha256": sha256_json(arm.judge_retries),
            "attempts": arm.judge_retries,
        },
    }


def _successful_retry_disclosures(
    population: dict[str, list[dict[str, Any]]],
    audit: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Inventory failed provider calls retained inside eventual final artifacts."""

    generation = []
    judges = []
    grades = _grade_map(population)
    for record in audit["population"]:
        episode_path = _recorded_file(record["episode_path"], label="source episode")
        grade_path = _recorded_file(record["grade_path"], label="source grade")
        try:
            episode_bytes = read_artifact_bytes(episode_path)
            if sha256_bytes(episode_bytes) != record["episode_sha256"]:
                raise ComparisonIntegrityError(
                    "source episode changed after strict report validation"
                )
            episode = parse_json(episode_bytes, label=f"episode {episode_path}")
            grade_bytes = read_artifact_bytes(grade_path)
            if sha256_bytes(grade_bytes) != record["grade_sha256"]:
                raise ComparisonIntegrityError(
                    "source grade changed after strict report validation"
                )
            stored_grade = parse_json(grade_bytes, label=f"grade {grade_path}")
        except (OSError, GradeIntegrityError) as exc:
            raise ComparisonIntegrityError("cannot inventory provider retries") from exc
        key = (record["budget"], record["rollout"], record["qid"])
        if stored_grade != grades[key]:
            raise ComparisonIntegrityError(
                "source grade changed after strict report validation"
            )
        attempts = episode.get("request_attempts", []) if isinstance(episode, dict) else []
        if not isinstance(attempts, list):
            raise ComparisonIntegrityError("episode request-attempt audit is invalid")
        for attempt in attempts:
            if not isinstance(attempt, dict):
                raise ComparisonIntegrityError("generation retry is not an object")
            if attempt.get("status") != "transport_error":
                continue
            generation.append({
                "task": record["task"],
                "qid": record["qid"],
                "budget": record["budget"],
                "rollout": record["rollout"],
                "episode_sha256": record["episode_sha256"],
                "logical_call": attempt.get("logical_call"),
                "attempt": attempt.get("attempt"),
                "request_sha256": attempt.get("request_sha256"),
                "error_type": attempt.get("error_type"),
                "error": attempt.get("error"),
                "usage": attempt.get("usage"),
            })
        judge_attempts = stored_grade.get("judge_attempts")
        if not isinstance(judge_attempts, list):
            raise ComparisonIntegrityError("grade judge-attempt audit is invalid")
        for attempt in judge_attempts:
            if not isinstance(attempt, dict):
                raise ComparisonIntegrityError("judge retry is not an object")
            if attempt.get("accepted") is not False:
                continue
            judges.append({
                "task": stored_grade["task"],
                "qid": stored_grade["qid"],
                "budget": stored_grade["budget"],
                "rollout": stored_grade["rollout"],
                "episode_sha256": stored_grade["episode_sha256"],
                "attempt": attempt.get("attempt"),
                "response_id": attempt.get("response_id"),
                "request_id": attempt.get("request_id"),
                "response_model": attempt.get("response_model"),
                "system_fingerprint": attempt.get("system_fingerprint"),
                "system_fingerprint_status": attempt.get(
                    "system_fingerprint_status"
                ),
                "system_fingerprint_observation": attempt.get(
                    "system_fingerprint_observation"
                ),
                "usage": attempt.get("usage"),
                "content_sha256": attempt.get("content_sha256"),
                "content_bytes": attempt.get("content_bytes"),
                "validation_error": attempt.get("validation_error"),
            })
    generation.sort(
        key=lambda item: (
            report.BUDGET_ORDER.index(item["budget"]),
            item["rollout"],
            item["qid"],
            item["logical_call"],
            item["attempt"],
        )
    )
    judges.sort(
        key=lambda item: (
            report.BUDGET_ORDER.index(item["budget"]),
            item["rollout"],
            item["qid"],
            item["attempt"],
        )
    )
    return generation, judges


def _pairing_records(control: LoadedArm, treatment: LoadedArm) -> list[dict[str, Any]]:
    control_grades = _grade_map(control.population)
    treatment_grades = _grade_map(treatment.population)
    control_records = _population_record_map(control)
    treatment_records = _population_record_map(treatment)
    seeds = control.spec["seed_policy"]["episode_seeds"]
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
            records.append(
                {
                    "task": control.task,
                    "qid": qid,
                    "question_sha256": question_hashes[qid],
                    "budget": budget,
                    "rollout": rollout,
                    "paired_seed": seeds[relative],
                    "control": {
                        "episode_sha256": control_records[key]["episode_sha256"],
                        "grade_sha256": control_records[key]["grade_sha256"],
                        "status": left["episode_status"],
                        "lenient": left["lenient"],
                        "strict": left["strict"],
                        "gen_tokens": left["gen_tokens"],
                    },
                    "treatment": {
                        "episode_sha256": treatment_records[key]["episode_sha256"],
                        "grade_sha256": treatment_records[key]["grade_sha256"],
                        "status": right["episode_status"],
                        "lenient": right["lenient"],
                        "strict": right["strict"],
                        "gen_tokens": right["gen_tokens"],
                    },
                }
            )
    return records


def build_comparison(
    control: LoadedArm,
    treatment: LoadedArm,
    *,
    intervention_description: str,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    intervention = validate_pair(
        control, treatment, intervention_description=intervention_description
    )
    analysis_policy = intervention["preregistration"]["analysis_policy"]
    if (
        bootstrap_replicates != analysis_policy["bootstrap_replicates"]
        or bootstrap_seed != analysis_policy["bootstrap_seed"]
    ):
        raise ComparisonIntegrityError(
            "bootstrap configuration differs from the preregistration"
        )
    pairing = _pairing_records(control, treatment)
    judge_revision_verified = (
        intervention["judge_revision_verification"]
        in _CLAIM_READY_JUDGE_REVISION_STATUSES
    )
    generation_revision_verified = (
        intervention["generation_revision_verification"]
        in _CLAIM_READY_GENERATION_REVISION_STATUSES
    )
    preregistration_verified = (
        intervention.get("preregistration_verification")
        == "matched_bound_two_arm_contract"
    )
    return {
        "comparison_schema_version": COMPARISON_SCHEMA_VERSION,
        "claim_ready": (
            generation_revision_verified
            and judge_revision_verified
            and preregistration_verified
        ),
        "task": control.task,
        "direction": "treatment_minus_control",
        "estimand": {
            "population": "all_manifest_planned_episodes_intention_to_treat",
            "model_no_answer": "zero",
            "infrastructure_failure": "excluded_only_after_successful_manifest_bound_retry",
            "judge_failure": "excluded_only_after_successful_manifest_bound_regrade",
            "study_tokens_in_expertise": False,
            "primary_score": "lenient",
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
        "point_estimates": point_estimates(control, treatment),
        "bootstrap": {
            "method": "paired_two_stage_question_then_rollout",
            "arm_pairing": "identical_sampled_rollout_indices",
            "confidence_interval": "percentile_95",
            "percentile_index": "round(p*(replicates-1))",
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "results": paired_bootstrap(
                control.population,
                treatment.population,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed,
            ),
        },
        "comparison_source": {
            "studybench/compare.py": file_sha256(Path(__file__).resolve()),
        },
    }


def write_comparison(
    artifact: dict[str, Any], *, output_root: str | Path = "comparisons"
) -> Path:
    """Write a deterministic content-addressed comparison artifact."""

    if artifact.get("comparison_schema_version") != COMPARISON_SCHEMA_VERSION:
        raise ComparisonIntegrityError("refusing to write an unknown comparison schema")
    intervention = artifact.get("intervention")
    revision_status = (
        intervention.get("judge_revision_verification")
        if isinstance(intervention, dict)
        else None
    )
    expected_claim_ready = revision_status in _CLAIM_READY_JUDGE_REVISION_STATUSES
    generation_status = (
        intervention.get("generation_revision_verification")
        if isinstance(intervention, dict)
        else None
    )
    expected_claim_ready = bool(
        expected_claim_ready
        and generation_status in _CLAIM_READY_GENERATION_REVISION_STATUSES
        and isinstance(intervention, dict)
        and intervention.get("preregistration_verification")
        == "matched_bound_two_arm_contract"
    )
    if artifact.get("claim_ready") is not expected_claim_ready:
        raise ComparisonIntegrityError(
            "comparison claim-ready status disagrees with its verification records"
        )
    preregistration = intervention.get("preregistration") if isinstance(
        intervention, dict
    ) else None
    analysis_policy = preregistration.get("analysis_policy") if isinstance(
        preregistration, dict
    ) else None
    bootstrap = artifact.get("bootstrap")
    if (
        not isinstance(analysis_policy, dict)
        or not isinstance(bootstrap, dict)
        or intervention.get("description") != preregistration.get("intervention")
        or bootstrap.get("replicates") != analysis_policy.get("bootstrap_replicates")
        or bootstrap.get("seed") != analysis_policy.get("bootstrap_seed")
    ):
        raise ComparisonIntegrityError(
            "comparison analysis differs from its preregistration"
        )
    try:
        sources = artifact["sources"]
        control_path = sources["control"]["report_path"]
        treatment_path = sources["treatment"]["report_path"]
        control = load_source_report(control_path)
        treatment = load_source_report(treatment_path)
        expected = build_comparison(
            control,
            treatment,
            intervention_description=intervention["description"],
            bootstrap_replicates=bootstrap["replicates"],
            bootstrap_seed=bootstrap["seed"],
        )
        matches = canonical_json_bytes(expected) == canonical_json_bytes(artifact)
    except (KeyError, TypeError, ValueError, GradeIntegrityError) as error:
        raise ComparisonIntegrityError(
            "comparison artifact cannot be independently recomputed"
        ) from error
    if not matches:
        raise ComparisonIntegrityError(
            "comparison artifact does not match independent recomputation"
        )
    control_id = validate_id(artifact["sources"]["control"]["run_id"])
    treatment_id = validate_id(artifact["sources"]["treatment"]["run_id"])
    task = artifact.get("task")
    if (
        not isinstance(task, str)
        or not task
        or task in {".", ".."}
        or Path(task).name != task
        or "/" in task
        or "\\" in task
    ):
        raise ComparisonIntegrityError("comparison task is invalid")
    digest = sha256_json(artifact)
    root = Path(output_root)
    if not root.is_absolute():
        root = ROOT / root
    path = (
        root
        / task
        / f"{control_id}--vs--{treatment_id}"
        / f"comparison-{digest}.json"
    )
    write_immutable_json(path, artifact)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-report", required=True)
    parser.add_argument("--treatment-report", required=True)
    parser.add_argument(
        "--intervention",
        required=True,
        choices=[INTERVENTION_KIND],
        help="the only manifest difference being tested",
    )
    parser.add_argument(
        "--intervention-description",
        required=True,
        help="concise, publication-facing description of the treatment minus control contrast",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--output-dir", default="comparisons")
    args = parser.parse_args()
    try:
        control = load_source_report(args.control_report)
        treatment = load_source_report(args.treatment_report)
        artifact = build_comparison(
            control,
            treatment,
            intervention_description=args.intervention_description,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
        )
        output = write_comparison(artifact, output_root=args.output_dir)
    except (ComparisonIntegrityError, GradeIntegrityError, KeyError, ValueError) as exc:
        raise SystemExit(f"INTEGRITY ERROR: {exc}") from exc

    readiness = "CLAIM-READY" if artifact["claim_ready"] else "DIAGNOSTIC ONLY"
    print(
        f"{artifact['task']}: {readiness} treatment-minus-control paired ITT "
        f"({args.bootstrap_replicates} bootstrap replicates)"
    )
    for budget in report.BUDGET_ORDER:
        point = artifact["point_estimates"]["budgets"][budget]["lenient"]
        interval = artifact["bootstrap"]["results"]["budgets"][budget]["lenient"]
        print(
            f"  {budget:8} lenient delta {point['treatment_minus_control']:6.2f} "
            f"[{interval['lower_95']:6.2f}, {interval['upper_95']:6.2f}]"
        )
    point = artifact["point_estimates"]["expertise"]["expertise_lenient"]
    interval = artifact["bootstrap"]["results"]["expertise"]["expertise_lenient"]
    print(
        f"  expertise lenient delta {point['treatment_minus_control']:6.2f} "
        f"[{interval['lower_95']:6.2f}, {interval['upper_95']:6.2f}]"
    )
    print(f"immutable comparison: {_display_path(output)}")


if __name__ == "__main__":
    main()
