"""Offline grading: sandbox compilation check, then GPT-5.4 rubric judge.

The judge prompt follows the paper's Appendix A.5 rubric-grading protocol. Claims
use the first-author correction of 0 or 1 only (the 0.5 partial-credit level was
removed because it increased variance), and the benchmark is described accurately
as source-grounded rather than private. Scores:

  lenient = weighted sum of claim scores (what Table 1 reports)
  strict  = 0 unless the compilation check passes AND every core claim scores 1;
            otherwise equal to the weighted sum

Writes grades/{run_id}/{judge}/{task}/{budget}/r{rollout}/{qid}.json for episodes
in runs/{run_id}/. Claim-ready grading requires an immutable run manifest; legacy
artifacts are preserved but are not silently mixed into new result populations.

The judge is selected by the GRADER_MODEL env var: "openai" (gpt-5.4, the paper's
grader — default) or "fugu" (Sakana API). Judge, evidence, and effort settings are
encoded in separate immutable grade namespaces so populations cannot mix.
"""

import argparse
import asyncio
import hashlib
import json
import logging
import math
import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from . import sandbox
from .dataset import CORPORA, ROOT, load_questions, read_pinned_code_bytes
from .env import load_private_env
from .human_audit import (
    HumanAuditError,
    validate_human_audit_protocol,
    validate_human_audit_result,
)
from .integrity import (canonical_json_bytes, exclusive_process_lock, read_artifact_bytes,
                        sha256_json, stable_seed, strict_json_loads, write_immutable_json)
from .preregistration import PreregistrationError, revalidate_run_preregistration
from .provenance import (
    environment_contract_is_valid,
    environment_is_claim_ready,
    validate_current_source,
    validate_environment_snapshot,
    validate_id,
)

CANONICAL_OPENAI_BASE_URL = "https://api.openai.com/v1"

GRADERS = {  # GRADER_MODEL env var -> (judge model id, base_url, api key env var)
    "openai": ("gpt-5.4", CANONICAL_OPENAI_BASE_URL, "OPENAI_API_KEY"),
    "fugu": ("fugu", "https://api.sakana.ai/v1", "SAKANA_API_KEY"),
}

GRADE_SCHEMA_VERSION = 4
MAX_JUDGE_ATTEMPTS = 2
FAILED_JUDGE_AUDIT_SCHEMA_VERSION = 2


class GradeIntegrityError(ValueError):
    """An episode, rubric, verdict, or stored grade is not safe to score."""


class JudgeAttemptsFailed(GradeIntegrityError):
    """No valid verdict was produced; carries a safe, non-verdict audit record."""

    def __init__(self, message: str, audit: dict[str, Any]):
        super().__init__(message)
        self.audit = audit


def grader_identity_for_model(judge_model: str) -> tuple[str, str]:
    """Return the unique configured grader name and its explicit API endpoint."""
    matches = [
        (grader, base_url)
        for grader, (model, base_url, _) in GRADERS.items()
        if model == judge_model
    ]
    if len(matches) != 1:
        raise GradeIntegrityError(
            f"judge model {judge_model!r} does not identify exactly one configured grader"
        )
    grader, base_url = matches[0]
    if not isinstance(base_url, str) or not base_url:
        raise GradeIntegrityError(f"grader {grader!r} has no explicit API endpoint")
    return grader, base_url


def _resolve_judge_base_url(
    judge_model: str, judge_base_url: str | None,
) -> str:
    """Resolve known models canonically; require explicit identity for test/local models."""
    configured = [
        base_url
        for model, base_url, _ in GRADERS.values()
        if model == judge_model
    ]
    if configured:
        if len(configured) != 1 or not isinstance(configured[0], str) or not configured[0]:
            raise GradeIntegrityError(
                f"judge model {judge_model!r} has an ambiguous or missing API endpoint"
            )
        if judge_base_url is not None and judge_base_url != configured[0]:
            raise GradeIntegrityError(
                f"judge endpoint does not match configured model {judge_model!r}"
            )
        return configured[0]
    if (
        not isinstance(judge_base_url, str)
        or not judge_base_url
        or judge_base_url != judge_base_url.strip()
    ):
        raise GradeIntegrityError(
            f"unconfigured judge model {judge_model!r} requires an explicit endpoint"
        )
    return judge_base_url


def _make_grader_client(grader: str, api_key: str) -> AsyncOpenAI:
    """Construct a grader client without consulting ambient SDK endpoint settings."""
    try:
        judge_model, judge_base_url, _ = GRADERS[grader]
    except KeyError as exc:
        raise GradeIntegrityError(f"unknown grader {grader!r}") from exc
    judge_base_url = _resolve_judge_base_url(judge_model, judge_base_url)
    return AsyncOpenAI(
        timeout=600,
        max_retries=0,
        base_url=judge_base_url,
        api_key=api_key,
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_bytes(read_artifact_bytes(path))


def stable_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode()
    return sha256_bytes(payload)


def parse_json(value: str | bytes, *, label: str) -> Any:
    """Parse strict JSON, rejecting duplicate keys and non-standard numbers."""
    try:
        return strict_json_loads(value, label=label)
    except ValueError as exc:
        raise GradeIntegrityError(f"invalid {label}: {exc}") from exc


def _safe_relative_file(root: Path, relative: object, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise GradeIntegrityError(f"{label} path is missing")
    logical = Path(relative)
    if logical.is_absolute() or any(part in ("", ".", "..") for part in logical.parts):
        raise GradeIntegrityError(f"{label} path is not normalized and relative")
    candidate = root / logical
    try:
        read_artifact_bytes(candidate)
    except (OSError, ValueError) as exc:
        raise GradeIntegrityError(
            f"{label} file is missing, non-regular, or reached through a symlink") from exc
    return candidate


def _valid_sha256(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _valid_git_commit(value: object) -> bool:
    return (isinstance(value, str) and len(value) in (40, 64)
            and all(character in "0123456789abcdef" for character in value))


def _path_component(value: object, *, label: str) -> str:
    if (not isinstance(value, str) or not value or value in {".", ".."}
            or Path(value).name != value or "/" in value or "\\" in value):
        raise GradeIntegrityError(f"invalid {label}: {value!r}")
    return value


def _validate_source_record(value: object, *, label: str) -> None:
    """Validate the exact clean-source record emitted by provenance.source_record."""
    if (not isinstance(value, dict)
            or set(value) != {"git_commit", "dirty", "files", "tree_sha256"}
            or value.get("dirty") is not False
            or not _valid_git_commit(value.get("git_commit"))):
        raise GradeIntegrityError(f"{label} source record is malformed or dirty")
    files = value.get("files")
    if (not isinstance(files, dict)
            or value.get("tree_sha256") != sha256_json(files)):
        raise GradeIntegrityError(f"{label} source record is malformed")
    for relative, record in files.items():
        logical = Path(relative) if isinstance(relative, str) else None
        if (logical is None or not relative or "\\" in relative or logical.is_absolute()
                or any(part in ("", ".", "..") for part in logical.parts)
                or not isinstance(record, dict)
                or set(record) != {"sha256", "bytes"}
                or not _valid_sha256(record.get("sha256"))
                or type(record.get("bytes")) is not int or record["bytes"] < 0):
            raise GradeIntegrityError(f"{label} source file record is malformed")


def _load_bundled_construction_dependencies(
    run_task_root: Path,
    provenance_bundle: dict,
    bundle_root: Path,
    inventory: object,
    inventory_sha256: object,
) -> dict[str, bytes]:
    """Validate and load an exact path-preserving construction snapshot."""
    construction_bundle = provenance_bundle.get("construction_artifacts")
    construction_root = bundle_root / "construction"
    if (not isinstance(inventory, dict) or not inventory
            or inventory_sha256 != sha256_json(inventory)
            or not isinstance(construction_bundle, dict)
            or set(construction_bundle) != {"root", "inventory_sha256", "artifacts"}
            or construction_bundle.get("root") != str(construction_root)
            or construction_bundle.get("inventory_sha256") != inventory_sha256):
        raise GradeIntegrityError(
            "bundled construction dependency inventory is missing or inconsistent")
    snapshots = construction_bundle.get("artifacts")
    if not isinstance(snapshots, dict) or set(snapshots) != set(inventory):
        raise GradeIntegrityError("bundled construction dependency set is incomplete")
    loaded = {}
    for raw_relative, source_artifact in inventory.items():
        relative = Path(raw_relative) if isinstance(raw_relative, str) else None
        if (relative is None or not raw_relative or "\\" in raw_relative
                or relative.is_absolute()
                or any(part in ("", ".", "..") for part in relative.parts)
                or not isinstance(source_artifact, dict)
                or set(source_artifact) != {"sha256", "bytes"}
                or not _valid_sha256(source_artifact.get("sha256"))
                or type(source_artifact.get("bytes")) is not int
                or source_artifact["bytes"] < 0):
            raise GradeIntegrityError(
                "construction dependency inventory contains an invalid record")
        snapshot_record = snapshots[raw_relative]
        expected_snapshot = construction_root / relative
        if (not isinstance(snapshot_record, dict)
                or set(snapshot_record) != {"sha256", "bytes", "snapshot"}
                or snapshot_record.get("sha256") != source_artifact["sha256"]
                or snapshot_record.get("bytes") != source_artifact["bytes"]
                or snapshot_record.get("snapshot") != str(expected_snapshot)):
            raise GradeIntegrityError(
                "bundled construction dependency record does not match its source")
        dependency_path = _safe_relative_file(
            run_task_root,
            snapshot_record["snapshot"],
            label=f"bundled construction dependency {raw_relative}",
        )
        dependency_bytes = read_artifact_bytes(dependency_path)
        if (sha256_bytes(dependency_bytes) != snapshot_record["sha256"]
                or len(dependency_bytes) != snapshot_record["bytes"]):
            raise GradeIntegrityError("bundled construction dependency bytes do not match")
        loaded[raw_relative] = dependency_bytes
    return loaded


def rubric_ids(row: dict) -> list[str]:
    rubric = row.get("rubric")
    if not isinstance(rubric, list) or not rubric:
        raise GradeIntegrityError(f"{row.get('id', '<unknown>')}: rubric is empty or invalid")
    ids = [claim.get("claim_id") for claim in rubric]
    if any(not isinstance(claim_id, str) or not claim_id for claim_id in ids):
        raise GradeIntegrityError(f"{row.get('id', '<unknown>')}: invalid rubric claim id")
    if len(ids) != len(set(ids)):
        raise GradeIntegrityError(f"{row.get('id', '<unknown>')}: duplicate rubric claim ids")
    if any(type(claim.get("weight")) is not int or claim["weight"] <= 0 for claim in rubric):
        raise GradeIntegrityError(f"{row.get('id', '<unknown>')}: invalid rubric weight")
    if sum(claim["weight"] for claim in rubric) != 100:
        raise GradeIntegrityError(f"{row.get('id', '<unknown>')}: rubric weights do not sum to 100")
    if any(claim.get("claim_type") not in {"core", "supporting"} for claim in rubric):
        raise GradeIntegrityError(f"{row.get('id', '<unknown>')}: invalid rubric claim type")
    if not any(claim["claim_type"] == "core" for claim in rubric):
        raise GradeIntegrityError(f"{row.get('id', '<unknown>')}: rubric has no core claim")
    return ids


def _validate_human_audit_integer_fields(
    construction: object,
    base: object,
    audit_result: object,
    audit_protocol: object,
) -> None:
    fields = (
        (construction, "round"),
        (base, "round"),
        (audit_result, "schema_version"),
        (audit_result, "round"),
        (audit_protocol, "schema_version"),
    )
    if any(
        not isinstance(record, dict) or type(record.get(field)) is not int
        for record, field in fields
    ):
        raise GradeIntegrityError(
            "bundled human-audit schema and round fields must be JSON integers")
    if (type(audit_result.get("blinding_preserved")) is not bool
            or type(audit_result.get("reviewer_independent")) is not bool):
        raise GradeIntegrityError(
            "bundled human-audit declarations must be JSON booleans")


def load_claim_manifest(run_task_root: Path, corpus, questions: list[dict]) -> dict:
    """Validate a claim-ready run manifest and its immutable note snapshot."""
    manifest_path = run_task_root / "manifest.json"
    try:
        manifest_bytes = read_artifact_bytes(manifest_path)
        manifest = parse_json(manifest_bytes, label="run manifest")
    except (OSError, ValueError, GradeIntegrityError) as exc:
        raise GradeIntegrityError(f"missing or invalid run manifest: {manifest_path}") from exc
    if manifest_bytes != canonical_json_bytes(manifest):
        raise GradeIntegrityError("run manifest is not canonically encoded")
    if (not isinstance(manifest, dict)
            or type(manifest.get("manifest_schema")) is not int
            or manifest["manifest_schema"] != 1):
        raise GradeIntegrityError("unknown run manifest schema")
    spec = manifest.get("spec")
    if (not isinstance(spec, dict)
            or type(spec.get("schema_version")) is not int
            or spec["schema_version"] != 1):
        raise GradeIntegrityError("unknown run specification schema")
    if spec.get("claim_ready") is not True or spec.get("purpose") != "confirmatory":
        raise GradeIntegrityError("run manifest is not claim-ready confirmatory research")
    if spec.get("task") != corpus.name:
        raise GradeIntegrityError("run manifest task does not match the requested corpus")
    if not isinstance(spec.get("run_id"), str) or not spec["run_id"]:
        raise GradeIntegrityError("run manifest has no run_id")
    try:
        preregistration = revalidate_run_preregistration(spec, run_task_root)
    except PreregistrationError as exc:
        raise GradeIntegrityError(
            f"run preregistration is invalid: {exc}"
        ) from exc
    extra = spec.get("extra")
    if (
        not isinstance(spec.get("model_revision"), str)
        or not spec["model_revision"]
        or not isinstance(extra, dict)
        or extra.get("model_revision") != spec["model_revision"]
        or not isinstance(extra.get("expected_response_model"), str)
        or not extra["expected_response_model"]
    ):
        raise GradeIntegrityError("run manifest model revision identity is incomplete")

    source = spec.get("source")
    _validate_source_record(source, label="run")
    try:
        validate_current_source(source)
    except ValueError as exc:
        raise GradeIntegrityError(str(exc)) from exc
    corpus_record = spec.get("corpus")
    if (not isinstance(corpus_record, dict) or corpus_record.get("dirty") is not False
            or corpus_record.get("name") != corpus.name):
        raise GradeIntegrityError("run corpus record is malformed or dirty")
    pinned_commit = getattr(corpus, "commit", None)
    if pinned_commit is None or corpus_record.get("commit") != pinned_commit:
        raise GradeIntegrityError("run corpus commit does not match the pinned corpus")
    expected_corpus_fields = {
        "roots": list(getattr(corpus, "roots", ())),
        "language": getattr(corpus, "language", None),
        "suffixes": sorted(getattr(corpus, "code_suffixes", ())),
    }
    for field, expected_value in expected_corpus_fields.items():
        if corpus_record.get(field) != expected_value:
            raise GradeIntegrityError(f"run corpus {field} does not match the pinned corpus")

    environment = spec.get("environment")
    if not isinstance(environment, dict) or not environment_is_claim_ready(environment):
        raise GradeIntegrityError("run environment is incomplete, inconsistent, or unpinned")
    if not environment_contract_is_valid(spec.get("environment_contract"), environment):
        raise GradeIntegrityError("run stable environment contract is invalid")

    if spec.get("question_bundle_sha256") != sha256_json(questions):
        raise GradeIntegrityError("run question bundle does not match the current dataset")
    question_records = spec.get("questions")
    if not isinstance(question_records, list):
        raise GradeIntegrityError("run manifest question records are invalid")
    for row in questions:
        _path_component(row.get("id"), label="question id")
    expected_questions = [
        {
            "id": row["id"],
            "sha256": sha256_json(row),
            "question_text_sha256": sha256_bytes(row["question"].encode("utf-8")),
        }
        for row in questions
    ]
    if question_records != expected_questions:
        raise GradeIntegrityError("run question records do not match the current dataset")

    budgets = spec.get("budgets")
    rollouts = spec.get("rollouts")
    if (not isinstance(budgets, list) or not budgets
            or len(budgets) != len(set(budgets))
            or type(rollouts) is not int or rollouts <= 0):
        raise GradeIntegrityError("run manifest has invalid budgets or rollouts")
    for budget in budgets:
        _path_component(budget, label="budget")
    expected = [
        f"{budget}/r{rollout}/{row['id']}.json"
        for budget in budgets
        for rollout in range(rollouts)
        for row in questions
    ]
    if spec.get("expected_episodes") != expected:
        raise GradeIntegrityError("run manifest expected_episodes is inconsistent")

    seed_policy = spec.get("seed_policy")
    expected_seed_parts = [
        "master_seed", "namespace", "seed_group", "task", "qid", "budget", "rollout"
    ]
    if (not isinstance(seed_policy, dict)
            or seed_policy.get("algorithm") != "sha256-canonical-json-mod-2147483647"
            or seed_policy.get("ordered_parts") != expected_seed_parts
            or not isinstance(seed_policy.get("namespace"), str)
            or not seed_policy["namespace"]
            or not isinstance(seed_policy.get("seed_group"), str)
            or not seed_policy["seed_group"]
            or not isinstance(seed_policy.get("episode_seeds"), dict)
            or type(spec.get("master_seed")) is not int):
        raise GradeIntegrityError("run seed policy is invalid")
    expected_seeds = {}
    for relative in expected:
        budget, rollout_dir, filename = relative.split("/")
        qid = filename.removesuffix(".json")
        rollout = int(rollout_dir.removeprefix("r"))
        expected_seeds[relative] = stable_seed(
            spec["master_seed"], seed_policy["namespace"], seed_policy["seed_group"],
            spec["task"], qid, budget, rollout,
        )
    if seed_policy["episode_seeds"] != expected_seeds:
        raise GradeIntegrityError("run episode seeds do not match the declared seed policy")

    note_record = spec.get("note")
    note = ""
    note_sha256 = None
    note_manifest = None
    if note_record is not None:
        if not isinstance(note_record, dict):
            raise GradeIntegrityError("run note record is invalid")
        note_path = _safe_relative_file(
            run_task_root, note_record.get("snapshot"), label="run note snapshot")
        try:
            note_bytes = read_artifact_bytes(note_path)
            note = note_bytes.decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise GradeIntegrityError("run note snapshot is missing or invalid") from exc
        note_sha256 = sha256_bytes(note_bytes)
        if (note_record.get("sha256") != note_sha256
                or type(note_record.get("bytes")) is not int
                or note_record.get("bytes") != len(note_bytes)):
            raise GradeIntegrityError("run note snapshot does not match its manifest record")
        if not note.strip():
            raise GradeIntegrityError("run note snapshot is empty")

        construction_record = note_record.get("construction_manifest")
        if not isinstance(construction_record, dict):
            raise GradeIntegrityError("research note has no construction manifest")
        construction_path = _safe_relative_file(
            run_task_root,
            construction_record.get("snapshot"),
            label="note construction manifest",
        )
        construction_bytes = read_artifact_bytes(construction_path)
        if construction_record.get("sha256") != sha256_bytes(construction_bytes):
            raise GradeIntegrityError("note construction manifest hash does not match")
        construction = parse_json(
            construction_bytes, label="note construction manifest")
        if not isinstance(construction, dict):
            raise GradeIntegrityError("note construction manifest is not an object")
        note_manifest = construction
        construction_claim_ready = construction.get("claim_ready")
        if construction_claim_ready is None and isinstance(construction.get("config"), dict):
            construction_claim_ready = construction["config"].get("claim_ready")
        if construction_claim_ready is not True:
            raise GradeIntegrityError("note construction manifest is not claim-ready")
        if construction.get("note_sha256") != note_sha256:
            raise GradeIntegrityError("note construction manifest names different note bytes")
        if construction.get("task") != corpus.name:
            raise GradeIntegrityError("note construction task does not match the run")
        if construction.get("corpus_commit") != pinned_commit:
            raise GradeIntegrityError("note construction corpus does not match the run")
        if not isinstance(construction.get("study_id"), str) or not construction["study_id"]:
            raise GradeIntegrityError("note construction manifest has no study ID")

        manifest_type = construction.get("manifest_type")
        if manifest_type == "human-audited-note":
            if "automated_readiness" not in construction:
                raise GradeIntegrityError(
                    "human-audited note has no automated readiness record")
            human = construction.get("human_audit")
            base_record = construction.get("construction_manifest")
            bundle = note_record.get("provenance_bundle")
            if (not isinstance(human, dict) or human.get("status") != "passed"
                    or not isinstance(base_record, dict) or not isinstance(bundle, dict)):
                raise GradeIntegrityError("human-audited note has no passing bundled audit chain")
            bundle_root = Path(str(bundle.get("root", "")))
            if (bundle_root.is_absolute() or not bundle_root.parts
                    or any(part in ("", ".", "..") for part in bundle_root.parts)):
                raise GradeIntegrityError("note provenance bundle has an unsafe root")

            bundled_manifest = _safe_relative_file(
                run_task_root, bundle.get("manifest_snapshot"),
                label="bundled audited-note manifest")
            bundled_note = _safe_relative_file(
                run_task_root, bundle.get("note_snapshot"),
                label="bundled audited note")
            if (read_artifact_bytes(bundled_manifest) != construction_bytes
                    or read_artifact_bytes(bundled_note) != note_bytes):
                raise GradeIntegrityError("note provenance bundle changed its manifest or note")

            artifact_records = bundle.get("artifacts")
            expected_artifacts = {
                "construction_manifest": (
                    base_record, "path", "sha256"),
                "human_audit_result": (
                    human, "result_path", "result_sha256"),
                "human_audit_protocol": (
                    human, "protocol_path", "protocol_sha256"),
            }
            if (not isinstance(artifact_records, dict)
                    or set(artifact_records) != set(expected_artifacts)):
                raise GradeIntegrityError("note provenance bundle artifact set is incomplete")
            loaded = {}
            for label, (source_record, path_key, hash_key) in expected_artifacts.items():
                relative_source = Path(str(source_record.get(path_key, "")))
                record = artifact_records[label]
                expected_snapshot = bundle_root / relative_source
                if (not isinstance(record, dict)
                        or record.get("snapshot") != str(expected_snapshot)
                        or record.get("sha256") != source_record.get(hash_key)):
                    raise GradeIntegrityError(f"bundled {label} record does not match the audit")
                artifact_path = _safe_relative_file(
                    run_task_root, record["snapshot"], label=f"bundled {label}")
                artifact_bytes = read_artifact_bytes(artifact_path)
                if sha256_bytes(artifact_bytes) != record["sha256"]:
                    raise GradeIntegrityError(f"bundled {label} hash does not match")
                loaded[label] = parse_json(artifact_bytes, label=f"bundled {label}")

            base = loaded["construction_manifest"]
            audit_result = loaded["human_audit_result"]
            audit_protocol = loaded["human_audit_protocol"]
            _validate_human_audit_integer_fields(
                construction, base, audit_result, audit_protocol)
            shared = (
                "study_id", "task", "round", "corpus_commit", "note_sha256",
                "note_path", "entry_ids", "entries", "usage",
                "automated_claim_ready", "automated_readiness",
                "construction_artifacts", "construction_artifacts_sha256",
            )
            if (not isinstance(base, dict) or base.get("claim_ready") is not False
                    or base.get("automated_claim_ready") is not True
                    or any(base.get(key) != construction.get(key) for key in shared)):
                raise GradeIntegrityError(
                    "audited note drifted from its automated construction manifest")

            construction_dependencies = _load_bundled_construction_dependencies(
                run_task_root,
                bundle,
                bundle_root,
                base.get("construction_artifacts"),
                base.get("construction_artifacts_sha256"),
            )
            try:
                audit_validation = validate_human_audit_result(
                    audit_result, base, construction_dependencies
                )
            except HumanAuditError as exc:
                raise GradeIntegrityError(
                    f"bundled human-audit population or decision is invalid: {exc}"
                ) from exc
            if not audit_validation.passed:
                raise GradeIntegrityError("bundled human audit is not passing")
            try:
                auditor_id = validate_id(audit_result.get("auditor_id"), "auditor ID")
            except (TypeError, ValueError) as exc:
                raise GradeIntegrityError(
                    "bundled human audit has an invalid auditor ID"
                ) from exc
            audit_expected = {
                "schema_version": 1,
                "study_id": construction["study_id"],
                "task": construction["task"],
                "round": construction["round"],
                "construction_manifest_sha256": base_record["sha256"],
                "note_sha256": note_sha256,
                "protocol_sha256": human["protocol_sha256"],
                "auditor_id": auditor_id,
                "blinding_preserved": True,
                "reviewer_independent": True,
                "decision": "pass",
            }
            if (not isinstance(audit_result, dict)
                    or human.get("auditor_id") != auditor_id
                    or any(audit_result.get(key) != value
                           for key, value in audit_expected.items())):
                raise GradeIntegrityError("human audit result does not bind the promoted note")
            try:
                validate_human_audit_protocol(audit_protocol)
            except HumanAuditError as exc:
                raise GradeIntegrityError(
                    "note audit protocol is not preregistered and blinded"
                ) from exc
        elif manifest_type == "forced-50-cheatsheet":
            config = construction.get("config")
            bundle = note_record.get("provenance_bundle")
            if (type(construction.get("manifest_schema")) is not int
                    or construction["manifest_schema"] != 1
                    or not isinstance(config, dict)
                    or type(config.get("schema_version")) is not int
                    or config["schema_version"] != 1
                    or config.get("method") != "forced-50-cheatsheet"
                    or config.get("claim_ready") is not True
                    or config.get("study_id") != construction["study_id"]
                    or config.get("task") != construction["task"]
                    or config.get("corpus") != spec.get("corpus")
                    or not isinstance(config.get("environment"), dict)
                    or not environment_is_claim_ready(config["environment"])
                    or not isinstance(config.get("model"), str)
                    or not config["model"]
                    or not isinstance(config.get("model_revision"), str)
                    or not config["model_revision"]
                    or not isinstance(config.get("expected_response_model"), str)
                    or not config["expected_response_model"]
                    or type(config.get("episode_seed")) is not int
                    or config.get("forced_iterations") != 50
                    or not _valid_sha256(config.get("study_question_sha256"))
                    or not isinstance(bundle, dict)):
                raise GradeIntegrityError("forced-50 construction manifest is incomplete")
            _validate_source_record(config.get("source"), label="forced-50 study")
            bundle_root = Path(str(bundle.get("root", "")))
            if (bundle_root.is_absolute() or not bundle_root.parts
                    or any(part in ("", ".", "..") for part in bundle_root.parts)
                    or set(bundle) != {
                        "root", "manifest_snapshot", "note_snapshot",
                        "construction_artifacts",
                    }):
                raise GradeIntegrityError("forced-50 provenance bundle is malformed")
            bundled_manifest = _safe_relative_file(
                run_task_root, bundle.get("manifest_snapshot"),
                label="bundled forced-50 manifest")
            bundled_note = _safe_relative_file(
                run_task_root, bundle.get("note_snapshot"),
                label="bundled forced-50 note")
            if (read_artifact_bytes(bundled_manifest) != construction_bytes
                    or read_artifact_bytes(bundled_note) != note_bytes):
                raise GradeIntegrityError(
                    "forced-50 provenance bundle changed its manifest or note")
            dependencies = _load_bundled_construction_dependencies(
                run_task_root,
                bundle,
                bundle_root,
                construction.get("construction_artifacts"),
                construction.get("construction_artifacts_sha256"),
            )
            if set(dependencies) != {"intent.json", "episode.json"}:
                raise GradeIntegrityError(
                    "forced-50 construction dependency set is not exact")
            intent_bytes = dependencies["intent.json"]
            episode_bytes = dependencies["episode.json"]
            intent = parse_json(intent_bytes, label="forced-50 study intent")
            episode = parse_json(episode_bytes, label="forced-50 study episode")
            if (intent_bytes != canonical_json_bytes(intent)
                    or episode_bytes != canonical_json_bytes(episode)
                    or not isinstance(intent, dict)
                    or not isinstance(episode, dict)):
                raise GradeIntegrityError(
                    "forced-50 dependencies are not canonical JSON objects")
            validate_episode(episode, {"id": "cheatsheet"})
            study_provider_identity = episode_provider_identity(episode)
            if (intent != config
                    or construction.get("intent_sha256") != sha256_json(intent)
                    or construction.get("episode_sha256") != sha256_json(episode)
                    or episode.get("study_intent_sha256")
                    != construction.get("intent_sha256")
                    or episode.get("question_sha256")
                    != config.get("study_question_sha256")
                    or episode.get("task") != corpus.name
                    or episode.get("qid") != "cheatsheet"
                    or episode.get("budget") != "s50"
                    or type(episode.get("rollout")) is not int
                    or episode["rollout"] != 0
                    or episode.get("harness") != "dspy.ReAct"
                    or episode.get("status") != "ok"
                    or episode.get("model") != config.get("model")
                    or episode.get("model_revision") != config.get("model_revision")
                    or episode.get("seed") != config.get("episode_seed")
                    or study_provider_identity["response_models"]
                    != [config["expected_response_model"]]
                    or not isinstance(episode.get("answer"), str)
                    or sha256_bytes(episode["answer"].encode("utf-8")) != note_sha256
                    or episode.get("n_react_iters") != config.get("forced_iterations")
                    or episode.get("n_tool_iters", 0) + episode.get("finish_catches", 0)
                    != config.get("forced_iterations")
                    or construction.get("study_generated_tokens")
                    != episode.get("completion_tokens")
                    or construction.get("study_prompt_tokens")
                    != episode.get("prompt_tokens")
                    or construction.get("study_total_tokens")
                    != episode.get("total_tokens")):
                raise GradeIntegrityError(
                    "forced-50 intent, episode, and note do not bind exactly")
        else:
            raise GradeIntegrityError(
                "unknown claim-ready note manifest type; add an explicit validator")

    prompt_policy = spec.get("prompt_policy")
    if not isinstance(prompt_policy, dict):
        raise GradeIntegrityError("run prompt policy is invalid")
    template = prompt_policy.get("note_prefix_template")
    if note_record:
        if not isinstance(template, str) or template.count("{note}") != 1:
            raise GradeIntegrityError("run note-prefix template is invalid")
        prefix = template.format(note=note)
    else:
        if template is not None:
            raise GradeIntegrityError("run without a note has a note-prefix template")
        prefix = ""
    presented_prompts = {
        row["id"]: sha256_bytes((prefix + row["question"]).encode("utf-8"))
        for row in questions
    }
    if prompt_policy.get("presented_prompt_sha256") != presented_prompts:
        raise GradeIntegrityError("run presented-prompt hashes do not match the prompt policy")
    return {
        "manifest": manifest,
        "manifest_sha256": sha256_bytes(manifest_bytes),
        "run_task_root": run_task_root,
        "spec": spec,
        "expected_episodes": expected,
        "question_sha256": {row["id"]: sha256_json(row) for row in questions},
        "prompt_sha256": presented_prompts,
        "episode_seeds": expected_seeds,
        "note_sha256": note_sha256,
        "note_construction_manifest_sha256": (
            note_record["construction_manifest"]["sha256"] if note_record else None
        ),
        "note_manifest": note_manifest,
        "preregistration": preregistration,
    }


def validate_preregistered_grading_policy(
    document: object,
    *,
    grader: str,
    judge_model: str,
    whole_files: bool,
    effort: str,
) -> None:
    """Require the requested grader invocation to equal the frozen policy."""

    if not isinstance(document, dict) or not isinstance(
        document.get("grading_policy"), dict
    ):
        raise GradeIntegrityError("run has no preregistered grading policy")
    policy = document["grading_policy"]
    expected = {
        "grader": grader,
        "judge_model": judge_model,
        "evidence_mode": "whole_files" if whole_files else "excerpt_evidence",
        "judge_effort": effort,
        "claim_scoring": "binary_0_1",
        "question_scoring": "weighted_claim_sum",
    }
    if policy != expected:
        raise GradeIntegrityError(
            "requested grading configuration differs from the preregistration"
        )


def validate_manifest_episode(ep: dict, row: dict, manifest_context: dict) -> None:
    relative = f"{ep.get('budget')}/r{ep.get('rollout')}/{row['id']}.json"
    expected = {
        "manifest_sha256": manifest_context["manifest_sha256"],
        "question_sha256": manifest_context["question_sha256"][row["id"]],
        "prompt_sha256": manifest_context["prompt_sha256"][row["id"]],
        "note_sha256": manifest_context["note_sha256"],
        "seed": manifest_context["episode_seeds"].get(relative),
    }
    for field, value in expected.items():
        if ep.get(field) != value:
            raise GradeIntegrityError(f"{row['id']}: episode {field} does not match manifest")
    spec = manifest_context["spec"]
    try:
        validate_environment_snapshot(
            manifest_context["run_task_root"],
            ep.get("environment_snapshot"),
            baseline=spec.get("environment"),
            require_claim_ready=True,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise GradeIntegrityError(
            f"{row['id']}: episode launch environment is invalid: {exc}"
        ) from exc
    extra = spec.get("extra")
    if not isinstance(extra, dict):
        raise GradeIntegrityError("run specification extra metadata is invalid")
    declared = {
        "task": spec["task"],
        "model": spec.get("model"),
        "model_revision": extra.get("model_revision"),
        "harness": spec.get("harness"),
    }
    for field, value in declared.items():
        if not isinstance(value, str) or not value or ep.get(field) != value:
            raise GradeIntegrityError(
                f"{row['id']}: episode {field} does not match the run specification")
    expected_response_model = extra.get("expected_response_model")
    if not isinstance(expected_response_model, str) or not expected_response_model:
        raise GradeIntegrityError(
            "run specification has no expected provider response model")
    if ep.get("status") in {"ok", "no_answer"}:
        observed = episode_provider_identity(ep)["response_models"]
        if observed != [expected_response_model]:
            raise GradeIntegrityError(
                f"{row['id']}: provider response model {observed} does not match "
                f"the run specification {expected_response_model!r}")


GRADER_PROMPT = """You are grading one model answer for a source-grounded {library_name} expert QA benchmark.

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
- Question ID: `{question_id}`
- Label: `{label}`
- Question: `{question}`
- Model answer:
{model_answer}

## Gold answer
{gold_answer}

## Claim rubric
{claim_rubric_json}

## Evidence spans
{evidence_spans_json}

## Whole evidence files
{whole_evidence_text}

Return JSON that matches the schema exactly."""

def judge_schema(row: dict) -> dict:
    """Constrain claim ids/scores/count; runtime validation enforces uniqueness."""
    ids = rubric_ids(row)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "grading",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "claims": {
                        "type": "array",
                        "minItems": len(ids),
                        "maxItems": len(ids),
                        "items": {
                            "type": "object",
                            "properties": {
                                "claim_id": {"type": "string", "enum": ids},
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

log = logging.getLogger("grade")


def build_prompt(corpus, row: dict, model_answer: str, whole_files: bool = False) -> str:
    if whole_files:
        # A.5-faithful: spans = the dataset's excerpts; whole files = full numbered
        # dumps of every evidence file from the pinned checkout
        spans_meta = row["evidence"]
        paths = list(dict.fromkeys(e["path"] for e in row["evidence"]))
        whole_text = "\n\n".join(
            f"### {p}\n" + "\n".join(
                f"{i:04d}: {line}" for i, line in enumerate(
                    read_pinned_code_bytes(corpus, p).decode("utf-8").splitlines(), 1)
            ) for p in paths
        )
    else:
        # dataset-README variant: the excerpts are the only code context
        spans_meta = [
            {k: e[k] for k in ("span_id", "path", "start_line", "end_line")}
            for e in row["evidence"]
        ]
        whole_text = "\n\n".join(
            f"### {e['path']} lines {e['start_line']}-{e['end_line']} ({e['span_id']})\n{e['excerpt']}"
            for e in row["evidence"]
        )
    return GRADER_PROMPT.format(
        library_name=corpus.display,
        question_id=row["id"],
        label=row["topic"],
        question=row["question"],
        model_answer=model_answer,
        gold_answer=row["gold_answer"],
        claim_rubric_json=json.dumps(row["rubric"], indent=2),
        evidence_spans_json=json.dumps(spans_meta, indent=2),
        whole_evidence_text=whole_text,
    )


@lru_cache(maxsize=None)
def sandbox_configuration_record(language: str) -> dict:
    """Hash expensive checker/container artifacts once per grading process."""
    return sandbox.configuration_record(language)


def sandbox_configuration_sha256(language: str) -> str:
    """Return the digest embedded in every deterministic checker result."""

    return stable_sha256(sandbox_configuration_record(language))


def grade_spec_sha256(corpus, row: dict, judge_model: str,
                      whole_files: bool = False, effort: str = "", *,
                      judge_base_url: str | None = None) -> str:
    """Hash every static input that defines how this question is graded."""
    judge_base_url = _resolve_judge_base_url(judge_model, judge_base_url)
    return stable_sha256({
        "grade_schema_version": GRADE_SCHEMA_VERSION,
        "grader_source": {
            "studybench/grade.py": file_sha256(Path(__file__).resolve()),
            "studybench/sandbox.py": file_sha256(Path(sandbox.__file__).resolve()),
        },
        "sandbox_configuration": sandbox_configuration_record(corpus.language),
        "judge_model": judge_model,
        "judge_base_url": judge_base_url,
        "whole_files": whole_files,
        "judge_effort": effort,
        "prompt": build_prompt(corpus, row, "<MODEL_ANSWER>", whole_files),
        "response_format": judge_schema(row),
    })


def score_from_claims(row: dict, claim_scores: dict[str, int], compile_ok: bool) -> dict:
    ids = rubric_ids(row)
    if set(claim_scores) != set(ids) or len(claim_scores) != len(ids):
        raise GradeIntegrityError(
            f"{row['id']}: claim score ids do not exactly match the rubric")
    if any(type(score) is not int or score not in (0, 1)
           for score in claim_scores.values()):
        raise GradeIntegrityError(f"{row['id']}: claim scores must be integer 0/1")
    if type(compile_ok) is not bool:
        raise GradeIntegrityError(f"{row['id']}: compile_ok must be boolean")
    lenient = sum(c["weight"] * claim_scores[c["claim_id"]] for c in row["rubric"])
    cores_ok = all(
        claim_scores[c["claim_id"]] == 1
        for c in row["rubric"] if c["claim_type"] == "core"
    )
    strict = lenient if (compile_ok and cores_ok) else 0
    return {"lenient": lenient, "strict": strict, "cores_ok": cores_ok}


def validate_verdict(row: dict, verdict: dict) -> tuple[list[dict], dict]:
    """Return canonical claims and scores, or reject the entire judge response."""
    if not isinstance(verdict, dict):
        raise GradeIntegrityError("judge verdict is not an object")
    if set(verdict) != {"claims", "question_score", "needs_regrade"}:
        raise GradeIntegrityError("judge verdict has missing or unexpected fields")
    claims = verdict.get("claims")
    ids = rubric_ids(row)
    if not isinstance(claims, list) or len(claims) != len(ids):
        got = len(claims) if isinstance(claims, list) else "non-list"
        raise GradeIntegrityError(
            f"{row['id']}: judge returned {got} claims; expected {len(ids)}")

    actual_ids = []
    by_id = {}
    for claim in claims:
        if not isinstance(claim, dict):
            raise GradeIntegrityError(f"{row['id']}: judge claim is not an object")
        if set(claim) != {"claim_id", "score", "rationale"}:
            raise GradeIntegrityError(
                f"{row['id']}: judge claim has missing or unexpected fields")
        claim_id = claim.get("claim_id")
        score = claim.get("score")
        rationale = claim.get("rationale")
        if not isinstance(claim_id, str):
            raise GradeIntegrityError(f"{row['id']}: judge claim id is invalid")
        if type(score) is not int or score not in (0, 1):
            raise GradeIntegrityError(f"{row['id']}/{claim_id}: score is not integer 0/1")
        if not isinstance(rationale, str):
            raise GradeIntegrityError(f"{row['id']}/{claim_id}: rationale is not a string")
        actual_ids.append(claim_id)
        by_id[claim_id] = {"claim_id": claim_id, "score": score,
                           "rationale": rationale}

    if len(actual_ids) != len(set(actual_ids)):
        raise GradeIntegrityError(f"{row['id']}: judge returned duplicate claim ids")
    if set(actual_ids) != set(ids):
        missing = sorted(set(ids) - set(actual_ids))
        extra = sorted(set(actual_ids) - set(ids))
        raise GradeIntegrityError(
            f"{row['id']}: judge claim ids mismatch (missing={missing}, extra={extra})")

    claim_scores = {claim_id: by_id[claim_id]["score"] for claim_id in ids}
    scores = score_from_claims(row, claim_scores, compile_ok=False)
    reported = verdict.get("question_score")
    if (isinstance(reported, bool) or not isinstance(reported, (int, float))
            or not math.isfinite(reported) or reported != scores["lenient"]):
        raise GradeIntegrityError(
            f"{row['id']}: judge question_score={reported!r}; "
            f"recomputed={scores['lenient']}")
    if type(verdict.get("needs_regrade")) is not bool:
        raise GradeIntegrityError(f"{row['id']}: needs_regrade is not boolean")
    if verdict["needs_regrade"]:
        raise GradeIntegrityError(f"{row['id']}: judge requested regrade")
    return [by_id[claim_id] for claim_id in ids], claim_scores


def validate_episode(ep: dict, row: dict) -> None:
    """Reject infrastructure/protocol failures before they can become grades."""
    for key in ("task", "qid", "budget", "rollout", "status", "gen_tokens"):
        if key not in ep:
            raise GradeIntegrityError(f"episode missing {key}")
    if ep["qid"] != row["id"]:
        raise GradeIntegrityError(f"episode qid {ep['qid']} != rubric qid {row['id']}")
    if ep["status"] not in {"ok", "no_answer"}:
        raise GradeIntegrityError(f"{ep['qid']}: non-evaluable status {ep['status']!r}")
    if type(ep["rollout"]) is not int or ep["rollout"] < 0:
        raise GradeIntegrityError(f"{ep['qid']}: invalid rollout")
    for field in ("prompt_tokens", "completion_tokens", "total_tokens", "gen_tokens"):
        if type(ep.get(field)) is not int or ep[field] < 0:
            raise GradeIntegrityError(f"{ep['qid']}: invalid {field}")
    if ep["gen_tokens"] != ep["completion_tokens"]:
        raise GradeIntegrityError(
            f"{ep['qid']}: gen_tokens does not equal completion_tokens")
    if type(ep.get("seed")) is not int:
        raise GradeIntegrityError(f"{ep['qid']}: missing deterministic episode seed")

    answer = ep.get("answer", "")
    if not isinstance(answer, str):
        raise GradeIntegrityError(f"{ep['qid']}: answer is not a string")
    if ep["status"] == "ok" and not answer.strip():
        raise GradeIntegrityError(f"{ep['qid']}: ok episode has an empty answer")
    if ep["status"] == "no_answer" and answer.strip():
        raise GradeIntegrityError(f"{ep['qid']}: no_answer episode has a non-empty answer")

    tool_iters = ep.get("n_tool_iters", 0)
    finish_catches = ep.get("finish_catches", 0)
    if type(tool_iters) is not int or tool_iters < 0:
        raise GradeIntegrityError(f"{ep['qid']}: invalid tool-iteration count")
    if type(finish_catches) is not int or finish_catches < 0:
        raise GradeIntegrityError(f"{ep['qid']}: invalid finish-catch count")

    turns = ep.get("turns", [])
    if not isinstance(turns, list):
        raise GradeIntegrityError(f"{ep['qid']}: turns are not a list")
    observed_tool_iters = 0
    observed_finish_catches = 0
    for index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            raise GradeIntegrityError(f"{ep['qid']}: turn {index} is not an object")
        calls = turn.get("tool_calls", [])
        observations = turn.get("observations", [])
        if not isinstance(calls, list) or not isinstance(observations, list):
            raise GradeIntegrityError(
                f"{ep['qid']}: turn {index} tool evidence is malformed")
        if len(calls) > 1:
            raise GradeIntegrityError(
                f"{ep['qid']}: turn {index} contains multiple tool calls")
        if len(observations) != len(calls):
            raise GradeIntegrityError(
                f"{ep['qid']}: turn {index} tool calls and observations differ")
        for call, observation in zip(calls, observations, strict=True):
            if (not isinstance(call, dict) or set(call) != {"name", "arguments"}
                    or not isinstance(call.get("name"), str) or not call["name"]
                    or not isinstance(call.get("arguments"), str)
                    or not isinstance(observation, str)):
                raise GradeIntegrityError(
                    f"{ep['qid']}: turn {index} tool evidence is invalid")
            if call["name"] == "finish":
                observed_finish_catches += 1
            else:
                observed_tool_iters += 1
    if (tool_iters != observed_tool_iters
            or finish_catches != observed_finish_catches):
        raise GradeIntegrityError(
            f"{ep['qid']}: tool counters do not match recorded calls and observations")
    if "n_react_iters" in ep:
        react_iters = ep["n_react_iters"]
        if (type(react_iters) is not int or react_iters < 0
                or react_iters != observed_tool_iters + observed_finish_catches):
            raise GradeIntegrityError(
                f"{ep['qid']}: ReAct iteration count does not match recorded turns")
    if ep["budget"] == "direct" and (tool_iters or finish_catches):
        raise GradeIntegrityError(f"{ep['qid']}: direct episode used tools")
    if ep["budget"] == "k5" and tool_iters + finish_catches > 5:
        raise GradeIntegrityError(f"{ep['qid']}: k5 episode exceeded its budget")
    if ep["budget"] == "k20" and tool_iters + finish_catches > 20:
        raise GradeIntegrityError(f"{ep['qid']}: k20 episode exceeded its budget")
    if ep["budget"] == "k20f" and tool_iters + finish_catches != 20:
        raise GradeIntegrityError(
            f"{ep['qid']}: forced k20 recorded {tool_iters + finish_catches} iterations")

    def usage_fields(record: object, label: str) -> tuple[int, int, int]:
        if not isinstance(record, dict):
            raise GradeIntegrityError(f"{ep['qid']}: {label} is not an object")
        values = []
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = record.get(field)
            if type(value) is not int or value < 0:
                raise GradeIntegrityError(
                    f"{ep['qid']}: {label} has invalid {field}")
            values.append(value)
        if values[2] != values[0] + values[1]:
            raise GradeIntegrityError(
                f"{ep['qid']}: {label} total_tokens is not prompt + completion")
        return tuple(values)

    if "usage_ledger" in ep:
        ledger = ep["usage_ledger"]
        if not isinstance(ledger, list):
            raise GradeIntegrityError(f"{ep['qid']}: usage ledger is not a list")
        if type(ep.get("n_lm_calls")) is not int or ep["n_lm_calls"] != len(ledger):
            raise GradeIntegrityError(f"{ep['qid']}: usage ledger length mismatch")
        totals = [0, 0, 0]
        for index, record in enumerate(ledger):
            if not isinstance(record, dict) or record.get("call") != index:
                raise GradeIntegrityError(f"{ep['qid']}: usage ledger call sequence mismatch")
            for position, value in enumerate(usage_fields(record, f"usage call {index}")):
                totals[position] += value
    else:
        turns = ep.get("turns")
        if not isinstance(turns, list) or not turns:
            raise GradeIntegrityError(f"{ep['qid']}: native episode has no usage-bearing turns")
        totals = [0, 0, 0]
        for index, turn in enumerate(turns):
            for position, value in enumerate(usage_fields(turn, f"turn {index}")):
                totals[position] += value
    scalars = [ep["prompt_tokens"], ep["completion_tokens"], ep["total_tokens"]]
    if scalars != totals:
        raise GradeIntegrityError(
            f"{ep['qid']}: token scalars {scalars} do not match usage records {totals}")
    episode_provider_identity(ep)


def episode_provider_identity(ep: dict) -> dict[str, Any]:
    """Validate and summarize provider-returned generation identity fields."""
    native = "usage_ledger" not in ep
    records = ep.get("turns") if native else ep.get("usage_ledger")
    if not isinstance(records, list) or not records:
        raise GradeIntegrityError(f"{ep.get('qid')}: no provider call records")
    models = set()
    fingerprints = set()
    missing_fingerprints = 0
    for index, record in enumerate(records):
        model = record.get("response_model") if isinstance(record, dict) else None
        if not isinstance(model, str) or not model:
            raise GradeIntegrityError(
                f"{ep.get('qid')}: provider call {index} has no response_model")
        models.add(model)
        response_id = record.get("response_id")
        if not isinstance(response_id, str) or not response_id:
            raise GradeIntegrityError(
                f"{ep.get('qid')}: provider call {index} has no response_id")
        fingerprint = record.get("system_fingerprint")
        if fingerprint is None:
            missing_fingerprints += 1
        elif not isinstance(fingerprint, str) or not fingerprint:
            raise GradeIntegrityError(
                f"{ep.get('qid')}: provider call {index} has invalid system_fingerprint")
        else:
            fingerprints.add(fingerprint)
        if not native:
            for hash_field in ("request_messages_sha256", "outputs_sha256"):
                if not _valid_sha256(record.get(hash_field)):
                    raise GradeIntegrityError(
                        f"{ep.get('qid')}: DSPy call {index} has invalid {hash_field}")
            provider_usage = record.get("provider_usage")
            if not isinstance(provider_usage, dict):
                raise GradeIntegrityError(
                    f"{ep.get('qid')}: DSPy call {index} has no provider usage")
            for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                if provider_usage.get(field) != record.get(field):
                    raise GradeIntegrityError(
                        f"{ep.get('qid')}: DSPy call {index} {field} disagrees with provider usage")
    if native:
        attempts = ep.get("request_attempts")
        if not isinstance(attempts, list) or not attempts:
            raise GradeIntegrityError(
                f"{ep.get('qid')}: native episode has no request-attempt audit")
        grouped: dict[int, list[dict[str, Any]]] = {}
        observed_order = []
        for record in attempts:
            if not isinstance(record, dict):
                raise GradeIntegrityError(
                    f"{ep.get('qid')}: native request attempt is not an object")
            logical_call = record.get("logical_call")
            attempt = record.get("attempt")
            if (type(logical_call) is not int or logical_call < 0
                    or type(attempt) is not int or attempt <= 0):
                raise GradeIntegrityError(
                    f"{ep.get('qid')}: native request attempt identity is invalid")
            grouped.setdefault(logical_call, []).append(record)
            observed_order.append((logical_call, attempt))
        if set(grouped) != set(range(len(records))):
            raise GradeIntegrityError(
                f"{ep.get('qid')}: native request audit does not cover every provider call")
        expected_order = []
        for logical_call, turn in enumerate(records):
            call_attempts = grouped[logical_call]
            attempt_numbers = [item.get("attempt") for item in call_attempts]
            if attempt_numbers != list(range(1, len(call_attempts) + 1)):
                raise GradeIntegrityError(
                    f"{ep.get('qid')}: native request retry sequence is invalid")
            expected_order.extend((logical_call, number) for number in attempt_numbers)
            request_hashes = {item.get("request_sha256") for item in call_attempts}
            if (len(request_hashes) != 1
                    or not _valid_sha256(next(iter(request_hashes), None))):
                raise GradeIntegrityError(
                    f"{ep.get('qid')}: native request payload hash is missing or changed on retry")
            for item in call_attempts[:-1]:
                if (item.get("status") != "transport_error"
                        or not isinstance(item.get("error_type"), str)
                        or not item["error_type"]
                        or not isinstance(item.get("error"), str)
                        or item.get("usage") != "unknown"):
                    raise GradeIntegrityError(
                        f"{ep.get('qid')}: native transport failure audit is incomplete")
            response = call_attempts[-1]
            if (response.get("status") != "response"
                    or response.get("response_id") != turn["response_id"]
                    or response.get("response_model") != turn["response_model"]):
                raise GradeIntegrityError(
                    f"{ep.get('qid')}: native response audit does not match its turn")
        if observed_order != expected_order:
            raise GradeIntegrityError(
                f"{ep.get('qid')}: native request attempts are not in execution order")
    if len(models) != 1:
        raise GradeIntegrityError(
            f"{ep.get('qid')}: episode resolved to multiple response models: {sorted(models)}")
    return {
        "harness_usage": "native_turns" if native else "dspy_usage_ledger",
        "response_models": sorted(models),
        "system_fingerprints": sorted(fingerprints),
        "missing_system_fingerprint_calls": missing_fingerprints,
    }


def _error_record(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def _audit_observation(value: object) -> dict[str, Any]:
    """Retain an invalid provider field without requiring it to be JSON-native."""
    try:
        payload = canonical_json_bytes(value)
    except Exception:
        try:
            representation = repr(value).encode("utf-8", errors="backslashreplace")
        except Exception:
            representation = f"<{type(value).__name__}>".encode("utf-8")
        return {
            "python_type": type(value).__name__,
            "json_serializable": False,
            "json_value": None,
            "value_sha256": sha256_bytes(representation),
        }
    return {
        "python_type": type(value).__name__,
        "json_serializable": True,
        "json_value": value,
        "value_sha256": sha256_bytes(payload),
    }


def _validate_audit_observation(value: object) -> None:
    if (not isinstance(value, dict)
            or set(value) != {
                "python_type", "json_serializable", "json_value", "value_sha256",
            }
            or not isinstance(value.get("python_type"), str)
            or not value["python_type"]
            or type(value.get("json_serializable")) is not bool
            or not _valid_sha256(value.get("value_sha256"))):
        raise GradeIntegrityError("incomplete judge response observation is invalid")
    if value["json_serializable"]:
        try:
            observed_hash = sha256_bytes(canonical_json_bytes(value["json_value"]))
        except (TypeError, ValueError) as exc:
            raise GradeIntegrityError(
                "judge response observation is not canonical JSON") from exc
        if observed_hash != value["value_sha256"]:
            raise GradeIntegrityError("judge response observation hash does not match")
    elif value.get("json_value") is not None:
        raise GradeIntegrityError(
            "non-JSON judge response observation retained an unsafe value")


def _provider_usage_raw(response: object) -> object:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump(mode="json")
    if isinstance(usage, dict):
        return dict(usage)
    return {
        field: getattr(usage, field)
        for field in (
            "prompt_tokens", "completion_tokens", "total_tokens",
            "input_tokens", "output_tokens",
        )
        if hasattr(usage, field)
    }


def _normalize_provider_usage(raw: object) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise GradeIntegrityError("provider response has no usage object")
    try:
        canonical_json_bytes(raw)
    except (TypeError, ValueError) as exc:
        raise GradeIntegrityError("provider usage is not JSON-serializable") from exc

    def token(*names: str) -> object:
        for name in names:
            if name in raw:
                return raw[name]
        return None

    normalized = {
        "prompt_tokens": token("prompt_tokens", "input_tokens"),
        "completion_tokens": token("completion_tokens", "output_tokens"),
        "total_tokens": token("total_tokens"),
        "provider_usage": raw,
    }
    _validate_usage(normalized)
    return normalized


def _provider_usage(response: object) -> dict[str, Any]:
    return _normalize_provider_usage(_provider_usage_raw(response))


def _validate_usage(usage: object) -> None:
    expected_keys = {
        "prompt_tokens", "completion_tokens", "total_tokens", "provider_usage",
    }
    if (not isinstance(usage, dict) or set(usage) != expected_keys
            or not isinstance(usage.get("provider_usage"), dict)):
        raise GradeIntegrityError("judge usage record is invalid")
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(field)
        if type(value) is not int or value < 0:
            raise GradeIntegrityError(f"judge usage has invalid {field}")
    if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
        raise GradeIntegrityError("judge usage total is not prompt + completion")
    raw = usage["provider_usage"]

    def token(*names: str) -> object:
        for name in names:
            if name in raw:
                return raw[name]
        return None

    expected = {
        "prompt_tokens": token("prompt_tokens", "input_tokens"),
        "completion_tokens": token("completion_tokens", "output_tokens"),
        "total_tokens": token("total_tokens"),
    }
    if any(usage[field] != expected[field] for field in expected):
        raise GradeIntegrityError("normalized judge usage disagrees with provider usage")
    try:
        canonical_json_bytes(raw)
    except (TypeError, ValueError) as exc:
        raise GradeIntegrityError("stored provider usage is not JSON-serializable") from exc


def _read_response_value(response: object, *names: str) -> tuple[object, BaseException | None]:
    last_error = None
    for name in names:
        try:
            value = getattr(response, name)
        except Exception as exc:
            last_error = exc
            continue
        if value is not None:
            return value, None
    return None, last_error


def _response_attempt(
    response: object, attempt: int,
) -> tuple[dict[str, Any], str | None, GradeIntegrityError | None, bool]:
    """Capture one received response completely enough to fail closed and audit it."""
    incomplete: dict[str, dict[str, Any]] = {}
    issues = []
    fatal = False
    normalized_identity: dict[str, str | None] = {}
    for field, names, label in (
        ("response_id", ("id",), "response ID"),
        ("request_id", ("_request_id", "request_id"), "request ID"),
        ("response_model", ("model",), "response model"),
    ):
        value, extraction_error = _read_response_value(response, *names)
        if isinstance(value, str) and value:
            normalized_identity[field] = value
            continue
        normalized_identity[field] = None
        incomplete[field] = _audit_observation(value)
        detail = (f" ({type(extraction_error).__name__}: {extraction_error})"
                  if extraction_error is not None else "")
        issues.append(f"provider {label} is missing or invalid{detail}")
        fatal = True

    fingerprint, _ = _read_response_value(response, "system_fingerprint")
    if isinstance(fingerprint, str) and fingerprint:
        fingerprint_status = "available"
        fingerprint_observation = None
    else:
        fingerprint_status = "unavailable"
        fingerprint_observation = _audit_observation(fingerprint)
        fingerprint = None

    try:
        content = response.choices[0].message.content
    except Exception as exc:
        content = None
        incomplete["content"] = _audit_observation(None)
        issues.append(f"judge response has no message content ({type(exc).__name__}: {exc})")
    if not isinstance(content, str):
        if "content" not in incomplete:
            incomplete["content"] = _audit_observation(content)
            issues.append("judge response content is not a string")

    usage = None
    usage_error = None
    try:
        raw_usage = _provider_usage_raw(response)
        usage = _normalize_provider_usage(raw_usage)
    except Exception as exc:
        raw_usage = locals().get("raw_usage")
        usage_error = _error_record(exc)
        incomplete["usage"] = _audit_observation(raw_usage)
        issues.append(f"judge response usage is unavailable ({type(exc).__name__}: {exc})")
        fatal = True

    record = {
        "attempt": attempt,
        "accepted": False,
        **normalized_identity,
        "system_fingerprint": fingerprint,
        "system_fingerprint_status": fingerprint_status,
        "system_fingerprint_observation": fingerprint_observation,
        "usage_status": "complete" if usage is not None else "unavailable",
        "usage": usage,
        "usage_error": usage_error,
        "content_sha256": sha256_bytes(content.encode("utf-8"))
        if isinstance(content, str) else None,
        "content_bytes": len(content.encode("utf-8")) if isinstance(content, str) else None,
        "invalid_content": None,
        "incomplete_response": incomplete or None,
        "validation_error": None,
    }
    error = GradeIntegrityError("; ".join(issues)) if issues else None
    return record, content if isinstance(content, str) else None, error, fatal


def _uninspectable_response_attempt(
    attempt: int, error: BaseException,
) -> dict[str, Any]:
    """Last-resort record: a response arrived but local inspection itself failed."""
    observation = _audit_observation(None)
    return {
        "attempt": attempt,
        "accepted": False,
        "response_id": None,
        "request_id": None,
        "response_model": None,
        "system_fingerprint": None,
        "system_fingerprint_status": "unavailable",
        "system_fingerprint_observation": dict(observation),
        "usage_status": "unavailable",
        "usage": None,
        "usage_error": _error_record(error),
        "content_sha256": None,
        "content_bytes": None,
        "invalid_content": None,
        "incomplete_response": {
            field: dict(observation)
            for field in ("response_id", "request_id", "response_model", "content", "usage")
        },
        "validation_error": None,
    }


def validate_judge_attempt_record(
    attempt: object, index: int, *, accepted: bool,
) -> None:
    expected_keys = {
        "attempt", "accepted", "response_id", "request_id", "response_model",
        "system_fingerprint", "system_fingerprint_status",
        "system_fingerprint_observation",
        "usage_status", "usage", "usage_error", "content_sha256", "content_bytes",
        "invalid_content", "incomplete_response", "validation_error",
    }
    if (not isinstance(attempt, dict) or set(attempt) != expected_keys
            or attempt.get("attempt") != index or attempt.get("accepted") is not accepted):
        raise GradeIntegrityError("stored judge attempt shape or sequence is invalid")
    for field in ("response_id", "request_id", "response_model"):
        value = attempt[field]
        if value is not None and (not isinstance(value, str) or not value):
            raise GradeIntegrityError(f"stored judge {field} is invalid")
    incomplete = attempt["incomplete_response"]
    if incomplete is not None:
        allowed = {"response_id", "request_id", "response_model", "content", "usage"}
        if (not isinstance(incomplete, dict) or not incomplete
                or not set(incomplete).issubset(allowed)):
            raise GradeIntegrityError("stored incomplete judge response is invalid")
        for observation in incomplete.values():
            _validate_audit_observation(observation)
    incomplete_fields = set(incomplete) if isinstance(incomplete, dict) else set()
    for field in ("response_id", "request_id", "response_model"):
        if (attempt[field] is None) != (field in incomplete_fields):
            raise GradeIntegrityError(
                f"stored judge {field} disagrees with its incomplete-response marker")
    fingerprint_status = attempt["system_fingerprint_status"]
    if fingerprint_status == "available":
        if (not isinstance(attempt["system_fingerprint"], str)
                or not attempt["system_fingerprint"]
                or attempt["system_fingerprint_observation"] is not None):
            raise GradeIntegrityError("available judge system fingerprint is invalid")
    elif fingerprint_status == "unavailable":
        if attempt["system_fingerprint"] is not None:
            raise GradeIntegrityError("unavailable judge system fingerprint has a value")
        _validate_audit_observation(attempt["system_fingerprint_observation"])
    else:
        raise GradeIntegrityError("judge system fingerprint status is invalid")
    usage_status = attempt["usage_status"]
    if usage_status == "complete":
        _validate_usage(attempt["usage"])
        if attempt["usage_error"] is not None or (
                isinstance(incomplete, dict) and "usage" in incomplete):
            raise GradeIntegrityError("complete judge usage has an unavailable marker")
    elif usage_status == "unavailable":
        error = attempt["usage_error"]
        if (attempt["usage"] is not None
                or not isinstance(incomplete, dict) or "usage" not in incomplete
                or not isinstance(error, dict)
                or set(error) != {"type", "message"}
                or not isinstance(error.get("type"), str)
                or not isinstance(error.get("message"), str)):
            raise GradeIntegrityError("unavailable judge usage record is invalid")
    else:
        raise GradeIntegrityError("stored judge usage status is invalid")

    content_hash = attempt["content_sha256"]
    content_bytes = attempt["content_bytes"]
    valid_content = (_valid_sha256(content_hash)
                     and type(content_bytes) is int and content_bytes >= 0)
    missing_content = content_hash is None and content_bytes is None
    if not (valid_content or missing_content):
        raise GradeIntegrityError("stored judge content identity is invalid")
    if missing_content != ("content" in incomplete_fields):
        raise GradeIntegrityError(
            "stored judge content disagrees with its incomplete-response marker")
    raw = attempt["invalid_content"]
    error = attempt["validation_error"]
    if accepted:
        if (error is not None or raw is not None or incomplete is not None
                or usage_status != "complete" or not valid_content
                or any(not attempt[field] for field in (
                    "response_id", "request_id", "response_model",
                ))):
            raise GradeIntegrityError("accepted judge attempt metadata is invalid")
    else:
        if (not isinstance(error, dict) or set(error) != {"type", "message"}
                or not isinstance(error.get("type"), str)
                or not isinstance(error.get("message"), str)):
            raise GradeIntegrityError("failed judge attempt has no validation error")
        if valid_content:
            if (not isinstance(raw, str)
                    or sha256_bytes(raw.encode("utf-8")) != content_hash
                    or len(raw.encode("utf-8")) != content_bytes):
                raise GradeIntegrityError("failed judge content does not match its identity")
        elif raw is not None:
            raise GradeIntegrityError("failed judge attempt stores content it did not receive")


def judge_usage_summary(
    attempts: list[dict[str, Any]], request_attempt_count: int,
) -> dict[str, Any]:
    known = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    missing_response = request_attempt_count != len(attempts)
    unavailable_response_usage = False
    for attempt in attempts:
        if attempt.get("usage_status") == "complete":
            _validate_usage(attempt.get("usage"))
            for field in known:
                known[field] += attempt["usage"][field]
        else:
            unavailable_response_usage = True
    if missing_response:
        status = "unavailable-for-request-without-response"
    elif unavailable_response_usage:
        status = "unavailable-for-response-without-usage"
    else:
        status = "complete"
    return {
        "status": status,
        "total": known if status == "complete" else None,
        "known_total": known,
    }


def _usage_total(attempts: list[dict[str, Any]]) -> dict[str, int]:
    summary = judge_usage_summary(attempts, len(attempts))
    if summary["total"] is None:
        raise GradeIntegrityError("cannot claim a cumulative total with unavailable judge usage")
    return summary["total"]


def _validate_judge_audit(grade: dict, corpus, row: dict, ep: dict,
                          judge_model: str, whole_files: bool) -> None:
    if grade.get("judge_requested_model") != judge_model:
        raise GradeIntegrityError("stored requested judge model does not match")
    attempts = grade.get("judge_attempts")
    count = grade.get("judge_attempt_count")
    accepted_attempt = grade.get("judge_accepted_attempt")
    if not isinstance(attempts, list) or type(count) is not int or count != len(attempts):
        raise GradeIntegrityError("stored judge attempt count is invalid")
    if count < 1 or count > MAX_JUDGE_ATTEMPTS or accepted_attempt != count:
        raise GradeIntegrityError("stored accepted judge attempt is invalid")
    for index, attempt in enumerate(attempts, 1):
        validate_judge_attempt_record(attempt, index, accepted=index == count)
        if (attempt["usage_status"] != "complete"
                or any(not attempt[field] for field in (
                    "response_id", "request_id", "response_model",
                ))
                or (isinstance(attempt["incomplete_response"], dict)
                    and set(attempt["incomplete_response"]) != {"content"})):
            raise GradeIntegrityError(
                "claim-ready grade contains an incompletely audited judge response")

    expected_prompt_hash = sha256_bytes(
        build_prompt(corpus, row, ep["answer"], whole_files).encode("utf-8"))
    if grade.get("judge_prompt_sha256") != expected_prompt_hash:
        raise GradeIntegrityError("stored judge prompt hash does not match")
    accepted = attempts[-1]
    accepted_content = grade.get("judge_accepted_content")
    if not isinstance(accepted_content, str):
        raise GradeIntegrityError("stored accepted judge content is unavailable")
    accepted_bytes = accepted_content.encode("utf-8")
    if (sha256_bytes(accepted_bytes) != accepted["content_sha256"]
            or len(accepted_bytes) != accepted["content_bytes"]):
        raise GradeIntegrityError(
            "stored accepted judge content does not match its response identity")
    accepted_verdict = parse_json(
        accepted_content, label="stored accepted judge verdict")
    accepted_claims, accepted_scores = validate_verdict(row, accepted_verdict)
    accepted_question_score = score_from_claims(
        row, accepted_scores, compile_ok=False)["lenient"]
    if (grade.get("claims") != accepted_claims
            or type(grade.get("question_score")) is not type(accepted_question_score)
            or grade.get("question_score") != accepted_question_score
            or grade.get("needs_regrade") is not False):
        raise GradeIntegrityError(
            "stored grade verdict does not match the accepted judge content")
    if grade.get("judge_response_model") != accepted["response_model"]:
        raise GradeIntegrityError("stored provider-returned judge model does not match")
    if grade.get("judge_usage") != accepted["usage"]:
        raise GradeIntegrityError("stored accepted judge usage does not match")
    usage_summary = judge_usage_summary(attempts, count)
    if usage_summary["status"] != "complete":
        raise GradeIntegrityError("claim-ready grade has unavailable cumulative judge usage")
    if grade.get("judge_usage_total") != usage_summary["total"]:
        raise GradeIntegrityError("stored cumulative judge usage does not match")


def _failed_judge_audit(*, ep: dict, episode_sha256: str,
                        grading_spec_sha256: str, judge_model: str,
                        judge_prompt_sha256: str,
                        attempts: list[dict[str, Any]], failure: BaseException,
                        request_attempt_count: int | None = None) -> dict:
    safe_failure_message = (
        str(failure) if isinstance(failure, GradeIntegrityError)
        else "provider request failed after prior invalid verdict(s)"
    )
    if request_attempt_count is None:
        request_attempt_count = len(attempts)
    usage_summary = judge_usage_summary(attempts, request_attempt_count)
    return {
        "failed_judge_audit_schema_version": FAILED_JUDGE_AUDIT_SCHEMA_VERSION,
        "episode_sha256": episode_sha256,
        "grading_spec_sha256": grading_spec_sha256,
        "manifest_sha256": ep.get("manifest_sha256"),
        "question_sha256": ep.get("question_sha256"),
        "prompt_sha256": ep.get("prompt_sha256"),
        "note_sha256": ep.get("note_sha256"),
        "task": ep["task"],
        "qid": ep["qid"],
        "budget": ep["budget"],
        "rollout": ep["rollout"],
        "judge_requested_model": judge_model,
        "judge_prompt_sha256": judge_prompt_sha256,
        "judge_request_attempt_count": request_attempt_count,
        "judge_attempt_count": len(attempts),
        "judge_attempts": attempts,
        "judge_usage_total": usage_summary["total"],
        "judge_usage_known_total": usage_summary["known_total"],
        "judge_usage_status": usage_summary["status"],
        "failure": {"type": type(failure).__name__, "message": safe_failure_message},
    }


def write_failed_judge_audit(out_root: Path, source_episode: str,
                             audit: dict[str, Any]) -> Path:
    """Persist failed judge costs/errors without creating a grade artifact."""
    artifact = {**audit, "source_episode": source_episode}
    digest = sha256_json(artifact)
    path = (
        out_root / "failed-judge-audits" / artifact["task"]
        / artifact["budget"] / f"r{artifact['rollout']}"
        / f"{artifact['qid']}-{digest}.json"
    )
    write_immutable_json(path, artifact)
    return path


async def grade_episode(client: AsyncOpenAI, judge_model: str, corpus, row: dict,
                        ep: dict, whole_files: bool = False, effort: str = "",
                        *, episode_sha256: str, grading_spec_sha256: str,
                        judge_base_url: str | None = None) -> dict:
    validate_episode(ep, row)
    judge_base_url = _resolve_judge_base_url(judge_model, judge_base_url)
    grade = {
        "grade_schema_version": GRADE_SCHEMA_VERSION,
        "episode_sha256": episode_sha256,
        "grading_spec_sha256": grading_spec_sha256,
        "manifest_sha256": ep.get("manifest_sha256"),
        "question_sha256": ep.get("question_sha256"),
        "prompt_sha256": ep.get("prompt_sha256"),
        "note_sha256": ep.get("note_sha256"),
        "task": ep["task"], "qid": ep["qid"], "budget": ep["budget"], "rollout": ep["rollout"],
        "judge_model": judge_model,
        "judge_requested_model": judge_model,
        "judge_base_url": judge_base_url,
        "episode_status": ep["status"],
        "gen_tokens": ep["gen_tokens"],
        "graded_at": datetime.now(timezone.utc).isoformat(),
    }
    answer = ep.get("answer", "")
    sandbox_config = sandbox_configuration_record(corpus.language)
    sandbox_config_sha256 = stable_sha256(sandbox_config)
    if ep["status"] == "no_answer":
        grade.update(compile_check={
                         "compile_ok": False,
                         "detail": "empty answer",
                         "configuration_sha256": sandbox_config_sha256,
                     },
                     claims=[], needs_regrade=False, question_score=0,
                     judge_question_score=0,
                     lenient=0, strict=0, cores_ok=False,
                     judge_prompt_sha256=None, judge_accepted_attempt=None,
                     judge_accepted_content=None,
                     judge_attempt_count=0, judge_attempts=[], judge_usage_total={
                         "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                     }, judge_response_model=None, judge_usage=None)
        return grade

    grade["compile_check"] = await asyncio.to_thread(
        sandbox.check,
        answer,
        corpus.language,
        expected_configuration=sandbox_config,
    )
    if (not isinstance(grade["compile_check"], dict)
            or type(grade["compile_check"].get("compile_ok")) is not bool
            or not isinstance(grade["compile_check"].get("detail"), str)
            or grade["compile_check"].get("configuration_sha256")
            != sandbox_config_sha256):
        raise GradeIntegrityError(
            "deterministic checker returned an invalid or unbound result; "
            "judge was not contacted")
    judge_prompt = build_prompt(corpus, row, answer, whole_files)
    grade["judge_prompt_sha256"] = sha256_bytes(judge_prompt.encode("utf-8"))

    last_error = None
    attempts: list[dict[str, Any]] = []
    for attempt in range(MAX_JUDGE_ATTEMPTS):
        try:
            resp = await client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": judge_prompt}],
                response_format=judge_schema(row),
                **({"reasoning_effort": effort} if effort else {}),
            )
        except Exception as exc:
            audit = _failed_judge_audit(
                ep=ep,
                episode_sha256=episode_sha256,
                grading_spec_sha256=grading_spec_sha256,
                judge_model=judge_model,
                judge_prompt_sha256=grade["judge_prompt_sha256"],
                attempts=attempts,
                failure=exc,
                request_attempt_count=attempt + 1,
            )
            raise JudgeAttemptsFailed(
                f"judge request {attempt + 1} failed after "
                f"{len(attempts)} invalid verdict(s)", audit
            ) from exc
        try:
            attempt_record, content, response_error, response_is_fatal = _response_attempt(
                resp, attempt + 1)
        except Exception as exc:
            response_error = GradeIntegrityError(
                f"received judge response could not be audited ({type(exc).__name__}: {exc})")
            attempt_record = _uninspectable_response_attempt(attempt + 1, response_error)
            content = None
            response_is_fatal = True
        if response_error is not None:
            last_error = response_error
            if isinstance(content, str):
                attempt_record["invalid_content"] = content
            attempt_record["validation_error"] = _error_record(response_error)
            attempts.append(attempt_record)
            log.warning("%s/%s/r%d judge attempt %d/%d incomplete: %s",
                        ep["budget"], ep["qid"], ep["rollout"], attempt + 1,
                        MAX_JUDGE_ATTEMPTS, response_error)
            if response_is_fatal:
                audit = _failed_judge_audit(
                    ep=ep,
                    episode_sha256=episode_sha256,
                    grading_spec_sha256=grading_spec_sha256,
                    judge_model=judge_model,
                    judge_prompt_sha256=grade["judge_prompt_sha256"],
                    attempts=attempts,
                    failure=response_error,
                    request_attempt_count=attempt + 1,
                )
                raise JudgeAttemptsFailed(
                    f"judge response {attempt + 1} had incomplete identity or usage; "
                    "no retry made and no grade written",
                    audit,
                ) from response_error
            continue
        try:
            verdict = parse_json(content, label="judge verdict")
            claims, claim_scores = validate_verdict(row, verdict)
        except GradeIntegrityError as exc:
            last_error = exc
            attempt_record["invalid_content"] = content
            attempt_record["validation_error"] = _error_record(exc)
            attempts.append(attempt_record)
            log.warning("%s/%s/r%d judge attempt %d/%d invalid: %s",
                        ep["budget"], ep["qid"], ep["rollout"], attempt + 1,
                        MAX_JUDGE_ATTEMPTS, exc)
            continue
        attempt_record["accepted"] = True
        attempts.append(attempt_record)
        break
    else:
        audit = _failed_judge_audit(
            ep=ep,
            episode_sha256=episode_sha256,
            grading_spec_sha256=grading_spec_sha256,
            judge_model=judge_model,
            judge_prompt_sha256=grade["judge_prompt_sha256"],
            attempts=attempts,
            failure=last_error,
        )
        raise JudgeAttemptsFailed(
            f"{ep['budget']}/{ep['qid']}/r{ep['rollout']}: judge returned "
            f"{MAX_JUDGE_ATTEMPTS} invalid verdicts; no grade written",
            audit,
        ) from last_error

    scores = score_from_claims(row, claim_scores,
                               grade["compile_check"]["compile_ok"])
    grade.update(
        claims=claims,
        needs_regrade=False,
        question_score=scores["lenient"],
        judge_question_score=scores["lenient"],
        judge_accepted_attempt=len(attempts),
        judge_accepted_content=content,
        judge_attempt_count=len(attempts),
        judge_attempts=attempts,
        judge_usage_total=_usage_total(attempts),
        judge_response_model=attempts[-1]["response_model"],
        judge_usage=attempts[-1]["usage"],
        **scores,
    )
    return grade


def validate_stored_grade(grade: dict, row: dict, ep: dict, *,
                          episode_sha256: str, grading_spec_sha256: str,
                          judge_model: str, judge_base_url: str | None = None,
                          corpus=None, whole_files: bool = False,
                          source_episode: str | None = None,
                          recheck_checker: bool = True) -> None:
    """Validate provenance and recompute every stored deterministic score."""
    validate_episode(ep, row)
    if not isinstance(grade, dict):
        raise GradeIntegrityError("grade is not an object")
    if grade.get("grade_schema_version") != GRADE_SCHEMA_VERSION:
        raise GradeIntegrityError("grade schema is legacy or unknown")
    if grade.get("episode_sha256") != episode_sha256:
        raise GradeIntegrityError("grade episode hash does not match the run file")
    if grade.get("grading_spec_sha256") != grading_spec_sha256:
        raise GradeIntegrityError("grade was produced by a different grading specification")
    if grade.get("judge_model") != judge_model:
        raise GradeIntegrityError("grade judge model does not match the requested judge")
    judge_base_url = _resolve_judge_base_url(judge_model, judge_base_url)
    if grade.get("judge_base_url") != judge_base_url:
        raise GradeIntegrityError("grade judge endpoint does not match the requested judge")
    try:
        graded_at = datetime.fromisoformat(grade.get("graded_at", ""))
    except (TypeError, ValueError) as exc:
        raise GradeIntegrityError("grade timestamp is invalid") from exc
    if graded_at.tzinfo is None:
        raise GradeIntegrityError("grade timestamp has no timezone")
    if source_episode is not None and grade.get("source_episode") != source_episode:
        raise GradeIntegrityError("grade source_episode does not match its run path")

    for key in ("task", "qid", "budget", "rollout"):
        if grade.get(key) != ep.get(key):
            raise GradeIntegrityError(f"grade {key} does not match the episode")
    for key in ("manifest_sha256", "question_sha256", "prompt_sha256", "note_sha256"):
        if grade.get(key) != ep.get(key):
            raise GradeIntegrityError(f"grade {key} does not match the episode")
    if grade.get("episode_status") != ep["status"]:
        raise GradeIntegrityError("grade episode_status does not match the episode")
    if grade.get("gen_tokens") != ep["gen_tokens"]:
        raise GradeIntegrityError("grade generated-token count does not match the episode")
    if grade.get("needs_regrade") is not False:
        raise GradeIntegrityError("stored grade is marked needs_regrade")

    if corpus is None:
        raise GradeIntegrityError(
            "corpus is required to validate deterministic checker provenance")
    expected_sandbox_configuration = sandbox_configuration_record(corpus.language)
    expected_sandbox_sha256 = stable_sha256(expected_sandbox_configuration)
    compile_check = grade.get("compile_check")
    if (not isinstance(compile_check, dict)
            or type(compile_check.get("compile_ok")) is not bool
            or not isinstance(compile_check.get("detail"), str)
            or compile_check.get("configuration_sha256")
            != expected_sandbox_sha256):
        raise GradeIntegrityError("stored grade has an invalid compile check")

    if ep["status"] == "no_answer":
        if "judge_accepted_content" not in grade:
            raise GradeIntegrityError(
                "no_answer grade is missing its accepted-content marker")
        expected = {
            "claims": [], "question_score": 0, "judge_question_score": 0,
            "lenient": 0, "strict": 0, "cores_ok": False,
            "judge_requested_model": judge_model,
            "judge_prompt_sha256": None, "judge_accepted_attempt": None,
            "judge_accepted_content": None,
            "judge_attempt_count": 0, "judge_attempts": [],
            "judge_usage_total": {
                "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            },
            "judge_response_model": None, "judge_usage": None,
        }
        for key, value in expected.items():
            if grade.get(key) != value:
                raise GradeIntegrityError(f"no_answer grade has invalid {key}")
        if compile_check["compile_ok"]:
            raise GradeIntegrityError("no_answer grade passed the compile check")
        if compile_check.get("detail") != "empty answer":
            raise GradeIntegrityError("no_answer grade has a non-canonical compile detail")
        return

    if type(recheck_checker) is not bool:
        raise GradeIntegrityError("recheck_checker must be a boolean")
    if recheck_checker:
        try:
            observed_compile_check = sandbox.check(
                ep.get("answer", ""),
                corpus.language,
                expected_configuration=expected_sandbox_configuration,
            )
        except Exception as exc:
            raise GradeIntegrityError(
                f"deterministic checker could not be independently rerun: {exc}"
            ) from exc
        if (
            not isinstance(observed_compile_check, dict)
            or type(observed_compile_check.get("compile_ok")) is not bool
            or observed_compile_check.get("configuration_sha256")
            != expected_sandbox_sha256
            or observed_compile_check["compile_ok"] != compile_check["compile_ok"]
        ):
            raise GradeIntegrityError(
                "stored compile outcome does not match an independent deterministic rerun"
            )

    _validate_judge_audit(grade, corpus, row, ep, judge_model, whole_files)

    verdict = {
        "claims": grade.get("claims"),
        "question_score": grade.get("question_score"),
        "needs_regrade": grade.get("needs_regrade"),
    }
    canonical_claims, claim_scores = validate_verdict(row, verdict)
    if grade["claims"] != canonical_claims:
        raise GradeIntegrityError("stored claims are not in canonical rubric order")
    scores = score_from_claims(row, claim_scores, compile_check["compile_ok"])
    for key, value in scores.items():
        if grade.get(key) != value:
            raise GradeIntegrityError(f"stored {key} does not match recomputation")
    if grade.get("judge_question_score") != scores["lenient"]:
        raise GradeIntegrityError("stored judge_question_score does not match recomputation")


def stored_grade_is_current(grade_path: Path, episode_path: Path, corpus, row: dict,
                            judge_model: str, whole_files: bool = False,
                            effort: str = "", *,
                            judge_base_url: str | None = None) -> bool:
    """Content-based replacement for unreliable mtime freshness checks."""
    try:
        episode_bytes = read_artifact_bytes(episode_path)
        ep = parse_json(episode_bytes, label=f"episode {episode_path}")
        grade = parse_json(read_artifact_bytes(grade_path), label=f"grade {grade_path}")
        validate_stored_grade(
            grade, row, ep,
            episode_sha256=sha256_bytes(episode_bytes),
            grading_spec_sha256=grade_spec_sha256(
                corpus, row, judge_model, whole_files, effort,
                judge_base_url=judge_base_url),
            judge_model=judge_model,
            judge_base_url=judge_base_url,
            corpus=corpus,
            whole_files=whole_files,
            source_episode=episode_path.relative_to(ROOT).as_posix(),
        )
    except (OSError, KeyError, GradeIntegrityError, ValueError):
        return False
    return True


def preflight_grade_population(*, runs_root: Path, out_root: Path, corpus,
                               questions: list[dict], manifest_context: dict,
                               judge_model: str, whole_files: bool,
                               effort: str,
                               judge_base_url: str | None = None) -> list[dict[str, Any]]:
    """Validate the entire run grid before allowing the first judge request."""
    rows = {question["id"]: question for question in questions}
    run_task_root = runs_root / corpus.name
    expected_run_files = {
        run_task_root / relative for relative in manifest_context["expected_episodes"]
    }
    actual_run_files = set(run_task_root.glob("*/r*/*.json"))
    expected_grade_files = {
        out_root / corpus.name / relative
        for relative in manifest_context["expected_episodes"]
    }
    actual_grade_files = {
        path for path in (out_root / corpus.name).rglob("*") if path.is_file()
    }
    errors = []
    for path in sorted(expected_run_files - actual_run_files):
        errors.append(f"missing run episode: {path.relative_to(ROOT)}")
    for path in sorted(actual_run_files - expected_run_files):
        errors.append(f"unexpected run episode: {path.relative_to(ROOT)}")
    for path in sorted(actual_grade_files - expected_grade_files):
        errors.append(f"unexpected grade outside manifest grid: {path.relative_to(ROOT)}")

    records = []
    for relative in manifest_context["expected_episodes"]:
        run_path = run_task_root / relative
        grade_path = out_root / corpus.name / relative
        if not run_path.is_file():
            continue
        try:
            episode_bytes = read_artifact_bytes(run_path)
            episode = parse_json(episode_bytes, label=f"episode {run_path}")
            if not isinstance(episode, dict):
                raise GradeIntegrityError("episode is not an object")
            budget, rollout_dir, filename = relative.split("/")
            expected_identity = {
                "task": corpus.name,
                "qid": filename.removesuffix(".json"),
                "budget": budget,
                "rollout": int(rollout_dir.removeprefix("r")),
            }
            for field, expected_value in expected_identity.items():
                if episode.get(field) != expected_value:
                    raise GradeIntegrityError(
                        f"episode {field}={episode.get(field)!r}; "
                        f"path requires {expected_value!r}")
            row = rows[expected_identity["qid"]]
            validate_episode(episode, row)
            validate_manifest_episode(episode, row, manifest_context)
            spec_sha256 = grade_spec_sha256(
                corpus, row, judge_model, whole_files, effort,
                judge_base_url=judge_base_url)
            source_episode = run_path.relative_to(ROOT).as_posix()
        except (OSError, KeyError, TypeError, ValueError, GradeIntegrityError) as exc:
            errors.append(f"invalid run episode {run_path.relative_to(ROOT)}: {exc}")
            continue

        if grade_path.exists() and not grade_path.is_file():
            errors.append(
                f"grade path exists but is not a file: {grade_path.relative_to(ROOT)}")
            continue
        if grade_path.is_file():
            try:
                stored = parse_json(
                    read_artifact_bytes(grade_path), label=f"stored grade {grade_path}")
                validate_stored_grade(
                    stored, row, episode,
                    episode_sha256=sha256_bytes(episode_bytes),
                    grading_spec_sha256=spec_sha256,
                    judge_model=judge_model,
                    judge_base_url=judge_base_url,
                    corpus=corpus,
                    whole_files=whole_files,
                    source_episode=source_episode,
                )
            except (OSError, KeyError, TypeError, ValueError, GradeIntegrityError) as exc:
                errors.append(
                    f"existing grade is stale or invalid and was preserved: "
                    f"{grade_path.relative_to(ROOT)} ({exc}); choose a new --grade-id")
            continue
        records.append({
            "run_path": run_path,
            "grade_path": grade_path,
            "episode_bytes": episode_bytes,
            "episode": episode,
            "row": row,
            "grading_spec_sha256": spec_sha256,
            "source_episode": source_episode,
        })

    if errors:
        preview = "\n".join(f"  - {error}" for error in errors[:20])
        remainder = (f"\n  - ... and {len(errors) - 20} more"
                     if len(errors) > 20 else "")
        raise GradeIntegrityError(
            f"refusing to contact the judge: {len(errors)} run preflight failure(s):\n"
            f"{preview}{remainder}")
    return records


async def main_async(args):
    corpus = CORPORA[args.task]
    questions = load_questions(args.task)
    rows = {q["id"]: q for q in questions}
    if len(rows) != len(questions):
        raise GradeIntegrityError(f"{args.task}: duplicate question ids")
    for row in questions:
        rubric_ids(row)
    grader = os.environ.get("GRADER_MODEL", "openai")
    if grader not in GRADERS:
        raise GradeIntegrityError(
            f"unknown GRADER_MODEL={grader!r}; choose one of {sorted(GRADERS)}")
    judge_model, judge_base_url, key_var = GRADERS[grader]
    judge_base_url = _resolve_judge_base_url(judge_model, judge_base_url)
    log.info(
        "grader=%s judge_model=%s judge_base_url=%s",
        grader, judge_model, judge_base_url,
    )
    runs_root = ROOT / "runs" / args.run_id
    judge_dir = (judge_model + ("-wholefiles" if args.whole_files else "-excerpts")
                 + (f"-effort-{args.judge_effort}" if args.judge_effort else ""))
    grade_id = args.grade_id or judge_dir
    out_root = ROOT / "grades" / args.run_id / grade_id

    manifest_context = load_claim_manifest(runs_root / args.task, corpus, questions)
    if manifest_context["spec"]["run_id"] != runs_root.name:
        raise GradeIntegrityError("run manifest ID does not match its directory")
    validate_preregistered_grading_policy(
        manifest_context["preregistration"],
        grader=grader,
        judge_model=judge_model,
        whole_files=args.whole_files,
        effort=args.judge_effort,
    )
    sandbox_config = sandbox_configuration_record(corpus.language)
    if sandbox_config.get("ready") is not True:
        raise GradeIntegrityError(
            "configured deterministic checker is not claim-ready: "
            f"{sandbox_config.get('error') or sandbox_config}")
    pending = preflight_grade_population(
        runs_root=runs_root,
        out_root=out_root,
        corpus=corpus,
        questions=questions,
        manifest_context=manifest_context,
        judge_model=judge_model,
        whole_files=args.whole_files,
        effort=args.judge_effort,
        judge_base_url=judge_base_url,
    )
    log.info("%d episodes to grade (task=%s)", len(pending), args.task)

    client = None
    if any(record["episode"]["status"] == "ok" for record in pending):
        api_key = os.environ.get(key_var)
        if not api_key:
            raise GradeIntegrityError(
                f"{key_var} is required for {grader} grading; no judge request made")
        client = _make_grader_client(grader, api_key)

    sem = asyncio.Semaphore(args.concurrency)
    done = 0
    failures = []

    async def one(record):
        nonlocal done
        async with sem:
            try:
                rf = record["run_path"]
                gf = record["grade_path"]
                lock_path = (
                    out_root / ".locks" / gf.relative_to(out_root)
                ).with_suffix(".lock")
                episode_bytes = record["episode_bytes"]
                ep = record["episode"]
                row = record["row"]
                with exclusive_process_lock(lock_path):
                    if read_artifact_bytes(rf) != episode_bytes:
                        raise GradeIntegrityError(
                            "episode changed after the global preflight")
                    if gf.exists():
                        if not gf.is_file():
                            raise GradeIntegrityError(
                                "grade path became a non-file after preflight")
                        current = parse_json(
                            read_artifact_bytes(gf), label=f"concurrently written grade {gf}")
                        validate_stored_grade(
                            current, row, ep,
                            episode_sha256=sha256_bytes(episode_bytes),
                            grading_spec_sha256=record["grading_spec_sha256"],
                            judge_model=judge_model,
                            judge_base_url=judge_base_url,
                            corpus=corpus,
                            whole_files=args.whole_files,
                            source_episode=record["source_episode"],
                        )
                        done += 1
                        log.info(
                            "[%d/%d] %s became current after the global preflight",
                            done, len(pending), gf,
                        )
                        return
                    grade = await grade_episode(
                        client, judge_model, corpus, row, ep, args.whole_files,
                        args.judge_effort,
                        episode_sha256=sha256_bytes(episode_bytes),
                        grading_spec_sha256=record["grading_spec_sha256"],
                        judge_base_url=judge_base_url,
                    )
                    grade["source_episode"] = record["source_episode"]
                    validate_stored_grade(
                        grade, row, ep,
                        episode_sha256=sha256_bytes(episode_bytes),
                        grading_spec_sha256=record["grading_spec_sha256"],
                        judge_model=judge_model,
                        judge_base_url=judge_base_url,
                        corpus=corpus,
                        whole_files=args.whole_files,
                        source_episode=grade["source_episode"],
                        # grade_episode just ran this exact checker under the
                        # bound configuration.  Existing artifacts and reports
                        # independently rerun it; doing so here would execute
                        # generated code twice before the immutable write.
                        recheck_checker=False,
                    )
                    if read_artifact_bytes(rf) != episode_bytes:
                        raise GradeIntegrityError(
                            "episode changed while its judge request ran")
                    write_immutable_json(gf, grade)
            except JudgeAttemptsFailed as exc:
                run_path = record["run_path"]
                audit_path = write_failed_judge_audit(
                    out_root, record["source_episode"], exc.audit)
                log.exception(
                    "grading %s failed (no grade written; judge audit saved to %s)",
                    run_path, audit_path,
                )
                failures.append((run_path, exc))
                return
            except Exception as exc:
                run_path = record["run_path"]
                log.exception("grading %s failed (no grade written; rerun to retry)", run_path)
                failures.append((run_path, exc))
                return
            done += 1
            log.info("[%d/%d] %s/%s/r%d lenient=%.0f strict=%.0f compile_ok=%s%s",
                     done, len(pending), grade["budget"], grade["qid"], grade["rollout"],
                     grade["lenient"], grade["strict"],
                     grade["compile_check"]["compile_ok"],
                     " NEEDS_REGRADE" if grade.get("needs_regrade") else "")
            if args.debug:
                log.debug("claims for %s/r%d: %s", grade["qid"], grade["rollout"],
                          json.dumps(grade["claims"], indent=2))

    try:
        await asyncio.gather(*(one(record) for record in pending))
    finally:
        if client is not None:
            await client.close()
    if failures:
        raise RuntimeError(
            f"grading failed for {len(failures)}/{len(pending)} episodes; "
            "no new grade was written for those episodes")
    log.info("all done: %d grades current", len(manifest_context["expected_episodes"]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True, choices=list(CORPORA))
    p.add_argument("--run-id", required=True,
                   help="immutable claim-ready run ID under runs/")
    p.add_argument(
        "--grade-id",
        help="immutable output namespace (default: judge/config name); choose a new one to regrade",
    )
    p.add_argument("--concurrency", type=int, default=8)
    evidence = p.add_mutually_exclusive_group(required=True)
    evidence.add_argument(
        "--whole-files", dest="whole_files", action="store_true",
        help="paper-faithful A.5 judge context: full evidence files",
    )
    evidence.add_argument(
        "--excerpt-evidence", dest="whole_files", action="store_false",
        help="local diagnostic variant: only dataset evidence excerpts",
    )
    p.add_argument("--judge-effort", default="",
                   choices=["", "low", "medium", "high", "xhigh"],
                   help="judge reasoning effort (default: API default)")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()
    try:
        args.run_id = validate_id(args.run_id)
        if args.grade_id is not None:
            args.grade_id = validate_id(args.grade_id, "grade ID")
    except ValueError as exc:
        p.error(str(exc))
    if args.concurrency <= 0:
        p.error("--concurrency must be positive")

    load_private_env(ROOT / ".env")
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(
                      ROOT / "logs" / (
                          f"grade-{args.run_id}-{args.grade_id or 'default'}-{args.task}.log"
                      ))],
    )
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
