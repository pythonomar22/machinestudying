"""Immutable, fail-closed preregistration contracts for confirmatory runs.

The checked-in JSON document fixes both experimental arms and every choice that
can affect generation, grading, analysis, or stopping.  Its ``source_commit``
names the implementation commit immediately before preregistration metadata was
added.  At execution time, HEAD must be its single-parent direct child, and that
one commit may only add direct ``preregistrations/*.json`` files.  This avoids
the impossible requirement that a Git commit contain its own hash while still
proving that code and existing preregistrations did not change afterward.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Any

from .dataset import ROOT
from .integrity import (
    canonical_json_bytes,
    read_artifact_bytes,
    sha256_bytes,
    strict_json_loads,
)


PREREGISTRATION_SCHEMA_VERSION = 1
_ID = re.compile(r"[a-z0-9][a-z0-9._-]{2,79}\Z")
_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_GIT_OID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")

RUN_FAILURE_POLICY = {
    "model_no_answer": "intention-to-run_zero",
    "infrastructure_error": "invalid_until_retried",
    "forced_short": "invalid_until_retried",
}

_TOP_LEVEL_KEYS = {
    "schema_version",
    "preregistration_id",
    "hypothesis",
    "intervention",
    "task",
    "corpus_commit",
    "source_commit",
    "question_bundle_sha256",
    "arms",
    "evaluation",
    "failure_policy",
    "grading_policy",
    "analysis_policy",
    "stopping_policy",
}
_EVALUATION_KEYS = {
    "harness",
    "model",
    "model_revision",
    "sampling",
    "master_seed",
    "seed_namespace",
    "seed_group",
    "budgets",
    "rollouts",
}
_GRADING_KEYS = {
    "grader",
    "judge_model",
    "evidence_mode",
    "judge_effort",
    "claim_scoring",
    "question_scoring",
}
_ANALYSIS_KEYS = {
    "primary_estimand",
    "primary_metric",
    "confidence_interval",
    "bootstrap_replicates",
    "bootstrap_seed",
    "multiplicity_policy",
}
_STOPPING_KEYS = {"population", "interim_looks", "stopping_rule"}
_BOUND_RECORD_KEYS = {
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


class PreregistrationError(ValueError):
    """A preregistration is malformed, mutable, or not bound to a run."""


@dataclass(frozen=True)
class LoadedPreregistration:
    path: Path
    relative_path: str
    data: bytes
    sha256: str
    document: dict[str, Any]
    head_commit: str


def _object(value: object, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise PreregistrationError(f"{label} must contain exactly {sorted(keys)}")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise PreregistrationError(f"{label} must be a nonempty, trimmed string")
    return value


def _integer(value: object, label: str, *, positive: bool = False) -> int:
    if type(value) is not int or (positive and value <= 0):
        qualifier = "positive " if positive else ""
        raise PreregistrationError(f"{label} must be a {qualifier}JSON integer")
    return value


def _hash(value: object, pattern: re.Pattern[str], label: str) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise PreregistrationError(f"{label} is not a lowercase immutable hash")
    return value


def _same_json(left: object, right: object) -> bool:
    """Compare JSON identities without Python's ``True == 1`` coercion."""

    try:
        return canonical_json_bytes(left) == canonical_json_bytes(right)
    except (TypeError, ValueError):
        return False


def validate_preregistration(document: object) -> dict[str, Any]:
    """Validate and return one strict schema-version-1 document."""

    try:
        canonical_json_bytes(document)
    except (TypeError, ValueError) as error:
        raise PreregistrationError("preregistration is not canonicalizable JSON") from error
    root = _object(document, _TOP_LEVEL_KEYS, "preregistration")
    if (
        type(root.get("schema_version")) is not int
        or root["schema_version"] != PREREGISTRATION_SCHEMA_VERSION
    ):
        raise PreregistrationError("preregistration schema_version must be JSON integer 1")

    preregistration_id = _string(root["preregistration_id"], "preregistration_id")
    if not _ID.fullmatch(preregistration_id):
        raise PreregistrationError("preregistration_id is not a safe lowercase identifier")
    _string(root["hypothesis"], "hypothesis")
    _string(root["intervention"], "intervention")
    task = _string(root["task"], "task")
    if not _SAFE_COMPONENT.fullmatch(task):
        raise PreregistrationError("task is not a safe identifier")
    _hash(root["corpus_commit"], _GIT_OID, "corpus_commit")
    _hash(root["source_commit"], _GIT_OID, "source_commit")
    _hash(root["question_bundle_sha256"], _SHA256, "question_bundle_sha256")

    arms = _object(root["arms"], {"control", "treatment"}, "arms")
    run_ids: list[str] = []
    for role in ("control", "treatment"):
        arm = _object(arms[role], {"run_id", "note_sha256"}, f"arms.{role}")
        run_id = _string(arm["run_id"], f"arms.{role}.run_id")
        if not _ID.fullmatch(run_id):
            raise PreregistrationError(f"arms.{role}.run_id is not a safe identifier")
        run_ids.append(run_id)
        note_sha256 = arm["note_sha256"]
        if note_sha256 is not None:
            _hash(note_sha256, _SHA256, f"arms.{role}.note_sha256")
    if len(set(run_ids)) != 2:
        raise PreregistrationError("control and treatment must use distinct run IDs")

    evaluation = _object(root["evaluation"], _EVALUATION_KEYS, "evaluation")
    _string(evaluation["harness"], "evaluation.harness")
    _string(evaluation["model"], "evaluation.model")
    _hash(evaluation["model_revision"], _GIT_OID, "evaluation.model_revision")
    if not isinstance(evaluation["sampling"], dict) or not evaluation["sampling"]:
        raise PreregistrationError("evaluation.sampling must be a nonempty JSON object")
    _integer(evaluation["master_seed"], "evaluation.master_seed")
    _string(evaluation["seed_namespace"], "evaluation.seed_namespace")
    seed_group = _string(evaluation["seed_group"], "evaluation.seed_group")
    if not _ID.fullmatch(seed_group):
        raise PreregistrationError("evaluation.seed_group is not a safe identifier")
    budgets = evaluation["budgets"]
    if (
        not isinstance(budgets, list)
        or not budgets
        or any(not isinstance(value, str) or not _SAFE_COMPONENT.fullmatch(value)
               for value in budgets)
        or len(budgets) != len(set(budgets))
    ):
        raise PreregistrationError("evaluation.budgets must be unique safe strings")
    if budgets != ["direct", "k5", "k20", "k20f"]:
        raise PreregistrationError(
            "evaluation.budgets must use the implemented confirmatory report grid"
        )
    _integer(evaluation["rollouts"], "evaluation.rollouts", positive=True)

    failure_policy = _object(
        root["failure_policy"], set(RUN_FAILURE_POLICY), "failure_policy"
    )
    if not _same_json(failure_policy, RUN_FAILURE_POLICY):
        raise PreregistrationError("failure_policy is not the implemented intention-to-run policy")

    grading = _object(root["grading_policy"], _GRADING_KEYS, "grading_policy")
    if not isinstance(grading["grader"], str) or grading["grader"] not in {
        "openai",
        "fugu",
    }:
        raise PreregistrationError("grading_policy.grader must be openai or fugu")
    expected_judge = {"openai": "gpt-5.4", "fugu": "fugu"}[grading["grader"]]
    if grading["judge_model"] != expected_judge:
        raise PreregistrationError(
            "grading_policy.judge_model does not match the selected grader"
        )
    if not isinstance(grading["evidence_mode"], str) or grading[
        "evidence_mode"
    ] not in {"whole_files", "excerpt_evidence"}:
        raise PreregistrationError("grading_policy.evidence_mode is unsupported")
    if not isinstance(grading["judge_effort"], str) or grading[
        "judge_effort"
    ] not in {"", "low", "medium", "high", "xhigh"}:
        raise PreregistrationError("grading_policy.judge_effort is unsupported")
    if grading["claim_scoring"] != "binary_0_1":
        raise PreregistrationError("grading_policy.claim_scoring must be binary_0_1")
    if grading["question_scoring"] != "weighted_claim_sum":
        raise PreregistrationError(
            "grading_policy.question_scoring must be weighted_claim_sum"
        )

    analysis = _object(root["analysis_policy"], _ANALYSIS_KEYS, "analysis_policy")
    if analysis["primary_estimand"] != "treatment_minus_control":
        raise PreregistrationError(
            "analysis_policy.primary_estimand must be treatment_minus_control"
        )
    if analysis["primary_metric"] != "expertise_lenient":
        raise PreregistrationError(
            "analysis_policy.primary_metric must be expertise_lenient"
        )
    if (
        analysis["confidence_interval"]
        != "paired_two_stage_question_then_rollout_percentile_95"
    ):
        raise PreregistrationError("analysis_policy.confidence_interval is unsupported")
    _integer(
        analysis["bootstrap_replicates"],
        "analysis_policy.bootstrap_replicates",
        positive=True,
    )
    _integer(analysis["bootstrap_seed"], "analysis_policy.bootstrap_seed")
    if analysis["multiplicity_policy"] != "single_preregistered_primary_no_adjustment":
        raise PreregistrationError(
            "analysis_policy.multiplicity_policy is not the implemented primary analysis"
        )

    stopping = _object(root["stopping_policy"], _STOPPING_KEYS, "stopping_policy")
    if stopping["population"] != "complete_manifest_grid":
        raise PreregistrationError("stopping_policy.population must be complete_manifest_grid")
    if type(stopping.get("interim_looks")) is not int or stopping["interim_looks"] != 0:
        raise PreregistrationError("stopping_policy.interim_looks must be JSON integer 0")
    if stopping["stopping_rule"] != "no_outcome_dependent_stopping":
        raise PreregistrationError(
            "stopping_policy.stopping_rule must be no_outcome_dependent_stopping"
        )
    return root


def _git(root: Path, *arguments: str) -> bytes:
    try:
        process = subprocess.run(
            ["git", "-C", str(root), *arguments],
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise PreregistrationError(f"cannot inspect preregistration Git state: {error}") from error
    if process.returncode:
        message = process.stderr.decode("utf-8", errors="replace").strip()
        raise PreregistrationError(
            f"git {' '.join(arguments)} failed while validating preregistration: {message}"
        )
    return process.stdout


def load_preregistration(
    path: Path, *, root: Path = ROOT
) -> LoadedPreregistration:
    """Load a canonical, committed preregistration whose code baseline is frozen."""

    root = Path(root).absolute()
    path = Path(path)
    path = path.absolute() if path.is_absolute() else (root / path).absolute()
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise PreregistrationError("preregistration must be inside the project root") from error
    if (
        len(relative.parts) != 2
        or relative.parts[0] != "preregistrations"
        or path.suffix != ".json"
    ):
        raise PreregistrationError(
            "preregistration must be a direct preregistrations/*.json file"
        )

    try:
        data = read_artifact_bytes(path)
    except (OSError, ValueError) as error:
        raise PreregistrationError(f"cannot read safe preregistration: {path}") from error
    try:
        document = strict_json_loads(data, label="preregistration")
    except ValueError as error:
        raise PreregistrationError(str(error)) from error
    document = validate_preregistration(document)
    if canonical_json_bytes(document) != data:
        raise PreregistrationError("preregistration bytes must use canonical JSON encoding")
    if path.name != f"{document['preregistration_id']}.json":
        raise PreregistrationError(
            "preregistration filename must equal <preregistration_id>.json"
        )

    top_level = _git(root, "rev-parse", "--show-toplevel").decode().strip()
    if Path(top_level).absolute() != root:
        raise PreregistrationError("preregistration root is not the Git worktree root")
    relative_posix = relative.as_posix()
    _git(root, "ls-files", "--error-unmatch", "--", relative_posix)
    if _git(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        relative_posix,
    ):
        raise PreregistrationError("preregistration source must be clean and committed")
    if _git(root, "show", f"HEAD:{relative_posix}") != data:
        raise PreregistrationError("preregistration bytes differ from the committed HEAD blob")

    head_commit = _git(root, "rev-parse", "HEAD").decode().strip()
    source_commit = document["source_commit"]
    if source_commit == head_commit:
        raise PreregistrationError(
            "source_commit must name the pre-preregistration implementation commit"
        )
    try:
        commit_and_parents = (
            _git(root, "rev-list", "--parents", "-n", "1", head_commit)
            .decode("ascii")
            .split()
        )
    except UnicodeDecodeError as error:
        raise PreregistrationError("Git returned a non-ASCII commit identity") from error
    if commit_and_parents != [head_commit, source_commit]:
        raise PreregistrationError(
            "execution HEAD must be the single-parent direct child of source_commit"
        )

    changed_raw = _git(
        root,
        "diff",
        "--name-status",
        "--no-renames",
        "-z",
        f"{source_commit}..{head_commit}",
        "--",
    )
    changed_fields = changed_raw.split(b"\0")
    if changed_fields and changed_fields[-1] == b"":
        changed_fields.pop()
    if len(changed_fields) % 2:
        raise PreregistrationError("Git returned malformed preregistration diff metadata")

    changed: list[str] = []
    for offset in range(0, len(changed_fields), 2):
        try:
            status = changed_fields[offset].decode("ascii")
            changed_path = changed_fields[offset + 1].decode("utf-8")
        except UnicodeDecodeError as error:
            raise PreregistrationError(
                "preregistration commit contains a non-UTF-8 path"
            ) from error
        if status != "A":
            raise PreregistrationError(
                "execution commit may only add direct preregistrations/*.json files; "
                f"found {status} {changed_path}"
            )
        logical = PurePosixPath(changed_path)
        if (
            len(logical.parts) != 2
            or logical.parts[0] != "preregistrations"
            or logical.suffix != ".json"
        ):
            raise PreregistrationError(
                "execution commit may only add direct preregistrations/*.json files; "
                f"found A {changed_path}"
            )
        changed.append(changed_path)

    if relative_posix not in changed:
        raise PreregistrationError("source_commit does not predate this preregistration")
    preregistration_commits = [
        value
        for value in _git(root, "log", "--format=%H", "--", relative_posix)
        .decode("utf-8")
        .splitlines()
        if value
    ]
    if len(preregistration_commits) != 1:
        raise PreregistrationError(
            "preregistration must be introduced once and never modified or renamed"
        )
    return LoadedPreregistration(
        path=path,
        relative_path=relative_posix,
        data=data,
        sha256=sha256_bytes(data),
        document=document,
        head_commit=head_commit,
    )


def _validate_runtime_binding(
    document: dict[str, Any],
    *,
    role: str,
    run_id: object,
    task: object,
    corpus_commit: object,
    source_head_commit: object,
    question_bundle_sha256: object,
    harness: object,
    model: object,
    model_revision: object,
    sampling: object,
    master_seed: object,
    seed_namespace: object,
    seed_group: object,
    budgets: object,
    rollouts: object,
    failure_policy: object,
    note_sha256: object,
) -> None:
    validate_preregistration(document)
    if not isinstance(role, str) or role not in {"control", "treatment"}:
        raise PreregistrationError("preregistration role must be control or treatment")
    if not isinstance(source_head_commit, str) or not _GIT_OID.fullmatch(source_head_commit):
        raise PreregistrationError("executed source commit is not an immutable Git hash")
    if type(master_seed) is not int:
        raise PreregistrationError("run master_seed must be a JSON integer")
    if type(rollouts) is not int:
        raise PreregistrationError("run rollouts must be a JSON integer")
    expected = document["evaluation"]
    arm = document["arms"][role]
    identities = {
        "run_id": (run_id, arm["run_id"]),
        "task": (task, document["task"]),
        "corpus_commit": (corpus_commit, document["corpus_commit"]),
        "question_bundle_sha256": (
            question_bundle_sha256,
            document["question_bundle_sha256"],
        ),
        "harness": (harness, expected["harness"]),
        "model": (model, expected["model"]),
        "model_revision": (model_revision, expected["model_revision"]),
        "master_seed": (master_seed, expected["master_seed"]),
        "seed_namespace": (seed_namespace, expected["seed_namespace"]),
        "seed_group": (seed_group, expected["seed_group"]),
        "rollouts": (rollouts, expected["rollouts"]),
        "note_sha256": (note_sha256, arm["note_sha256"]),
    }
    for label, (observed, preregistered) in identities.items():
        if type(observed) is not type(preregistered) or observed != preregistered:
            raise PreregistrationError(f"run {label} differs from the preregistration")
    for label, observed, preregistered in (
        ("sampling", sampling, expected["sampling"]),
        ("budgets", budgets, expected["budgets"]),
        ("failure_policy", failure_policy, document["failure_policy"]),
    ):
        if not _same_json(observed, preregistered):
            raise PreregistrationError(f"run {label} differs from the preregistration")


def bind_preregistration(
    path: Path,
    *,
    role: str,
    run_id: object,
    task: object,
    corpus_commit: object,
    source_head_commit: object,
    question_bundle_sha256: object,
    harness: object,
    model: object,
    model_revision: object,
    sampling: object,
    master_seed: object,
    seed_namespace: object,
    seed_group: object,
    budgets: object,
    rollouts: object,
    failure_policy: object,
    note_sha256: object,
    root: Path = ROOT,
) -> LoadedPreregistration:
    """Load and exactly bind one committed preregistration to runtime inputs."""

    loaded = load_preregistration(path, root=root)
    if loaded.head_commit != source_head_commit:
        raise PreregistrationError(
            "executed source commit differs from the preregistration worktree HEAD"
        )
    _validate_runtime_binding(
        loaded.document,
        role=role,
        run_id=run_id,
        task=task,
        corpus_commit=corpus_commit,
        source_head_commit=source_head_commit,
        question_bundle_sha256=question_bundle_sha256,
        harness=harness,
        model=model,
        model_revision=model_revision,
        sampling=sampling,
        master_seed=master_seed,
        seed_namespace=seed_namespace,
        seed_group=seed_group,
        budgets=budgets,
        rollouts=rollouts,
        failure_policy=failure_policy,
        note_sha256=note_sha256,
    )
    return loaded


def revalidate_run_preregistration(
    spec: object, run_root: Path
) -> dict[str, Any]:
    """Revalidate a stored bound preregistration without Git or any writes.

    Grading and comparison code can call this on an immutable run manifest.  It
    validates the snapshotted bytes, normalized binding record, and every run
    identity covered by the preregistration.
    """

    if not isinstance(spec, dict):
        raise PreregistrationError("run spec is not an object")
    record = _object(
        spec.get("preregistration"), _BOUND_RECORD_KEYS, "run preregistration record"
    )
    if (
        type(record.get("schema_version")) is not int
        or record["schema_version"] != PREREGISTRATION_SCHEMA_VERSION
        or record.get("status") != "bound"
    ):
        raise PreregistrationError("run does not contain a bound schema-version-1 preregistration")
    document = validate_preregistration(record["document"])
    expected_source = f"preregistrations/{document['preregistration_id']}.json"
    expected_snapshot = f"inputs/preregistration-{record.get('sha256')}.json"
    if record.get("source_path") != expected_source or record.get("snapshot") != expected_snapshot:
        raise PreregistrationError("run preregistration paths are not normalized")
    if type(record.get("bytes")) is not int or record["bytes"] < 0:
        raise PreregistrationError("run preregistration byte count is not a JSON integer")
    document_bytes = canonical_json_bytes(document)
    if (
        not isinstance(record.get("sha256"), str)
        or not _SHA256.fullmatch(record["sha256"])
        or record["bytes"] != len(document_bytes)
        or record["sha256"] != sha256_bytes(document_bytes)
    ):
        raise PreregistrationError("run preregistration document identity is inconsistent")
    snapshot_path = Path(run_root).absolute() / PurePosixPath(expected_snapshot)
    try:
        snapshot_bytes = read_artifact_bytes(snapshot_path)
    except (OSError, ValueError) as error:
        raise PreregistrationError("run preregistration snapshot is missing or unsafe") from error
    if snapshot_bytes != document_bytes:
        raise PreregistrationError("run preregistration snapshot bytes changed")

    corpus = spec.get("corpus")
    source = spec.get("source")
    seed_policy = spec.get("seed_policy")
    note = spec.get("note")
    extra = spec.get("extra")
    if not all(
        isinstance(value, dict) for value in (corpus, source, seed_policy, extra)
    ):
        raise PreregistrationError("run spec lacks corpus, source, seed, or model identities")
    if (
        extra.get("model_revision") != spec.get("model_revision")
        or not isinstance(extra.get("expected_response_model"), str)
        or not extra["expected_response_model"]
    ):
        raise PreregistrationError("run model identity diverges from its episode identity")
    executed_source_commit = source.get("git_commit")
    if record.get("executed_source_commit") != executed_source_commit:
        raise PreregistrationError("run preregistration executed source commit drifted")
    note_sha256 = note.get("sha256") if isinstance(note, dict) else None
    _validate_runtime_binding(
        document,
        role=record.get("role"),
        run_id=spec.get("run_id"),
        task=spec.get("task"),
        corpus_commit=corpus.get("commit"),
        source_head_commit=executed_source_commit,
        question_bundle_sha256=spec.get("question_bundle_sha256"),
        harness=spec.get("harness"),
        model=spec.get("model"),
        model_revision=spec.get("model_revision"),
        sampling=spec.get("sampling"),
        master_seed=spec.get("master_seed"),
        seed_namespace=seed_policy.get("namespace"),
        seed_group=seed_policy.get("seed_group"),
        budgets=spec.get("budgets"),
        rollouts=spec.get("rollouts"),
        failure_policy=spec.get("failure_policy"),
        note_sha256=note_sha256,
    )
    return document
