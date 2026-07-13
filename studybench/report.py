"""Aggregate a complete, provenance-checked evaluation population into per-budget
accuracy, mean generated tokens, and expertise (weighted AUC per Appendix C).
Paper Table 1 values can be shown as a contextual reference only after explicit
known-regime checks. They are never labelled a byte-exact replication because
the original dependency/tool-cap artifact was not archived here.
Each grade embeds its episode's gen_tokens and status, so scores and tokens come
from one population.

Score definitions (author-confirmed, docs/jacob.md: "lenient is just weights
summed together"; the core-conjunctive rule from the earlier DMs belongs to strict):
  lenient  = raw weighted claim sum, no gates; THE Table 1 comparison.
  len-cc   = weighted sum if every core claim scored 1, else 0 (core-conjunctive;
             reported for the strict-adjacent analysis, NOT Table 1).
  strict   = core-conjunctive plus the compile-gate zero.

The expertise formula was verified against the paper: with x = log10(tokens/3000),
w(x) = ln(10)·10^(-x), the weight of the segment between consecutive budgets is
3000/tok_i - 3000/tok_{i+1}; performance is the best-score-so-far envelope; the region
below the first budget is floored to 0 and the last score carries the tail. It
reproduces the worked example (10.8) and DSPy base (6.49) exactly; the paper's own
Table 1 values for OpenClaw base give 7.66 vs the published 7.64, consistent with the
table's tokens being rounded to 0.1k.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import mean

from .dataset import CORPORA, ROOT, load_questions
from .grade import (FAILED_JUDGE_AUDIT_SCHEMA_VERSION, GRADE_SCHEMA_VERSION,
                    GRADERS, MAX_JUDGE_ATTEMPTS, GradeIntegrityError, file_sha256,
                    episode_provider_identity, grade_spec_sha256,
                    grader_identity_for_model, judge_usage_summary,
                    load_claim_manifest, parse_json, sha256_bytes,
                    validate_judge_attempt_record, validate_manifest_episode,
                    validate_stored_grade)
from .grade import validate_preregistered_grading_policy
from .integrity import (canonical_json_bytes, read_artifact_bytes, sha256_json,
                        write_immutable_json)
from .provenance import validate_id

BUDGET_ORDER = ["direct", "k5", "k20", "k20f"]

DIAGNOSTIC_BANNER = (
    "\n"
    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
    "!!! DIAGNOSTIC ONLY: LEGACY/PARTIAL/UNVERIFIED ARTIFACTS             !!!\n"
    "!!! These numbers are not valid research results and must not be cited. !!!\n"
    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
)


class ReportIntegrityError(RuntimeError):
    """The requested result population is incomplete, stale, or unverifiable."""


REPORT_SCHEMA_VERSION = 4

PAPER = {  # Table 1, lenient: (task, variant) -> budget -> (acc %, tokens k)
    ("dspy", ""): {"direct": (3.3, 4.1), "k5": (8.6, 7.9), "k20": (9.6, 8.6),
                   "k20f": (29.4, 34.6), "expertise": 6.49},
    ("openclaw", ""): {"direct": (2.3, 4.1), "k5": (6.9, 4.6), "k20": (15.8, 9.7),
                       "k20f": (17.6, 24.3), "expertise": 7.64},
    ("dspy", "cheatsheet"): {"direct": (6.3, 3.9), "k5": (14.4, 6.1), "k20": (14.1, 7.1),
                             "k20f": (23.1, 29.9), "expertise": 9.65},
    ("openclaw", "cheatsheet"): {"direct": (4.3, 3.8), "k5": (8.6, 6.0), "k20": (15.2, 9.1),
                                 "k20f": (18.1, 20.1), "expertise": 8.18},
}
PAPER[("dspy", "react")] = PAPER[("dspy", "")]  # react = the paper's own base harness
PAPER[("openclaw", "react")] = PAPER[("openclaw", "")]
PAPER[("dspy", "react-cheatsheet")] = PAPER[("dspy", "cheatsheet")]
PAPER[("openclaw", "react-cheatsheet")] = PAPER[("openclaw", "cheatsheet")]

PAPER_MODEL = "openai/Qwen/Qwen3.5-9B"
PAPER_MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
PAPER_SAMPLING = {
    "temperature": 1.0,
    "top_p": 0.95,
    "max_tokens": 32768,
    "presence_penalty": 1.5,
    "extra_body": {
        "top_k": 20,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
    },
}
PAPER_REFERENCE_LIMITATIONS = [
    "the exact original DSPy dependency artifact is not archived locally",
    "not every original tool-output cap and harness implementation byte is known",
    "the public 50-question benchmark has been reused across adaptive local arms",
]


def paper_comparability_errors(audit: dict, *, variant: str,
                               judge_model: str, whole_files: bool) -> list[str]:
    """Check known regime fields before displaying Table 1 as context."""
    spec = audit["run_manifest"]["spec"]
    errors = []
    extra = spec.get("extra") if isinstance(spec.get("extra"), dict) else {}
    checks = {
        "harness is not dspy.ReAct": spec.get("harness") == "dspy.ReAct",
        "generation model differs": spec.get("model") == PAPER_MODEL,
        "generation model revision differs": (
            extra.get("model_revision") == PAPER_MODEL_REVISION),
        "provider-returned generation model differs": (
            audit.get("generation_runtime", {}).get("response_models")
            == [extra.get("expected_response_model")]),
        "sampling configuration differs": spec.get("sampling") == PAPER_SAMPLING,
        "paper requires exactly three rollouts": spec.get("rollouts") == 3,
        "paper budget grid differs": spec.get("budgets") == BUDGET_ORDER,
        "paper grader is GPT-5.4": judge_model == "gpt-5.4",
        "paper A.5 comparison requires whole evidence files": whole_files is True,
    }
    errors.extend(message for message, passed in checks.items() if not passed)
    note_present = spec.get("note") is not None
    if variant == "base" and note_present:
        errors.append("paper base comparison requires no study note")
    if variant == "cheatsheet":
        if not note_present:
            errors.append("paper cheatsheet comparison requires a study note")
        if audit.get("note_provenance", {}).get("method") != "forced-50-cheatsheet":
            errors.append("study note is not the paper's forced-50 cheatsheet method")
    return errors


def expertise(points: list[tuple[float, float]]) -> float:
    """Weighted AUC from (mean_tokens, accuracy) budget points; 3k-token anchor."""
    pts = sorted(p for p in points if p[0] > 0)
    e, best = 0.0, 0.0
    for i, (tok, acc) in enumerate(pts):
        best = max(best, acc)
        next_w = min(3000 / pts[i + 1][0], 1.0) if i + 1 < len(pts) else 0.0
        e += (min(3000 / tok, 1.0) - next_w) * best
    return e


def _legacy_aggregate(task: str, grades_dir: str = "grades",
                      runs_dir: str = "runs") -> dict:
    """Historical count-only behavior, available only in diagnostic mode."""
    budgets = {}
    for budget in BUDGET_ORDER:
        gdir = ROOT / grades_dir / task / budget
        grades = [json.loads(f.read_text()) for f in sorted(gdir.rglob("*.json"))] \
            if gdir.exists() else []
        if not grades:
            continue
        n_runs = len(list((ROOT / runs_dir / task / budget).rglob("*.json")))
        if n_runs != len(grades):
            print(f"WARNING: {task}/{budget} has {n_runs} runs but {len(grades)} grades "
                  "— aggregating the graded subset only")
        budgets[budget] = {
            "n": len(grades),
            "lenient": mean(g["lenient"] for g in grades),
            "len_cc": mean(g["lenient"] if g["cores_ok"] else 0 for g in grades),
            "strict": mean(g["strict"] for g in grades),
            "compile_rate": mean(g["compile_check"]["compile_ok"] for g in grades),
            "needs_regrade": sum(bool(g.get("needs_regrade")) for g in grades),
            "tokens": mean(g["gen_tokens"] for g in grades),
            "no_answer": sum(g.get("episode_status") == "no_answer" for g in grades),
            "bad_episodes": sum(g["episode_status"] != "ok" for g in grades),
        }
    out = {"budgets": budgets}
    for kind in ("lenient", "strict"):
        pts = [(b["tokens"], b[kind]) for b in budgets.values()]
        out[f"expertise_{kind}"] = expertise(pts) if len(pts) == 4 else None
    return out


def _rooted(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _expected_paths(root: Path, task: str, qids: list[str],
                    rollouts: int) -> dict[tuple[str, int, str], Path]:
    return {
        (budget, rollout, qid):
            root / task / budget / f"r{rollout}" / f"{qid}.json"
        for budget in BUDGET_ORDER
        for rollout in range(rollouts)
        for qid in qids
    }


def _inventory_failed_attempts(run_task_root: Path, rows: dict[str, dict],
                               manifest_context: dict) -> tuple[list[dict], list[str]]:
    """Validate and inventory retained infrastructure attempts outside the ITT grid."""
    root = run_task_root / "failed-attempts"
    if not root.exists():
        return [], []
    if not root.is_dir() or root.is_symlink():
        return [], [f"failed-attempt root is not a real directory: {_display_path(root)}"]
    records = []
    errors = []
    attempts_by_episode: dict[str, list[int]] = {}
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        if path.suffix != ".json":
            errors.append(f"unexpected non-JSON failed-attempt artifact: {_display_path(path)}")
            continue
        try:
            relative = path.relative_to(root)
            if len(relative.parts) != 4:
                raise GradeIntegrityError("failed-attempt path has the wrong depth")
            budget, rollout_dir, qid, filename = relative.parts
            if not rollout_dir.startswith("r") or not rollout_dir[1:].isdecimal():
                raise GradeIntegrityError("failed-attempt path has an invalid rollout")
            if not filename.startswith("attempt-") or not filename[:-5][8:].isdecimal():
                raise GradeIntegrityError("failed-attempt path has an invalid attempt index")
            rollout = int(rollout_dir[1:])
            attempt = int(filename[:-5][8:])
            expected_episode = f"{budget}/{rollout_dir}/{qid}.json"
            if expected_episode not in manifest_context["expected_episodes"]:
                raise GradeIntegrityError("failed attempt is outside the manifest grid")
            artifact_bytes = read_artifact_bytes(path)
            artifact = parse_json(artifact_bytes, label=f"failed attempt {path}")
            if not isinstance(artifact, dict):
                raise GradeIntegrityError("failed attempt is not an object")
            expected_identity = {
                "task": manifest_context["spec"]["task"],
                "qid": qid,
                "budget": budget,
                "rollout": rollout,
                "failure_attempt": attempt,
                "expected_episode": expected_episode,
            }
            for field, expected_value in expected_identity.items():
                if artifact.get(field) != expected_value:
                    raise GradeIntegrityError(
                        f"failed attempt {field} does not match its path")
            if type(artifact.get("rollout")) is not int or type(
                    artifact.get("failure_attempt")) is not int:
                raise GradeIntegrityError("failed attempt has invalid integer identity")
            if artifact.get("status") not in {"error", "forced_short"}:
                raise GradeIntegrityError("failed attempt has an unknown or final status")
            validate_manifest_episode(artifact, rows[qid], manifest_context)
            attempts_by_episode.setdefault(expected_episode, []).append(attempt)
            records.append({
                "path": _display_path(path),
                "sha256": sha256_bytes(artifact_bytes),
                "expected_episode": expected_episode,
                "attempt": attempt,
                "status": artifact.get("status"),
            })
        except (OSError, KeyError, TypeError, ValueError, GradeIntegrityError) as exc:
            errors.append(f"invalid failed attempt {_display_path(path)}: {exc}")
    for expected_episode, attempts in attempts_by_episode.items():
        if sorted(attempts) != list(range(1, len(attempts) + 1)):
            errors.append(f"failed-attempt sequence has gaps for {expected_episode}")
    records.sort(key=lambda record: (record["expected_episode"], record["attempt"]))
    return records, errors


def _valid_sha256(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _inventory_failed_judge_audits(
    grade_root: Path,
    task: str,
    expected_runs: dict[tuple[str, int, str], Path],
    manifest_context: dict,
    grading_specs: dict[str, str],
    judge_model: str,
) -> tuple[list[dict], list[str]]:
    """Validate and disclose judge calls that produced no grade."""
    root = grade_root / "failed-judge-audits" / task
    if not root.exists():
        return [], []
    if not root.is_dir() or root.is_symlink():
        return [], [f"failed-judge root is not a real directory: {_display_path(root)}"]
    records = []
    errors = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        try:
            if path.suffix != ".json":
                raise GradeIntegrityError("failed-judge audit is not JSON")
            relative = path.relative_to(root)
            if len(relative.parts) != 3:
                raise GradeIntegrityError("failed-judge audit path has the wrong depth")
            budget, rollout_dir, filename = relative.parts
            if not rollout_dir.startswith("r") or not rollout_dir[1:].isdecimal():
                raise GradeIntegrityError("failed-judge audit path has an invalid rollout")
            stem = filename.removesuffix(".json")
            if len(stem) < 66 or stem[-65] != "-" or not _valid_sha256(stem[-64:]):
                raise GradeIntegrityError("failed-judge audit filename is not content-addressed")
            qid, filename_digest = stem[:-65], stem[-64:]
            rollout = int(rollout_dir[1:])
            key = (budget, rollout, qid)
            run_path = expected_runs.get(key)
            if run_path is None:
                raise GradeIntegrityError("failed-judge audit is outside the manifest grid")

            artifact_bytes = read_artifact_bytes(path)
            artifact = parse_json(artifact_bytes, label=f"failed-judge audit {path}")
            if not isinstance(artifact, dict):
                raise GradeIntegrityError("failed-judge audit is not an object")
            if artifact_bytes != canonical_json_bytes(artifact):
                raise GradeIntegrityError("failed-judge audit is not canonically encoded")
            if sha256_json(artifact) != filename_digest:
                raise GradeIntegrityError("failed-judge audit filename hash does not match")
            identity = {
                "failed_judge_audit_schema_version": FAILED_JUDGE_AUDIT_SCHEMA_VERSION,
                "task": task,
                "qid": qid,
                "budget": budget,
                "rollout": rollout,
                "source_episode": _display_path(run_path),
            }
            for field, value in identity.items():
                if artifact.get(field) != value:
                    raise GradeIntegrityError(
                        f"failed-judge audit {field} does not match its path")
            for field in (
                "episode_sha256", "grading_spec_sha256", "manifest_sha256",
                "question_sha256", "prompt_sha256", "judge_prompt_sha256",
            ):
                if not _valid_sha256(artifact.get(field)):
                    raise GradeIntegrityError(
                        f"failed-judge audit has invalid {field}")
            note_hash = artifact.get("note_sha256")
            if note_hash is not None and not _valid_sha256(note_hash):
                raise GradeIntegrityError("failed-judge audit has invalid note_sha256")
            requested_model = artifact.get("judge_requested_model")
            if not isinstance(requested_model, str) or not requested_model:
                raise GradeIntegrityError("failed-judge audit has no requested model")

            attempts = artifact.get("judge_attempts")
            request_count = artifact.get("judge_request_attempt_count")
            attempt_count = artifact.get("judge_attempt_count")
            if (not isinstance(attempts, list)
                    or type(request_count) is not int
                    or not 1 <= request_count <= MAX_JUDGE_ATTEMPTS
                    or type(attempt_count) is not int
                    or attempt_count != len(attempts)
                    or not 0 <= attempt_count <= request_count):
                raise GradeIntegrityError("failed-judge request/response counts are invalid")
            response_models = set()
            system_fingerprints = set()
            missing_system_fingerprints = 0
            incomplete_response_fields = set()
            for index, attempt in enumerate(attempts, 1):
                validate_judge_attempt_record(attempt, index, accepted=False)
                if attempt.get("response_model"):
                    response_models.add(attempt["response_model"])
                if attempt["system_fingerprint_status"] == "available":
                    system_fingerprints.add(attempt["system_fingerprint"])
                else:
                    missing_system_fingerprints += 1
                if isinstance(attempt.get("incomplete_response"), dict):
                    incomplete_response_fields.update(attempt["incomplete_response"])
            usage_summary = judge_usage_summary(attempts, request_count)
            status = usage_summary["status"]
            if (request_count not in {attempt_count, attempt_count + 1}
                    or artifact.get("judge_usage_status") != status
                    or artifact.get("judge_usage_total") != usage_summary["total"]
                    or artifact.get("judge_usage_known_total")
                    != usage_summary["known_total"]):
                raise GradeIntegrityError("failed-judge cumulative usage is invalid")
            failure = artifact.get("failure")
            if (not isinstance(failure, dict)
                    or not isinstance(failure.get("type"), str)
                    or not isinstance(failure.get("message"), str)):
                raise GradeIntegrityError("failed-judge failure record is invalid")

            episode_bytes = read_artifact_bytes(run_path)
            episode = parse_json(episode_bytes, label=f"episode {run_path}")
            current_bindings = {
                "episode": artifact["episode_sha256"] == sha256_bytes(episode_bytes),
                "grading_spec": artifact["grading_spec_sha256"] == grading_specs[qid],
                "manifest": artifact["manifest_sha256"] == manifest_context["manifest_sha256"],
                "question": artifact["question_sha256"]
                == manifest_context["question_sha256"][qid],
                "presented_prompt": artifact["prompt_sha256"]
                == manifest_context["prompt_sha256"][qid],
                "note": artifact["note_sha256"] == manifest_context["note_sha256"],
                "judge_model": requested_model == judge_model,
            }
            records.append({
                "path": _display_path(path),
                "sha256": sha256_bytes(artifact_bytes),
                "source_episode": artifact["source_episode"],
                "qid": qid,
                "budget": budget,
                "rollout": rollout,
                "judge_requested_model": requested_model,
                "judge_response_models": sorted(response_models),
                "judge_system_fingerprints": sorted(system_fingerprints),
                "missing_judge_system_fingerprint_calls": missing_system_fingerprints,
                "incomplete_response_fields": sorted(incomplete_response_fields),
                "judge_request_attempt_count": request_count,
                "judge_attempt_count": attempt_count,
                "judge_usage_status": status,
                "judge_usage_total": usage_summary["total"],
                "judge_usage_known_total": usage_summary["known_total"],
                "current_bindings": current_bindings,
                "all_bindings_current": all(current_bindings.values()),
            })
        except (OSError, KeyError, TypeError, ValueError, GradeIntegrityError) as exc:
            errors.append(f"invalid failed-judge audit {_display_path(path)}: {exc}")
    records.sort(key=lambda record: (
        BUDGET_ORDER.index(record["budget"]), record["rollout"],
        record["qid"], record["path"],
    ))
    return records, errors


def _load_complete_evaluation(task: str, grades_dir: str | Path,
                              runs_dir: str | Path, *, rollouts: int | None,
                              judge_model: str, whole_files: bool = False,
                              effort: str = "") -> tuple[dict[str, list[dict]], dict]:
    """Load one exact evaluation grid and its content-addressed audit record."""
    if rollouts is not None and (type(rollouts) is not int or rollouts <= 0):
        raise ReportIntegrityError("rollouts must be a positive integer")
    if task not in CORPORA:
        raise ReportIntegrityError(f"unknown task {task!r}")

    rows_list = load_questions(task)
    rows = {row["id"]: row for row in rows_list}
    if len(rows) != len(rows_list):
        raise ReportIntegrityError(f"{task}: dataset has duplicate question ids")
    qids = list(rows)
    run_root = _rooted(runs_dir)
    grade_root = _rooted(grades_dir)
    corpus = CORPORA[task]
    try:
        manifest_context = load_claim_manifest(run_root / task, corpus, rows_list)
    except GradeIntegrityError as exc:
        raise ReportIntegrityError(f"{task}: invalid claim-ready run manifest: {exc}") from exc
    try:
        grader, judge_base_url = grader_identity_for_model(judge_model)
    except GradeIntegrityError as exc:
        raise ReportIntegrityError(
            "strict reporting does not recognize the requested preregistered judge"
        ) from exc
    try:
        validate_preregistered_grading_policy(
            manifest_context["preregistration"],
            grader=grader,
            judge_model=judge_model,
            whole_files=whole_files,
            effort=effort,
        )
    except GradeIntegrityError as exc:
        raise ReportIntegrityError(
            f"{task}: grading configuration differs from preregistration: {exc}"
        ) from exc
    run_id = manifest_context["spec"]["run_id"]
    if run_root.name != run_id:
        raise ReportIntegrityError("run manifest ID does not match its directory")
    if grade_root.parent.name != run_id:
        raise ReportIntegrityError("grade directory is not scoped to the manifest run ID")
    manifest_rollouts = manifest_context["spec"]["rollouts"]
    if rollouts is not None and rollouts != manifest_rollouts:
        raise ReportIntegrityError(
            f"declared rollouts={rollouts}, manifest requires {manifest_rollouts}")
    rollouts = manifest_rollouts
    if manifest_context["spec"]["budgets"] != BUDGET_ORDER:
        raise ReportIntegrityError(
            f"expertise requires budgets {BUDGET_ORDER}, manifest has "
            f"{manifest_context['spec']['budgets']}")
    expected_runs = _expected_paths(run_root, task, qids, rollouts)
    expected_grades = _expected_paths(grade_root, task, qids, rollouts)
    manifest_paths = {
        run_root / task / relative for relative in manifest_context["expected_episodes"]
    }
    if set(expected_runs.values()) != manifest_paths:
        raise ReportIntegrityError("run manifest does not describe the complete benchmark grid")
    expected_run_set = set(expected_runs.values())
    expected_grade_set = set(expected_grades.values())
    actual_runs = set((run_root / task).glob("*/r*/*.json"))
    actual_grades = {
        path for path in (grade_root / task).rglob("*") if path.is_file()
    }

    errors = []
    failed_attempts, failed_attempt_errors = _inventory_failed_attempts(
        run_root / task, rows, manifest_context)
    errors.extend(failed_attempt_errors)
    for path in sorted(expected_run_set - actual_runs):
        errors.append(f"missing run: {_display_path(path)}")
    for path in sorted(expected_grade_set - actual_grades):
        errors.append(f"missing grade: {_display_path(path)}")
    for path in sorted(actual_runs - expected_run_set):
        errors.append(
            f"unexpected run outside the declared {rollouts}-rollout grid: "
            f"{_display_path(path)}")
    for path in sorted(actual_grades - expected_grade_set):
        errors.append(
            f"unexpected grade outside the declared {rollouts}-rollout grid: "
            f"{_display_path(path)}")

    population = {budget: [] for budget in BUDGET_ORDER}
    population_records = []
    grading_specs = {
        qid: grade_spec_sha256(
            corpus,
            row,
            judge_model,
            whole_files,
            effort,
            judge_base_url=judge_base_url,
        )
        for qid, row in rows.items()
    }
    failed_judge_audits, failed_judge_errors = _inventory_failed_judge_audits(
        grade_root, task, expected_runs, manifest_context, grading_specs, judge_model)
    errors.extend(failed_judge_errors)
    response_models = set()
    # Only the final accepted attempt determines a stored score. Rejected
    # attempts remain visible in the grade/retry audit, but must not be allowed
    # to make two graders look revision-matched.
    judge_system_fingerprints = set()
    accepted_judge_system_fingerprint_by_episode = {}
    missing_judge_system_fingerprints = 0
    generation_models = set()
    generation_fingerprints = set()
    missing_generation_fingerprints = 0
    environment_snapshot_by_episode = {}
    for key, run_path in expected_runs.items():
        budget, rollout, qid = key
        grade_path = expected_grades[key]
        if not run_path.is_file() or not grade_path.is_file():
            continue
        try:
            episode_bytes = read_artifact_bytes(run_path)
            grade_bytes = read_artifact_bytes(grade_path)
            ep = parse_json(episode_bytes, label=f"episode {run_path}")
            grade = parse_json(grade_bytes, label=f"grade {grade_path}")
            expected_identity = {
                "task": task, "qid": qid, "budget": budget, "rollout": rollout,
            }
            for field, value in expected_identity.items():
                if ep.get(field) != value:
                    raise GradeIntegrityError(
                        f"episode {field}={ep.get(field)!r}; path requires {value!r}")
            validate_manifest_episode(ep, rows[qid], manifest_context)
            validate_stored_grade(
                grade, rows[qid], ep,
                episode_sha256=sha256_bytes(episode_bytes),
                grading_spec_sha256=grading_specs[qid],
                judge_model=judge_model,
                judge_base_url=judge_base_url,
                corpus=corpus,
                whole_files=whole_files,
                source_episode=_display_path(run_path),
            )
            population[budget].append(grade)
            generation_identity = episode_provider_identity(ep)
            generation_models.update(generation_identity["response_models"])
            generation_fingerprints.update(generation_identity["system_fingerprints"])
            missing_generation_fingerprints += generation_identity[
                "missing_system_fingerprint_calls"]
            relative = f"{budget}/r{rollout}/{qid}.json"
            environment_snapshot_by_episode[relative] = ep[
                "environment_snapshot"
            ]["sha256"]
            if grade["episode_status"] == "ok":
                response_models.add(grade["judge_response_model"])
                accepted_attempt = grade["judge_attempts"][-1]
                if accepted_attempt["system_fingerprint_status"] == "available":
                    fingerprint = accepted_attempt["system_fingerprint"]
                    judge_system_fingerprints.add(fingerprint)
                    accepted_judge_system_fingerprint_by_episode[relative] = fingerprint
                else:
                    missing_judge_system_fingerprints += 1
                    accepted_judge_system_fingerprint_by_episode[relative] = None
            population_records.append({
                "task": task,
                "qid": qid,
                "budget": budget,
                "rollout": rollout,
                "episode_path": _display_path(run_path),
                "episode_sha256": sha256_bytes(episode_bytes),
                "grade_path": _display_path(grade_path),
                "grade_sha256": sha256_bytes(grade_bytes),
            })
        except (OSError, json.JSONDecodeError, KeyError, ValueError,
                GradeIntegrityError) as exc:
            errors.append(f"invalid {_display_path(grade_path)}: {exc}")

    if len(response_models) > 1:
        errors.append(
            "provider resolved the requested judge alias to multiple models: "
            f"{sorted(response_models)}")
    if len(generation_models) > 1:
        errors.append(
            "generation provider resolved the requested model to multiple models: "
            f"{sorted(generation_models)}")
    extra = manifest_context["spec"].get("extra")
    expected_generation_model = (
        extra.get("expected_response_model") if isinstance(extra, dict) else None
    )
    if (not isinstance(expected_generation_model, str)
            or not expected_generation_model
            or generation_models != {expected_generation_model}):
        errors.append(
            "generation response model does not match the run manifest: "
            f"expected={expected_generation_model!r}, observed={sorted(generation_models)}")

    if errors:
        preview = "\n".join(f"  - {error}" for error in errors[:20])
        remainder = (f"\n  - ... and {len(errors) - 20} more"
                     if len(errors) > 20 else "")
        raise ReportIntegrityError(
            f"{task}: refusing to aggregate {len(errors)} integrity failure(s):\n"
            f"{preview}{remainder}\n"
            "Use --legacy-partial only for conspicuously labeled diagnostics.")
    population_records.sort(
        key=lambda record: (
            BUDGET_ORDER.index(record["budget"]), record["rollout"], record["qid"])
    )
    grading_config = {
        "grade_schema_version": GRADE_SCHEMA_VERSION,
        "judge_requested_model": judge_model,
        "judge_base_url": judge_base_url,
        "judge_response_models": sorted(response_models),
        # Fingerprints identify mutable serving builds, not a stable model revision.
        # This summary and per-episode map contain accepted attempts only.
        "judge_system_fingerprint_scope": "accepted_final_attempts_only",
        "judge_system_fingerprints": sorted(judge_system_fingerprints),
        "accepted_judge_system_fingerprint_by_episode": (
            accepted_judge_system_fingerprint_by_episode
        ),
        "missing_judge_system_fingerprint_calls": missing_judge_system_fingerprints,
        "whole_files": whole_files,
        "judge_effort": effort,
        "grading_spec_sha256_by_question": grading_specs,
    }
    note_manifest = manifest_context.get("note_manifest")
    note_config = (
        note_manifest.get("config")
        if isinstance(note_manifest, dict) and isinstance(note_manifest.get("config"), dict)
        else {}
    )
    audit = {
        "run_manifest": {
            "path": _display_path(run_root / task / "manifest.json"),
            "sha256": manifest_context["manifest_sha256"],
            "spec_sha256": sha256_json(manifest_context["spec"]),
            "spec": manifest_context["spec"],
        },
        "grading_manifest": {
            "sha256": sha256_json(grading_config),
            "config": grading_config,
        },
        "generation_runtime": {
            "response_models": sorted(generation_models),
            "system_fingerprints": sorted(generation_fingerprints),
            "missing_system_fingerprint_calls": missing_generation_fingerprints,
            "environment_snapshot_scope": "final_manifest_episodes",
            "environment_snapshot_sha256s": sorted(
                set(environment_snapshot_by_episode.values())
            ),
            "environment_snapshot_sha256_by_episode": (
                environment_snapshot_by_episode
            ),
        },
        "note_provenance": {
            "construction_manifest_sha256": manifest_context.get(
                "note_construction_manifest_sha256"),
            "study_id": note_manifest.get("study_id")
            if isinstance(note_manifest, dict) else None,
            "method": note_manifest.get("method", note_config.get("method"))
            if isinstance(note_manifest, dict) else None,
            "manifest_type": note_manifest.get("manifest_type")
            if isinstance(note_manifest, dict) else None,
        },
        "failed_attempts": {
            "count": len(failed_attempts),
            "sha256": sha256_json(failed_attempts),
            "artifacts": failed_attempts,
        },
        "failed_judge_audits": {
            "count": len(failed_judge_audits),
            "sha256": sha256_json(failed_judge_audits),
            "artifacts": failed_judge_audits,
        },
        "population": population_records,
        "population_sha256": sha256_json(population_records),
    }
    return population, audit


def load_complete_evaluation(task: str, grades_dir: str | Path,
                             runs_dir: str | Path, *, rollouts: int | None,
                             judge_model: str, whole_files: bool = False,
                             effort: str = "") -> tuple[dict[str, list[dict]], dict]:
    """Public strict loader returning both the population and complete audit."""
    return _load_complete_evaluation(
        task, grades_dir, runs_dir, rollouts=rollouts,
        judge_model=judge_model, whole_files=whole_files, effort=effort)


def load_complete_population(task: str, grades_dir: str | Path,
                             runs_dir: str | Path, *, rollouts: int | None,
                             judge_model: str, whole_files: bool = False,
                             effort: str = "") -> dict[str, list[dict]]:
    """Load one exact evaluation grid, rejecting every unverifiable artifact."""
    population, _ = load_complete_evaluation(
        task, grades_dir, runs_dir, rollouts=rollouts,
        judge_model=judge_model, whole_files=whole_files, effort=effort)
    return population


def aggregate_population(population: dict[str, list[dict]]) -> dict:
    budgets = {}
    for budget in BUDGET_ORDER:
        grades = population[budget]
        budgets[budget] = {
            "n": len(grades),
            "lenient": mean(g["lenient"] for g in grades),
            "len_cc": mean(g["lenient"] if g["cores_ok"] else 0 for g in grades),
            "strict": mean(g["strict"] for g in grades),
            "compile_rate": mean(g["compile_check"]["compile_ok"] for g in grades),
            "needs_regrade": 0,
            "tokens": mean(g["gen_tokens"] for g in grades),
            "no_answer": sum(g["episode_status"] == "no_answer" for g in grades),
            "bad_episodes": 0,
        }
    out = {"budgets": budgets}
    for kind in ("lenient", "strict"):
        points = [(budgets[budget]["tokens"], budgets[budget][kind])
                  for budget in BUDGET_ORDER]
        out[f"expertise_{kind}"] = expertise(points)
    return out


def aggregate(task: str, grades_dir: str = "grades", runs_dir: str = "runs",
              *, rollouts: int | None = None, judge_model: str | None = None,
              whole_files: bool = False, effort: str = "",
              legacy_partial: bool = False) -> dict:
    if legacy_partial:
        return _legacy_aggregate(task, grades_dir, runs_dir)
    if judge_model is None:
        raise ReportIntegrityError("strict aggregation requires an explicit judge_model")
    population = load_complete_population(
        task, grades_dir, runs_dir, rollouts=rollouts,
        judge_model=judge_model, whole_files=whole_files, effort=effort)
    return aggregate_population(population)


def _bootstrap_data(data: dict, n_boot: int, seed: int) -> dict:
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    qids = sorted(data[BUDGET_ORDER[0]])
    if not qids:
        raise ReportIntegrityError("cannot bootstrap an empty population")
    expected_qids = set(qids)
    if any(set(data[budget]) != expected_qids for budget in BUDGET_ORDER):
        raise ReportIntegrityError("bootstrap budgets do not contain the same questions")
    rollout_counts = {
        len(data[budget][qid]) for budget in BUDGET_ORDER for qid in qids
    }
    if len(rollout_counts) != 1 or 0 in rollout_counts:
        raise ReportIntegrityError("bootstrap cells do not have one complete rollout count")

    rng = random.Random(seed)
    stats = {b: [] for b in BUDGET_ORDER} | {"wauc": [], "wauc_cc": []}
    for _ in range(n_boot):
        qs = rng.choices(qids, k=len(qids))
        pts, pts_cc = [], []
        for budget in BUDGET_ORDER:
            cc = rub = tok = n = 0
            for qid in qs:
                pool = data[budget][qid]
                for ep in rng.choices(pool, k=len(pool)):
                    cc += ep[0]
                    rub += ep[1]
                    tok += ep[2]
                    n += 1
            stats[budget].append(rub / n)
            pts.append((tok / n, rub / n))
            pts_cc.append((tok / n, cc / n))
        stats["wauc"].append(expertise(pts))
        stats["wauc_cc"].append(expertise(pts_cc))

    def ci(xs):
        xs = sorted(xs)
        return xs[round(0.025 * (len(xs) - 1))], xs[round(0.975 * (len(xs) - 1))]

    return {key: (mean(values), *ci(values)) for key, values in stats.items()}


def bootstrap_population(population: dict[str, list[dict]], n_boot: int,
                         seed: int = 0) -> dict:
    """Two-stage question/rollout bootstrap over a validated population."""
    data = {}
    for budget in BUDGET_ORDER:
        episodes = {}
        for grade in population[budget]:
            episodes.setdefault(grade["qid"], []).append(
                (grade["lenient"] if grade["cores_ok"] else 0,
                 grade["lenient"], grade["gen_tokens"]))
        data[budget] = episodes
    return _bootstrap_data(data, n_boot, seed)


def _legacy_bootstrap(task: str, n_boot: int, seed: int = 0,
                      grades_dir: str = "grades") -> dict:
    """Historical unverified bootstrap, diagnostic mode only."""
    data = {}  # budget -> qid -> [(lenient_cc, rubric, tokens)]
    for budget in BUDGET_ORDER:
        eps = {}
        for f in sorted((ROOT / grades_dir / task / budget).rglob("*.json")):
            g = json.loads(f.read_text())
            eps.setdefault(g["qid"], []).append(
                (g["lenient"] if g["cores_ok"] else 0, g["lenient"], g["gen_tokens"]))
        data[budget] = eps
    return _bootstrap_data(data, n_boot, seed)


def bootstrap(task: str, n_boot: int, seed: int = 0,
              grades_dir: str = "grades", runs_dir: str = "runs", *,
              rollouts: int | None = None, judge_model: str | None = None,
              whole_files: bool = False, effort: str = "",
              legacy_partial: bool = False) -> dict:
    if legacy_partial:
        return _legacy_bootstrap(task, n_boot, seed, grades_dir)
    if judge_model is None:
        raise ReportIntegrityError("strict bootstrap requires an explicit judge_model")
    population = load_complete_population(
        task, grades_dir, runs_dir, rollouts=rollouts,
        judge_model=judge_model, whole_files=whole_files, effort=effort)
    return bootstrap_population(population, n_boot, seed)


def write_report_artifact(*, task: str, run_id: str, judge_dir: str,
                          aggregate_result: dict, bootstrap_result: dict | None,
                          bootstrap_replicates: int, bootstrap_seed: int,
                          audit: dict, paper_comparison: dict | None = None,
                          output_root: str | Path = "reports") -> Path:
    """Recompute and write one deterministic content-addressed result.

    The writer is a research boundary, not a serialization convenience.  It
    reloads the exact run/grade population named by ``audit`` and rejects any
    caller-supplied aggregate, interval, audit, or paper reference that does
    not follow from those immutable inputs.
    """

    try:
        run_id = validate_id(run_id)
        judge_dir = validate_id(judge_dir, "grade ID")
    except ValueError as error:
        raise ReportIntegrityError(str(error)) from error
    if task not in CORPORA:
        raise ReportIntegrityError(f"unknown task {task!r}")
    if type(bootstrap_replicates) is not int or bootstrap_replicates < 0:
        raise ReportIntegrityError("bootstrap replicates must be a nonnegative integer")
    if type(bootstrap_seed) is not int:
        raise ReportIntegrityError("bootstrap seed must be an integer")
    if not isinstance(audit, dict):
        raise ReportIntegrityError("report audit is not an object")

    run_manifest = audit.get("run_manifest")
    population_records = audit.get("population")
    grading_manifest = audit.get("grading_manifest")
    if (
        not isinstance(run_manifest, dict)
        or not isinstance(run_manifest.get("path"), str)
        or not isinstance(population_records, list)
        or not population_records
        or not isinstance(grading_manifest, dict)
        or not isinstance(grading_manifest.get("config"), dict)
    ):
        raise ReportIntegrityError("report audit cannot identify its immutable inputs")

    def recorded_path(raw: object, *, label: str) -> Path:
        if not isinstance(raw, str) or not raw or "\\" in raw:
            raise ReportIntegrityError(f"{label} path is invalid")
        logical = Path(raw)
        if not logical.is_absolute() and any(
            part in ("", ".", "..") for part in logical.parts
        ):
            raise ReportIntegrityError(f"{label} path is unsafe")
        path = logical.absolute() if logical.is_absolute() else (ROOT / logical).absolute()
        try:
            path.relative_to(ROOT.absolute())
            read_artifact_bytes(path)
        except (OSError, ValueError) as error:
            raise ReportIntegrityError(f"{label} path is missing or unsafe") from error
        return path

    manifest_path = recorded_path(run_manifest["path"], label="run manifest")
    if (
        manifest_path.name != "manifest.json"
        or manifest_path.parent.name != task
        or manifest_path.parent.parent.name != run_id
    ):
        raise ReportIntegrityError("run-manifest path disagrees with report identity")
    run_root = manifest_path.parent.parent

    grade_roots = set()
    for record in population_records:
        if not isinstance(record, dict):
            raise ReportIntegrityError("report population record is invalid")
        grade_path = recorded_path(record.get("grade_path"), label="grade")
        if len(grade_path.parents) < 4:
            raise ReportIntegrityError("grade path is too shallow")
        grade_roots.add(grade_path.parents[3])
    if len(grade_roots) != 1:
        raise ReportIntegrityError("report population mixes grading roots")
    grade_root = grade_roots.pop()
    if grade_root.name != judge_dir or grade_root.parent.name != run_id:
        raise ReportIntegrityError("grading root disagrees with report identity")

    config = grading_manifest["config"]
    judge_model = config.get("judge_requested_model")
    judge_base_url = config.get("judge_base_url")
    whole_files = config.get("whole_files")
    effort = config.get("judge_effort")
    spec = run_manifest.get("spec")
    if (
        not isinstance(judge_model, str)
        or not isinstance(judge_base_url, str)
        or type(whole_files) is not bool
        or not isinstance(effort, str)
        or not isinstance(spec, dict)
        or type(spec.get("rollouts")) is not int
    ):
        raise ReportIntegrityError("grading or run configuration is invalid")
    try:
        _, configured_base_url = grader_identity_for_model(judge_model)
    except GradeIntegrityError as error:
        raise ReportIntegrityError("grading endpoint identity is invalid") from error
    if judge_base_url != configured_base_url:
        raise ReportIntegrityError("grading endpoint differs from the configured grader")
    fresh_population, fresh_audit = _load_complete_evaluation(
        task,
        grade_root,
        run_root,
        rollouts=spec["rollouts"],
        judge_model=judge_model,
        whole_files=whole_files,
        effort=effort,
    )
    if canonical_json_bytes(fresh_audit) != canonical_json_bytes(audit):
        raise ReportIntegrityError("caller-supplied report audit is not current")

    expected_aggregate = aggregate_population(fresh_population)
    if canonical_json_bytes(aggregate_result) != canonical_json_bytes(expected_aggregate):
        raise ReportIntegrityError("caller-supplied report aggregate does not recompute")
    expected_bootstrap = (
        bootstrap_population(fresh_population, bootstrap_replicates, bootstrap_seed)
        if bootstrap_replicates
        else None
    )
    if canonical_json_bytes(bootstrap_result) != canonical_json_bytes(expected_bootstrap):
        raise ReportIntegrityError("caller-supplied report bootstrap does not recompute")

    if paper_comparison is not None:
        if not isinstance(paper_comparison, dict) or set(paper_comparison) != {
            "variant",
            "table",
            "status",
            "known_regime_checks_passed",
            "limitations",
        }:
            raise ReportIntegrityError("paper comparison record is invalid")
        variant = paper_comparison.get("variant")
        paper_key = "" if variant == "base" else "cheatsheet" if variant == "cheatsheet" else None
        expected_table = PAPER.get((task, paper_key)) if paper_key is not None else None
        if (
            expected_table is None
            or paper_comparison.get("table") != expected_table
            or paper_comparison.get("status")
            != "contextual-reference-not-byte-exact-replication"
            or paper_comparison.get("known_regime_checks_passed") is not True
            or paper_comparison.get("limitations") != PAPER_REFERENCE_LIMITATIONS
            or paper_comparability_errors(
                fresh_audit,
                variant=variant,
                judge_model=judge_model,
                whole_files=whole_files,
            )
        ):
            raise ReportIntegrityError("paper comparison does not follow from the report inputs")

    artifact = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "claim_ready": True,
        "task": task,
        "run_id": run_id,
        "budget_order": BUDGET_ORDER,
        "run_manifest": audit["run_manifest"],
        "generation_runtime": audit["generation_runtime"],
        "note_provenance": audit["note_provenance"],
        "failed_attempts": audit["failed_attempts"],
        "failed_judge_audits": audit["failed_judge_audits"],
        "grading_manifest": audit["grading_manifest"],
        "population": audit["population"],
        "population_sha256": audit["population_sha256"],
        "aggregate": aggregate_result,
        "bootstrap": {
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "results": bootstrap_result,
        },
        "paper_comparison": paper_comparison,
        "report_source": {
            "studybench/report.py": file_sha256(Path(__file__).resolve()),
        },
    }
    artifact_sha256 = sha256_json(artifact)
    root = _rooted(output_root)
    path = root / run_id / judge_dir / task / f"report-{artifact_sha256}.json"
    write_immutable_json(path, artifact)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", default="dspy,openclaw")
    p.add_argument("--run-id",
                   help="immutable run ID (required unless --legacy-partial is used)")
    p.add_argument("--grader", default="openai", choices=["openai", "fugu"])
    p.add_argument(
        "--grade-id",
        help="immutable grading namespace (default: judge/config name used by grade.py)",
    )
    p.add_argument("--rollouts", type=int,
                   help="optional assertion; strict mode normally reads this from the manifest")
    evidence = p.add_mutually_exclusive_group()
    evidence.add_argument(
        "--whole-files", dest="whole_files", action="store_true",
        help="paper-faithful A.5 judge context: full evidence files",
    )
    evidence.add_argument(
        "--excerpt-evidence", dest="whole_files", action="store_false",
        help="local variant: only dataset evidence excerpts",
    )
    p.set_defaults(whole_files=None)
    p.add_argument("--judge-effort", default="",
                   choices=["", "low", "medium", "high", "xhigh"])
    p.add_argument(
        "--legacy-partial", action="store_true",
        help="DIAGNOSTIC ONLY: bypass manifests/provenance/completeness for historical artifacts",
    )
    p.add_argument("--ci", type=int, default=0, metavar="N",
                   help="add 95%% bootstrap CIs from N replicates (e.g. 10000)")
    p.add_argument("--ci-seed", type=int, default=0,
                   help="deterministic bootstrap seed (recorded in the report artifact)")
    p.add_argument("--report-dir", default="reports",
                   help="root for immutable strict report artifacts")
    p.add_argument(
        "--paper-variant", choices=["base", "cheatsheet"],
        help="request a contextual Table 1 reference; known regime checks must pass",
    )
    p.add_argument(
        "--legacy-grades-dir",
        help="diagnostic-only explicit historical grade root (passed to legacy loader)",
    )
    p.add_argument(
        "--legacy-runs-dir",
        help="diagnostic-only explicit historical run root (passed to legacy loader)",
    )
    args = p.parse_args()

    if args.ci < 0:
        p.error("--ci must be nonnegative")
    if not args.legacy_partial and not args.run_id:
        p.error("--run-id is required for strict reporting")
    if not args.legacy_partial and args.whole_files is None:
        p.error("strict reporting requires --whole-files or --excerpt-evidence")
    if args.legacy_partial and args.paper_variant:
        p.error("paper comparison is unavailable for legacy/partial diagnostics")
    if not args.legacy_partial and (args.legacy_grades_dir or args.legacy_runs_dir):
        p.error("--legacy-*-dir options require --legacy-partial")
    run_id = args.run_id or "base"
    try:
        run_id = validate_id(run_id)
        if args.grade_id is not None:
            args.grade_id = validate_id(args.grade_id, "grade ID")
    except ValueError as exc:
        p.error(str(exc))
    if args.legacy_partial:
        print(DIAGNOSTIC_BANNER, file=sys.stderr)
    judge_model = GRADERS[args.grader][0]
    if args.legacy_partial and args.whole_files is None:
        args.whole_files = False
        judge_dir = judge_model
    else:
        judge_dir = (judge_model + ("-wholefiles" if args.whole_files else "-excerpts")
                     + (f"-effort-{args.judge_effort}" if args.judge_effort else ""))
    grade_id = args.grade_id or judge_dir
    grades = args.legacy_grades_dir or f"grades/{run_id}/{grade_id}"
    runs = args.legacy_runs_dir or f"runs/{run_id}"
    for task in args.tasks.split(","):
        try:
            if args.legacy_partial:
                agg = _legacy_aggregate(task, grades, runs)
                audit = population = None
            else:
                population, audit = load_complete_evaluation(
                    task, grades, runs, rollouts=args.rollouts,
                    judge_model=judge_model, whole_files=args.whole_files,
                    effort=args.judge_effort)
                agg = aggregate_population(population)
        except ReportIntegrityError as exc:
            raise SystemExit(f"INTEGRITY ERROR: {exc}") from exc
        paper = None
        if args.paper_variant:
            comparison_errors = paper_comparability_errors(
                audit,
                variant=args.paper_variant,
                judge_model=judge_model,
                whole_files=args.whole_files,
            )
            if comparison_errors:
                details = "\n".join(f"  - {error}" for error in comparison_errors)
                raise SystemExit(
                    "INTEGRITY ERROR: requested paper comparison is not valid:\n"
                    f"{details}"
                )
            paper_key = "" if args.paper_variant == "base" else "cheatsheet"
            paper = PAPER.get((task, paper_key))
            if paper is None:
                raise SystemExit(
                    f"INTEGRITY ERROR: no paper Table 1 values exist for {task!r}")
            print(
                "CONTEXT ONLY: Table 1 is not a byte-exact replication target; "
                + "; ".join(PAPER_REFERENCE_LIMITATIONS),
                file=sys.stderr,
            )
        label = f"run={run_id}, grade-id={grade_id}, judge={args.grader}"
        if args.legacy_partial:
            label += ", DIAGNOSTIC-LEGACY-PARTIAL-NOT-A-RESULT"
        print(f"\n== {CORPORA[task].display} ({label}) ==")
        if not args.legacy_partial:
            print(
                "retained non-final attempts (excluded from ITT population): "
                f"{audit['failed_attempts']['count']}"
            )
            print(
                "retained failed judge audits (no grade; excluded from ITT population): "
                f"{audit['failed_judge_audits']['count']}"
            )
        header = (
            f"{'budget':8} {'n':>4} {'lenient':>8} {'len-cc':>7} "
            f"{'strict':>7} {'tok(k)':>7} {'compile':>8} {'no-ans':>6} "
            f"{'regrade':>8} {'bad':>4}"
        )
        if paper is not None:
            header += "   paper-ref-lenient  paper-ref-tok(k)"
        print(header)
        for budget, b in agg["budgets"].items():
            line = (
                f"{budget:8} {b['n']:>4} {b['lenient']:>8.1f} "
                f"{b['len_cc']:>7.1f} {b['strict']:>7.1f} "
                f"{b['tokens'] / 1000:>7.1f} {b['compile_rate']:>8.1%} "
                f"{b['no_answer']:>6} {b['needs_regrade']:>8} "
                f"{b['bad_episodes']:>4}"
            )
            if paper is not None:
                paper_accuracy, paper_tokens = paper[budget]
                line += f"   {paper_accuracy:>13.1f} {paper_tokens:>12.1f}"
            print(line)
        if agg.get("expertise_lenient") is not None:
            line = f"expertise (lenient WAUC): {agg['expertise_lenient']:.2f}"
            if paper is not None:
                line += f" (paper reference: {paper['expertise']:.2f})"
            line += f"; strict WAUC: {agg['expertise_strict']:.2f}"
            print(line)
        bootstrap_result = None
        if args.ci:
            try:
                if args.legacy_partial:
                    b = _legacy_bootstrap(task, args.ci, args.ci_seed, grades)
                else:
                    b = bootstrap_population(population, args.ci, args.ci_seed)
            except (ReportIntegrityError, KeyError) as exc:
                raise SystemExit(f"INTEGRITY ERROR: cannot bootstrap: {exc}") from exc
            bootstrap_result = b
            print(f"95% CIs ({args.ci} bootstrap replicates over questions×rollouts):")
            for budget in BUDGET_ORDER:
                m, lo, hi = b[budget]
                print(f"  {budget:8} lenient {m:5.1f} [{lo:5.1f}, {hi:5.1f}]")
            m, lo, hi = b["wauc"]
            line = f"  WAUC lenient {m:5.2f} [{lo:5.2f}, {hi:5.2f}]"
            if paper is not None:
                line += f" (paper reference: {paper['expertise']:.2f})"
            print(line)
            m, lo, hi = b["wauc_cc"]
            print(f"  WAUC len-cc  {m:5.2f} [{lo:5.2f}, {hi:5.2f}]")
        if not args.legacy_partial:
            report_path = write_report_artifact(
                task=task,
                run_id=run_id,
                judge_dir=grade_id,
                aggregate_result=agg,
                bootstrap_result=bootstrap_result,
                bootstrap_replicates=args.ci,
                bootstrap_seed=args.ci_seed,
                audit=audit,
                paper_comparison=(
                    {
                        "variant": args.paper_variant,
                        "table": paper,
                        "status": "contextual-reference-not-byte-exact-replication",
                        "known_regime_checks_passed": True,
                        "limitations": PAPER_REFERENCE_LIMITATIONS,
                    }
                    if paper is not None else None
                ),
                output_root=args.report_dir,
            )
            print(f"immutable report: {_display_path(report_path)}")


if __name__ == "__main__":
    main()
