"""Deterministic one-round study over a bounded full-source DSPy call relation.

This runner deliberately has a smaller scientific claim than a Python call
graph.  :mod:`studybench.static_graph` freezes a conservative syntactic
relation, supplies exact gold edge sets, and scores strict JSON predictions.
The model is used only for the open-book ATTEMPT phase.  Verification,
correction construction, note rendering, and the paired internal development
exam are otherwise deterministic.

The full procedure studies 16 corpus-selected targets spanning every eligible
top-level package stratum from an empty note, then evaluates four location-
disjoint held-out targets twice with identical seeds and protocols; only the
note differs. Smoke mode runs only the first frozen training target and cannot produce an
automated-claim-ready note manifest.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import logging
from pathlib import Path, PurePosixPath
import platform
import re
from typing import Any, Iterable

from .dataset import CORPORA, ROOT
from .integrity import (
    canonical_json_bytes,
    exclusive_process_lock,
    load_json_artifact,
    read_artifact_bytes,
    sha256_bytes,
    sha256_json,
    sha256_text,
    stable_seed,
    strict_json_loads,
    write_immutable_json,
    write_immutable_text,
)
from .provenance import (
    corpus_record,
    environment_contract_is_valid,
    environment_contract_record,
    environment_is_claim_ready,
    environment_record,
    environments_compatible,
    source_record,
    validate_environment_snapshot,
    validate_id,
    validate_local_server_urls,
    write_environment_snapshot,
)
from .react import MODEL_ID, MODEL_REVISION, READ_MAX_LINES, SAMPLING, make_tools
from .selfquiz import (
    _attempt,
    _attempt_protocol as attempt_protocol_record,
    _record_id,
    _server_url,
    _validate_attempt_record,
    artifact_usage_consistent,
    repo_map,
    usage_by_phase,
    usage_ledger_audit,
    usage_totals,
)
from .static_graph import (
    CONTRACT_VERSION,
    FROZEN_DEV_TARGETS,
    FROZEN_TRAIN_TARGETS,
    SOURCE_ROOT,
    build_question_bank,
    contract_sha256,
    resolver_contract_record,
    verify_prediction,
)
from .study_protocol import (
    STATIC_GRAPH_METHOD,
    STATIC_GRAPH_NOTE_MANIFEST_TYPE,
    STATIC_GRAPH_TASK_MANIFEST_TYPE,
    StudyProtocolError,
    derive_protocol_summary,
    validate_study_note_archive,
)
from .tools import RepoTools


SCHEMA_VERSION = 1
TASK = "dspy"
ROUND = 1
# This is a fixed analyzer scope, not a configurable study subsetting knob.
# The cross-method protocol summary consequently records ``focus: null``.
ATTEMPT_ACCESS = "react-corpus"
SMOKE_TARGET = FROZEN_TRAIN_TARGETS[0]
TRAIN_QUESTION_COUNT = len(FROZEN_TRAIN_TARGETS)
DEV_QUESTION_COUNT = len(FROZEN_DEV_TARGETS)
VERDICTS = frozenset({"exact", "partial", "wrong"})
SHA256 = re.compile(r"[0-9a-f]{64}")

log = logging.getLogger("graph-study")


# ---------------------------------------------------------------------------
# Deterministic question, score, and correction records


def _question_name(question: dict[str, Any]) -> str:
    target = question.get("target")
    if not isinstance(target, str) or not target.startswith("dspy."):
        raise ValueError("graph question has an invalid DSPy target")
    return target.rsplit(".", 1)[-1]


def _static_json_sha256(value: object) -> str:
    """Match static_graph's newline-free canonical JSON fingerprint."""

    return sha256_text(json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ))


def _dspy_python_path(value: object) -> bool:
    if not isinstance(value, str):
        return False
    logical = PurePosixPath(value)
    return (
        value.startswith(SOURCE_ROOT + "/")
        and value.endswith(".py")
        and value == value.strip()
        and "\\" not in value
        and "\x00" not in value
        and not logical.is_absolute()
        and logical.as_posix() == value
        and all(part not in ("", ".", "..") for part in logical.parts)
    )


def _valid_gold_edge(edge: object, target: str) -> bool:
    if not isinstance(edge, dict) or set(edge) != {
        "caller", "callee", "path", "line", "direction",
    }:
        return False
    caller, callee = edge.get("caller"), edge.get("callee")
    if (
        not isinstance(caller, str)
        or not caller.startswith("dspy.")
        or not isinstance(callee, str)
        or not callee.startswith("dspy.")
        or not _dspy_python_path(edge.get("path"))
        or type(edge.get("line")) is not int
        or edge["line"] < 1
    ):
        return False
    direction = edge.get("direction")
    return (
        direction == "self" and caller == target and callee == target
        or direction == "incoming" and caller != target and callee == target
        or direction == "outgoing" and caller == target and callee != target
    )


def _validate_bank(
    bank: object,
    resolver: object,
    *,
    smoke: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate the frozen bank and return ``(full, selected)``.

    The exact target list and split live in the resolver contract.  These
    additional integration checks prevent a future analyzer change from
    silently turning the held-out item into training data.
    """

    if type(smoke) is not bool or not isinstance(bank, list) or not bank:
        raise ValueError("graph question bank is invalid")
    if not isinstance(resolver, dict):
        raise ValueError("graph resolver contract is invalid")
    resolver_fields = {
        "version", "source_root", "contract_sha256", "source_sha256",
        "source_files",
        "question_bank_sha256", "python_implementation", "python_version",
        "analyzer_module", "analyzer_source_sha256", "corpus_name",
        "corpus_commit", "target_symbols", "target_selection",
        "target_selection_sha256", "candidate_inventory",
        "candidate_inventory_sha256", "candidate_count",
        "eligible_candidate_count", "train_question_count",
        "dev_question_count", "excluded_candidates",
        "excluded_candidates_sha256", "excluded_candidate_count",
    }
    excluded = resolver.get("excluded_candidates")
    source_files = resolver.get("source_files")
    candidate_inventory = resolver.get("candidate_inventory")
    target_selection = resolver.get("target_selection")
    exclusion_fields = {
        "caller", "path", "line", "column", "callee_syntax", "reason",
    }
    exclusion_reasons = {
        "unresolved-bare-callee",
        "unresolved-attribute-callee",
        "dynamic-callee-expression",
    }
    hash_fields = (
        "contract_sha256", "source_sha256", "question_bank_sha256",
        "analyzer_source_sha256", "excluded_candidates_sha256",
        "candidate_inventory_sha256", "target_selection_sha256",
    )
    if (
        set(resolver) != resolver_fields
        or resolver.get("version") != CONTRACT_VERSION
        or resolver.get("source_root") != SOURCE_ROOT
        or resolver.get("corpus_name") != TASK
        or resolver.get("corpus_commit") != CORPORA[TASK].commit
        or resolver.get("contract_sha256") != contract_sha256()
        or resolver.get("python_implementation") != platform.python_implementation()
        or resolver.get("python_version") != platform.python_version()
        or resolver.get("analyzer_module") != "studybench.static_graph"
        or not all(
            isinstance(resolver.get(field), str)
            and SHA256.fullmatch(resolver[field])
            for field in hash_fields
        )
        or not isinstance(source_files, list)
        or not source_files
        or any(
            not isinstance(source_file, dict)
            or set(source_file) != {"path", "sha256", "line_count"}
            or not _dspy_python_path(source_file.get("path"))
            or not isinstance(source_file.get("sha256"), str)
            or not SHA256.fullmatch(source_file["sha256"])
            or type(source_file.get("line_count")) is not int
            or source_file["line_count"] < 0
            for source_file in source_files
        )
        or [source_file["path"] for source_file in source_files]
        != sorted({source_file["path"] for source_file in source_files})
        or resolver.get("source_sha256") != _static_json_sha256([
            {"path": source_file["path"], "sha256": source_file["sha256"]}
            for source_file in source_files
        ])
        or not isinstance(candidate_inventory, list)
        or not candidate_inventory
        or type(resolver.get("candidate_count")) is not int
        or resolver["candidate_count"] != len(candidate_inventory)
        or type(resolver.get("eligible_candidate_count")) is not int
        or resolver["eligible_candidate_count"] != sum(
            item.get("eligible") is True
            for item in candidate_inventory
            if isinstance(item, dict)
        )
        or resolver["candidate_inventory_sha256"]
        != _static_json_sha256(candidate_inventory)
        or any(
            not isinstance(item, dict)
            or set(item) != {
                "target", "stratum", "definition_path", "definition_line",
                "neighborhood_edge_count", "eligible", "exclusion",
                "stratum_rank", "global_rank",
            }
            or not isinstance(item.get("target"), str)
            or not item["target"].startswith("dspy.")
            or not isinstance(item.get("stratum"), str)
            or not item["stratum"]
            or not _dspy_python_path(item.get("definition_path"))
            or type(item.get("definition_line")) is not int
            or item["definition_line"] < 1
            or type(item.get("neighborhood_edge_count")) is not int
            or item["neighborhood_edge_count"] < 0
            or type(item.get("eligible")) is not bool
            or (
                item["eligible"]
                and (
                    item.get("exclusion") is not None
                    or type(item.get("stratum_rank")) is not int
                    or item["stratum_rank"] < 1
                    or type(item.get("global_rank")) is not int
                    or item["global_rank"] < 1
                    or not 1 <= item["neighborhood_edge_count"] <= 10
                )
            )
            or (
                not item["eligible"]
                and (
                    item.get("exclusion") not in {
                        "below-minimum-neighborhood",
                        "above-maximum-neighborhood",
                    }
                    or item.get("stratum_rank") is not None
                    or item.get("global_rank") is not None
                )
            )
            for item in candidate_inventory
        )
        or not isinstance(target_selection, list)
        or resolver["target_selection_sha256"]
        != _static_json_sha256(target_selection)
        or any(
            not isinstance(item, dict)
            or set(item) != {
                "target", "stratum", "split", "selection_stage",
                "neighborhood_edge_count", "stratum_rank", "global_rank",
            }
            or item.get("split") not in {"train", "dev"}
            or not isinstance(item.get("target"), str)
            or not item["target"].startswith("dspy.")
            or not isinstance(item.get("stratum"), str)
            or not item["stratum"]
            or item.get("selection_stage") not in {
                "base-stratum-train", "global-extra-train",
                "global-held-out-dev",
            }
            or type(item.get("neighborhood_edge_count")) is not int
            or not 1 <= item["neighborhood_edge_count"] <= 10
            or type(item.get("stratum_rank")) is not int
            or item["stratum_rank"] < 1
            or type(item.get("global_rank")) is not int
            or item["global_rank"] < 1
            for item in target_selection
        )
        or resolver.get("target_symbols") != [
            item.get("target") for item in target_selection
        ]
        or not isinstance(excluded, list)
        or not excluded
        or type(resolver.get("excluded_candidate_count")) is not int
        or resolver["excluded_candidate_count"] != len(excluded)
        or resolver["excluded_candidates_sha256"] != _static_json_sha256(excluded)
        or any(
            not isinstance(candidate, dict)
            or set(candidate) != exclusion_fields
            or not all(
                isinstance(candidate.get(field), str) and candidate[field]
                for field in ("caller", "path", "callee_syntax", "reason")
            )
            or not _dspy_python_path(candidate.get("path"))
            or type(candidate.get("line")) is not int
            or candidate["line"] < 1
            or type(candidate.get("column")) is not int
            or candidate["column"] < 0
            or candidate.get("reason") not in exclusion_reasons
            for candidate in excluded
        )
    ):
        raise ValueError("graph resolver provenance or exclusion inventory drifted")
    expected_fields = {
        "id", "target", "question", "anchors", "target_definition",
        "gold_edges", "split", "stratum", "selection_stage",
        "neighborhood_edge_count", "stratum_rank", "global_rank",
    }
    targets: list[str] = []
    ids: list[str] = []
    for question in bank:
        if not isinstance(question, dict) or set(question) != expected_fields:
            raise ValueError("graph question schema drifted")
        _question_name(question)
        target = question["target"]
        target_definition = question["target_definition"]
        gold_edges = question["gold_edges"]
        gold_identity = [
            (
                edge.get("caller"), edge.get("callee"), edge.get("path"),
                edge.get("line"), edge.get("direction"),
            )
            for edge in gold_edges
            if isinstance(edge, dict)
        ] if isinstance(gold_edges, list) else []
        if (
            not isinstance(question["id"], str)
            or question["id"] != f"static-call-neighborhood::{target}"
            or not isinstance(question["question"], str)
            or not question["question"].strip()
            or not isinstance(question["anchors"], list)
            or len(question["anchors"]) != 1
            or not all(
                _dspy_python_path(anchor)
                for anchor in question["anchors"]
            )
            or not isinstance(target_definition, dict)
            or set(target_definition) != {"path", "line"}
            or target_definition["path"] != question["anchors"][0]
            or type(target_definition["line"]) is not int
            or target_definition["line"] < 1
            or not isinstance(gold_edges, list)
            or not all(_valid_gold_edge(edge, target) for edge in gold_edges)
            or len(gold_identity) != len(set(gold_identity))
            or gold_identity != sorted(
                gold_identity,
                key=lambda identity: (
                    identity[2], identity[3], identity[0], identity[1], identity[4]
                ),
            )
            or question["split"] not in {"train", "dev"}
            or not isinstance(question["stratum"], str)
            or not question["stratum"]
            or question["selection_stage"] not in {
                "base-stratum-train", "global-extra-train",
                "global-held-out-dev",
            }
            or (
                question["split"] == "dev"
                and question["selection_stage"] != "global-held-out-dev"
            )
            or (
                question["split"] == "train"
                and question["selection_stage"] == "global-held-out-dev"
            )
            or type(question["neighborhood_edge_count"]) is not int
            or question["neighborhood_edge_count"] != len(gold_edges)
            or not 1 <= question["neighborhood_edge_count"] <= 10
            or type(question["stratum_rank"]) is not int
            or question["stratum_rank"] < 1
            or type(question["global_rank"]) is not int
            or question["global_rank"] < 1
        ):
            raise ValueError(f"invalid graph question: {question.get('id')!r}")
        targets.append(target)
        ids.append(question["id"])
    if len(set(ids)) != len(ids) or len(set(targets)) != len(targets):
        raise ValueError("graph question IDs and targets must be unique")
    if resolver["question_bank_sha256"] != _static_json_sha256(bank):
        raise ValueError("graph question bank hash drifted from the resolver")
    expected_targets = resolver.get("target_symbols")
    if targets != expected_targets:
        raise ValueError("graph question order or targets drifted from the resolver")
    observed_selection = [
        {
            key: question[key]
            for key in (
                "target", "stratum", "split", "selection_stage",
                "neighborhood_edge_count", "stratum_rank", "global_rank",
            )
        }
        for question in bank
    ]
    if observed_selection != target_selection:
        raise ValueError("question metadata drifted from target selection")
    train = [question for question in bank if question["split"] == "train"]
    dev = [question for question in bank if question["split"] == "dev"]
    if (
        len(bank) != TRAIN_QUESTION_COUNT + DEV_QUESTION_COUNT
        or len(train) != TRAIN_QUESTION_COUNT
        or len(dev) != DEV_QUESTION_COUNT
        or tuple(question["target"] for question in train) != FROZEN_TRAIN_TARGETS
        or tuple(question["target"] for question in dev) != FROZEN_DEV_TARGETS
        or resolver.get("train_question_count") != TRAIN_QUESTION_COUNT
        or resolver.get("dev_question_count") != DEV_QUESTION_COUNT
    ):
        raise ValueError("the frozen 16-train/four-dev split drifted")
    base_strata = [
        question["stratum"]
        for question in train
        if question["selection_stage"] == "base-stratum-train"
    ]
    if len(base_strata) != 12 or len(set(base_strata)) != 12:
        raise ValueError("base training targets do not cover all 12 strata")
    train_locations = {
        (edge["path"], edge["line"])
        for question in train
        for edge in question["gold_edges"]
    }
    dev_locations = {
        (edge["path"], edge["line"])
        for question in dev
        for edge in question["gold_edges"]
    }
    if train_locations & dev_locations:
        raise ValueError("development evidence locations overlap training evidence")
    selected = (
        [question for question in bank if question["target"] == SMOKE_TARGET]
        if smoke
        else list(bank)
    )
    if smoke and (
        len(selected) != 1
        or selected[0]["split"] != "train"
        or selected[0]["target"] != SMOKE_TARGET
    ):
        raise ValueError("smoke must select exactly the first frozen train target")
    return list(bank), selected


def score_verdict(score: object) -> str:
    """Map an exact verifier result to the frozen three-level verdict."""

    if not isinstance(score, dict) or type(score.get("exact")) is not bool:
        raise ValueError("verifier score is invalid")
    f1 = score.get("f1")
    if not isinstance(f1, (int, float)) or isinstance(f1, bool) or not 0 <= f1 <= 1:
        raise ValueError("verifier F1 is invalid")
    if score["exact"]:
        if f1 != 1:
            raise ValueError("exact verifier result must have F1=1")
        return "exact"
    return "partial" if f1 > 0 else "wrong"


def _source_line(rt: Any, path: str, line: int) -> str:
    text = getattr(rt, "text", {}).get(path)
    if not isinstance(text, str) or type(line) is not int or line < 1:
        raise ValueError(f"invalid source evidence location: {path}:{line}")
    lines = text.splitlines()
    if line > len(lines):
        raise ValueError(f"source evidence is out of range: {path}:{line}")
    return lines[line - 1]


def _edge_evidence(rt: Any, edge: dict[str, Any]) -> dict[str, Any]:
    return {
        "edge": edge,
        "file": edge["path"],
        "line": edge["line"],
        "quote": _source_line(rt, edge["path"], edge["line"]),
    }


def build_correction_entry(
    question: dict[str, Any],
    score: dict[str, Any],
    rt: Any,
) -> dict[str, Any] | None:
    """Build one model-free correction from an exact set difference.

    Every gold edge is paired with the exact source line containing its call.
    A target-definition line is retained for the degenerate empty-gold case.
    No development question should ever be passed to this function.
    """

    if question.get("split") != "train":
        raise ValueError("development gold cannot enter correction construction")
    verdict = score_verdict(score)
    if verdict == "exact":
        return None
    missing = score.get("missing_edges")
    spurious = score.get("spurious_edges")
    if not isinstance(missing, list) or not isinstance(spurious, list):
        raise ValueError("verifier set difference is invalid")
    evidence = [_edge_evidence(rt, edge) for edge in question["gold_edges"]]
    if not evidence:
        definition = question["target_definition"]
        evidence = [{
            "edge": None,
            "file": definition["path"],
            "line": definition["line"],
            "quote": _source_line(rt, definition["path"], definition["line"]),
        }]
    belief = (
        f"Your submitted bounded neighborhood for {question['target']} "
        f"omitted {len(missing)} gold edge(s) and added "
        f"{len(spurious)} spurious edge(s)."
    )
    if score.get("schema_valid") is not True:
        belief = (
            f"Your submitted answer for {question['target']} violated the "
            f"strict JSON edge schema: {score.get('schema_error')}."
        )
    correction = json.dumps(
        {"edges": question["gold_edges"]},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    payload = {
        "origin_question_id": question["id"],
        "target": question["target"],
        "chapter": SOURCE_ROOT,
        "verdict": verdict,
        "belief": belief,
        "correction": correction,
        "gold_edges": question["gold_edges"],
        "missing_edges": missing,
        "spurious_edges": spurious,
        "schema_error": score.get("schema_error"),
        "evidence": evidence,
        "evidence_class": "exact-source-line",
    }
    return {
        "entry_id": _record_id("graph-entry", question["id"], payload),
        **payload,
    }


def _render_edge(edge: dict[str, Any]) -> str:
    return (
        f"`{edge['direction']}: {edge['caller']} -> {edge['callee']}` "
        f"(`{edge['path']}:{edge['line']}`)"
    )


def render_graph_note(rt: Any, entries: list[dict[str, Any]]) -> str:
    """Render only training-derived correction entries into the study note."""

    if any(
        entry.get("chapter") != SOURCE_ROOT
        or entry.get("target") in FROZEN_DEV_TARGETS
        for entry in entries
    ):
        raise ValueError("held-out or out-of-scope entry cannot enter the graph note")
    parts = [
        "# DSPy — deterministic static-neighborhood corrections",
        "",
        (
            "This note records corrections under the frozen conservative static "
            "resolver. It is not a Python runtime call graph."
        ),
        "",
        repo_map(rt, sorted({
            "/".join(str(entry["target"]).split(".")[:2])
            for entry in entries
        })),
        "",
    ]
    for entry in entries:
        parts.extend([
            f"## `{entry['target']}`",
            "",
            entry["belief"],
            "",
            "Exact bounded neighborhood:",
            "",
        ])
        evidence_by_edge = {
            canonical_json_bytes(evidence["edge"]): evidence
            for evidence in entry["evidence"]
            if evidence["edge"] is not None
        }
        if not entry["gold_edges"]:
            evidence = entry["evidence"][0]
            parts.extend([
                "- No resolved edges.",
                f"  Target definition: `{evidence['file']}:{evidence['line']}`",
                f"  > `{evidence['quote']}`",
            ])
        for edge in entry["gold_edges"]:
            evidence = evidence_by_edge[canonical_json_bytes(edge)]
            parts.extend([
                f"- {_render_edge(edge)}",
                f"  > `{evidence['quote']}`",
            ])
        parts.append("")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Artifact construction and exact validation


def _question_artifact(question: dict[str, Any], resolver: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "static-graph-question",
        "resolver_contract_sha256": sha256_json(resolver),
        "question_id": question["id"],
        "target": question["target"],
        "split": question["split"],
        "stratum": question["stratum"],
        "selection_stage": question["selection_stage"],
        "neighborhood_edge_count": question["neighborhood_edge_count"],
        "stratum_rank": question["stratum_rank"],
        "global_rank": question["global_rank"],
        "question": question["question"],
        "question_sha256": sha256_text(question["question"]),
        "anchors": question["anchors"],
        "target_definition": question["target_definition"],
        "gold_edges": question["gold_edges"],
        "gold_sha256": sha256_json(question["gold_edges"]),
    }


def _question_filename(index: int, question: dict[str, Any]) -> str:
    target = question.get("target")
    if not isinstance(target, str) or not target.startswith("dspy."):
        raise ValueError("question target is invalid")
    return f"q{index:02d}-{sha256_text(target)[:16]}.json"


def _question_paths(
    rdir: Path, selected: list[dict[str, Any]], resolver: dict[str, Any]
) -> dict[str, tuple[Path, dict[str, Any]]]:
    return {
        question["id"]: (
            rdir / "questions" / _question_filename(index, question),
            _question_artifact(question, resolver),
        )
        for index, question in enumerate(selected)
    }


def _attempt_delta(score: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_error": score["schema_error"],
        "missing_edges": score["missing_edges"],
        "spurious_edges": score["spurious_edges"],
        "invalid_location_edges": score["invalid_location_edges"],
    }


def _training_record(
    question: dict[str, Any],
    question_artifact: dict[str, Any],
    rt: Any,
    tools: Iterable[Any],
    url: str,
    *,
    study_id: str,
    master_seed: int,
    launch_environment: dict[str, Any],
) -> dict[str, Any]:
    owner_id = _record_id("graph-train", study_id, question["id"])
    seed = stable_seed(master_seed, study_id, question["id"], "train-attempt")
    attempt, calls = _attempt(
        question["question"],
        "",
        tools,
        url,
        seed=seed,
        owner_id=owner_id,
        phase="graph-train-attempt",
        attempt_access=ATTEMPT_ACCESS,
    )
    score = verify_prediction(attempt["answer"], question["gold_edges"], source_view=rt)
    # A transport/model failure is not a substantive answer and must never be
    # promoted into a gold correction.  The immutable attempt and deterministic
    # score are retained for diagnosis; the run is stopped before note creation.
    entry = (
        build_correction_entry(question, score, rt)
        if attempt["status"] == "ok"
        else None
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "static-graph-train-attempt",
        "question_id": question["id"],
        "question_artifact_sha256": sha256_json(question_artifact),
        "target": question["target"],
        "split": "train",
        "owner_id": owner_id,
        "seed": seed,
        "input_note_sha256": sha256_text(""),
        "input_note_bytes": 0,
        "attempt_protocol": attempt_protocol_record(ATTEMPT_ACCESS),
        "attempt": attempt,
        "prediction_sha256": sha256_text(attempt["answer"]),
        "score": score,
        "score_sha256": sha256_json(score),
        "verdict": score_verdict(score),
        "delta": _attempt_delta(score),
        "entry": entry,
        "calls": calls,
        "usage": usage_totals(calls),
        "environment_snapshot": launch_environment,
    }


def _dev_record(
    question: dict[str, Any],
    question_artifact: dict[str, Any],
    note: str,
    rt: Any,
    tools: Iterable[Any],
    url: str,
    *,
    study_id: str,
    master_seed: int,
    launch_environment: dict[str, Any],
) -> dict[str, Any]:
    owner_id = _record_id("graph-dev", study_id, question["id"])
    paired_seed = stable_seed(master_seed, study_id, question["id"], "paired-attempt")
    attempts: dict[str, dict[str, Any]] = {}
    scores: dict[str, dict[str, Any]] = {}
    calls: list[dict[str, Any]] = []
    for arm, arm_note in (("with_note", note), ("bare", "")):
        attempt, arm_calls = _attempt(
            question["question"],
            arm_note,
            tools,
            url,
            seed=paired_seed,
            owner_id=owner_id,
            phase=f"graph-dev-attempt-{arm}",
            attempt_access=ATTEMPT_ACCESS,
        )
        attempts[arm] = attempt
        scores[arm] = verify_prediction(
            attempt["answer"], question["gold_edges"], source_view=rt
        )
        calls.extend(arm_calls)
    protocol = {
        **attempt_protocol_record(ATTEMPT_ACCESS),
        "paired_seed": paired_seed,
        "only_manipulated_field": "note",
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "static-graph-dev-exam",
        "question_id": question["id"],
        "question_artifact_sha256": sha256_json(question_artifact),
        "target": question["target"],
        "split": "dev",
        "owner_id": owner_id,
        "paired_seed": paired_seed,
        "attempt_protocol": protocol,
        "presented_inputs": {
            "with_note": {
                "question_sha256": sha256_text(question["question"]),
                "note_sha256": sha256_text(note),
                "note_bytes": len(note.encode("utf-8")),
            },
            "bare": {
                "question_sha256": sha256_text(question["question"]),
                "note_sha256": sha256_text(""),
                "note_bytes": 0,
            },
        },
        "attempts": attempts,
        "scores": scores,
        "score_sha256": {arm: sha256_json(score) for arm, score in scores.items()},
        "verdicts": {arm: score_verdict(score) for arm, score in scores.items()},
        "deltas": {arm: _attempt_delta(score) for arm, score in scores.items()},
        "calls": calls,
        "usage": usage_totals(calls),
        "environment_snapshot": launch_environment,
    }


def _validate_question_artifact(
    path: Path,
    question: dict[str, Any],
    resolver: dict[str, Any],
) -> dict[str, Any]:
    observed = load_json_artifact(path)
    expected = _question_artifact(question, resolver)
    if canonical_json_bytes(observed) != canonical_json_bytes(expected):
        raise SystemExit(f"question/gold artifact drifted: {path}")
    return observed


def _validate_environment_binding(
    sdir: Path,
    task_manifest: dict[str, Any],
    record: object,
    *,
    require_claim_ready: bool,
) -> None:
    try:
        validate_environment_snapshot(
            sdir,
            record,
            baseline=task_manifest["environment"],
            require_claim_ready=require_claim_ready,
        )
    except (KeyError, OSError, ValueError) as error:
        raise SystemExit(f"invalid launch-environment provenance: {error}") from error


def _validate_calls(
    calls: object,
    *,
    owner_id: str,
    phases: set[str],
    seed: int,
    require_nonempty: bool,
) -> list[dict[str, Any]]:
    expected_fields = {
        "call_id", "owner_id", "phase", "seed", "model", "model_revision",
        "response_model", "response_id", "system_fingerprint",
        "request_messages_sha256", "request_messages_available",
        "outputs_sha256", "outputs_available", "prompt_tokens",
        "completion_tokens", "total_tokens", "usage_reported", "provider_usage",
    }
    if not isinstance(calls, list) or (require_nonempty and not calls):
        raise SystemExit("attempt model-call ledger is missing")
    for call in calls:
        if (
            not isinstance(call, dict)
            or set(call) != expected_fields
            or call.get("owner_id") != owner_id
            or call.get("phase") not in phases
            or type(call.get("seed")) is not int
            or call["seed"] != seed
        ):
            raise SystemExit("attempt model-call lineage drifted")
    audit = usage_ledger_audit(calls, calls)
    if not audit["complete"]:
        raise SystemExit("attempt model-call usage is incomplete: " + "; ".join(audit["errors"]))
    return calls


def _validate_training_record(
    record: object,
    question: dict[str, Any],
    question_artifact: dict[str, Any],
    rt: Any,
    *,
    study_id: str,
    master_seed: int,
    sdir: Path,
    task_manifest: dict[str, Any],
    smoke: bool,
) -> dict[str, Any]:
    owner_id = _record_id("graph-train", study_id, question["id"])
    seed = stable_seed(master_seed, study_id, question["id"], "train-attempt")
    if not isinstance(record, dict):
        raise SystemExit("training artifact is not an object")
    expected_keys = {
        "schema_version", "record_type", "question_id",
        "question_artifact_sha256", "target", "split", "owner_id", "seed",
        "input_note_sha256", "input_note_bytes", "attempt_protocol", "attempt",
        "prediction_sha256", "score", "score_sha256", "verdict", "delta",
        "entry", "calls", "usage", "environment_snapshot",
    }
    if set(record) != expected_keys:
        raise SystemExit("training artifact schema drifted")
    expected_fixed = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "static-graph-train-attempt",
        "question_id": question["id"],
        "question_artifact_sha256": sha256_json(question_artifact),
        "target": question["target"],
        "split": "train",
        "owner_id": owner_id,
        "seed": seed,
        "input_note_sha256": sha256_text(""),
        "input_note_bytes": 0,
        "attempt_protocol": attempt_protocol_record(ATTEMPT_ACCESS),
    }
    if any(record.get(key) != value for key, value in expected_fixed.items()):
        raise SystemExit("training artifact lineage or empty-note contract drifted")
    attempt = _validate_attempt_record(
        record.get("attempt"), seed=seed, label="graph training attempt"
    )
    score = verify_prediction(attempt["answer"], question["gold_edges"], source_view=rt)
    expected_derived = {
        "prediction_sha256": sha256_text(attempt["answer"]),
        "score": score,
        "score_sha256": sha256_json(score),
        "verdict": score_verdict(score),
        "delta": _attempt_delta(score),
        "entry": (
            build_correction_entry(question, score, rt)
            if attempt["status"] == "ok"
            else None
        ),
    }
    if any(record.get(key) != value for key, value in expected_derived.items()):
        raise SystemExit("training score or model-free correction drifted")
    calls = _validate_calls(
        record.get("calls"),
        owner_id=owner_id,
        phases={"graph-train-attempt"},
        seed=seed,
        require_nonempty=attempt["status"] == "ok",
    )
    if record.get("usage") != usage_totals(calls):
        raise SystemExit("training artifact usage drifted")
    _validate_environment_binding(
        sdir,
        task_manifest,
        record.get("environment_snapshot"),
        require_claim_ready=not smoke,
    )
    return record


def _validate_dev_record(
    record: object,
    question: dict[str, Any],
    question_artifact: dict[str, Any],
    note: str,
    rt: Any,
    *,
    study_id: str,
    master_seed: int,
    sdir: Path,
    task_manifest: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise SystemExit("development exam is not an object")
    expected_keys = {
        "schema_version", "record_type", "question_id",
        "question_artifact_sha256", "target", "split", "owner_id",
        "paired_seed", "attempt_protocol", "presented_inputs", "attempts",
        "scores", "score_sha256", "verdicts", "deltas", "calls", "usage",
        "environment_snapshot",
    }
    if set(record) != expected_keys:
        raise SystemExit("development exam schema drifted")
    owner_id = _record_id("graph-dev", study_id, question["id"])
    paired_seed = stable_seed(master_seed, study_id, question["id"], "paired-attempt")
    protocol = {
        **attempt_protocol_record(ATTEMPT_ACCESS),
        "paired_seed": paired_seed,
        "only_manipulated_field": "note",
    }
    expected_fixed = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "static-graph-dev-exam",
        "question_id": question["id"],
        "question_artifact_sha256": sha256_json(question_artifact),
        "target": question["target"],
        "split": "dev",
        "owner_id": owner_id,
        "paired_seed": paired_seed,
        "attempt_protocol": protocol,
        "presented_inputs": {
            "with_note": {
                "question_sha256": sha256_text(question["question"]),
                "note_sha256": sha256_text(note),
                "note_bytes": len(note.encode("utf-8")),
            },
            "bare": {
                "question_sha256": sha256_text(question["question"]),
                "note_sha256": sha256_text(""),
                "note_bytes": 0,
            },
        },
    }
    if any(record.get(key) != value for key, value in expected_fixed.items()):
        raise SystemExit("paired development lineage or note-only contract drifted")
    attempts = record.get("attempts")
    if not isinstance(attempts, dict) or set(attempts) != {"with_note", "bare"}:
        raise SystemExit("paired development attempts are incomplete")
    for arm in ("with_note", "bare"):
        _validate_attempt_record(
            attempts[arm], seed=paired_seed, label=f"graph development {arm} attempt"
        )
    scores = {
        arm: verify_prediction(
            attempts[arm]["answer"], question["gold_edges"], source_view=rt
        )
        for arm in ("with_note", "bare")
    }
    expected_derived = {
        "scores": scores,
        "score_sha256": {arm: sha256_json(score) for arm, score in scores.items()},
        "verdicts": {arm: score_verdict(score) for arm, score in scores.items()},
        "deltas": {arm: _attempt_delta(score) for arm, score in scores.items()},
    }
    if any(record.get(key) != value for key, value in expected_derived.items()):
        raise SystemExit("development score drifted from deterministic verification")
    calls = _validate_calls(
        record.get("calls"),
        owner_id=owner_id,
        phases={"graph-dev-attempt-with_note", "graph-dev-attempt-bare"},
        seed=paired_seed,
        require_nonempty=any(
            attempts[arm]["status"] == "ok" for arm in ("with_note", "bare")
        ),
    )
    phase_counts = Counter(call["phase"] for call in calls)
    missing_successful_calls = [
        arm
        for arm in ("with_note", "bare")
        if attempts[arm]["status"] == "ok"
        and not phase_counts[f"graph-dev-attempt-{arm}"]
    ]
    if missing_successful_calls:
        raise SystemExit(
            "successful development arms must retain model calls: "
            + ", ".join(missing_successful_calls)
        )
    if record.get("usage") != usage_totals(calls):
        raise SystemExit("development usage drifted")
    _validate_environment_binding(
        sdir,
        task_manifest,
        record.get("environment_snapshot"),
        require_claim_ready=True,
    )
    return record


def _require_successful_training(records: Iterable[dict[str, Any]]) -> None:
    failed = sorted(
        record["question_id"]
        for record in records
        if record["attempt"]["status"] != "ok"
    )
    if failed:
        raise SystemExit(
            "graph-study training attempt failed; its immutable study ID cannot "
            f"be resumed ({', '.join(failed)}); this frozen construction has no "
            "treatment result"
        )


def _require_successful_development(records: Iterable[dict[str, Any]]) -> None:
    failed = sorted(
        f"{record['question_id']}:{arm}"
        for record in records
        for arm in ("with_note", "bare")
        if record["attempts"][arm]["status"] != "ok"
    )
    if failed:
        raise SystemExit(
            "graph-study development attempt failed; its immutable study ID cannot "
            f"be resumed ({', '.join(failed)}); this frozen construction has no "
            "treatment result"
        )


def _jsonl(records: Iterable[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    data = read_artifact_bytes(path)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SystemExit(f"invalid UTF-8 JSONL artifact: {path}") from error
    records = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line:
            raise SystemExit(f"blank JSONL record at {path}:{line_number}")
        value = strict_json_loads(line, label=f"{path}:{line_number}")
        if not isinstance(value, dict):
            raise SystemExit(f"non-object JSONL record at {path}:{line_number}")
        records.append(value)
    return records


# ---------------------------------------------------------------------------
# Provenance and filesystem layout


def _study_dir(args: Any) -> Path:
    try:
        study_id = validate_id(args.study_id, "study ID")
    except (TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    sdir = ROOT / "study-graph" / "studies" / study_id / TASK
    cursor = ROOT
    for part in ("study-graph", "studies", study_id, TASK):
        cursor /= part
        if cursor.is_symlink():
            raise SystemExit(f"graph-study path must not traverse a symlink: {cursor}")
    return sdir


def _lock_path(args: Any) -> Path:
    return ROOT / ".studybench-locks" / "graph-study" / args.study_id / f"{TASK}-r1.lock"


@contextmanager
def _study_lock(args: Any):
    try:
        with exclusive_process_lock(_lock_path(args)):
            yield
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"graph study is already active or has an unsafe lock: {error}") from error


def _provenance_readiness(
    corpus: dict[str, Any],
    source: dict[str, Any],
    environment: dict[str, Any],
    urls: list[str],
) -> dict[str, bool]:
    try:
        server_count_matches = int(environment["server_count"]) == len(urls)
    except (KeyError, TypeError, ValueError):
        server_count_matches = False
    return {
        "corpus_pinned_clean": (
            corpus.get("name") == TASK
            and corpus.get("commit") == CORPORA[TASK].commit
            and corpus.get("dirty") is False
        ),
        "source_pinned_clean": bool(
            source.get("git_commit")
            and source.get("tree_sha256")
            and source.get("files")
            and source.get("dirty") is False
        ),
        "environment_complete": environment_is_claim_ready(environment),
        "model_revision_pinned": bool(MODEL_REVISION),
        "server_count_matches_environment": server_count_matches,
    }


def _write_task_manifest(
    args: Any,
    sdir: Path,
    urls: list[str],
    resolver: dict[str, Any],
    full_bank: list[dict[str, Any]],
) -> dict[str, Any]:
    try:
        corpus = corpus_record(CORPORA[TASK])
        source = source_record()
        current_environment = environment_record()
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"cannot record graph-study provenance: {error}") from error
    path = sdir / "manifest.json"
    existing = load_json_artifact(path) if path.exists() else None
    baseline = existing.get("environment") if isinstance(existing, dict) else current_environment
    if not isinstance(baseline, dict):
        raise SystemExit("existing graph-study manifest has no environment baseline")
    if existing is not None and not environments_compatible(baseline, current_environment):
        raise SystemExit("graph-study environment drifted; choose a new --study-id")
    readiness = _provenance_readiness(corpus, source, baseline, urls)
    current_readiness = _provenance_readiness(corpus, source, current_environment, urls)
    if not args.smoke and (not all(readiness.values()) or not all(current_readiness.values())):
        failed = sorted({
            key for values in (readiness, current_readiness)
            for key, ready in values.items() if not ready
        })
        raise SystemExit(
            "research graph study requires complete clean provenance; failed: "
            + ", ".join(failed)
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "manifest_type": STATIC_GRAPH_TASK_MANIFEST_TYPE,
        "study_id": args.study_id,
        "task": TASK,
        "round": ROUND,
        "master_seed": args.seed,
        "source_root": SOURCE_ROOT,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "sampling": SAMPLING,
        "corpus_commit": CORPORA[TASK].commit,
        "corpus": corpus,
        "source": source,
        "environment": baseline,
        "environment_contract": environment_contract_record(baseline),
        "provenance_readiness": readiness,
        "automated_provenance_ready": all(readiness.values()),
        "resolver_contract": resolver,
        "resolver_contract_sha256": sha256_json(resolver),
        "question_bank_sha256": resolver["question_bank_sha256"],
        "question_bank_artifact_sha256": sha256_json(full_bank),
        "server_transport": {
            "scope": "loopback",
            "protocol": "openai-compatible-http",
            "server_count": len(urls),
            "assignment": "stable_seed(master_seed, owner_id, server) modulo server_count",
        },
        "config": {
            "method": STATIC_GRAPH_METHOD,
            "smoke": args.smoke,
            "concurrency": args.concurrency,
            "attempt_protocol": attempt_protocol_record(ATTEMPT_ACCESS),
            "train_input_note_sha256": sha256_text(""),
            "train_question_count": 1 if args.smoke else TRAIN_QUESTION_COUNT,
            "dev_question_count": 0 if args.smoke else DEV_QUESTION_COUNT,
            "dev_holdout_targets": (
                [] if args.smoke else list(FROZEN_DEV_TARGETS)
            ),
            "provider_retries": 0,
            "read_max_lines": READ_MAX_LINES,
        },
    }
    if existing is not None:
        if canonical_json_bytes(existing) != canonical_json_bytes(manifest):
            raise SystemExit("graph-study task manifest drifted; choose a new --study-id")
        return existing
    write_immutable_json(path, manifest)
    return manifest


def _snapshot_launch(
    sdir: Path,
    task_manifest: dict[str, Any],
    *,
    smoke: bool,
) -> dict[str, Any]:
    try:
        environment = environment_record()
        if not environments_compatible(task_manifest["environment"], environment):
            raise ValueError("launch environment differs from task baseline")
        if not smoke and not environment_is_claim_ready(environment):
            raise ValueError("launch environment is not claim-ready")
        record = write_environment_snapshot(
            sdir, PurePosixPath("r1/environments"), environment
        )
        validate_environment_snapshot(
            sdir,
            record,
            baseline=task_manifest["environment"],
            require_claim_ready=not smoke,
        )
        return record
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"cannot snapshot graph-study launch: {error}") from error


def _round_contract(
    args: Any,
    task_manifest: dict[str, Any],
    resolver: dict[str, Any],
    selected: list[dict[str, Any]],
    initial_environment: dict[str, Any] | None,
) -> dict[str, Any]:
    train = [question["id"] for question in selected if question["split"] == "train"]
    dev = [question["id"] for question in selected if question["split"] == "dev"]
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_type": "deterministic-static-graph-study-round",
        "study_id": args.study_id,
        "task": TASK,
        "round": ROUND,
        "master_seed": args.seed,
        "source_root": SOURCE_ROOT,
        "smoke": args.smoke,
        "task_manifest_sha256": sha256_json(task_manifest),
        "resolver_contract_sha256": sha256_json(resolver),
        "question_bank_sha256": resolver["question_bank_sha256"],
        "selected_question_ids": [question["id"] for question in selected],
        "selected_question_bank_sha256": sha256_json(selected),
        "train_question_ids": train,
        "dev_question_ids": dev,
        "train_input_note_sha256": sha256_text(""),
        "initial_environment_snapshot": initial_environment,
    }


def _write_or_validate_round_manifest(
    args: Any,
    rdir: Path,
    task_manifest: dict[str, Any],
    resolver: dict[str, Any],
    selected: list[dict[str, Any]],
    initial_environment: dict[str, Any] | None,
) -> dict[str, Any]:
    path = rdir / "manifest.json"
    if path.exists():
        existing = load_json_artifact(path)
        expected = _round_contract(
            args,
            task_manifest,
            resolver,
            selected,
            existing.get("initial_environment_snapshot") if isinstance(existing, dict) else None,
        )
        if canonical_json_bytes(existing) != canonical_json_bytes(expected):
            raise SystemExit("graph-study round manifest drifted")
        if existing["initial_environment_snapshot"] is None:
            raise SystemExit("graph-study round has no initial launch provenance")
        _validate_environment_binding(
            rdir.parent,
            task_manifest,
            existing["initial_environment_snapshot"],
            require_claim_ready=not args.smoke,
        )
        return existing
    if initial_environment is None:
        raise SystemExit("first graph-study launch has no environment snapshot")
    manifest = _round_contract(
        args, task_manifest, resolver, selected, initial_environment
    )
    write_immutable_json(path, manifest)
    return manifest


def _regular_file_inventory(sdir: Path, paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for path in sorted(set(paths)):
        try:
            relative = path.relative_to(sdir).as_posix()
            data = read_artifact_bytes(path)
        except (OSError, ValueError) as error:
            raise SystemExit(f"construction artifact is missing or unsafe: {path}") from error
        inventory[relative] = {"sha256": sha256_bytes(data), "bytes": len(data)}
    return inventory


def _validate_launch_environments(
    sdir: Path,
    task_manifest: dict[str, Any],
    *,
    smoke: bool,
) -> set[str]:
    """Validate every launch snapshot, including crash-orphaned checkpoints."""

    paths = sorted((sdir / "r1" / "environments").glob("environment-*.json"))
    if not paths:
        raise SystemExit("graph study has no launch-environment snapshot")
    relatives: set[str] = set()
    for path in paths:
        try:
            data = read_artifact_bytes(path)
        except (OSError, ValueError) as error:
            raise SystemExit(f"invalid launch-environment artifact: {path}") from error
        digest = sha256_bytes(data)
        if path.name != f"environment-{digest}.json":
            raise SystemExit(f"launch-environment filename is not content addressed: {path}")
        relative = path.relative_to(sdir).as_posix()
        record = {
            "schema_version": 1,
            "sha256": digest,
            "bytes": len(data),
            "snapshot": relative,
        }
        _validate_environment_binding(
            sdir,
            task_manifest,
            record,
            require_claim_ready=not smoke,
        )
        relatives.add(relative)
    return relatives


def _all_files(sdir: Path) -> set[Path]:
    if not sdir.exists():
        return set()
    files = set()
    for path in sdir.rglob("*"):
        if path.is_symlink():
            raise SystemExit(f"graph-study artifact tree contains a symlink: {path}")
        if path.is_file():
            files.add(path)
    return files


def _expected_paths(
    sdir: Path,
    question_paths: dict[str, tuple[Path, dict[str, Any]]],
    train_paths: dict[str, Path],
    dev_paths: dict[str, Path],
    *,
    note_sha256: str | None,
    include_final: bool,
) -> set[Path]:
    rdir = sdir / "r1"
    paths = {
        sdir / "manifest.json",
        rdir / "manifest.json",
        rdir / "question-bank.json",
        *(path for path, _ in question_paths.values()),
        *train_paths.values(),
        *(rdir / "environments").glob("environment-*.json"),
    }
    if train_paths and all(path.is_file() for path in train_paths.values()):
        paths.add(rdir / "items.jsonl")
    if note_sha256 is not None:
        paths.update({
            sdir / "notes" / "note-r1.md",
            sdir / "notes" / "by-sha256" / f"{note_sha256}.md",
        })
    paths.update(path for path in dev_paths.values() if path.is_file())
    if dev_paths and all(path.is_file() for path in dev_paths.values()):
        paths.add(rdir / "dev-exam.jsonl")
    if include_final:
        paths.update({rdir / "usage.jsonl", rdir / "summary.json"})
    return paths


def _reject_unknown_files(sdir: Path, allowed: set[Path]) -> None:
    note_manifest = sdir / "notes" / "note-r1.manifest.json"
    unexpected = _all_files(sdir) - allowed - {note_manifest}
    if unexpected:
        relative = sorted(path.relative_to(sdir).as_posix() for path in unexpected)
        raise SystemExit(f"unexpected graph-study artifacts: {relative}")


def _reject_unknown_layout(
    sdir: Path,
    question_paths: dict[str, tuple[Path, dict[str, Any]]],
    train_paths: dict[str, Path],
    dev_paths: dict[str, Path],
) -> None:
    """Reject foreign files before any resumed model call can be issued."""

    rdir = sdir / "r1"
    allowed = {
        sdir / "manifest.json",
        rdir / "manifest.json",
        rdir / "question-bank.json",
        *(path for path, _ in question_paths.values()),
        *train_paths.values(),
        *dev_paths.values(),
        rdir / "items.jsonl",
        rdir / "dev-exam.jsonl",
        rdir / "usage.jsonl",
        rdir / "summary.json",
        sdir / "notes" / "note-r1.md",
        sdir / "notes" / "note-r1.manifest.json",
        *(rdir / "environments").glob("environment-*.json"),
    }
    content_notes = list((sdir / "notes" / "by-sha256").glob("*.md"))
    if len(content_notes) > 1 or any(
        not re.fullmatch(r"[0-9a-f]{64}\.md", path.name)
        for path in content_notes
    ):
        raise SystemExit("graph-study note content-addressed layout drifted")
    allowed.update(content_notes)
    unexpected = _all_files(sdir) - allowed
    if unexpected:
        relative = sorted(path.relative_to(sdir).as_posix() for path in unexpected)
        raise SystemExit(f"unexpected graph-study artifacts before resume: {relative}")

    train_pending = any(not path.is_file() for path in train_paths.values())
    dev_pending = any(not path.is_file() for path in dev_paths.values())
    if train_pending and any(path.exists() for path in (
        rdir / "items.jsonl",
        sdir / "notes" / "note-r1.md",
        *content_notes,
        *dev_paths.values(),
        rdir / "dev-exam.jsonl",
        rdir / "usage.jsonl",
        rdir / "summary.json",
        sdir / "notes" / "note-r1.manifest.json",
    )):
        raise SystemExit("downstream graph-study artifacts exist before training is complete")
    if dev_pending and any(path.exists() for path in (
        rdir / "dev-exam.jsonl",
        rdir / "usage.jsonl",
        rdir / "summary.json",
        sdir / "notes" / "note-r1.manifest.json",
    )):
        raise SystemExit("downstream graph-study artifacts exist before dev is complete")


def _note_paths(sdir: Path, note: str) -> tuple[Path, Path, str]:
    digest = sha256_text(note)
    alias = sdir / "notes" / "note-r1.md"
    content = sdir / "notes" / "by-sha256" / f"{digest}.md"
    return alias, content, digest


def _response_models(calls: list[dict[str, Any]]) -> list[str]:
    return sorted({
        call["response_model"]
        for call in calls
        if isinstance(call.get("response_model"), str) and call["response_model"]
    })


def _summary(
    args: Any,
    train_records: list[dict[str, Any]],
    dev_records: list[dict[str, Any]],
    note: str,
    calls: list[dict[str, Any]],
    usage_audit: dict[str, Any],
) -> dict[str, Any]:
    train_verdicts = Counter(record["verdict"] for record in train_records)
    dev_verdicts = {
        arm: Counter(record["verdicts"][arm] for record in dev_records)
        for arm in ("with_note", "bare")
    } if dev_records else {}
    return {
        "schema_version": SCHEMA_VERSION,
        "study_id": args.study_id,
        "task": TASK,
        "round": ROUND,
        "source_root": SOURCE_ROOT,
        "smoke": args.smoke,
        "train_items": len(train_records),
        "dev_items": len(dev_records),
        "train_verdicts": dict(sorted(train_verdicts.items())),
        "train_mean_f1": (
            sum(record["score"]["f1"] for record in train_records) / len(train_records)
            if train_records else None
        ),
        "dev_verdicts": {
            arm: dict(sorted(counts.items())) for arm, counts in dev_verdicts.items()
        },
        "dev_mean_f1": {
            arm: sum(record["scores"][arm]["f1"] for record in dev_records)
            / len(dev_records)
            for arm in ("with_note", "bare")
        } if dev_records else {},
        "entries_admitted": sum(record["entry"] is not None for record in train_records),
        "note_sha256": sha256_text(note),
        "note_bytes": len(note.encode("utf-8")),
        "response_models": _response_models(calls),
        "usage": usage_totals(calls),
        "usage_by_phase": usage_by_phase(calls),
        "usage_audit": usage_audit,
        "claim_ready": False,
        "publication_claim_ready": False,
        "confirmatory_claim_ready": False,
    }


def _readiness(
    args: Any,
    task_manifest: dict[str, Any],
    resolver: dict[str, Any],
    full_bank: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    train_records: list[dict[str, Any]],
    dev_records: list[dict[str, Any]],
    note: str,
    entries: list[dict[str, Any]],
    calls: list[dict[str, Any]],
    usage_audit: dict[str, Any],
    rt: Any,
) -> dict[str, bool]:
    expected_note = render_graph_note(rt, entries)
    response_models = _response_models(calls)
    train_ids = {question["id"] for question in selected if question["split"] == "train"}
    dev_ids = {question["id"] for question in selected if question["split"] == "dev"}
    return {
        "non_smoke": not args.smoke,
        "provenance_complete": task_manifest.get("automated_provenance_ready") is True,
        "environment_contract_valid": environment_contract_is_valid(
            task_manifest.get("environment_contract"), task_manifest.get("environment")
        ),
        "resolver_contract_recomputed": (
            task_manifest.get("resolver_contract") == resolver
            and task_manifest.get("resolver_contract_sha256") == sha256_json(resolver)
        ),
        "question_bank_recomputed": (
            task_manifest.get("question_bank_sha256") == resolver.get("question_bank_sha256")
            and task_manifest.get("question_bank_artifact_sha256") == sha256_json(full_bank)
        ),
        "selection_exact": (
            len(train_ids) == TRAIN_QUESTION_COUNT
            and len(dev_ids) == DEV_QUESTION_COUNT
            and not args.smoke
        ),
        "training_complete": (
            len(train_records) == TRAIN_QUESTION_COUNT
            and all(record["attempt"]["status"] == "ok" for record in train_records)
        ),
        "training_empty_note": all(
            record["input_note_sha256"] == sha256_text("")
            and record["input_note_bytes"] == 0
            for record in train_records
        ),
        "training_scores_recomputed": all(
            record["score_sha256"] == sha256_json(record["score"])
            and record["verdict"] in VERDICTS
            for record in train_records
        ),
        "corrections_recomputed": (
            entries == [record["entry"] for record in train_records if record["entry"] is not None]
        ),
        "note_recomputed": note == expected_note,
        "dev_holdout_isolated": (
            len(dev_records) == DEV_QUESTION_COUNT
            and all(entry["origin_question_id"] in train_ids for entry in entries)
            and all(entry["origin_question_id"] not in dev_ids for entry in entries)
        ),
        "dev_evidence_locations_disjoint": not ({
            (edge["path"], edge["line"])
            for question in selected if question["split"] == "train"
            for edge in question["gold_edges"]
        } & {
            (edge["path"], edge["line"])
            for question in selected if question["split"] == "dev"
            for edge in question["gold_edges"]
        }),
        "dev_pair_complete": (
            len(dev_records) == DEV_QUESTION_COUNT
            and all(
                set(record["attempts"]) == {"with_note", "bare"}
                and all(record["attempts"][arm]["status"] == "ok"
                        for arm in ("with_note", "bare"))
                for record in dev_records
            )
        ),
        "dev_pair_note_only": all(
            record["attempts"]["with_note"]["seed"]
            == record["attempts"]["bare"]["seed"]
            == record["paired_seed"]
            and record["attempt_protocol"].get("only_manipulated_field") == "note"
            for record in dev_records
        ),
        "dev_scores_recomputed": all(
            record["score_sha256"]
            == {arm: sha256_json(record["scores"][arm]) for arm in ("with_note", "bare")}
            for record in dev_records
        ),
        "usage_complete": (
            usage_audit.get("complete") is True
            and artifact_usage_consistent(train_records + dev_records)
        ),
        "launch_environments_bound": all(
            isinstance(record.get("environment_snapshot"), dict)
            for record in train_records + dev_records
        ),
        "response_model_homogeneous": len(response_models) == 1,
        "response_model_expected": response_models == [MODEL_ID.removeprefix("openai/")],
    }


def _write_note_manifest(
    args: Any,
    sdir: Path,
    task_manifest: dict[str, Any],
    resolver: dict[str, Any],
    full_bank: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    train_records: list[dict[str, Any]],
    dev_records: list[dict[str, Any]],
    entries: list[dict[str, Any]],
    note: str,
    calls: list[dict[str, Any]],
    usage_audit: dict[str, Any],
    inventory: dict[str, dict[str, Any]],
    *,
    rt: Any,
) -> dict[str, Any]:
    readiness = _readiness(
        args,
        task_manifest,
        resolver,
        full_bank,
        selected,
        train_records,
        dev_records,
        note,
        entries,
        calls,
        usage_audit,
        rt,
    )
    readiness["construction_inventory_complete"] = bool(inventory)
    automated = all(readiness.values()) and not args.smoke
    digest = sha256_text(note)
    try:
        protocol_summary = derive_protocol_summary(
            read_artifact_bytes(sdir / "manifest.json")
        )
    except (OSError, ValueError, StudyProtocolError) as error:
        raise SystemExit(
            f"cannot bind graph note to study task manifest: {error}"
        ) from error
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "manifest_type": STATIC_GRAPH_NOTE_MANIFEST_TYPE,
        "method": STATIC_GRAPH_METHOD,
        "protocol_summary": protocol_summary,
        "study_id": args.study_id,
        "task": TASK,
        "round": ROUND,
        "corpus_commit": CORPORA[TASK].commit,
        "claim_ready": False,
        "publication_claim_ready": False,
        "confirmatory_claim_ready": False,
        "automated_claim_ready": automated,
        "automated_readiness": readiness,
        "human_audit": {
            "required_for_publication": True,
            "status": "not_performed",
        },
        "note_sha256": digest,
        "note_path": f"by-sha256/{digest}.md",
        "input_note_sha256": sha256_text(""),
        "entry_ids": [entry["entry_id"] for entry in entries],
        "entries": entries,
        "train_question_ids": [
            question["id"] for question in selected if question["split"] == "train"
        ],
        "held_out_dev_question_ids": [
            question["id"] for question in selected if question["split"] == "dev"
        ],
        "resolver_contract_sha256": sha256_json(resolver),
        "question_bank_sha256": resolver["question_bank_sha256"],
        "construction_artifacts": inventory,
        "construction_artifacts_sha256": sha256_json(inventory),
        "usage": usage_totals(calls),
        "usage_by_phase": usage_by_phase(calls),
        "usage_audit": usage_audit,
        "note_chars": len(note),
        "note_bytes": len(note.encode("utf-8")),
    }
    write_immutable_json(sdir / "notes" / "note-r1.manifest.json", manifest)
    return manifest


# ---------------------------------------------------------------------------
# Driver and completed-study audit


def _artifact_path(rdir: Path, directory: str, question: dict[str, Any]) -> Path:
    digest = sha256_text(question["id"])[:20]
    return rdir / directory / f"{digest}.json"


def _write_bank_artifacts(
    rdir: Path,
    resolver: dict[str, Any],
    full_bank: list[dict[str, Any]],
    selected: list[dict[str, Any]],
) -> dict[str, tuple[Path, dict[str, Any]]]:
    bank_artifact = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "static-graph-question-bank",
        "resolver_contract": resolver,
        "resolver_contract_sha256": sha256_json(resolver),
        "question_bank_sha256": resolver["question_bank_sha256"],
        "question_bank_artifact_sha256": sha256_json(full_bank),
        "full_bank": full_bank,
        "selected_question_ids": [question["id"] for question in selected],
        "selected_question_bank_sha256": sha256_json(selected),
    }
    write_immutable_json(rdir / "question-bank.json", bank_artifact)
    paths = _question_paths(rdir, selected, resolver)
    for path, artifact in paths.values():
        write_immutable_json(path, artifact)
        _validate_question_artifact(
            path,
            next(question for question in selected if question["id"] == artifact["question_id"]),
            resolver,
        )
    return paths


def _load_validate_records(
    args: Any,
    sdir: Path,
    task_manifest: dict[str, Any],
    selected: list[dict[str, Any]],
    question_paths: dict[str, tuple[Path, dict[str, Any]]],
    rt: Any,
) -> tuple[list[dict[str, Any]], dict[str, Path], dict[str, Path]]:
    rdir = sdir / "r1"
    train_questions = [question for question in selected if question["split"] == "train"]
    dev_questions = [question for question in selected if question["split"] == "dev"]
    train_paths = {
        question["id"]: _artifact_path(rdir, "items", question)
        for question in train_questions
    }
    dev_paths = {
        question["id"]: _artifact_path(rdir, "dev-exam", question)
        for question in dev_questions
    }
    train_records = []
    for question in train_questions:
        path = train_paths[question["id"]]
        if path.exists():
            train_records.append(_validate_training_record(
                load_json_artifact(path),
                question,
                question_paths[question["id"]][1],
                rt,
                study_id=args.study_id,
                master_seed=args.seed,
                sdir=sdir,
                task_manifest=task_manifest,
                smoke=args.smoke,
            ))
    return train_records, train_paths, dev_paths


def _validate_completed(
    args: Any,
    sdir: Path,
    task_manifest: dict[str, Any],
    resolver: dict[str, Any],
    full_bank: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    question_paths: dict[str, tuple[Path, dict[str, Any]]],
    train_paths: dict[str, Path],
    dev_paths: dict[str, Path],
    rt: Any,
) -> dict[str, Any]:
    manifest_path = sdir / "notes" / "note-r1.manifest.json"
    if not manifest_path.exists():
        raise SystemExit("graph study is incomplete: note manifest is absent")
    train_questions = [question for question in selected if question["split"] == "train"]
    dev_questions = [question for question in selected if question["split"] == "dev"]
    train_records = [
        _validate_training_record(
            load_json_artifact(train_paths[question["id"]]),
            question,
            question_paths[question["id"]][1],
            rt,
            study_id=args.study_id,
            master_seed=args.seed,
            sdir=sdir,
            task_manifest=task_manifest,
            smoke=args.smoke,
        )
        for question in train_questions
    ]
    _require_successful_training(train_records)
    entries = [record["entry"] for record in train_records if record["entry"] is not None]
    note = render_graph_note(rt, entries)
    alias, content, digest = _note_paths(sdir, note)
    if read_artifact_bytes(alias).decode("utf-8") != note \
            or read_artifact_bytes(content).decode("utf-8") != note:
        raise SystemExit("graph-study note bytes drifted")
    dev_records = [
        _validate_dev_record(
            load_json_artifact(dev_paths[question["id"]]),
            question,
            question_paths[question["id"]][1],
            note,
            rt,
            study_id=args.study_id,
            master_seed=args.seed,
            sdir=sdir,
            task_manifest=task_manifest,
        )
        for question in dev_questions
    ]
    _require_successful_development(dev_records)
    rdir = sdir / "r1"
    if _read_jsonl(rdir / "items.jsonl") != train_records:
        raise SystemExit("aggregate training JSONL drifted")
    if dev_records and _read_jsonl(rdir / "dev-exam.jsonl") != dev_records:
        raise SystemExit("aggregate development JSONL drifted")
    if not dev_records and (rdir / "dev-exam.jsonl").exists():
        raise SystemExit("smoke must not contain a development exam")
    calls = sorted(
        [call for record in train_records + dev_records for call in record["calls"]],
        key=lambda call: call["call_id"],
    )
    ledger = _read_jsonl(rdir / "usage.jsonl")
    usage_audit = usage_ledger_audit(calls, ledger)
    if not usage_audit["complete"] or ledger != calls:
        raise SystemExit("graph-study cumulative usage ledger drifted")
    expected_summary = _summary(
        args, train_records, dev_records, note, calls, usage_audit
    )
    if load_json_artifact(rdir / "summary.json") != expected_summary:
        raise SystemExit("graph-study summary drifted")
    allowed = _expected_paths(
        sdir,
        question_paths,
        train_paths,
        dev_paths,
        note_sha256=digest,
        include_final=True,
    )
    _validate_launch_environments(
        sdir, task_manifest, smoke=args.smoke
    )
    _reject_unknown_files(sdir, allowed)
    inventory = _regular_file_inventory(sdir, allowed)
    manifest = load_json_artifact(manifest_path)
    readiness = _readiness(
        args,
        task_manifest,
        resolver,
        full_bank,
        selected,
        train_records,
        dev_records,
        note,
        entries,
        calls,
        usage_audit,
        rt,
    )
    readiness["construction_inventory_complete"] = bool(inventory)
    try:
        task_bytes = read_artifact_bytes(sdir / "manifest.json")
        protocol_summary = derive_protocol_summary(task_bytes)
    except (OSError, ValueError, StudyProtocolError) as error:
        raise SystemExit(
            f"cannot bind graph note to study task manifest: {error}"
        ) from error
    expected_manifest = {
        "schema_version": SCHEMA_VERSION,
        "manifest_type": STATIC_GRAPH_NOTE_MANIFEST_TYPE,
        "method": STATIC_GRAPH_METHOD,
        "protocol_summary": protocol_summary,
        "study_id": args.study_id,
        "task": TASK,
        "round": ROUND,
        "corpus_commit": CORPORA[TASK].commit,
        "claim_ready": False,
        "publication_claim_ready": False,
        "confirmatory_claim_ready": False,
        "automated_claim_ready": all(readiness.values()) and not args.smoke,
        "automated_readiness": readiness,
        "human_audit": {"required_for_publication": True, "status": "not_performed"},
        "note_sha256": digest,
        "note_path": f"by-sha256/{digest}.md",
        "input_note_sha256": sha256_text(""),
        "entry_ids": [entry["entry_id"] for entry in entries],
        "entries": entries,
        "train_question_ids": [question["id"] for question in train_questions],
        "held_out_dev_question_ids": [question["id"] for question in dev_questions],
        "resolver_contract_sha256": sha256_json(resolver),
        "question_bank_sha256": resolver["question_bank_sha256"],
        "construction_artifacts": inventory,
        "construction_artifacts_sha256": sha256_json(inventory),
        "usage": usage_totals(calls),
        "usage_by_phase": usage_by_phase(calls),
        "usage_audit": usage_audit,
        "note_chars": len(note),
        "note_bytes": len(note.encode("utf-8")),
    }
    if canonical_json_bytes(manifest) != canonical_json_bytes(expected_manifest):
        raise SystemExit("graph-study note manifest or construction inventory drifted")
    try:
        dependency_bytes = {
            path.relative_to(sdir).as_posix(): read_artifact_bytes(path)
            for path in allowed
        }
        validate_study_note_archive(
            manifest,
            dependency_bytes,
            note.encode("utf-8"),
            expected_task=TASK,
            expected_model=MODEL_ID,
            expected_model_revision=MODEL_REVISION,
            expected_sampling=SAMPLING,
            expected_corpus_commit=CORPORA[TASK].commit,
            expected_corpus=task_manifest["corpus"],
            expected_source=task_manifest["source"],
            expected_environment=task_manifest["environment"],
            environments_compatible=environments_compatible,
            allow_smoke=True,
            deep_semantics=False,
        )
    except (KeyError, ValueError, StudyProtocolError) as error:
        raise SystemExit(
            f"graph-study note does not bind its construction protocol: {error}"
        ) from error
    if args.smoke and (
        manifest["automated_claim_ready"] is not False
        or manifest["automated_readiness"]["non_smoke"] is not False
        or dev_records
    ):
        raise SystemExit("smoke artifact was incorrectly made promotable")
    return manifest


def validate_bundled_graph_archive(
    note_manifest: dict[str, Any],
    construction_dependencies: dict[str, bytes],
    note_bytes: bytes,
) -> None:
    """Re-attest a graph archive against the live pinned DSPy source.

    The archive is first closed and hash-checked by ``study_protocol``.  This
    pass then reuses the constructor's exact AST resolver, frozen selection,
    deterministic verifier/correction renderer, paired-dev, usage, environment,
    summary, and manifest validation over a private materialization of those
    bytes.  It therefore does not trust self-consistent stored scores or gold.
    """

    import tempfile

    try:
        task_manifest = strict_json_loads(
            construction_dependencies["manifest.json"],
            label="bundled graph task manifest",
        )
        if not isinstance(task_manifest, dict):
            raise ValueError("bundled graph task manifest is not an object")
        smoke = task_manifest["config"]["smoke"]
        if type(smoke) is not bool:
            raise ValueError("bundled graph smoke identity is invalid")
        args = argparse.Namespace(
            study_id=task_manifest["study_id"],
            seed=task_manifest["master_seed"],
            smoke=smoke,
        )

        with tempfile.TemporaryDirectory(prefix="studybench-graph-audit-") as raw:
            sdir = Path(raw) / "study"
            for relative, data in construction_dependencies.items():
                path = sdir.joinpath(*PurePosixPath(relative).parts)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            manifest_path = sdir / "notes" / "note-r1.manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_bytes(canonical_json_bytes(note_manifest))

            rt = RepoTools(CORPORA[TASK], read_max_lines=READ_MAX_LINES)
            full_bank_raw = build_question_bank(rt)
            resolver = resolver_contract_record(rt)
            full_bank, selected = _validate_bank(
                full_bank_raw, resolver, smoke=smoke
            )
            rdir = sdir / "r1"
            observed_round = load_json_artifact(rdir / "manifest.json")
            if not isinstance(observed_round, dict):
                raise ValueError("bundled graph round manifest is not an object")
            expected_round = _round_contract(
                args,
                task_manifest,
                resolver,
                selected,
                observed_round.get("initial_environment_snapshot"),
            )
            if canonical_json_bytes(observed_round) != canonical_json_bytes(expected_round):
                raise ValueError("bundled graph round manifest drifted")
            _validate_environment_binding(
                sdir,
                task_manifest,
                observed_round["initial_environment_snapshot"],
                require_claim_ready=not smoke,
            )
            expected_bank = {
                "schema_version": SCHEMA_VERSION,
                "record_type": "static-graph-question-bank",
                "resolver_contract": resolver,
                "resolver_contract_sha256": sha256_json(resolver),
                "question_bank_sha256": resolver["question_bank_sha256"],
                "question_bank_artifact_sha256": sha256_json(full_bank),
                "full_bank": full_bank,
                "selected_question_ids": [
                    question["id"] for question in selected
                ],
                "selected_question_bank_sha256": sha256_json(selected),
            }
            if canonical_json_bytes(load_json_artifact(rdir / "question-bank.json")) \
                    != canonical_json_bytes(expected_bank):
                raise ValueError("bundled graph question bank drifted")
            question_paths = _question_paths(rdir, selected, resolver)
            for question_id, (path, _) in question_paths.items():
                question = next(
                    value for value in selected if value["id"] == question_id
                )
                _validate_question_artifact(path, question, resolver)
            train_paths = {
                question["id"]: _artifact_path(rdir, "items", question)
                for question in selected
                if question["split"] == "train"
            }
            dev_paths = {
                question["id"]: _artifact_path(rdir, "dev-exam", question)
                for question in selected
                if question["split"] == "dev"
            }
            observed = _validate_completed(
                args,
                sdir,
                task_manifest,
                resolver,
                full_bank,
                selected,
                question_paths,
                train_paths,
                dev_paths,
                rt,
            )
            if (
                canonical_json_bytes(observed) != canonical_json_bytes(note_manifest)
                or read_artifact_bytes(sdir / "notes" / "note-r1.md") != note_bytes
            ):
                raise ValueError("bundled graph final identity drifted")
    except StudyProtocolError:
        raise
    except SystemExit as error:
        raise StudyProtocolError(
            f"graph constructor re-attestation failed: {error}"
        ) from error
    except (KeyError, OSError, TypeError, UnicodeError, ValueError) as error:
        raise StudyProtocolError(
            f"graph constructor re-attestation failed: {error}"
        ) from error


def _run_locked(args: Any) -> Path:
    sdir = _study_dir(args)
    rdir = sdir / "r1"
    if any(path.is_symlink() for path in (sdir, rdir, sdir / "notes")):
        raise SystemExit("graph-study artifact directories must not be symlinks")
    try:
        urls = validate_local_server_urls(args.base_urls)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    rt = RepoTools(CORPORA[TASK], read_max_lines=READ_MAX_LINES)
    full_bank_raw = build_question_bank(rt)
    resolver = resolver_contract_record(rt)
    full_bank, selected = _validate_bank(full_bank_raw, resolver, smoke=args.smoke)
    task_manifest = _write_task_manifest(args, sdir, urls, resolver, full_bank)
    if not args.smoke:
        try:
            validate_local_server_urls(
                args.base_urls,
                expected_count=int(task_manifest["environment"]["server_count"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise SystemExit(
                "graph study requires loopback endpoints matching SB_NSERVE"
            ) from error

    # The first launch snapshot is part of the immutable round contract.  On
    # resume, new snapshots are created only when a model artifact is missing.
    round_path = rdir / "manifest.json"
    initial_environment = None if round_path.exists() else _snapshot_launch(
        sdir, task_manifest, smoke=args.smoke
    )
    round_manifest = _write_or_validate_round_manifest(
        args, rdir, task_manifest, resolver, selected, initial_environment
    )
    bank_path = rdir / "question-bank.json"
    question_paths = _write_bank_artifacts(rdir, resolver, full_bank, selected)
    bank_observed = load_json_artifact(bank_path)
    if (
        bank_observed.get("resolver_contract") != resolver
        or bank_observed.get("full_bank") != full_bank
        or bank_observed.get("selected_question_ids")
        != round_manifest["selected_question_ids"]
    ):
        raise SystemExit("graph question-bank artifact drifted")
    train_records, train_paths, dev_paths = _load_validate_records(
        args, sdir, task_manifest, selected, question_paths, rt
    )
    _reject_unknown_layout(sdir, question_paths, train_paths, dev_paths)
    _validate_launch_environments(sdir, task_manifest, smoke=args.smoke)
    final_manifest = sdir / "notes" / "note-r1.manifest.json"
    if final_manifest.exists():
        _validate_completed(
            args, sdir, task_manifest, resolver, full_bank, selected,
            question_paths, train_paths, dev_paths, rt,
        )
        return final_manifest

    # An error checkpoint is intentionally immutable: retaining it makes the
    # failed launch auditable, while failing here prevents silent retries under
    # the same stochastic/provenance identity or construction from failed text.
    _require_successful_training(train_records)

    train_questions = [question for question in selected if question["split"] == "train"]
    pending_train = [
        question for question in train_questions
        if not train_paths[question["id"]].exists()
    ]
    dev_questions = [question for question in selected if question["split"] == "dev"]
    pending_dev = [
        question for question in dev_questions
        if not dev_paths[question["id"]].exists()
    ]
    launch = None
    if pending_train or pending_dev:
        launch = _snapshot_launch(sdir, task_manifest, smoke=args.smoke)
    tools = make_tools(rt)

    def run_train(question: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        assert launch is not None
        artifact = question_paths[question["id"]][1]
        owner_id = _record_id("graph-train", args.study_id, question["id"])
        record = _training_record(
            question,
            artifact,
            rt,
            tools,
            _server_url(urls, args.seed, owner_id),
            study_id=args.study_id,
            master_seed=args.seed,
            launch_environment=launch,
        )
        path = train_paths[question["id"]]
        write_immutable_json(path, record)
        return question["id"], _validate_training_record(
            record,
            question,
            artifact,
            rt,
            study_id=args.study_id,
            master_seed=args.seed,
            sdir=sdir,
            task_manifest=task_manifest,
            smoke=args.smoke,
        )

    records_by_id = {record["question_id"]: record for record in train_records}
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for question_id, record in pool.map(run_train, pending_train):
            records_by_id[question_id] = record
    train_records = [records_by_id[question["id"]] for question in train_questions]
    write_immutable_text(rdir / "items.jsonl", _jsonl(train_records))
    _require_successful_training(train_records)

    entries = [record["entry"] for record in train_records if record["entry"] is not None]
    note = render_graph_note(rt, entries)
    alias, content, note_digest = _note_paths(sdir, note)
    write_immutable_text(content, note)
    write_immutable_text(alias, note)

    dev_records_by_id: dict[str, dict[str, Any]] = {}
    for question in dev_questions:
        path = dev_paths[question["id"]]
        artifact = question_paths[question["id"]][1]
        if path.exists():
            record = _validate_dev_record(
                load_json_artifact(path), question, artifact, note, rt,
                study_id=args.study_id, master_seed=args.seed, sdir=sdir,
                task_manifest=task_manifest,
            )
        else:
            assert launch is not None
            owner_id = _record_id("graph-dev", args.study_id, question["id"])
            record = _dev_record(
                question,
                artifact,
                note,
                rt,
                tools,
                _server_url(urls, args.seed, owner_id),
                study_id=args.study_id,
                master_seed=args.seed,
                launch_environment=launch,
            )
            write_immutable_json(path, record)
            record = _validate_dev_record(
                record, question, artifact, note, rt,
                study_id=args.study_id, master_seed=args.seed, sdir=sdir,
                task_manifest=task_manifest,
            )
        dev_records_by_id[question["id"]] = record
        _require_successful_development([record])
    dev_records = [dev_records_by_id[question["id"]] for question in dev_questions]
    _require_successful_development(dev_records)
    if dev_records:
        write_immutable_text(rdir / "dev-exam.jsonl", _jsonl(dev_records))

    calls = sorted(
        [call for record in train_records + dev_records for call in record["calls"]],
        key=lambda call: call["call_id"],
    )
    usage_audit = usage_ledger_audit(calls, calls)
    write_immutable_text(rdir / "usage.jsonl", _jsonl(calls))
    write_immutable_json(
        rdir / "summary.json",
        _summary(args, train_records, dev_records, note, calls, usage_audit),
    )
    allowed = _expected_paths(
        sdir,
        question_paths,
        train_paths,
        dev_paths,
        note_sha256=note_digest,
        include_final=True,
    )
    _validate_launch_environments(
        sdir, task_manifest, smoke=args.smoke
    )
    _reject_unknown_files(sdir, allowed)
    inventory = _regular_file_inventory(sdir, allowed)
    _write_note_manifest(
        args,
        sdir,
        task_manifest,
        resolver,
        full_bank,
        selected,
        train_records,
        dev_records,
        entries,
        note,
        calls,
        usage_audit,
        inventory,
        rt=rt,
    )
    _validate_completed(
        args, sdir, task_manifest, resolver, full_bank, selected,
        question_paths, train_paths, dev_paths, rt,
    )
    log.info(
        "graph study complete: train=%d dev=%d note=%s",
        len(train_records), len(dev_records), note_digest,
    )
    return final_manifest


def run_study(args: Any) -> Path:
    _study_dir(args)
    if type(args.seed) is not int:
        raise SystemExit("--seed must be an integer")
    if type(args.concurrency) is not int or args.concurrency < 1:
        raise SystemExit("--concurrency must be a positive integer")
    if type(args.smoke) is not bool or type(args.debug) is not bool:
        raise SystemExit("--smoke and --debug must be booleans")
    with _study_lock(args):
        return _run_locked(args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--base-urls", required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    try:
        validate_id(args.study_id, "study ID")
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(ROOT / "logs" / f"graph-study-{args.study_id}.log"),
        ],
    )
    print(run_study(args))


if __name__ == "__main__":
    main()
