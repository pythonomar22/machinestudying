"""Build and validate blinded manual-review packets for the frozen raw-Qwen screen.

This module never changes, retries, or substitutes for a Qwen grade.  It creates
the full-census sensitivity review specified in experiment 012: 120 answer rows
and 619 claim rows, blinded to arm, budget, rollout, Qwen labels, and aggregate
results.  The builder is deliberately pinned to the one completed SmallDSPy
raw screen so it cannot silently become a general post-hoc grading path.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re
from typing import Any, Callable

from . import provenance
from .dataset import CORPORA, ROOT, load_questions
from .integrity import (
    canonical_json_bytes,
    read_artifact_bytes,
    sha256_bytes,
    sha256_json,
    strict_json_loads,
    write_immutable_json,
    write_immutable_text,
)


SCREEN_ID = "smalldspy-base-cheatsheet-raw-qwen-20260714n"
SCREEN_ROOT = ROOT / "raw-qwen-screens" / SCREEN_ID
SCREEN_HASHES = {
    "intent.json": "85c0fc11ebd5e6f91264fb2aefc852ed1eab3751a5b1e2f7fecdbcc7b21601ab",
    "raw-audit.json": "59c1434044ee567aee090862b9ebe31bd362f048c4aee577e608a80af7674e42",
    "result.json": "d10ee48bc7e0ef85c696fddf195bdfe4348aeb1f85d288cc8995bc4f79631413",
}
FIRST_ORDER_PREFIX = "46001:"
CELL_ORDER_PREFIX = "46000:"
SECOND_ORDER_PREFIX = "46002:"
FROZEN_MANUAL_AUDIT_ID = "smalldspy-raw-qwen-n-manual-20260714-v3"
FIRST_PASS_MANIFEST_SHA256 = (
    "37a61ac8c843c38be9127e525d3c2e677e18c542af669632fed2bf9b5859305b"
)
FIRST_PASS_VALIDATION_SHA256 = (
    "72ad9ed6c187278c27f02a428f1e23281a17f267f1e31f62bfdcbeae3ad93eea"
)
PACKET_SCHEMA_VERSION = 1
GRADE_VIEW_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
REVIEW_SCHEMA_VERSION = 1
SECOND_MANIFEST_SCHEMA_VERSION = 1
SECOND_REVIEW_SCHEMA_VERSION = 1
SECOND_VALIDATION_SCHEMA_VERSION = 1
POST_REVIEW_SUMMARY_SCHEMA_VERSION = 1
EXPECTED_SECOND_PASS_SELECTION = {"answer": 0, "claim": 71, "total": 71}
ACCESS_TOOLS = [
    "filesystem reads limited to the assigned packet and cited corpus",
    "apply_patch for the assigned review output",
    "local JSON parse/coverage validation of packet and assigned output",
]
FORBIDDEN_CONTEXT = [
    "arm", "budget", "rollout", "run path", "grade path",
    "server slot", "Qwen label", "weighted total", "aggregate result",
]
SECOND_FORBIDDEN_CONTEXT = FORBIDDEN_CONTEXT + [
    "first-pass decision", "selection reason",
]

REVIEW_PROMPT = """You are an independent first-pass reviewer of frozen candidate answers.
Read only your packet and, when necessary, the cited files inside the pinned
SmallDSPy corpus. Do not inspect runs, raw-Qwen artifacts, reports, experiments,
other reviewers' files, or aggregate outcomes.

Review every row independently. Never infer a decision from a neighboring row
that repeats the same candidate. For an answer row, record answer_ok only when
the candidate is relevant, source-grounded, factually correct, and has no
material omission; otherwise record answer_incorrect, or uncertain when the
provided bundle and allowed corpus cannot resolve it. For a claim row, record 1
only when the candidate itself clearly, correctly, and completely satisfies
every requirement in that claim without contradiction. Missing, partial,
vague, merely implied, or contradicted content is 0. The gold answer and cited
evidence verify candidate assertions; they never fill in missing content.
For every answer row also verify whether `candidate_status` truthfully describes
the displayed candidate and mention any mismatch in the note. A truthful
`no_answer` status for an empty candidate is still `answer_incorrect` for task
correctness; it must never become `answer_ok` merely because its status is honest.

For each answer include a concise reason and flag any corpus/evidence issue. For
each claim include 0, 1, or uncertain; confidence high, medium, or low; an
ambiguity flag; a rubric/evidence-defect flag; and a concise source-grounded
note. Preserve malformed or incomplete candidate text exactly as shown. Do not
repair it. Do not attempt to discover arm, budget, rollout, Qwen label, or score.

Write one canonical JSON object with exactly these top-level fields:
`review_schema_version` equal to 1; `reviewer` equal to your packet's reviewer;
`packet_sha256` equal to the SHA-256 of the exact assigned packet-file bytes;
`review_prompt_sha256` copied from your packet; and
`reviews`, containing exactly one object for every packet row. An answer object
has `row_id`, `unit` equal to `answer`, `decision` equal to `answer_ok`,
`answer_incorrect`, or `uncertain`, `corpus_evidence_issue` as a Boolean, and a
nonempty concise `note`. A claim object has `row_id`, `unit` equal to `claim`,
`decision` as integer 0 or 1 or string `uncertain`, `confidence` equal to
`high`, `medium`, or `low`, `ambiguity` as a Boolean,
`rubric_evidence_defect` as a Boolean, and a nonempty concise `note`. Add no
other top-level fields. Save it only at the review output path named in your
packet. Parse it and verify exact row coverage before declaring completion.
Use canonical UTF-8 JSON: `json.dumps` with `allow_nan=False`,
`ensure_ascii=False`, `separators=(",", ":")`, and `sort_keys=True`, followed
by exactly one newline. Do not add any top-level or review-row fields beyond
the exact fields listed above.
"""

SECOND_REVIEW_PROMPT = """You are an independent second-pass reviewer of frozen candidate answers.
Read only your assigned packet and, when necessary, the cited files inside the
pinned SmallDSPy corpus. Do not inspect first-pass reviews, runs, raw-Qwen
artifacts, reports, experiments, manifests, validation files, other reviewers'
files, or aggregate outcomes.

Review every displayed row independently. Never infer a decision from a
neighboring row that repeats the same candidate. For an answer row, record
answer_ok only when the candidate is relevant, source-grounded, factually
correct, and has no material omission; otherwise record answer_incorrect, or
uncertain when the provided bundle and allowed corpus cannot resolve it. For a
claim row, record 1 only when the candidate itself clearly, correctly, and
completely satisfies every requirement in that claim without contradiction.
Missing, partial, vague, merely implied, or contradicted content is 0. The gold
answer and cited evidence verify candidate assertions; they never fill in
missing content. For every answer row also verify whether `candidate_status`
truthfully describes the displayed candidate and mention any mismatch in the
note. A truthful `no_answer` status for an empty candidate is still
`answer_incorrect` for task correctness.

For each answer include a concise reason and flag any corpus/evidence issue. For
each claim include 0, 1, or uncertain; confidence high, medium, or low; an
ambiguity flag; a rubric/evidence-defect flag; and a concise source-grounded
note. Preserve malformed or incomplete candidate text exactly as shown. Do not
repair it. Do not attempt to discover arm, budget, rollout, Qwen label, score,
first-pass decision, or why a row was selected.

Write one canonical JSON object with exactly these top-level fields:
`second_pass_review_schema_version` equal to 1; `reviewer` equal to your
packet's reviewer; `packet_sha256` equal to the SHA-256 of the exact assigned
packet-file bytes; `review_prompt_sha256` copied from your packet; and
`reviews`, containing exactly one object for every packet row. An answer object
has `row_id`, `unit` equal to `answer`, `decision` equal to `answer_ok`,
`answer_incorrect`, or `uncertain`, `corpus_evidence_issue` as a Boolean, and a
nonempty concise `note`. A claim object has `row_id`, `unit` equal to `claim`,
`decision` as integer 0 or 1 or string `uncertain`, `confidence` equal to
`high`, `medium`, or `low`, `ambiguity` as a Boolean,
`rubric_evidence_defect` as a Boolean, and a nonempty concise `note`. Add no
other fields. Save it only at the review output path named in your packet.
Parse it and verify exact row coverage before declaring completion. Use
canonical UTF-8 JSON: `json.dumps` with `allow_nan=False`,
`ensure_ascii=False`, `separators=(",", ":")`, and `sort_keys=True`, followed
by exactly one newline.
"""


class ManualAuditError(RuntimeError):
    """The frozen screen or a manual-audit artifact failed validation."""


def _canonical_object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = read_artifact_bytes(path)
        value = strict_json_loads(raw, label=label)
    except (OSError, ValueError) as exc:
        raise ManualAuditError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict) or raw != canonical_json_bytes(value):
        raise ManualAuditError(f"{label} is not a canonical JSON object")
    return value, raw


def _load_screen() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    loaded: dict[str, tuple[dict[str, Any], bytes]] = {}
    for name, expected in SCREEN_HASHES.items():
        value, raw = _canonical_object(SCREEN_ROOT / name, f"raw screen {name}")
        if sha256_bytes(raw) != expected:
            raise ManualAuditError(f"raw screen {name} differs from its frozen hash")
        loaded[name] = (value, raw)
    intent, intent_raw = loaded["intent.json"]
    audit, audit_raw = loaded["raw-audit.json"]
    result, _ = loaded["result.json"]
    if (
        audit.get("complete") is not True
        or audit.get("terminal_error") is not None
        or audit.get("accepted_count") != 119
        or audit.get("request_count") != 119
        or audit.get("expected_request_count") != 119
        or result.get("claim_ready") is not False
        or result.get("judge_qualified") is not False
        or intent.get("claim_ready") is not False
        or intent.get("judge_qualified") is not False
        or audit.get("intent_sha256") != sha256_bytes(intent_raw)
        or result.get("intent", {}).get("sha256") != sha256_bytes(intent_raw)
        or result.get("raw_audit", {}).get("sha256") != sha256_bytes(audit_raw)
    ):
        raise ManualAuditError("raw screen linkage or terminal census is invalid")
    cells = intent.get("cells")
    requests = intent.get("requests")
    responses = audit.get("responses")
    if not isinstance(cells, list) or len(cells) != 120:
        raise ManualAuditError("raw screen must contain exactly 120 cells")
    if not isinstance(requests, list) or not isinstance(responses, list):
        raise ManualAuditError("raw screen request/response inventory is invalid")
    if len(requests) != 119 or len(responses) != 119:
        raise ManualAuditError("raw screen must contain exactly 119 judge calls")
    identity = ("request_index", "arm", "relative", "qid", "server_slot", "url", "payload_sha256")
    for request, response in zip(requests, responses, strict=True):
        if any(request.get(key) != response.get(key) for key in identity):
            raise ManualAuditError("raw response identity differs from request intent")
        if response.get("accepted") is not True:
            raise ManualAuditError("raw screen contains a rejected response")
    return intent, audit, result


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except (OSError, ValueError):
        return str(path.resolve())


def _order(prefix: str, digest: str) -> str:
    return sha256_bytes(f"{prefix}{digest}".encode("utf-8"))


def _row_id(cell_binding_sha256: str, unit: str, claim_id: str | None) -> str:
    return sha256_json({
        "audit_schema": 1,
        "cell_binding_sha256": cell_binding_sha256,
        "unit": unit,
        "claim_id": claim_id,
    })


def _validate_reviewers(reviewers: list[str]) -> list[str]:
    if len(reviewers) != 3 or len(set(reviewers)) != 3:
        raise ManualAuditError("exactly three distinct reviewer names are required")
    for reviewer in reviewers:
        if re.fullmatch(r"/root/raw_first_pass_[a-z0-9_]+", reviewer) is None:
            raise ManualAuditError(f"invalid canonical reviewer name: {reviewer!r}")
    return reviewers


def _builder_source_attestation() -> dict[str, Any]:
    """Bind packet construction to clean source already present on a remote."""

    from .raw_qwen_screen import _clean_pushed_source

    try:
        source = _clean_pushed_source()
        module_raw = read_artifact_bytes(Path(__file__))
    except (OSError, RuntimeError, ValueError) as exc:
        raise ManualAuditError("cannot attest clean pushed packet-builder source") from exc
    return {
        "clean_pushed_source": source,
        "module": {
            "path": _display_path(Path(__file__)),
            "sha256": sha256_bytes(module_raw),
            "bytes": len(module_raw),
        },
    }


def _validate_builder_source_attestation(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {"clean_pushed_source", "module"}:
        raise ManualAuditError("packet-builder source attestation is invalid")
    clean = value.get("clean_pushed_source")
    module = value.get("module")
    if (
        not isinstance(clean, dict)
        or set(clean) != {
            "policy", "source", "source_sha256", "remote_tracking_refs",
        }
        or not isinstance(module, dict)
        or set(module) != {"path", "sha256", "bytes"}
    ):
        raise ManualAuditError("packet-builder source attestation is incomplete")
    source = clean.get("source")
    refs = clean.get("remote_tracking_refs")
    path = module.get("path")
    if (
        clean.get("policy") != "clean-head-contained-in-remote-tracking-ref-v1"
        or not isinstance(source, dict)
        or source.get("dirty") is not False
        or clean.get("source_sha256") != sha256_json(source)
        or not isinstance(refs, list)
        or not refs
        or len(refs) != len(set(refs))
        or not all(isinstance(ref, str) and ref.startswith("refs/remotes/") for ref in refs)
        or any(ref.endswith("/HEAD") for ref in refs)
        or path != "studybench/raw_qwen_manual_audit.py"
        or type(module.get("bytes")) is not int
        or module["bytes"] <= 0
        or not isinstance(module.get("sha256"), str)
    ):
        raise ManualAuditError("packet-builder source attestation is invalid")
    try:
        provenance.validate_frozen_source_commit(source)
    except ValueError as exc:
        raise ManualAuditError("packet-builder source commit is invalid") from exc
    source_file = source.get("files", {}).get(path)
    if (
        not isinstance(source_file, dict)
        or source_file.get("sha256") != module["sha256"]
        or source_file.get("bytes") != module["bytes"]
    ):
        raise ManualAuditError("packet-builder module is not bound to source")


def build_packets(
    *, audit_id: str, reviewers: list[str], output_root: Path,
    source_attestor: Callable[[], dict[str, Any]] = _builder_source_attestation,
) -> Path:
    """Write immutable blinded packets and their pre-open manifest."""

    reviewers = _validate_reviewers(reviewers)
    if not audit_id or any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789-" for ch in audit_id):
        raise ManualAuditError("audit ID must be lowercase alphanumeric/hyphen")
    builder_source = source_attestor()
    _validate_builder_source_attestation(builder_source)
    intent, audit, _ = _load_screen()
    questions = {row["id"]: row for row in load_questions("smalldspy")}
    if len(questions) != 5:
        raise ManualAuditError("SmallDSPy question bundle is not the frozen five rows")
    responses = {(item["arm"], item["relative"]): item for item in audit["responses"]}
    audit_root = output_root / audit_id
    binding_root = audit_root / "operator-bindings"
    packet_root = audit_root / "first-pass"
    prompt_path = audit_root / "review-prompt.txt"
    write_immutable_text(prompt_path, REVIEW_PROMPT)
    prompt_raw = read_artifact_bytes(prompt_path)

    row_records: list[dict[str, Any]] = []
    cell_records: list[dict[str, Any]] = []
    arm_names = {"base": "base", "treatment": "cheatsheet"}
    no_answer_count = 0
    for cell_index, cell in enumerate(intent["cells"]):
        try:
            episode_path = ROOT / cell["episode_path"]
            episode, episode_raw = _canonical_object(episode_path, "frozen episode")
            row = questions[cell["qid"]]
        except (KeyError, TypeError) as exc:
            raise ManualAuditError("invalid raw-screen cell identity") from exc
        if (
            sha256_bytes(episode_raw) != cell.get("episode_sha256")
            or episode.get("qid") != cell.get("qid")
            or episode.get("budget") != cell.get("budget")
            or episode.get("rollout") != cell.get("rollout")
            or episode.get("status") != cell.get("status")
        ):
            raise ManualAuditError("raw-screen cell differs from its episode")
        response = responses.get((cell["arm"], cell["relative"]))
        no_answer = episode["status"] == "no_answer"
        if no_answer != (response is None):
            raise ManualAuditError("episode status differs from judge contact census")
        no_answer_count += int(no_answer)
        grade_view = {
            "raw_grade_view_schema_version": GRADE_VIEW_SCHEMA_VERSION,
            "screen_id": SCREEN_ID,
            "screen_intent_sha256": SCREEN_HASHES["intent.json"],
            "screen_raw_audit_sha256": SCREEN_HASHES["raw-audit.json"],
            "arm": arm_names[cell["arm"]],
            "qid": cell["qid"],
            "budget": cell["budget"],
            "rollout": cell["rollout"],
            "episode_sha256": cell["episode_sha256"],
            "episode_status": episode["status"],
            "judge_response": response,
            "itt_lenient": 0 if no_answer else response["lenient"],
        }
        grade_path = binding_root / f"cell-{cell_index:03d}.json"
        write_immutable_json(grade_path, grade_view)
        grade_raw = read_artifact_bytes(grade_path)
        grade_sha256 = sha256_bytes(grade_raw)
        cell_binding = sha256_json({
            "audit_schema": 1,
            "arm": arm_names[cell["arm"]],
            "qid": cell["qid"],
            "budget": cell["budget"],
            "rollout": cell["rollout"],
            "episode_sha256": cell["episode_sha256"],
            "grade_sha256": grade_sha256,
        })
        bundle_id = sha256_bytes(f"bundle:{cell_binding}".encode("utf-8"))
        answer_id = _row_id(cell_binding, "answer", None)
        row_records.append({
            "row_id": answer_id,
            "order_key": _order(FIRST_ORDER_PREFIX, answer_id),
            "cell_binding_sha256": cell_binding,
            "bundle_id": bundle_id,
            "unit": "answer",
            "candidate_status": episode["status"],
            "question": row["question"],
            "candidate_answer": episode["answer"],
            "gold_answer": row["gold_answer"],
            "claim_rubric": row["rubric"],
            "evidence": row["evidence"],
        })
        if not no_answer:
            evidence = {span["span_id"]: span for span in row["evidence"]}
            for claim in row["rubric"]:
                claim_id = claim["claim_id"]
                cited = []
                try:
                    cited = [evidence[span_id] for span_id in claim["span_ids"]]
                except KeyError as exc:
                    raise ManualAuditError("rubric cites a missing evidence span") from exc
                claim_row_id = _row_id(cell_binding, "claim", claim_id)
                row_records.append({
                    "row_id": claim_row_id,
                    "order_key": _order(FIRST_ORDER_PREFIX, claim_row_id),
                    "cell_binding_sha256": cell_binding,
                    "bundle_id": bundle_id,
                    "unit": "claim",
                    "question": row["question"],
                    "candidate_answer": episode["answer"],
                    "gold_answer": row["gold_answer"],
                    "claim": claim,
                    "evidence": cited,
                })
        cell_records.append({
            "cell_binding_sha256": cell_binding,
            "cell_order_key": _order(CELL_ORDER_PREFIX, cell_binding),
            "grade_path": _display_path(grade_path),
            "grade_sha256": grade_sha256,
        })

    units = Counter(item["unit"] for item in row_records)
    if len(cell_records) != 120 or units != {"answer": 120, "claim": 619} or no_answer_count != 1:
        raise ManualAuditError("manual-review census is not exactly 120 + 619 rows")
    row_ids = [item["row_id"] for item in row_records]
    order_keys = [item["order_key"] for item in row_records]
    if len(set(row_ids)) != 739 or len(set(order_keys)) != 739:
        raise ManualAuditError("manual-review row identity collision")

    assignment: dict[str, str] = {}
    for index, cell in enumerate(sorted(cell_records, key=lambda item: item["cell_order_key"])):
        assignment[cell["cell_binding_sha256"]] = reviewers[index % len(reviewers)]
    packet_records = []
    for packet_index, reviewer in enumerate(reviewers):
        review_output_path = audit_root / "first-pass-reviews" / f"review-{packet_index}.json"
        review_output_path.parent.mkdir(parents=True, exist_ok=True)
        allowed_access = {
            "tools": ACCESS_TOOLS,
            "filesystem_read": [
                f"assigned blinded packet-{packet_index}.json",
                str(CORPORA["smalldspy"].repo.resolve()),
            ],
            "filesystem_write": [_display_path(review_output_path)],
            "corpus_roots": list(CORPORA["smalldspy"].roots),
            "network": "none",
            "other_repository_files": "forbidden",
        }
        visible_rows = []
        for source in sorted(row_records, key=lambda item: item["order_key"]):
            if assignment[source["cell_binding_sha256"]] != reviewer:
                continue
            visible = {key: value for key, value in source.items() if key not in {
                "cell_binding_sha256", "order_key",
            }}
            visible_rows.append(visible)
        packet = {
            "packet_schema_version": PACKET_SCHEMA_VERSION,
            "independence_role": "blinded-first-pass",
            "reviewer": reviewer,
            "reviewer_model": "unavailable",
            "review_prompt": REVIEW_PROMPT,
            "review_prompt_sha256": sha256_bytes(prompt_raw),
            "review_output_path": _display_path(review_output_path),
            "allowed_access": allowed_access,
            "allowed_corpus": {
                "path": str(CORPORA["smalldspy"].repo.resolve()),
                "commit": CORPORA["smalldspy"].commit,
                "roots": list(CORPORA["smalldspy"].roots),
                "access": "read-only cited-source verification",
            },
            "forbidden_context": FORBIDDEN_CONTEXT,
            "rows": visible_rows,
        }
        packet_path = packet_root / f"packet-{packet_index}.json"
        write_immutable_json(packet_path, packet)
        packet_raw = read_artifact_bytes(packet_path)
        packet_records.append({
            "reviewer": reviewer,
            "reviewer_model": "unavailable",
            "independence_role": "blinded-first-pass",
            "review_prompt_sha256": sha256_bytes(prompt_raw),
            "review_output_path": _display_path(review_output_path),
            "allowed_access": allowed_access,
            "path": _display_path(packet_path),
            "sha256": sha256_bytes(packet_raw),
            "bytes": len(packet_raw),
            "answer_rows": sum(item["unit"] == "answer" for item in visible_rows),
            "claim_rows": sum(item["unit"] == "claim" for item in visible_rows),
        })
    if [item["answer_rows"] for item in packet_records] != [40, 40, 40]:
        raise ManualAuditError("whole-cell assignment did not give 40 answers per reviewer")

    manifest = {
        "manual_audit_manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "audit_id": audit_id,
        "screen_id": SCREEN_ID,
        "screen_artifacts": SCREEN_HASHES,
        "packet_builder_source": builder_source,
        "stage": "first-pass-packets-frozen-before-open",
        "blinded": True,
        "row_counts": {"answer": 120, "claim": 619, "total": 739},
        "assignment": "whole-cell-round-robin-after-sha256(46000:cell_binding_sha256)",
        "row_order": "sha256(46001:row_id)",
        "review_prompt": {
            "path": _display_path(prompt_path),
            "sha256": sha256_bytes(prompt_raw),
            "bytes": len(prompt_raw),
        },
        "allowed_corpus": {
            "path": str(CORPORA["smalldspy"].repo.resolve()),
            "commit": CORPORA["smalldspy"].commit,
            "roots": list(CORPORA["smalldspy"].roots),
        },
        "reviewers": packet_records,
        "operator_binding_views": {
            "count": 120,
            "inventory_sha256": sha256_json(cell_records),
            "not_exposed_to_reviewers": True,
        },
    }
    final_builder_source = source_attestor()
    _validate_builder_source_attestation(final_builder_source)
    if canonical_json_bytes(final_builder_source) != canonical_json_bytes(builder_source):
        raise ManualAuditError("packet-builder source changed during packet construction")
    manifest_path = audit_root / "pre-open-manifest.json"
    write_immutable_json(manifest_path, manifest)
    return manifest_path


def validate_first_pass(*, manifest_path: Path, review_paths: list[Path]) -> dict[str, Any]:
    """Validate full blinded first-pass coverage without revealing Qwen labels."""

    manifest, manifest_raw = _canonical_object(manifest_path, "manual-audit manifest")
    audit_root = manifest_path.resolve().parent
    manifest_keys = {
        "manual_audit_manifest_schema_version", "audit_id", "screen_id",
        "screen_artifacts", "packet_builder_source", "stage", "blinded",
        "row_counts", "assignment", "row_order", "review_prompt",
        "allowed_corpus", "reviewers", "operator_binding_views",
    }
    expected_corpus = {
        "path": str(CORPORA["smalldspy"].repo.resolve()),
        "commit": CORPORA["smalldspy"].commit,
        "roots": list(CORPORA["smalldspy"].roots),
    }
    if (
        set(manifest) != manifest_keys
        or manifest_path.resolve() != audit_root / "pre-open-manifest.json"
        or manifest.get("audit_id") != audit_root.name
        or type(manifest.get("manual_audit_manifest_schema_version")) is not int
        or manifest.get("manual_audit_manifest_schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("screen_id") != SCREEN_ID
        or manifest.get("screen_artifacts") != SCREEN_HASHES
        or manifest.get("stage") != "first-pass-packets-frozen-before-open"
        or manifest.get("blinded") is not True
        or manifest.get("row_counts") != {"answer": 120, "claim": 619, "total": 739}
        or manifest.get("assignment")
            != "whole-cell-round-robin-after-sha256(46000:cell_binding_sha256)"
        or manifest.get("row_order") != "sha256(46001:row_id)"
        or manifest.get("allowed_corpus") != expected_corpus
    ):
        raise ManualAuditError("manual-audit manifest is not a pre-open first pass")
    _validate_builder_source_attestation(manifest.get("packet_builder_source"))
    prompt_record = manifest.get("review_prompt")
    if not isinstance(prompt_record, dict) or set(prompt_record) != {"path", "sha256", "bytes"}:
        raise ManualAuditError("manual-audit manifest has no review prompt binding")
    prompt_path = ROOT / prompt_record.get("path", "")
    try:
        prompt_raw = read_artifact_bytes(prompt_path)
    except (OSError, ValueError) as exc:
        raise ManualAuditError("manual-audit review prompt is unavailable") from exc
    if (
        prompt_path.resolve() != audit_root / "review-prompt.txt"
        or prompt_raw != REVIEW_PROMPT.encode("utf-8")
        or prompt_record.get("sha256") != sha256_bytes(prompt_raw)
        or prompt_record.get("bytes") != len(prompt_raw)
    ):
        raise ManualAuditError("manual-audit review prompt binding is invalid")
    reviewer_records = manifest.get("reviewers")
    if not isinstance(reviewer_records, list) or len(reviewer_records) != 3:
        raise ManualAuditError("first pass requires exactly three reviewer records")
    expected_packets = {item.get("reviewer"): item for item in reviewer_records if isinstance(item, dict)}
    if len(expected_packets) != 3 or len(review_paths) != 3:
        raise ManualAuditError("first pass requires exactly three reviewer files")
    _validate_reviewers(list(expected_packets))
    operator = manifest.get("operator_binding_views")
    if (
        not isinstance(operator, dict)
        or set(operator) != {"count", "inventory_sha256", "not_exposed_to_reviewers"}
        or operator.get("count") != 120
        or operator.get("not_exposed_to_reviewers") is not True
        or not isinstance(operator.get("inventory_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", operator["inventory_sha256"]) is None
    ):
        raise ManualAuditError("operator binding-view inventory is invalid")
    reviewer_indices = {record["reviewer"]: index for index, record in enumerate(reviewer_records)}
    seen: set[str] = set()
    seen_reviewers: set[str] = set()
    expected_global: set[str] = set()
    expected_units = Counter()
    no_answer_rows = 0
    decisions = Counter()
    review_records = []
    for path in review_paths:
        review, review_raw = _canonical_object(path, "first-pass review")
        reviewer = review.get("reviewer")
        packet_record = expected_packets.get(reviewer)
        if packet_record is None or reviewer in seen_reviewers:
            raise ManualAuditError("reviewer is not bound by the pre-open manifest")
        seen_reviewers.add(reviewer)
        index = reviewer_indices[reviewer]
        expected_packet_path = audit_root / "first-pass" / f"packet-{index}.json"
        expected_output_path = audit_root / "first-pass-reviews" / f"review-{index}.json"
        reviewer_keys = {
            "reviewer", "reviewer_model", "independence_role",
            "review_prompt_sha256", "review_output_path", "allowed_access",
            "path", "sha256", "bytes", "answer_rows", "claim_rows",
        }
        if (
            set(packet_record) != reviewer_keys
            or packet_record.get("reviewer_model") != "unavailable"
            or packet_record.get("independence_role") != "blinded-first-pass"
            or packet_record.get("review_prompt_sha256") != sha256_bytes(prompt_raw)
            or not isinstance(packet_record.get("allowed_access"), dict)
            or (ROOT / packet_record.get("path", "")).resolve() != expected_packet_path
            or (ROOT / packet_record.get("review_output_path", "")).resolve()
                != expected_output_path
            or type(packet_record.get("answer_rows")) is not int
            or type(packet_record.get("claim_rows")) is not int
        ):
            raise ManualAuditError("reviewer pre-open binding is incomplete")
        access = packet_record["allowed_access"]
        expected_access = {
            "tools": ACCESS_TOOLS,
            "filesystem_read": [
                f"assigned blinded packet-{index}.json",
                str(CORPORA["smalldspy"].repo.resolve()),
            ],
            "filesystem_write": [_display_path(expected_output_path)],
            "corpus_roots": list(CORPORA["smalldspy"].roots),
            "network": "none",
            "other_repository_files": "forbidden",
        }
        if (
            set(access) != set(expected_access)
            or access != expected_access
        ):
            raise ManualAuditError("reviewer allowed-access binding is invalid")
        if (ROOT / packet_record.get("review_output_path", "")).resolve() != path.resolve():
            raise ManualAuditError("review file is not at its predeclared output path")
        packet_path = expected_packet_path
        packet, packet_raw = _canonical_object(packet_path, "blinded packet")
        if (
            sha256_bytes(packet_raw) != packet_record.get("sha256")
            or len(packet_raw) != packet_record.get("bytes")
        ):
            raise ManualAuditError("blinded packet changed after pre-open freeze")
        if (
            set(packet) != {
                "packet_schema_version", "independence_role", "reviewer",
                "reviewer_model", "review_prompt", "review_prompt_sha256",
                "review_output_path", "allowed_access", "allowed_corpus",
                "forbidden_context", "rows",
            }
            or type(packet.get("packet_schema_version")) is not int
            or packet.get("packet_schema_version") != PACKET_SCHEMA_VERSION
            or packet.get("reviewer") != reviewer
            or packet.get("reviewer_model") != "unavailable"
            or packet.get("independence_role") != "blinded-first-pass"
            or packet.get("review_prompt") != REVIEW_PROMPT
            or packet.get("review_prompt_sha256") != sha256_bytes(prompt_raw)
            or packet.get("review_output_path") != packet_record.get("review_output_path")
            or packet.get("allowed_access") != packet_record.get("allowed_access")
            or packet.get("allowed_corpus") != {
                **expected_corpus,
                "access": "read-only cited-source verification",
            }
            or packet.get("forbidden_context") != FORBIDDEN_CONTEXT
        ):
            raise ManualAuditError("blinded packet differs from its reviewer binding")
        packet_rows = packet.get("rows")
        if not isinstance(packet_rows, list):
            raise ManualAuditError("blinded packet has no row list")
        answer_packet_keys = {
            "row_id", "bundle_id", "unit", "candidate_status", "question",
            "candidate_answer", "gold_answer", "claim_rubric", "evidence",
        }
        claim_packet_keys = {
            "row_id", "bundle_id", "unit", "question", "candidate_answer",
            "gold_answer", "claim", "evidence",
        }
        for row in packet_rows:
            if not isinstance(row, dict):
                raise ManualAuditError("blinded packet row is not an object")
            unit = row.get("unit")
            expected_keys = answer_packet_keys if unit == "answer" else claim_packet_keys
            if (
                unit not in {"answer", "claim"}
                or set(row) != expected_keys
                or not isinstance(row.get("bundle_id"), str)
                or re.fullmatch(r"[0-9a-f]{64}", row["bundle_id"]) is None
                or not isinstance(row.get("question"), str)
                or not isinstance(row.get("candidate_answer"), str)
                or not isinstance(row.get("gold_answer"), str)
                or not isinstance(row.get("evidence"), list)
                or (unit == "answer" and row.get("candidate_status") not in {"ok", "no_answer"})
                or (unit == "answer" and not isinstance(row.get("claim_rubric"), list))
                or (unit == "claim" and not isinstance(row.get("claim"), dict))
            ):
                raise ManualAuditError("blinded packet row schema is invalid")
            expected_units[unit] += 1
            no_answer_rows += int(unit == "answer" and row["candidate_status"] == "no_answer")
        packet_ids = [row.get("row_id") for row in packet_rows if isinstance(row, dict)]
        if (
            len(packet_ids) != len(packet_rows)
            or any(
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in packet_ids
            )
            or any(row.get("unit") not in {"answer", "claim"} for row in packet_rows)
            or len(packet_ids) != len(set(packet_ids))
            or packet_ids != sorted(packet_ids, key=lambda value: _order(FIRST_ORDER_PREFIX, value))
            or sum(row.get("unit") == "answer" for row in packet_rows)
                != packet_record.get("answer_rows")
            or sum(row.get("unit") == "claim" for row in packet_rows)
                != packet_record.get("claim_rows")
            or set(packet_ids) & expected_global
        ):
            raise ManualAuditError("blinded packet row inventory/order is invalid")
        expected_global.update(packet_ids)
        if (
            set(review) != {
                "review_schema_version", "reviewer", "packet_sha256",
                "review_prompt_sha256", "reviews",
            }
            or type(review.get("review_schema_version")) is not int
            or review.get("review_schema_version") != REVIEW_SCHEMA_VERSION
            or review.get("packet_sha256") != packet_record["sha256"]
            or review.get("review_prompt_sha256") != packet["review_prompt_sha256"]
        ):
            raise ManualAuditError("review is not bound to its packet and prompt")
        expected = {row["row_id"]: row["unit"] for row in packet_rows}
        rows = review.get("reviews")
        if not isinstance(rows, list) or len(rows) != len(expected):
            raise ManualAuditError("review row count differs from packet")
        local_seen: set[str] = set()
        for item in rows:
            if not isinstance(item, dict):
                raise ManualAuditError("review row is not an object")
            row_id = item.get("row_id")
            unit = expected.get(row_id)
            if unit is None or row_id in local_seen or item.get("unit") != unit:
                raise ManualAuditError("review row identity is missing, duplicated, or wrong")
            local_seen.add(row_id)
            note = item.get("note")
            if not isinstance(note, str) or not note.strip():
                raise ManualAuditError("every review row requires a concise note")
            if unit == "answer":
                if (
                    set(item) != {
                        "row_id", "unit", "decision",
                        "corpus_evidence_issue", "note",
                    }
                    or item.get("decision") not in {"answer_ok", "answer_incorrect", "uncertain"}
                    or type(item.get("corpus_evidence_issue")) is not bool
                ):
                    raise ManualAuditError("invalid answer review decision")
            else:
                decision = item.get("decision")
                if (
                    set(item) != {
                        "row_id", "unit", "decision", "confidence",
                        "ambiguity", "rubric_evidence_defect", "note",
                    }
                    or not (
                        (type(decision) is int and decision in (0, 1))
                        or decision == "uncertain"
                    )
                    or item.get("confidence") not in {"high", "medium", "low"}
                    or type(item.get("ambiguity")) is not bool
                    or type(item.get("rubric_evidence_defect")) is not bool
                ):
                    raise ManualAuditError("invalid claim review decision")
            decisions[(unit, str(item["decision"]))] += 1
        if seen & local_seen:
            raise ManualAuditError("first-pass review rows overlap across reviewers")
        seen.update(local_seen)
        review_records.append({
            "reviewer": reviewer,
            "path": _display_path(path),
            "sha256": sha256_bytes(review_raw),
            "bytes": len(review_raw),
            "rows": len(rows),
        })
    if (
        len(expected_global) != 739
        or seen != expected_global
        or expected_units != {"answer": 120, "claim": 619}
        or no_answer_rows != 1
    ):
        raise ManualAuditError("first-pass reviews do not cover all 739 rows")
    return {
        "first_pass_validation_schema_version": 1,
        "manifest_path": _display_path(manifest_path),
        "manifest_sha256": sha256_bytes(manifest_raw),
        "complete": True,
        "blinded": True,
        "row_count": len(seen),
        "decision_counts": {
            f"{unit}:{decision}": count
            for (unit, decision), count in sorted(decisions.items())
        },
        "reviews": sorted(review_records, key=lambda item: item["reviewer"]),
    }


def write_first_pass_validation(
    *, manifest_path: Path, review_paths: list[Path], output_path: Path,
) -> Path:
    """Freeze the complete pre-reveal review census and exact review hashes."""

    result = validate_first_pass(
        manifest_path=manifest_path, review_paths=review_paths
    )
    expected_output = manifest_path.resolve().parent / "first-pass-validation.json"
    if output_path.resolve() != expected_output:
        raise ManualAuditError("first-pass validation path is not canonical")
    write_immutable_json(output_path, result)
    raw = read_artifact_bytes(output_path)
    if raw != canonical_json_bytes(result):
        raise ManualAuditError("durable first-pass validation failed verification")
    return output_path


def _bound_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _validate_second_reviewers(
    reviewers: list[str], *, first_reviewers: set[str],
) -> list[str]:
    if len(reviewers) != 3 or len(set(reviewers)) != 3:
        raise ManualAuditError("exactly three distinct second-pass reviewers are required")
    for reviewer in reviewers:
        if re.fullmatch(r"/root/raw_second_pass_[a-z0-9_]+", reviewer) is None:
            raise ManualAuditError(f"invalid canonical second-pass reviewer: {reviewer!r}")
    if set(reviewers) & first_reviewers:
        raise ManualAuditError("second-pass reviewers must be fresh identities")
    return reviewers


def _reconstruct_operator_rows(
    *, manifest_path: Path, manifest: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Rebuild every blinded row and its hidden grade binding from frozen bytes."""

    intent, audit, _ = _load_screen()
    questions = {row["id"]: row for row in load_questions("smalldspy")}
    if len(questions) != 5:
        raise ManualAuditError("SmallDSPy question bundle is not the frozen five rows")
    responses = {(item["arm"], item["relative"]): item for item in audit["responses"]}
    if len(responses) != 119:
        raise ManualAuditError("raw-screen response identity collision")

    visible_from_packets: dict[str, dict[str, Any]] = {}
    audit_root = manifest_path.resolve().parent
    for index, record in enumerate(manifest["reviewers"]):
        packet_path = audit_root / "first-pass" / f"packet-{index}.json"
        packet, _ = _canonical_object(packet_path, "frozen first-pass packet")
        if _bound_path(record["path"]).resolve() != packet_path:
            raise ManualAuditError("first-pass packet path changed")
        for row in packet["rows"]:
            row_id = row["row_id"]
            if row_id in visible_from_packets:
                raise ManualAuditError("first-pass packet row is duplicated")
            visible_from_packets[row_id] = row

    arm_names = {"base": "base", "treatment": "cheatsheet"}
    reconstructed: dict[str, dict[str, Any]] = {}
    cell_records: list[dict[str, Any]] = []
    for cell_index, cell in enumerate(intent["cells"]):
        try:
            episode_path = ROOT / cell["episode_path"]
            episode, episode_raw = _canonical_object(episode_path, "frozen episode")
            question = questions[cell["qid"]]
            arm = arm_names[cell["arm"]]
        except (KeyError, TypeError) as exc:
            raise ManualAuditError("invalid raw-screen cell identity") from exc
        response = responses.get((cell["arm"], cell["relative"]))
        no_answer = episode.get("status") == "no_answer"
        if (
            sha256_bytes(episode_raw) != cell.get("episode_sha256")
            or episode.get("qid") != cell.get("qid")
            or episode.get("budget") != cell.get("budget")
            or episode.get("rollout") != cell.get("rollout")
            or episode.get("status") != cell.get("status")
            or no_answer != (response is None)
        ):
            raise ManualAuditError("raw-screen cell differs from its frozen episode")
        expected_grade = {
            "raw_grade_view_schema_version": GRADE_VIEW_SCHEMA_VERSION,
            "screen_id": SCREEN_ID,
            "screen_intent_sha256": SCREEN_HASHES["intent.json"],
            "screen_raw_audit_sha256": SCREEN_HASHES["raw-audit.json"],
            "arm": arm,
            "qid": cell["qid"],
            "budget": cell["budget"],
            "rollout": cell["rollout"],
            "episode_sha256": cell["episode_sha256"],
            "episode_status": episode["status"],
            "judge_response": response,
            "itt_lenient": 0 if no_answer else response["lenient"],
        }
        grade_path = audit_root / "operator-bindings" / f"cell-{cell_index:03d}.json"
        grade, grade_raw = _canonical_object(grade_path, "operator grade binding")
        if grade != expected_grade:
            raise ManualAuditError("operator grade binding differs from the raw screen")
        grade_sha256 = sha256_bytes(grade_raw)
        cell_binding = sha256_json({
            "audit_schema": 1,
            "arm": arm,
            "qid": cell["qid"],
            "budget": cell["budget"],
            "rollout": cell["rollout"],
            "episode_sha256": cell["episode_sha256"],
            "grade_sha256": grade_sha256,
        })
        bundle_id = sha256_bytes(f"bundle:{cell_binding}".encode("utf-8"))
        cell_records.append({
            "cell_binding_sha256": cell_binding,
            "cell_order_key": _order(CELL_ORDER_PREFIX, cell_binding),
            "grade_path": _display_path(grade_path),
            "grade_sha256": grade_sha256,
        })
        qwen_claims: dict[str, int] = {}
        if response is not None:
            claims = response.get("claims")
            if not isinstance(claims, list):
                raise ManualAuditError("Qwen grade has no atomic claim list")
            for claim_grade in claims:
                if (
                    not isinstance(claim_grade, dict)
                    or set(claim_grade) != {"claim_id", "score"}
                    or not isinstance(claim_grade.get("claim_id"), str)
                    or type(claim_grade.get("score")) is not int
                    or claim_grade["score"] not in (0, 1)
                    or claim_grade["claim_id"] in qwen_claims
                ):
                    raise ManualAuditError("Qwen atomic claim label is invalid")
                qwen_claims[claim_grade["claim_id"]] = claim_grade["score"]
        expected_claim_ids = {claim["claim_id"] for claim in question["rubric"]}
        if not no_answer and set(qwen_claims) != expected_claim_ids:
            raise ManualAuditError("Qwen claim IDs differ from the frozen rubric")

        answer_id = _row_id(cell_binding, "answer", None)
        answer_visible = {
            "row_id": answer_id,
            "bundle_id": bundle_id,
            "unit": "answer",
            "candidate_status": episode["status"],
            "question": question["question"],
            "candidate_answer": episode["answer"],
            "gold_answer": question["gold_answer"],
            "claim_rubric": question["rubric"],
            "evidence": question["evidence"],
        }
        reconstructed[answer_id] = {
            "visible": answer_visible,
            "arm": arm,
            "qid": cell["qid"],
            "budget": cell["budget"],
            "rollout": cell["rollout"],
            "cell_binding_sha256": cell_binding,
            "qwen_label": None,
            "qwen_claims": [
                {"claim_id": claim_id, "score": qwen_claims[claim_id]}
                for claim_id in sorted(qwen_claims)
            ],
            "claim_weight": None,
        }
        if not no_answer:
            evidence = {span["span_id"]: span for span in question["evidence"]}
            for claim in question["rubric"]:
                claim_id = claim["claim_id"]
                weight = claim.get("weight")
                if type(weight) is not int or weight <= 0:
                    raise ManualAuditError("rubric claim weight is not a positive integer")
                try:
                    cited = [evidence[span_id] for span_id in claim["span_ids"]]
                except (KeyError, TypeError) as exc:
                    raise ManualAuditError("rubric cites a missing evidence span") from exc
                row_id = _row_id(cell_binding, "claim", claim_id)
                visible = {
                    "row_id": row_id,
                    "bundle_id": bundle_id,
                    "unit": "claim",
                    "question": question["question"],
                    "candidate_answer": episode["answer"],
                    "gold_answer": question["gold_answer"],
                    "claim": claim,
                    "evidence": cited,
                }
                if row_id in reconstructed:
                    raise ManualAuditError("reconstructed row identity collision")
                reconstructed[row_id] = {
                    "visible": visible,
                    "arm": arm,
                    "qid": cell["qid"],
                    "budget": cell["budget"],
                    "rollout": cell["rollout"],
                    "cell_binding_sha256": cell_binding,
                    "qwen_label": qwen_claims[claim_id],
                    "qwen_claims": None,
                    "claim_weight": weight,
                }

    operator = manifest["operator_binding_views"]
    if (
        len(cell_records) != 120
        or sha256_json(cell_records) != operator["inventory_sha256"]
        or len(reconstructed) != 739
        or set(reconstructed) != set(visible_from_packets)
        or any(
            reconstructed[row_id]["visible"] != visible
            for row_id, visible in visible_from_packets.items()
        )
    ):
        raise ManualAuditError("first-pass rows do not match frozen operator bindings")
    return reconstructed


def _load_frozen_first_pass(
    validation_path: Path,
) -> dict[str, Any]:
    """Revalidate the exact durable v3 first pass before any Qwen-label reveal."""

    validation_path = validation_path.resolve()
    audit_root = validation_path.parent
    if (
        audit_root.name != FROZEN_MANUAL_AUDIT_ID
        or validation_path != audit_root / "first-pass-validation.json"
    ):
        raise ManualAuditError("first-pass validation is not the frozen v3 namespace")
    validation, validation_raw = _canonical_object(
        validation_path, "frozen first-pass validation"
    )
    if sha256_bytes(validation_raw) != FIRST_PASS_VALIDATION_SHA256:
        raise ManualAuditError("first-pass validation differs from its frozen hash")
    manifest_path = _bound_path(validation.get("manifest_path", "")).resolve()
    if manifest_path != audit_root / "pre-open-manifest.json":
        raise ManualAuditError("first-pass manifest path is not canonical")
    manifest, manifest_raw = _canonical_object(manifest_path, "frozen first-pass manifest")
    if (
        sha256_bytes(manifest_raw) != FIRST_PASS_MANIFEST_SHA256
        or validation.get("manifest_sha256") != FIRST_PASS_MANIFEST_SHA256
    ):
        raise ManualAuditError("first-pass manifest differs from its frozen hash")
    review_records = validation.get("reviews")
    if not isinstance(review_records, list) or len(review_records) != 3:
        raise ManualAuditError("frozen first-pass validation has no three-review census")
    review_paths: list[Path] = []
    for record in review_records:
        if not isinstance(record, dict):
            raise ManualAuditError("frozen first-pass review binding is invalid")
        path = _bound_path(record.get("path", "")).resolve()
        try:
            raw = read_artifact_bytes(path)
        except (OSError, ValueError) as exc:
            raise ManualAuditError("frozen first-pass review is unavailable") from exc
        if (
            sha256_bytes(raw) != record.get("sha256")
            or len(raw) != record.get("bytes")
        ):
            raise ManualAuditError("frozen first-pass review differs from its binding")
        review_paths.append(path)
    recomputed = validate_first_pass(
        manifest_path=manifest_path, review_paths=review_paths
    )
    if validation != recomputed:
        raise ManualAuditError("first-pass validation does not exactly recompute")

    reviews: dict[str, dict[str, Any]] = {}
    for path in review_paths:
        review, _ = _canonical_object(path, "frozen first-pass review")
        reviewer = review["reviewer"]
        for decision in review["reviews"]:
            row_id = decision["row_id"]
            if row_id in reviews:
                raise ManualAuditError("first-pass decision is duplicated")
            reviews[row_id] = {"reviewer": reviewer, **decision}
    rows = _reconstruct_operator_rows(
        manifest_path=manifest_path, manifest=manifest
    )
    if set(reviews) != set(rows):
        raise ManualAuditError("first-pass decisions differ from reconstructed rows")
    return {
        "audit_root": audit_root,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_raw": manifest_raw,
        "validation": validation,
        "validation_path": validation_path,
        "validation_raw": validation_raw,
        "review_paths": review_paths,
        "reviews": reviews,
        "rows": rows,
    }


def _second_pass_selection(context: dict[str, Any]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    order_keys: set[str] = set()
    for row_id, row in context["rows"].items():
        review = context["reviews"][row_id]
        reasons: list[str] = []
        if row["visible"]["unit"] == "answer":
            if review["decision"] == "uncertain":
                reasons.append("answer_uncertain")
            if review["corpus_evidence_issue"]:
                reasons.append("answer_corpus_evidence_issue")
        else:
            decision = review["decision"]
            if decision == "uncertain":
                reasons.append("claim_uncertain")
            elif decision != row["qwen_label"]:
                reasons.append("reviewer_qwen_disagreement")
            if review["rubric_evidence_defect"]:
                reasons.append("claim_rubric_evidence_defect")
        if not reasons:
            continue
        order_key = _order(SECOND_ORDER_PREFIX, row_id)
        if order_key in order_keys:
            raise ManualAuditError("second-pass order-key collision")
        order_keys.add(order_key)
        selected.append({
            "row_id": row_id,
            "bundle_id": row["visible"]["bundle_id"],
            "unit": row["visible"]["unit"],
            "order_key": order_key,
            "selection_reasons": reasons,
            "first_pass_decision": review["decision"],
            "qwen_label": row["qwen_label"],
        })
    return sorted(selected, key=lambda item: item["order_key"])


def _require_frozen_selection_census(selected: list[dict[str, Any]]) -> None:
    census = {
        "answer": sum(item["unit"] == "answer" for item in selected),
        "claim": sum(item["unit"] == "claim" for item in selected),
        "total": len(selected),
    }
    if census != EXPECTED_SECOND_PASS_SELECTION:
        raise ManualAuditError(
            "second-pass selection differs from the frozen 0-answer/71-claim census"
        )


def _second_pass_access(*, audit_root: Path, packet_index: int) -> dict[str, Any]:
    output_path = audit_root / "second-pass-reviews" / f"review-{packet_index}.json"
    return {
        "tools": ACCESS_TOOLS,
        "filesystem_read": [
            f"assigned blinded second-pass packet-{packet_index}.json",
            str(CORPORA["smalldspy"].repo.resolve()),
        ],
        "filesystem_write": [_display_path(output_path)],
        "corpus_roots": list(CORPORA["smalldspy"].roots),
        "network": "none",
        "other_repository_files": "forbidden",
    }


def _selection_inventory(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "row_id": item["row_id"],
            "bundle_id": item["bundle_id"],
            "unit": item["unit"],
            "order_key": item["order_key"],
            "selection_reasons": item["selection_reasons"],
            "first_pass_decision": item["first_pass_decision"],
            "qwen_label": item["qwen_label"],
        }
        for item in selected
    ]


def _second_assignment(
    selected: list[dict[str, Any]], reviewers: list[str],
) -> dict[str, str]:
    bundle_keys: dict[str, str] = {}
    for item in selected:
        current = bundle_keys.get(item["bundle_id"])
        if current is None or item["order_key"] < current:
            bundle_keys[item["bundle_id"]] = item["order_key"]
    return {
        bundle_id: reviewers[index % len(reviewers)]
        for index, (bundle_id, _) in enumerate(
            sorted(bundle_keys.items(), key=lambda item: item[1])
        )
    }


def build_second_pass_packets(
    *, first_pass_validation_path: Path, reviewers: list[str],
    source_attestor: Callable[[], dict[str, Any]] = _builder_source_attestation,
) -> Path:
    """Freeze blinded second-pass packets after the durable first-pass census."""

    context = _load_frozen_first_pass(first_pass_validation_path)
    first_reviewers = {
        record["reviewer"] for record in context["validation"]["reviews"]
    }
    reviewers = _validate_second_reviewers(
        reviewers, first_reviewers=first_reviewers
    )
    builder_source = source_attestor()
    _validate_builder_source_attestation(builder_source)
    audit_root = context["audit_root"]
    prompt_path = audit_root / "second-pass-review-prompt.txt"
    manifest_path = audit_root / "second-pass-pre-open-manifest.json"
    reserved = [
        prompt_path,
        manifest_path,
        audit_root / "second-pass",
        audit_root / "second-pass-reviews",
        audit_root / "second-pass-validation.json",
        audit_root / "post-review-summary.json",
    ]
    if any(path.exists() for path in reserved):
        raise ManualAuditError("second-pass namespace already contains reserved artifacts")
    selected = _second_pass_selection(context)
    _require_frozen_selection_census(selected)
    assignment = _second_assignment(selected, reviewers)
    write_immutable_text(prompt_path, SECOND_REVIEW_PROMPT)
    prompt_raw = read_artifact_bytes(prompt_path)
    packet_root = audit_root / "second-pass"
    review_root = audit_root / "second-pass-reviews"
    review_root.mkdir(parents=True, exist_ok=False)
    packet_records: list[dict[str, Any]] = []
    for index, reviewer in enumerate(reviewers):
        output_path = review_root / f"review-{index}.json"
        access = _second_pass_access(audit_root=audit_root, packet_index=index)
        visible_rows = [
            context["rows"][item["row_id"]]["visible"]
            for item in selected
            if assignment[item["bundle_id"]] == reviewer
        ]
        packet = {
            "packet_schema_version": PACKET_SCHEMA_VERSION,
            "independence_role": "blinded-second-pass",
            "reviewer": reviewer,
            "reviewer_model": "unavailable",
            "review_prompt": SECOND_REVIEW_PROMPT,
            "review_prompt_sha256": sha256_bytes(prompt_raw),
            "review_output_path": _display_path(output_path),
            "allowed_access": access,
            "allowed_corpus": {
                "path": str(CORPORA["smalldspy"].repo.resolve()),
                "commit": CORPORA["smalldspy"].commit,
                "roots": list(CORPORA["smalldspy"].roots),
                "access": "read-only cited-source verification",
            },
            "forbidden_context": SECOND_FORBIDDEN_CONTEXT,
            "rows": visible_rows,
        }
        packet_path = packet_root / f"packet-{index}.json"
        write_immutable_json(packet_path, packet)
        packet_raw = read_artifact_bytes(packet_path)
        packet_records.append({
            "reviewer": reviewer,
            "reviewer_model": "unavailable",
            "independence_role": "blinded-second-pass",
            "review_prompt_sha256": sha256_bytes(prompt_raw),
            "review_output_path": _display_path(output_path),
            "allowed_access": access,
            "path": _display_path(packet_path),
            "sha256": sha256_bytes(packet_raw),
            "bytes": len(packet_raw),
            "answer_rows": sum(row["unit"] == "answer" for row in visible_rows),
            "claim_rows": sum(row["unit"] == "claim" for row in visible_rows),
        })
    first_pass_binding = {
        "validation_path": _display_path(context["validation_path"]),
        "validation_sha256": sha256_bytes(context["validation_raw"]),
        "validation_bytes": len(context["validation_raw"]),
        "manifest_path": _display_path(context["manifest_path"]),
        "manifest_sha256": sha256_bytes(context["manifest_raw"]),
        "reviews": context["validation"]["reviews"],
    }
    manifest = {
        "second_pass_manifest_schema_version": SECOND_MANIFEST_SCHEMA_VERSION,
        "audit_id": audit_root.name,
        "screen_id": SCREEN_ID,
        "screen_artifacts": SCREEN_HASHES,
        "packet_builder_source": builder_source,
        "stage": "second-pass-packets-frozen-before-open",
        "blinded": True,
        "first_pass": first_pass_binding,
        "selection": {
            "rule": (
                "all first-pass reviewer-Qwen claim disagreements, claim uncertain "
                "rows, claim rubric/evidence defects, and answers marked uncertain "
                "or corpus/evidence issue"
            ),
            "answer_rows": sum(item["unit"] == "answer" for item in selected),
            "claim_rows": sum(item["unit"] == "claim" for item in selected),
            "total": len(selected),
            "inventory_sha256": sha256_json(_selection_inventory(selected)),
            "qwen_labels_exposed_to_reviewers": False,
            "first_pass_decisions_exposed_to_reviewers": False,
            "selection_reasons_exposed_to_reviewers": False,
        },
        "assignment": (
            "whole-selected-bundle-round-robin-after-min-sha256(46002:row_id)"
        ),
        "row_order": "sha256(46002:row_id)",
        "review_prompt": {
            "path": _display_path(prompt_path),
            "sha256": sha256_bytes(prompt_raw),
            "bytes": len(prompt_raw),
        },
        "allowed_corpus": {
            "path": str(CORPORA["smalldspy"].repo.resolve()),
            "commit": CORPORA["smalldspy"].commit,
            "roots": list(CORPORA["smalldspy"].roots),
        },
        "reviewers": packet_records,
    }
    _load_screen()
    final_source = source_attestor()
    _validate_builder_source_attestation(final_source)
    if canonical_json_bytes(final_source) != canonical_json_bytes(builder_source):
        raise ManualAuditError("packet-builder source changed during second-pass construction")
    write_immutable_json(manifest_path, manifest)
    return manifest_path


def _load_second_pass_manifest(manifest_path: Path) -> dict[str, Any]:
    """Recompute a blinded second-pass manifest and every packet byte binding."""

    manifest_path = manifest_path.resolve()
    audit_root = manifest_path.parent
    if (
        audit_root.name != FROZEN_MANUAL_AUDIT_ID
        or manifest_path != audit_root / "second-pass-pre-open-manifest.json"
    ):
        raise ManualAuditError("second-pass manifest path is not canonical")
    manifest, manifest_raw = _canonical_object(
        manifest_path, "second-pass pre-open manifest"
    )
    expected_keys = {
        "second_pass_manifest_schema_version", "audit_id", "screen_id",
        "screen_artifacts", "packet_builder_source", "stage", "blinded",
        "first_pass", "selection", "assignment", "row_order",
        "review_prompt", "allowed_corpus", "reviewers",
    }
    if (
        set(manifest) != expected_keys
        or type(manifest.get("second_pass_manifest_schema_version")) is not int
        or manifest.get("second_pass_manifest_schema_version")
            != SECOND_MANIFEST_SCHEMA_VERSION
        or manifest.get("audit_id") != audit_root.name
        or manifest.get("screen_id") != SCREEN_ID
        or manifest.get("screen_artifacts") != SCREEN_HASHES
        or manifest.get("stage") != "second-pass-packets-frozen-before-open"
        or manifest.get("blinded") is not True
        or manifest.get("assignment")
            != "whole-selected-bundle-round-robin-after-min-sha256(46002:row_id)"
        or manifest.get("row_order") != "sha256(46002:row_id)"
    ):
        raise ManualAuditError("second-pass manifest schema or protocol is invalid")
    _validate_builder_source_attestation(manifest.get("packet_builder_source"))
    first_binding = manifest.get("first_pass")
    if not isinstance(first_binding, dict):
        raise ManualAuditError("second-pass manifest has no first-pass binding")
    first_validation_path = _bound_path(
        first_binding.get("validation_path", "")
    ).resolve()
    context = _load_frozen_first_pass(first_validation_path)
    expected_first_binding = {
        "validation_path": _display_path(context["validation_path"]),
        "validation_sha256": sha256_bytes(context["validation_raw"]),
        "validation_bytes": len(context["validation_raw"]),
        "manifest_path": _display_path(context["manifest_path"]),
        "manifest_sha256": sha256_bytes(context["manifest_raw"]),
        "reviews": context["validation"]["reviews"],
    }
    if first_binding != expected_first_binding:
        raise ManualAuditError("second-pass first-pass binding is incomplete or changed")
    prompt_record = manifest.get("review_prompt")
    if not isinstance(prompt_record, dict) or set(prompt_record) != {
        "path", "sha256", "bytes",
    }:
        raise ManualAuditError("second-pass prompt binding is invalid")
    prompt_path = _bound_path(prompt_record.get("path", "")).resolve()
    try:
        prompt_raw = read_artifact_bytes(prompt_path)
    except (OSError, ValueError) as exc:
        raise ManualAuditError("second-pass review prompt is unavailable") from exc
    if (
        prompt_path != audit_root / "second-pass-review-prompt.txt"
        or prompt_raw != SECOND_REVIEW_PROMPT.encode("utf-8")
        or prompt_record.get("sha256") != sha256_bytes(prompt_raw)
        or prompt_record.get("bytes") != len(prompt_raw)
    ):
        raise ManualAuditError("second-pass review prompt differs from its binding")
    expected_corpus = {
        "path": str(CORPORA["smalldspy"].repo.resolve()),
        "commit": CORPORA["smalldspy"].commit,
        "roots": list(CORPORA["smalldspy"].roots),
    }
    if manifest.get("allowed_corpus") != expected_corpus:
        raise ManualAuditError("second-pass corpus binding is invalid")
    selected = _second_pass_selection(context)
    _require_frozen_selection_census(selected)
    expected_selection = {
        "rule": (
            "all first-pass reviewer-Qwen claim disagreements, claim uncertain "
            "rows, claim rubric/evidence defects, and answers marked uncertain "
            "or corpus/evidence issue"
        ),
        "answer_rows": sum(item["unit"] == "answer" for item in selected),
        "claim_rows": sum(item["unit"] == "claim" for item in selected),
        "total": len(selected),
        "inventory_sha256": sha256_json(_selection_inventory(selected)),
        "qwen_labels_exposed_to_reviewers": False,
        "first_pass_decisions_exposed_to_reviewers": False,
        "selection_reasons_exposed_to_reviewers": False,
    }
    if manifest.get("selection") != expected_selection:
        raise ManualAuditError("second-pass selection is incomplete or changed")
    records = manifest.get("reviewers")
    if not isinstance(records, list) or len(records) != 3:
        raise ManualAuditError("second pass requires exactly three reviewer bindings")
    reviewers = [record.get("reviewer") for record in records if isinstance(record, dict)]
    if len(reviewers) != 3:
        raise ManualAuditError("second-pass reviewer binding is invalid")
    _validate_second_reviewers(
        reviewers,
        first_reviewers={
            record["reviewer"] for record in context["validation"]["reviews"]
        },
    )
    assignment = _second_assignment(selected, reviewers)
    reviewer_rows: dict[str, list[str]] = {}
    reviewer_by_row: dict[str, str] = {}
    for index, (reviewer, record) in enumerate(zip(reviewers, records, strict=True)):
        packet_path = audit_root / "second-pass" / f"packet-{index}.json"
        output_path = audit_root / "second-pass-reviews" / f"review-{index}.json"
        access = _second_pass_access(audit_root=audit_root, packet_index=index)
        visible_rows = [
            context["rows"][item["row_id"]]["visible"]
            for item in selected
            if assignment[item["bundle_id"]] == reviewer
        ]
        expected_packet = {
            "packet_schema_version": PACKET_SCHEMA_VERSION,
            "independence_role": "blinded-second-pass",
            "reviewer": reviewer,
            "reviewer_model": "unavailable",
            "review_prompt": SECOND_REVIEW_PROMPT,
            "review_prompt_sha256": sha256_bytes(prompt_raw),
            "review_output_path": _display_path(output_path),
            "allowed_access": access,
            "allowed_corpus": {
                **expected_corpus,
                "access": "read-only cited-source verification",
            },
            "forbidden_context": SECOND_FORBIDDEN_CONTEXT,
            "rows": visible_rows,
        }
        packet, packet_raw = _canonical_object(packet_path, "second-pass packet")
        expected_record = {
            "reviewer": reviewer,
            "reviewer_model": "unavailable",
            "independence_role": "blinded-second-pass",
            "review_prompt_sha256": sha256_bytes(prompt_raw),
            "review_output_path": _display_path(output_path),
            "allowed_access": access,
            "path": _display_path(packet_path),
            "sha256": sha256_bytes(packet_raw),
            "bytes": len(packet_raw),
            "answer_rows": sum(row["unit"] == "answer" for row in visible_rows),
            "claim_rows": sum(row["unit"] == "claim" for row in visible_rows),
        }
        if packet != expected_packet or record != expected_record:
            raise ManualAuditError(
                "second-pass packet leaked context or differs from its exact binding"
            )
        row_ids = [row["row_id"] for row in visible_rows]
        if row_ids != sorted(
            row_ids, key=lambda row_id: _order(SECOND_ORDER_PREFIX, row_id)
        ):
            raise ManualAuditError("second-pass packet order is invalid")
        reviewer_rows[reviewer] = row_ids
        for row_id in row_ids:
            if row_id in reviewer_by_row:
                raise ManualAuditError("second-pass row appears in multiple packets")
            reviewer_by_row[row_id] = reviewer
    expected_ids = {item["row_id"] for item in selected}
    if set(reviewer_by_row) != expected_ids:
        raise ManualAuditError("second-pass packets omit selected rows")
    return {
        **context,
        "second_manifest": manifest,
        "second_manifest_path": manifest_path,
        "second_manifest_raw": manifest_raw,
        "second_prompt_raw": prompt_raw,
        "selected": selected,
        "second_reviewers": records,
        "second_reviewer_rows": reviewer_rows,
        "second_reviewer_by_row": reviewer_by_row,
    }


def _validate_second_decision(item: object, *, unit: str) -> None:
    if not isinstance(item, dict):
        raise ManualAuditError("second-pass review row is not an object")
    note = item.get("note")
    if not isinstance(note, str) or not note.strip():
        raise ManualAuditError("every second-pass review row requires a note")
    if unit == "answer":
        if (
            set(item) != {
                "row_id", "unit", "decision", "corpus_evidence_issue", "note",
            }
            or item.get("decision")
                not in {"answer_ok", "answer_incorrect", "uncertain"}
            or type(item.get("corpus_evidence_issue")) is not bool
        ):
            raise ManualAuditError("invalid second-pass answer decision")
        return
    decision = item.get("decision")
    if (
        set(item) != {
            "row_id", "unit", "decision", "confidence", "ambiguity",
            "rubric_evidence_defect", "note",
        }
        or not (
            (type(decision) is int and decision in (0, 1))
            or decision == "uncertain"
        )
        or item.get("confidence") not in {"high", "medium", "low"}
        or type(item.get("ambiguity")) is not bool
        or type(item.get("rubric_evidence_defect")) is not bool
    ):
        raise ManualAuditError("invalid second-pass claim decision")


def validate_second_pass(
    *, manifest_path: Path, review_paths: list[Path],
) -> dict[str, Any]:
    """Validate exact second-pass coverage while keeping Qwen labels concealed."""

    context = _load_second_pass_manifest(manifest_path)
    if len(review_paths) != 3:
        raise ManualAuditError("second pass requires exactly three review files")
    records = {
        record["reviewer"]: record for record in context["second_reviewers"]
    }
    seen_reviewers: set[str] = set()
    seen_rows: set[str] = set()
    decisions = Counter()
    frozen_reviews: list[dict[str, Any]] = []
    for path in review_paths:
        path = path.resolve()
        review, review_raw = _canonical_object(path, "second-pass review")
        reviewer = review.get("reviewer")
        record = records.get(reviewer)
        if record is None or reviewer in seen_reviewers:
            raise ManualAuditError("second-pass reviewer is missing, duplicated, or unbound")
        seen_reviewers.add(reviewer)
        expected_output = _bound_path(record["review_output_path"]).resolve()
        if path != expected_output:
            raise ManualAuditError("second-pass review is not at its bound output path")
        if (
            set(review) != {
                "second_pass_review_schema_version", "reviewer", "packet_sha256",
                "review_prompt_sha256", "reviews",
            }
            or type(review.get("second_pass_review_schema_version")) is not int
            or review.get("second_pass_review_schema_version")
                != SECOND_REVIEW_SCHEMA_VERSION
            or review.get("packet_sha256") != record["sha256"]
            or review.get("review_prompt_sha256")
                != record["review_prompt_sha256"]
        ):
            raise ManualAuditError("second-pass review is not bound to packet and prompt")
        expected_ids = context["second_reviewer_rows"][reviewer]
        rows = review.get("reviews")
        if not isinstance(rows, list) or len(rows) != len(expected_ids):
            raise ManualAuditError("second-pass review row count differs from packet")
        expected_units = {
            row_id: context["rows"][row_id]["visible"]["unit"]
            for row_id in expected_ids
        }
        local_seen: set[str] = set()
        for item in rows:
            if not isinstance(item, dict):
                raise ManualAuditError("second-pass review row is not an object")
            row_id = item.get("row_id")
            unit = expected_units.get(row_id)
            if unit is None or row_id in local_seen or item.get("unit") != unit:
                raise ManualAuditError(
                    "second-pass review identity is missing, duplicated, or wrong"
                )
            _validate_second_decision(item, unit=unit)
            local_seen.add(row_id)
            decisions[(unit, str(item["decision"]))] += 1
        if local_seen != set(expected_ids) or seen_rows & local_seen:
            raise ManualAuditError("second-pass review coverage overlaps or is incomplete")
        seen_rows.update(local_seen)
        frozen_reviews.append({
            "reviewer": reviewer,
            "path": _display_path(path),
            "sha256": sha256_bytes(review_raw),
            "bytes": len(review_raw),
            "rows": len(rows),
        })
    expected_global = {item["row_id"] for item in context["selected"]}
    if seen_rows != expected_global or set(records) != seen_reviewers:
        raise ManualAuditError("second-pass reviews do not cover every selected row")
    _load_screen()
    return {
        "second_pass_validation_schema_version": SECOND_VALIDATION_SCHEMA_VERSION,
        "manifest_path": _display_path(context["second_manifest_path"]),
        "manifest_sha256": sha256_bytes(context["second_manifest_raw"]),
        "first_pass_validation": {
            "path": _display_path(context["validation_path"]),
            "sha256": sha256_bytes(context["validation_raw"]),
            "reviews": context["validation"]["reviews"],
        },
        "complete": True,
        "blinded": True,
        "row_count": len(seen_rows),
        "decision_counts": {
            f"{unit}:{decision}": count
            for (unit, decision), count in sorted(decisions.items())
        },
        "reviews": sorted(frozen_reviews, key=lambda item: item["reviewer"]),
    }


def write_second_pass_validation(
    *, manifest_path: Path, review_paths: list[Path], output_path: Path,
) -> Path:
    """Freeze second-review bytes without revealing or replacing Qwen labels."""

    result = validate_second_pass(
        manifest_path=manifest_path, review_paths=review_paths
    )
    expected_output = manifest_path.resolve().parent / "second-pass-validation.json"
    if output_path.resolve() != expected_output:
        raise ManualAuditError("second-pass validation path is not canonical")
    write_immutable_json(output_path, result)
    raw = read_artifact_bytes(output_path)
    if raw != canonical_json_bytes(result):
        raise ManualAuditError("durable second-pass validation failed verification")
    return output_path


def _load_frozen_second_pass(validation_path: Path) -> dict[str, Any]:
    validation_path = validation_path.resolve()
    audit_root = validation_path.parent
    if validation_path != audit_root / "second-pass-validation.json":
        raise ManualAuditError("second-pass validation path is not canonical")
    validation, validation_raw = _canonical_object(
        validation_path, "frozen second-pass validation"
    )
    manifest_path = _bound_path(validation.get("manifest_path", "")).resolve()
    if manifest_path != audit_root / "second-pass-pre-open-manifest.json":
        raise ManualAuditError("second-pass validation has a noncanonical manifest")
    review_records = validation.get("reviews")
    if not isinstance(review_records, list) or len(review_records) != 3:
        raise ManualAuditError("second-pass validation has no three-review census")
    review_paths: list[Path] = []
    for record in review_records:
        if not isinstance(record, dict):
            raise ManualAuditError("second-pass review binding is invalid")
        path = _bound_path(record.get("path", "")).resolve()
        try:
            raw = read_artifact_bytes(path)
        except (OSError, ValueError) as exc:
            raise ManualAuditError("second-pass review is unavailable") from exc
        if (
            sha256_bytes(raw) != record.get("sha256")
            or len(raw) != record.get("bytes")
        ):
            raise ManualAuditError("second-pass review differs from validation binding")
        review_paths.append(path)
    recomputed = validate_second_pass(
        manifest_path=manifest_path, review_paths=review_paths
    )
    if validation != recomputed:
        raise ManualAuditError("second-pass validation does not exactly recompute")
    context = _load_second_pass_manifest(manifest_path)
    second_reviews: dict[str, dict[str, Any]] = {}
    for path in review_paths:
        review, _ = _canonical_object(path, "frozen second-pass review")
        reviewer = review["reviewer"]
        for decision in review["reviews"]:
            row_id = decision["row_id"]
            if row_id in second_reviews:
                raise ManualAuditError("second-pass decision is duplicated")
            second_reviews[row_id] = {"reviewer": reviewer, **decision}
    expected = {item["row_id"] for item in context["selected"]}
    if set(second_reviews) != expected:
        raise ManualAuditError("second-pass decisions differ from selected rows")
    return {
        **context,
        "second_validation": validation,
        "second_validation_path": validation_path,
        "second_validation_raw": validation_raw,
        "second_review_paths": review_paths,
        "second_reviews": second_reviews,
    }


def _exact_ratio(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "fraction": None if denominator == 0 else f"{numerator}/{denominator}",
    }


def _claim_audit_metrics(
    context: dict[str, Any], *, arm: str | None,
    reviews: dict[str, dict[str, Any]] | None = None,
    row_ids: set[str] | None = None,
) -> dict[str, Any]:
    reviews = context["reviews"] if reviews is None else reviews
    claims = [
        (row_id, row)
        for row_id, row in context["rows"].items()
        if row["visible"]["unit"] == "claim"
        and (arm is None or row["arm"] == arm)
        and (row_ids is None or row_id in row_ids)
    ]
    confusion = Counter({
        "qwen_0_reviewer_0": 0,
        "qwen_0_reviewer_1": 0,
        "qwen_1_reviewer_0": 0,
        "qwen_1_reviewer_1": 0,
    })
    determinate = agreements = disagreement_weight = determinate_weight = 0
    total_weight = sum(row["claim_weight"] for _, row in claims)
    uncertain = ambiguity = defects = 0
    for row_id, row in claims:
        review = reviews[row_id]
        decision = review["decision"]
        ambiguity += int(review["ambiguity"])
        defects += int(review["rubric_evidence_defect"])
        if decision == "uncertain":
            uncertain += 1
            continue
        determinate += 1
        weight = row["claim_weight"]
        determinate_weight += weight
        qwen = row["qwen_label"]
        confusion[f"qwen_{qwen}_reviewer_{decision}"] += 1
        if decision == qwen:
            agreements += 1
        else:
            disagreement_weight += weight
    return {
        "claim_rows": len(claims),
        "determinate_rows": determinate,
        "uncertain_rows": uncertain,
        "agreement_rows": agreements,
        "disagreement_rows": determinate - agreements,
        "determinate_agreement": _exact_ratio(agreements, determinate),
        "confusion_0_1": dict(sorted(confusion.items())),
        "ambiguity_rows": ambiguity,
        "rubric_evidence_defect_rows": defects,
        "claim_weighted_disagreement": {
            "disagreement_weight": disagreement_weight,
            "determinate_weight": determinate_weight,
            "total_weight": total_weight,
            "among_determinate": _exact_ratio(
                disagreement_weight, determinate_weight
            ),
            "among_all_claim_weight": _exact_ratio(
                disagreement_weight, total_weight
            ),
        },
    }


def _selected_claim_sensitivity(
    context: dict[str, Any], *, arm: str | None,
) -> dict[str, Any]:
    selected_ids = {
        item["row_id"] for item in context["selected"]
        if item["unit"] == "claim"
        and (arm is None or context["rows"][item["row_id"]]["arm"] == arm)
    }
    confusion = Counter({
        "first_0_second_0": 0,
        "first_0_second_1": 0,
        "first_1_second_0": 0,
        "first_1_second_1": 0,
    })
    both_determinate = agreements = 0
    selected_weight = both_determinate_weight = 0
    first_second_disagreement_weight = 0
    confirmed_qwen_agreement_weight = 0
    confirmed_qwen_disagreement_weight = 0
    uncertain_in_either_weight = 0
    for row_id in selected_ids:
        row = context["rows"][row_id]
        weight = row["claim_weight"]
        selected_weight += weight
        first = context["reviews"][row_id]["decision"]
        second = context["second_reviews"][row_id]["decision"]
        if type(first) is not int or type(second) is not int:
            uncertain_in_either_weight += weight
            continue
        both_determinate += 1
        both_determinate_weight += weight
        confusion[f"first_{first}_second_{second}"] += 1
        if first == second:
            agreements += 1
            if first == row["qwen_label"]:
                confirmed_qwen_agreement_weight += weight
            else:
                confirmed_qwen_disagreement_weight += weight
        else:
            first_second_disagreement_weight += weight
    return {
        "conditional_on_first_pass_selection": True,
        "not_an_all_rows_estimate": True,
        "selected_claim_rows": len(selected_ids),
        "first_vs_qwen": _claim_audit_metrics(
            context, arm=arm, reviews=context["reviews"], row_ids=selected_ids
        ),
        "second_vs_qwen": _claim_audit_metrics(
            context,
            arm=arm,
            reviews=context["second_reviews"],
            row_ids=selected_ids,
        ),
        "first_vs_second": {
            "both_determinate_rows": both_determinate,
            "uncertain_in_either_rows": len(selected_ids) - both_determinate,
            "agreement_rows": agreements,
            "disagreement_rows": both_determinate - agreements,
            "determinate_agreement": _exact_ratio(agreements, both_determinate),
            "confusion_0_1": dict(sorted(confusion.items())),
        },
        "weight_accounting": {
            "selected_weight": selected_weight,
            "both_determinate_weight": both_determinate_weight,
            "uncertain_in_either_weight": uncertain_in_either_weight,
            "first_second_disagreement_weight": first_second_disagreement_weight,
            "both_confirm_qwen_agreement_weight": confirmed_qwen_agreement_weight,
            "both_confirm_qwen_disagreement_weight": (
                confirmed_qwen_disagreement_weight
            ),
            "first_second_disagreement_among_selected": _exact_ratio(
                first_second_disagreement_weight, selected_weight
            ),
            "confirmed_qwen_disagreement_among_selected": _exact_ratio(
                confirmed_qwen_disagreement_weight, selected_weight
            ),
        },
    }


def _review_without_identity(review: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in review.items()
        if key not in {"row_id", "unit"}
    }


def _row_identity(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "arm": row["arm"],
        "qid": row["qid"],
        "budget": row["budget"],
        "rollout": row["rollout"],
    }


def _claim_followup_entry(
    context: dict[str, Any], selected: dict[str, Any],
) -> dict[str, Any]:
    row_id = selected["row_id"]
    row = context["rows"][row_id]
    visible = row["visible"]
    return {
        "row_id": row_id,
        **_row_identity(row),
        "question": visible["question"],
        "candidate_answer": visible["candidate_answer"],
        "gold_answer": visible["gold_answer"],
        "claim": visible["claim"],
        "evidence": visible["evidence"],
        "qwen_label": row["qwen_label"],
        "selection_reasons": selected["selection_reasons"],
        "first_review": _review_without_identity(context["reviews"][row_id]),
        "second_review": _review_without_identity(context["second_reviews"][row_id]),
    }


def build_post_review_summary(context: dict[str, Any]) -> dict[str, Any]:
    """Reveal labels only after both immutable review passes validate."""

    first_counts = Counter()
    first_second_counts = Counter()
    selection_reasons = Counter()
    for item in context["selected"]:
        row_id = item["row_id"]
        unit = item["unit"]
        first = context["reviews"][row_id]["decision"]
        second = context["second_reviews"][row_id]["decision"]
        first_counts[(unit, str(first))] += 1
        first_second_counts[(unit, str(first), str(second))] += 1
        selection_reasons.update(item["selection_reasons"])

    uncertainty_defects = Counter()
    answer_reviews = []
    answer_non_ok_details = []
    for row_id, row in sorted(context["rows"].items()):
        review = context["reviews"][row_id]
        unit = row["visible"]["unit"]
        if unit == "claim":
            uncertainty_defects["claim_uncertain"] += int(
                review["decision"] == "uncertain"
            )
            uncertainty_defects["claim_ambiguity"] += int(review["ambiguity"])
            uncertainty_defects["claim_rubric_evidence_defect"] += int(
                review["rubric_evidence_defect"]
            )
            continue
        uncertainty_defects["answer_uncertain"] += int(
            review["decision"] == "uncertain"
        )
        uncertainty_defects["answer_incorrect"] += int(
            review["decision"] == "answer_incorrect"
        )
        uncertainty_defects["answer_corpus_evidence_issue"] += int(
            review["corpus_evidence_issue"]
        )
        concise = {
            "row_id": row_id,
            **_row_identity(row),
            "candidate_status": row["visible"]["candidate_status"],
            "first_review": _review_without_identity(review),
            "second_review": (
                _review_without_identity(context["second_reviews"][row_id])
                if row_id in context["second_reviews"] else None
            ),
        }
        answer_reviews.append(concise)
        if review["decision"] != "answer_ok" or review["corpus_evidence_issue"]:
            visible = row["visible"]
            answer_non_ok_details.append({
                **concise,
                "question": visible["question"],
                "candidate_answer": visible["candidate_answer"],
                "gold_answer": visible["gold_answer"],
                "claim_rubric": visible["claim_rubric"],
                "evidence": visible["evidence"],
                "qwen_claims": row["qwen_claims"],
            })

    disagreement_table = []
    other_followups = []
    for selected in context["selected"]:
        if selected["unit"] != "claim":
            continue
        entry = _claim_followup_entry(context, selected)
        qwen = entry["qwen_label"]
        first = entry["first_review"]["decision"]
        second = entry["second_review"]["decision"]
        if (
            (type(first) is int and first != qwen)
            or (type(second) is int and second != qwen)
        ):
            disagreement_table.append(entry)
        else:
            other_followups.append(entry)

    return {
        "post_review_summary_schema_version": POST_REVIEW_SUMMARY_SCHEMA_VERSION,
        "screen_id": SCREEN_ID,
        "screen_artifacts": SCREEN_HASHES,
        "diagnostic_only": True,
        "claim_ready": False,
        "grade_policy": {
            "qwen_grades_changed": False,
            "qwen_grades_retried": False,
            "reviewer_labels_substituted_for_qwen": False,
            "raw_qwen_scores_kept_separate": True,
        },
        "first_pass_validation": {
            "path": _display_path(context["validation_path"]),
            "sha256": sha256_bytes(context["validation_raw"]),
        },
        "second_pass_validation": {
            "path": _display_path(context["second_validation_path"]),
            "sha256": sha256_bytes(context["second_validation_raw"]),
        },
        "first_pass_claim_audit": {
            "overall": _claim_audit_metrics(context, arm=None),
            "by_arm": {
                arm: _claim_audit_metrics(context, arm=arm)
                for arm in ("base", "cheatsheet")
            },
        },
        "first_pass_uncertainty_and_defects": {
            key: uncertainty_defects[key]
            for key in (
                "answer_incorrect", "answer_uncertain",
                "answer_corpus_evidence_issue", "claim_uncertain",
                "claim_ambiguity", "claim_rubric_evidence_defect",
            )
        },
        "second_pass_followup": {
            "selected_rows": len(context["selected"]),
            "selected_answer_rows": sum(
                item["unit"] == "answer" for item in context["selected"]
            ),
            "selected_claim_rows": sum(
                item["unit"] == "claim" for item in context["selected"]
            ),
            "selection_reason_counts": dict(sorted(selection_reasons.items())),
            "first_decision_counts": {
                f"{unit}:{decision}": count
                for (unit, decision), count in sorted(first_counts.items())
            },
            "first_second_decision_counts": {
                f"{unit}:first_{first}:second_{second}": count
                for (unit, first, second), count
                in sorted(first_second_counts.items())
            },
        },
        "selected_subset_sensitivity": {
            "scope_warning": (
                "These metrics are conditional on first-pass selection and must "
                "not be extrapolated to all 619 claim rows."
            ),
            "overall": _selected_claim_sensitivity(context, arm=None),
            "by_arm": {
                arm: _selected_claim_sensitivity(context, arm=arm)
                for arm in ("base", "cheatsheet")
            },
        },
        "answer_reviews": answer_reviews,
        "answer_non_ok_details": answer_non_ok_details,
        "claim_disagreements": disagreement_table,
        "claim_other_followups": other_followups,
    }


def write_post_review_summary(
    *, second_pass_validation_path: Path, output_path: Path,
    source_attestor: Callable[[], dict[str, Any]] = _builder_source_attestation,
) -> Path:
    """Write the separate post-review reveal; never modify a raw Qwen grade."""

    context = _load_frozen_second_pass(second_pass_validation_path)
    expected_output = context["audit_root"] / "post-review-summary.json"
    if output_path.resolve() != expected_output:
        raise ManualAuditError("post-review summary path is not canonical")
    source = source_attestor()
    _validate_builder_source_attestation(source)
    if canonical_json_bytes(source) != canonical_json_bytes(
        context["second_manifest"]["packet_builder_source"]
    ):
        raise ManualAuditError("summary source differs from pre-open builder source")
    result = build_post_review_summary(context)
    _load_screen()
    final_source = source_attestor()
    _validate_builder_source_attestation(final_source)
    if canonical_json_bytes(final_source) != canonical_json_bytes(source):
        raise ManualAuditError("summary source changed during reveal construction")
    write_immutable_json(output_path, result)
    raw = read_artifact_bytes(output_path)
    if raw != canonical_json_bytes(result):
        raise ManualAuditError("durable post-review summary failed verification")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--audit-id", required=True)
    build.add_argument("--reviewer", action="append", required=True)
    build.add_argument("--output-root", default="manual-audits")
    validate = subparsers.add_parser("validate-first-pass")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--review", action="append", required=True)
    validate.add_argument("--output", required=True)
    build_second = subparsers.add_parser("build-second-pass")
    build_second.add_argument("--validation", required=True)
    build_second.add_argument("--reviewer", action="append", required=True)
    validate_second = subparsers.add_parser("validate-second-pass")
    validate_second.add_argument("--manifest", required=True)
    validate_second.add_argument("--review", action="append", required=True)
    validate_second.add_argument("--output", required=True)
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--validation", required=True)
    summarize.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        if args.command == "build":
            path = build_packets(
                audit_id=args.audit_id,
                reviewers=args.reviewer,
                output_root=ROOT / args.output_root,
            )
            print(_display_path(path))
        elif args.command == "validate-first-pass":
            path = write_first_pass_validation(
                manifest_path=ROOT / args.manifest,
                review_paths=[ROOT / value for value in args.review],
                output_path=ROOT / args.output,
            )
            raw = read_artifact_bytes(path)
            print(f"{_display_path(path)} sha256={sha256_bytes(raw)}")
        elif args.command == "build-second-pass":
            path = build_second_pass_packets(
                first_pass_validation_path=ROOT / args.validation,
                reviewers=args.reviewer,
            )
            print(_display_path(path))
        elif args.command == "validate-second-pass":
            path = write_second_pass_validation(
                manifest_path=ROOT / args.manifest,
                review_paths=[ROOT / value for value in args.review],
                output_path=ROOT / args.output,
            )
            raw = read_artifact_bytes(path)
            print(f"{_display_path(path)} sha256={sha256_bytes(raw)}")
        else:
            path = write_post_review_summary(
                second_pass_validation_path=ROOT / args.validation,
                output_path=ROOT / args.output,
            )
            raw = read_artifact_bytes(path)
            print(f"{_display_path(path)} sha256={sha256_bytes(raw)}")
    except (OSError, TypeError, ValueError, ManualAuditError) as exc:
        raise SystemExit(f"MANUAL AUDIT ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
