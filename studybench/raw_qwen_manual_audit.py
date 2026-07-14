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
PACKET_SCHEMA_VERSION = 1
GRADE_VIEW_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
REVIEW_SCHEMA_VERSION = 1
ACCESS_TOOLS = [
    "filesystem reads limited to the assigned packet and cited corpus",
    "apply_patch for the assigned review output",
    "local JSON parse/coverage validation of packet and assigned output",
]
FORBIDDEN_CONTEXT = [
    "arm", "budget", "rollout", "run path", "grade path",
    "server slot", "Qwen label", "weighted total", "aggregate result",
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
    args = parser.parse_args()
    try:
        if args.command == "build":
            path = build_packets(
                audit_id=args.audit_id,
                reviewers=args.reviewer,
                output_root=ROOT / args.output_root,
            )
            print(_display_path(path))
        else:
            path = write_first_pass_validation(
                manifest_path=ROOT / args.manifest,
                review_paths=[ROOT / value for value in args.review],
                output_path=ROOT / args.output,
            )
            raw = read_artifact_bytes(path)
            print(f"{_display_path(path)} sha256={sha256_bytes(raw)}")
    except (OSError, TypeError, ValueError, ManualAuditError) as exc:
        raise SystemExit(f"MANUAL AUDIT ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
