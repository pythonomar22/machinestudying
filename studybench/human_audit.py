"""Pure validation for blinded self-study human-audit protocols and results.

This module deliberately depends only on the Python standard library so the
same validation can run during study promotion, provenance snapshotting, and
offline grading.  Callers are responsible for verifying artifact hashes before
passing the exact construction-artifact bytes here.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import PurePosixPath
import re


HUMAN_AUDIT_SCHEMA_VERSION = 1
HUMAN_AUDIT_DECISION_RULE_ID = "all-required-reviews-pass-v1"
HUMAN_AUDIT_BLINDING_ID = "condition_and_method_labels_hidden"
HUMAN_AUDIT_POPULATION_ID = "all_train_dev_verdicts_and_admitted_entries"
HUMAN_AUDIT_PROTOCOL_FIELDS = frozenset({
    "schema_version",
    "protocol_id",
    "blinding",
    "population",
    "decision_rule",
})
_AUDIT_ID = re.compile(r"[a-z0-9][a-z0-9._-]{2,79}\Z")
_UNSPECIFIED_PROTOCOL_ID = object()


class HumanAuditError(ValueError):
    """A human-audit protocol, construction population, or result is not exact."""


@dataclass(frozen=True)
class HumanAuditPopulation:
    """The exact cumulative records and admitted entries requiring review."""

    record_ids: tuple[str, ...]
    entry_ids: tuple[str, ...]


@dataclass(frozen=True)
class HumanAuditValidation:
    """A validated audit and the population from which its decision follows."""

    population: HumanAuditPopulation
    passed: bool


def validate_human_audit_protocol(
    protocol: object,
    *,
    expected_protocol_id: object = _UNSPECIFIED_PROTOCOL_ID,
) -> str:
    """Validate the complete protocol schema and return its protocol ID.

    The schema is deliberately closed.  Accepting descriptive extensions would
    let a pre-registration appear to require procedures that promotion and
    downstream grading never enforce.
    """

    if not isinstance(protocol, dict):
        raise HumanAuditError("human-audit protocol is not a JSON object")
    fields = set(protocol)
    if fields != HUMAN_AUDIT_PROTOCOL_FIELDS:
        missing = sorted(HUMAN_AUDIT_PROTOCOL_FIELDS - fields)
        extra = sorted(fields - HUMAN_AUDIT_PROTOCOL_FIELDS, key=repr)
        details = []
        if missing:
            details.append(f"missing fields {missing!r}")
        if extra:
            details.append(f"unsupported fields {extra!r}")
        raise HumanAuditError(
            "human-audit protocol must use the exact closed schema: "
            + "; ".join(details)
        )
    if (type(protocol["schema_version"]) is not int
            or protocol["schema_version"] != HUMAN_AUDIT_SCHEMA_VERSION):
        raise HumanAuditError(
            f"human-audit protocol must declare schema_version="
            f"{HUMAN_AUDIT_SCHEMA_VERSION}"
        )
    required = {
        "blinding": HUMAN_AUDIT_BLINDING_ID,
        "population": HUMAN_AUDIT_POPULATION_ID,
        "decision_rule": HUMAN_AUDIT_DECISION_RULE_ID,
    }
    for field, expected in required.items():
        if protocol[field] != expected:
            raise HumanAuditError(
                f"human-audit protocol must declare {field}={expected!r}"
            )
    protocol_id = protocol["protocol_id"]
    if not isinstance(protocol_id, str) or not _AUDIT_ID.fullmatch(protocol_id):
        raise HumanAuditError(
            "audit protocol ID must be 3-80 lowercase letters, digits, '.', '_' or '-'"
        )
    if (expected_protocol_id is not _UNSPECIFIED_PROTOCOL_ID
            and protocol_id != expected_protocol_id):
        raise HumanAuditError(
            "human-audit protocol ID does not match its immutable manifest record"
        )
    return protocol_id


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise HumanAuditError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise HumanAuditError(f"non-finite JSON number: {value}")


def _jsonl_records(data: bytes, *, label: str) -> list[dict]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise HumanAuditError(f"{label} is not valid UTF-8") from error

    records = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(
                line,
                object_pairs_hook=_strict_object,
                parse_constant=_reject_nonfinite,
            )
        except (json.JSONDecodeError, HumanAuditError) as error:
            raise HumanAuditError(
                f"{label}:{line_number} is not strict JSON: {error}"
            ) from error
        if not isinstance(record, dict):
            raise HumanAuditError(f"{label}:{line_number} is not a JSON object")
        records.append(record)
    if not records:
        raise HumanAuditError(f"{label} contains no reviewable records")
    return records


def _canonical_artifacts(artifacts: Mapping[str, bytes]) -> dict[str, bytes]:
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise HumanAuditError("construction artifacts are missing")
    canonical = {}
    for raw_path, data in artifacts.items():
        if not isinstance(raw_path, str) or not raw_path or "\\" in raw_path:
            raise HumanAuditError("construction artifacts contain an unsafe path")
        path = PurePosixPath(raw_path)
        if (path.is_absolute() or any(part in ("", ".", "..") for part in path.parts)
                or str(path) != raw_path):
            raise HumanAuditError("construction artifacts contain an unsafe path")
        if not isinstance(data, bytes):
            raise HumanAuditError(
                f"construction artifact {raw_path!r} was not supplied as exact bytes"
            )
        canonical[raw_path] = data
    return canonical


def _strict_positive_integer(value: object) -> bool:
    return type(value) is int and value >= 1


def _record_id(record: dict, *, label: str) -> str:
    value = record.get("item_id")
    if not isinstance(value, str) or not value or value != value.strip():
        raise HumanAuditError(f"{label} has an invalid item_id")
    return value


def _is_original_train_item(record: dict, *, schema_version: int) -> bool:
    """Mirror the construction rule for records allowed to admit note entries."""

    return (
        type(record.get("schema_version")) is int
        and record["schema_version"] == schema_version
        and isinstance(record.get("item_id"), str)
        and bool(record["item_id"])
        and record.get("origin_item_id") == record["item_id"]
        and _strict_positive_integer(record.get("round"))
        and record.get("origin_round") == record["round"]
        and record.get("kind") == "quiz"
        and record.get("split") == "train"
        and record.get("retest_of") is None
    )


def derive_human_audit_population(
    construction: object,
    artifacts: Mapping[str, bytes],
) -> HumanAuditPopulation:
    """Derive the exact cumulative audit population from construction bytes.

    ``artifacts`` maps the construction manifest's study-relative paths to the
    already hash-verified bytes at those paths.  The current construction
    manifest is checked against entries reconstructed from original training
    records; its summary lists are never trusted as the source of population.
    """

    if not isinstance(construction, dict):
        raise HumanAuditError("construction manifest is not a JSON object")
    round_number = construction.get("round")
    if not _strict_positive_integer(round_number):
        raise HumanAuditError("construction manifest has no valid round")
    schema_version = construction.get("schema_version")
    if not _strict_positive_integer(schema_version):
        raise HumanAuditError("construction manifest has no valid schema version")
    artifact_bytes = _canonical_artifacts(artifacts)

    train_records: list[dict] = []
    dev_records: list[dict] = []
    for current_round in range(1, round_number + 1):
        train_path = f"r{current_round}/items.jsonl"
        dev_path = f"r{current_round}/dev-exam.jsonl"
        if train_path not in artifact_bytes or dev_path not in artifact_bytes:
            raise HumanAuditError(
                f"construction artifacts omit the round-{current_round} audit population"
            )
        train_records.extend(_jsonl_records(
            artifact_bytes[train_path], label=train_path))
        dev_records.extend(_jsonl_records(
            artifact_bytes[dev_path], label=dev_path))

    record_ids = tuple(
        _record_id(record, label="train/dev record")
        for record in train_records + dev_records
    )
    if len(record_ids) != len(set(record_ids)):
        raise HumanAuditError("construction train/dev record IDs are not unique")

    entries_by_id: dict[str, dict] = {}
    for record in dev_records:
        if record.get("entry") is not None:
            raise HumanAuditError("a dev record cannot admit a note entry")

    for record in train_records:
        entry = record.get("entry")
        if entry is None:
            continue
        if not _is_original_train_item(record, schema_version=schema_version):
            raise HumanAuditError(
                "only an original training item can admit a note entry"
            )
        if not isinstance(entry, dict):
            raise HumanAuditError("an admitted note entry is not a JSON object")
        entry_id = entry.get("entry_id")
        if (not isinstance(entry_id, str) or not entry_id
                or entry_id != entry_id.strip()):
            raise HumanAuditError("an admitted note entry has an invalid entry_id")
        if entry_id in entries_by_id and entries_by_id[entry_id] != entry:
            raise HumanAuditError(f"conflicting admitted note entry: {entry_id}")
        entries_by_id[entry_id] = entry

    expected_entries = [entries_by_id[key] for key in sorted(entries_by_id)]
    entry_ids = tuple(entry["entry_id"] for entry in expected_entries)
    if (construction.get("entries") != expected_entries
            or construction.get("entry_ids") != list(entry_ids)):
        raise HumanAuditError(
            "construction manifest entry lineage differs from its item artifacts"
        )
    return HumanAuditPopulation(record_ids=record_ids, entry_ids=entry_ids)


def _validate_reviews(
    audit: dict,
    *,
    field: str,
    id_field: str,
    expected_ids: tuple[str, ...],
    boolean_fields: tuple[str, ...],
    label: str,
) -> bool:
    reviews = audit.get(field)
    if not isinstance(reviews, list) or not all(isinstance(review, dict)
                                                for review in reviews):
        raise HumanAuditError(f"human audit has invalid {field}")
    reviewed_ids = [review.get(id_field) for review in reviews]
    if (not all(isinstance(value, str) and bool(value) for value in reviewed_ids)
            or len(reviewed_ids) != len(set(reviewed_ids))
            or sorted(reviewed_ids) != sorted(expected_ids)):
        raise HumanAuditError(f"human audit does not cover every {label} exactly once")
    if any(type(review.get(field_name)) is not bool
           for review in reviews for field_name in boolean_fields):
        raise HumanAuditError(f"human audit {label} decisions must be JSON booleans")
    return all(review[field_name]
               for review in reviews for field_name in boolean_fields)


def validate_human_audit_result(
    audit: object,
    construction: object,
    artifacts: Mapping[str, bytes],
) -> HumanAuditValidation:
    """Validate complete review coverage and the pre-registered decision rule.

    Artifact/protocol hashes and study identifiers are binding concerns checked
    by callers.  This function owns the population and decision semantics so
    promotion, provenance, and grading cannot disagree about them.
    """

    if not isinstance(audit, dict):
        raise HumanAuditError("human audit is not a JSON object")
    population = derive_human_audit_population(construction, artifacts)
    records_pass = _validate_reviews(
        audit,
        field="record_reviews",
        id_field="record_id",
        expected_ids=population.record_ids,
        boolean_fields=("verdict_valid", "evidence_valid", "leakage_free"),
        label="cumulative train/dev record",
    )
    entries_pass = _validate_reviews(
        audit,
        field="entry_reviews",
        id_field="entry_id",
        expected_ids=population.entry_ids,
        boolean_fields=("correction_supported", "citation_exact", "leakage_free"),
        label="admitted note entry",
    )
    if (type(audit.get("blinding_preserved")) is not bool
            or type(audit.get("reviewer_independent")) is not bool):
        raise HumanAuditError("human audit process declarations must be JSON booleans")
    passed = (
        audit["blinding_preserved"]
        and audit["reviewer_independent"]
        and records_pass
        and entries_pass
    )
    expected_decision = "pass" if passed else "fail"
    if audit.get("decision") != expected_decision:
        raise HumanAuditError(
            f"human audit decision must be {expected_decision!r} for its declared reviews"
        )
    return HumanAuditValidation(population=population, passed=passed)
