from copy import deepcopy
import json
import unittest

from studybench.human_audit import (
    HUMAN_AUDIT_DECISION_RULE_ID,
    HumanAuditError,
    derive_human_audit_population,
    validate_human_audit_protocol,
    validate_human_audit_result,
)


def _jsonl(*records: dict) -> bytes:
    return b"".join(
        json.dumps(record, sort_keys=True).encode("utf-8") + b"\n"
        for record in records
    )


def _train(item_id: str, round_number: int, entry: dict | None = None) -> dict:
    return {
        "schema_version": 2,
        "item_id": item_id,
        "origin_item_id": item_id,
        "origin_round": round_number,
        "round": round_number,
        "kind": "quiz",
        "split": "train",
        "retest_of": None,
        "entry": entry,
    }


def _fixture() -> tuple[dict, dict[str, bytes], dict]:
    entry_a = {"entry_id": "entry-a", "correction": "A"}
    entry_b = {"entry_id": "entry-b", "correction": "B"}
    artifacts = {
        "r1/items.jsonl": _jsonl(_train("train-1", 1, entry_b)),
        "r1/dev-exam.jsonl": _jsonl({"item_id": "dev-1"}),
        "r2/items.jsonl": _jsonl(_train("train-2", 2, entry_a)),
        "r2/dev-exam.jsonl": _jsonl({"item_id": "dev-2"}),
        "notes/note-r2.md": b"note\n",
    }
    construction = {
        "schema_version": 2,
        "round": 2,
        "entry_ids": ["entry-a", "entry-b"],
        "entries": [entry_a, entry_b],
    }
    audit = {
        "blinding_preserved": True,
        "reviewer_independent": True,
        "decision": "pass",
        "record_reviews": [
            {
                "record_id": record_id,
                "verdict_valid": True,
                "evidence_valid": True,
                "leakage_free": True,
            }
            for record_id in ("train-1", "train-2", "dev-1", "dev-2")
        ],
        "entry_reviews": [
            {
                "entry_id": entry_id,
                "correction_supported": True,
                "citation_exact": True,
                "leakage_free": True,
            }
            for entry_id in ("entry-a", "entry-b")
        ],
    }
    return construction, artifacts, audit


class HumanAuditValidationTests(unittest.TestCase):
    def test_decision_rule_identifier_is_canonical_and_versioned(self):
        self.assertEqual(
            HUMAN_AUDIT_DECISION_RULE_ID,
            "all-required-reviews-pass-v1",
        )

    def test_protocol_schema_is_closed_to_unenforced_clauses(self):
        protocol = {
            "schema_version": 1,
            "protocol_id": "blind-audit-01",
            "blinding": "condition_and_method_labels_hidden",
            "population": "all_train_dev_verdicts_and_admitted_entries",
            "decision_rule": HUMAN_AUDIT_DECISION_RULE_ID,
        }
        self.assertEqual(
            validate_human_audit_protocol(protocol),
            "blind-audit-01",
        )
        for field in ("assignment", "escalation", "adjudication"):
            with self.subTest(field=field):
                extended = deepcopy(protocol)
                extended[field] = {"required": True}
                with self.assertRaisesRegex(HumanAuditError, "unsupported fields"):
                    validate_human_audit_protocol(extended)

    def test_population_is_derived_from_every_round_and_item_entries(self):
        construction, artifacts, audit = _fixture()
        population = derive_human_audit_population(construction, artifacts)
        self.assertEqual(
            population.record_ids,
            ("train-1", "train-2", "dev-1", "dev-2"),
        )
        self.assertEqual(population.entry_ids, ("entry-a", "entry-b"))
        validation = validate_human_audit_result(audit, construction, artifacts)
        self.assertTrue(validation.passed)
        self.assertEqual(validation.population, population)

    def test_reviews_must_cover_each_exact_id_once_with_json_booleans(self):
        construction, artifacts, audit = _fixture()
        cases = []
        missing = deepcopy(audit)
        missing["record_reviews"].pop()
        cases.append(missing)
        duplicate = deepcopy(audit)
        duplicate["entry_reviews"][1]["entry_id"] = "entry-a"
        cases.append(duplicate)
        integer_boolean = deepcopy(audit)
        integer_boolean["record_reviews"][0]["verdict_valid"] = 1
        cases.append(integer_boolean)
        for candidate in cases:
            with self.subTest(candidate=candidate), self.assertRaises(HumanAuditError):
                validate_human_audit_result(candidate, construction, artifacts)

    def test_decision_is_a_pure_function_of_every_review_and_declaration(self):
        construction, artifacts, audit = _fixture()
        failed = deepcopy(audit)
        failed["entry_reviews"][0]["citation_exact"] = False
        with self.assertRaisesRegex(HumanAuditError, "decision must be 'fail'"):
            validate_human_audit_result(failed, construction, artifacts)
        failed["decision"] = "fail"
        self.assertFalse(
            validate_human_audit_result(failed, construction, artifacts).passed)

    def test_construction_summary_cannot_invent_entries_or_omit_population(self):
        construction, artifacts, audit = _fixture()
        invented = deepcopy(construction)
        invented["entry_ids"].append("invented")
        with self.assertRaisesRegex(HumanAuditError, "entry lineage"):
            validate_human_audit_result(audit, invented, artifacts)

        incomplete = dict(artifacts)
        del incomplete["r1/dev-exam.jsonl"]
        with self.assertRaisesRegex(HumanAuditError, "omit"):
            derive_human_audit_population(construction, incomplete)

        ambiguous = dict(artifacts)
        ambiguous["r1/items.jsonl"] = b'{"item_id":"a","item_id":"b"}\n'
        with self.assertRaisesRegex(HumanAuditError, "duplicate JSON key"):
            derive_human_audit_population(construction, ambiguous)

    def test_population_rejects_nonfinite_json_and_nontraining_entries(self):
        construction, artifacts, _audit = _fixture()

        nonfinite = dict(artifacts)
        nonfinite["r1/dev-exam.jsonl"] = b'{"item_id":"dev-1","value":NaN}\n'
        with self.assertRaisesRegex(HumanAuditError, "non-finite"):
            derive_human_audit_population(construction, nonfinite)

        dev_entry = dict(artifacts)
        dev_entry["r1/dev-exam.jsonl"] = _jsonl({
            "item_id": "dev-1",
            "entry": {"entry_id": "hidden-dev-entry"},
        })
        with self.assertRaisesRegex(HumanAuditError, "dev record"):
            derive_human_audit_population(construction, dev_entry)

        retest_entry = dict(artifacts)
        retest = _train(
            "retest-1", 1, {"entry_id": "hidden-retest-entry"}
        )
        retest["origin_item_id"] = "train-0"
        retest["retest_of"] = "train-0"
        retest_entry["r1/items.jsonl"] = _jsonl(retest)
        with self.assertRaisesRegex(HumanAuditError, "original training item"):
            derive_human_audit_population(construction, retest_entry)


if __name__ == "__main__":
    unittest.main()
