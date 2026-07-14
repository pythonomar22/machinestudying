from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from studybench import raw_qwen_manual_audit as manual
from studybench.integrity import (
    atomic_write_json,
    canonical_json_bytes,
    read_artifact_bytes,
    sha256_bytes,
    sha256_json,
    strict_json_loads,
    write_immutable_json,
)


REVIEWERS = [
    "/root/raw_first_pass_a",
    "/root/raw_first_pass_b",
    "/root/raw_first_pass_c",
]


class IdentityTests(unittest.TestCase):
    def test_row_identity_uses_the_frozen_literal_object(self) -> None:
        binding = "a" * 64
        expected = sha256_json({
            "audit_schema": 1,
            "cell_binding_sha256": binding,
            "unit": "claim",
            "claim_id": "c7",
        })
        self.assertEqual(manual._row_id(binding, "claim", "c7"), expected)
        self.assertEqual(
            manual._order("46001:", expected),
            sha256_bytes(f"46001:{expected}".encode("utf-8")),
        )

    def test_reviewer_names_are_exact_and_distinct(self) -> None:
        self.assertEqual(manual._validate_reviewers(REVIEWERS), REVIEWERS)
        for reviewers in (
            REVIEWERS[:2],
            [REVIEWERS[0], REVIEWERS[0], REVIEWERS[2]],
            ["reviewer-a", REVIEWERS[1], REVIEWERS[2]],
            ["/root/raw_first_pass_a/escape", REVIEWERS[1], REVIEWERS[2]],
        ):
            with self.subTest(reviewers=reviewers), self.assertRaises(
                manual.ManualAuditError
            ):
                manual._validate_reviewers(reviewers)

    def test_forged_source_attestation_is_rejected(self) -> None:
        with self.assertRaises(manual.ManualAuditError):
            manual._validate_builder_source_attestation({
                "clean_pushed_source": {
                    "policy": "clean-head-contained-in-remote-tracking-ref-v1",
                    "source": {"dirty": False, "files": {}},
                    "source_sha256": "0" * 64,
                    "remote_tracking_refs": ["refs/remotes/origin/main"],
                },
                "module": {
                    "path": "studybench/raw_qwen_manual_audit.py",
                    "sha256": "f" * 64,
                    "bytes": 1,
                },
            })

    def test_second_pass_selection_does_not_treat_incorrect_answer_as_qwen_disagreement(
        self,
    ) -> None:
        rows = {
            "a" * 64: {
                "visible": {"unit": "answer", "bundle_id": "1" * 64},
                "qwen_label": None,
            },
            "b" * 64: {
                "visible": {"unit": "answer", "bundle_id": "2" * 64},
                "qwen_label": None,
            },
        }
        reviews = {
            "a" * 64: {
                "decision": "answer_incorrect", "corpus_evidence_issue": False,
            },
            "b" * 64: {
                "decision": "uncertain", "corpus_evidence_issue": False,
            },
        }
        selected = manual._second_pass_selection({"rows": rows, "reviews": reviews})
        self.assertEqual([item["row_id"] for item in selected], ["b" * 64])
        self.assertEqual(selected[0]["selection_reasons"], ["answer_uncertain"])

    def test_second_pass_reviewer_names_are_fresh_and_exact(self) -> None:
        reviewers = [
            "/root/raw_second_pass_a",
            "/root/raw_second_pass_b",
            "/root/raw_second_pass_c",
        ]
        self.assertEqual(
            manual._validate_second_reviewers(
                reviewers, first_reviewers=set(REVIEWERS)
            ),
            reviewers,
        )
        for invalid in (
            reviewers[:2],
            [reviewers[0], reviewers[0], reviewers[2]],
            [REVIEWERS[0], reviewers[1], reviewers[2]],
            ["/root/second_pass_a", reviewers[1], reviewers[2]],
        ):
            with self.subTest(invalid=invalid), self.assertRaises(
                manual.ManualAuditError
            ):
                manual._validate_second_reviewers(
                    invalid, first_reviewers=set(REVIEWERS)
                )

    def test_frozen_second_pass_census_is_exact(self) -> None:
        selected = [
            {"unit": "claim"} for _ in range(71)
        ]
        manual._require_frozen_selection_census(selected)
        for wrong in (selected[:-1], selected + [{"unit": "answer"}]):
            with self.assertRaisesRegex(manual.ManualAuditError, "71-claim"):
                manual._require_frozen_selection_census(wrong)


class FrozenPacketTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source_validator = patch.object(
            manual.provenance,
            "validate_frozen_source_commit",
            side_effect=lambda source: source,
        )
        self.source_validator.start()

    def tearDown(self) -> None:
        self.source_validator.stop()

    def _source_attestation(self) -> dict:
        source = {
            "dirty": False,
            "files": {
                "studybench/raw_qwen_manual_audit.py": {
                    "sha256": "f" * 64,
                    "bytes": 1,
                }
            },
        }
        return {
            "clean_pushed_source": {
                "policy": "clean-head-contained-in-remote-tracking-ref-v1",
                "source": source,
                "source_sha256": sha256_json(source),
                "remote_tracking_refs": ["refs/remotes/origin/main"],
            },
            "module": {
                "path": "studybench/raw_qwen_manual_audit.py",
                "sha256": "f" * 64,
                "bytes": 1,
            },
        }

    def _build(self, root: Path) -> tuple[Path, dict]:
        manifest_path = manual.build_packets(
            audit_id="test-raw-manual-audit",
            reviewers=REVIEWERS,
            output_root=root,
            source_attestor=self._source_attestation,
        )
        raw = read_artifact_bytes(manifest_path)
        manifest = strict_json_loads(raw, label="test manifest")
        self.assertEqual(raw, canonical_json_bytes(manifest))
        return manifest_path, manifest

    def test_full_census_is_blinded_and_bound_before_open(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, manifest = self._build(root)
            self.assertEqual(
                manifest["row_counts"], {"answer": 120, "claim": 619, "total": 739}
            )
            self.assertEqual(
                [record["answer_rows"] for record in manifest["reviewers"]],
                [40, 40, 40],
            )
            self.assertEqual(
                sum(record["claim_rows"] for record in manifest["reviewers"]), 619
            )
            self.assertEqual(len(list(root.rglob("operator-bindings/cell-*.json"))), 120)
            row_ids = []
            forbidden = {
                "arm", "budget", "rollout", "episode_path", "grade_path",
                "server_slot", "claims", "lenient", "cores_ok", "qwen_label",
            }
            for record in manifest["reviewers"]:
                packet_path = manual.ROOT / record["path"]
                # Test output roots may be outside the repository; use the
                # basename beneath the temporary audit root in that case.
                if not packet_path.exists():
                    packet_path = next(root.rglob(Path(record["path"]).name))
                packet_raw = read_artifact_bytes(packet_path)
                self.assertEqual(sha256_bytes(packet_raw), record["sha256"])
                packet = strict_json_loads(packet_raw, label="test packet")
                self.assertEqual(packet["reviewer"], record["reviewer"])
                for row in packet["rows"]:
                    self.assertFalse(forbidden.intersection(row))
                    row_ids.append(row["row_id"])
            self.assertEqual(len(row_ids), 739)
            self.assertEqual(len(set(row_ids)), 739)

    def test_manifest_outcome_leak_is_rejected_before_reviews(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, manifest = self._build(root)
            manifest["qwen_totals"] = {"base": 1, "treatment": 2}
            atomic_write_json(manifest_path, manifest)
            with self.assertRaisesRegex(manual.ManualAuditError, "pre-open"):
                manual.validate_first_pass(
                    manifest_path=manifest_path,
                    review_paths=[root / f"missing-{index}.json" for index in range(3)],
                )

    def test_first_pass_validator_requires_exact_739_row_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, manifest = self._build(root)
            review_paths = []
            for index, record in enumerate(manifest["reviewers"]):
                packet_path = next(root.rglob(Path(record["path"]).name))
                packet = strict_json_loads(
                    read_artifact_bytes(packet_path), label="test packet"
                )
                reviews = []
                for row in packet["rows"]:
                    if row["unit"] == "answer":
                        reviews.append({
                            "row_id": row["row_id"],
                            "unit": "answer",
                            "decision": "uncertain",
                            "corpus_evidence_issue": False,
                            "note": "Synthetic validator fixture; no semantic decision.",
                        })
                    else:
                        reviews.append({
                            "row_id": row["row_id"],
                            "unit": "claim",
                            "decision": "uncertain",
                            "confidence": "low",
                            "ambiguity": True,
                            "rubric_evidence_defect": False,
                            "note": "Synthetic validator fixture; no semantic decision.",
                        })
                review = {
                    "review_schema_version": 1,
                    "reviewer": record["reviewer"],
                    "packet_sha256": record["sha256"],
                    "review_prompt_sha256": packet["review_prompt_sha256"],
                    "reviews": reviews,
                }
                path = Path(record["review_output_path"])
                if not path.is_absolute():
                    path = manual.ROOT / path
                write_immutable_json(path, review)
                review_paths.append(path)
            validated = manual.validate_first_pass(
                manifest_path=manifest_path, review_paths=review_paths
            )
            self.assertTrue(validated["complete"])
            self.assertEqual(validated["row_count"], 739)
            validation_path = manifest_path.parent / "first-pass-validation.json"
            self.assertEqual(
                manual.write_first_pass_validation(
                    manifest_path=manifest_path,
                    review_paths=review_paths,
                    output_path=validation_path,
                ),
                validation_path,
            )
            frozen_validation = strict_json_loads(
                read_artifact_bytes(validation_path), label="frozen validation"
            )
            self.assertTrue(frozen_validation["complete"])
            self.assertEqual(len(frozen_validation["reviews"]), 3)
            with self.assertRaisesRegex(manual.ManualAuditError, "not canonical"):
                manual.write_first_pass_validation(
                    manifest_path=manifest_path,
                    review_paths=review_paths,
                    output_path=root / "wrong-validation.json",
                )

            original = json.loads(read_artifact_bytes(review_paths[0]))
            broken = json.loads(read_artifact_bytes(review_paths[0]))
            broken["reviews"] = broken["reviews"][:-1]
            atomic_write_json(review_paths[0], broken)
            with self.assertRaisesRegex(manual.ManualAuditError, "row count"):
                manual.validate_first_pass(
                    manifest_path=manifest_path,
                    review_paths=review_paths,
                )

            atomic_write_json(review_paths[0], original)
            boolean = json.loads(read_artifact_bytes(review_paths[0]))
            claim = next(item for item in boolean["reviews"] if item["unit"] == "claim")
            claim["decision"] = True
            atomic_write_json(review_paths[0], boolean)
            with self.assertRaisesRegex(manual.ManualAuditError, "claim review"):
                manual.validate_first_pass(
                    manifest_path=manifest_path,
                    review_paths=review_paths,
                )

            # Even if a malicious operator rebinds all visible hashes, a
            # Qwen-label field in a blinded packet is outside the exact schema.
            atomic_write_json(review_paths[0], original)
            packet_path = Path(manifest["reviewers"][0]["path"])
            if not packet_path.is_absolute():
                packet_path = manual.ROOT / packet_path
            leaked_packet = json.loads(read_artifact_bytes(packet_path))
            leaked_packet["rows"][0]["qwen_label"] = 1
            atomic_write_json(packet_path, leaked_packet)
            leaked_raw = read_artifact_bytes(packet_path)
            leaked_manifest = json.loads(read_artifact_bytes(manifest_path))
            leaked_manifest["reviewers"][0]["sha256"] = sha256_bytes(leaked_raw)
            leaked_manifest["reviewers"][0]["bytes"] = len(leaked_raw)
            atomic_write_json(manifest_path, leaked_manifest)
            leaked_review = json.loads(read_artifact_bytes(review_paths[0]))
            leaked_review["packet_sha256"] = sha256_bytes(leaked_raw)
            atomic_write_json(review_paths[0], leaked_review)
            with self.assertRaisesRegex(manual.ManualAuditError, "row schema"):
                manual.validate_first_pass(
                    manifest_path=manifest_path,
                    review_paths=review_paths,
                )

    def test_second_pass_is_exact_blinded_and_fail_closed_through_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, manifest = self._build(root)
            first_review_paths = []
            answer_index = 0
            claim_index = 0
            for record in manifest["reviewers"]:
                packet_path = Path(record["path"])
                packet = strict_json_loads(
                    read_artifact_bytes(packet_path), label="test first packet"
                )
                reviews = []
                for row in packet["rows"]:
                    if row["unit"] == "answer":
                        decision = "answer_ok"
                        evidence_issue = False
                        if answer_index == 0:
                            decision = "answer_incorrect"
                        elif answer_index == 1:
                            decision = "uncertain"
                        elif answer_index == 2:
                            evidence_issue = True
                        answer_index += 1
                        reviews.append({
                            "row_id": row["row_id"],
                            "unit": "answer",
                            "decision": decision,
                            "corpus_evidence_issue": evidence_issue,
                            "note": "Synthetic answer decision for audit plumbing.",
                        })
                    else:
                        decision: int | str = 0
                        defect = False
                        if claim_index == 0:
                            decision = "uncertain"
                        elif claim_index == 1:
                            defect = True
                        claim_index += 1
                        reviews.append({
                            "row_id": row["row_id"],
                            "unit": "claim",
                            "decision": decision,
                            "confidence": "low" if decision == "uncertain" else "high",
                            "ambiguity": decision == "uncertain",
                            "rubric_evidence_defect": defect,
                            "note": "Synthetic claim decision for audit plumbing.",
                        })
                review = {
                    "review_schema_version": 1,
                    "reviewer": record["reviewer"],
                    "packet_sha256": record["sha256"],
                    "review_prompt_sha256": packet["review_prompt_sha256"],
                    "reviews": reviews,
                }
                path = Path(record["review_output_path"])
                write_immutable_json(path, review)
                first_review_paths.append(path)
            first_validation_path = manifest_path.parent / "first-pass-validation.json"
            manual.write_first_pass_validation(
                manifest_path=manifest_path,
                review_paths=first_review_paths,
                output_path=first_validation_path,
            )
            manifest_sha = sha256_bytes(read_artifact_bytes(manifest_path))
            validation_sha = sha256_bytes(read_artifact_bytes(first_validation_path))
            second_reviewers = [
                "/root/raw_second_pass_test_a",
                "/root/raw_second_pass_test_b",
                "/root/raw_second_pass_test_c",
            ]
            frozen = (
                patch.object(
                    manual, "FROZEN_MANUAL_AUDIT_ID", manifest_path.parent.name
                ),
                patch.object(manual, "FIRST_PASS_MANIFEST_SHA256", manifest_sha),
                patch.object(
                    manual, "FIRST_PASS_VALIDATION_SHA256", validation_sha
                ),
            )
            for mocked in frozen:
                mocked.start()
            selection_patch = None
            try:
                first_context = manual._load_frozen_first_pass(first_validation_path)
                selected = manual._second_pass_selection(first_context)
                selected_ids = {item["row_id"] for item in selected}
                self.assertEqual(
                    sum(item["unit"] == "answer" for item in selected), 2
                )
                incorrect_answer_id = next(
                    row_id for row_id, review in first_context["reviews"].items()
                    if review["decision"] == "answer_incorrect"
                )
                self.assertNotIn(incorrect_answer_id, selected_ids)
                selection_patch = patch.object(
                    manual,
                    "EXPECTED_SECOND_PASS_SELECTION",
                    {
                        "answer": sum(item["unit"] == "answer" for item in selected),
                        "claim": sum(item["unit"] == "claim" for item in selected),
                        "total": len(selected),
                    },
                )
                selection_patch.start()

                second_manifest_path = manual.build_second_pass_packets(
                    first_pass_validation_path=first_validation_path,
                    reviewers=second_reviewers,
                    source_attestor=self._source_attestation,
                )
                second_manifest = strict_json_loads(
                    read_artifact_bytes(second_manifest_path),
                    label="test second manifest",
                )
                packet_row_ids = []
                for record in second_manifest["reviewers"]:
                    packet = strict_json_loads(
                        read_artifact_bytes(Path(record["path"])),
                        label="test second packet",
                    )
                    for row in packet["rows"]:
                        self.assertNotIn("qwen_label", row)
                        self.assertNotIn("first_pass_decision", row)
                        self.assertNotIn("selection_reasons", row)
                        packet_row_ids.append(row["row_id"])
                    ids = [row["row_id"] for row in packet["rows"]]
                    self.assertEqual(
                        ids,
                        sorted(
                            ids,
                            key=lambda row_id: manual._order("46002:", row_id),
                        ),
                    )
                self.assertEqual(set(packet_row_ids), selected_ids)
                self.assertEqual(len(packet_row_ids), len(set(packet_row_ids)))

                original_packet_path = Path(second_manifest["reviewers"][0]["path"])
                original_packet = json.loads(read_artifact_bytes(original_packet_path))
                original_manifest = json.loads(read_artifact_bytes(second_manifest_path))
                leaked_packet = json.loads(read_artifact_bytes(original_packet_path))
                leaked_packet["rows"][0]["qwen_label"] = 1
                atomic_write_json(original_packet_path, leaked_packet)
                leaked_raw = read_artifact_bytes(original_packet_path)
                rebound_manifest = json.loads(read_artifact_bytes(second_manifest_path))
                rebound_manifest["reviewers"][0]["sha256"] = sha256_bytes(leaked_raw)
                rebound_manifest["reviewers"][0]["bytes"] = len(leaked_raw)
                atomic_write_json(second_manifest_path, rebound_manifest)
                with self.assertRaisesRegex(manual.ManualAuditError, "leaked context"):
                    manual._load_second_pass_manifest(second_manifest_path)
                atomic_write_json(original_packet_path, original_packet)
                atomic_write_json(second_manifest_path, original_manifest)

                second_review_paths = []
                for record in original_manifest["reviewers"]:
                    packet = strict_json_loads(
                        read_artifact_bytes(Path(record["path"])),
                        label="test second packet",
                    )
                    reviews = []
                    for row in packet["rows"]:
                        if row["unit"] == "answer":
                            reviews.append({
                                "row_id": row["row_id"],
                                "unit": "answer",
                                "decision": "answer_incorrect",
                                "corpus_evidence_issue": False,
                                "note": "Independent synthetic second answer review.",
                            })
                        else:
                            reviews.append({
                                "row_id": row["row_id"],
                                "unit": "claim",
                                "decision": 0,
                                "confidence": "high",
                                "ambiguity": False,
                                "rubric_evidence_defect": False,
                                "note": "Independent synthetic second claim review.",
                            })
                    review = {
                        "second_pass_review_schema_version": 1,
                        "reviewer": record["reviewer"],
                        "packet_sha256": record["sha256"],
                        "review_prompt_sha256": record["review_prompt_sha256"],
                        "reviews": reviews,
                    }
                    path = Path(record["review_output_path"])
                    write_immutable_json(path, review)
                    second_review_paths.append(path)
                second_validation_path = (
                    manifest_path.parent / "second-pass-validation.json"
                )
                manual.write_second_pass_validation(
                    manifest_path=second_manifest_path,
                    review_paths=second_review_paths,
                    output_path=second_validation_path,
                )
                summary_path = manifest_path.parent / "post-review-summary.json"
                manual.write_post_review_summary(
                    second_pass_validation_path=second_validation_path,
                    output_path=summary_path,
                    source_attestor=self._source_attestation,
                )
                summary = strict_json_loads(
                    read_artifact_bytes(summary_path), label="test summary"
                )
                self.assertEqual(len(summary["answer_reviews"]), 120)
                self.assertEqual(len(summary["answer_non_ok_details"]), 3)
                self.assertEqual(
                    summary["first_pass_claim_audit"]["overall"]["claim_rows"],
                    619,
                )
                self.assertFalse(summary["grade_policy"]["qwen_grades_changed"])
                sensitivity = summary["selected_subset_sensitivity"]
                overall = sensitivity["overall"]
                selected_claims = sum(
                    item["unit"] == "claim" for item in selected
                )
                self.assertTrue(overall["conditional_on_first_pass_selection"])
                self.assertTrue(overall["not_an_all_rows_estimate"])
                self.assertEqual(overall["selected_claim_rows"], selected_claims)
                self.assertEqual(
                    overall["second_vs_qwen"]["claim_rows"], selected_claims
                )
                self.assertEqual(
                    sum(
                        item["selected_claim_rows"]
                        for item in sensitivity["by_arm"].values()
                    ),
                    selected_claims,
                )
                weights = overall["weight_accounting"]
                self.assertEqual(
                    weights["selected_weight"],
                    weights["both_determinate_weight"]
                    + weights["uncertain_in_either_weight"],
                )
                self.assertEqual(
                    weights["both_determinate_weight"],
                    weights["first_second_disagreement_weight"]
                    + weights["both_confirm_qwen_agreement_weight"]
                    + weights["both_confirm_qwen_disagreement_weight"],
                )
                first_second = overall["first_vs_second"]
                self.assertEqual(
                    sum(first_second["confusion_0_1"].values()),
                    first_second["both_determinate_rows"],
                )

                original_review = json.loads(
                    read_artifact_bytes(second_review_paths[0])
                )
                wrong_identity = json.loads(
                    read_artifact_bytes(second_review_paths[0])
                )
                wrong_identity["reviewer"] = second_reviewers[1]
                atomic_write_json(second_review_paths[0], wrong_identity)
                with self.assertRaisesRegex(
                    manual.ManualAuditError, "unbound|bound output path"
                ):
                    manual.validate_second_pass(
                        manifest_path=second_manifest_path,
                        review_paths=second_review_paths,
                    )
                atomic_write_json(second_review_paths[0], original_review)

                original_first_review = json.loads(
                    read_artifact_bytes(first_review_paths[0])
                )
                changed_first_review = json.loads(
                    read_artifact_bytes(first_review_paths[0])
                )
                changed_first_review["reviews"][0]["note"] += " changed"
                atomic_write_json(first_review_paths[0], changed_first_review)
                with self.assertRaisesRegex(
                    manual.ManualAuditError, "differs from its binding"
                ):
                    manual._load_second_pass_manifest(second_manifest_path)
                atomic_write_json(first_review_paths[0], original_first_review)
            finally:
                if selection_patch is not None:
                    selection_patch.stop()
                for mocked in reversed(frozen):
                    mocked.stop()


if __name__ == "__main__":
    unittest.main()
