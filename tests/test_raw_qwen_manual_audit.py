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

    def _build(self, root: Path) -> tuple[Path, dict]:
        source = {
            "dirty": False,
            "files": {
                "studybench/raw_qwen_manual_audit.py": {
                    "sha256": "f" * 64,
                    "bytes": 1,
                }
            },
        }
        manifest_path = manual.build_packets(
            audit_id="test-raw-manual-audit",
            reviewers=REVIEWERS,
            output_root=root,
            source_attestor=lambda: {
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
            },
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


if __name__ == "__main__":
    unittest.main()
