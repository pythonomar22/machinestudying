from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from studybench import grade, report, screen_compare
from studybench.integrity import canonical_json_bytes, sha256_bytes, sha256_json
from studybench.provenance import server_assignment_record
from tests.test_compare_integrity import loaded_arm, set_generation_identities


DESCRIPTION = "Compare the exploratory study note with the paired control."
LOADER_CHECKER_CONFIGURATION = {
    "language": "python",
    "ready": False,
    "check_level": "syntax-only",
}
TEST_GRADING_RUNTIME = {"schema_version": 1, "identity": "test-grading-runtime"}
TEST_LOCAL_JUDGE_RUNTIME = {
    "schema_version": 1,
    "identity": "test-local-judge-runtime",
}


def source_record(commit: str = "a" * 40) -> dict:
    files = {
        "studybench/example.py": {"sha256": "b" * 64, "bytes": 1},
    }
    return {
        "git_commit": commit,
        "dirty": False,
        "files": files,
        "tree_sha256": sha256_json(files),
    }


def source_validation(source: dict | None = None) -> dict:
    source = source_record() if source is None else deepcopy(source)
    return {
        "schema_version": 1,
        "policy": grade.CURRENT_GENERATION_SOURCE_POLICY,
        "claim_ready": False,
        "paper_comparison_allowed": False,
        "generation_source": source,
        "generation_source_sha256": sha256_json(source),
        "grader_source": deepcopy(source),
        "grader_source_sha256": sha256_json(source),
    }


def historical_source_validation(
    source: dict, *, grader_commit: str = "c" * 40
) -> dict:
    validation = source_validation(source)
    grader_source = source_record(grader_commit)
    validation.update({
        "policy": grade.HISTORICAL_EXPLORATORY_SOURCE_POLICY,
        "grader_source": grader_source,
        "grader_source_sha256": sha256_json(grader_source),
    })
    return validation


def replace_source_validation(
    arm: screen_compare.LoadedScreenArm, validation: dict
) -> None:
    config = arm.audit["grading_manifest"]["config"]
    config["generation_source_validation"] = validation
    arm.audit["grading_manifest"]["sha256"] = sha256_json(config)


def local_arm(
    run_id: str,
    *,
    treatment: bool,
    score_offset: int,
    port: int,
    checker_ready: bool = False,
) -> screen_compare.LoadedScreenArm:
    strict = loaded_arm(
        run_id,
        treatment=treatment,
        score_offset=score_offset,
    )
    spec = strict.audit["run_manifest"]["spec"]
    spec.update({
        "purpose": "exploratory",
        "claim_ready": False,
        "source": source_record(),
        "preregistration": {
            "schema_version": 1,
            "status": "not_provided",
            "reason": "exploratory",
        },
    })
    strict.audit["run_manifest"]["spec_sha256"] = sha256_json(spec)
    url = f"http://localhost:{port}/v1"
    previous = strict.audit["grading_manifest"]["config"]
    checker = {
        "language": "python",
        "ready": checker_ready,
        "check_level": "contained-execution" if checker_ready else "syntax-only",
        "sandboxed": checker_ready,
        "error": None if checker_ready else "test checker is not configured",
    }
    checker_sha256 = grade.stable_sha256(checker)
    judge_intents = [{
        "expected_episode": f"{run_id}/direct/r0/q1.json",
        "outcome": "final",
    }]
    for grades in strict.population.values():
        for stored in grades:
            stored["compile_check"]["configuration_sha256"] = checker_sha256
            stored["judge_response_model"] = grade.LOCAL_GRADER_MODEL
    config = {
        "grade_schema_version": grade.GRADE_SCHEMA_VERSION,
        "judge_requested_model": grade.LOCAL_GRADER_MODEL,
        "judge_attempt_policy": grade.JUDGE_ATTEMPT_POLICY,
        "max_judge_attempts": grade.MAX_JUDGE_ATTEMPTS,
        "judge_base_url": url,
        "judge_response_models": [grade.LOCAL_GRADER_MODEL],
        "judge_system_fingerprint_scope": "accepted_final_attempts_only",
        "judge_system_fingerprints": previous["judge_system_fingerprints"],
        "accepted_judge_system_fingerprint_by_episode": previous[
            "accepted_judge_system_fingerprint_by_episode"
        ],
        "missing_judge_system_fingerprint_calls": previous[
            "missing_judge_system_fingerprint_calls"
        ],
        "whole_files": False,
        "judge_effort": "",
        "grading_runtime_sha256": sha256_json(TEST_GRADING_RUNTIME),
        "grading_runtime": TEST_GRADING_RUNTIME,
        "grading_spec_sha256_by_question": previous[
            "grading_spec_sha256_by_question"
        ],
        "judge_attempt_intents": {
            "policy": grade.JUDGE_ATTEMPT_POLICY,
            "count": len(judge_intents),
            "sha256": sha256_json(judge_intents),
            "artifacts": judge_intents,
        },
        "claim_ready": False,
        "grading_tier": "diagnostic-local-proxy",
        "local_proxy": True,
        "judge_endpoint_identity": grade.LOCAL_GRADER_ENDPOINT_IDENTITY,
        "judge_transport_urls": [url],
        "judge_model_revision": grade.LOCAL_GRADER_MODEL_REVISION,
        "judge_request_policy": grade.LOCAL_GRADER_REQUEST_POLICY,
        "judge_request_options": grade.LOCAL_GRADER_REQUEST_OPTIONS,
        "generation_source_validation": source_validation(spec["source"]),
        "local_judge_runtime_sha256": sha256_json(TEST_LOCAL_JUDGE_RUNTIME),
        "local_judge_runtime": TEST_LOCAL_JUDGE_RUNTIME,
        "checker_interpretation": {
            "language": "python",
            "sandbox_configuration_sha256": checker_sha256,
            "ready": checker_ready,
            "score_interpretation": (
                "all-metrics"
                if checker_ready
                else "lenient-and-core-conjunctive-checker-unavailable"
            ),
        },
    }
    strict.audit["grading_manifest"] = {
        "sha256": sha256_json(config),
        "config": config,
    }
    return screen_compare.LoadedScreenArm(
        report_path=strict.report_path,
        report_sha256=strict.report_sha256,
        population=strict.population,
        audit=strict.audit,
        aggregate=strict.aggregate,
        checker_configuration=checker,
    )


class LocalPairTests(unittest.TestCase):
    def setUp(self) -> None:
        self.control = local_arm(
            "control-a", treatment=False, score_offset=0, port=8123,
        )
        self.treatment = local_arm(
            "treatment-a", treatment=True, score_offset=10, port=9123
        )

    def test_different_loopback_ports_are_transport_only(self) -> None:
        intervention = screen_compare.validate_pair(
            self.control,
            self.treatment,
            intervention_description=DESCRIPTION,
        )
        self.assertEqual(
            intervention["grading_transport"]["control_validation_url"],
            "http://localhost:8123/v1",
        )
        self.assertEqual(
            intervention["grading_transport"]["treatment_validation_url"],
            "http://localhost:9123/v1",
        )
        self.assertEqual(
            intervention["seed_pairing"]["episode_seeds_sha256"],
            sha256_json(self.control.spec["seed_policy"]["episode_seeds"]),
        )

    def test_arm_specific_judge_intent_outcomes_are_normalized(self) -> None:
        treatment = deepcopy(self.treatment)
        config = treatment.audit["grading_manifest"]["config"]
        config["judge_attempt_intents"] = {
            "policy": grade.JUDGE_ATTEMPT_POLICY,
            "count": 0,
            "sha256": sha256_json([]),
            "artifacts": [],
        }
        treatment.audit["grading_manifest"]["sha256"] = sha256_json(config)
        screen_compare.validate_pair(
            self.control,
            treatment,
            intervention_description=DESCRIPTION,
        )

    def test_malformed_judge_intent_ledger_is_fatal(self) -> None:
        for field, value in (
            ("policy", "different-policy"),
            ("sha256", "0" * 64),
        ):
            with self.subTest(field=field):
                drifted = deepcopy(self.treatment)
                config = drifted.audit["grading_manifest"]["config"]
                config["judge_attempt_intents"][field] = value
                drifted.audit["grading_manifest"]["sha256"] = sha256_json(config)
                with self.assertRaisesRegex(
                    screen_compare.ScreenComparisonIntegrityError,
                    "judge-attempt intent ledger is invalid",
                ):
                    screen_compare.validate_pair(
                        self.control,
                        drifted,
                        intervention_description=DESCRIPTION,
                    )

    def test_fresh_report_port_need_not_match_observed_grade_ports(self) -> None:
        control = deepcopy(self.control)
        config = control.audit["grading_manifest"]["config"]
        config["judge_base_url"] = "http://localhost:8223/v1"
        control.audit["grading_manifest"]["sha256"] = sha256_json(config)
        intervention = screen_compare.validate_pair(
            control,
            self.treatment,
            intervention_description=DESCRIPTION,
        )
        transport = intervention["grading_transport"]
        self.assertEqual(
            transport["control_validation_url"],
            "http://localhost:8223/v1",
        )
        self.assertEqual(
            transport["control_observed_urls"],
            ["http://localhost:8123/v1"],
        )

    def test_remote_grading_transport_is_fatal(self) -> None:
        drifted = deepcopy(self.treatment)
        config = drifted.audit["grading_manifest"]["config"]
        config["judge_base_url"] = "https://grader.invalid/v1"
        config["judge_transport_urls"] = ["https://grader.invalid/v1"]
        drifted.audit["grading_manifest"]["sha256"] = sha256_json(config)
        with self.assertRaisesRegex(
            screen_compare.ScreenComparisonIntegrityError,
            "non-loopback",
        ):
            screen_compare.validate_pair(
                self.control,
                drifted,
                intervention_description=DESCRIPTION,
            )

        drifted = deepcopy(self.treatment)
        config = drifted.audit["grading_manifest"]["config"]
        config["judge_request_options"] = {"temperature": 0.1}
        drifted.audit["grading_manifest"]["sha256"] = sha256_json(config)
        with self.assertRaisesRegex(
            screen_compare.ScreenComparisonIntegrityError,
            "pinned diagnostic local-Qwen",
        ):
            screen_compare.validate_pair(
                self.control,
                drifted,
                intervention_description=DESCRIPTION,
            )

        drifted = deepcopy(self.treatment)
        config = drifted.audit["grading_manifest"]["config"]
        config["local_judge_runtime"] = {
            "schema_version": 1,
            "identity": "different-local-judge-runtime",
        }
        config["local_judge_runtime_sha256"] = sha256_json(
            config["local_judge_runtime"]
        )
        drifted.audit["grading_manifest"]["sha256"] = sha256_json(config)
        with self.assertRaisesRegex(
            screen_compare.ScreenComparisonIntegrityError,
            "substantive grading contracts",
        ):
            screen_compare.validate_pair(
                self.control,
                drifted,
                intervention_description=DESCRIPTION,
            )

    def test_seed_and_substantive_grading_drift_are_fatal(self) -> None:
        drifted = deepcopy(self.treatment)
        drifted.audit["run_manifest"]["spec"]["seed_policy"][
            "episode_seeds"
        ]["direct/r0/q1.json"] += 1
        with self.assertRaisesRegex(
            screen_compare.ScreenComparisonIntegrityError,
            "seed-policy episode_seeds",
        ):
            screen_compare.validate_pair(
                self.control,
                drifted,
                intervention_description=DESCRIPTION,
            )

    def test_source_stage_digest_and_manifest_mismatches_are_fatal(self) -> None:
        digest_tamper = deepcopy(self.treatment)
        validation = deepcopy(
            digest_tamper.audit["grading_manifest"]["config"]
            ["generation_source_validation"]
        )
        validation["grader_source_sha256"] = "0" * 64
        replace_source_validation(digest_tamper, validation)
        with self.assertRaisesRegex(
            screen_compare.ScreenComparisonIntegrityError,
            "invalid generation/grader source binding",
        ):
            screen_compare.validate_pair(
                self.control,
                digest_tamper,
                intervention_description=DESCRIPTION,
            )

        manifest_mismatch = deepcopy(self.treatment)
        different = source_record("d" * 40)
        replace_source_validation(
            manifest_mismatch, source_validation(different)
        )
        with self.assertRaisesRegex(
            screen_compare.ScreenComparisonIntegrityError,
            "does not match the run manifest generation source",
        ):
            screen_compare.validate_pair(
                self.control,
                manifest_mismatch,
                intervention_description=DESCRIPTION,
            )

    def test_pair_rejects_grader_source_and_source_policy_mismatches(self) -> None:
        generation_mismatch = deepcopy(self.treatment)
        different_generation = source_record("d" * 40)
        generation_mismatch.audit["run_manifest"]["spec"]["source"] = (
            different_generation
        )
        generation_mismatch.audit["run_manifest"]["spec_sha256"] = sha256_json(
            generation_mismatch.spec
        )
        replace_source_validation(
            generation_mismatch,
            source_validation(different_generation),
        )
        with self.assertRaisesRegex(
            screen_compare.ScreenComparisonIntegrityError,
            "outside the note intervention",
        ):
            screen_compare.validate_pair(
                self.control,
                generation_mismatch,
                intervention_description=DESCRIPTION,
            )

        control = deepcopy(self.control)
        treatment = deepcopy(self.treatment)
        replace_source_validation(
            control,
            historical_source_validation(
                control.spec["source"], grader_commit="c" * 40
            ),
        )
        replace_source_validation(
            treatment,
            historical_source_validation(
                treatment.spec["source"], grader_commit="d" * 40
            ),
        )
        with self.assertRaisesRegex(
            screen_compare.ScreenComparisonIntegrityError,
            "different substantive grading contracts",
        ):
            screen_compare.validate_pair(
                control,
                treatment,
                intervention_description=DESCRIPTION,
            )

        historical_treatment = deepcopy(self.treatment)
        replace_source_validation(
            historical_treatment,
            historical_source_validation(historical_treatment.spec["source"]),
        )
        with self.assertRaisesRegex(
            screen_compare.ScreenComparisonIntegrityError,
            "different substantive grading contracts",
        ):
            screen_compare.validate_pair(
                self.control,
                historical_treatment,
                intervention_description=DESCRIPTION,
            )

    def test_screen_rejects_grading_smoke_source_policy(self) -> None:
        smoke = deepcopy(self.treatment)
        validation = deepcopy(
            smoke.audit["grading_manifest"]["config"]
            ["generation_source_validation"]
        )
        validation["policy"] = grade.CURRENT_SMOKE_SOURCE_POLICY
        replace_source_validation(smoke, validation)
        with self.assertRaisesRegex(
            screen_compare.ScreenComparisonIntegrityError,
            "grading-smoke source binding",
        ):
            screen_compare.validate_pair(
                self.control,
                smoke,
                intervention_description=DESCRIPTION,
            )

    def test_non_note_manifest_drift_is_fatal(self) -> None:
        drifted = deepcopy(self.treatment)
        drifted.audit["run_manifest"]["spec"]["sampling"]["temperature"] = 0.2
        with self.assertRaisesRegex(
            screen_compare.ScreenComparisonIntegrityError,
            "outside the note intervention",
        ):
            screen_compare.validate_pair(
                self.control,
                drifted,
                intervention_description=DESCRIPTION,
            )

    def test_generation_missing_fingerprint_counts_are_disclosed(self) -> None:
        treatment = deepcopy(self.treatment)
        identities = deepcopy(
            treatment.audit["generation_runtime"]["provider_identity_by_episode"]
        )
        target = sorted(identities)[0]
        identities[target]["missing_system_fingerprint_calls"] = 7
        identities[target]["provider_call_count"] = 8
        set_generation_identities(treatment, identities)
        intervention = screen_compare.validate_pair(
            self.control,
            treatment,
            intervention_description=DESCRIPTION,
        )
        runtime = intervention["generation_runtime_pairing"]
        self.assertFalse(runtime["missing_call_counts_are_equality_gated"])
        self.assertFalse(runtime["provider_call_counts_are_equality_gated"])
        target_record = next(
            record
            for record in runtime["records"]
            if f"{record['budget']}/r{record['rollout']}/{record['qid']}.json"
            == target
        )
        self.assertEqual(
            target_record["control"]["missing_system_fingerprint_calls"], 0
        )
        self.assertEqual(
            target_record["treatment"]["missing_system_fingerprint_calls"], 7
        )
        self.assertEqual(
            runtime["verification"],
            "matched_pinned_generation_runtime_with_missing_fingerprints_"
            "disclosed",
        )

        mismatched = deepcopy(self.treatment)
        identities = deepcopy(
            mismatched.audit["generation_runtime"]["provider_identity_by_episode"]
        )
        for identity in identities.values():
            identity["system_fingerprints"] = [
                "different-generation-fingerprint"
            ]
        set_generation_identities(mismatched, identities)
        with self.assertRaisesRegex(
            screen_compare.ScreenComparisonIntegrityError,
            "paired generation identities differ",
        ):
            screen_compare.validate_pair(
                self.control,
                mismatched,
                intervention_description=DESCRIPTION,
            )

    def test_generation_fingerprint_swaps_cannot_hide_in_arm_level_sets(self) -> None:
        control = deepcopy(self.control)
        treatment = deepcopy(self.treatment)
        control_identities = deepcopy(
            control.audit["generation_runtime"]["provider_identity_by_episode"]
        )
        treatment_identities = deepcopy(
            treatment.audit["generation_runtime"]["provider_identity_by_episode"]
        )
        first, second = sorted(control_identities)[:2]
        control_identities[first]["system_fingerprints"] = ["fingerprint-a"]
        control_identities[second]["system_fingerprints"] = ["fingerprint-b"]
        treatment_identities[first]["system_fingerprints"] = ["fingerprint-b"]
        treatment_identities[second]["system_fingerprints"] = ["fingerprint-a"]
        set_generation_identities(control, control_identities)
        set_generation_identities(treatment, treatment_identities)
        self.assertEqual(
            control.audit["generation_runtime"]["system_fingerprints"],
            treatment.audit["generation_runtime"]["system_fingerprints"],
        )
        with self.assertRaisesRegex(
            screen_compare.ScreenComparisonIntegrityError,
            "multiple available provider fingerprints",
        ):
            screen_compare.validate_pair(
                control, treatment, intervention_description=DESCRIPTION
            )

    def test_unready_checker_omits_strict_and_compile_metrics(self) -> None:
        artifact = screen_compare.build_screen_comparison(
            self.control,
            self.treatment,
            intervention_description=DESCRIPTION,
            bootstrap_replicates=20,
            bootstrap_seed=7,
        )
        self.assertFalse(artifact["claim_ready"])
        self.assertTrue(artifact["diagnostic_only"])
        self.assertEqual(
            artifact["intervention"]["checker_interpretation"]["score_policy"],
            "lenient_and_core_conjunctive_strict_and_compile_unavailable",
        )
        for budget in report.BUDGET_ORDER:
            point = artifact["point_estimates"]["budgets"][budget]
            interval = artifact["bootstrap"]["results"]["budgets"][budget]
            self.assertNotIn("strict", point)
            self.assertIn("len_cc", point)
            self.assertNotIn("compile_rate", point)
            self.assertNotIn("strict", interval)
            self.assertIn("len_cc", interval)
            self.assertNotIn("compile_rate", interval)
            self.assertEqual(point["lenient"]["treatment_minus_control"], 10)
            self.assertEqual(point["len_cc"]["treatment_minus_control"], 10)
            self.assertEqual(interval["lenient"]["lower_95"], 10)
            self.assertEqual(interval["lenient"]["upper_95"], 10)
        self.assertNotIn(
            "expertise_strict", artifact["point_estimates"]["expertise"]
        )
        self.assertIn("same-model", " ".join(artifact["limitations"]))
        self.assertIn("systematic error", " ".join(artifact["limitations"]))
        self.assertIn("inconclusive", " ".join(artifact["limitations"]))
        self.assertEqual(
            artifact["bootstrap"]["zero_in_interval_interpretation"],
            "inconclusive_not_parity_or_equivalence",
        )

    def test_smoke_or_partial_arms_are_rejected(self) -> None:
        smoke = deepcopy(self.control)
        smoke.spec["purpose"] = "smoke"
        smoke.spec["preregistration"]["reason"] = "smoke"
        with self.assertRaisesRegex(
            screen_compare.ScreenComparisonIntegrityError,
            "smoke, partial",
        ):
            screen_compare.validate_pair(
                smoke,
                self.treatment,
                intervention_description=DESCRIPTION,
            )

        partial = deepcopy(self.control)
        partial.population["direct"].pop()
        with self.assertRaisesRegex(
            screen_compare.ScreenComparisonIntegrityError,
            "partial",
        ):
            screen_compare.validate_pair(
                partial,
                self.treatment,
                intervention_description=DESCRIPTION,
            )

    def test_checker_interpretation_must_match_bound_configuration(self) -> None:
        drifted = deepcopy(self.treatment)
        config = drifted.audit["grading_manifest"]["config"]
        config["checker_interpretation"]["ready"] = True
        drifted.audit["grading_manifest"]["sha256"] = sha256_json(config)
        with self.assertRaisesRegex(
            screen_compare.ScreenComparisonIntegrityError,
            "checker interpretation",
        ):
            screen_compare.validate_pair(
                self.control,
                drifted,
                intervention_description=DESCRIPTION,
            )

    def test_treatment_no_answer_remains_analyzable_as_itt_zero(self) -> None:
        treatment = deepcopy(self.treatment)
        stored = next(
            item
            for item in treatment.population["direct"]
            if item["rollout"] == 0 and item["qid"] == "q1"
        )
        stored.update({
            "episode_status": "no_answer",
            "lenient": 0,
            "strict": 0,
            "cores_ok": False,
            "judge_response_model": None,
        })
        stored["compile_check"]["compile_ok"] = False
        config = treatment.audit["grading_manifest"]["config"]
        del config["accepted_judge_system_fingerprint_by_episode"][
            "direct/r0/q1.json"
        ]
        treatment.audit["grading_manifest"]["sha256"] = sha256_json(config)
        object.__setattr__(
            treatment,
            "aggregate",
            report.aggregate_population(treatment.population),
        )

        artifact = screen_compare.build_screen_comparison(
            self.control,
            treatment,
            intervention_description=DESCRIPTION,
            bootstrap_replicates=5,
            bootstrap_seed=0,
        )
        runtime = artifact["intervention"]["accepted_judge_runtime_pairing"]
        self.assertEqual(runtime["cells_with_exactly_one_no_answer"], 1)
        self.assertEqual(
            artifact["point_estimates"]["budgets"]["direct"][
                "treatment_no_answer"
            ],
            1,
        )

    def test_joint_judge_fingerprint_mismatch_is_fatal_or_disclosed(self) -> None:
        relative = "direct/r0/q1.json"
        mismatch = deepcopy(self.treatment)
        config = mismatch.audit["grading_manifest"]["config"]
        config["accepted_judge_system_fingerprint_by_episode"][
            relative
        ] = "different-fingerprint"
        config["judge_system_fingerprints"] = [
            "different-fingerprint",
            "judge-fingerprint",
        ]
        mismatch.audit["grading_manifest"]["sha256"] = sha256_json(config)
        with self.assertRaisesRegex(
            screen_compare.ScreenComparisonIntegrityError,
            "accepted judge fingerprints differ",
        ):
            screen_compare.validate_pair(
                self.control,
                mismatch,
                intervention_description=DESCRIPTION,
            )

        incomplete = deepcopy(self.treatment)
        config = incomplete.audit["grading_manifest"]["config"]
        config["accepted_judge_system_fingerprint_by_episode"][relative] = None
        config["missing_judge_system_fingerprint_calls"] = 1
        incomplete.audit["grading_manifest"]["sha256"] = sha256_json(config)
        intervention = screen_compare.validate_pair(
            self.control,
            incomplete,
            intervention_description=DESCRIPTION,
        )
        runtime = intervention["accepted_judge_runtime_pairing"]
        self.assertEqual(
            runtime["verification"],
            "matched_models_fingerprint_incomplete_and_disclosed",
        )
        self.assertEqual(
            runtime["jointly_judged_cells_with_missing_fingerprint"], 1
        )

    def test_ready_checker_allows_secondary_strict_metrics(self) -> None:
        control = local_arm(
            "control-a", treatment=False, score_offset=0, port=8123,
            checker_ready=True,
        )
        treatment = local_arm(
            "treatment-a", treatment=True, score_offset=10, port=9123,
            checker_ready=True,
        )
        artifact = screen_compare.build_screen_comparison(
            control,
            treatment,
            intervention_description=DESCRIPTION,
            bootstrap_replicates=5,
            bootstrap_seed=0,
        )
        for budget in report.BUDGET_ORDER:
            self.assertIn("strict", artifact["point_estimates"]["budgets"][budget])
            self.assertIn(
                "compile_rate", artifact["point_estimates"]["budgets"][budget]
            )

    def test_writer_is_content_addressed_and_recomputes(self) -> None:
        artifact = screen_compare.build_screen_comparison(
            self.control,
            self.treatment,
            intervention_description=DESCRIPTION,
            bootstrap_replicates=5,
            bootstrap_seed=3,
        )
        with tempfile.TemporaryDirectory() as directory, patch.object(
            screen_compare,
            "load_local_report",
            side_effect=[
                self.control,
                self.treatment,
                self.control,
                self.treatment,
            ],
        ):
            first = screen_compare.write_screen_comparison(
                artifact, output_root=directory
            )
            repeated = screen_compare.write_screen_comparison(
                artifact, output_root=directory
            )
            self.assertEqual(first, repeated)
            self.assertEqual(json.loads(first.read_bytes()), artifact)
            self.assertEqual(
                first.name, f"screen-comparison-{sha256_json(artifact)}.json"
            )

        forged = deepcopy(artifact)
        forged["claim_ready"] = True
        with self.assertRaisesRegex(
            screen_compare.ScreenComparisonIntegrityError,
            "claim-ready",
        ):
            screen_compare.write_screen_comparison(forged)


class LocalReportLoaderTests(unittest.TestCase):
    def _fixture(
        self,
        root: Path,
        *,
        checker_configuration: dict = LOADER_CHECKER_CONFIGURATION,
        bootstrap_replicates: int = 0,
    ) -> tuple[Path, dict[str, list[dict]], dict]:
        run_id = "run-a"
        task = "fake"
        run_root = root / "runs" / run_id
        grade_root = root / "grades" / run_id / "local-qwen"
        manifest_path = run_root / task / "manifest.json"
        manifest_path.parent.mkdir(parents=True)
        manifest_path.write_bytes(b"{}")
        population = {budget: [] for budget in report.BUDGET_ORDER}
        records = []
        for index, budget in enumerate(report.BUDGET_ORDER):
            relative = Path(task) / budget / "r0" / "q1.json"
            episode_path = run_root / relative
            grade_path = grade_root / relative
            episode_path.parent.mkdir(parents=True, exist_ok=True)
            grade_path.parent.mkdir(parents=True, exist_ok=True)
            episode_path.write_bytes(b"{}")
            grade_path.write_bytes(b"{}")
            stored = {
                "task": task,
                "qid": "q1",
                "budget": budget,
                "rollout": 0,
                "episode_status": "ok",
                "judge_response_model": grade.LOCAL_GRADER_MODEL,
                "lenient": 10 + index,
                "strict": 0,
                "cores_ok": False,
                "compile_check": {
                    "compile_ok": False,
                    "configuration_sha256": grade.stable_sha256(
                        checker_configuration
                    ),
                },
                "gen_tokens": 4_000 + index,
            }
            population[budget].append(stored)
            records.append({
                "task": task,
                "qid": "q1",
                "budget": budget,
                "rollout": 0,
                "episode_path": str(episode_path),
                "episode_sha256": sha256_bytes(b"{}"),
                "grade_path": str(grade_path),
                "grade_sha256": sha256_bytes(b"{}"),
            })
        expected_episodes = [
            f"{budget}/r0/q1.json" for budget in report.BUDGET_ORDER
        ]
        spec = {
            "run_id": run_id,
            "task": task,
            "master_seed": 7,
            "rollouts": 1,
            "purpose": "exploratory",
            "claim_ready": False,
            "budgets": report.BUDGET_ORDER,
            "questions": [{"id": "q1", "sha256": "1" * 64}],
            "expected_episodes": expected_episodes,
            "server_assignment": server_assignment_record(expected_episodes, 1),
            "seed_policy": {
                "algorithm": "sha256-canonical-json-mod-2147483647",
                "namespace": "native-react",
                "seed_group": "screen-pair",
                "ordered_parts": [
                    "master_seed",
                    "namespace",
                    "seed_group",
                    "task",
                    "qid",
                    "budget",
                    "rollout",
                ],
                "episode_seeds": {
                    relative: index
                    for index, relative in enumerate(expected_episodes)
                },
            },
            "preregistration": {
                "schema_version": 1,
                "status": "not_provided",
                "reason": "exploratory",
            },
            "source": source_record(),
            "extra": {"server_transport": {"server_count": 1}},
        }
        config = {
            "judge_attempt_intents": {
                "policy": grade.JUDGE_ATTEMPT_POLICY,
                "count": 0,
                "sha256": sha256_json([]),
                "artifacts": [],
            },
            "grade_schema_version": grade.GRADE_SCHEMA_VERSION,
            "judge_requested_model": grade.LOCAL_GRADER_MODEL,
            "judge_attempt_policy": grade.JUDGE_ATTEMPT_POLICY,
            "max_judge_attempts": grade.MAX_JUDGE_ATTEMPTS,
            # The population was graded on 8123 and revalidated/report-written
            # in a fresh allocation on 8223.
            "judge_base_url": "http://localhost:8223/v1",
            "judge_response_models": [grade.LOCAL_GRADER_MODEL],
            "judge_system_fingerprint_scope": "accepted_final_attempts_only",
            "judge_system_fingerprints": ["local-fingerprint"],
            "accepted_judge_system_fingerprint_by_episode": {
                relative: "local-fingerprint" for relative in expected_episodes
            },
            "missing_judge_system_fingerprint_calls": 0,
            "whole_files": False,
            "judge_effort": "",
            "grading_runtime_sha256": sha256_json(TEST_GRADING_RUNTIME),
            "grading_runtime": TEST_GRADING_RUNTIME,
            "grading_spec_sha256_by_question": {"q1": "1" * 64},
            "claim_ready": False,
            "grading_tier": "diagnostic-local-proxy",
            "local_proxy": True,
            "judge_endpoint_identity": grade.LOCAL_GRADER_ENDPOINT_IDENTITY,
            "judge_transport_urls": ["http://localhost:8123/v1"],
            "judge_model_revision": grade.LOCAL_GRADER_MODEL_REVISION,
            "judge_request_policy": grade.LOCAL_GRADER_REQUEST_POLICY,
            "judge_request_options": grade.LOCAL_GRADER_REQUEST_OPTIONS,
            "generation_source_validation": source_validation(spec["source"]),
            "local_judge_runtime_sha256": sha256_json(
                TEST_LOCAL_JUDGE_RUNTIME
            ),
            "local_judge_runtime": TEST_LOCAL_JUDGE_RUNTIME,
            "checker_interpretation": {
                "language": "python",
                "sandbox_configuration_sha256": grade.stable_sha256(
                    checker_configuration
                ),
                "ready": checker_configuration["ready"],
                "score_interpretation": (
                    "all-metrics"
                    if checker_configuration["ready"]
                    else "lenient-and-core-conjunctive-checker-unavailable"
                ),
            },
        }
        audit = {
            "run_manifest": {
                "path": str(manifest_path),
                "sha256": sha256_bytes(b"{}"),
                "spec_sha256": sha256_json(spec),
                "spec": spec,
            },
            "generation_runtime": {
                "response_models": ["generation-model"],
                "system_fingerprints": [],
                "provider_call_count": 4,
                "missing_system_fingerprint_calls": 4,
                "provider_identity_scope": "final_manifest_episodes",
                "provider_identity_by_episode": {
                    relative: {
                        "harness_usage": "native_turns",
                        "provider_call_count": 1,
                        "response_models": ["generation-model"],
                        "system_fingerprints": [],
                        "missing_system_fingerprint_calls": 1,
                        "server_slot": spec["server_assignment"]["episode_slots"][
                            relative
                        ],
                    }
                    for relative in expected_episodes
                },
            },
            "note_provenance": {
                "construction_manifest_sha256": None,
                "study_id": None,
                "method": None,
                "manifest_type": None,
            },
            "failed_attempts": {
                "count": 0,
                "sha256": sha256_json([]),
                "artifacts": [],
            },
            "failed_judge_audits": {
                "count": 0,
                "sha256": sha256_json([]),
                "artifacts": [],
            },
            "grading_manifest": {
                "sha256": sha256_json(config),
                "config": config,
            },
            "population": records,
            "population_sha256": sha256_json(records),
        }
        artifact = {
            "report_schema_version": report.REPORT_SCHEMA_VERSION,
            "claim_ready": False,
            "task": task,
            "run_id": run_id,
            "budget_order": report.BUDGET_ORDER,
            **audit,
            "aggregate": report.reportable_aggregate(
                report.aggregate_population(population),
                checker_ready=checker_configuration["ready"],
            ),
            "bootstrap": {
                "replicates": bootstrap_replicates,
                "seed": 0,
                "results": report.reportable_bootstrap(
                    report.bootstrap_population(
                        population, bootstrap_replicates, seed=0
                    )
                    if bootstrap_replicates
                    else None,
                    checker_ready=checker_configuration["ready"],
                ),
            },
            "paper_comparison": None,
            "report_source": {
                "studybench/report.py": grade.file_sha256(
                    Path(report.__file__).resolve()
                )
            },
        }
        data = canonical_json_bytes(artifact)
        path = root / f"report-{sha256_bytes(data)}.json"
        path.write_bytes(data)
        return path, population, audit

    def test_loader_revalidates_complete_local_population(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, population, audit = self._fixture(root)
            corpus = SimpleNamespace(language="python")
            with (
                patch.object(report, "ROOT", root),
                patch.object(report, "CORPORA", {"fake": corpus}),
                patch.object(
                    report,
                    "revalidate_recorded_local_diagnostic_evaluation",
                    return_value=(population, audit),
                ) as loader,
                patch.object(
                    report,
                    "load_local_diagnostic_evaluation",
                    side_effect=AssertionError("live local runtime path was called"),
                ),
                patch.object(
                    screen_compare.sandbox,
                    "configuration_record",
                    return_value=LOADER_CHECKER_CONFIGURATION,
                ),
            ):
                arm = screen_compare.load_local_report(path)
            self.assertEqual(arm.report_sha256, sha256_bytes(path.read_bytes()))
            loader.assert_called_once_with(
                "fake",
                root / "grades/run-a/local-qwen",
                root / "runs/run-a",
                rollouts=1,
                judge_base_url="http://localhost:8223/v1",
                grading_runtime=TEST_GRADING_RUNTIME,
                local_judge_runtime=TEST_LOCAL_JUDGE_RUNTIME,
                whole_files=False,
                historical_exploratory_source_commit=None,
            )

    def test_loader_rejects_unavailable_checker_zero_sentinels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, population, audit = self._fixture(
                root, bootstrap_replicates=3
            )
            artifact = json.loads(path.read_bytes())
            self.assertIsNone(
                artifact["aggregate"]["budgets"]["direct"]["strict"]
            )
            self.assertIsNotNone(artifact["bootstrap"]["results"]["wauc_cc"])
            artifact["aggregate"]["budgets"]["direct"]["strict"] = 0
            data = canonical_json_bytes(artifact)
            forged = root / f"report-{sha256_bytes(data)}.json"
            forged.write_bytes(data)
            corpus = SimpleNamespace(language="python")
            with (
                patch.object(report, "ROOT", root),
                patch.object(report, "CORPORA", {"fake": corpus}),
                patch.object(
                    report,
                    "revalidate_recorded_local_diagnostic_evaluation",
                    return_value=(population, audit),
                ),
                patch.object(
                    screen_compare.sandbox,
                    "configuration_record",
                    return_value=LOADER_CHECKER_CONFIGURATION,
                ),
                self.assertRaisesRegex(
                    screen_compare.ScreenComparisonIntegrityError,
                    "aggregate no longer recomputes",
                ),
            ):
                screen_compare.load_local_report(forged)

    def test_loader_rejects_checker_ready_nulls(self) -> None:
        ready_checker = {
            "language": "python",
            "ready": True,
            "check_level": "contained-execution",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, population, audit = self._fixture(
                root, checker_configuration=ready_checker
            )
            artifact = json.loads(path.read_bytes())
            self.assertEqual(
                artifact["aggregate"]["budgets"]["direct"]["strict"], 0
            )
            artifact["aggregate"]["budgets"]["direct"]["strict"] = None
            data = canonical_json_bytes(artifact)
            forged = root / f"report-{sha256_bytes(data)}.json"
            forged.write_bytes(data)
            corpus = SimpleNamespace(language="python")
            with (
                patch.object(report, "ROOT", root),
                patch.object(report, "CORPORA", {"fake": corpus}),
                patch.object(
                    report,
                    "revalidate_recorded_local_diagnostic_evaluation",
                    return_value=(population, audit),
                ),
                patch.object(
                    screen_compare.sandbox,
                    "configuration_record",
                    return_value=ready_checker,
                ),
                self.assertRaisesRegex(
                    screen_compare.ScreenComparisonIntegrityError,
                    "aggregate no longer recomputes",
                ),
            ):
                screen_compare.load_local_report(forged)

    def test_loader_rejects_claim_ready_or_non_content_addressed_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _, _ = self._fixture(root)
            artifact = json.loads(path.read_bytes())
            artifact["claim_ready"] = True
            data = canonical_json_bytes(artifact)
            forged = root / f"report-{sha256_bytes(data)}.json"
            forged.write_bytes(data)
            with self.assertRaisesRegex(
                screen_compare.ScreenComparisonIntegrityError,
                "not diagnostic",
            ):
                screen_compare.load_local_report(forged)

            wrong_name = root / "report-wrong.json"
            wrong_name.write_bytes(path.read_bytes())
            with (
                patch.object(
                    report,
                    "CORPORA",
                    {"fake": SimpleNamespace(language="python")},
                ),
                self.assertRaisesRegex(
                    screen_compare.ScreenComparisonIntegrityError,
                    "content-addressed",
                ),
            ):
                screen_compare.load_local_report(wrong_name)


if __name__ == "__main__":
    unittest.main()
