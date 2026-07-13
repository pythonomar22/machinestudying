from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from studybench import compare, grade as grade_module, report
from studybench.integrity import canonical_json_bytes, sha256_bytes, sha256_json, stable_seed
from studybench.provenance import environment_contract_record


QIDS = ("q1", "q2")
ROLLOUTS = 2
INTERVENTION = "Add the preregistered study note."
TEST_JUDGE_BASE_URL = "https://judge.test/v1"


def population(offset: int = 0) -> dict[str, list[dict]]:
    result = {budget: [] for budget in report.BUDGET_ORDER}
    for budget_index, budget in enumerate(report.BUDGET_ORDER):
        for rollout in range(ROLLOUTS):
            for qid_index, qid in enumerate(QIDS):
                score = 10 + budget_index * 5 + rollout + qid_index + offset
                result[budget].append({
                    "task": "fake",
                    "qid": qid,
                    "budget": budget,
                    "rollout": rollout,
                    "episode_status": "ok",
                    "lenient": score,
                    "strict": score,
                    "cores_ok": True,
                    "compile_check": {"compile_ok": True},
                    "gen_tokens": 4_000 * (budget_index + 1),
                })
    return result


def run_spec(run_id: str, *, treatment: bool) -> dict:
    expected = []
    seeds = {}
    for budget in report.BUDGET_ORDER:
        for rollout in range(ROLLOUTS):
            for qid in QIDS:
                relative = f"{budget}/r{rollout}/{qid}.json"
                expected.append(relative)
                seeds[relative] = stable_seed(
                    19, "native-react", "paired-study", "fake", qid, budget, rollout)
    note = None
    template = None
    prompt_hashes = {qid: sha256_bytes(f"base-{qid}".encode()) for qid in QIDS}
    if treatment:
        note = {
            "sha256": "b" * 64,
            "bytes": 4,
            "snapshot": "inputs/note.md",
            "construction_manifest": {
                "sha256": "c" * 64,
                "snapshot": "inputs/note-manifest.json",
            },
        }
        template = "Study note:\n{note}\nQuestion:\n"
        prompt_hashes = {
            qid: sha256_bytes(f"treatment-{qid}".encode()) for qid in QIDS
        }
    allocation_inventory = {
        "schema_version": 1,
        "hostname": f"node-{run_id}",
        "cuda_visible_devices": "0",
        "gpu_count": 1,
        "gpus": [{
            "cuda_identifier": "0",
            "uuid": f"GPU-{run_id}",
            "name": "L40S",
            "memory_mib": 46_068,
            "driver_version": "590.48",
        }],
        "slurm": {
            "job_id": f"job-{run_id}",
            "job_gpus": f"GPU-{run_id}",
            "step_gpus": None,
            "job_nodelist": f"node-{run_id}",
            "node_id": "0",
        },
    }
    environment = {
        "vllm_environment_sha256": "1" * 64,
        "vllm_environment": {
            "path": f"logs/{run_id}.packages.txt",
            "sha256": "1" * 64,
            "bytes": 10,
            "lines": ["vllm==0.24.0"],
        },
        "vllm_runtime": {
            "path": f"logs/{run_id}.runtime.json",
            "sha256": "2" * 64,
            "bytes": 20,
            "inventory": {"runtime": "same"},
        },
        "model_cache": {
            "path": f"logs/{run_id}.model.json",
            "sha256": "3" * 64,
            "bytes": 30,
            "inventory": {"tree_sha256": "4" * 64},
        },
        "allocation": {
            "path": f"logs/{run_id}.gpus.json",
            "sha256": sha256_json(allocation_inventory),
            "bytes": len(canonical_json_bytes(allocation_inventory)),
            "inventory": allocation_inventory,
        },
        "cuda_visible_devices": "0",
        "slurm_job_id": f"job-{run_id}",
        "server_launch_id": sha256_bytes(f"key-{run_id}".encode()),
        "vllm_api_key_sha256": sha256_bytes(f"key-{run_id}".encode()),
        "gpu_models": ["L40S"],
        "nvidia_driver": ["590.48"],
        "visible_gpu_count": "1",
        "server_count": "1",
        "tensor_parallel_size": "1",
    }
    preregistration_document = {
        "schema_version": 1,
        "preregistration_id": "paired-study-r1",
        "hypothesis": "The study note improves expertise_lenient.",
        "intervention": INTERVENTION,
        "task": "fake",
        "corpus_commit": "e" * 40,
        "source_commit": "1" * 40,
        "question_bundle_sha256": "d" * 64,
        "arms": {
            "control": {"run_id": "control-a", "note_sha256": None},
            "treatment": {"run_id": "treatment-a", "note_sha256": "b" * 64},
        },
        "evaluation": {
            "harness": "native-react",
            "model": "model",
            "model_revision": "a" * 40,
            "sampling": {"temperature": 1.0},
            "master_seed": 19,
            "seed_namespace": "native-react",
            "seed_group": "paired-study",
            "budgets": report.BUDGET_ORDER,
            "rollouts": ROLLOUTS,
        },
        "failure_policy": {
            "model_no_answer": "intention-to-run_zero",
            "infrastructure_error": "invalid_until_retried",
            "forced_short": "invalid_until_retried",
        },
        "grading_policy": {
            "grader": "openai",
            "judge_model": "gpt-5.4",
            "evidence_mode": "whole_files",
            "judge_effort": "high",
            "claim_scoring": "binary_0_1",
            "question_scoring": "weighted_claim_sum",
        },
        "analysis_policy": {
            "primary_estimand": "treatment_minus_control",
            "primary_metric": "expertise_lenient",
            "confidence_interval": (
                "paired_two_stage_question_then_rollout_percentile_95"
            ),
            "bootstrap_replicates": 20,
            "bootstrap_seed": 9,
            "multiplicity_policy": "single_preregistered_primary_no_adjustment",
        },
        "stopping_policy": {
            "population": "complete_manifest_grid",
            "interim_looks": 0,
            "stopping_rule": "no_outcome_dependent_stopping",
        },
    }
    preregistration_bytes = canonical_json_bytes(preregistration_document)
    preregistration_sha256 = sha256_bytes(preregistration_bytes)
    spec = {
        "schema_version": 1,
        "run_id": run_id,
        "task": "fake",
        "purpose": "confirmatory",
        "claim_ready": True,
        "harness": "native-react",
        "model": "model",
        "model_revision": "a" * 40,
        "sampling": {"temperature": 1.0},
        "master_seed": 19,
        "seed_policy": {
            "algorithm": "sha256-canonical-json-mod-2147483647",
            "namespace": "native-react",
            "seed_group": "paired-study",
            "ordered_parts": [
                "master_seed",
                "namespace",
                "seed_group",
                "task",
                "qid",
                "budget",
                "rollout",
            ],
            "episode_seeds": seeds,
        },
        "budgets": report.BUDGET_ORDER,
        "rollouts": ROLLOUTS,
        "questions": [{"id": qid, "sha256": sha256_json({"id": qid})} for qid in QIDS],
        "question_bundle_sha256": "d" * 64,
        "prompt_policy": {
            "note_prefix_template": template,
            "presented_prompt_sha256": prompt_hashes,
        },
        "expected_episodes": expected,
        "failure_policy": {
            "model_no_answer": "intention-to-run_zero",
            "infrastructure_error": "invalid_until_retried",
            "forced_short": "invalid_until_retried",
        },
        "corpus": {"name": "fake", "commit": "e" * 40},
        "source": {
            "git_commit": "f" * 40,
            "tree_sha256": "f" * 64,
            "dirty": False,
        },
        "environment": environment,
        "note": note,
        "preregistration": {
            "schema_version": 1,
            "status": "bound",
            "role": "treatment" if treatment else "control",
            "source_path": "preregistrations/paired-study-r1.json",
            "sha256": preregistration_sha256,
            "bytes": len(preregistration_bytes),
            "snapshot": f"inputs/preregistration-{preregistration_sha256}.json",
            "executed_source_commit": "f" * 40,
            "document": preregistration_document,
        },
        "extra": {
            "model_revision": "a" * 40,
            "expected_response_model": "generation-revision",
        },
    }
    spec["environment_contract"] = environment_contract_record(environment)
    return spec


def loaded_arm(run_id: str, *, treatment: bool, score_offset: int) -> compare.LoadedArm:
    spec = run_spec(run_id, treatment=treatment)
    grades = population(score_offset)
    records = []
    for budget in report.BUDGET_ORDER:
        for grade in grades[budget]:
            key = f"{budget}-r{grade['rollout']}-{grade['qid']}"
            records.append({
                "task": "fake",
                "qid": grade["qid"],
                "budget": budget,
                "rollout": grade["rollout"],
                "episode_path": (
                    f"runs/{run_id}/fake/{budget}/r{grade['rollout']}/"
                    f"{grade['qid']}.json"
                ),
                "episode_sha256": sha256_bytes(f"episode-{run_id}-{key}".encode()),
                "grade_path": (
                    f"grades/{run_id}/judge/fake/{budget}/r{grade['rollout']}/"
                    f"{grade['qid']}.json"
                ),
                "grade_sha256": sha256_bytes(f"grade-{run_id}-{key}".encode()),
            })
    grading_config = {
        "grade_schema_version": grade_module.GRADE_SCHEMA_VERSION,
        "judge_requested_model": "gpt-5.4",
        "judge_base_url": grade_module.CANONICAL_OPENAI_BASE_URL,
        "judge_response_models": ["judge-revision"],
        "judge_system_fingerprint_scope": "accepted_final_attempts_only",
        "judge_system_fingerprints": ["judge-fingerprint"],
        "accepted_judge_system_fingerprint_by_episode": {
            f"{budget}/r{grade['rollout']}/{grade['qid']}.json": "judge-fingerprint"
            for budget in report.BUDGET_ORDER
            for grade in grades[budget]
        },
        "missing_judge_system_fingerprint_calls": 0,
        "whole_files": True,
        "judge_effort": "high",
        "grading_spec_sha256_by_question": {qid: "2" * 64 for qid in QIDS},
    }
    audit = {
        "run_manifest": {
            "path": f"runs/{run_id}/fake/manifest.json",
            "sha256": "3" * 64,
            "spec_sha256": sha256_json(spec),
            "spec": spec,
        },
        "grading_manifest": {
            "sha256": sha256_json(grading_config),
            "config": grading_config,
        },
        "generation_runtime": {
            "response_models": ["generation-revision"],
            "system_fingerprints": ["generation-fingerprint"],
            "missing_system_fingerprint_calls": 0,
        },
        "note_provenance": {
            "construction_manifest_sha256": "c" * 64 if treatment else None,
            "study_id": "study-a" if treatment else None,
            "method": "method-a" if treatment else None,
            "manifest_type": None,
        },
        "failed_attempts": {
            "count": 1 if treatment else 0,
            "sha256": sha256_json([{"attempt": 1}] if treatment else []),
            "artifacts": [{"attempt": 1}] if treatment else [],
        },
        "failed_judge_audits": {
            "count": 0,
            "sha256": sha256_json([]),
            "artifacts": [],
        },
        "population": records,
        "population_sha256": sha256_json(records),
    }
    usage = {
        "available": treatment,
        "note_present": treatment,
        "accounting_status": (
            "validated_construction_manifest_metadata"
            if treatment
            else "not_applicable"
        ),
        "metadata": {"study_generated_tokens": 100} if treatment else {},
    }
    return compare.LoadedArm(
        report_path=Path(f"/tmp/report-{run_id}.json"),
        report_sha256=sha256_bytes(run_id.encode()),
        population=grades,
        audit=audit,
        aggregate=report.aggregate_population(grades),
        study_usage=usage,
        generation_retries=[],
        judge_retries=[],
    )


def set_accepted_judge_fingerprints(
    arm: compare.LoadedArm, values: dict[str, str | None]
) -> None:
    config = arm.audit["grading_manifest"]["config"]
    config["accepted_judge_system_fingerprint_by_episode"] = values
    config["judge_system_fingerprints"] = sorted(
        {value for value in values.values() if value is not None}
    )
    config["missing_judge_system_fingerprint_calls"] = sum(
        value is None for value in values.values()
    )
    arm.audit["grading_manifest"]["sha256"] = sha256_json(config)


class PairContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.control = loaded_arm("control-a", treatment=False, score_offset=0)
        self.treatment = loaded_arm("treatment-a", treatment=True, score_offset=10)

    def test_exact_note_only_pair_and_point_delta(self) -> None:
        intervention = compare.validate_pair(
            self.control,
            self.treatment,
            intervention_description="Add the preregistered study note.",
        )
        self.assertEqual(intervention["seed_pairing"]["seed_group"], "paired-study")
        self.assertEqual(
            intervention["judge_revision_verification"],
            "matched_complete_accepted_fingerprints_by_paired_cell",
        )
        self.assertEqual(
            intervention["generation_revision_verification"],
            "matched_complete_provider_fingerprint_set",
        )
        self.assertIn(
            "/environment/slurm_job_id",
            intervention["observed_environment_nuisance_leaf_paths"],
        )
        self.assertTrue(intervention["observed_intervention_leaf_paths"])
        point = compare.point_estimates(self.control, self.treatment)
        for budget in report.BUDGET_ORDER:
            self.assertEqual(
                point["budgets"][budget]["lenient"]["treatment_minus_control"],
                10,
            )

    def test_preregistration_roles_and_intervention_are_exact(self) -> None:
        with self.assertRaisesRegex(
            compare.ComparisonIntegrityError, "intervention differs"
        ):
            compare.validate_pair(
                self.control,
                self.treatment,
                intervention_description="A post-hoc description.",
            )
        drifted = deepcopy(self.treatment)
        drifted.audit["run_manifest"]["spec"]["preregistration"]["role"] = "control"
        with self.assertRaisesRegex(
            compare.ComparisonIntegrityError, "bound preregistration"
        ):
            compare.validate_pair(
                self.control, drifted, intervention_description=INTERVENTION
            )

    def test_seed_or_undisclosed_sampling_drift_is_fatal(self) -> None:
        drifted = deepcopy(self.treatment)
        drifted.audit["run_manifest"]["spec"]["seed_policy"]["seed_group"] = "other"
        with self.assertRaises(compare.ComparisonIntegrityError):
            compare.validate_pair(
                self.control, drifted, intervention_description=INTERVENTION
            )

        drifted = loaded_arm("treatment-a", treatment=True, score_offset=10)
        drifted.audit["run_manifest"]["spec"]["sampling"]["temperature"] = 0.5
        with self.assertRaises(compare.ComparisonIntegrityError):
            compare.validate_pair(
                self.control, drifted, intervention_description=INTERVENTION
            )

    def test_substantive_environment_drift_is_fatal(self) -> None:
        drifted = loaded_arm("treatment-a", treatment=True, score_offset=10)
        drifted.audit["run_manifest"]["spec"]["environment"]["allocation"][
            "inventory"
        ]["gpus"][0]["name"] = "different-gpu"
        with self.assertRaisesRegex(
            compare.ComparisonIntegrityError, "outside the disclosed note intervention"
        ):
            compare.validate_pair(
                self.control, drifted, intervention_description=INTERVENTION
            )

        drifted = loaded_arm("treatment-a", treatment=True, score_offset=10)
        drifted.audit["run_manifest"]["spec"]["environment"]["model_cache"][
            "sha256"
        ] = "9" * 64
        with self.assertRaises(compare.ComparisonIntegrityError):
            compare.validate_pair(
                self.control, drifted, intervention_description=INTERVENTION
            )

    def test_provider_model_heterogeneity_is_fatal(self) -> None:
        drifted = loaded_arm("treatment-a", treatment=True, score_offset=10)
        drifted.audit["generation_runtime"]["response_models"].append("other-model")
        with self.assertRaises(compare.ComparisonIntegrityError):
            compare.validate_pair(
                self.control, drifted, intervention_description=INTERVENTION
            )

    def test_paired_accepted_judge_fingerprint_mismatch_is_fatal(self) -> None:
        control = loaded_arm("control-a", treatment=False, score_offset=0)
        treatment = loaded_arm("treatment-a", treatment=True, score_offset=10)
        control_values = deepcopy(
            control.audit["grading_manifest"]["config"][
                "accepted_judge_system_fingerprint_by_episode"
            ]
        )
        treatment_values = deepcopy(control_values)
        keys = sorted(control_values)
        control_values[keys[0]] = "other-fingerprint"
        treatment_values[keys[1]] = "other-fingerprint"
        set_accepted_judge_fingerprints(control, control_values)
        set_accepted_judge_fingerprints(treatment, treatment_values)
        with self.assertRaisesRegex(
            compare.ComparisonIntegrityError,
            "paired accepted judge fingerprints differ",
        ):
            compare.validate_pair(
                control, treatment, intervention_description=INTERVENTION
            )

    def test_missing_accepted_judge_fingerprint_is_diagnostic(self) -> None:
        control = loaded_arm("control-a", treatment=False, score_offset=0)
        treatment = loaded_arm("treatment-a", treatment=True, score_offset=10)
        for missing, arm in ((3, control), (4, treatment)):
            values = deepcopy(
                arm.audit["grading_manifest"]["config"][
                    "accepted_judge_system_fingerprint_by_episode"
                ]
            )
            for key in sorted(values)[:missing]:
                values[key] = None
            set_accepted_judge_fingerprints(arm, values)
        control.audit["generation_runtime"]["missing_system_fingerprint_calls"] = 2
        treatment.audit["generation_runtime"]["missing_system_fingerprint_calls"] = 5
        intervention = compare.validate_pair(
            control, treatment, intervention_description=INTERVENTION
        )
        self.assertEqual(
            intervention["judge_revision_verification"],
            "accepted_provider_fingerprint_incomplete_and_disclosed",
        )
        self.assertEqual(
            intervention["provider_fingerprint_disclosure"]["judge"],
            {
                "control_system_fingerprints": ["judge-fingerprint"],
                "treatment_system_fingerprints": ["judge-fingerprint"],
                "control_missing_calls": 3,
                "treatment_missing_calls": 4,
            },
        )
        artifact = compare.build_comparison(
            control,
            treatment,
            intervention_description=INTERVENTION,
            bootstrap_replicates=20,
            bootstrap_seed=9,
        )
        self.assertFalse(artifact["claim_ready"])
        forged = deepcopy(artifact)
        forged["claim_ready"] = True
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(
            compare.ComparisonIntegrityError, "claim-ready status disagrees"
        ):
            compare.write_comparison(forged, output_root=directory)

        with self.assertRaisesRegex(
            compare.ComparisonIntegrityError, "bootstrap configuration"
        ):
            compare.build_comparison(
                control,
                treatment,
                intervention_description=INTERVENTION,
                bootstrap_replicates=19,
                bootstrap_seed=9,
            )

    def test_missing_generation_fingerprint_is_diagnostic(self) -> None:
        control = loaded_arm("control-a", treatment=False, score_offset=0)
        treatment = loaded_arm("treatment-a", treatment=True, score_offset=10)
        control.audit["generation_runtime"]["missing_system_fingerprint_calls"] = 1
        treatment.audit["generation_runtime"]["missing_system_fingerprint_calls"] = 1
        artifact = compare.build_comparison(
            control,
            treatment,
            intervention_description=INTERVENTION,
            bootstrap_replicates=20,
            bootstrap_seed=9,
        )
        self.assertEqual(
            artifact["intervention"]["generation_revision_verification"],
            "provider_fingerprint_incomplete_and_disclosed",
        )
        self.assertFalse(artifact["claim_ready"])

    def test_static_grading_contract_drift_is_fatal(self) -> None:
        drifted = loaded_arm("treatment-a", treatment=True, score_offset=10)
        config = drifted.audit["grading_manifest"]["config"]
        config["whole_files"] = False
        drifted.audit["grading_manifest"]["sha256"] = sha256_json(config)
        with self.assertRaisesRegex(
            compare.ComparisonIntegrityError, "grading differs"
        ):
            compare.validate_pair(
                self.control, drifted, intervention_description=INTERVENTION
            )

    def test_no_answer_must_remain_zero(self) -> None:
        invalid = loaded_arm("treatment-a", treatment=True, score_offset=10)
        invalid.population["direct"][0].update(episode_status="no_answer", lenient=1)
        with self.assertRaises(compare.ComparisonIntegrityError):
            compare.validate_pair(
                self.control, invalid, intervention_description=INTERVENTION
            )


class PairedBootstrapTests(unittest.TestCase):
    def test_identical_rollout_draws_preserve_a_constant_paired_effect(self) -> None:
        result = compare.paired_bootstrap(
            population(0), population(10), replicates=200, seed=73
        )
        for budget in report.BUDGET_ORDER:
            interval = result["budgets"][budget]["lenient"]
            self.assertEqual(interval, {"mean": 10.0, "lower_95": 10.0, "upper_95": 10.0})

    def test_bootstrap_is_reproducible(self) -> None:
        first = compare.paired_bootstrap(
            population(0), population(10), replicates=25, seed=4
        )
        second = compare.paired_bootstrap(
            population(0), population(10), replicates=25, seed=4
        )
        self.assertEqual(first, second)


class SourceReportTests(unittest.TestCase):
    def _fixture(self, root: Path, *, aggregate_drift: bool = False) -> tuple[Path, dict, dict]:
        run_id = "source-a"
        task = "fake"
        run_root = root / "runs" / run_id
        grade_root = root / "grades" / run_id / "judge"
        manifest_path = run_root / task / "manifest.json"
        manifest_path.parent.mkdir(parents=True)
        manifest_path.write_text("{}\n", encoding="utf-8")

        grades = {budget: [] for budget in report.BUDGET_ORDER}
        records = []
        for index, budget in enumerate(report.BUDGET_ORDER):
            grade = {
                "task": task,
                "qid": "q1",
                "budget": budget,
                "rollout": 0,
                "episode_status": "ok",
                "lenient": 10 + index,
                "strict": 10 + index,
                "cores_ok": True,
                "compile_check": {"compile_ok": True},
                "gen_tokens": 4_000 * (index + 1),
                "episode_sha256": "a" * 64,
                "judge_attempts": ([{
                    "attempt": 1,
                    "accepted": False,
                    "response_id": "judge-response",
                    "request_id": "judge-request",
                    "response_model": "judge-revision",
                    "system_fingerprint": "rejected-fingerprint",
                    "system_fingerprint_status": "available",
                    "system_fingerprint_observation": None,
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 2,
                        "total_tokens": 12,
                        "provider_usage": {},
                    },
                    "content_sha256": "b" * 64,
                    "content_bytes": 2,
                    "validation_error": {
                        "type": "GradeIntegrityError",
                        "message": "invalid verdict",
                    },
                }] if index == 0 else []),
            }
            grades[budget].append(grade)
            episode_path = run_root / task / budget / "r0" / "q1.json"
            grade_path = grade_root / task / budget / "r0" / "q1.json"
            episode_path.parent.mkdir(parents=True)
            grade_path.parent.mkdir(parents=True)
            episode = {
                "request_attempts": ([{
                    "logical_call": 0,
                    "attempt": 1,
                    "status": "transport_error",
                    "request_sha256": "c" * 64,
                    "error_type": "TimeoutError",
                    "error": "timed out",
                    "usage": "unknown",
                }] if index == 0 else []),
            }
            episode_path.write_bytes(canonical_json_bytes(episode))
            grade_path.write_bytes(canonical_json_bytes(grade))
            records.append({
                "task": task,
                "qid": "q1",
                "budget": budget,
                "rollout": 0,
                "episode_path": episode_path.relative_to(root).as_posix(),
                "episode_sha256": sha256_bytes(episode_path.read_bytes()),
                "grade_path": grade_path.relative_to(root).as_posix(),
                "grade_sha256": sha256_bytes(grade_path.read_bytes()),
            })
        spec = {"run_id": run_id, "task": task, "rollouts": 1, "note": None}
        grading_config = {
            "judge_requested_model": "judge",
            "judge_base_url": TEST_JUDGE_BASE_URL,
            "judge_response_models": ["judge-revision"],
            "judge_system_fingerprint_scope": "accepted_final_attempts_only",
            "judge_system_fingerprints": ["judge-fingerprint"],
            "accepted_judge_system_fingerprint_by_episode": {
                f"{budget}/r0/q1.json": "judge-fingerprint"
                for budget in report.BUDGET_ORDER
            },
            "missing_judge_system_fingerprint_calls": 0,
            "whole_files": True,
            "judge_effort": "",
        }
        audit = {
            "run_manifest": {
                "path": manifest_path.relative_to(root).as_posix(),
                "sha256": sha256_bytes(manifest_path.read_bytes()),
                "spec_sha256": sha256_json(spec),
                "spec": spec,
            },
            "generation_runtime": {
                "response_models": ["generation-revision"],
                "system_fingerprints": [],
                "missing_system_fingerprint_calls": 4,
            },
            "note_provenance": {
                "construction_manifest_sha256": None,
                "study_id": None,
                "method": None,
                "manifest_type": None,
            },
            "failed_attempts": {"count": 0, "sha256": sha256_json([]), "artifacts": []},
            "failed_judge_audits": {
                "count": 0,
                "sha256": sha256_json([]),
                "artifacts": [],
            },
            "grading_manifest": {
                "sha256": sha256_json(grading_config),
                "config": grading_config,
            },
            "population": records,
            "population_sha256": sha256_json(records),
        }
        aggregate = report.aggregate_population(grades)
        if aggregate_drift:
            aggregate = deepcopy(aggregate)
            aggregate["budgets"]["direct"]["lenient"] += 1
        artifact = {
            "report_schema_version": report.REPORT_SCHEMA_VERSION,
            "claim_ready": True,
            "task": task,
            "run_id": run_id,
            "budget_order": report.BUDGET_ORDER,
            **audit,
            "aggregate": aggregate,
            "bootstrap": {"replicates": 0, "seed": 0, "results": None},
            "paper_comparison": None,
            "report_source": {
                "studybench/report.py": compare.file_sha256(Path(report.__file__).resolve())
            },
        }
        raw = canonical_json_bytes(artifact)
        report_path = root / "reports" / f"report-{sha256_bytes(raw)}.json"
        report_path.parent.mkdir()
        report_path.write_bytes(raw)
        return report_path, grades, audit

    def test_report_is_reloaded_before_it_is_trusted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, grades, audit = self._fixture(root)
            with (
                patch.object(compare, "ROOT", root),
                patch.object(report, "ROOT", root),
                patch.object(
                    report,
                    "load_complete_evaluation",
                    return_value=(grades, audit),
                ) as loader,
            ):
                arm = compare.load_source_report(path)
            self.assertEqual(arm.report_sha256, sha256_bytes(path.read_bytes()))
            self.assertEqual(arm.audit["population_sha256"], audit["population_sha256"])
            self.assertEqual(len(arm.generation_retries), 1)
            self.assertEqual(len(arm.judge_retries), 1)
            self.assertEqual(
                arm.judge_retries[0]["system_fingerprint"],
                "rejected-fingerprint",
            )
            self.assertEqual(
                arm.judge_retries[0]["system_fingerprint_status"], "available"
            )
            loader.assert_called_once()

    def test_stale_report_aggregate_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, grades, audit = self._fixture(root, aggregate_drift=True)
            with (
                patch.object(compare, "ROOT", root),
                patch.object(report, "ROOT", root),
                patch.object(
                    report,
                    "load_complete_evaluation",
                    return_value=(grades, audit),
                ),
                self.assertRaises(compare.ComparisonIntegrityError),
            ):
                compare.load_source_report(path)

    def test_study_token_metadata_is_bound_and_totals_are_checked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            manifest_path = run_root / "inputs" / "construction.json"
            manifest_path.parent.mkdir()
            construction = {
                "study_id": "study-a",
                "note_sha256": "a" * 64,
                "config": {"method": "forced-50-cheatsheet"},
                "study_prompt_tokens": 80,
                "study_generated_tokens": 20,
                "study_total_tokens": 100,
            }
            manifest_path.write_bytes(canonical_json_bytes(construction))
            spec = {
                "note": {
                    "sha256": "a" * 64,
                    "construction_manifest": {
                        "snapshot": "inputs/construction.json",
                        "sha256": sha256_bytes(manifest_path.read_bytes()),
                    },
                }
            }
            metadata = compare._study_usage(spec, run_root)
            self.assertEqual(metadata["metadata"]["study_generated_tokens"], 20)

            construction["study_total_tokens"] = 99
            manifest_path.write_bytes(canonical_json_bytes(construction))
            spec["note"]["construction_manifest"]["sha256"] = sha256_bytes(
                manifest_path.read_bytes()
            )
            with self.assertRaises(compare.ComparisonIntegrityError):
                compare._study_usage(spec, run_root)


class ComparisonArtifactTests(unittest.TestCase):
    def test_recorded_inputs_reject_intermediate_symlink_components(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "real"
            real.mkdir()
            (real / "report.json").write_text("{}\n", encoding="utf-8")
            linked = root / "linked"
            linked.symlink_to(real, target_is_directory=True)
            with self.assertRaises(compare.ComparisonIntegrityError):
                compare._recorded_file(
                    str(linked / "report.json"), label="test report"
                )

    def test_artifact_is_deterministic_content_addressed_and_discloses_attempts(self) -> None:
        control = loaded_arm("control-a", treatment=False, score_offset=0)
        treatment = loaded_arm("treatment-a", treatment=True, score_offset=10)
        artifact = compare.build_comparison(
            control,
            treatment,
            intervention_description="Add the preregistered study note.",
            bootstrap_replicates=20,
            bootstrap_seed=9,
        )
        self.assertTrue(artifact["claim_ready"])
        self.assertEqual(
            artifact["sources"]["treatment"]["failed_generation_attempts"]["count"],
            1,
        )
        self.assertTrue(artifact["sources"]["treatment"]["study_usage"]["available"])
        with tempfile.TemporaryDirectory() as directory, patch.object(
            compare, "ROOT", Path(directory)
        ), patch.object(
            compare,
            "load_source_report",
            side_effect=[control, treatment, control, treatment, control, treatment],
        ):
            first = compare.write_comparison(artifact)
            before = first.read_bytes()
            second = compare.write_comparison(artifact)
            self.assertEqual(first, second)
            self.assertEqual(before, second.read_bytes())
            self.assertEqual(first.stem.removeprefix("comparison-"), sha256_bytes(before))

            forged = deepcopy(artifact)
            forged["point_estimates"]["expertise"]["expertise_lenient"][
                "treatment_minus_control"
            ] += 1
            with self.assertRaisesRegex(
                compare.ComparisonIntegrityError,
                "does not match independent recomputation",
            ):
                compare.write_comparison(forged)


if __name__ == "__main__":
    unittest.main()
