from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import subprocess
import tempfile
import unittest

from studybench.integrity import canonical_json_bytes, sha256_bytes
from studybench.preregistration import (
    PREREGISTRATION_SCHEMA_VERSION,
    RUN_FAILURE_POLICY,
    PreregistrationError,
    bind_preregistration,
    load_preregistration,
    revalidate_run_preregistration,
    validate_preregistration,
)


def preregistration_document(source_commit: str = "b" * 40) -> dict[str, object]:
    return {
        "schema_version": PREREGISTRATION_SCHEMA_VERSION,
        "preregistration_id": "study-method-r1",
        "hypothesis": "The treatment improves retained repository expertise.",
        "intervention": "Provide the preregistered treatment study note.",
        "task": "dspy",
        "corpus_commit": "a" * 40,
        "source_commit": source_commit,
        "question_bundle_sha256": "d" * 64,
        "arms": {
            "control": {"run_id": "control-r1", "note_sha256": None},
            "treatment": {"run_id": "treatment-r1", "note_sha256": "e" * 64},
        },
        "evaluation": {
            "harness": "dspy.ReAct",
            "model": "openai/Qwen/Qwen3.5-9B",
            "model_revision": "c" * 40,
            "sampling": {"temperature": 0.0, "top_p": 0.95},
            "master_seed": 44001,
            "seed_namespace": "dspy-react",
            "seed_group": "paired-r1",
            "budgets": ["direct", "k5", "k20", "k20f"],
            "rollouts": 6,
        },
        "failure_policy": dict(RUN_FAILURE_POLICY),
        "grading_policy": {
            "grader": "openai",
            "judge_model": "gpt-5.4",
            "evidence_mode": "whole_files",
            "judge_effort": "",
            "claim_scoring": "binary_0_1",
            "question_scoring": "weighted_claim_sum",
        },
        "analysis_policy": {
            "primary_estimand": "treatment_minus_control",
            "primary_metric": "expertise_lenient",
            "confidence_interval": (
                "paired_two_stage_question_then_rollout_percentile_95"
            ),
            "bootstrap_replicates": 10_000,
            "bootstrap_seed": 45001,
            "multiplicity_policy": "single_preregistered_primary_no_adjustment",
        },
        "stopping_policy": {
            "population": "complete_manifest_grid",
            "interim_looks": 0,
            "stopping_rule": "no_outcome_dependent_stopping",
        },
    }


def git(root: Path, *arguments: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        text=True,
        check=True,
    )
    return process.stdout.strip()


def committed_preregistration(root: Path) -> tuple[Path, dict[str, object], str]:
    git(root, "init", "-q")
    git(root, "config", "user.email", "research@example.test")
    git(root, "config", "user.name", "Research Test")
    (root / "implementation.py").write_text("VALUE = 1\n", encoding="utf-8")
    git(root, "add", "implementation.py")
    git(root, "commit", "-q", "-m", "freeze implementation")
    source_commit = git(root, "rev-parse", "HEAD")
    document = preregistration_document(source_commit)
    directory = root / "preregistrations"
    directory.mkdir()
    path = directory / "study-method-r1.json"
    path.write_bytes(canonical_json_bytes(document))
    git(root, "add", path.relative_to(root).as_posix())
    git(root, "commit", "-q", "-m", "preregister study")
    return path, document, git(root, "rev-parse", "HEAD")


class PreregistrationTests(unittest.TestCase):
    def test_schema_rejects_boolean_integer_identities(self) -> None:
        document = preregistration_document()
        validate_preregistration(document)
        paths = (
            ("schema_version",),
            ("evaluation", "master_seed"),
            ("evaluation", "rollouts"),
            ("analysis_policy", "bootstrap_replicates"),
            ("analysis_policy", "bootstrap_seed"),
            ("stopping_policy", "interim_looks"),
        )
        for keys in paths:
            invalid = deepcopy(document)
            target = invalid
            for key in keys[:-1]:
                target = target[key]
            target[keys[-1]] = True
            with self.subTest(path=".".join(keys)), self.assertRaises(
                PreregistrationError
            ):
                validate_preregistration(invalid)

    def test_schema_rejects_unimplemented_analysis_and_grader_contracts(self) -> None:
        for mutate in (
            lambda value: value["evaluation"].update(
                {"budgets": ["direct", "k5"]}
            ),
            lambda value: value["grading_policy"].update(
                {"judge_model": "different-judge"}
            ),
            lambda value: value["analysis_policy"].update(
                {"primary_metric": "post_hoc_metric"}
            ),
            lambda value: value["analysis_policy"].update(
                {"multiplicity_policy": "unspecified"}
            ),
        ):
            invalid = preregistration_document()
            mutate(invalid)
            with self.assertRaises(PreregistrationError):
                validate_preregistration(invalid)

    def test_loader_requires_canonical_clean_committed_preregistration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, document, head = committed_preregistration(root)
            loaded = load_preregistration(path, root=root)
            self.assertEqual(loaded.document, document)
            self.assertEqual(loaded.head_commit, head)
            self.assertEqual(loaded.sha256, sha256_bytes(path.read_bytes()))

            path.write_bytes(path.read_bytes() + b" ")
            with self.assertRaises(PreregistrationError):
                load_preregistration(path, root=root)

    def test_loader_rejects_implementation_changes_after_preregistration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _, _ = committed_preregistration(root)
            (root / "implementation.py").write_text("VALUE = 2\n", encoding="utf-8")
            git(root, "add", "implementation.py")
            git(root, "commit", "-q", "--amend", "--no-edit")
            with self.assertRaisesRegex(
                PreregistrationError, r"may only add direct preregistrations/\*\.json"
            ):
                load_preregistration(path, root=root)

    def test_loader_rejects_nonpreregistration_additions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _, _ = committed_preregistration(root)
            (root / "analysis.py").write_text("RESULT = 1\n", encoding="utf-8")
            git(root, "add", "analysis.py")
            git(root, "commit", "-q", "--amend", "--no-edit")
            with self.assertRaisesRegex(
                PreregistrationError,
                r"may only add direct preregistrations/\*\.json files; "
                r"found A analysis\.py",
            ):
                load_preregistration(path, root=root)

    def test_loader_rejects_a_committed_preregistration_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, document, _ = committed_preregistration(root)
            document["hypothesis"] = "A rewritten hypothesis."
            path.write_bytes(canonical_json_bytes(document))
            git(root, "add", path.relative_to(root).as_posix())
            git(root, "commit", "-q", "-m", "rewrite preregistration")
            with self.assertRaisesRegex(PreregistrationError, "single-parent direct child"):
                load_preregistration(path, root=root)

    def test_loader_rejects_a_rewritten_and_renamed_preregistration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, document, preregistration_commit = committed_preregistration(root)
            document["source_commit"] = preregistration_commit
            document["preregistration_id"] = "study-method-r2"
            document["hypothesis"] = "A rewritten and renamed hypothesis."
            renamed = path.with_name("study-method-r2.json")
            renamed.write_bytes(canonical_json_bytes(document))
            path.unlink()
            git(root, "add", "--all", "preregistrations")
            git(root, "commit", "-q", "-m", "rewrite and rename preregistration")

            with self.assertRaisesRegex(
                PreregistrationError,
                r"may only add direct preregistrations/\*\.json files; found D ",
            ):
                load_preregistration(renamed, root=root)

    def test_runtime_binding_and_offline_snapshot_revalidation_are_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, document, head = committed_preregistration(root)
            evaluation = document["evaluation"]
            loaded = bind_preregistration(
                path,
                role="control",
                run_id="control-r1",
                task="dspy",
                corpus_commit="a" * 40,
                source_head_commit=head,
                question_bundle_sha256="d" * 64,
                harness="dspy.ReAct",
                model="openai/Qwen/Qwen3.5-9B",
                model_revision="c" * 40,
                sampling=evaluation["sampling"],
                master_seed=44001,
                seed_namespace="dspy-react",
                seed_group="paired-r1",
                budgets=["direct", "k5", "k20", "k20f"],
                rollouts=6,
                failure_policy=RUN_FAILURE_POLICY,
                note_sha256=None,
                root=root,
            )
            run_root = root / "run"
            snapshot = run_root / "inputs" / f"preregistration-{loaded.sha256}.json"
            snapshot.parent.mkdir(parents=True)
            snapshot.write_bytes(loaded.data)
            spec = {
                "run_id": "control-r1",
                "task": "dspy",
                "harness": "dspy.ReAct",
                "model": "openai/Qwen/Qwen3.5-9B",
                "model_revision": "c" * 40,
                "sampling": evaluation["sampling"],
                "master_seed": 44001,
                "seed_policy": {
                    "namespace": "dspy-react",
                    "seed_group": "paired-r1",
                },
                "budgets": ["direct", "k5", "k20", "k20f"],
                "rollouts": 6,
                "question_bundle_sha256": "d" * 64,
                "failure_policy": dict(RUN_FAILURE_POLICY),
                "corpus": {"commit": "a" * 40},
                "source": {"git_commit": head},
                "note": None,
                "extra": {
                    "model_revision": "c" * 40,
                    "expected_response_model": "Qwen/Qwen3.5-9B",
                },
                "preregistration": {
                    "schema_version": 1,
                    "status": "bound",
                    "role": "control",
                    "source_path": "preregistrations/study-method-r1.json",
                    "sha256": loaded.sha256,
                    "bytes": len(loaded.data),
                    "snapshot": f"inputs/preregistration-{loaded.sha256}.json",
                    "executed_source_commit": head,
                    "document": document,
                },
            }
            self.assertEqual(revalidate_run_preregistration(spec, run_root), document)

            wrong_type = deepcopy(spec)
            wrong_type["master_seed"] = True
            with self.assertRaises(PreregistrationError):
                revalidate_run_preregistration(wrong_type, run_root)
            divergent_model = deepcopy(spec)
            divergent_model["extra"]["model_revision"] = "f" * 40
            with self.assertRaisesRegex(PreregistrationError, "model identity"):
                revalidate_run_preregistration(divergent_model, run_root)
            snapshot.write_bytes(b"{}\n")
            with self.assertRaisesRegex(PreregistrationError, "snapshot bytes changed"):
                revalidate_run_preregistration(spec, run_root)


if __name__ == "__main__":
    unittest.main()
