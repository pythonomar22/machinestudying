from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from copy import deepcopy
from unittest.mock import patch

from studybench.dataset import ROOT
from studybench.integrity import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    sha256_json,
    sha256_text,
    write_immutable_json,
)
from studybench.model_cache import ATTESTATION_POLICY as MODEL_CACHE_ATTESTATION_POLICY
from studybench.provenance import (
    MODEL_ID,
    MODEL_REVISION,
    VLLM_PYTHON_VERSION,
    VLLM_VERSION,
    _build_installed_distribution_inventory,
    _load_note,
    RunContext,
    corpus_record,
    environment_is_claim_ready,
    environment_record,
    environments_compatible,
    episode_identity,
    normalized_environment,
    prepare_run,
    source_record,
    validate_current_source,
    validate_id,
    validate_environment_snapshot,
    validate_resumable_episode,
    validate_local_server_urls,
    write_environment_snapshot,
    write_episode_result,
)


def claim_ready_environment(*, include_dspy: bool = True) -> dict[str, object]:
    lock_path = ROOT / "scripts" / "vllm-requirements.lock"
    lock_bytes = lock_path.read_bytes()
    locked = [
        line.strip()
        for line in lock_bytes.decode("utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    runner_packages = [
        {"name": "openai", "version": "1.0"},
        {"name": "pydantic", "version": "2.0"},
    ]
    if include_dspy:
        runner_packages.append({"name": "dspy", "version": "3.0"})
    runner_packages.sort(key=lambda row: (row["name"], row["version"]))
    pyvenv_text = "home = /python\n"
    runner_python_version = VLLM_PYTHON_VERSION if include_dspy else "3.14.6"
    runner = {
        "python": {
            "version": runner_python_version,
            "implementation": "CPython",
            "executable": "/venv/bin/python",
            "resolved_executable": "/python/bin/python3.14",
            "executable_sha256": "1" * 64,
            "prefix": "/venv",
            "base_prefix": "/python",
            "pyvenv_cfg": {
                "path": "/venv/pyvenv.cfg",
                "sha256": sha256_text(pyvenv_text),
                "bytes": len(pyvenv_text.encode("utf-8")),
                "text": pyvenv_text,
            },
        },
        "packages": runner_packages,
        "packages_sha256": sha256_json(runner_packages),
    }
    installed_distributions = []
    for requirement in locked:
        name, version = requirement.split("==", 1)
        record_path = (
            f"lib/python3.12/site-packages/{name.replace('-', '_')}-{version}.dist-info/"
            "RECORD"
        )
        record_bytes = (requirement + "\n").encode("utf-8")
        files = [{
            "path": record_path,
            "bytes": len(record_bytes),
            "sha256": sha256_bytes(record_bytes),
        }]
        installed_distributions.append({
            "name": name,
            "version": version,
            "record_path": record_path,
            "record_sha256": files[0]["sha256"],
            "file_count": len(files),
            "total_bytes": len(record_bytes),
            "files": files,
            "tree_sha256": sha256_json(files),
        })
    installed_code = {
        "schema_version": 1,
        "python_version": VLLM_PYTHON_VERSION,
        "prefix": "/vllm",
        "distribution_count": len(installed_distributions),
        "file_count": sum(row["file_count"] for row in installed_distributions),
        "total_bytes": sum(row["total_bytes"] for row in installed_distributions),
        "distributions": installed_distributions,
        "tree_sha256": sha256_json(installed_distributions),
    }
    vllm_bytes = canonical_json_bytes(installed_code)
    vllm_environment = {
        "path": "logs/test.packages.txt",
        "sha256": sha256_bytes(vllm_bytes),
        "bytes": len(vllm_bytes),
        "inventory": installed_code,
    }
    runtime_inventory = {
        "schema_version": 1,
        "python": {
            "version": VLLM_PYTHON_VERSION,
            "implementation": "CPython",
            "executable": "/vllm/bin/python",
            "resolved_executable": "/python/bin/python3.12",
            "executable_sha256": "2" * 64,
            "prefix": "/vllm",
            "base_prefix": "/python",
        },
        "vllm_entrypoint": {"path": "/vllm/bin/vllm", "sha256": "3" * 64},
        "cuda_toolkit": {
            "cuda_home": "/usr/local/cuda",
            "nvcc": {
                "path": "/usr/local/cuda/bin/nvcc",
                "resolved_path": "/usr/local/cuda/bin/nvcc",
                "sha256": "7" * 64,
                "version_text": "Cuda compilation tools, release 13.0\n",
                "version_sha256": sha256_text(
                    "Cuda compilation tools, release 13.0\n"
                ),
            },
        },
        "torch": {"version": "2.9.0", "cuda_version": "13.0"},
        "package_inventory_sha256": vllm_environment["sha256"],
        "lock_sha256": sha256_bytes(lock_bytes),
    }
    model_files = [
        {
            "path": "config.json",
            "storage_path": "blobs/config",
            "bytes": 10,
            "sha256": "4" * 64,
        },
        {
            "path": "model.safetensors",
            "storage_path": "blobs/weights",
            "bytes": 20,
            "sha256": "5" * 64,
        },
    ]
    model_inventory = {
        "schema_version": 1,
        "attestation_policy": MODEL_CACHE_ATTESTATION_POLICY,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "cache_root": "/cache/hub",
        "snapshot": f"/cache/hub/snapshots/{MODEL_REVISION}",
        "file_count": len(model_files),
        "total_bytes": 30,
        "files": model_files,
        "tree_sha256": sha256_json(model_files),
    }
    allocation_inventory = {
        "schema_version": 1,
        "hostname": "compute.example",
        "cuda_visible_devices": "0",
        "gpu_count": 1,
        "gpus": [{
            "cuda_identifier": "0",
            "uuid": "GPU-test",
            "name": "test-gpu",
            "memory_mib": 48_000,
            "driver_version": "test-driver",
        }],
        "slurm": {
            "job_id": "123",
            "job_gpus": "0",
            "step_gpus": None,
            "job_nodelist": "compute",
            "node_id": "0",
        },
    }

    def snapshot(path: str, inventory: dict[str, object]) -> dict[str, object]:
        payload = canonical_json_bytes(inventory)
        return {
            "path": path,
            "sha256": sha256_bytes(payload),
            "bytes": len(payload),
            "inventory": inventory,
        }

    selected = {
        "dspy": "3.0" if include_dspy else None,
        "openai": "1.0",
        "pydantic": "2.0",
    }
    project_root = ROOT / "corpora" / "dspy" if include_dspy else ROOT

    def source_record(path: Path) -> dict[str, object]:
        data = path.read_bytes()
        return {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256_bytes(data),
            "bytes": len(data),
        }

    arguments = ["sync", "--project", str(project_root), "--frozen"]
    if include_dspy:
        arguments.append("--no-dev")
    arguments.append("--check")
    runner_lock = {
        "schema_version": 1,
        "kind": "dspy" if include_dspy else "main",
        "python_version": runner_python_version,
        "lock": source_record(project_root / "uv.lock"),
        "project": source_record(project_root / "pyproject.toml"),
        "uv": {
            "path": "/usr/bin/uv",
            "sha256": "8" * 64,
            "version": "uv 1.0.0",
        },
        "sync_check": {"status": "synchronized", "arguments": arguments},
        "dspy_corpus": (
            {"commit": "9cdb0aac28b2a04b064e40697ccd301872cf6a43", "dirty": False}
            if include_dspy
            else None
        ),
        "dspy_import": (
            {
                "version": "3.0",
                "origin": "/venv/lib/python/site-packages/dspy/__init__.py",
                "origin_sha256": "9" * 64,
            }
            if include_dspy
            else None
        ),
    }
    launch_id = "6" * 64
    return {
        "python": runner_python_version,
        "implementation": "CPython",
        "machine": "x86_64",
        "platform": "test",
        "packages": selected,
        "runner": runner,
        "runner_lock": runner_lock,
        "gpu_models": ["test-gpu"],
        "nvidia_driver": ["test-driver"],
        "allocation": snapshot("logs/test.gpus.json", allocation_inventory),
        "vllm_version": VLLM_VERSION,
        "vllm_environment_sha256": vllm_environment["sha256"],
        "vllm_environment": vllm_environment,
        "vllm_runtime": snapshot("logs/test.vllm-runtime.json", runtime_inventory),
        "model_cache": snapshot("logs/test.model-cache.json", model_inventory),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tensor_parallel_size": "1",
        "visible_gpu_count": "1",
        "server_count": "1",
        "cuda_visible_devices": "0",
        "slurm_job_id": "123",
        "runner_allocation": {
            "slurm_job_id": "123",
            "slurm_job_gpus": "0",
            "slurm_step_gpus": None,
            "slurm_job_nodelist": "compute",
            "slurm_node_id": "0",
            "cuda_visible_devices": "0",
            "hostname": "compute.example",
        },
        "server_launch_id": launch_id,
        "vllm_api_key_sha256": launch_id,
        "inventory_errors": {},
    }


def reallocated_environment(environment: dict[str, object]) -> dict[str, object]:
    """Change only identities that legitimately vary in a fresh Slurm launch."""

    value = deepcopy(environment)
    for field in ("vllm_environment", "vllm_runtime", "model_cache"):
        value[field]["path"] = value[field]["path"].replace("test", "retry")
    inventory = value["allocation"]["inventory"]
    inventory["hostname"] = "retry.example"
    inventory["cuda_visible_devices"] = "7"
    inventory["gpus"][0]["cuda_identifier"] = "7"
    inventory["gpus"][0]["uuid"] = "GPU-retry"
    inventory["slurm"] = {
        "job_id": "456",
        "job_gpus": "7",
        "step_gpus": None,
        "job_nodelist": "retry",
        "node_id": "1",
    }
    allocation_bytes = canonical_json_bytes(inventory)
    value["allocation"].update({
        "path": "logs/retry.gpus.json",
        "sha256": sha256_bytes(allocation_bytes),
        "bytes": len(allocation_bytes),
    })
    value["cuda_visible_devices"] = "7"
    value["slurm_job_id"] = "456"
    value["runner_allocation"] = {
        "slurm_job_id": "456",
        "slurm_job_gpus": "7",
        "slurm_step_gpus": None,
        "slurm_job_nodelist": "retry",
        "slurm_node_id": "1",
        "cuda_visible_devices": "7",
        "hostname": "retry.example",
    }
    value["server_launch_id"] = "a" * 64
    value["vllm_api_key_sha256"] = "a" * 64
    return value


class ProvenanceTests(unittest.TestCase):
    @staticmethod
    def _initialize_source_repo(root: Path) -> None:
        (root / "studybench").mkdir()
        (root / "studybench" / "example.py").write_text(
            "VALUE = 1\n", encoding="utf-8"
        )
        (root / "tests" / "unit").mkdir(parents=True)
        (root / "tests" / "unit" / "test_example.py").write_text(
            "def test_example():\n    assert True\n", encoding="utf-8"
        )
        (root / "tests" / "fixtures").mkdir()
        (root / "tests" / "fixtures" / "case.json").write_text(
            '{"case": 1}\n', encoding="utf-8"
        )
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        subprocess.run(
            ["git", "-C", str(root), "config", "user.email", "test@example.com"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(root), "config", "user.name", "Test"], check=True
        )
        subprocess.run(["git", "-C", str(root), "add", "--all"], check=True)
        subprocess.run(
            ["git", "-C", str(root), "commit", "-q", "-m", "fixture"], check=True
        )

    def test_claim_ready_environment_rejects_inherited_proxy_or_loader_hooks(self) -> None:
        for variable in ("HTTP_PROXY", "PYTHONPATH", "LD_PRELOAD"):
            with self.subTest(variable=variable), patch.dict(
                os.environ, {variable: "unsafe"}, clear=True
            ), patch(
                "studybench.provenance._runner_environment_record",
                side_effect=ValueError("not needed for this preflight"),
            ):
                observed = environment_record()
                self.assertIn(
                    "runner_process_environment", observed["inventory_errors"]
                )
                self.assertFalse(environment_is_claim_ready(observed))

    def test_corpus_record_requires_the_strict_pinned_snapshot_validator(self) -> None:
        corpus = SimpleNamespace(
            name="fixture",
            commit="a" * 40,
            repo=Path("/unused"),
            roots=("src",),
            language="python",
            code_suffixes=(".py",),
        )
        with patch("studybench.provenance.validate_corpus_snapshot") as validate:
            record = corpus_record(corpus)
        validate.assert_called_once_with(corpus)
        self.assertEqual(record["commit"], corpus.commit)
        self.assertIs(record["dirty"], False)

    def test_installed_distribution_inventory_hashes_actual_record_files(self) -> None:
        class Distribution:
            metadata = {"Name": "Example_Package"}
            version = "1.2.3"

            def __init__(self, root: Path):
                self.root = root
                self.files = (
                    PurePosixPath("lib/python3.12/site-packages/example/__init__.py"),
                    PurePosixPath(
                        "lib/python3.12/site-packages/"
                        "example_package-1.2.3.dist-info/RECORD"
                    ),
                )

            def locate_file(self, relative: PurePosixPath) -> Path:
                return self.root.joinpath(*relative.parts)

            def read_text(self, filename: str) -> str | None:
                if filename != "RECORD":
                    return None
                return (
                    self.root
                    / "lib/python3.12/site-packages/"
                    "example_package-1.2.3.dist-info/RECORD"
                ).read_text(encoding="utf-8")

        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory).resolve()
            package = prefix / "lib/python3.12/site-packages/example/__init__.py"
            package.parent.mkdir(parents=True)
            package.write_text("VALUE = 1\n", encoding="utf-8")
            record = (
                prefix
                / "lib/python3.12/site-packages/"
                "example_package-1.2.3.dist-info/RECORD"
            )
            record.parent.mkdir()
            record.write_text("example/__init__.py,,\n", encoding="utf-8")
            distribution = Distribution(prefix)

            first = _build_installed_distribution_inventory(
                [distribution], prefix=prefix, python_version="3.12.11"
            )
            package.write_text("VALUE = 2\n", encoding="utf-8")
            second = _build_installed_distribution_inventory(
                [distribution], prefix=prefix, python_version="3.12.11"
            )

        self.assertEqual(first["distributions"][0]["name"], "example-package")
        self.assertNotEqual(first["tree_sha256"], second["tree_sha256"])
        self.assertNotEqual(
            first["distributions"][0]["files"][0]["sha256"],
            second["distributions"][0]["files"][0]["sha256"],
        )

    def test_downstream_source_must_match_the_launch_snapshot_exactly(self) -> None:
        expected = {
            "git_commit": "a" * 40,
            "dirty": False,
            "files": {"studybench/example.py": {"sha256": "b" * 64, "bytes": 1}},
            "tree_sha256": "c" * 64,
        }
        with patch(
            "studybench.provenance.source_record", return_value=deepcopy(expected)
        ):
            self.assertEqual(validate_current_source(expected), expected)
            drifted = deepcopy(expected)
            drifted["files"]["studybench/example.py"]["sha256"] = "d" * 64
            with self.assertRaisesRegex(ValueError, "differs from the run"):
                validate_current_source(drifted)

    def test_source_record_rejects_symlinked_research_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._initialize_source_repo(root)
            scripts = root / "scripts"
            scripts.mkdir()
            target = root / "target.sh"
            target.write_text("#!/bin/sh\n", encoding="utf-8")
            (scripts / "linked.sh").symlink_to(target)
            with (
                patch("studybench.provenance.ROOT", root),
                self.assertRaisesRegex(ValueError, "must not be a symlink"),
            ):
                source_record()

    def test_source_record_freezes_nested_test_code_and_fixtures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._initialize_source_repo(root)
            with patch("studybench.provenance.ROOT", root):
                frozen = source_record()
            self.assertIs(frozen["dirty"], False)
            self.assertIn("tests/unit/test_example.py", frozen["files"])
            self.assertIn("tests/fixtures/case.json", frozen["files"])

            fixture = root / "tests" / "fixtures" / "case.json"
            fixture.write_text('{"case": 2}\n', encoding="utf-8")
            with patch("studybench.provenance.ROOT", root):
                changed = source_record()
                with self.assertRaisesRegex(ValueError, "differs from the run"):
                    validate_current_source(frozen)
            self.assertIs(changed["dirty"], True)
            self.assertNotEqual(
                frozen["files"]["tests/fixtures/case.json"]["sha256"],
                changed["files"]["tests/fixtures/case.json"]["sha256"],
            )

    def test_source_record_hidden_index_flags_force_dirty(self) -> None:
        for flag in ("--assume-unchanged", "--skip-worktree"):
            with self.subTest(flag=flag), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self._initialize_source_repo(root)
                subprocess.run(
                    [
                        "git", "-C", str(root), "update-index", flag,
                        "studybench/example.py",
                    ],
                    check=True,
                )
                with patch("studybench.provenance.ROOT", root):
                    self.assertIs(source_record()["dirty"], True)

    def test_source_record_ignored_test_fixture_forces_dirty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._initialize_source_repo(root)
            (root / ".gitignore").write_text(
                "tests/ignored-fixture.bin\n", encoding="utf-8"
            )
            subprocess.run(["git", "-C", str(root), "add", ".gitignore"], check=True)
            subprocess.run(
                ["git", "-C", str(root), "commit", "-q", "-m", "ignore fixture"],
                check=True,
            )
            ignored = root / "tests" / "ignored-fixture.bin"
            ignored.write_bytes(b"research fixture\n")
            status = subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain", "--", "tests"],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(status.stdout, "")
            with patch("studybench.provenance.ROOT", root):
                record = source_record()
            self.assertIs(record["dirty"], True)
            self.assertIn("tests/ignored-fixture.bin", record["files"])

    def test_source_record_checks_live_mode_when_core_filemode_is_false(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._initialize_source_repo(root)
            subprocess.run(
                ["git", "-C", str(root), "config", "core.filemode", "false"],
                check=True,
            )
            source = root / "studybench" / "example.py"
            source.chmod(0o755)
            status = subprocess.run(
                [
                    "git", "-C", str(root), "status", "--porcelain", "--",
                    "studybench/example.py",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(status.stdout, "")
            with patch("studybench.provenance.ROOT", root):
                self.assertIs(source_record()["dirty"], True)

    def test_model_endpoints_are_loopback_only_and_counted(self) -> None:
        urls = validate_local_server_urls(
            "http://localhost:8100/v1,http://127.0.0.1:8101/v1/",
            expected_count=2,
        )
        self.assertEqual(
            urls,
            ["http://localhost:8100/v1", "http://localhost:8101/v1"],
        )
        for bad in (
            "https://localhost:8100/v1",
            "http://example.com:8100/v1",
            "http://localhost/v1",
            "http://localhost:8100/v1?x=1",
            "http://user@localhost:8100/v1",
            "http://localhost:8100/other",
        ):
            with self.subTest(url=bad), self.assertRaises(ValueError):
                validate_local_server_urls(bad)
        with self.assertRaises(ValueError):
            validate_local_server_urls("http://localhost:8100/v1", expected_count=2)
        for duplicate in (
            "http://localhost:8100/v1,http://127.0.0.1:8100/v1/",
            "http://localhost:8100/v1/,http://[::1]:8100/v1",
        ):
            with self.subTest(duplicate=duplicate), self.assertRaises(ValueError):
                validate_local_server_urls(duplicate)

    def test_claim_ready_environment_binds_every_inventory(self) -> None:
        environment = claim_ready_environment()
        self.assertTrue(environment_is_claim_ready(environment))
        self.assertTrue(
            environment_is_claim_ready(claim_ready_environment(include_dspy=False))
        )

        missing_model_cache = deepcopy(environment)
        missing_model_cache["model_cache"] = None
        self.assertFalse(environment_is_claim_ready(missing_model_cache))

        wrong_lock = deepcopy(environment)
        runtime = wrong_lock["vllm_runtime"]["inventory"]
        runtime["lock_sha256"] = "f" * 64
        payload = canonical_json_bytes(runtime)
        wrong_lock["vllm_runtime"].update({
            "sha256": sha256_bytes(payload),
            "bytes": len(payload),
        })
        self.assertFalse(environment_is_claim_ready(wrong_lock))

        wrong_gpu = deepcopy(environment)
        wrong_gpu["allocation"]["inventory"]["gpus"][0]["cuda_identifier"] = "1"
        payload = canonical_json_bytes(wrong_gpu["allocation"]["inventory"])
        wrong_gpu["allocation"].update({
            "sha256": sha256_bytes(payload),
            "bytes": len(payload),
        })
        self.assertFalse(environment_is_claim_ready(wrong_gpu))

        stale_cache_policy = deepcopy(environment)
        model_inventory = stale_cache_policy["model_cache"]["inventory"]
        model_inventory["attestation_policy"] = "path-hash-only-v0"
        payload = canonical_json_bytes(model_inventory)
        stale_cache_policy["model_cache"].update({
            "sha256": sha256_bytes(payload),
            "bytes": len(payload),
        })
        self.assertFalse(environment_is_claim_ready(stale_cache_policy))

        wrong_runner_lock = deepcopy(environment)
        wrong_runner_lock["runner_lock"]["sync_check"]["status"] = "unchecked"
        self.assertFalse(environment_is_claim_ready(wrong_runner_lock))

        wrong_live_allocation = deepcopy(environment)
        wrong_live_allocation["runner_allocation"]["slurm_job_id"] = "other"
        self.assertFalse(environment_is_claim_ready(wrong_live_allocation))

        wrong_top_level_python = deepcopy(environment)
        wrong_top_level_python["python"] = "different"
        self.assertFalse(environment_is_claim_ready(wrong_top_level_python))
        wrong_top_level_implementation = deepcopy(environment)
        wrong_top_level_implementation["implementation"] = "DifferentPython"
        self.assertFalse(environment_is_claim_ready(wrong_top_level_implementation))

        for inventory_name, field in (
            ("vllm_runtime", "schema_version"),
            ("model_cache", "schema_version"),
            ("model_cache", "file_count"),
            ("allocation", "schema_version"),
            ("allocation", "gpu_count"),
        ):
            wrong_integer_type = deepcopy(environment)
            inventory = wrong_integer_type[inventory_name]["inventory"]
            inventory[field] = True
            payload = canonical_json_bytes(inventory)
            wrong_integer_type[inventory_name].update({
                "sha256": sha256_bytes(payload),
                "bytes": len(payload),
            })
            with self.subTest(inventory=inventory_name, field=field):
                self.assertFalse(environment_is_claim_ready(wrong_integer_type))

    def test_same_versions_with_different_installed_bytes_are_incompatible(self) -> None:
        baseline = claim_ready_environment()
        changed = deepcopy(baseline)
        installed_code = changed["vllm_environment"]["inventory"]
        distribution = installed_code["distributions"][0]
        distribution["files"][0]["sha256"] = "f" * 64
        distribution["record_sha256"] = "f" * 64
        distribution["tree_sha256"] = sha256_json(distribution["files"])
        installed_code["tree_sha256"] = sha256_json(
            installed_code["distributions"]
        )
        package_bytes = canonical_json_bytes(installed_code)
        changed["vllm_environment"].update({
            "sha256": sha256_bytes(package_bytes),
            "bytes": len(package_bytes),
        })
        changed["vllm_environment_sha256"] = changed["vllm_environment"]["sha256"]
        runtime = changed["vllm_runtime"]["inventory"]
        runtime["package_inventory_sha256"] = changed["vllm_environment"]["sha256"]
        runtime_bytes = canonical_json_bytes(runtime)
        changed["vllm_runtime"].update({
            "sha256": sha256_bytes(runtime_bytes),
            "bytes": len(runtime_bytes),
        })

        self.assertTrue(environment_is_claim_ready(baseline))
        self.assertTrue(environment_is_claim_ready(changed))
        self.assertFalse(environments_compatible(baseline, changed))

    def test_environment_compatibility_removes_only_launch_nuisances(self) -> None:
        baseline = claim_ready_environment()
        retry = reallocated_environment(baseline)
        self.assertTrue(environment_is_claim_ready(retry))
        self.assertTrue(environments_compatible(baseline, retry))
        self.assertEqual(
            normalized_environment(baseline), normalized_environment(retry)
        )

        drifted = deepcopy(retry)
        drifted["allocation"]["inventory"]["gpus"][0]["driver_version"] = "new-driver"
        payload = canonical_json_bytes(drifted["allocation"]["inventory"])
        drifted["allocation"].update({
            "sha256": sha256_bytes(payload),
            "bytes": len(payload),
        })
        drifted["nvidia_driver"] = ["new-driver"]
        self.assertTrue(environment_is_claim_ready(drifted))
        self.assertFalse(environments_compatible(baseline, drifted))
        self.assertFalse(environments_compatible({"field": 1}, {"field": True}))

    def test_environment_snapshots_are_exact_content_addressed_artifacts(self) -> None:
        environment = claim_ready_environment()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = write_environment_snapshot(
                root, PurePosixPath("inputs/environments"), environment
            )
            self.assertEqual(
                validate_environment_snapshot(
                    root,
                    record,
                    baseline=environment,
                    require_claim_ready=True,
                ),
                environment,
            )
            unsafe = {**record, "snapshot": "../outside.json"}
            with self.assertRaisesRegex(ValueError, "unsafe"):
                validate_environment_snapshot(
                    root,
                    unsafe,
                    baseline=environment,
                    require_claim_ready=True,
                )
            (root / record["snapshot"]).write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "do not match"):
                validate_environment_snapshot(
                    root,
                    record,
                    baseline=environment,
                    require_claim_ready=True,
                )

    def test_environment_record_reads_only_secure_launcher_snapshots(self) -> None:
        fixture = claim_ready_environment()
        original_lock = (ROOT / "scripts" / "vllm-requirements.lock").read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "logs").mkdir()
            (root / "scripts").mkdir()
            (root / "scripts" / "vllm-requirements.lock").write_bytes(original_lock)
            package_path = root / fixture["vllm_environment"]["path"]
            package_path.write_bytes(
                canonical_json_bytes(fixture["vllm_environment"]["inventory"])
            )
            snapshots = {
                "SB_VLLM_RUNTIME_INVENTORY": fixture["vllm_runtime"],
                "SB_MODEL_CACHE_INVENTORY": fixture["model_cache"],
                "SB_GPU_INVENTORY": fixture["allocation"],
            }
            for snapshot in snapshots.values():
                path = root / snapshot["path"]
                path.write_bytes(canonical_json_bytes(snapshot["inventory"]))
                path.chmod(0o600)
            package_path.chmod(0o600)
            api_key = "ephemeral-test-key"
            launch_id = sha256_text(api_key)
            variables = {
                "SB_VLLM_VERSION": VLLM_VERSION,
                "SB_VLLM_ENV_INVENTORY": fixture["vllm_environment"]["path"],
                "SB_VLLM_ENV_SHA256": fixture["vllm_environment"]["sha256"],
                "SB_VLLM_RUNTIME_INVENTORY": fixture["vllm_runtime"]["path"],
                "SB_VLLM_RUNTIME_SHA256": fixture["vllm_runtime"]["sha256"],
                "SB_MODEL_CACHE_INVENTORY": fixture["model_cache"]["path"],
                "SB_MODEL_CACHE_SHA256": fixture["model_cache"]["sha256"],
                "SB_GPU_INVENTORY": fixture["allocation"]["path"],
                "SB_GPU_INVENTORY_SHA256": fixture["allocation"]["sha256"],
                "SB_MODEL_ID": MODEL_ID,
                "SB_MODEL_REVISION": MODEL_REVISION,
                "SB_TP_EFFECTIVE": "1",
                "SB_NGPU": "1",
                "SB_NSERVE": "1",
                "SB_CUDA_VISIBLE_DEVICES": "0",
                "SB_SLURM_JOB_ID": "123",
                "SB_SERVER_HOSTNAME": "compute.example",
                "SB_SERVER_LAUNCH_ID": launch_id,
                "SB_VLLM_API_KEY": api_key,
                "SB_VLLM_API_KEY_SHA256": launch_id,
                "SLURM_JOB_ID": "123",
                "SLURM_JOB_GPUS": "0",
                "SLURM_JOB_NODELIST": "compute",
                "SLURM_NODEID": "0",
                "CUDA_VISIBLE_DEVICES": "0",
            }
            with (
                patch("studybench.provenance.ROOT", root),
                patch(
                    "studybench.provenance._runner_environment_record",
                    return_value=fixture["runner"],
                ),
                patch(
                    "studybench.provenance._runner_lock_attestation",
                    return_value=fixture["runner_lock"],
                ),
                patch("studybench.provenance._runner_lock_is_valid", return_value=True),
                patch("studybench.provenance.platform.platform", return_value="test"),
                patch(
                    "studybench.provenance.platform.python_version",
                    return_value=fixture["python"],
                ),
                patch(
                    "studybench.provenance.platform.python_implementation",
                    return_value=fixture["implementation"],
                ),
                patch("studybench.provenance.socket.gethostname", return_value="compute.example"),
                patch("studybench.provenance.subprocess.run", side_effect=AssertionError),
                patch.dict(os.environ, variables, clear=True),
            ):
                observed = environment_record()
                self.assertTrue(environment_is_claim_ready(observed))
                self.assertEqual(
                    observed["allocation"]["inventory"]["gpus"],
                    fixture["allocation"]["inventory"]["gpus"],
                )

            wrong_live = {**variables, "SLURM_JOB_ID": "different-job"}
            with (
                patch("studybench.provenance.ROOT", root),
                patch(
                    "studybench.provenance._runner_environment_record",
                    return_value=fixture["runner"],
                ),
                patch(
                    "studybench.provenance._runner_lock_attestation",
                    return_value=fixture["runner_lock"],
                ),
                patch("studybench.provenance._runner_lock_is_valid", return_value=True),
                patch("studybench.provenance.socket.gethostname", return_value="compute.example"),
                patch.dict(os.environ, wrong_live, clear=True),
            ):
                observed = environment_record()
                self.assertIn("runner_allocation", observed["inventory_errors"])
                self.assertFalse(environment_is_claim_ready(observed))

            package_path.chmod(0o644)
            with (
                patch("studybench.provenance.ROOT", root),
                patch(
                    "studybench.provenance._runner_environment_record",
                    return_value=fixture["runner"],
                ),
                patch(
                    "studybench.provenance._runner_lock_attestation",
                    return_value=fixture["runner_lock"],
                ),
                patch("studybench.provenance._runner_lock_is_valid", return_value=True),
                patch("studybench.provenance.socket.gethostname", return_value="compute.example"),
                patch.dict(os.environ, variables, clear=True),
            ):
                observed = environment_record()
                self.assertIn("vllm_environment", observed["inventory_errors"])
                self.assertFalse(environment_is_claim_ready(observed))

    def test_note_inputs_reject_symlink_components_and_duplicate_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "real"
            real.mkdir()
            note = real / "note.md"
            note.write_text("note\n", encoding="utf-8")
            alias = root / "alias"
            alias.symlink_to(real, target_is_directory=True)
            with self.assertRaises(ValueError):
                _load_note(
                    root / "run", alias / "note.md", None, require_manifest=False
                )

            duplicate = root / "duplicate.json"
            note_hash = sha256_text("note\n")
            duplicate.write_text(
                '{"note_sha256":"' + note_hash + '","note_sha256":"' + note_hash + '"}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "invalid note manifest"):
                _load_note(
                    root / "run", note, duplicate, require_manifest=False
                )

            manifest = real / "manifest.json"
            manifest.write_text(
                json.dumps({"note_sha256": note_hash}), encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                _load_note(
                    root / "run", note, alias / "manifest.json",
                    require_manifest=False,
                )

    def test_ids_are_path_safe(self) -> None:
        self.assertEqual(validate_id("confirm-r1"), "confirm-r1")
        for bad in ("x", "Upper", "../escape", "has space", "/absolute"):
            with self.assertRaises(ValueError):
                validate_id(bad)

    def test_note_manifest_must_match_exact_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            note = root / "source.md"
            note.write_text("exact\n", encoding="utf-8")
            config = {
                "study_id": "study-r1",
                "task": "dspy",
                "method": "forced-50-cheatsheet",
                "claim_ready": True,
                "model": "model",
                "model_revision": "revision",
                "episode_seed": 7,
                "study_question_sha256": "q" * 64,
                "corpus": {"commit": "abc"},
            }
            write_immutable_json(root / "intent.json", config)
            episode = {
                "study_intent_sha256": sha256_json(config),
                "question_sha256": "q" * 64,
                "status": "ok",
                "answer": "exact\n",
                "model": "model",
                "model_revision": "revision",
                "seed": 7,
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            }
            write_immutable_json(root / "episode.json", episode)
            inventory = {
                relative: {
                    "sha256": sha256_file(root / relative),
                    "bytes": (root / relative).stat().st_size,
                }
                for relative in ("intent.json", "episode.json")
            }
            manifest = root / "source.manifest.json"
            write_immutable_json(manifest, {
                "manifest_type": "forced-50-cheatsheet",
                "note_sha256": sha256_text("exact\n"),
                "note_path": "source.md",
                "study_id": "study-r1",
                "task": "dspy",
                "corpus_commit": "abc",
                "claim_ready": True,
                "config": config,
                "intent_sha256": sha256_json(config),
                "episode_sha256": sha256_json(episode),
                "study_generated_tokens": 5,
                "study_prompt_tokens": 10,
                "study_total_tokens": 15,
                "construction_artifacts": inventory,
                "construction_artifacts_sha256": sha256_json(inventory),
            })
            text, record = _load_note(
                root / "run", note, manifest, require_manifest=True,
                expected_task="dspy", expected_corpus_commit="abc",
            )
            self.assertEqual(text, "exact\n")
            self.assertEqual(record["sha256"], sha256_text(text))
            self.assertEqual(
                set(record["provenance_bundle"]["construction_artifacts"]["artifacts"]),
                {"intent.json", "episode.json"},
            )
            note.write_text("drift\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                _load_note(
                    root / "other", note, manifest, require_manifest=True,
                    expected_task="dspy", expected_corpus_commit="abc",
                )

    def test_unknown_claim_ready_note_manifest_type_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            note = root / "note.md"
            note.write_text("note", encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "note_sha256": sha256_text("note"),
                "note_path": "note.md",
                "study_id": "study-r1",
                "task": "dspy",
                "corpus_commit": "abc",
                "claim_ready": True,
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown claim-ready"):
                _load_note(
                    root / "run", note, manifest, require_manifest=True,
                    expected_task="dspy", expected_corpus_commit="abc",
                )

    def test_research_note_requires_construction_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            note = Path(directory) / "note.md"
            note.write_text("n", encoding="utf-8")
            with self.assertRaises(ValueError):
                _load_note(Path(directory) / "run", note, None, require_manifest=True)

    def test_resume_requires_every_identity_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.json"
            identity = {
                "manifest_sha256": "m",
                "question_sha256": "q",
                "prompt_sha256": "p",
                "note_sha256": None,
                "seed": 1,
                "task": "dspy",
                "qid": "q1",
                "budget": "direct",
                "rollout": 0,
            }
            path.write_text(json.dumps({**identity, "status": "ok"}), encoding="utf-8")
            self.assertEqual(validate_resumable_episode(path, identity)["status"], "ok")
            changed = {**identity, "seed": 2}
            with self.assertRaises(ValueError):
                validate_resumable_episode(path, changed)
            for field in ("seed", "rollout"):
                boolean_identity = {**identity, field: bool(identity[field])}
                path.write_text(
                    json.dumps({**boolean_identity, "status": "ok"}), encoding="utf-8"
                )
                with self.subTest(field=field), self.assertRaises(ValueError):
                    validate_resumable_episode(path, identity)

    def test_manifest_freezes_grid_seed_and_presented_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            (root / "scripts" / "vllm-requirements.lock").write_bytes(
                (ROOT / "scripts" / "vllm-requirements.lock").read_bytes()
            )
            corpus = SimpleNamespace(name="dspy", commit="abc", repo=root / "corpus")
            questions = [{"id": "q1", "question": "Question?", "rubric": []}]
            corpus_record = {
                "name": "dspy",
                "commit": "abc",
                "dirty": False,
                "roots": ["dspy"],
                "language": "python",
                "suffixes": [".py"],
            }
            source_record = {
                "git_commit": "def",
                "dirty": False,
                "files": {},
                "tree_sha256": "tree",
            }
            kwargs = dict(
                run_id="confirm-r1",
                task="dspy",
                corpus=corpus,
                questions=questions,
                budgets=["direct", "k5"],
                rollouts=2,
                harness="test",
                model="model",
                model_revision="c" * 40,
                sampling={"temperature": 0},
                master_seed=7,
                seed_namespace="test-harness",
                seed_group="paired-r1",
                note_path=None,
                note_manifest_path=None,
                note_prefix_template=None,
                smoke=False,
                exploratory=True,
                allow_dirty=False,
                preregistration_path=None,
                preregistration_role=None,
                extra={
                    "model_revision": "c" * 40,
                    "expected_response_model": "model",
                },
            )
            with (
                patch("studybench.provenance.ROOT", root),
                patch("studybench.provenance.corpus_record", return_value=corpus_record),
                patch("studybench.provenance.source_record", return_value=source_record),
                patch(
                    "studybench.provenance.environment_record",
                    return_value=claim_ready_environment(),
                ),
                patch("studybench.provenance.environment_is_claim_ready", return_value=True),
            ):
                context = prepare_run(**kwargs)
                spec = context.manifest["spec"]
                self.assertEqual(spec["purpose"], "exploratory")
                self.assertIs(spec["claim_ready"], False)
                self.assertEqual(
                    spec["preregistration"],
                    {"schema_version": 1, "status": "not_provided", "reason": "exploratory"},
                )
                self.assertEqual(
                    spec["expected_episodes"],
                    [
                        "direct/r0/q1.json", "direct/r1/q1.json",
                        "k5/r0/q1.json", "k5/r1/q1.json",
                    ],
                )
                seed = spec["seed_policy"]["episode_seeds"]["k5/r1/q1.json"]
                identity = episode_identity(
                    context, q=questions[0], prompt="Question?",
                    budget="k5", rollout=1, seed=seed,
                )
                self.assertEqual(identity["seed"], seed)
                self.assertEqual(
                    identity["environment_snapshot"],
                    context.launch_environment_record,
                )
                episode_path = context.root / "k5" / "r1" / "q1.json"
                write_immutable_json(episode_path, {**identity, "status": "ok"})
                resumed = prepare_run(**kwargs)
                self.assertEqual(resumed.manifest_sha256, context.manifest_sha256)
                retry_environment = reallocated_environment(
                    claim_ready_environment()
                )
                with patch(
                    "studybench.provenance.environment_record",
                    return_value=retry_environment,
                ):
                    cross_allocation = prepare_run(**kwargs)
                self.assertEqual(
                    cross_allocation.manifest_sha256, context.manifest_sha256
                )
                self.assertNotEqual(
                    cross_allocation.launch_environment_record,
                    context.launch_environment_record,
                )
                retry_identity = episode_identity(
                    cross_allocation,
                    q=questions[0],
                    prompt="Question?",
                    budget="k5",
                    rollout=1,
                    seed=seed,
                )
                with self.assertRaisesRegex(ValueError, "current launch"):
                    write_episode_result(
                        cross_allocation,
                        context.root / "failed-new-episode.json",
                        {**identity, "status": "error"},
                    )
                self.assertEqual(
                    validate_resumable_episode(
                        episode_path,
                        retry_identity,
                        context=cross_allocation,
                    )["status"],
                    "ok",
                )

                substantive_drift = deepcopy(retry_environment)
                substantive_drift["allocation"]["inventory"]["gpus"][0][
                    "driver_version"
                ] = "new-driver"
                substantive_payload = canonical_json_bytes(
                    substantive_drift["allocation"]["inventory"]
                )
                substantive_drift["allocation"].update({
                    "sha256": sha256_bytes(substantive_payload),
                    "bytes": len(substantive_payload),
                })
                substantive_drift["nvidia_driver"] = ["new-driver"]
                with (
                    patch(
                        "studybench.provenance.environment_record",
                        return_value=substantive_drift,
                    ),
                    self.assertRaisesRegex(ValueError, "substantive drift"),
                ):
                    prepare_run(**kwargs)
                paired = prepare_run(**{**kwargs, "run_id": "treatment-r1"})
                self.assertEqual(
                    paired.manifest["spec"]["seed_policy"]["episode_seeds"],
                    spec["seed_policy"]["episode_seeds"],
                )
                with self.assertRaises(ValueError):
                    prepare_run(**{**kwargs, "master_seed": 8})
                with (
                    patch(
                        "studybench.provenance.environment_record",
                        return_value=claim_ready_environment(include_dspy=False),
                    ),
                    self.assertRaisesRegex(ValueError, "environment is incomplete"),
                ):
                    prepare_run(
                        **{
                            **kwargs,
                            "run_id": "missing-dspy-r1",
                            "harness": "dspy.ReAct",
                        }
                    )

    def test_confirmatory_manifest_snapshots_and_revalidates_preregistration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            questions = [{"id": "q1", "question": "Question?", "rubric": []}]
            preregistration_data = canonical_json_bytes({"frozen": "contract"})
            preregistration_sha256 = sha256_bytes(preregistration_data)
            loaded = SimpleNamespace(
                data=preregistration_data,
                sha256=preregistration_sha256,
                relative_path="preregistrations/confirm-r1.json",
                head_commit="f" * 40,
                document={"frozen": "contract"},
            )
            corpus = SimpleNamespace(name="dspy", commit="a" * 40, repo=root / "corpus")
            with (
                patch("studybench.provenance.ROOT", root),
                patch("studybench.provenance.corpus_record", return_value={
                    "name": "dspy", "commit": "a" * 40, "dirty": False,
                }),
                patch("studybench.provenance.source_record", return_value={
                    "git_commit": "f" * 40, "dirty": False, "files": {},
                    "tree_sha256": "tree",
                }),
                patch(
                    "studybench.provenance.environment_record",
                    return_value=claim_ready_environment(),
                ),
                patch(
                    "studybench.provenance.environment_is_claim_ready",
                    return_value=True,
                ),
                patch(
                    "studybench.provenance.bind_preregistration",
                    return_value=loaded,
                ) as bind,
                patch(
                    "studybench.provenance.revalidate_run_preregistration"
                ) as revalidate,
            ):
                context = prepare_run(
                    run_id="control-r1",
                    task="dspy",
                    corpus=corpus,
                    questions=questions,
                    budgets=["direct", "k5", "k20", "k20f"],
                    rollouts=6,
                    harness="dspy.ReAct",
                    model="openai/Qwen/Qwen3.5-9B",
                    model_revision="c" * 40,
                    sampling={"temperature": 0.0},
                    master_seed=44001,
                    seed_namespace="dspy-react",
                    seed_group="paired-r1",
                    note_path=None,
                    note_manifest_path=None,
                    note_prefix_template=None,
                    smoke=False,
                    exploratory=False,
                    allow_dirty=False,
                    preregistration_path=Path("preregistrations/confirm-r1.json"),
                    preregistration_role="control",
                    extra={
                        "model_revision": "c" * 40,
                        "expected_response_model": "Qwen/Qwen3.5-9B",
                    },
                )
            spec = context.manifest["spec"]
            self.assertEqual(spec["purpose"], "confirmatory")
            self.assertIs(spec["claim_ready"], True)
            self.assertEqual(spec["model_revision"], spec["extra"]["model_revision"])
            snapshot = context.root / spec["preregistration"]["snapshot"]
            self.assertEqual(snapshot.read_bytes(), preregistration_data)
            bind.assert_called_once()
            revalidate.assert_called_once_with(spec, context.root)

    def test_manifest_is_never_added_to_a_nonempty_legacy_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "runs" / "legacy-run" / "dspy" / "direct"
            legacy.mkdir(parents=True)
            (legacy / "old.json").write_text("{}", encoding="utf-8")
            corpus = SimpleNamespace(name="dspy", commit="abc", repo=root / "corpus")
            ready_environment = claim_ready_environment()
            with (
                patch("studybench.provenance.ROOT", root),
                patch("studybench.provenance.corpus_record", return_value={
                    "name": "dspy", "commit": "abc", "dirty": False,
                }),
                patch("studybench.provenance.source_record", return_value={
                    "git_commit": "def", "dirty": False, "files": {},
                    "tree_sha256": "tree",
                }),
                patch("studybench.provenance.environment_record", return_value=ready_environment),
            ):
                with self.assertRaises(ValueError):
                    prepare_run(
                        run_id="legacy-run", task="dspy", corpus=corpus,
                        questions=[{"id": "q1", "question": "Q"}],
                        budgets=["direct"], rollouts=1, harness="h", model="m",
                        model_revision="c" * 40, sampling={"temperature": 0},
                        master_seed=1, seed_namespace="h",
                        seed_group="paired-r1", note_path=None,
                        note_manifest_path=None, note_prefix_template=None,
                        smoke=False, exploratory=True, allow_dirty=False,
                        preregistration_path=None, preregistration_role=None,
                        extra={
                            "model_revision": "c" * 40,
                            "expected_response_model": "m",
                        },
                    )

    def test_failed_attempts_never_overwrite_the_expected_episode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = RunContext(root, {"spec": {"note": None}}, "manifest", "", "")
            expected = root / "k20f" / "r0" / "q1.json"
            first = write_episode_result(context, expected, {"status": "forced_short"})
            second = write_episode_result(context, expected, {"status": "error"})
            self.assertFalse(expected.exists())
            self.assertEqual(first.name, "attempt-1.json")
            self.assertEqual(second.name, "attempt-2.json")
            final = write_episode_result(context, expected, {"status": "ok"})
            self.assertEqual(final, expected)
            self.assertTrue(expected.is_file())


if __name__ == "__main__":
    unittest.main()
