from __future__ import annotations

from pathlib import Path
import subprocess
import unittest

from studybench.dataset import CORPORA, ROOT


SCRIPTS = ROOT / "scripts"


class ScriptContractTests(unittest.TestCase):
    def read(self, name: str) -> str:
        return (SCRIPTS / name).read_text(encoding="utf-8")

    def test_every_runner_is_valid_bash(self) -> None:
        paths = sorted((*SCRIPTS.glob("*.sh"), *SCRIPTS.glob("*.sbatch")))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(path=path.name):
                subprocess.run(["bash", "-n", str(path)], check=True)

    def test_setup_is_frozen_pinned_and_non_destructive(self) -> None:
        common = self.read("setup_common.sh")
        combined = "\n".join(
            self.read(name)
            for name in ("setup.sh", "setup_grading.sh", "setup_common.sh")
        )
        self.assertIn(CORPORA["dspy"].commit, common)
        self.assertIn(CORPORA["openclaw"].commit, common)
        self.assertIn("MAIN_PYTHON_VERSION=3.14.6", common)
        self.assertIn("AUX_PYTHON_VERSION=3.12.11", common)
        self.assertIn("uv sync --frozen", common)
        self.assertIn("uv pip sync", common)
        self.assertIn("vllm-requirements.lock", common)
        self.assertNotIn("rm -rf", combined)
        self.assertNotIn("--extra optuna", combined)
        self.assertIn(".env must have mode 0600", common)
        self.assertIn("[ ! -L \"$path\" ]", common)

        failed_version = subprocess.run(
            [
                "bash",
                "-c",
                "source scripts/setup_common.sh; "
                "verify_python_version /bin/false 3.14.6 test",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(failed_version.returncode, 0)

    def test_vllm_lock_matches_the_present_environment(self) -> None:
        python = ROOT / ".venv-vllm/bin/python"
        if not python.exists():
            self.skipTest("vLLM environment is intentionally absent in grading-only setup")
        expected = sorted(
            line.strip()
            for line in (SCRIPTS / "vllm-requirements.lock")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip() and not line.startswith("#")
        )
        code = (
            "from importlib.metadata import distributions; import re; "
            "print('\\n'.join(sorted("
            "f\"{re.sub(r'[-_.]+', '-', d.metadata['Name']).lower()}=={d.version}\" "
            "for d in distributions())))"
        )
        observed = subprocess.check_output([str(python), "-c", code], text=True)
        self.assertEqual(observed.splitlines(), expected)

    def test_server_requires_explicit_slurm_gpus_and_records_provenance(self) -> None:
        serve = self.read("serve_vllm.sh")
        wait = self.read("serve_and_wait.sh")
        self.assertIn("SLURM_JOB_ID:?", serve)
        self.assertIn("CUDA_VISIBLE_DEVICES:?", serve)
        self.assertNotIn("--query-gpu=index", serve)
        self.assertIn(
            '"${SANITIZED_ENV[@]}" "$NVIDIA_SMI" -i "$selector"', serve
        )
        self.assertNotIn('value=$(nvidia-smi -i "$selector"', serve)
        self.assertNotIn('nvidia-smi --query-gpu=', serve)
        self.assertIn("torch.cuda.get_device_properties(logical_index).uuid", serve)
        self.assertIn('query_allocated_gpu "$uuid"', serve)
        self.assertNotIn('query_allocated_gpu "$gpu" uuid', serve)
        self.assertIn("MODEL_REVISION=c202236235762e1c871ad0ccb60c8ee5ba337b9a", serve)
        for field in (
            "SB_VLLM_VERSION",
            "SB_TP_EFFECTIVE",
            "SB_VLLM_ENV_INVENTORY",
            "SB_VLLM_ENV_SHA256",
            "SB_VLLM_RUNTIME_INVENTORY",
            "SB_VLLM_RUNTIME_SHA256",
            "SB_MODEL_CACHE_INVENTORY",
            "SB_MODEL_CACHE_SHA256",
            "SB_GPU_INVENTORY",
            "SB_GPU_INVENTORY_SHA256",
            "SB_CUDA_VISIBLE_DEVICES",
            "SB_SLURM_JOB_ID",
            "SB_SERVER_HOSTNAME",
            "SB_SERVER_LAUNCH_ID",
            "SB_VLLM_API_KEY_SHA256",
        ):
            self.assertIn(field, serve)
            self.assertIn(field, wait)
        self.assertIn("write_vllm_inventory", serve)
        self.assertIn("installed_distribution_inventory", self.read("setup_common.sh"))
        self.assertIn("canonical_json_bytes", self.read("setup_common.sh"))

    def test_server_identity_and_readiness_are_collision_safe(self) -> None:
        serve = self.read("serve_vllm.sh")
        wait = self.read("serve_and_wait.sh")
        for text in (serve, wait):
            self.assertIn("verify_env_file", text)
            self.assertIn("never reads secret contents", text)
        self.assertIn("secrets.token_urlsafe(48)", serve)
        self.assertNotIn('--api-key "$SB_VLLM_API_KEY"', serve)
        self.assertNotIn('"VLLM_API_KEY=$SB_VLLM_API_KEY"', serve)
        self.assertIn("secret = sys.stdin.buffer.read()", serve)
        self.assertIn('os.environ["VLLM_API_KEY"] = api_key', serve)
        self.assertIn("os.execv(sys.argv[1], sys.argv[1:])", serve)
        self.assertIn("env -i", serve)
        self.assertIn('"policy": "clear-and-allowlist-v1"', serve)
        self.assertIn('"proxy_policy": "cleared"', serve)
        self.assertIn("--host 127.0.0.1", serve)
        self.assertIn("chmod 600 \"$TOPOLOGY_TMP\"", serve)
        self.assertIn("$LOG_PREFIX.topology", serve)
        self.assertIn("missing or concurrently changing snapshot", serve)
        self.assertIn("HF_HUB_OFFLINE=1", serve)
        self.assertIn("TRANSFORMERS_OFFLINE=1", serve)
        self.assertIn("duplicate CUDA_VISIBLE_DEVICES identifier", serve)
        self.assertIn("CUDA_VISIBLE_DEVICES contains an empty identifier", serve)
        self.assertIn("allocated GPU identifiers alias the same UUID", serve)
        self.assertIn("tree_sha256", (ROOT / "studybench/model_cache.py").read_text())
        self.assertIn("studybench/model_cache.py", serve)
        self.assertIn(
            'verify "$MODEL_ID" "$MODEL_REVISION" "$MODEL_CACHE_INVENTORY"',
            serve,
        )
        self.assertIn("studybench/model_cache.py verify", wait)
        self.assertIn(
            '"$SB_MODEL_ID" "$SB_MODEL_REVISION" "$SB_MODEL_CACHE_INVENTORY"',
            wait,
        )
        self.assertIn('"stable-openat-sha256-v1"', wait)
        self.assertIn('"cuda_toolkit"', serve)
        self.assertIn('"cuda_version": torch.version.cuda', serve)
        self.assertIn("resolved_nvcc", serve)
        self.assertIn("version_sha256", wait)
        self.assertIn("wait -n", serve)
        self.assertNotIn('--header "Authorization: Bearer $SB_VLLM_API_KEY"', wait)
        self.assertIn('--config - --output "$response"', wait)
        self.assertIn('header = "Authorization: Bearer $SB_VLLM_API_KEY"', wait)
        self.assertIn("--noproxy '*'", wait)
        self.assertIn("unset HTTP_PROXY HTTPS_PROXY ALL_PROXY", wait)
        self.assertIn("export NO_PROXY=localhost,127.0.0.1,::1", wait)
        self.assertIn("unset PYTHONHOME PYTHONPATH LD_PRELOAD LD_AUDIT", wait)
        self.assertIn('"$url/models"', wait)
        self.assertNotIn("/health", wait)
        self.assertIn("assert_launcher_alive", wait)
        readiness = wait[wait.index("for ((i = 0; i < SB_NSERVE; i++))") :]
        self.assertGreaterEqual(readiness.count("assert_launcher_alive"), 3)
        self.assertIn('[ "$SB_LAUNCHER_PID" = "$SERVE_PID" ]', wait)
        self.assertIn("verify_owner_only_file \"$TOPOLOGY\"", wait)

    def test_runner_flags_are_explicit_and_task_bound(self) -> None:
        react = self.read("react.sbatch")
        rollout = self.read("rollout.sbatch")
        retry = self.read("retry-errors.sbatch")
        selfquiz = self.read("selfquiz.sbatch")
        for text in (react, rollout, retry):
            self.assertIn("SB_SEED_GROUP", text)
            self.assertIn("--seed-group", text)
            self.assertIn("require_single_csv_value SB_TASKS", text)
            self.assertIn("SB_PREREGISTRATION", text)
            self.assertIn("SB_PREREGISTRATION_ROLE", text)
            self.assertIn("--preregistration", text)
            self.assertIn("--preregistration-role", text)
            self.assertIn("SB_EXPLORATORY", text)
            self.assertIn("--exploratory", text)
        for text in (react, rollout, retry, selfquiz):
            self.assertIn("#SBATCH --partition=matx", text)
            self.assertIn("#SBATCH --gpus-per-node=6", text)
            self.assertNotIn("#SBATCH --partition=a3", text)
        launch_loop = react[react.index("for task in ${TASKS//,/ }") :]
        study_launch, evaluation_launch = launch_loop.split("    else\n", 1)
        self.assertNotIn("--preregistration", study_launch)
        self.assertNotIn("--exploratory", study_launch)
        self.assertIn("--preregistration", evaluation_launch)
        self.assertIn("--exploratory", evaluation_launch)
        self.assertIn("--study-id", selfquiz)
        self.assertIn("--seed", selfquiz)
        self.assertIn("SB_AUDIT_PROTOCOL", selfquiz)
        self.assertIn("--audit-protocol", selfquiz)
        self.assertIn("may only be registered in round 1", selfquiz)
        self.assertIn('[ ! -L "$AUDIT_PROTOCOL" ]', selfquiz)
        self.assertNotIn("--promote-human-audit", selfquiz)
        self.assertIn("sync_dspy_environment", selfquiz)
        self.assertNotIn("tree_sitter", selfquiz)
        for removed in ("SB_COMPACT", "SB_USAGE", "SB_STUDIED", "SB_SELECT"):
            self.assertNotIn(removed, selfquiz)
        self.assertIn("sync_dspy_environment", retry)
        self.assertIn("sync_main_environment", retry)
        self.assertIn("sync_dspy_environment", react)
        self.assertIn("sync_main_environment", rollout)

    def test_shell_ids_match_python_provenance_constraints(self) -> None:
        valid = subprocess.run(
            [
                "bash",
                "-c",
                "source scripts/run_args.sh; validate_id TEST confirm-r1",
            ],
            cwd=ROOT,
        )
        self.assertEqual(valid.returncode, 0)
        for invalid in ("x", "Upper", "../escape", "a" * 81):
            with self.subTest(invalid=invalid):
                proc = subprocess.run(
                    [
                        "bash",
                        "-c",
                        "source scripts/run_args.sh; validate_id TEST \"$1\"",
                        "bash",
                        invalid,
                    ],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(proc.returncode, 0)
                self.assertNotIn("contents", proc.stdout)
        duplicate = subprocess.run(
            [
                "bash",
                "-c",
                "source scripts/run_args.sh; "
                "validate_csv_members TEST dspy,dspy dspy,openclaw",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(duplicate.returncode, 0)
        self.assertIn("duplicate", duplicate.stderr)


if __name__ == "__main__":
    unittest.main()
