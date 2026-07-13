from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from studybench import sandbox


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SandboxTests(unittest.TestCase):
    def test_frozen_checker_configuration_is_rechecked_before_and_after(self) -> None:
        expected = {"language": "python", "ready": True, "identity": "a"}
        changed = {**expected, "identity": "b"}
        with (
            patch("studybench.sandbox.configuration_record", return_value=changed),
            patch("studybench.sandbox._check_python") as checker,
        ):
            with self.assertRaises(sandbox.CheckerConfigurationChanged):
                sandbox.check(
                    "```python\npass\n```",
                    "python",
                    expected_configuration=expected,
                )
        checker.assert_not_called()

        with (
            patch(
                "studybench.sandbox.configuration_record",
                side_effect=[expected, changed],
            ),
            patch(
                "studybench.sandbox._check_python",
                return_value={"compile_ok": True, "detail": "ok"},
            ) as checker,
        ):
            with self.assertRaises(sandbox.CheckerConfigurationChanged):
                sandbox.check(
                    "```python\npass\n```",
                    "python",
                    expected_configuration=expected,
                )
        checker.assert_called_once()

    def test_python_execution_is_disabled_without_a_pinned_container(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("studybench.sandbox._run_limited") as run,
        ):
            result = sandbox._check_python("print('must not execute')")
        run.assert_not_called()
        self.assertTrue(result["syntax_ok"])
        self.assertFalse(result["compile_ok"])
        self.assertFalse(result["sandboxed"])
        self.assertEqual(result["check_level"], "syntax-only")
        self.assertIn(sandbox.PYTHON_IMAGE_ENV, result["detail"])

    def test_python_syntax_failure_remains_fail_closed(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            result = sandbox._check_python("def broken(:\n    pass")
        self.assertFalse(result["syntax_ok"])
        self.assertFalse(result["compile_ok"])
        self.assertFalse(result["sandboxed"])

    def test_pinned_container_builds_the_required_offline_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            image = root / "python.sif"
            image.write_bytes(b"pinned image fixture")
            runtime = root / "apptainer"
            runtime.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
            runtime.chmod(0o755)
            image_digest = sha256(image)
            runtime_digest = sha256(runtime)
            environment = {
                sandbox.PYTHON_IMAGE_ENV: str(image),
                sandbox.PYTHON_IMAGE_SHA256_ENV: image_digest,
                sandbox.APPTAINER_BIN_ENV: str(runtime),
                sandbox.APPTAINER_SHA256_ENV: runtime_digest,
            }
            with (
                patch.dict(os.environ, environment, clear=True),
                patch(
                    "studybench.sandbox._run_limited",
                    return_value=(0, "", False, True),
                ) as run,
            ):
                result = sandbox._check_python("print('contained')")
                record = sandbox.configuration_record("python")

        command = run.call_args.args[0]
        self.assertEqual(command[:2], [str(runtime), "exec"])
        for flag in ("--containall", "--cleanenv", "--no-home", "--net"):
            self.assertIn(flag, command)
        self.assertIn("--network", command)
        self.assertIn("none", command)
        self.assertIn("--bind", command)
        self.assertTrue(any(value.endswith(":/work:rw") for value in command))
        self.assertEqual(command[-3:], ["python3", "-I", "/work/answer.py"])
        self.assertEqual(result["check_level"], "contained-execution")
        self.assertTrue(result["compile_ok"])
        self.assertTrue(result["sandboxed"])
        self.assertEqual(result["image_sha256"], image_digest)
        self.assertEqual(result["checker_sha256"], runtime_digest)
        self.assertTrue(record["ready"])
        self.assertEqual(record["checker"], str(runtime))

    def test_container_hash_drift_never_launches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            image = root / "python.sif"
            image.write_bytes(b"changed")
            runtime = root / "apptainer"
            runtime.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            runtime.chmod(0o755)
            image_digest = sha256(image)
            environment = {
                sandbox.PYTHON_IMAGE_ENV: str(image),
                sandbox.PYTHON_IMAGE_SHA256_ENV: "0" * 64,
                sandbox.APPTAINER_BIN_ENV: str(runtime),
                sandbox.APPTAINER_SHA256_ENV: sha256(runtime),
            }
            with (
                patch.dict(os.environ, environment, clear=True),
                patch("studybench.sandbox._run_limited") as run,
            ):
                result = sandbox._check_python("pass")
        run.assert_not_called()
        self.assertFalse(result["compile_ok"])
        self.assertEqual(result["image_sha256"], image_digest)
        self.assertIn("hash mismatch", result["detail"])

    def test_missing_resource_launcher_makes_container_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            image = root / "python.sif"
            image.write_bytes(b"image")
            runtime = root / "apptainer"
            runtime.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            runtime.chmod(0o755)
            environment = {
                sandbox.PYTHON_IMAGE_ENV: str(image),
                sandbox.PYTHON_IMAGE_SHA256_ENV: sha256(image),
                sandbox.APPTAINER_BIN_ENV: str(runtime),
                sandbox.APPTAINER_SHA256_ENV: sha256(runtime),
            }
            with (
                patch.dict(os.environ, environment, clear=True),
                patch("studybench.sandbox.PYTHON_BIN", root / "missing-python"),
            ):
                record = sandbox.configuration_record("python")
        self.assertFalse(record["ready"])
        self.assertFalse(record["sandboxed"])
        self.assertIn("launcher", record["error"])

    def test_typescript_tree_sitter_is_syntax_only(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            valid = sandbox._check_typescript("const value: number = 1;")
            invalid = sandbox._check_typescript("const value = ;")
            record = sandbox.configuration_record("typescript")
        self.assertTrue(valid["syntax_ok"])
        self.assertFalse(valid["compile_ok"])
        self.assertFalse(valid["sandboxed"])
        self.assertEqual(valid["check_level"], "syntax-only")
        self.assertTrue(Path(valid["checker"]).is_absolute())
        self.assertRegex(valid["checker_sha256"], r"^[0-9a-f]{64}$")
        self.assertTrue(Path(valid["syntax_core"]).is_absolute())
        self.assertRegex(valid["syntax_core_sha256"], r"^[0-9a-f]{64}$")
        self.assertFalse(invalid["syntax_ok"])
        self.assertFalse(record["ready"])
        self.assertEqual(record["syntax_core_version"], "0.26.0")

    def test_typescript_compiler_must_be_content_pinned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checker = (Path(temporary) / "ts-checker").resolve()
            checker.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            checker.chmod(0o755)
            with patch("studybench.sandbox._run_limited") as run:
                unpinned = sandbox._check_typescript(
                    "const value = 1;", checker=checker
                )
            run.assert_not_called()
            self.assertFalse(unpinned["compile_ok"])

            with patch(
                "studybench.sandbox._run_limited",
                return_value=(0, "", False, True),
            ) as run:
                pinned = sandbox._check_typescript(
                    "const value = 1;",
                    checker=checker,
                    checker_sha256=sha256(checker),
                )
            run.assert_called_once()
            self.assertTrue(pinned["compile_ok"])
            self.assertEqual(pinned["check_level"], "configured-compiler")
            self.assertFalse(pinned["sandboxed"])
            self.assertEqual(pinned["checker_sha256"], sha256(checker))

    def test_child_environment_does_not_inherit_credentials_or_gpus(self) -> None:
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "secret", "CUDA_VISIBLE_DEVICES": "0,1"},
            clear=True,
        ):
            child = sandbox._child_environment("/tmp/studybench-test")
        self.assertNotIn("OPENAI_API_KEY", child)
        self.assertEqual(child["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(child["NVIDIA_VISIBLE_DEVICES"], "none")
        launcher = sandbox._limit_launcher()
        for limit in ("RLIMIT_CPU", "RLIMIT_AS", "RLIMIT_FSIZE", "RLIMIT_NPROC"):
            self.assertIn(limit, launcher)


if __name__ == "__main__":
    unittest.main()
