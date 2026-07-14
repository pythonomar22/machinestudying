import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from studybench import local_judge_qualification as qualification


class FakeUsage:
    def model_dump(self, *, mode):
        assert mode == "json"
        return {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_tokens_details": None,
            "completion_tokens_details": None,
        }


class FakeClient:
    def __init__(self, *, base_url, flip_case=None, intent_path=None, **_):
        self.base_url = base_url
        self.flip_case = flip_case
        self.intent_path = intent_path
        self.counter = 0
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create)
        )
        self.models = SimpleNamespace(list=self.list_models)

    async def create(self, **request):
        if self.intent_path is not None and not self.intent_path.is_file():
            raise AssertionError("qualification intent was not durable before contact")
        self.counter += 1
        payload = json.loads(request["messages"][-1]["content"])
        case = next(
            item for item in qualification.QUALIFICATION_CASES
            if item.case_id == payload["question_id"]
        )
        scores = case.expected_scores()
        if self.flip_case == case.case_id:
            claim_id = next(iter(scores))
            scores[claim_id] = 1 - scores[claim_id]
        content = json.dumps({
            "claims": scores,
            "needs_regrade": case.needs_regrade,
        })
        return SimpleNamespace(
            id=f"response-{self.counter}",
            _request_id=f"request-{self.counter}",
            model=qualification.LOCAL_GRADER_MODEL,
            system_fingerprint="fake-fingerprint",
            usage=FakeUsage(),
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason="stop",
            )],
        )

    async def list_models(self):
        return SimpleNamespace(data=[SimpleNamespace(
            id=qualification.LOCAL_GRADER_MODEL
        )])

    async def close(self):
        return None


class QualificationSuiteTests(unittest.IsolatedAsyncioTestCase):
    SERVER_LAUNCH_ID = "b" * 64

    @classmethod
    def _runtime(cls):
        return {
            "server_launch_id": cls.SERVER_LAUNCH_ID,
            "server": {"server_count": 3},
        }

    @staticmethod
    def _source():
        return {
            "git_commit": "a" * 40,
            "dirty": False,
            "files": {},
            "tree_sha256": qualification.sha256_json({}),
        }

    @classmethod
    def _output(cls, directory: str) -> Path:
        with patch.object(qualification, "ROOT", Path(directory)):
            return qualification.qualification_audit_path(cls._runtime())

    def test_suite_is_exact_balanced_and_metamorphic(self):
        suite = qualification.validate_qualification_suite()
        self.assertEqual(suite["case_count"], 20)
        self.assertEqual(suite["claim_count"], 44)
        self.assertEqual(suite["positive_labels"], 22)
        self.assertEqual(suite["negative_labels"], 22)
        self.assertEqual(suite["sha256"], qualification.EXPECTED_SUITE_SHA256)
        self.assertEqual(
            [item["case_id"] for item in suite["cases"]],
            [f"C{index:02d}" for index in range(1, 21)],
        )
        self.assertEqual(
            sum(item["expected_needs_regrade"] for item in suite["cases"]),
            1,
        )

    def test_audit_path_is_canonical_for_one_validated_launch_id(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            qualification, "ROOT", Path(directory)
        ):
            canonical = qualification.qualification_audit_path(self._runtime())
            self.assertEqual(
                canonical,
                Path(directory)
                / "logs"
                / f"local-judge-qualification-{self.SERVER_LAUNCH_ID}.json",
            )
            self.assertEqual(
                qualification.qualification_audit_path({
                    "server_launch_id": self.SERVER_LAUNCH_ID,
                    "unrelated_runtime_field": "different",
                }),
                canonical,
            )
            self.assertEqual(
                qualification._require_qualification_audit_path(
                    canonical.relative_to(Path(directory)), self._runtime()
                ),
                canonical,
            )
            with self.assertRaisesRegex(
                qualification.QualificationIntegrityError, "cannot traverse"
            ):
                qualification._require_qualification_audit_path(
                    Path("logs") / ".." / canonical.name,
                    self._runtime(),
                )
            for launch_id in (None, "B" * 64, "b" * 63, "g" * 64):
                with self.subTest(launch_id=launch_id), self.assertRaises(
                    qualification.QualificationIntegrityError
                ):
                    qualification.qualification_audit_path({
                        "server_launch_id": launch_id
                    })

    def test_inconsistent_bundle_can_be_qualified_without_weakening_grader(self):
        case = qualification.QUALIFICATION_CASES[-1]
        value = {
            "claims": case.expected_scores(),
            "needs_regrade": True,
        }
        scores, needs_regrade = qualification._validate_qualification_verdict(
            case, value
        )
        self.assertEqual(scores, {"c1": 0, "c2": 1})
        self.assertTrue(needs_regrade)

    async def _run(self, directory: str, *, flip_case=None):
        clients = []
        root = Path(directory)
        runtime = self._runtime()
        output = self._output(directory)
        intent_path = qualification.qualification_intent_path(output)
        source = self._source()
        urls = [f"http://localhost:{36000 + index}/v1" for index in range(3)]

        def factory(**kwargs):
            client = FakeClient(
                **kwargs, flip_case=flip_case, intent_path=intent_path
            )
            clients.append(client)
            return client

        with (
            patch.object(qualification, "ROOT", root),
            patch.object(qualification, "AsyncOpenAI", side_effect=factory),
            patch.object(
                qualification, "local_judge_runtime_record",
                return_value=runtime,
            ),
            patch.object(
                qualification, "local_judge_runtime_sha256",
                return_value="d" * 64,
            ),
            patch.object(qualification, "source_record", return_value=source),
            patch.dict(os.environ, {"SB_VLLM_API_KEY": "test-key"}),
        ):
            artifact = await qualification._run_qualification(
                urls, output.relative_to(root)
            )
        return output, artifact, clients, runtime, source, urls

    def _validate(self, output, *, runtime, source, urls, root=None):
        output = Path(output)
        root = output.parent.parent if root is None else Path(root)
        with (
            patch.object(qualification, "ROOT", root),
            patch.object(
                qualification, "local_judge_runtime_record",
                return_value=runtime,
            ),
            patch.object(
                qualification, "local_judge_runtime_sha256",
                return_value="d" * 64,
            ),
            patch.object(qualification, "source_record", return_value=source),
        ):
            return qualification.validate_qualification_audit(
                output,
                expected_urls=urls,
                expected_source=source,
                expected_runtime=runtime,
            )

    async def test_runner_writes_complete_pass_before_returning(self):
        with tempfile.TemporaryDirectory() as directory:
            output, artifact, clients, runtime, source, urls = await self._run(
                directory
            )
            self.assertTrue(artifact["all_passed"])
            self.assertTrue(artifact["source_stable"])
            self.assertEqual(len(artifact["requests"]), 60)
            self.assertEqual(len(artifact["responses"]), 60)
            self.assertEqual(sum(client.counter for client in clients), 60)
            self.assertTrue(all(item["passed"] for item in artifact["responses"]))
            self.assertTrue(all(
                item["passed"] for item in artifact["post_qualification_health"]
            ))
            self.assertTrue(all(
                item["attempt"]["accepted"]
                for item in artifact["responses"]
            ))
            self.assertEqual(json.loads(output.read_text()), artifact)
            intent_path = qualification.qualification_intent_path(output)
            self.assertTrue(intent_path.is_file())
            intent = json.loads(intent_path.read_text())
            self.assertEqual(len(intent["requests"]), 60)
            self.assertTrue(all(
                set(item) == {"case_id", "slot", "url", "request_sha256"}
                for item in intent["requests"]
            ))
            self.assertEqual(
                intent["request_manifest_sha256"],
                qualification.sha256_json(intent["requests"]),
            )
            self.assertEqual(
                [item["request_sha256"] for item in intent["requests"]],
                [item["request_sha256"] for item in artifact["requests"]],
            )
            self.assertEqual(
                artifact["qualification_intent_sha256"],
                qualification.sha256_bytes(
                    qualification.read_artifact_bytes(intent_path)
                ),
            )
            binding = self._validate(
                output, runtime=runtime, source=source, urls=urls
            )
            relative_binding = self._validate(
                output.relative_to(Path(directory)),
                runtime=runtime,
                source=source,
                urls=urls,
                root=directory,
            )
            self.assertEqual(relative_binding, binding)
            self.assertEqual(
                binding["audit_sha256"],
                qualification.sha256_bytes(
                    qualification.read_artifact_bytes(output)
                ),
            )
            self.assertEqual(binding["intent_sha256"], artifact[
                "qualification_intent_sha256"
            ])
            self.assertEqual(binding["chat_request_count"], 60)
            self.assertEqual(len(binding["binding_sha256"]), 64)

    async def test_runner_preserves_failure_audit_before_terminal_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            output = self._output(directory)
            with self.assertRaisesRegex(SystemExit, "namespace is terminal"):
                await self._run(directory, flip_case="C01")
            artifact = json.loads(output.read_text())
            self.assertFalse(artifact["all_passed"])
            failures = [
                item for item in artifact["responses"]
                if item["case_id"] == "C01"
            ]
            self.assertEqual(len(failures), 3)
            self.assertTrue(all(not item["passed"] for item in failures))
            self.assertTrue(all(
                item["attempt"]["accepted"] is False
                and item["attempt"]["validation_error"]["type"]
                == "QualificationError"
                for item in failures
            ))
            self.assertEqual(len(artifact["responses"]), 60)
            with self.assertRaises(qualification.QualificationIntegrityError):
                self._validate(
                    output,
                    runtime=artifact["local_judge_runtime"],
                    source=artifact["source"],
                    urls=artifact["ordered_urls"],
                )

    async def test_runner_rejects_alternate_output_before_client_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = self._runtime()
            source = self._source()
            canonical = self._output(directory)
            alternate = Path("logs") / "alternate-qualification.json"
            urls = [
                f"http://localhost:{36000 + index}/v1" for index in range(3)
            ]
            with (
                patch.object(qualification, "ROOT", root),
                patch.object(
                    qualification, "local_judge_runtime_record",
                    return_value=runtime,
                ),
                patch.object(
                    qualification, "local_judge_runtime_sha256",
                    return_value="d" * 64,
                ),
                patch.object(qualification, "source_record", return_value=source),
                patch.object(qualification, "AsyncOpenAI") as constructor,
                self.assertRaisesRegex(
                    qualification.QualificationIntegrityError,
                    "canonical namespace",
                ),
            ):
                await qualification._run_qualification(urls, alternate)
            constructor.assert_not_called()
            self.assertFalse(canonical.exists())
            self.assertFalse(
                qualification.qualification_intent_path(canonical).exists()
            )
            self.assertFalse(Path(f"{root / alternate}.lock").exists())

    async def test_runner_rejects_canonical_preexisting_namespace_before_contact(self):
        with tempfile.TemporaryDirectory() as directory:
            output, _, _, runtime, source, urls = await self._run(directory)
            with (
                patch.object(qualification, "ROOT", Path(directory)),
                patch.object(
                    qualification, "local_judge_runtime_record",
                    return_value=runtime,
                ),
                patch.object(
                    qualification, "local_judge_runtime_sha256",
                    return_value="d" * 64,
                ),
                patch.object(qualification, "source_record", return_value=source),
            ):
                with patch.object(qualification, "AsyncOpenAI") as constructor:
                    with self.assertRaisesRegex(FileExistsError, "audit.*terminal"):
                        await qualification._run_qualification(urls, output)
                    constructor.assert_not_called()

                output.unlink()
                self.assertTrue(
                    qualification.qualification_intent_path(output).is_file()
                )
                with patch.object(qualification, "AsyncOpenAI") as constructor:
                    with self.assertRaisesRegex(FileExistsError, "intent.*terminal"):
                        await qualification._run_qualification(urls, output)
                    constructor.assert_not_called()

    async def test_runner_lock_prevents_concurrent_precontact_writers(self):
        with tempfile.TemporaryDirectory() as directory:
            output = self._output(directory)
            lock = Path(f"{output}.lock")
            runtime = self._runtime()
            source = self._source()
            urls = [
                f"http://localhost:{36000 + index}/v1" for index in range(3)
            ]
            with (
                patch.object(qualification, "ROOT", Path(directory)),
                patch.object(
                    qualification, "local_judge_runtime_record",
                    return_value=runtime,
                ),
                patch.object(
                    qualification, "local_judge_runtime_sha256",
                    return_value="d" * 64,
                ),
                patch.object(qualification, "source_record", return_value=source),
                qualification.exclusive_process_lock(lock),
                patch.object(qualification, "AsyncOpenAI") as constructor,
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "another process is already working"
                ):
                    await qualification._run_qualification(urls, output)
                constructor.assert_not_called()

    async def test_validator_rejects_context_and_canonical_audit_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            output, artifact, _, runtime, source, urls = await self._run(directory)
            with self.assertRaisesRegex(
                qualification.QualificationIntegrityError,
                "canonical namespace",
            ):
                self._validate(
                    output.with_name("alternate-qualification.json"),
                    runtime=runtime,
                    source=source,
                    urls=urls,
                )
            wrong_urls = [
                f"http://localhost:{36100 + index}/v1" for index in range(3)
            ]
            with self.assertRaises(qualification.QualificationIntegrityError):
                self._validate(
                    output, runtime=runtime, source=source, urls=wrong_urls
                )
            wrong_source = {**source, "git_commit": "e" * 40}
            with self.assertRaises(qualification.QualificationIntegrityError):
                self._validate(
                    output, runtime=runtime, source=wrong_source, urls=urls
                )
            wrong_runtime = {**runtime, "server_launch_id": "f" * 64}
            with self.assertRaises(qualification.QualificationIntegrityError):
                self._validate(
                    output, runtime=wrong_runtime, source=source, urls=urls
                )

            output.unlink()
            output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
            with self.assertRaisesRegex(
                qualification.QualificationIntegrityError, "canonical"
            ):
                self._validate(
                    output, runtime=runtime, source=source, urls=urls
                )

    async def test_validator_reconstructs_request_and_attempt_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            output, artifact, _, runtime, source, urls = await self._run(directory)
            artifact["responses"][0]["request_sha256"] = "0" * 64
            output.unlink()
            output.write_bytes(qualification.canonical_json_bytes(artifact))
            with self.assertRaisesRegex(
                qualification.QualificationIntegrityError,
                "reconstructed request",
            ):
                self._validate(
                    output, runtime=runtime, source=source, urls=urls
                )

    async def test_validator_rejects_intent_request_hash_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            output, _, _, runtime, source, urls = await self._run(directory)
            intent_path = qualification.qualification_intent_path(output)
            intent = json.loads(intent_path.read_text())
            intent["requests"][0]["request_sha256"] = "0" * 64
            intent["request_manifest_sha256"] = qualification.sha256_json(
                intent["requests"]
            )
            intent_path.unlink()
            intent_path.write_bytes(qualification.canonical_json_bytes(intent))
            with self.assertRaisesRegex(
                qualification.QualificationIntegrityError,
                "reconstructed request population",
            ):
                self._validate(
                    output, runtime=runtime, source=source, urls=urls
                )


if __name__ == "__main__":
    unittest.main()
