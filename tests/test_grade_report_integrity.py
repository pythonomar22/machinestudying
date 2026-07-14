from __future__ import annotations

import asyncio
from copy import deepcopy
from io import StringIO
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from studybench import grade, provenance, report
from studybench.integrity import canonical_json_bytes, sha256_json, stable_seed
from studybench.provenance import (
    _load_note,
    environment_contract_record,
    server_assignment_record,
)
from studybench.study_protocol import (
    DSPY_ADAPTER_NAME,
    DSPY_ADAPTER_POLICY,
    DSPY_REPOSITORY_TOOL_CONTRACT,
    DSPY_REQUEST_AUDIT_SCHEMA_VERSION,
    FORCED50_CONFIG_SCHEMA_VERSION,
    FORCED50_ITERATIONS,
    REACT_SAMPLING,
    SEMANTIC_SELFQUIZ_METHOD,
    SEMANTIC_SELFQUIZ_NOTE_MANIFEST_TYPE,
    SEMANTIC_SELFQUIZ_TASK_MANIFEST_TYPE,
    StudyProtocolError,
    derive_protocol_summary,
    forced50_study_question,
    openbook_attempt_protocol,
)
from studybench.tools import DSPY_READ_MAX_LINES


TEST_JUDGE_BASE_URL = "https://judge.test/v1"


def fake_grading_runtime(*, tree_sha256: str = "a" * 64) -> dict:
    packages = [{"name": "openai", "version": "1.0"}]
    pyvenv_text = "home = /usr/bin\n"
    return {
        "schema_version": 1,
        "attestation_policy": provenance.GRADING_RUNTIME_ATTESTATION_POLICY,
        "python": {
            "version": provenance.MAIN_PYTHON_VERSION,
            "implementation": "CPython",
            "executable": "/test/.venv/bin/python",
            "resolved_executable": "/usr/bin/python",
            "executable_sha256": "b" * 64,
            "prefix": "/test/.venv",
            "base_prefix": "/usr",
            "pyvenv_cfg": {
                "path": "/test/.venv/pyvenv.cfg",
                "sha256": grade.sha256_bytes(pyvenv_text.encode()),
                "bytes": len(pyvenv_text.encode()),
                "text": pyvenv_text,
            },
        },
        "packages": packages,
        "packages_sha256": sha256_json(packages),
        "runner_lock": {"schema_version": 1},
        "installed_code": {
            "schema_version": 1,
            "python_version": provenance.MAIN_PYTHON_VERSION,
            "prefix": "/test/.venv",
            "distribution_count": 1,
            "file_count": 2,
            "total_bytes": 10,
            "tree_sha256": tree_sha256,
        },
    }


def fake_local_judge_runtime(*, contract_sha256: str = "c" * 64) -> dict:
    return {
        "schema_version": 1,
        "attestation_policy": provenance.LOCAL_JUDGE_RUNTIME_ATTESTATION_POLICY,
        "environment_contract": {
            "schema_version": 1,
            "policy": provenance.ENVIRONMENT_COMPATIBILITY_POLICY,
            "sha256": contract_sha256,
        },
        "model": {
            "id": provenance.MODEL_ID,
            "revision": provenance.MODEL_REVISION,
            "cache_inventory_sha256": "d" * 64,
            "cache_file_count": 1,
            "cache_total_bytes": 10,
            "cache_tree_sha256": "e" * 64,
        },
        "server": {
            "vllm_version": provenance.VLLM_VERSION,
            "installed_inventory_sha256": "f" * 64,
            "installed_distribution_count": 1,
            "installed_file_count": 2,
            "installed_total_bytes": 10,
            "installed_tree_sha256": "1" * 64,
            "runtime_inventory_sha256": "2" * 64,
            "tensor_parallel_size": 1,
            "visible_gpu_count": 1,
            "server_count": 1,
        },
        "hardware": {
            "gpu_models": ["test-gpu"],
            "nvidia_driver": ["test-driver"],
            "gpu_profiles": [{
                "name": "test-gpu",
                "memory_mib": 1024,
                "driver_version": "test-driver",
                "count": 1,
            }],
        },
    }


def question() -> dict:
    return {
        "id": "q1",
        "topic": "testing",
        "question": "How does it work?",
        "gold_answer": "It works exactly as documented.",
        "evidence": [{
            "span_id": "s1",
            "path": "src/example.py",
            "start_line": 1,
            "end_line": 1,
            "excerpt": "0001: pass",
        }],
        "rubric": [
            {
                "claim_id": "core",
                "claim_type": "core",
                "statement": "States the core behavior.",
                "weight": 60,
                "span_ids": ["s1"],
            },
            {
                "claim_id": "detail",
                "claim_type": "supporting",
                "statement": "States the supporting detail.",
                "weight": 40,
                "span_ids": ["s1"],
            },
        ],
    }


def verdict(*, score: int = 60, duplicate: bool = False) -> dict:
    second_id = "core" if duplicate else "detail"
    return {
        "claims": [
            {"claim_id": "core", "score": 1, "rationale": "present"},
            {"claim_id": second_id, "score": 0, "rationale": "missing"},
        ],
        "question_score": score,
        "needs_regrade": False,
    }


def checker_result(compile_ok: object = True, detail: str = "ok") -> dict:
    return {
        "compile_ok": compile_ok,
        "detail": detail,
        "configuration_sha256": grade.sandbox_configuration_sha256("python"),
    }


def native_episode(*, budget: str = "direct", status: str = "ok") -> dict:
    answer = "```python\npass\n```" if status == "ok" else ""
    tool_iters = 20 if budget == "k20f" else 0
    turn_count = max(tool_iters, 1)
    prompt_parts = [90 // turn_count + (index < 90 % turn_count)
                    for index in range(turn_count)]
    completion_parts = [10 // turn_count + (index < 10 % turn_count)
                        for index in range(turn_count)]
    turns = []
    request_attempts = []
    for index, (prompt_tokens, completion_tokens) in enumerate(
            zip(prompt_parts, completion_parts, strict=True)):
        response_id = f"generation-response-{index + 1}"
        calls = ([{"name": "grep", "arguments": "{}"}] if tool_iters else [])
        turns.append({
            "response_id": response_id,
            "response_model": "generation-revision",
            "system_fingerprint": "generation-fingerprint",
            "tool_calls": calls,
            "observations": (["source"] if calls else []),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        })
        request_attempts.append({
            "logical_call": index,
            "attempt": 1,
            "status": "response",
            "request_sha256": "f" * 64,
            "response_id": response_id,
            "response_model": "generation-revision",
        })
    return {
        "task": "fake",
        "qid": "q1",
        "budget": budget,
        "rollout": 0,
        "model": "model",
        "model_revision": "revision-a",
        "harness": "native-react",
        "seed": 7,
        "status": status,
        "answer": answer,
        "n_tool_iters": tool_iters,
        "finish_catches": 0,
        "prompt_tokens": 90,
        "completion_tokens": 10,
        "total_tokens": 100,
        "gen_tokens": 10,
        "turns": turns,
        "request_attempts": request_attempts,
    }


class FakeUsage:
    def __init__(self, prompt: int = 100, completion: int = 20) -> None:
        self.values = {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }

    def model_dump(self, *, mode: str) -> dict:
        assert mode == "json"
        return dict(self.values)


class FakeCompletions:
    def __init__(self, payloads: list[dict | str | BaseException],
                 response_model: str = "judge-revision"):
        self.payloads = list(payloads)
        self.response_model = response_model
        self.calls = 0

    async def create(self, **kwargs):
        payload = self.payloads[self.calls]
        self.calls += 1
        if isinstance(payload, BaseException):
            raise payload
        content = payload if isinstance(payload, str) else json.dumps(payload)
        return SimpleNamespace(
            id=f"response-{self.calls}",
            _request_id=f"request-{self.calls}",
            model=self.response_model,
            system_fingerprint="judge-fingerprint",
            usage=FakeUsage(100 * self.calls, 20 * self.calls),
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        )


class FakeClient:
    def __init__(self, payloads: list[dict | str | BaseException],
                 response_model: str = "judge-revision"):
        self.completions = FakeCompletions(payloads, response_model)
        self.chat = SimpleNamespace(completions=self.completions)


class FailingClient:
    def __init__(self) -> None:
        async def create(**kwargs):
            raise RuntimeError("provider unavailable")

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


class FixedResponseClient:
    def __init__(self, response: object) -> None:
        self.calls = 0

        async def create(**kwargs):
            self.calls += 1
            return response

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


def fixed_response(*, usage: object = None, response_model: object = "judge-revision",
                   system_fingerprint: object = "judge-fingerprint"):
    return SimpleNamespace(
        id="response-1",
        _request_id="request-1",
        model=response_model,
        system_fingerprint=system_fingerprint,
        usage=usage,
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(verdict())))],
    )


class GradeVerdictTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grading_runtime = fake_grading_runtime()
        self.local_judge_runtime = fake_local_judge_runtime()
        self.grading_runtime_patch = patch.object(
            grade, "grading_runtime_record", return_value=self.grading_runtime
        )
        self.local_judge_runtime_patch = patch.object(
            grade,
            "local_judge_runtime_record",
            return_value=self.local_judge_runtime,
        )
        self.grading_runtime_patch.start()
        self.local_judge_runtime_patch.start()

    def tearDown(self) -> None:
        self.local_judge_runtime_patch.stop()
        self.grading_runtime_patch.stop()

    def test_preregistered_grading_policy_is_exact(self) -> None:
        document = {
            "grading_policy": {
                "grader": "openai",
                "judge_model": "gpt-5.4",
                "evidence_mode": "excerpt_evidence",
                "judge_effort": "",
                "claim_scoring": "binary_0_1",
                "question_scoring": "weighted_claim_sum",
            }
        }
        grade.validate_preregistered_grading_policy(
            document,
            grader="openai",
            judge_model="gpt-5.4",
            whole_files=False,
            effort="",
        )
        with self.assertRaisesRegex(
            grade.GradeIntegrityError, "differs from the preregistration"
        ):
            grade.validate_preregistered_grading_policy(
                document,
                grader="openai",
                judge_model="gpt-5.4",
                whole_files=True,
                effort="",
            )

    def test_grader_schedules_each_pending_episode_once_and_uses_a_lock(self) -> None:
        source = Path(grade.__file__).read_text()
        self.assertEqual(source.count("await asyncio.gather"), 1)
        self.assertIn("with exclusive_process_lock(lock_path):", source)
        self.assertIn('out_root / ".locks" / "grading.lock"', source)
        self.assertIn("await _main_async_locked(args)", source)
        self.assertIn("if gf.exists():", source)

    def test_grader_disables_hidden_sdk_retries(self) -> None:
        source = Path(grade.__file__).read_text()
        self.assertEqual(source.count("AsyncOpenAI("), 1)
        self.assertIn("max_retries=0", source)

    def test_grade_run_lock_blocks_a_second_invocation_before_preflight(self) -> None:
        args = SimpleNamespace(
            run_id="run-a",
            grade_id="grade-a",
            whole_files=False,
            judge_effort="",
            local_smoke=False,
        )
        with tempfile.TemporaryDirectory() as directory, patch.object(
            grade, "ROOT", Path(directory)
        ), patch.dict(os.environ, {"GRADER_MODEL": "openai"}):
            _, out_root = grade._grade_namespace(args)
            lock_path = out_root / ".locks" / "grading.lock"
            with grade.exclusive_process_lock(lock_path), self.assertRaisesRegex(
                RuntimeError, "already working"
            ):
                asyncio.run(grade.main_async(args))

    def test_local_smoke_selects_one_answer_only_from_a_fresh_grid(self) -> None:
        pending = [
            {"episode": {"status": "no_answer"}, "name": "zero"},
            {"episode": {"status": "ok"}, "name": "judged"},
            {"episode": {"status": "ok"}, "name": "later"},
        ]
        self.assertEqual(
            grade.select_local_smoke_record(
                pending, expected_count=3, grader="local"
            ),
            [pending[1]],
        )
        with self.assertRaisesRegex(grade.GradeIntegrityError, "fresh empty"):
            grade.select_local_smoke_record(
                pending[1:], expected_count=3, grader="local"
            )
        with self.assertRaisesRegex(grade.GradeIntegrityError, "only for local"):
            grade.select_local_smoke_record(
                pending, expected_count=3, grader="openai"
            )

    def test_openai_grader_ignores_ambient_base_url(self) -> None:
        ambient = "https://redirect.invalid/v1"
        with patch.dict(os.environ, {"OPENAI_BASE_URL": ambient}), patch.object(
            grade, "AsyncOpenAI"
        ) as constructor:
            client = grade._make_grader_client("openai", "test-key")
        self.assertIs(client, constructor.return_value)
        self.assertEqual(
            constructor.call_args.kwargs["base_url"],
            grade.CANONICAL_OPENAI_BASE_URL,
        )
        self.assertNotEqual(constructor.call_args.kwargs["base_url"], ambient)

    def test_local_grader_uses_only_the_explicit_loopback_endpoint(self) -> None:
        ambient = "https://redirect.invalid/v1"
        with patch.dict(os.environ, {"OPENAI_BASE_URL": ambient}), patch.object(
            grade, "AsyncOpenAI"
        ) as constructor:
            client = grade._make_grader_client(
                "local",
                "test-key",
                judge_base_url="http://127.0.0.1:8123/v1",
            )
        self.assertIs(client, constructor.return_value)
        self.assertEqual(
            constructor.call_args.kwargs["base_url"],
            "http://localhost:8123/v1",
        )
        self.assertNotEqual(constructor.call_args.kwargs["base_url"], ambient)

    def test_local_grader_requires_authenticated_pinned_launcher_identity(self) -> None:
        key = "ephemeral-local-key"
        key_sha256 = grade.sha256_bytes(key.encode())
        environment = {
            "SB_VLLM_API_KEY": key,
            "SB_VLLM_API_KEY_SHA256": key_sha256,
            "SB_SERVER_LAUNCH_ID": key_sha256,
            "SB_MODEL_ID": grade.LOCAL_GRADER_MODEL,
            "SB_MODEL_REVISION": grade.LOCAL_GRADER_MODEL_REVISION,
            "BASE_URLS": "http://127.0.0.1:8123/v1",
        }
        with patch.dict(os.environ, environment, clear=True):
            grade._validate_local_grader_environment(
                "http://localhost:8123/v1"
            )
        for field, invalid in (
            ("SB_VLLM_API_KEY_SHA256", "0" * 64),
            ("SB_MODEL_REVISION", "0" * 40),
            ("BASE_URLS", "https://remote.invalid/v1"),
        ):
            with self.subTest(field=field), patch.dict(
                os.environ, {**environment, field: invalid}, clear=True
            ), self.assertRaises(grade.GradeIntegrityError):
                grade._validate_local_grader_environment(
                    "http://localhost:8123/v1"
                )

    def test_local_grader_request_options_are_fixed_and_recorded(self) -> None:
        calls = []

        async def create(**kwargs):
            calls.append(kwargs)
            return fixed_response(
                usage=FakeUsage(), response_model=grade.LOCAL_GRADER_MODEL)

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        corpus = SimpleNamespace(name="fake", display="Fake", language="python")
        with patch("studybench.grade.sandbox.check", return_value=checker_result()):
            result = asyncio.run(grade.grade_episode(
                client,
                grade.LOCAL_GRADER_MODEL,
                corpus,
                question(),
                native_episode(),
                judge_base_url="http://localhost:8123/v1",
                episode_sha256="a" * 64,
                grading_spec_sha256="b" * 64,
            ))
        self.assertEqual(len(calls), 1)
        for key, value in grade.LOCAL_GRADER_REQUEST_OPTIONS.items():
            self.assertEqual(calls[0][key], value)
        self.assertNotIn("reasoning_effort", calls[0])
        self.assertEqual(
            result["judge_request_options"], grade.LOCAL_GRADER_REQUEST_OPTIONS)
        self.assertFalse(result["claim_ready"])
        self.assertTrue(result["local_proxy"])
        self.assertEqual(
            result["judge_endpoint_identity"],
            grade.LOCAL_GRADER_ENDPOINT_IDENTITY,
        )
        with self.assertRaisesRegex(grade.GradeIntegrityError, "must be empty"):
            grade._judge_request_options(grade.LOCAL_GRADER_MODEL, "high")

    def test_grading_spec_binds_explicit_judge_endpoint(self) -> None:
        corpus = SimpleNamespace(name="fake", display="Fake", language="python")
        first = grade.grade_spec_sha256(
            corpus,
            question(),
            "judge",
            judge_base_url="https://judge-a.test/v1",
        )
        second = grade.grade_spec_sha256(
            corpus,
            question(),
            "judge",
            judge_base_url="https://judge-b.test/v1",
        )
        self.assertNotEqual(first, second)

    def test_local_grading_spec_treats_loopback_port_as_transport(self) -> None:
        corpus = SimpleNamespace(name="fake", display="Fake", language="python")
        first = grade.grade_spec_sha256(
            corpus,
            question(),
            grade.LOCAL_GRADER_MODEL,
            judge_base_url="http://localhost:8123/v1",
        )
        second = grade.grade_spec_sha256(
            corpus,
            question(),
            grade.LOCAL_GRADER_MODEL,
            judge_base_url="http://127.0.0.1:9123/v1",
        )
        self.assertEqual(first, second)

    def test_grading_runtime_attestation_compacts_the_full_byte_inventory(self) -> None:
        runner = {
            "python": self.grading_runtime["python"],
            "packages": self.grading_runtime["packages"],
            "packages_sha256": self.grading_runtime["packages_sha256"],
        }
        inventory = {
            **self.grading_runtime["installed_code"],
            "distributions": [{"name": "openai", "version": "1.0"}],
        }
        provenance._grading_runtime_record_bytes.cache_clear()
        try:
            with (
                patch.object(
                    provenance, "_runner_environment_record", return_value=runner
                ),
                patch.object(
                    provenance,
                    "_runner_lock_attestation",
                    return_value=self.grading_runtime["runner_lock"],
                ),
                patch.object(provenance, "_runner_lock_is_valid", return_value=True),
                patch.object(
                    provenance,
                    "installed_distribution_inventory",
                    return_value=inventory,
                ),
                patch.object(
                    provenance, "_validate_installed_distribution_inventory"
                ),
            ):
                observed = provenance.grading_runtime_record()
        finally:
            provenance._grading_runtime_record_bytes.cache_clear()
        self.assertEqual(
            observed["installed_code"]["tree_sha256"],
            inventory["tree_sha256"],
        )
        self.assertNotIn("distributions", observed["installed_code"])
        self.assertNotIn(
            '"distributions":', canonical_json_bytes(observed).decode()
        )

    def test_grading_spec_binds_installed_package_bytes(self) -> None:
        corpus = SimpleNamespace(name="fake", display="Fake", language="python")
        first_runtime = fake_grading_runtime(tree_sha256="1" * 64)
        second_runtime = fake_grading_runtime(tree_sha256="2" * 64)
        first = grade.grade_spec_sha256(
            corpus,
            question(),
            "judge",
            judge_base_url=TEST_JUDGE_BASE_URL,
            grading_runtime=first_runtime,
        )
        second = grade.grade_spec_sha256(
            corpus,
            question(),
            "judge",
            judge_base_url=TEST_JUDGE_BASE_URL,
            grading_runtime=second_runtime,
        )
        self.assertNotEqual(first, second)

    def test_grade_records_runtime_digest_and_rejects_runtime_drift(self) -> None:
        corpus = SimpleNamespace(name="fake", display="Fake", language="python")
        episode = native_episode(status="no_answer")
        stored = asyncio.run(grade.grade_episode(
            FakeClient([]),
            "judge",
            corpus,
            question(),
            episode,
            judge_base_url=TEST_JUDGE_BASE_URL,
            episode_sha256="a" * 64,
            grading_spec_sha256="b" * 64,
            grading_runtime=self.grading_runtime,
        ))
        self.assertEqual(
            stored["grading_runtime_sha256"],
            provenance.grading_runtime_sha256(self.grading_runtime),
        )
        self.assertNotIn("grading_runtime", stored)
        with self.assertRaisesRegex(
            grade.GradeIntegrityError, "different Python/package runtime"
        ):
            grade.validate_stored_grade(
                stored,
                question(),
                episode,
                episode_sha256="a" * 64,
                grading_spec_sha256="b" * 64,
                judge_model="judge",
                judge_base_url=TEST_JUDGE_BASE_URL,
                corpus=corpus,
                recheck_checker=False,
                grading_runtime=fake_grading_runtime(tree_sha256="2" * 64),
            )

    def test_local_grading_spec_binds_substantive_runtime_not_transport(self) -> None:
        corpus = SimpleNamespace(name="fake", display="Fake", language="python")
        common = {
            "grading_runtime": self.grading_runtime,
            "judge_base_url": "http://localhost:8123/v1",
        }
        first = grade.grade_spec_sha256(
            corpus,
            question(),
            grade.LOCAL_GRADER_MODEL,
            local_judge_runtime=fake_local_judge_runtime(
                contract_sha256="1" * 64
            ),
            **common,
        )
        second = grade.grade_spec_sha256(
            corpus,
            question(),
            grade.LOCAL_GRADER_MODEL,
            local_judge_runtime=fake_local_judge_runtime(
                contract_sha256="2" * 64
            ),
            **common,
        )
        self.assertNotEqual(first, second)

    def test_local_runtime_attestation_discards_large_launcher_inventories(self) -> None:
        environment = {
            "vllm_environment": {
                "sha256": "1" * 64,
                "inventory": {
                    "distribution_count": 2,
                    "file_count": 3,
                    "total_bytes": 100,
                    "tree_sha256": "2" * 64,
                    "distributions": [{"large": "inventory"}],
                },
            },
            "vllm_runtime": {"sha256": "3" * 64},
            "model_cache": {
                "sha256": "4" * 64,
                "inventory": {
                    "file_count": 1,
                    "total_bytes": 50,
                    "tree_sha256": "5" * 64,
                    "files": [{"large": "inventory"}],
                },
            },
            "allocation": {"inventory": {"gpus": [{
                "name": "test-gpu",
                "memory_mib": 1024,
                "driver_version": "test-driver",
            }]}},
            "model_id": provenance.MODEL_ID,
            "model_revision": provenance.MODEL_REVISION,
            "vllm_version": provenance.VLLM_VERSION,
            "tensor_parallel_size": "1",
            "visible_gpu_count": "1",
            "server_count": "1",
            "gpu_models": ["test-gpu"],
            "nvidia_driver": ["test-driver"],
        }
        provenance._local_judge_runtime_record_bytes.cache_clear()
        try:
            with (
                patch.object(provenance, "environment_record", return_value=environment),
                patch.object(
                    provenance, "environment_is_claim_ready", return_value=True
                ),
            ):
                observed = provenance.local_judge_runtime_record()
        finally:
            provenance._local_judge_runtime_record_bytes.cache_clear()
        serialized = canonical_json_bytes(observed).decode()
        self.assertNotIn('"distributions":', serialized)
        self.assertNotIn('"files":', serialized)
        self.assertEqual(
            observed["environment_contract"]["sha256"],
            provenance.environment_contract_record(environment)["sha256"],
        )

    def test_clean_source_records_are_exact_and_self_consistent(self) -> None:
        valid = {
            "git_commit": "a" * 40,
            "dirty": False,
            "files": {},
            "tree_sha256": sha256_json({}),
        }
        grade._validate_source_record(valid, label="test")
        for mutation in (
            lambda value: value.update(dirty=True),
            lambda value: value.update(tree_sha256="0" * 64),
            lambda value: value.update(extra=True),
        ):
            invalid = deepcopy(valid)
            mutation(invalid)
            with self.assertRaises(grade.GradeIntegrityError):
                grade._validate_source_record(invalid, label="test")

    def test_exact_unique_rubric_ids_are_required(self) -> None:
        row = question()
        with self.assertRaises(grade.GradeIntegrityError):
            grade.validate_verdict(row, verdict(duplicate=True))
        missing = verdict()
        missing["claims"] = missing["claims"][:1]
        with self.assertRaises(grade.GradeIntegrityError):
            grade.validate_verdict(row, missing)
        extra = verdict()
        extra["claims"].append(
            {"claim_id": "extra", "score": 1, "rationale": "not in rubric"})
        with self.assertRaises(grade.GradeIntegrityError):
            grade.validate_verdict(row, extra)
        extra_field = verdict()
        extra_field["unrecognized"] = True
        with self.assertRaises(grade.GradeIntegrityError):
            grade.validate_verdict(row, extra_field)

    def test_question_score_is_recomputed_and_claims_are_canonicalized(self) -> None:
        row = question()
        wrong = verdict(score=100)
        with self.assertRaises(grade.GradeIntegrityError):
            grade.validate_verdict(row, wrong)
        out_of_order = verdict()
        out_of_order["claims"].reverse()
        claims, scores = grade.validate_verdict(row, out_of_order)
        self.assertEqual([claim["claim_id"] for claim in claims], ["core", "detail"])
        self.assertEqual(scores, {"core": 1, "detail": 0})

    def test_missing_score_cannot_silently_become_zero(self) -> None:
        with self.assertRaises(grade.GradeIntegrityError):
            grade.score_from_claims(question(), {"core": 1}, compile_ok=True)

    def test_token_scalar_must_match_turn_usage(self) -> None:
        episode = native_episode()
        episode["gen_tokens"] = 999
        with self.assertRaises(grade.GradeIntegrityError):
            grade.validate_episode(episode, question())
        episode = native_episode()
        episode["turns"][0]["completion_tokens"] = 9
        with self.assertRaises(grade.GradeIntegrityError):
            grade.validate_episode(episode, question())
        episode = native_episode()
        episode["turns"][0]["total_tokens"] = 99
        episode["total_tokens"] = 99
        with self.assertRaises(grade.GradeIntegrityError):
            grade.validate_episode(episode, question())
        episode = native_episode()
        episode["request_attempts"][0]["response_id"] = "different"
        with self.assertRaises(grade.GradeIntegrityError):
            grade.validate_episode(episode, question())
        episode = native_episode()
        del episode["request_attempts"][0]["request_sha256"]
        with self.assertRaises(grade.GradeIntegrityError):
            grade.validate_episode(episode, question())

    def test_tool_counters_require_one_observed_call_per_iteration(self) -> None:
        episode = native_episode(budget="k20f")
        grade.validate_episode(episode, question())
        mismatched = deepcopy(episode)
        mismatched["n_tool_iters"] = 19
        with self.assertRaises(grade.GradeIntegrityError):
            grade.validate_episode(mismatched, question())
        missing_observation = deepcopy(episode)
        missing_observation["turns"][0]["observations"] = []
        with self.assertRaises(grade.GradeIntegrityError):
            grade.validate_episode(missing_observation, question())
        parallel = deepcopy(episode)
        parallel["turns"][0]["tool_calls"].append(
            {"name": "glob", "arguments": "{}"})
        parallel["turns"][0]["observations"].append("more source")
        with self.assertRaises(grade.GradeIntegrityError):
            grade.validate_episode(parallel, question())

    def test_invalid_checker_result_stops_before_judge_request(self) -> None:
        client = FakeClient([verdict()])
        corpus = SimpleNamespace(name="fake", display="Fake", language="python")
        with patch(
            "studybench.grade.sandbox.check",
            return_value=checker_result("not-a-boolean", "broken checker"),
        ):
            with self.assertRaisesRegex(
                grade.GradeIntegrityError, "judge was not contacted"
            ):
                asyncio.run(grade.grade_episode(
                    client, "judge", corpus, question(), native_episode(),
                    judge_base_url=TEST_JUDGE_BASE_URL,
                    episode_sha256="a" * 64, grading_spec_sha256="b" * 64,
                ))
        self.assertEqual(client.completions.calls, 0)

    def test_bundled_note_dependencies_and_audit_integer_types_are_checked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            bundle_root = Path("inputs/provenance")
            relative = "r1/attempt.json"
            data = b"immutable construction input\n"
            digest = grade.sha256_bytes(data)
            inventory = {relative: {"sha256": digest, "bytes": len(data)}}
            snapshot = bundle_root / "construction" / relative
            snapshot_path = run_root / snapshot
            snapshot_path.parent.mkdir(parents=True)
            snapshot_path.write_bytes(data)
            bundle = {"construction_artifacts": {
                "root": str(bundle_root / "construction"),
                "inventory_sha256": sha256_json(inventory),
                "artifacts": {relative: {
                    "sha256": digest,
                    "bytes": len(data),
                    "snapshot": str(snapshot),
                }},
            }}
            loaded = grade._load_bundled_construction_dependencies(
                run_root, bundle, bundle_root, inventory, sha256_json(inventory))
            self.assertEqual(loaded, {relative: data})
            snapshot_path.write_bytes(b"tampered\n")
            with self.assertRaises(grade.GradeIntegrityError):
                grade._load_bundled_construction_dependencies(
                    run_root, bundle, bundle_root, inventory, sha256_json(inventory))

        valid = (
            {"round": 1},
            {"round": 1},
            {
                "schema_version": 1,
                "round": 4,
                "blinding_preserved": True,
                "reviewer_independent": True,
            },
            {"schema_version": 1},
        )
        grade._validate_human_audit_integer_fields(*valid)
        for record_index, field in (
            (0, "round"),
            (1, "round"),
            (2, "schema_version"),
            (2, "round"),
            (3, "schema_version"),
        ):
            with self.subTest(record_index=record_index, field=field):
                invalid = deepcopy(valid)
                invalid[record_index][field] = True
                with self.assertRaisesRegex(
                    grade.GradeIntegrityError, "must be JSON integers"
                ):
                    grade._validate_human_audit_integer_fields(*invalid)
        for field in ("blinding_preserved", "reviewer_independent"):
            with self.subTest(field=field):
                invalid = deepcopy(valid)
                invalid[2][field] = 1
                with self.assertRaisesRegex(
                    grade.GradeIntegrityError, "must be JSON booleans"
                ):
                    grade._validate_human_audit_integer_fields(*invalid)

    def test_invalid_attempt_is_audited_before_valid_retry(self) -> None:
        client = FakeClient([verdict(duplicate=True), verdict()])
        corpus = SimpleNamespace(name="fake", display="Fake", language="python")
        with patch("studybench.grade.sandbox.check", return_value=checker_result()):
            result = asyncio.run(grade.grade_episode(
                client, "judge", corpus, question(), native_episode(),
                judge_base_url=TEST_JUDGE_BASE_URL,
                episode_sha256="a" * 64, grading_spec_sha256="b" * 64,
            ))
        self.assertEqual(client.completions.calls, 2)
        self.assertEqual(result["question_score"], 60)
        self.assertEqual(result["judge_accepted_attempt"], 2)
        self.assertFalse(result["judge_attempts"][0]["accepted"])
        self.assertEqual(
            result["judge_attempts"][0]["validation_error"]["type"],
            "GradeIntegrityError",
        )
        self.assertIsInstance(result["judge_attempts"][0]["invalid_content"], str)
        accepted_content = result["judge_accepted_content"]
        self.assertEqual(
            grade.sha256_bytes(accepted_content.encode("utf-8")),
            result["judge_attempts"][-1]["content_sha256"],
        )
        self.assertEqual(json.loads(accepted_content), verdict())
        self.assertEqual(result["judge_response_model"], "judge-revision")
        self.assertEqual(result["judge_usage_total"]["total_tokens"], 360)

    def test_no_answer_grade_has_no_accepted_judge_content(self) -> None:
        corpus = SimpleNamespace(name="fake", display="Fake", language="python")
        result = asyncio.run(grade.grade_episode(
            FakeClient([]),
            "judge",
            corpus,
            question(),
            native_episode(status="no_answer"),
            judge_base_url=TEST_JUDGE_BASE_URL,
            episode_sha256="a" * 64,
            grading_spec_sha256="b" * 64,
        ))
        self.assertIsNone(result["judge_accepted_content"])
        self.assertEqual(result["judge_attempts"], [])

    def test_judge_intent_precedes_contact_and_is_port_stable(self) -> None:
        corpus = SimpleNamespace(name="fake", display="Fake", language="python")
        episode = native_episode()
        source_episode = "runs/run-a/fake/direct/r0/q1.json"
        episode_digest = "a" * 64
        spec_digest = "b" * 64
        runtime_digest = provenance.grading_runtime_sha256(self.grading_runtime)
        local_digest = provenance.local_judge_runtime_sha256(
            self.local_judge_runtime
        )
        client = FakeClient(
            [verdict()], response_model=grade.LOCAL_GRADER_MODEL
        )
        with tempfile.TemporaryDirectory() as directory:
            out_root = Path(directory)

            def write_intent(prompt_digest: str) -> str:
                self.assertEqual(client.completions.calls, 0)
                _, digest = grade.write_judge_attempt_intent(
                    out_root,
                    source_episode=source_episode,
                    episode=episode,
                    episode_sha256=episode_digest,
                    grading_spec_sha256=spec_digest,
                    grading_runtime_sha256=runtime_digest,
                    local_judge_runtime_sha256=local_digest,
                    judge_model=grade.LOCAL_GRADER_MODEL,
                    judge_base_url="http://localhost:8123/v1",
                    judge_prompt_sha256=prompt_digest,
                )
                return digest

            with patch(
                "studybench.grade.sandbox.check", return_value=checker_result()
            ):
                stored = asyncio.run(grade.grade_episode(
                    client,
                    grade.LOCAL_GRADER_MODEL,
                    corpus,
                    question(),
                    episode,
                    judge_base_url="http://localhost:8123/v1",
                    episode_sha256=episode_digest,
                    grading_spec_sha256=spec_digest,
                    grading_runtime=self.grading_runtime,
                    local_judge_runtime=self.local_judge_runtime,
                    judge_attempt_intent_writer=write_intent,
                ))
            self.assertEqual(client.completions.calls, 1)
            intent_path = grade._judge_attempt_intent_path(out_root, episode)
            intent = json.loads(intent_path.read_bytes())
            self.assertNotIn("judge_base_url", intent)
            self.assertEqual(
                intent["judge_endpoint_identity"],
                grade.LOCAL_GRADER_ENDPOINT_IDENTITY,
            )
            observed = grade.validate_judge_attempt_intent(
                out_root,
                source_episode=source_episode,
                episode=episode,
                episode_sha256=episode_digest,
                grading_spec_sha256=spec_digest,
                grading_runtime_sha256=runtime_digest,
                local_judge_runtime_sha256=local_digest,
                judge_model=grade.LOCAL_GRADER_MODEL,
                judge_base_url="http://localhost:9123/v1",
                judge_prompt_sha256=stored["judge_prompt_sha256"],
            )
            self.assertIsNotNone(observed)
            self.assertEqual(
                stored["judge_attempt_intent_sha256"], observed[1]
            )
            stored["source_episode"] = source_episode
            grade.validate_stored_grade(
                stored,
                question(),
                episode,
                episode_sha256=episode_digest,
                grading_spec_sha256=spec_digest,
                judge_model=grade.LOCAL_GRADER_MODEL,
                judge_base_url="http://localhost:8123/v1",
                corpus=corpus,
                source_episode=source_episode,
                recheck_checker=False,
                grading_runtime=self.grading_runtime,
                local_judge_runtime=self.local_judge_runtime,
            )

    def test_failed_judge_audit_requires_prior_intent(self) -> None:
        corpus = SimpleNamespace(name="fake", display="Fake", language="python")
        episode = native_episode()
        source_episode = "runs/run-a/fake/direct/r0/q1.json"
        runtime_digest = provenance.grading_runtime_sha256(self.grading_runtime)
        client = FakeClient([RuntimeError("provider unavailable")])
        with tempfile.TemporaryDirectory() as directory:
            out_root = Path(directory)

            def write_intent(prompt_digest: str) -> str:
                _, digest = grade.write_judge_attempt_intent(
                    out_root,
                    source_episode=source_episode,
                    episode=episode,
                    episode_sha256="a" * 64,
                    grading_spec_sha256="b" * 64,
                    grading_runtime_sha256=runtime_digest,
                    local_judge_runtime_sha256=None,
                    judge_model="judge",
                    judge_base_url=TEST_JUDGE_BASE_URL,
                    judge_prompt_sha256=prompt_digest,
                )
                return digest

            with patch(
                "studybench.grade.sandbox.check", return_value=checker_result()
            ), self.assertRaises(grade.JudgeAttemptsFailed) as caught:
                asyncio.run(grade.grade_episode(
                    client,
                    "judge",
                    corpus,
                    question(),
                    episode,
                    judge_base_url=TEST_JUDGE_BASE_URL,
                    episode_sha256="a" * 64,
                    grading_spec_sha256="b" * 64,
                    grading_runtime=self.grading_runtime,
                    judge_attempt_intent_writer=write_intent,
                ))
            intent_digest = caught.exception.audit[
                "judge_attempt_intent_sha256"
            ]
            audit_path = grade.write_failed_judge_audit(
                out_root, source_episode, caught.exception.audit
            )
            self.assertEqual(
                grade.existing_failed_judge_audit(
                    out_root,
                    source_episode=source_episode,
                    episode=episode,
                    episode_sha256="a" * 64,
                    grading_spec_sha256="b" * 64,
                    grading_runtime_sha256=runtime_digest,
                    judge_model="judge",
                    judge_prompt_sha256=caught.exception.audit[
                        "judge_prompt_sha256"
                    ],
                    judge_attempt_intent_sha256=intent_digest,
                    require_judge_attempt_intent=True,
                ),
                audit_path,
            )

    def test_second_invalid_attempt_is_fatal(self) -> None:
        client = FakeClient([verdict(duplicate=True), verdict(score=100)])
        corpus = SimpleNamespace(name="fake", display="Fake", language="python")
        with patch("studybench.grade.sandbox.check", return_value=checker_result()):
            with self.assertRaises(grade.JudgeAttemptsFailed) as caught:
                asyncio.run(grade.grade_episode(
                    client, "judge", corpus, question(), native_episode(),
                    judge_base_url=TEST_JUDGE_BASE_URL,
                    episode_sha256="a" * 64, grading_spec_sha256="b" * 64,
                ))
        self.assertEqual(client.completions.calls, 2)
        self.assertEqual(caught.exception.audit["judge_attempt_count"], 2)
        self.assertEqual(caught.exception.audit["judge_usage_total"]["total_tokens"], 360)
        with tempfile.TemporaryDirectory() as directory:
            path = grade.write_failed_judge_audit(
                Path(directory), "runs/run-a/fake/direct/r0/q1.json",
                caught.exception.audit,
            )
            self.assertTrue(path.is_file())
            self.assertNotIn("claims", json.loads(path.read_bytes()))
            self.assertEqual(
                grade.existing_failed_judge_audit(
                    Path(directory),
                    source_episode="runs/run-a/fake/direct/r0/q1.json",
                    episode=native_episode(),
                    episode_sha256="a" * 64,
                    grading_spec_sha256="b" * 64,
                ),
                path,
            )

    def test_first_request_failure_has_an_immutable_audit_and_no_usage_claim(self) -> None:
        corpus = SimpleNamespace(name="fake", display="Fake", language="python")
        with patch(
            "studybench.grade.sandbox.check",
            return_value=checker_result(False, "unavailable"),
        ):
            with self.assertRaises(grade.JudgeAttemptsFailed) as caught:
                asyncio.run(grade.grade_episode(
                    FailingClient(), "judge", corpus, question(), native_episode(),
                    judge_base_url=TEST_JUDGE_BASE_URL,
                    episode_sha256="a" * 64, grading_spec_sha256="b" * 64,
                ))
        audit = caught.exception.audit
        self.assertEqual(audit["judge_request_attempt_count"], 1)
        self.assertEqual(audit["judge_attempt_count"], 0)
        self.assertEqual(
            audit["judge_usage_status"],
            "unavailable-for-request-without-response",
        )
        self.assertIsNone(audit["judge_usage_total"])
        self.assertEqual(audit["judge_usage_known_total"]["total_tokens"], 0)

    def test_request_failure_after_a_response_keeps_only_a_known_lower_bound(self) -> None:
        corpus = SimpleNamespace(name="fake", display="Fake", language="python")
        client = FakeClient([verdict(duplicate=True), RuntimeError("provider unavailable")])
        with patch("studybench.grade.sandbox.check", return_value=checker_result()):
            with self.assertRaises(grade.JudgeAttemptsFailed) as caught:
                asyncio.run(grade.grade_episode(
                    client, "judge", corpus, question(), native_episode(),
                    judge_base_url=TEST_JUDGE_BASE_URL,
                    episode_sha256="a" * 64, grading_spec_sha256="b" * 64,
                ))
        audit = caught.exception.audit
        self.assertEqual(client.completions.calls, 2)
        self.assertEqual(audit["judge_request_attempt_count"], 2)
        self.assertEqual(audit["judge_attempt_count"], 1)
        self.assertEqual(
            audit["judge_usage_status"],
            "unavailable-for-request-without-response",
        )
        self.assertIsNone(audit["judge_usage_total"])
        self.assertEqual(audit["judge_usage_known_total"]["total_tokens"], 120)

    def test_incomplete_response_usage_is_audited_without_retry_or_zero(self) -> None:
        corpus = SimpleNamespace(name="fake", display="Fake", language="python")
        malformed = FakeUsage()
        malformed.values["total_tokens"] = 999
        for label, usage in (("missing", None), ("inconsistent", malformed)):
            with self.subTest(label=label):
                client = FixedResponseClient(fixed_response(usage=usage))
                with patch(
                    "studybench.grade.sandbox.check", return_value=checker_result()
                ):
                    with self.assertRaises(grade.JudgeAttemptsFailed) as caught:
                        asyncio.run(grade.grade_episode(
                            client, "judge", corpus, question(), native_episode(),
                            judge_base_url=TEST_JUDGE_BASE_URL,
                            episode_sha256="a" * 64, grading_spec_sha256="b" * 64,
                        ))
                self.assertEqual(client.calls, 1)
                audit = caught.exception.audit
                self.assertEqual(audit["judge_request_attempt_count"], 1)
                self.assertEqual(audit["judge_attempt_count"], 1)
                self.assertEqual(
                    audit["judge_usage_status"],
                    "unavailable-for-response-without-usage",
                )
                self.assertIsNone(audit["judge_usage_total"])
                self.assertEqual(audit["judge_usage_known_total"]["total_tokens"], 0)
                attempt = audit["judge_attempts"][0]
                self.assertIsNone(attempt["usage"])
                self.assertEqual(attempt["usage_status"], "unavailable")
                self.assertIn("usage", attempt["incomplete_response"])
                self.assertIsInstance(attempt["invalid_content"], str)
                grade.validate_judge_attempt_record(attempt, 1, accepted=False)

    def test_incomplete_response_identity_is_retained_and_fatal(self) -> None:
        corpus = SimpleNamespace(name="fake", display="Fake", language="python")
        client = FixedResponseClient(
            fixed_response(usage=FakeUsage(), response_model={"invalid": "model"}))
        with patch("studybench.grade.sandbox.check", return_value=checker_result()):
            with self.assertRaises(grade.JudgeAttemptsFailed) as caught:
                asyncio.run(grade.grade_episode(
                    client, "judge", corpus, question(), native_episode(),
                    judge_base_url=TEST_JUDGE_BASE_URL,
                    episode_sha256="a" * 64, grading_spec_sha256="b" * 64,
                ))
        self.assertEqual(client.calls, 1)
        audit = caught.exception.audit
        attempt = audit["judge_attempts"][0]
        self.assertIsNone(attempt["response_model"])
        observation = attempt["incomplete_response"]["response_model"]
        self.assertEqual(observation["json_value"], {"invalid": "model"})
        self.assertEqual(audit["judge_usage_status"], "complete")
        self.assertEqual(audit["judge_usage_total"]["total_tokens"], 120)
        grade.validate_judge_attempt_record(attempt, 1, accepted=False)

    def test_missing_judge_system_fingerprint_is_explicit_but_not_fabricated(self) -> None:
        corpus = SimpleNamespace(name="fake", display="Fake", language="python")
        client = FixedResponseClient(fixed_response(
            usage=FakeUsage(), system_fingerprint=None))
        with patch("studybench.grade.sandbox.check", return_value=checker_result()):
            result = asyncio.run(grade.grade_episode(
                client, "judge", corpus, question(), native_episode(),
                judge_base_url=TEST_JUDGE_BASE_URL,
                episode_sha256="a" * 64, grading_spec_sha256="b" * 64,
            ))
        attempt = result["judge_attempts"][0]
        self.assertEqual(attempt["system_fingerprint_status"], "unavailable")
        self.assertIsNone(attempt["system_fingerprint"])
        self.assertEqual(
            attempt["system_fingerprint_observation"]["json_value"], None)
        grade.validate_judge_attempt_record(attempt, 1, accepted=True)


class ReportMathTests(unittest.TestCase):
    def test_expertise_matches_appendix_c_worked_example(self) -> None:
        points = [(5_000, 10), (10_000, 20), (20_000, 30), (100_000, 40)]
        self.assertAlmostEqual(report.expertise(points), 10.8)

    def test_expertise_matches_paper_dspy_base(self) -> None:
        points = [(4_100, 3.3), (7_900, 8.6), (8_600, 9.6), (34_600, 29.4)]
        self.assertEqual(round(report.expertise(points), 2), 6.49)

    def test_expertise_uses_best_so_far_and_holds_the_tail(self) -> None:
        points = [(6_000, 10), (1_000, 50)]
        self.assertAlmostEqual(report.expertise(points), 50.0)

    def test_checker_availability_projection_is_explicit_and_nonmutating(self) -> None:
        population = {
            budget: [{
                "qid": "q1",
                "lenient": 60,
                "strict": 0,
                "cores_ok": False,
                "compile_check": {"compile_ok": False},
                "gen_tokens": 4_000,
                "episode_status": "ok",
            }]
            for budget in report.BUDGET_ORDER
        }
        aggregate = report.aggregate_population(population)
        bootstrap = report.bootstrap_population(population, 3, seed=1)

        unavailable = report.reportable_aggregate(
            aggregate, checker_ready=False
        )
        for budget in report.BUDGET_ORDER:
            self.assertEqual(unavailable["budgets"][budget]["len_cc"], 0)
            self.assertIsNone(unavailable["budgets"][budget]["strict"])
            self.assertIsNone(unavailable["budgets"][budget]["compile_rate"])
            self.assertEqual(unavailable["budgets"][budget]["lenient"], 60)
            self.assertEqual(aggregate["budgets"][budget]["strict"], 0)
        self.assertIsNone(unavailable["expertise_strict"])
        self.assertEqual(report.reportable_bootstrap(
            bootstrap, checker_ready=False
        )["wauc_cc"], bootstrap["wauc_cc"])
        self.assertIsNotNone(bootstrap["wauc_cc"])

        self.assertEqual(
            report.reportable_aggregate(aggregate, checker_ready=True),
            aggregate,
        )
        self.assertEqual(
            report.reportable_bootstrap(bootstrap, checker_ready=True),
            bootstrap,
        )
        with self.assertRaises(report.ReportIntegrityError):
            report.reportable_aggregate(aggregate, checker_ready=1)


class EvaluationFixture:
    def __init__(
        self, root: Path, *, local: bool = False, task: str = "fake"
    ) -> None:
        self.root = root
        self.local = local
        self.task = task
        self.run_id = "run-a"
        self.judge_model = (
            grade.LOCAL_GRADER_MODEL if local else "gpt-5.4"
        )
        self.judge_base_url = (
            "http://localhost:8123/v1" if local else None
        )
        self.judge_dir = (
            "local-qwen3.5-9b-excerpts" if local else "gpt-5.4-excerpts"
        )
        self.corpus = SimpleNamespace(
            name=task,
            display="DSPy" if task == "dspy" else "Fake",
            repo=root / "corpus",
            roots=("src",),
            language="python",
            commit="c" * 40,
            code_suffixes=(".py",),
        )
        self.questions = [question()]
        self.run_root = root / "runs" / self.run_id
        self.run_task_root = self.run_root / task
        self.grade_root = root / "grades" / self.run_id / self.judge_dir
        self.expected = [
            f"{budget}/r0/q1.json" for budget in report.BUDGET_ORDER
        ]
        self.manifest = self._manifest()
        self.run_task_root.mkdir(parents=True)
        (self.run_task_root / "manifest.json").write_bytes(
            canonical_json_bytes(self.manifest))
        self.manifest_sha256 = grade.sha256_bytes(
            (self.run_task_root / "manifest.json").read_bytes())
        self._write_population()

    def _manifest(self) -> dict:
        seed_group = "paired-a"
        episode_seeds = {}
        for relative in self.expected:
            budget, _, _ = relative.split("/")
            episode_seeds[relative] = stable_seed(
                11, "native-react", seed_group, self.task, "q1", budget, 0)
        spec = {
            "schema_version": 1,
            "run_id": self.run_id,
            "task": self.task,
            "purpose": "confirmatory",
            "claim_ready": True,
            "harness": "native-react",
            "model": "model",
            "model_revision": "revision-a",
            "sampling": {"temperature": 0},
            "master_seed": 11,
            "seed_policy": {
                "algorithm": "sha256-canonical-json-mod-2147483647",
                "namespace": "native-react",
                "seed_group": seed_group,
                "ordered_parts": [
                    "master_seed", "namespace", "seed_group", "task", "qid",
                    "budget", "rollout",
                ],
                "episode_seeds": episode_seeds,
            },
            "server_assignment": server_assignment_record(self.expected, 1),
            "budgets": report.BUDGET_ORDER,
            "rollouts": 1,
            "questions": [{
                "id": "q1",
                "sha256": sha256_json(self.questions[0]),
                "question_text_sha256": grade.sha256_bytes(
                    self.questions[0]["question"].encode("utf-8")),
            }],
            "question_bundle_sha256": sha256_json(self.questions),
            "prompt_policy": {
                "note_prefix_template": None,
                "presented_prompt_sha256": {
                    "q1": grade.sha256_bytes(
                        self.questions[0]["question"].encode("utf-8")),
                },
            },
            "expected_episodes": self.expected,
            "failure_policy": {
                "model_no_answer": "intention-to-run_zero",
                "infrastructure_error": "invalid_until_retried",
                "forced_short": "invalid_until_retried",
            },
            "corpus": {
                "name": self.task,
                "commit": self.corpus.commit,
                "dirty": False,
                "roots": ["src"],
                "language": "python",
                "suffixes": [".py"],
            },
            "source": {
                "git_commit": "a" * 40,
                "dirty": False,
                "files": {},
                "tree_sha256": sha256_json({}),
            },
            "environment": {
                "gpu_models": ["test-gpu"],
                "nvidia_driver": ["test-driver"],
                "vllm_version": "0.24.0",
                "vllm_environment_sha256": "e" * 64,
                "tensor_parallel_size": "1",
                "visible_gpu_count": "1",
                "server_count": "1",
            },
            "note": None,
            "extra": {
                "model_revision": "revision-a",
                "expected_response_model": "generation-revision",
                "server_transport": {"server_count": 1},
            },
        }
        if self.local:
            spec.update({
                "purpose": "exploratory",
                "claim_ready": False,
                "failure_policy": deepcopy(grade.SCREEN_FAILURE_POLICY),
                "preregistration": {
                    "schema_version": 1,
                    "status": "not_provided",
                    "reason": "exploratory",
                },
            })
        spec["environment_contract"] = environment_contract_record(spec["environment"])
        return {"manifest_schema": 1, "spec": spec}

    def _write_population(self) -> None:
        client = FakeClient(
            [verdict()] * 3,
            response_model=(
                grade.LOCAL_GRADER_MODEL if self.local else "judge-revision"
            ),
        )
        for budget in report.BUDGET_ORDER:
            status = "no_answer" if budget == "direct" else "ok"
            episode = native_episode(budget=budget, status=status)
            relative = f"{budget}/r0/q1.json"
            episode.update({
                "manifest_sha256": self.manifest_sha256,
                "question_sha256": sha256_json(self.questions[0]),
                "prompt_sha256": grade.sha256_bytes(
                    self.questions[0]["question"].encode("utf-8")),
                "note_sha256": None,
                "seed": self.manifest["spec"]["seed_policy"]["episode_seeds"][relative],
                "server_slot": self.manifest["spec"]["server_assignment"][
                    "episode_slots"
                ][relative],
                "environment_snapshot": {
                    "schema_version": 1,
                    "sha256": "9" * 64,
                    "bytes": 2,
                    "snapshot": "inputs/environments/environment-" + "9" * 64 + ".json",
                },
            })
            episode_path = self.run_task_root / relative
            episode_path.parent.mkdir(parents=True, exist_ok=True)
            episode_path.write_bytes(canonical_json_bytes(episode))
            episode_bytes = episode_path.read_bytes()
            source_episode = episode_path.relative_to(self.root).as_posix()
            spec_sha256 = grade.grade_spec_sha256(
                self.corpus,
                self.questions[0],
                self.judge_model,
                judge_base_url=self.judge_base_url,
            )
            if status == "ok":
                def write_intent(prompt_digest: str) -> str:
                    _, digest = grade.write_judge_attempt_intent(
                        self.grade_root,
                        source_episode=source_episode,
                        episode=episode,
                        episode_sha256=grade.sha256_bytes(episode_bytes),
                        grading_spec_sha256=spec_sha256,
                        grading_runtime_sha256=provenance.grading_runtime_sha256(
                            grade.grading_runtime_record()
                        ),
                        local_judge_runtime_sha256=(
                            provenance.local_judge_runtime_sha256(
                                grade.local_judge_runtime_record()
                            )
                            if self.local
                            else None
                        ),
                        judge_model=self.judge_model,
                        judge_base_url=(
                            self.judge_base_url or grade.CANONICAL_OPENAI_BASE_URL
                        ),
                        judge_prompt_sha256=prompt_digest,
                    )
                    return digest

                with patch(
                    "studybench.grade.sandbox.check", return_value=checker_result()
                ):
                    stored = asyncio.run(grade.grade_episode(
                        client,
                        self.judge_model,
                        self.corpus,
                        self.questions[0],
                        episode,
                        episode_sha256=grade.sha256_bytes(episode_bytes),
                        grading_spec_sha256=spec_sha256,
                        judge_base_url=self.judge_base_url,
                        judge_attempt_intent_writer=write_intent,
                    ))
            else:
                stored = asyncio.run(grade.grade_episode(
                    client,
                    self.judge_model,
                    self.corpus,
                    self.questions[0],
                    episode,
                    episode_sha256=grade.sha256_bytes(episode_bytes),
                    grading_spec_sha256=spec_sha256,
                    judge_base_url=self.judge_base_url,
                ))
            stored["source_episode"] = source_episode
            grade_path = self.grade_root / self.task / relative
            grade_path.parent.mkdir(parents=True, exist_ok=True)
            grade_path.write_bytes(canonical_json_bytes(stored))

    def patches(self):
        return (
            patch.object(report, "ROOT", self.root),
            patch.object(report, "CORPORA", {self.task: self.corpus}),
            patch.object(report, "load_questions", return_value=self.questions),
        )


def judge_population_bindings(fixture: EvaluationFixture) -> list[dict]:
    """Build the complete report-facing binding list for one test fixture."""

    bindings = []
    for relative in fixture.expected:
        episode_path = fixture.run_task_root / relative
        episode_bytes = episode_path.read_bytes()
        episode = json.loads(episode_bytes)
        bindings.append({
            "source_episode": episode_path.relative_to(fixture.root).as_posix(),
            "episode": episode,
            "episode_sha256": grade.sha256_bytes(episode_bytes),
            "grading_spec_sha256": grade.grade_spec_sha256(
                fixture.corpus,
                fixture.questions[0],
                fixture.judge_model,
                judge_base_url=fixture.judge_base_url,
            ),
            "judge_prompt_sha256": (
                grade.sha256_bytes(
                    grade.build_prompt(
                        fixture.corpus, fixture.questions[0], episode["answer"]
                    ).encode("utf-8")
                )
                if episode["status"] == "ok"
                else None
            ),
        })
    return bindings


class StrictReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grading_runtime = fake_grading_runtime()
        self.local_judge_runtime = fake_local_judge_runtime()
        self.runtime_patches = [
            patch.object(
                module, "grading_runtime_record", return_value=self.grading_runtime
            )
            for module in (grade, report)
        ] + [
            patch.object(
                module,
                "local_judge_runtime_record",
                return_value=self.local_judge_runtime,
            )
            for module in (grade, report)
        ]
        for runtime_patch in self.runtime_patches:
            runtime_patch.start()
        # Provenance owns the detailed environment schema; these tests isolate
        # grading/report behavior with a manifest declared claim-ready.
        self.environment_patch = patch(
            "studybench.grade.environment_is_claim_ready", return_value=True)
        self.environment_patch.start()
        self.environment_snapshot_patch = patch(
            "studybench.grade.validate_environment_snapshot",
            return_value={"claim_ready": True},
        )
        self.environment_snapshot_patch.start()
        self.screen_attempt_tree_patch = patch(
            "studybench.grade.provenance.validate_persisted_screen_attempt_tree",
            side_effect=lambda run_root, manifest, require_complete: [
                {
                    "expected_episode": relative,
                    "path": f"attempt-intents/{relative}",
                    "sha256": "8" * 64,
                    "bytes": 2,
                    "outcome": "final",
                }
                for relative in manifest["spec"]["expected_episodes"]
            ],
        )
        self.screen_attempt_tree_patch.start()
        # EvaluationFixture intentionally stores a synthetic successful
        # checker outcome.  Keep report-time independent rechecks synthetic as
        # well; production validation still invokes the real checker.
        self.checker_patch = patch(
            "studybench.grade.sandbox.check",
            side_effect=lambda *args, **kwargs: checker_result(),
        )
        self.checker_patch.start()
        self.preregistration_patch = patch(
            "studybench.grade.revalidate_run_preregistration",
            return_value={
                "grading_policy": {
                    "grader": "openai",
                    "judge_model": "gpt-5.4",
                    "evidence_mode": "excerpt_evidence",
                    "judge_effort": "",
                    "claim_scoring": "binary_0_1",
                    "question_scoring": "weighted_claim_sum",
                }
            },
        )
        self.preregistration_patch.start()
        self.current_source_patch = patch(
            "studybench.grade.validate_current_source", return_value={})
        self.current_source_patch.start()

    def tearDown(self) -> None:
        self.current_source_patch.stop()
        self.preregistration_patch.stop()
        self.checker_patch.stop()
        self.screen_attempt_tree_patch.stop()
        self.environment_snapshot_patch.stop()
        self.environment_patch.stop()
        for runtime_patch in reversed(self.runtime_patches):
            runtime_patch.stop()

    def _load(self, fixture: EvaluationFixture):
        root_patch, corpora_patch, questions_patch = fixture.patches()
        with root_patch, corpora_patch, questions_patch:
            return report._load_complete_evaluation(
                "fake",
                fixture.grade_root,
                fixture.run_root,
                rollouts=1,
                judge_model=fixture.judge_model,
            )

    def _load_local(self, fixture: EvaluationFixture):
        self.assertTrue(fixture.local)
        root_patch, corpora_patch, questions_patch = fixture.patches()
        with root_patch, corpora_patch, questions_patch:
            return report.load_local_diagnostic_evaluation(
                "fake",
                fixture.grade_root,
                fixture.run_root,
                rollouts=1,
                judge_base_url=fixture.judge_base_url,
            )

    def test_paper_comparison_requires_an_exact_explicit_configuration(self) -> None:
        expected_response_model = "Qwen/Qwen3.5-9B"
        audit = {
            "run_manifest": {"spec": {
                "harness": "dspy.ReAct",
                "model": report.PAPER_MODEL,
                "sampling": deepcopy(report.PAPER_SAMPLING),
                "rollouts": 3,
                "budgets": report.BUDGET_ORDER,
                "note": None,
                "extra": {
                    "model_revision": report.PAPER_MODEL_REVISION,
                    "expected_response_model": expected_response_model,
                },
            }},
            "generation_runtime": {"response_models": [expected_response_model]},
            "note_provenance": {"method": None},
        }
        self.assertEqual(report.paper_comparability_errors(
            audit, variant="base", judge_model="gpt-5.4", whole_files=True), [])
        audit["run_manifest"]["spec"]["sampling"]["temperature"] = 0.0
        errors = report.paper_comparability_errors(
            audit, variant="base", judge_model="gpt-5.4", whole_files=True)
        self.assertIn("sampling configuration differs", errors)

    def test_complete_population_and_immutable_report_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvaluationFixture(Path(directory))
            population, audit = self._load(fixture)
            aggregate = report.aggregate_population(population)
            self.assertEqual(aggregate["budgets"]["direct"]["no_answer"], 1)
            self.assertEqual(len(audit["population"]), 4)
            self.assertEqual(
                audit["generation_runtime"]["environment_snapshot_sha256s"],
                ["9" * 64],
            )
            self.assertEqual(
                len(audit["generation_runtime"][
                    "environment_snapshot_sha256_by_episode"
                ]),
                4,
            )
            generation_runtime = audit["generation_runtime"]
            self.assertEqual(
                generation_runtime["provider_identity_scope"],
                "final_manifest_episodes",
            )
            self.assertEqual(
                set(generation_runtime["provider_identity_by_episode"]),
                set(fixture.expected),
            )
            self.assertEqual(
                generation_runtime["provider_identity_by_episode"][
                    "direct/r0/q1.json"
                ]["response_models"],
                ["generation-revision"],
            )
            root_patch, corpora_patch, questions_patch = fixture.patches()
            with root_patch, corpora_patch, questions_patch:
                artifact_path = report.write_report_artifact(
                    task="fake",
                    run_id=fixture.run_id,
                    judge_dir=fixture.judge_dir,
                    aggregate_result=aggregate,
                    bootstrap_result=None,
                    bootstrap_replicates=0,
                    bootstrap_seed=17,
                    audit=audit,
                )
                first_bytes = artifact_path.read_bytes()
                repeated = report.write_report_artifact(
                    task="fake",
                    run_id=fixture.run_id,
                    judge_dir=fixture.judge_dir,
                    aggregate_result=aggregate,
                    bootstrap_result=None,
                    bootstrap_replicates=0,
                    bootstrap_seed=17,
                    audit=audit,
                )

                forged_aggregate = deepcopy(aggregate)
                forged_aggregate["expertise_lenient"] += 1
                with self.assertRaisesRegex(
                    report.ReportIntegrityError, "aggregate does not recompute"
                ):
                    report.write_report_artifact(
                        task="fake",
                        run_id=fixture.run_id,
                        judge_dir=fixture.judge_dir,
                        aggregate_result=forged_aggregate,
                        bootstrap_result=None,
                        bootstrap_replicates=0,
                        bootstrap_seed=17,
                        audit=audit,
                    )
            self.assertEqual(repeated, artifact_path)
            self.assertEqual(first_bytes, repeated.read_bytes())
            artifact = json.loads(first_bytes)
            self.assertTrue(artifact["claim_ready"])
            self.assertEqual(artifact["bootstrap"]["seed"], 17)
            self.assertIsNone(artifact["paper_comparison"])
            self.assertEqual(len(artifact["population"]), 4)
            self.assertEqual(artifact["failed_attempts"]["count"], 0)
            self.assertEqual(artifact["failed_judge_audits"]["count"], 0)
            grading = artifact["grading_manifest"]["config"]
            self.assertEqual(
                grading["judge_base_url"], grade.CANONICAL_OPENAI_BASE_URL)
            self.assertEqual(
                grading["judge_system_fingerprint_scope"],
                "accepted_final_attempts_only",
            )
            self.assertEqual(grading["judge_system_fingerprints"], ["judge-fingerprint"])
            self.assertEqual(
                grading["accepted_judge_system_fingerprint_by_episode"],
                {
                    f"{budget}/r0/q1.json": "judge-fingerprint"
                    for budget in ("k5", "k20", "k20f")
                },
            )
            self.assertEqual(grading["missing_judge_system_fingerprint_calls"], 0)
            self.assertEqual(grading["grading_runtime"], self.grading_runtime)
            self.assertEqual(
                grading["grading_runtime_sha256"],
                provenance.grading_runtime_sha256(self.grading_runtime),
            )
            self.assertNotIn(
                '"distributions":',
                canonical_json_bytes(grading["grading_runtime"]).decode(),
            )

    def test_report_loader_rejects_live_grading_runtime_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvaluationFixture(Path(directory))
            drifted = fake_grading_runtime(tree_sha256="2" * 64)
            with patch.object(
                report, "grading_runtime_record", return_value=drifted
            ), self.assertRaisesRegex(
                report.ReportIntegrityError, "integrity failure"
            ):
                self._load(fixture)

    def test_local_population_writes_deterministic_non_claim_ready_report(self) -> None:
        expected_report_fields = {
            "report_schema_version",
            "claim_ready",
            "task",
            "run_id",
            "budget_order",
            "run_manifest",
            "generation_runtime",
            "note_provenance",
            "failed_attempts",
            "failed_judge_audits",
            "grading_manifest",
            "population",
            "population_sha256",
            "aggregate",
            "bootstrap",
            "paper_comparison",
            "report_source",
        }
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvaluationFixture(Path(directory), local=True)
            population, audit = self._load_local(fixture)
            aggregate = report.aggregate_population(population)
            config = audit["grading_manifest"]["config"]
            self.assertEqual(config["judge_requested_model"], grade.LOCAL_GRADER_MODEL)
            self.assertEqual(config["judge_base_url"], fixture.judge_base_url)
            self.assertFalse(config["claim_ready"])
            self.assertEqual(config["grading_tier"], "diagnostic-local-proxy")
            self.assertTrue(config["local_proxy"])
            self.assertEqual(
                config["judge_endpoint_identity"],
                grade.LOCAL_GRADER_ENDPOINT_IDENTITY,
            )
            self.assertEqual(
                config["judge_transport_urls"], [fixture.judge_base_url]
            )
            self.assertEqual(
                config["judge_model_revision"], grade.LOCAL_GRADER_MODEL_REVISION)
            self.assertEqual(
                config["judge_request_options"], grade.LOCAL_GRADER_REQUEST_OPTIONS)
            self.assertEqual(
                config["local_judge_runtime"], self.local_judge_runtime
            )
            self.assertEqual(
                config["local_judge_runtime_sha256"],
                provenance.local_judge_runtime_sha256(
                    self.local_judge_runtime
                ),
            )
            checker = config["checker_interpretation"]
            self.assertEqual(checker["language"], "python")
            self.assertEqual(
                checker["sandbox_configuration_sha256"],
                grade.sandbox_configuration_sha256("python"),
            )
            expected_ready = (
                grade.sandbox_configuration_record("python").get("ready") is True
            )
            self.assertIs(checker["ready"], expected_ready)
            self.assertEqual(
                checker["score_interpretation"],
                "all-metrics"
                if expected_ready
                else "lenient-and-core-conjunctive-checker-unavailable",
            )

            root_patch, corpora_patch, questions_patch = fixture.patches()
            with root_patch, corpora_patch, questions_patch:
                first = report.write_report_artifact(
                    task="fake",
                    run_id=fixture.run_id,
                    judge_dir=fixture.judge_dir,
                    aggregate_result=aggregate,
                    bootstrap_result=None,
                    bootstrap_replicates=0,
                    bootstrap_seed=23,
                    audit=audit,
                )
                repeated = report.write_report_artifact(
                    task="fake",
                    run_id=fixture.run_id,
                    judge_dir=fixture.judge_dir,
                    aggregate_result=aggregate,
                    bootstrap_result=None,
                    bootstrap_replicates=0,
                    bootstrap_seed=23,
                    audit=audit,
                )
                with self.assertRaisesRegex(
                    report.ReportIntegrityError,
                    "paper comparison is prohibited",
                ):
                    report.write_report_artifact(
                        task="fake",
                        run_id=fixture.run_id,
                        judge_dir=fixture.judge_dir,
                        aggregate_result=aggregate,
                        bootstrap_result=None,
                        bootstrap_replicates=0,
                        bootstrap_seed=23,
                        audit=audit,
                        paper_comparison={},
                    )

            self.assertEqual(repeated, first)
            artifact = json.loads(first.read_bytes())
            self.assertEqual(set(artifact), expected_report_fields)
            self.assertFalse(artifact["claim_ready"])
            self.assertIsNone(artifact["paper_comparison"])
            self.assertEqual(artifact["grading_manifest"], audit["grading_manifest"])

    def test_local_report_nulls_unavailable_checker_metrics_after_recompute(self) -> None:
        checker_configuration = {
            "language": "python",
            "ready": False,
            "check_level": "syntax-only",
            "sandboxed": False,
            "error": "test checker unavailable",
        }
        with (
            patch.object(
                grade,
                "sandbox_configuration_record",
                return_value=checker_configuration,
            ),
            patch.object(
                report,
                "sandbox_configuration_record",
                return_value=checker_configuration,
            ),
            tempfile.TemporaryDirectory() as directory,
        ):
            fixture = EvaluationFixture(Path(directory), local=True)
            population, audit = self._load_local(fixture)
            raw_aggregate = report.aggregate_population(population)
            raw_bootstrap = report.bootstrap_population(population, 3, seed=23)
            root_patch, corpora_patch, questions_patch = fixture.patches()
            with root_patch, corpora_patch, questions_patch:
                artifact_path = report.write_report_artifact(
                    task="fake",
                    run_id=fixture.run_id,
                    judge_dir=fixture.judge_dir,
                    aggregate_result=raw_aggregate,
                    bootstrap_result=raw_bootstrap,
                    bootstrap_replicates=3,
                    bootstrap_seed=23,
                    audit=audit,
                )
                with self.assertRaisesRegex(
                    report.ReportIntegrityError,
                    "aggregate does not recompute",
                ):
                    report.write_report_artifact(
                        task="fake",
                        run_id=fixture.run_id,
                        judge_dir=fixture.judge_dir,
                        aggregate_result=report.reportable_aggregate(
                            raw_aggregate, checker_ready=False
                        ),
                        bootstrap_result=raw_bootstrap,
                        bootstrap_replicates=3,
                        bootstrap_seed=23,
                        audit=audit,
                    )

            artifact = json.loads(artifact_path.read_bytes())
            self.assertIsNone(artifact["aggregate"]["expertise_strict"])
            for budget in report.BUDGET_ORDER:
                values = artifact["aggregate"]["budgets"][budget]
                self.assertIsNotNone(values["len_cc"])
                self.assertIsNone(values["strict"])
                self.assertIsNone(values["compile_rate"])
                self.assertIsNotNone(values["lenient"])
            self.assertIsNotNone(artifact["bootstrap"]["results"]["wauc_cc"])
            self.assertIsNotNone(artifact["bootstrap"]["results"]["wauc"])

    def test_local_console_keeps_core_conjunctive_metrics_without_checker(self) -> None:
        population = {
            budget: [{
                "lenient": 50.0,
                "cores_ok": True,
                "strict": 0.0,
                "compile_check": {"compile_ok": False},
                "gen_tokens": 100,
                "episode_status": "ok",
            }]
            for budget in report.BUDGET_ORDER
        }
        audit = {
            "grading_manifest": {
                "config": {
                    "checker_interpretation": {
                        "ready": False,
                        "score_interpretation": (
                            "lenient-and-core-conjunctive-checker-unavailable"
                        ),
                    }
                }
            },
            "failed_attempts": {"count": 0},
            "failed_judge_audits": {"count": 0},
        }
        bootstrap = {
            budget: (50.0, 40.0, 60.0) for budget in report.BUDGET_ORDER
        } | {
            "wauc": (50.0, 40.0, 60.0),
            "wauc_cc": (49.0, 39.0, 59.0),
        }
        argv = [
            "report",
            "--tasks", "dspy",
            "--run-id", "fake-run",
            "--grader", "local",
            "--grade-id", "fake-grade",
            "--judge-base-url", "http://localhost:30000/v1",
            "--excerpt-evidence",
            "--ci", "3",
        ]
        output = StringIO()
        with (
            patch("sys.argv", argv),
            patch("sys.stdout", output),
            patch.object(
                report,
                "load_local_diagnostic_evaluation",
                return_value=(population, audit),
            ),
            patch.object(report, "bootstrap_population", return_value=bootstrap),
            patch.object(
                report,
                "write_report_artifact",
                return_value=Path("report.json"),
            ),
        ):
            report.main()

        rendered = output.getvalue()
        self.assertIn("len-cc", rendered)
        self.assertIn("WAUC len-cc", rendered)
        self.assertNotIn("compile", rendered)
        self.assertNotIn("strict WAUC", rendered)

    def test_recorded_local_revalidation_needs_no_live_judge_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvaluationFixture(Path(directory), local=True)
            expected_population, expected_audit = self._load_local(fixture)
            root_patch, corpora_patch, questions_patch = fixture.patches()
            with (
                root_patch,
                corpora_patch,
                questions_patch,
                patch.object(
                    report,
                    "local_judge_runtime_record",
                    side_effect=AssertionError(
                        "post-hoc revalidation called the live judge runtime"
                    ),
                ),
            ):
                population, audit = (
                    report.revalidate_recorded_local_diagnostic_evaluation(
                        "fake",
                        fixture.grade_root,
                        fixture.run_root,
                        rollouts=1,
                        judge_base_url=fixture.judge_base_url,
                        grading_runtime=self.grading_runtime,
                        local_judge_runtime=self.local_judge_runtime,
                    )
                )
            self.assertEqual(population, expected_population)
            self.assertEqual(audit, expected_audit)

    def test_recorded_local_revalidation_rejects_runtime_drift_or_tampering(
        self,
    ) -> None:
        mutations = {
            "current grading runtime drift": (
                self.grading_runtime,
                self.local_judge_runtime,
                fake_grading_runtime(tree_sha256="2" * 64),
            ),
            "stored grading runtime tamper": (
                fake_grading_runtime(tree_sha256="2" * 64),
                self.local_judge_runtime,
                self.grading_runtime,
            ),
            "stored local runtime cross-binding tamper": (
                self.grading_runtime,
                fake_local_judge_runtime(contract_sha256="2" * 64),
                self.grading_runtime,
            ),
        }
        for label, (stored_grading, stored_local, current_grading) in (
            mutations.items()
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                fixture = EvaluationFixture(Path(directory), local=True)
                root_patch, corpora_patch, questions_patch = fixture.patches()
                with (
                    root_patch,
                    corpora_patch,
                    questions_patch,
                    patch.object(
                        report,
                        "grading_runtime_record",
                        return_value=current_grading,
                    ),
                    patch.object(
                        report,
                        "local_judge_runtime_record",
                        side_effect=AssertionError(
                            "post-hoc revalidation called the live judge runtime"
                        ),
                    ),
                    self.assertRaises(report.ReportIntegrityError),
                ):
                    report.revalidate_recorded_local_diagnostic_evaluation(
                        "fake",
                        fixture.grade_root,
                        fixture.run_root,
                        rollouts=1,
                        judge_base_url=fixture.judge_base_url,
                        grading_runtime=stored_grading,
                        local_judge_runtime=stored_local,
                    )

    def test_strict_report_loader_rejects_local_exploratory_population(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvaluationFixture(Path(directory), local=True)
            root_patch, corpora_patch, questions_patch = fixture.patches()
            with (
                root_patch,
                corpora_patch,
                questions_patch,
                self.assertRaisesRegex(
                    report.ReportIntegrityError,
                    "explicit diagnostic-local path",
                ),
            ):
                report.load_complete_evaluation(
                    "fake",
                    fixture.grade_root,
                    fixture.run_root,
                    rollouts=1,
                    judge_model=fixture.judge_model,
                )

    def test_local_report_rejects_remote_endpoint_and_provenance_tampering(self) -> None:
        mutations = {
            "endpoint": lambda stored: stored.update(
                judge_base_url="https://redirect.invalid/v1"),
            "model revision": lambda stored: stored.update(
                judge_model_revision="0" * 40),
        }
        for label, mutation in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                fixture = EvaluationFixture(Path(directory), local=True)
                path = fixture.grade_root / "fake/k5/r0/q1.json"
                stored = json.loads(path.read_bytes())
                mutation(stored)
                path.write_bytes(canonical_json_bytes(stored))
                with self.assertRaises(report.ReportIntegrityError):
                    self._load_local(fixture)

    def test_local_report_accepts_grades_resumed_on_a_new_loopback_port(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvaluationFixture(Path(directory), local=True)
            path = fixture.grade_root / "fake/k5/r0/q1.json"
            stored = json.loads(path.read_bytes())
            stored["judge_base_url"] = "http://localhost:9123/v1"
            path.write_bytes(canonical_json_bytes(stored))
            _, audit = self._load_local(fixture)
            self.assertEqual(
                audit["grading_manifest"]["config"]["judge_transport_urls"],
                ["http://localhost:8123/v1", "http://localhost:9123/v1"],
            )

    def test_accepted_judge_content_tampering_is_fatal(self) -> None:
        def mutate_hash_only(stored: dict) -> None:
            stored["judge_accepted_content"] += "\n"

        def mutate_verdict_and_identity(stored: dict) -> None:
            payload = json.loads(stored["judge_accepted_content"])
            payload["claims"][0]["rationale"] = "tampered rationale"
            content = json.dumps(payload, sort_keys=True)
            stored["judge_accepted_content"] = content
            accepted = stored["judge_attempts"][-1]
            accepted["content_sha256"] = grade.sha256_bytes(content.encode("utf-8"))
            accepted["content_bytes"] = len(content.encode("utf-8"))

        for label, mutation in (
                ("content hash", mutate_hash_only),
                ("stored verdict", mutate_verdict_and_identity)):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                fixture = EvaluationFixture(Path(directory))
                path = fixture.grade_root / "fake/k5/r0/q1.json"
                stored = json.loads(path.read_bytes())
                mutation(stored)
                path.write_bytes(canonical_json_bytes(stored))
                with self.assertRaises(report.ReportIntegrityError):
                    self._load(fixture)

    def test_judge_endpoint_tampering_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvaluationFixture(Path(directory))
            path = fixture.grade_root / "fake/k5/r0/q1.json"
            stored = json.loads(path.read_bytes())
            stored["judge_base_url"] = "https://redirect.invalid/v1"
            path.write_bytes(canonical_json_bytes(stored))
            with self.assertRaises(report.ReportIntegrityError):
                self._load(fixture)

    def test_current_source_drift_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvaluationFixture(Path(directory))
            with patch(
                "studybench.grade.validate_current_source",
                side_effect=ValueError("source drift"),
            ), self.assertRaisesRegex(
                report.ReportIntegrityError, "source drift"
            ):
                self._load(fixture)

    def test_no_answer_grade_requires_a_null_accepted_content_marker(self) -> None:
        for label, mutation in (
                ("non-null", lambda stored: stored.update(
                    judge_accepted_content="{}")),
                ("missing", lambda stored: stored.pop("judge_accepted_content"))):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                fixture = EvaluationFixture(Path(directory))
                path = fixture.grade_root / "fake/direct/r0/q1.json"
                stored = json.loads(path.read_bytes())
                mutation(stored)
                path.write_bytes(canonical_json_bytes(stored))
                with self.assertRaises(report.ReportIntegrityError):
                    self._load(fixture)

    def test_report_discloses_mutable_and_missing_judge_fingerprints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvaluationFixture(Path(directory))
            missing_path = fixture.grade_root / "fake/k5/r0/q1.json"
            missing = json.loads(missing_path.read_bytes())
            missing_attempt = missing["judge_attempts"][0]
            missing_attempt["system_fingerprint"] = None
            missing_attempt["system_fingerprint_status"] = "unavailable"
            missing_attempt["system_fingerprint_observation"] = (
                grade._audit_observation(None))
            missing_path.write_bytes(canonical_json_bytes(missing))

            changed_path = fixture.grade_root / "fake/k20/r0/q1.json"
            changed = json.loads(changed_path.read_bytes())
            changed["judge_attempts"][0]["system_fingerprint"] = "other-fingerprint"
            changed_path.write_bytes(canonical_json_bytes(changed))

            _, audit = self._load(fixture)
            grading = audit["grading_manifest"]["config"]
            self.assertEqual(
                grading["judge_system_fingerprints"],
                ["judge-fingerprint", "other-fingerprint"],
            )
            self.assertEqual(grading["missing_judge_system_fingerprint_calls"], 1)
            accepted = grading["accepted_judge_system_fingerprint_by_episode"]
            self.assertIsNone(accepted["k5/r0/q1.json"])
            self.assertEqual(accepted["k20/r0/q1.json"], "other-fingerprint")
            self.assertEqual(accepted["k20f/r0/q1.json"], "judge-fingerprint")

    def test_rejected_judge_fingerprint_is_not_an_accepted_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvaluationFixture(Path(directory))
            grade_path = fixture.grade_root / "fake/k5/r0/q1.json"
            stored = json.loads(grade_path.read_bytes())
            accepted = deepcopy(stored["judge_attempts"][0])
            rejected = deepcopy(accepted)
            invalid_content = "{}"
            rejected.update({
                "attempt": 1,
                "accepted": False,
                "system_fingerprint": "rejected-fingerprint",
                "content_sha256": grade.sha256_bytes(invalid_content.encode("utf-8")),
                "content_bytes": len(invalid_content.encode("utf-8")),
                "invalid_content": invalid_content,
                "validation_error": {
                    "type": "GradeIntegrityError",
                    "message": "invalid verdict",
                },
            })
            accepted.update({
                "attempt": 2,
                "response_id": "response-2",
                "request_id": "request-2",
            })
            stored["judge_attempts"] = [rejected, accepted]
            stored["judge_attempt_count"] = 2
            stored["judge_accepted_attempt"] = 2
            stored["judge_usage_total"] = {
                field: rejected["usage"][field] + accepted["usage"][field]
                for field in ("prompt_tokens", "completion_tokens", "total_tokens")
            }
            grade_path.write_bytes(canonical_json_bytes(stored))

            _, audit = self._load(fixture)
            grading = audit["grading_manifest"]["config"]
            self.assertEqual(
                grading["judge_system_fingerprints"], ["judge-fingerprint"]
            )
            self.assertEqual(
                grading["accepted_judge_system_fingerprint_by_episode"][
                    "k5/r0/q1.json"
                ],
                "judge-fingerprint",
            )

    def test_failed_attempts_are_disclosed_but_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvaluationFixture(Path(directory))
            final_path = fixture.run_task_root / "k5/r0/q1.json"
            failed = json.loads(final_path.read_bytes())
            failed.update({
                "status": "error",
                "error": "provider failure",
                "failure_attempt": 1,
                "expected_episode": "k5/r0/q1.json",
            })
            failed_path = (
                fixture.run_task_root
                / "failed-attempts/k5/r0/q1/attempt-1.json"
            )
            failed_path.parent.mkdir(parents=True, exist_ok=True)
            failed_path.write_bytes(canonical_json_bytes(failed))
            population, audit = self._load(fixture)
            self.assertEqual(sum(len(values) for values in population.values()), 4)
            self.assertEqual(audit["failed_attempts"]["count"], 1)
            self.assertEqual(
                audit["failed_attempts"]["artifacts"][0]["status"], "error")

    def test_report_rejects_failed_judge_audit_with_complete_grade(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvaluationFixture(Path(directory))
            episode_path = fixture.run_task_root / "k5/r0/q1.json"
            episode = json.loads(episode_path.read_bytes())
            spec_sha256 = grade.grade_spec_sha256(
                fixture.corpus, fixture.questions[0], fixture.judge_model)
            client = FakeClient([verdict(duplicate=True), verdict(score=100)])
            with patch(
                "studybench.grade.sandbox.check", return_value=checker_result()
            ):
                with self.assertRaises(grade.JudgeAttemptsFailed) as caught:
                    asyncio.run(grade.grade_episode(
                        client,
                        fixture.judge_model,
                        fixture.corpus,
                        fixture.questions[0],
                        episode,
                        episode_sha256=grade.sha256_bytes(episode_path.read_bytes()),
                        grading_spec_sha256=spec_sha256,
                    ))
            with patch.object(grade, "ROOT", fixture.root):
                grade.write_failed_judge_audit(
                    fixture.grade_root,
                    episode_path.relative_to(fixture.root).as_posix(),
                    caught.exception.audit,
                )
            with self.assertRaisesRegex(
                report.ReportIntegrityError, "judge-attempt intent ledger"
            ):
                self._load(fixture)

    def test_report_rejects_unavailable_usage_failure_with_complete_grade(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvaluationFixture(Path(directory))
            episode_path = fixture.run_task_root / "k5/r0/q1.json"
            episode = json.loads(episode_path.read_bytes())
            spec_sha256 = grade.grade_spec_sha256(
                fixture.corpus, fixture.questions[0], fixture.judge_model)
            client = FixedResponseClient(fixed_response(usage=None))
            with patch(
                "studybench.grade.sandbox.check", return_value=checker_result()
            ):
                with self.assertRaises(grade.JudgeAttemptsFailed) as caught:
                    asyncio.run(grade.grade_episode(
                        client,
                        fixture.judge_model,
                        fixture.corpus,
                        fixture.questions[0],
                        episode,
                        episode_sha256=grade.sha256_bytes(episode_path.read_bytes()),
                        grading_spec_sha256=spec_sha256,
                    ))
            with patch.object(grade, "ROOT", fixture.root):
                grade.write_failed_judge_audit(
                    fixture.grade_root,
                    episode_path.relative_to(fixture.root).as_posix(),
                    caught.exception.audit,
                )
            with self.assertRaisesRegex(
                report.ReportIntegrityError, "judge-attempt intent ledger"
            ):
                self._load(fixture)

    def test_missing_grade_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvaluationFixture(Path(directory))
            (fixture.grade_root / "fake/k5/r0/q1.json").unlink()
            with self.assertRaises(report.ReportIntegrityError):
                self._load(fixture)

    def test_report_rejects_deleted_judge_attempt_intent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvaluationFixture(Path(directory))
            intent = (
                fixture.grade_root
                / "judge-attempt-intents/fake/k5/r0/q1.json"
            )
            intent.unlink()
            with self.assertRaisesRegex(
                report.ReportIntegrityError, "judge-attempt intent ledger"
            ):
                self._load(fixture)

    def test_episode_drift_makes_grade_stale(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvaluationFixture(Path(directory))
            path = fixture.run_task_root / "k5/r0/q1.json"
            episode = json.loads(path.read_bytes())
            episode["answer"] += "\nchanged"
            path.write_bytes(canonical_json_bytes(episode))
            with self.assertRaises(report.ReportIntegrityError):
                self._load(fixture)

    def test_episode_launch_environment_is_revalidated_downstream(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvaluationFixture(Path(directory))
            with patch(
                "studybench.grade.validate_environment_snapshot",
                side_effect=ValueError("substantive drift"),
            ), self.assertRaisesRegex(
                report.ReportIntegrityError, "launch environment"
            ):
                self._load(fixture)

    def test_mixed_provider_model_revisions_are_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvaluationFixture(Path(directory))
            path = fixture.grade_root / "fake/k5/r0/q1.json"
            stored = json.loads(path.read_bytes())
            stored["judge_response_model"] = "different-revision"
            stored["judge_attempts"][-1]["response_model"] = "different-revision"
            path.write_bytes(canonical_json_bytes(stored))
            with self.assertRaises(report.ReportIntegrityError):
                self._load(fixture)

    def test_mixed_generation_model_revisions_are_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvaluationFixture(Path(directory))
            episode_path = fixture.run_task_root / "k5/r0/q1.json"
            episode = json.loads(episode_path.read_bytes())
            episode["turns"][0]["response_model"] = "different-generation-revision"
            episode_path.write_bytes(canonical_json_bytes(episode))
            stored_path = fixture.grade_root / "fake/k5/r0/q1.json"
            stored = json.loads(stored_path.read_bytes())
            stored["episode_sha256"] = grade.sha256_bytes(episode_path.read_bytes())
            stored_path.write_bytes(canonical_json_bytes(stored))
            with self.assertRaises(report.ReportIntegrityError):
                self._load(fixture)

    def test_preflight_rejects_invalid_episode_before_grading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvaluationFixture(Path(directory))
            path = fixture.run_task_root / "k20/r0/q1.json"
            episode = json.loads(path.read_bytes())
            episode["gen_tokens"] += 1
            path.write_bytes(canonical_json_bytes(episode))
            context = grade.load_claim_manifest(
                fixture.run_task_root, fixture.corpus, fixture.questions)
            with patch.object(grade, "ROOT", fixture.root):
                with self.assertRaises(grade.GradeIntegrityError):
                    grade.preflight_grade_population(
                        runs_root=fixture.run_root,
                        out_root=fixture.grade_root,
                        corpus=fixture.corpus,
                        questions=fixture.questions,
                        manifest_context=context,
                        judge_model=fixture.judge_model,
                        whole_files=False,
                        effort="",
                    )

    def test_preflight_globally_blocks_an_orphan_judge_intent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvaluationFixture(Path(directory))
            (fixture.grade_root / "fake/k5/r0/q1.json").unlink()
            context = grade.load_claim_manifest(
                fixture.run_task_root, fixture.corpus, fixture.questions
            )
            with patch.object(grade, "ROOT", fixture.root), self.assertRaisesRegex(
                grade.GradeIntegrityError, "orphan judge-attempt intent"
            ):
                grade.preflight_grade_population(
                    runs_root=fixture.run_root,
                    out_root=fixture.grade_root,
                    corpus=fixture.corpus,
                    questions=fixture.questions,
                    manifest_context=context,
                    judge_model=fixture.judge_model,
                    whole_files=False,
                    effort="",
                )

    def test_preflight_globally_blocks_failed_audit_and_grade_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvaluationFixture(Path(directory))
            episode_path = fixture.run_task_root / "k5/r0/q1.json"
            episode_bytes = episode_path.read_bytes()
            episode = json.loads(episode_bytes)
            source_episode = episode_path.relative_to(fixture.root).as_posix()
            spec_digest = grade.grade_spec_sha256(
                fixture.corpus, fixture.questions[0], fixture.judge_model
            )
            intent_path = grade._judge_attempt_intent_path(
                fixture.grade_root, episode
            )
            intent_digest = grade.sha256_bytes(intent_path.read_bytes())
            audit = grade._failed_judge_audit(
                ep=episode,
                episode_sha256=grade.sha256_bytes(episode_bytes),
                grading_spec_sha256=spec_digest,
                grading_runtime_sha256=provenance.grading_runtime_sha256(
                    self.grading_runtime
                ),
                local_judge_runtime_sha256=None,
                judge_model=fixture.judge_model,
                judge_prompt_sha256=grade.sha256_bytes(
                    grade.build_prompt(
                        fixture.corpus, fixture.questions[0], episode["answer"]
                    ).encode("utf-8")
                ),
                attempts=[],
                failure=RuntimeError("provider unavailable"),
                request_attempt_count=1,
                judge_attempt_intent_sha256=intent_digest,
            )
            grade.write_failed_judge_audit(
                fixture.grade_root, source_episode, audit
            )
            context = grade.load_claim_manifest(
                fixture.run_task_root, fixture.corpus, fixture.questions
            )
            with patch.object(grade, "ROOT", fixture.root), self.assertRaisesRegex(
                grade.GradeIntegrityError,
                "whole grading invocation is blocked|coexist",
            ):
                grade.preflight_grade_population(
                    runs_root=fixture.run_root,
                    out_root=fixture.grade_root,
                    corpus=fixture.corpus,
                    questions=fixture.questions,
                    manifest_context=context,
                    judge_model=fixture.judge_model,
                    whole_files=False,
                    effort="",
                )

    def test_preflight_rejects_symlinked_grade_destination_before_contact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvaluationFixture(Path(directory))
            alternate = fixture.root / "grades/run-a/alternate"
            target = fixture.root / "grade-target"
            target.mkdir()
            (alternate / "fake").mkdir(parents=True)
            (alternate / "fake/k5").symlink_to(
                target, target_is_directory=True
            )
            context = grade.load_claim_manifest(
                fixture.run_task_root, fixture.corpus, fixture.questions
            )
            with patch.object(grade, "ROOT", fixture.root), self.assertRaisesRegex(
                grade.GradeIntegrityError, "grade destination tree contains a symlink"
            ):
                grade.preflight_grade_population(
                    runs_root=fixture.run_root,
                    out_root=alternate,
                    corpus=fixture.corpus,
                    questions=fixture.questions,
                    manifest_context=context,
                    judge_model=fixture.judge_model,
                    whole_files=False,
                    effort="",
                )

    def test_preflight_rejects_broken_symlinked_grade_root_before_contact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvaluationFixture(Path(directory))
            alternate = fixture.root / "grades/run-a/alternate"
            alternate.mkdir(parents=True)
            (alternate / "fake").symlink_to(
                fixture.root / "missing-grade-target", target_is_directory=True
            )
            context = grade.load_claim_manifest(
                fixture.run_task_root, fixture.corpus, fixture.questions
            )
            with patch.object(grade, "ROOT", fixture.root), self.assertRaisesRegex(
                grade.GradeIntegrityError, "grade destination root is not a safe directory"
            ):
                grade.preflight_grade_population(
                    runs_root=fixture.run_root,
                    out_root=alternate,
                    corpus=fixture.corpus,
                    questions=fixture.questions,
                    manifest_context=context,
                    judge_model=fixture.judge_model,
                    whole_files=False,
                    effort="",
                )

    def test_judge_attempt_inventory_is_complete_sorted_and_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvaluationFixture(Path(directory))
            records = grade.validate_judge_attempt_inventory(
                fixture.grade_root,
                judge_population_bindings(fixture),
                judge_model=fixture.judge_model,
                judge_base_url=grade.CANONICAL_OPENAI_BASE_URL,
                grading_runtime_sha256=provenance.grading_runtime_sha256(
                    self.grading_runtime
                ),
                local_judge_runtime_sha256=None,
            )
            self.assertEqual(len(records), 3)
            self.assertEqual(
                [record["expected_episode"] for record in records],
                sorted(record["expected_episode"] for record in records),
            )
            self.assertEqual({record["outcome"] for record in records}, {"grade"})
            self.assertTrue(all(record["path"].startswith(
                "judge-attempt-intents/fake/"
            ) for record in records))

            direct_intent = (
                fixture.grade_root
                / "judge-attempt-intents/fake/direct/r0/q1.json"
            )
            direct_intent.parent.mkdir(parents=True, exist_ok=True)
            direct_intent.symlink_to(
                fixture.grade_root
                / "judge-attempt-intents/fake/k5/r0/q1.json"
            )
            with self.assertRaisesRegex(
                grade.GradeIntegrityError,
                "judge-attempt intent tree contains (?:an unknown directory|a symlink)",
            ):
                grade.validate_judge_attempt_inventory(
                    fixture.grade_root,
                    judge_population_bindings(fixture),
                    judge_model=fixture.judge_model,
                    judge_base_url=grade.CANONICAL_OPENAI_BASE_URL,
                    grading_runtime_sha256=provenance.grading_runtime_sha256(
                        self.grading_runtime
                    ),
                    local_judge_runtime_sha256=None,
                )

    def test_judge_attempt_inventory_accepts_one_bound_failed_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvaluationFixture(Path(directory))
            relative = "k5/r0/q1.json"
            episode_path = fixture.run_task_root / relative
            episode_bytes = episode_path.read_bytes()
            episode = json.loads(episode_bytes)
            grade_path = fixture.grade_root / "fake" / relative
            grade_path.unlink()
            intent_path = grade._judge_attempt_intent_path(
                fixture.grade_root, episode
            )
            intent_digest = grade.sha256_bytes(intent_path.read_bytes())
            source_episode = episode_path.relative_to(fixture.root).as_posix()
            spec_digest = grade.grade_spec_sha256(
                fixture.corpus, fixture.questions[0], fixture.judge_model
            )
            prompt_digest = grade.sha256_bytes(
                grade.build_prompt(
                    fixture.corpus, fixture.questions[0], episode["answer"]
                ).encode("utf-8")
            )
            audit = grade._failed_judge_audit(
                ep=episode,
                episode_sha256=grade.sha256_bytes(episode_bytes),
                grading_spec_sha256=spec_digest,
                grading_runtime_sha256=provenance.grading_runtime_sha256(
                    self.grading_runtime
                ),
                local_judge_runtime_sha256=None,
                judge_model=fixture.judge_model,
                judge_prompt_sha256=prompt_digest,
                attempts=[],
                failure=RuntimeError("provider unavailable"),
                request_attempt_count=1,
                judge_attempt_intent_sha256=intent_digest,
            )
            grade.write_failed_judge_audit(
                fixture.grade_root, source_episode, audit
            )

            records = grade.validate_judge_attempt_inventory(
                fixture.grade_root,
                judge_population_bindings(fixture),
                judge_model=fixture.judge_model,
                judge_base_url=grade.CANONICAL_OPENAI_BASE_URL,
                grading_runtime_sha256=provenance.grading_runtime_sha256(
                    self.grading_runtime
                ),
                local_judge_runtime_sha256=None,
            )
            failed = [record for record in records if record["outcome"] == "failed"]
            self.assertEqual(len(failed), 1)
            self.assertEqual(failed[0]["expected_episode"], source_episode)
            self.assertTrue(failed[0]["terminal_path"].startswith(
                "failed-judge-audits/fake/k5/r0/q1-"
            ))

    def test_manifest_and_population_inputs_reject_symlink_components(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvaluationFixture(Path(directory))
            linked_run = fixture.root / "linked-run"
            linked_run.symlink_to(fixture.run_root, target_is_directory=True)
            with self.assertRaises(grade.GradeIntegrityError):
                grade.load_claim_manifest(
                    linked_run / "fake", fixture.corpus, fixture.questions)

            context = grade.load_claim_manifest(
                fixture.run_task_root, fixture.corpus, fixture.questions)
            episode_path = fixture.run_task_root / "k20/r0/q1.json"
            real_copy = fixture.root / "episode-copy.json"
            real_copy.write_bytes(episode_path.read_bytes())
            episode_path.unlink()
            episode_path.symlink_to(real_copy)
            with patch.object(grade, "ROOT", fixture.root):
                with self.assertRaises(grade.GradeIntegrityError):
                    grade.preflight_grade_population(
                        runs_root=fixture.run_root,
                        out_root=fixture.grade_root,
                        corpus=fixture.corpus,
                        questions=fixture.questions,
                        manifest_context=context,
                        judge_model=fixture.judge_model,
                        whole_files=False,
                        effort="",
                    )

    def test_stale_existing_grade_is_preserved_and_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvaluationFixture(Path(directory))
            grade_path = fixture.grade_root / "fake/k5/r0/q1.json"
            stale = json.loads(grade_path.read_bytes())
            stale["lenient"] = 0
            grade_path.write_bytes(canonical_json_bytes(stale))
            before = grade_path.read_bytes()
            context = grade.load_claim_manifest(
                fixture.run_task_root, fixture.corpus, fixture.questions)
            with patch.object(grade, "ROOT", fixture.root):
                with self.assertRaisesRegex(
                    grade.GradeIntegrityError, "preserved.*new --grade-id"
                ):
                    grade.preflight_grade_population(
                        runs_root=fixture.run_root,
                        out_root=fixture.grade_root,
                        corpus=fixture.corpus,
                        questions=fixture.questions,
                        manifest_context=context,
                        judge_model=fixture.judge_model,
                        whole_files=False,
                        effort="",
                    )
            self.assertEqual(grade_path.read_bytes(), before)

    def test_checker_configuration_binding_cannot_be_rewritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvaluationFixture(Path(directory))
            grade_path = fixture.grade_root / "fake/k5/r0/q1.json"
            stored = json.loads(grade_path.read_bytes())
            stored["compile_check"]["configuration_sha256"] = "0" * 64
            # Even a self-consistent rewrite of the derived strict score cannot
            # detach the result from the frozen checker contract.
            stored["strict"] = grade.score_from_claims(
                fixture.questions[0],
                {"core": 1, "detail": 0},
                stored["compile_check"]["compile_ok"],
            )["strict"]
            grade_path.write_bytes(canonical_json_bytes(stored))
            with self.assertRaises(report.ReportIntegrityError):
                self._load(fixture)

    def test_checker_outcome_is_independently_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvaluationFixture(Path(directory))
            grade_path = fixture.grade_root / "fake/k5/r0/q1.json"
            stored = json.loads(grade_path.read_bytes())
            stored["compile_check"]["compile_ok"] = False
            scores = grade.score_from_claims(
                fixture.questions[0], {"core": 1, "detail": 0}, False)
            for key, value in scores.items():
                stored[key] = value
            stored["judge_question_score"] = scores["lenient"]
            grade_path.write_bytes(canonical_json_bytes(stored))
            with self.assertRaisesRegex(
                report.ReportIntegrityError, "independent deterministic rerun"
            ):
                self._load(fixture)

    def test_manifest_schema_environment_and_seed_policy_are_independently_checked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvaluationFixture(Path(directory))
            manifest_path = fixture.run_task_root / "manifest.json"
            for field_path in (("manifest_schema",), ("spec", "schema_version")):
                with self.subTest(field_path=field_path):
                    manifest = deepcopy(fixture.manifest)
                    target = manifest
                    for field in field_path[:-1]:
                        target = target[field]
                    target[field_path[-1]] = True
                    manifest_path.write_bytes(canonical_json_bytes(manifest))
                    with self.assertRaisesRegex(
                        grade.GradeIntegrityError, "unknown run .* schema"
                    ):
                        grade.load_claim_manifest(
                            fixture.run_task_root, fixture.corpus, fixture.questions)

            manifest = deepcopy(fixture.manifest)
            manifest["spec"]["environment"]["vllm_environment_sha256"] = None
            manifest_path.write_bytes(canonical_json_bytes(manifest))
            with patch(
                "studybench.grade.environment_is_claim_ready",
                side_effect=lambda value: (
                    value.get("vllm_environment_sha256") is not None),
            ):
                with self.assertRaises(grade.GradeIntegrityError):
                    grade.load_claim_manifest(
                        fixture.run_task_root, fixture.corpus, fixture.questions)

            manifest = deepcopy(fixture.manifest)
            manifest["spec"]["purpose"] = "exploratory"
            manifest_path.write_bytes(canonical_json_bytes(manifest))
            with self.assertRaisesRegex(grade.GradeIntegrityError, "confirmatory"):
                grade.load_claim_manifest(
                    fixture.run_task_root, fixture.corpus, fixture.questions)

            manifest_path.write_bytes(canonical_json_bytes(fixture.manifest))
            with patch(
                "studybench.grade.revalidate_run_preregistration",
                side_effect=grade.PreregistrationError("changed snapshot"),
            ), self.assertRaisesRegex(grade.GradeIntegrityError, "preregistration"):
                grade.load_claim_manifest(
                    fixture.run_task_root, fixture.corpus, fixture.questions)

    def test_local_smoke_manifest_is_narrow_explicit_and_can_be_dirty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvaluationFixture(Path(directory), local=True)
            manifest = deepcopy(fixture.manifest)
            spec = manifest["spec"]
            spec["purpose"] = "smoke"
            spec["claim_ready"] = False
            spec["preregistration"] = {
                "schema_version": 1,
                "status": "not_provided",
                "reason": "smoke",
            }
            spec["source"]["dirty"] = True
            path = fixture.run_task_root / "manifest.json"
            path.write_bytes(canonical_json_bytes(manifest))

            with self.assertRaisesRegex(
                grade.GradeIntegrityError, "exploratory"
            ):
                grade.load_claim_manifest(
                    fixture.run_task_root,
                    fixture.corpus,
                    fixture.questions,
                    require_claim_ready=False,
                )
            with patch(
                "studybench.grade.environment_is_claim_ready", return_value=False
            ):
                context = grade.load_claim_manifest(
                    fixture.run_task_root,
                    fixture.corpus,
                    fixture.questions,
                    require_claim_ready=False,
                    allow_smoke=True,
                )
            self.assertEqual(context["spec"]["purpose"], "smoke")
            self.assertFalse(context["spec"]["claim_ready"])

    def test_exploratory_manifest_accepts_only_bundled_automated_ready_note(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvaluationFixture(Path(directory), task="dspy")
            fixture.manifest["spec"]["sampling"] = deepcopy(REACT_SAMPLING)
            spec_contract = fixture.manifest["spec"]
            note_bytes = b"automatically validated exploratory note\n"
            dependency_bytes = canonical_json_bytes({"record": "complete"})
            task_manifest = {
                "schema_version": 4,
                "manifest_type": SEMANTIC_SELFQUIZ_TASK_MANIFEST_TYPE,
                "method": SEMANTIC_SELFQUIZ_METHOD,
                "study_id": "study-a",
                "task": "dspy",
                "master_seed": 11,
                "model": spec_contract["model"],
                "model_revision": spec_contract["model_revision"],
                "sampling": deepcopy(spec_contract["sampling"]),
                "corpus_commit": fixture.corpus.commit,
                "corpus": deepcopy(spec_contract["corpus"]),
                "source": deepcopy(spec_contract["source"]),
                "environment": deepcopy(spec_contract["environment"]),
                "environment_contract": deepcopy(
                    spec_contract["environment_contract"]
                ),
                "server_transport": {
                    "scope": "loopback",
                    "protocol": "openai-compatible-http",
                    "server_count": 1,
                    "assignment": (
                        "stable_seed(master_seed, stochastic_namespace, server) modulo server_count"
                    ),
                },
                "provenance_readiness": {
                    "corpus_pinned_clean": True,
                    "source_pinned_clean": True,
                    "environment_complete": True,
                    "model_revision_pinned": True,
                    "server_count_matches_environment": True,
                },
                "automated_provenance_ready": True,
                "human_audit_protocol": None,
                "config": {
                    "chapter_syllabus": [
                        "dspy/teleprompt", "dspy/adapters", "dspy/clients",
                        "dspy/predict", "dspy/primitives", "dspy/utils",
                        "dspy/dsp", "dspy/signatures", "dspy/datasets",
                        "dspy/retrievers", "dspy/streaming", "dspy/evaluate",
                        "dspy/propose", "dspy", "dspy/experimental",
                    ],
                    "chapters_per_round": 4,
                    "final_round": 4,
                    "questions_per_chapter": 5,
                    "attempt_access": "react-corpus",
                    "smoke": False,
                    "quiz_max_iters": 15,
                    "attempt_protocol": openbook_attempt_protocol(),
                    "derive_max_iters": 15,
                    "train_ensemble": 2,
                    "dev_ensemble": 2,
                    "retest_fraction": 0.2,
                    "freshness_near_jaccard": 0.8,
                    "max_freshness_near_rate": 0.1,
                    "concurrency": 8,
                    "provider_retries": 0,
                },
            }
            task_manifest_bytes = canonical_json_bytes(task_manifest)
            note_sha256 = grade.sha256_bytes(note_bytes)
            dependency_sha256 = grade.sha256_bytes(dependency_bytes)
            inventory = {
                "manifest.json": {
                    "sha256": grade.sha256_bytes(task_manifest_bytes),
                    "bytes": len(task_manifest_bytes),
                },
                "rounds/round-1/record.json": {
                    "sha256": dependency_sha256,
                    "bytes": len(dependency_bytes),
                }
            }
            construction = {
                "schema_version": 2,
                "manifest_type": SEMANTIC_SELFQUIZ_NOTE_MANIFEST_TYPE,
                "method": SEMANTIC_SELFQUIZ_METHOD,
                "protocol_summary": derive_protocol_summary(task_manifest_bytes),
                "study_id": "study-a",
                "task": "dspy",
                "round": 4,
                "corpus_commit": fixture.corpus.commit,
                "claim_ready": False,
                "publication_claim_ready": False,
                "confirmatory_claim_ready": False,
                "automated_claim_ready": True,
                "automated_readiness": {"construction_complete": True},
                "note_sha256": note_sha256,
                "note_path": "by-sha256/note.md",
                "construction_artifacts": inventory,
                "construction_artifacts_sha256": sha256_json(inventory),
            }
            construction_bytes = canonical_json_bytes(construction)
            construction_sha256 = grade.sha256_bytes(construction_bytes)
            bundle_root = Path("inputs") / f"note-provenance-{construction_sha256}"
            construction_snapshot = Path("inputs/construction.json")
            note_snapshot = Path("inputs/note.md")
            bundled_manifest = bundle_root / "note-r4.manifest.json"
            bundled_note = bundle_root / construction["note_path"]
            bundled_dependency = (
                bundle_root / "construction/rounds/round-1/record.json")
            bundled_task_manifest = bundle_root / "construction/manifest.json"
            for relative, data in (
                (construction_snapshot, construction_bytes),
                (note_snapshot, note_bytes),
                (bundled_manifest, construction_bytes),
                (bundled_note, note_bytes),
                (bundled_task_manifest, task_manifest_bytes),
                (bundled_dependency, dependency_bytes),
            ):
                path = fixture.run_task_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)

            manifest = deepcopy(fixture.manifest)
            spec = manifest["spec"]
            spec["purpose"] = "exploratory"
            spec["claim_ready"] = False
            spec["failure_policy"] = deepcopy(grade.SCREEN_FAILURE_POLICY)
            spec["preregistration"] = {
                "schema_version": 1,
                "status": "not_provided",
                "reason": "exploratory",
            }
            template = "Study note:\n{note}\n\n"
            spec["prompt_policy"] = {
                "note_prefix_template": template,
                "presented_prompt_sha256": {
                    "q1": grade.sha256_bytes(
                        (template.format(note=note_bytes.decode())
                         + fixture.questions[0]["question"]).encode()),
                },
            }
            spec["note"] = {
                "sha256": note_sha256,
                "bytes": len(note_bytes),
                "snapshot": str(note_snapshot),
                "source_name": "note.md",
                "construction_manifest": {
                    "sha256": construction_sha256,
                    "snapshot": str(construction_snapshot),
                },
                "provenance_bundle": {
                    "root": str(bundle_root),
                    "manifest_snapshot": str(bundled_manifest),
                    "note_snapshot": str(bundled_note),
                    "construction_artifacts": {
                        "root": str(bundle_root / "construction"),
                        "inventory_sha256": sha256_json(inventory),
                        "artifacts": {
                            "manifest.json": {
                                **inventory["manifest.json"],
                                "snapshot": str(bundled_task_manifest),
                            },
                            "rounds/round-1/record.json": {
                                **inventory["rounds/round-1/record.json"],
                                "snapshot": str(bundled_dependency),
                            },
                        },
                    },
                },
            }
            manifest_path = fixture.run_task_root / "manifest.json"
            manifest_path.write_bytes(canonical_json_bytes(manifest))
            with self.assertRaisesRegex(
                grade.GradeIntegrityError, "unknown schema"
            ):
                grade.load_claim_manifest(
                    fixture.run_task_root,
                    fixture.corpus,
                    fixture.questions,
                    require_claim_ready=False,
                )
            with patch.object(
                grade,
                "validate_study_note_archive",
                return_value=construction["protocol_summary"],
            ):
                context = grade.load_claim_manifest(
                    fixture.run_task_root,
                    fixture.corpus,
                    fixture.questions,
                    require_claim_ready=False,
                )
            self.assertFalse(context["note_manifest"]["claim_ready"])

            smoke_manifest = deepcopy(manifest)
            smoke_spec = smoke_manifest["spec"]
            smoke_spec["purpose"] = "smoke"
            smoke_spec["claim_ready"] = False
            smoke_spec["preregistration"] = {
                "schema_version": 1,
                "status": "not_provided",
                "reason": "smoke",
            }
            manifest_path.write_bytes(canonical_json_bytes(smoke_manifest))
            with patch.object(
                grade,
                "validate_study_note_archive",
                return_value=construction["protocol_summary"],
            ) as smoke_archive_validator:
                smoke_context = grade.load_claim_manifest(
                    fixture.run_task_root,
                    fixture.corpus,
                    fixture.questions,
                    require_claim_ready=False,
                    allow_smoke=True,
                )
            self.assertTrue(
                smoke_archive_validator.call_args.kwargs["allow_smoke"]
            )
            self.assertEqual(
                smoke_context["note_protocol_summary"],
                construction["protocol_summary"],
            )

            round_three = {**construction, "round": 3}
            round_three_bytes = canonical_json_bytes(round_three)
            for relative in (construction_snapshot, bundled_manifest):
                (fixture.run_task_root / relative).write_bytes(round_three_bytes)
            smoke_manifest["spec"]["note"]["construction_manifest"]["sha256"] = (
                grade.sha256_bytes(round_three_bytes)
            )
            manifest_path.write_bytes(canonical_json_bytes(smoke_manifest))
            with patch.object(
                grade,
                "validate_study_note_archive",
                side_effect=StudyProtocolError(
                    "semantic evaluation requires the final construction round"
                ),
            ), self.assertRaisesRegex(
                grade.GradeIntegrityError, "final construction round"
            ):
                grade.load_claim_manifest(
                    fixture.run_task_root,
                    fixture.corpus,
                    fixture.questions,
                    require_claim_ready=False,
                    allow_smoke=True,
                )

            for relative in (construction_snapshot, bundled_manifest):
                (fixture.run_task_root / relative).write_bytes(construction_bytes)
            manifest_path.write_bytes(canonical_json_bytes(manifest))

            contradictory = {**construction, "publication_claim_ready": True}
            contradictory_bytes = canonical_json_bytes(contradictory)
            for relative in (construction_snapshot, bundled_manifest):
                (fixture.run_task_root / relative).write_bytes(contradictory_bytes)
            manifest["spec"]["note"]["construction_manifest"]["sha256"] = (
                grade.sha256_bytes(contradictory_bytes)
            )
            manifest_path.write_bytes(canonical_json_bytes(manifest))
            with self.assertRaisesRegex(
                grade.GradeIntegrityError,
                "automated construction gates",
            ):
                grade.load_claim_manifest(
                    fixture.run_task_root,
                    fixture.corpus,
                    fixture.questions,
                    require_claim_ready=False,
                )

            for relative in (construction_snapshot, bundled_manifest):
                (fixture.run_task_root / relative).write_bytes(construction_bytes)
            manifest["spec"]["note"]["construction_manifest"]["sha256"] = (
                construction_sha256
            )
            manifest_path.write_bytes(canonical_json_bytes(manifest))

            (fixture.run_task_root / bundled_dependency).write_bytes(b"changed")
            with self.assertRaisesRegex(
                grade.GradeIntegrityError,
                "dependency bytes do not match",
            ):
                grade.load_claim_manifest(
                    fixture.run_task_root,
                    fixture.corpus,
                    fixture.questions,
                    require_claim_ready=False,
                )

    def test_unknown_claim_ready_note_manifest_type_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvaluationFixture(Path(directory))
            note = "verified study note\n"
            note_bytes = note.encode("utf-8")
            note_sha256 = grade.sha256_bytes(note_bytes)
            note_path = fixture.run_task_root / "inputs/note.md"
            note_path.parent.mkdir(parents=True, exist_ok=True)
            note_path.write_bytes(note_bytes)
            construction = {
                "schema_version": 2,
                "study_id": "study-a",
                "task": "fake",
                "corpus_commit": fixture.corpus.commit,
                "claim_ready": True,
                "note_sha256": note_sha256,
                "note_path": "by-sha256/note.md",
            }
            construction_path = fixture.run_task_root / "inputs/construction.json"
            construction_path.write_bytes(canonical_json_bytes(construction))
            manifest = deepcopy(fixture.manifest)
            manifest["spec"]["note"] = {
                "sha256": note_sha256,
                "bytes": len(note_bytes),
                "snapshot": "inputs/note.md",
                "source_name": "note.md",
                "construction_manifest": {
                    "sha256": grade.sha256_bytes(construction_path.read_bytes()),
                    "snapshot": "inputs/construction.json",
                },
            }
            template = "Study note:\n{note}\nQuestion:\n"
            manifest["spec"]["prompt_policy"] = {
                "note_prefix_template": template,
                "presented_prompt_sha256": {
                    "q1": grade.sha256_bytes(
                        (template.format(note=note) + question()["question"]).encode("utf-8")),
                },
            }
            manifest_path = fixture.run_task_root / "manifest.json"
            manifest_path.write_bytes(canonical_json_bytes(manifest))
            with self.assertRaisesRegex(
                grade.GradeIntegrityError, "unknown claim-ready note manifest type"
            ):
                grade.load_claim_manifest(
                    fixture.run_task_root, fixture.corpus, fixture.questions)

    def test_forced_50_note_rebinds_its_intent_episode_and_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvaluationFixture(Path(directory))
            study_root = fixture.root / "study"
            study_root.mkdir()
            note = "verified forced study note\n"
            note_sha256 = grade.sha256_bytes(note.encode("utf-8"))
            study_question = forced50_study_question(fixture.corpus.display)
            question_sha256 = sha256_json(study_question)
            master_seed = 31
            episode_seed = stable_seed(
                master_seed, "cheatsheet", "study-a", "fake"
            )
            config = {
                "schema_version": FORCED50_CONFIG_SCHEMA_VERSION,
                "study_id": "study-a",
                "task": "fake",
                "method": "forced-50-cheatsheet",
                "model": "model",
                "model_revision": "revision-a",
                "expected_response_model": "generation-revision",
                "sampling": REACT_SAMPLING,
                "adapter": DSPY_ADAPTER_NAME,
                "adapter_fallback_policy": DSPY_ADAPTER_POLICY,
                "dspy_request_audit_schema": DSPY_REQUEST_AUDIT_SCHEMA_VERSION,
                "master_seed": master_seed,
                "episode_seed": episode_seed,
                "study_prompt_sha256": grade.sha256_bytes(
                    study_question["question"].encode("utf-8")
                ),
                "study_question_sha256": question_sha256,
                "tool_contract": DSPY_REPOSITORY_TOOL_CONTRACT,
                "tool_schema_sha256": sha256_json(
                    DSPY_REPOSITORY_TOOL_CONTRACT
                ),
                "read_max_lines": DSPY_READ_MAX_LINES,
                "forced_iterations": FORCED50_ITERATIONS,
                "repository_tool_scope": "full-pinned-corpus",
                "corpus": deepcopy(fixture.manifest["spec"]["corpus"]),
                "source": deepcopy(fixture.manifest["spec"]["source"]),
                "environment": deepcopy(fixture.manifest["spec"]["environment"]),
                "claim_ready": True,
                "server_transport": {
                    "scope": "loopback",
                    "protocol": "openai-compatible-http",
                    "available_server_count": 1,
                    "selected_server_index": 0,
                },
            }
            episode = {
                "task": "fake",
                "qid": "cheatsheet",
                "budget": "s50",
                "rollout": 0,
                "model": "model",
                "model_revision": "revision-a",
                "harness": "dspy.ReAct",
                "seed": episode_seed,
                "study_intent_sha256": sha256_json(config),
                "question_sha256": question_sha256,
                "status": "ok",
                "started": "2026-01-01T00:00:00+00:00",
                "finished": "2026-01-01T00:01:00+00:00",
                "answer": note,
                "n_react_iters": 50,
                "n_tool_iters": 50,
                "finish_catches": 0,
                "turns": [{
                    "reasoning": f"step {index}",
                    "tool_calls": [{"name": "grep", "arguments": "{}"}],
                    "observations": ["source"],
                } for index in range(50)],
                "prompt_tokens": 90,
                "completion_tokens": 10,
                "total_tokens": 100,
                "gen_tokens": 10,
                "n_lm_calls": 1,
                "usage_ledger": [{
                    "call": 0,
                    "response_id": "study-response",
                    "response_model": "generation-revision",
                    "system_fingerprint": "study-fingerprint",
                    "request_messages_sha256": "2" * 64,
                    "outputs_sha256": "3" * 64,
                    "provider_usage": {
                        "prompt_tokens": 90,
                        "completion_tokens": 10,
                        "total_tokens": 100,
                    },
                    "prompt_tokens": 90,
                    "completion_tokens": 10,
                    "total_tokens": 100,
                }],
            }
            intent_path = study_root / "intent.json"
            episode_path = study_root / "episode.json"
            note_path = study_root / f"note-{note_sha256}.md"
            intent_path.write_bytes(canonical_json_bytes(config))
            episode_path.write_bytes(canonical_json_bytes(episode))
            note_path.write_text(note)
            inventory = {
                path.name: {
                    "sha256": grade.sha256_bytes(path.read_bytes()),
                    "bytes": len(path.read_bytes()),
                }
                for path in (intent_path, episode_path)
            }
            construction = {
                "manifest_schema": 1,
                "manifest_type": "forced-50-cheatsheet",
                "claim_ready": True,
                "study_id": "study-a",
                "task": "fake",
                "corpus_commit": fixture.corpus.commit,
                "config": config,
                "note_sha256": note_sha256,
                "note_path": note_path.name,
                "episode_sha256": sha256_json(episode),
                "intent_sha256": sha256_json(config),
                "study_generated_tokens": 10,
                "study_prompt_tokens": 90,
                "study_total_tokens": 100,
                "construction_artifacts": inventory,
                "construction_artifacts_sha256": sha256_json(inventory),
            }
            construction_path = study_root / "manifest.json"
            construction_path.write_bytes(canonical_json_bytes(construction))
            _, note_record = _load_note(
                fixture.run_task_root,
                note_path,
                construction_path,
                require_manifest=True,
                expected_task="fake",
                expected_model="model",
                expected_model_revision="revision-a",
                expected_response_model="generation-revision",
                expected_sampling=REACT_SAMPLING,
                expected_corpus_commit=fixture.corpus.commit,
                expected_corpus=fixture.manifest["spec"]["corpus"],
                expected_source=fixture.manifest["spec"]["source"],
                expected_environment=fixture.manifest["spec"]["environment"],
                expected_corpus_display=fixture.corpus.display,
            )
            manifest = deepcopy(fixture.manifest)
            manifest["spec"]["sampling"] = deepcopy(REACT_SAMPLING)
            manifest["spec"]["note"] = note_record
            template = "Study note:\n{note}\nQuestion:\n"
            manifest["spec"]["prompt_policy"] = {
                "note_prefix_template": template,
                "presented_prompt_sha256": {
                    "q1": grade.sha256_bytes(
                        (template.format(note=note) + question()["question"]).encode("utf-8")),
                },
            }
            manifest_path = fixture.run_task_root / "manifest.json"
            manifest_path.write_bytes(canonical_json_bytes(manifest))
            context = grade.load_claim_manifest(
                fixture.run_task_root, fixture.corpus, fixture.questions)
            self.assertEqual(context["note_sha256"], note_sha256)

            parity_mutations = {
                "turns": lambda value: value.update(turns=[]),
                "gen-tokens": lambda value: value.update(gen_tokens=9),
            }
            for label, mutate in parity_mutations.items():
                with self.subTest(preflight_parity=label):
                    invalid_episode = deepcopy(episode)
                    mutate(invalid_episode)
                    invalid_episode_bytes = canonical_json_bytes(invalid_episode)
                    invalid_construction = deepcopy(construction)
                    invalid_construction["episode_sha256"] = sha256_json(
                        invalid_episode
                    )
                    invalid_construction["construction_artifacts"]["episode.json"] = {
                        "sha256": grade.sha256_bytes(invalid_episode_bytes),
                        "bytes": len(invalid_episode_bytes),
                    }
                    invalid_construction["construction_artifacts_sha256"] = sha256_json(
                        invalid_construction["construction_artifacts"]
                    )
                    episode_path.write_bytes(invalid_episode_bytes)
                    construction_path.write_bytes(
                        canonical_json_bytes(invalid_construction)
                    )
                    with self.assertRaisesRegex(ValueError, "study episode"):
                        _load_note(
                            fixture.root / f"preflight-{label}",
                            note_path,
                            construction_path,
                            require_manifest=True,
                            expected_task="fake",
                            expected_model="model",
                            expected_model_revision="revision-a",
                            expected_response_model="generation-revision",
                            expected_sampling=REACT_SAMPLING,
                            expected_corpus_commit=fixture.corpus.commit,
                            expected_corpus=fixture.manifest["spec"]["corpus"],
                            expected_source=fixture.manifest["spec"]["source"],
                            expected_environment=fixture.manifest["spec"]["environment"],
                            expected_corpus_display=fixture.corpus.display,
                        )

                    _, legacy_record = _load_note(
                        fixture.run_task_root,
                        note_path,
                        construction_path,
                        require_manifest=True,
                        expected_task="fake",
                        expected_corpus_commit=fixture.corpus.commit,
                    )
                    invalid_manifest = deepcopy(fixture.manifest)
                    invalid_manifest["spec"]["sampling"] = deepcopy(REACT_SAMPLING)
                    invalid_manifest["spec"]["note"] = legacy_record
                    invalid_manifest["spec"]["prompt_policy"] = deepcopy(
                        manifest["spec"]["prompt_policy"]
                    )
                    manifest_path.write_bytes(canonical_json_bytes(invalid_manifest))
                    with self.assertRaises(grade.GradeIntegrityError):
                        grade.load_claim_manifest(
                            fixture.run_task_root, fixture.corpus, fixture.questions
                        )

            episode_path.write_bytes(canonical_json_bytes(episode))
            construction_path.write_bytes(canonical_json_bytes(construction))
            manifest_path.write_bytes(canonical_json_bytes(manifest))

            construction_snapshot = fixture.run_task_root / note_record[
                "construction_manifest"
            ]["snapshot"]
            for field_path in (("manifest_schema",), ("config", "schema_version")):
                with self.subTest(field_path=field_path):
                    invalid_construction = deepcopy(construction)
                    target = invalid_construction
                    for field in field_path[:-1]:
                        target = target[field]
                    target[field_path[-1]] = True
                    invalid_bytes = canonical_json_bytes(invalid_construction)
                    construction_snapshot.write_bytes(invalid_bytes)
                    invalid_manifest = deepcopy(manifest)
                    invalid_manifest["spec"]["note"]["construction_manifest"][
                        "sha256"
                    ] = grade.sha256_bytes(invalid_bytes)
                    manifest_path.write_bytes(canonical_json_bytes(invalid_manifest))
                    with self.assertRaisesRegex(
                        grade.GradeIntegrityError,
                        "forced-50 (construction manifest|protocol binding)",
                    ):
                        grade.load_claim_manifest(
                            fixture.run_task_root, fixture.corpus, fixture.questions)

            def invalid_config(label, mutate) -> None:
                invalid_construction = deepcopy(construction)
                mutate(invalid_construction["config"])
                invalid_bytes = canonical_json_bytes(invalid_construction)
                construction_snapshot.write_bytes(invalid_bytes)
                invalid_manifest = deepcopy(manifest)
                invalid_manifest["spec"]["note"]["construction_manifest"][
                    "sha256"
                ] = grade.sha256_bytes(invalid_bytes)
                manifest_path.write_bytes(canonical_json_bytes(invalid_manifest))
                with self.subTest(protocol_field=label), self.assertRaises(
                    grade.GradeIntegrityError
                ):
                    grade.load_claim_manifest(
                        fixture.run_task_root, fixture.corpus, fixture.questions
                    )

            invalid_config(
                "unknown-key", lambda value: value.__setitem__("unknown", True)
            )
            invalid_config(
                "sampling", lambda value: value.__setitem__(
                    "sampling", {"temperature": 0}
                )
            )
            invalid_config(
                "read-max", lambda value: value.__setitem__("read_max_lines", 201)
            )
            invalid_config(
                "tool-contract", lambda value: value["tool_contract"].__setitem__(
                    "adapter", "other"
                )
            )
            invalid_config(
                "tool-hash", lambda value: value.__setitem__(
                    "tool_schema_sha256", "0" * 64
                )
            )
            invalid_config(
                "prompt-hash", lambda value: value.__setitem__(
                    "study_prompt_sha256", "0" * 64
                )
            )
            invalid_config(
                "question-hash", lambda value: value.__setitem__(
                    "study_question_sha256", "0" * 64
                )
            )
            invalid_config(
                "master-seed", lambda value: value.__setitem__(
                    "master_seed", value["master_seed"] + 1
                )
            )
            invalid_config(
                "episode-seed", lambda value: value.__setitem__(
                    "episode_seed", value["episode_seed"] + 1
                )
            )
            invalid_config(
                "server-index-bool", lambda value: value[
                    "server_transport"
                ].__setitem__("selected_server_index", False)
            )
            invalid_config(
                "server-count", lambda value: value[
                    "server_transport"
                ].__setitem__("available_server_count", 2)
            )
            invalid_config(
                "model", lambda value: value.__setitem__("model", "other")
            )
            invalid_config(
                "response-model", lambda value: value.__setitem__(
                    "expected_response_model", "other"
                )
            )
            invalid_config(
                "source", lambda value: value["source"].__setitem__(
                    "git_commit", "b" * 40
                )
            )
            invalid_config(
                "environment", lambda value: value["environment"].__setitem__(
                    "vllm_version", "other"
                )
            )

            construction_snapshot.write_bytes(canonical_json_bytes(construction))
            manifest_path.write_bytes(canonical_json_bytes(manifest))
            with patch(
                "studybench.grade.provenance.environments_compatible",
                return_value=False,
            ), self.assertRaises(grade.GradeIntegrityError):
                grade.load_claim_manifest(
                    fixture.run_task_root, fixture.corpus, fixture.questions
                )

            construction_snapshot.write_bytes(canonical_json_bytes(construction))
            manifest_path.write_bytes(canonical_json_bytes(manifest))

            snapshot = note_record["provenance_bundle"]["construction_artifacts"][
                "artifacts"
            ]["episode.json"]["snapshot"]
            (fixture.run_task_root / snapshot).write_bytes(b"tampered\n")
            with self.assertRaises(grade.GradeIntegrityError):
                grade.load_claim_manifest(
                    fixture.run_task_root, fixture.corpus, fixture.questions)


if __name__ == "__main__":
    unittest.main()
