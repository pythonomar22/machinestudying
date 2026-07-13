from __future__ import annotations

import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from studybench import grade, report
from studybench.integrity import canonical_json_bytes, sha256_json, stable_seed
from studybench.provenance import _load_note, environment_contract_record


TEST_JUDGE_BASE_URL = "https://judge.test/v1"


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
        self.assertIn("if gf.exists():", source)

    def test_grader_disables_hidden_sdk_retries(self) -> None:
        source = Path(grade.__file__).read_text()
        self.assertEqual(source.count("AsyncOpenAI("), 1)
        self.assertIn("max_retries=0", source)

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
                "round": 1,
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


class EvaluationFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.run_id = "run-a"
        self.judge_model = "gpt-5.4"
        self.judge_dir = "gpt-5.4-excerpts"
        self.corpus = SimpleNamespace(
            name="fake",
            display="Fake",
            repo=root / "corpus",
            roots=("src",),
            language="python",
            commit="c" * 40,
            code_suffixes=(".py",),
        )
        self.questions = [question()]
        self.run_root = root / "runs" / self.run_id
        self.run_task_root = self.run_root / "fake"
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
                11, "native-react", seed_group, "fake", "q1", budget, 0)
        spec = {
            "schema_version": 1,
            "run_id": self.run_id,
            "task": "fake",
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
                "name": "fake",
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
            },
        }
        spec["environment_contract"] = environment_contract_record(spec["environment"])
        return {"manifest_schema": 1, "spec": spec}

    def _write_population(self) -> None:
        client = FakeClient([verdict()] * 3)
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
            spec_sha256 = grade.grade_spec_sha256(
                self.corpus, self.questions[0], self.judge_model)
            if status == "ok":
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
                ))
            stored["source_episode"] = episode_path.relative_to(self.root).as_posix()
            grade_path = self.grade_root / "fake" / relative
            grade_path.parent.mkdir(parents=True, exist_ok=True)
            grade_path.write_bytes(canonical_json_bytes(stored))

    def patches(self):
        return (
            patch.object(report, "ROOT", self.root),
            patch.object(report, "CORPORA", {"fake": self.corpus}),
            patch.object(report, "load_questions", return_value=self.questions),
        )


class StrictReportTests(unittest.TestCase):
    def setUp(self) -> None:
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
        self.environment_snapshot_patch.stop()
        self.environment_patch.stop()

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

    def test_failed_judge_audits_are_validated_and_disclosed(self) -> None:
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
            population, audit = self._load(fixture)
            self.assertEqual(sum(len(values) for values in population.values()), 4)
            inventory = audit["failed_judge_audits"]
            self.assertEqual(inventory["count"], 1)
            self.assertTrue(inventory["artifacts"][0]["all_bindings_current"])
            self.assertEqual(
                inventory["artifacts"][0]["judge_usage_total"]["total_tokens"],
                360,
            )

    def test_report_discloses_failed_response_with_unavailable_usage(self) -> None:
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
            population, audit = self._load(fixture)
            self.assertEqual(sum(len(values) for values in population.values()), 4)
            artifact = audit["failed_judge_audits"]["artifacts"][0]
            self.assertEqual(
                artifact["judge_usage_status"],
                "unavailable-for-response-without-usage",
            )
            self.assertIsNone(artifact["judge_usage_total"])
            self.assertEqual(artifact["judge_usage_known_total"]["total_tokens"], 0)
            self.assertEqual(artifact["incomplete_response_fields"], ["usage"])

    def test_missing_grade_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = EvaluationFixture(Path(directory))
            (fixture.grade_root / "fake/k5/r0/q1.json").unlink()
            with self.assertRaises(report.ReportIntegrityError):
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
            question_sha256 = "1" * 64
            config = {
                "schema_version": 1,
                "study_id": "study-a",
                "task": "fake",
                "method": "forced-50-cheatsheet",
                "model": "model",
                "model_revision": "revision-a",
                "expected_response_model": "generation-revision",
                "episode_seed": 19,
                "study_question_sha256": question_sha256,
                "forced_iterations": 50,
                "corpus": deepcopy(fixture.manifest["spec"]["corpus"]),
                "source": deepcopy(fixture.manifest["spec"]["source"]),
                "environment": deepcopy(fixture.manifest["spec"]["environment"]),
                "claim_ready": True,
            }
            episode = {
                "task": "fake",
                "qid": "cheatsheet",
                "budget": "s50",
                "rollout": 0,
                "model": "model",
                "model_revision": "revision-a",
                "harness": "dspy.ReAct",
                "seed": 19,
                "study_intent_sha256": sha256_json(config),
                "question_sha256": question_sha256,
                "status": "ok",
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
                expected_corpus_commit=fixture.corpus.commit,
            )
            manifest = deepcopy(fixture.manifest)
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
                        "forced-50 construction manifest is incomplete",
                    ):
                        grade.load_claim_manifest(
                            fixture.run_task_root, fixture.corpus, fixture.questions)

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
