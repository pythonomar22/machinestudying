from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from studybench import grade, raw_qwen_screen as raw, report
from studybench.dataset import CORPORA, ROOT
from studybench.integrity import read_artifact_bytes, sha256_bytes, strict_json_loads


def row() -> dict:
    return {
        "id": "q1",
        "topic": "topic",
        "question": "Full question",
        "gold_answer": "Full gold answer",
        "rubric": [
            {
                "claim_id": "c1",
                "claim_type": "core",
                "weight": 60,
                "statement": "First claim",
                "span_ids": ["s1"],
            },
            {
                "claim_id": "c2",
                "claim_type": "supporting",
                "weight": 40,
                "statement": "Second claim",
                "span_ids": ["s1"],
            },
        ],
        "evidence": [
            {
                "span_id": "s1",
                "path": "dspy/predict/example.py",
                "start_line": 1,
                "end_line": 2,
                "excerpt": "full evidence\nsecond line",
            }
        ],
    }


def response(content: str, *, finish_reason: str = "stop") -> SimpleNamespace:
    return SimpleNamespace(
        id="response-1",
        _request_id="request-1",
        model=grade.LOCAL_GRADER_MODEL,
        system_fingerprint="fingerprint-1",
        usage={
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content),
            finish_reason=finish_reason,
        )],
    )


class Client:
    def __init__(self, returned, before_call=None):
        self.returned = returned
        self.before_call = before_call
        self.calls = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create)
        )

    async def create(self, **kwargs):
        if self.before_call is not None:
            self.before_call()
        self.calls.append(kwargs)
        return self.returned


class RawRequestTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_bundle_and_fixed_request_are_sent_once(self) -> None:
        candidate = "candidate start\n" + ("x" * 10_000) + "\ncandidate end"
        question = row()
        messages = grade.build_judge_messages(
            CORPORA[raw.TASK], question, candidate, False,
            grade.LOCAL_GRADER_MODEL,
        )
        payload = {
            "model": grade.LOCAL_GRADER_MODEL,
            "messages": messages,
            "response_format": grade.judge_schema(
                question, grade.LOCAL_GRADER_MODEL
            ),
            **raw.RAW_REQUEST_OPTIONS,
        }
        request = {
            "request_index": 0,
            "arm": raw.BASE_ARM,
            "relative": "direct/r0/q1.json",
            "qid": "q1",
            "server_slot": 0,
            "url": "http://localhost:8000/v1",
            "payload": payload,
            "payload_sha256": raw.sha256_json(payload),
        }
        client = Client(response('{"claims":{"c1":1,"c2":0},"needs_regrade":false}'))
        observed = await raw._one_request(
            request, question, client, asyncio.Semaphore(1)
        )

        self.assertTrue(observed["accepted"])
        self.assertEqual(observed["lenient"], 60)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0], payload)
        user = json.loads(client.calls[0]["messages"][1]["content"])
        self.assertEqual(user["candidate_answer"], candidate)
        self.assertEqual(user["question"], question["question"])
        self.assertEqual(user["gold_answer"], question["gold_answer"])
        self.assertEqual(user["claim_rubric"], question["rubric"])
        self.assertEqual(user["evidence"], question["evidence"])
        self.assertEqual(
            {key: client.calls[0][key] for key in raw.RAW_REQUEST_OPTIONS},
            raw.RAW_REQUEST_OPTIONS,
        )

    async def test_malformed_length_and_needs_regrade_fail_closed(self) -> None:
        question = row()
        base_request = {
            "request_index": 0,
            "arm": raw.BASE_ARM,
            "relative": "direct/r0/q1.json",
            "qid": "q1",
            "server_slot": 0,
            "url": "http://localhost:8000/v1",
            "payload": {},
            "payload_sha256": "a" * 64,
        }
        cases = (
            ("not json", "stop"),
            ('{"claims":{"c1":1,"c2":1},"needs_regrade":false}', "length"),
            ('{"claims":{"c1":1,"c2":1},"needs_regrade":true}', "stop"),
        )
        for content, finish_reason in cases:
            with self.subTest(content=content, finish_reason=finish_reason):
                client = Client(response(content, finish_reason=finish_reason))
                observed = await raw._one_request(
                    base_request, question, client, asyncio.Semaphore(1)
                )
                self.assertFalse(observed["accepted"])
                self.assertIsNotNone(observed["validation_error"])
                self.assertEqual(observed["content"], content)
                self.assertEqual(len(client.calls), 1)


class PinnedScopeTests(unittest.TestCase):
    def test_exact_failed_qualification_and_manifests_match_frozen_hashes(self) -> None:
        qualification = raw._failed_qualification(
            ROOT / raw.FAILED_QUALIFICATION_PATH
        )
        self.assertEqual(
            qualification["sha256"], raw.FAILED_QUALIFICATION_SHA256
        )
        for run_id, expected in raw.RUN_MANIFEST_SHA256.items():
            observed = sha256_bytes(read_artifact_bytes(
                ROOT / "runs" / run_id / raw.TASK / "manifest.json"
            ))
            self.assertEqual(observed, expected)

    def test_alternate_runs_and_failed_audits_are_rejected(self) -> None:
        with self.assertRaisesRegex(raw.RawQwenScreenError, "pinned"):
            raw.prepare_screen(
                base_run_id="alternate-base",
                treatment_run_id=raw.TREATMENT_RUN_ID,
                judge_base_urls="http://localhost:8000/v1",
                failed_qualification_audit=ROOT / raw.FAILED_QUALIFICATION_PATH,
            )
        with tempfile.TemporaryDirectory() as directory:
            alternate = Path(directory) / Path(raw.FAILED_QUALIFICATION_PATH).name
            alternate.write_bytes(read_artifact_bytes(
                ROOT / raw.FAILED_QUALIFICATION_PATH
            ))
            with self.assertRaisesRegex(raw.RawQwenScreenError, "exact frozen"):
                raw._failed_qualification(alternate)


class DurableAuditTests(unittest.TestCase):
    def test_response_census_rejects_missing_duplicate_and_content_drift(self) -> None:
        request = {
            "request_index": 0,
            "arm": raw.BASE_ARM,
            "relative": "direct/r0/q1.json",
            "qid": "q1",
            "server_slot": 0,
            "url": "http://localhost:8000/v1",
            "payload_sha256": "a" * 64,
        }
        content = '{"claims":{"c1":1,"c2":0},"needs_regrade":false}'
        accepted = {
            **request,
            "accepted": True,
            "content": content,
            "response": {
                "content_sha256": raw.sha256_bytes(content.encode("utf-8")),
                "content_bytes": len(content.encode("utf-8")),
            },
        }
        intent = {"requests": [request]}
        raw._validate_response_census(intent, [accepted])
        for responses in ([], [accepted, accepted]):
            with self.subTest(count=len(responses)), self.assertRaises(
                raw.RawQwenScreenError
            ):
                raw._validate_response_census(intent, responses)
        drifted = {**accepted, "content": content + " "}
        with self.assertRaisesRegex(raw.RawQwenScreenError, "content differs"):
            raw._validate_response_census(intent, [drifted])

    def test_intent_precedes_contact_and_failure_writes_terminal_raw_audit(self) -> None:
        question = row()
        url = "http://localhost:8000/v1"
        request = {
            "request_index": 0,
            "arm": raw.BASE_ARM,
            "relative": "direct/r0/q1.json",
            "qid": "q1",
            "server_slot": 0,
            "url": url,
            "payload": {},
            "payload_sha256": "a" * 64,
        }
        intent = {
            "claim_ready": False,
            "judge_qualified": False,
            "failed_qualification_audit_sha256": "f" * 64,
            "judge": {"ordered_urls": [url]},
            "estimand": {},
            "requests": [request],
        }
        prepared = raw.PreparedScreen(intent, [], {"q1": question})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            client = Client(
                response("malformed"),
                before_call=lambda: self.assertTrue((root / "intent.json").is_file()),
            )
            with self.assertRaisesRegex(raw.RawQwenScreenError, "failed closed"):
                asyncio.run(raw.run_prepared_screen(
                    prepared,
                    output_dir=root,
                    client_factory=lambda _: client,
                    provenance_revalidator=lambda _: None,
                ))
            audit = strict_json_loads(
                read_artifact_bytes(root / "raw-audit.json"), label="raw audit"
            )
            self.assertFalse(audit["complete"])
            self.assertEqual(audit["request_count"], 1)
            self.assertEqual(audit["accepted_count"], 0)
            self.assertEqual(audit["responses"][0]["content"], "malformed")
            self.assertEqual(len(client.calls), 1)
            self.assertFalse((root / "result.json").exists())


class AggregateTests(unittest.TestCase):
    def test_expected_population_is_exactly_120_cells_and_119_requests(self) -> None:
        relatives = [
            f"{budget}/r{rollout}/q{question}.json"
            for budget in report.BUDGET_ORDER
            for rollout in range(3)
            for question in range(5)
        ]
        spec = {
            "task": raw.TASK,
            "budgets": report.BUDGET_ORDER,
            "rollouts": 3,
            "questions": [f"q{question}" for question in range(5)],
            "expected_episodes": relatives,
        }
        base = [
            {"episode": {"status": "ok"}} for _ in relatives
        ]
        treatment = [
            {"episode": {"status": "no_answer" if index == 0 else "ok"}}
            for index, _ in enumerate(relatives)
        ]
        raw._paired_inputs(({"spec": spec}, base), ({"spec": spec}, treatment))
        self.assertEqual(len(base) + len(treatment), 120)
        self.assertEqual(
            sum(item["episode"]["status"] == "ok" for item in base + treatment),
            119,
        )

    def test_treatment_no_answer_is_an_itt_zero(self) -> None:
        cells = []
        responses = []
        for budget_index, budget in enumerate(report.BUDGET_ORDER, 1):
            for rollout in range(3):
                for question in range(5):
                    relative = f"{budget}/r{rollout}/q{question}.json"
                    no_answer = budget == "direct" and rollout == 0 and question == 0
                    cells.append({
                        "arm": raw.TREATMENT_ARM,
                        "relative": relative,
                        "qid": f"q{question}",
                        "budget": budget,
                        "rollout": rollout,
                        "status": "no_answer" if no_answer else "ok",
                        "gen_tokens": budget_index * 100,
                    })
                    if not no_answer:
                        responses.append({
                            "arm": raw.TREATMENT_ARM,
                            "relative": relative,
                            "lenient": 100,
                            "cores_ok": True,
                        })
        summary, population = raw._arm_summary(
            raw.TREATMENT_ARM, cells, responses
        )
        direct = summary["budgets"]["direct"]
        self.assertEqual(direct["answered"], 14)
        self.assertEqual(direct["no_answer"], 1)
        self.assertAlmostEqual(direct["mean_lenient"], 1400 / 15)
        self.assertEqual(population["direct"][0]["lenient"], 0)
        expected = report.expertise([
            (
                summary["budgets"][budget]["mean_gen_tokens"],
                summary["budgets"][budget]["mean_lenient"],
            )
            for budget in report.BUDGET_ORDER
        ])
        self.assertEqual(summary["lenient_wauc"], expected)
        self.assertEqual(summary["bootstrap"]["replicates"], 10_000)
        self.assertEqual(summary["bootstrap"]["seed"], 45_001)


if __name__ == "__main__":
    unittest.main()
