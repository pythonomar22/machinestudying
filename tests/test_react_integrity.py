from __future__ import annotations

import copy
import json
from pathlib import Path
import random
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import dspy
from dspy.adapters.base import Adapter
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.json_adapter import JSONAdapter
from dspy.utils.exceptions import AdapterParseError

from studybench.integrity import (
    canonical_json_bytes,
    sha256_json,
    sha256_text,
    write_immutable_json,
)
from studybench.grade import GradeIntegrityError, validate_episode
from studybench.react import (
    READ_MAX_LINES,
    ParseOnlyFallbackChatAdapter,
    TrajectoryRecordingReAct,
    _artifact_inventory,
    _dspy_final_binding,
    _dspy_usage_record,
    _episode_server_url,
    _stored_completed_study_config,
    _validate_completed_study,
    make_tools,
    runtime_dspy_tool_contract,
)
from studybench.rollout import _validate_final_episode
from studybench.provenance import (
    server_assignment_record,
    validate_local_server_urls,
)
from studybench.study_protocol import (
    _DspyAdapterParseFailure,
    _dspy_output_field_contract,
    _replay_dspy_chat_parse,
    _replay_dspy_json_parse,
    DSPY_REPOSITORY_TOOL_CONTRACT,
    DSPY_REQUEST_AUDIT_LEGACY_SCHEMA_VERSION,
    DSPY_REQUEST_AUDIT_SCHEMA_VERSION,
)
from studybench.tools import READ_MAX_LINES as NATIVE_READ_MAX_LINES


class ReactStudyIntegrityTests(unittest.TestCase):
    @staticmethod
    def make_current_call(
        index: int,
        output: str | None,
        *,
        finish_reason: str = "stop",
        reasoning_content: str | None = None,
        tool_calls: list[dict] | None = None,
        choice_index: object = 0,
    ) -> dict:
        usage = {
            "prompt_tokens": 2,
            "completion_tokens": 3,
            "total_tokens": 5,
        }
        response = {
            "id": f"response-{index}",
            "model": "served-model",
            "system_fingerprint": "fingerprint",
            "choices": [{
                "index": choice_index,
                "finish_reason": finish_reason,
                "message": {
                    "role": "assistant",
                    "content": output,
                    "reasoning_content": reasoning_content,
                    "tool_calls": tool_calls,
                },
            }],
            "usage": usage,
            "_hidden_params": {"api_key": "must-not-survive"},
            "headers": {"authorization": "must-not-survive"},
        }
        return _dspy_usage_record({
            "usage": usage,
            "messages": [{"role": "user", "content": "question"}],
            "outputs": [
                {
                    "text": output,
                    **(
                        {"reasoning_content": reasoning_content}
                        if reasoning_content
                        else {}
                    ),
                    **({"tool_calls": tool_calls} if tool_calls else {}),
                }
            ] if reasoning_content or tool_calls else [output],
            "response": response,
            "response_model": "served-model",
        }, index)

    @staticmethod
    def make_current_episode(
        ledger: list[dict],
        parse_audit: dict,
        *,
        answer: str,
        status: str = "ok",
        stage: str = "direct",
        budget: str = "direct",
    ) -> dict:
        episode = {
            "task": "dspy",
            "qid": "q1",
            "budget": budget,
            "rollout": 0,
            "model": "model",
            "model_revision": "revision",
            "harness": "dspy.ReAct",
            "seed": 17,
            "started": "start",
            "finished": "finish",
            "status": status,
            "answer": answer,
            "turns": [],
            "n_react_iters": 0,
            "n_tool_iters": 0,
            "finish_catches": 0,
            "n_lm_calls": len(ledger),
            "usage_ledger": ledger,
            "prompt_tokens": sum(call["prompt_tokens"] for call in ledger),
            "completion_tokens": sum(call["completion_tokens"] for call in ledger),
            "total_tokens": sum(call["total_tokens"] for call in ledger),
            "gen_tokens": sum(call["completion_tokens"] for call in ledger),
            "dspy_request_audit_schema": DSPY_REQUEST_AUDIT_SCHEMA_VERSION,
        }
        binding = _dspy_final_binding(
            parse_audit,
            ledger,
            stage=stage,
            status=status,
            answer=answer,
        )
        episode[
            "answer_audit" if status == "ok" else "non_answer_audit"
        ] = binding
        return episode

    def test_episode_routing_uses_canonical_slot_after_pending_filtering(self) -> None:
        episodes = [f"direct/r0/q{index}.json" for index in range(6)]
        slots = server_assignment_record(episodes, 3)["episode_slots"]
        pending = [episodes[2], episodes[5]]
        urls = validate_local_server_urls(
            "http://localhost:8302/v1,http://localhost:8300/v1,"
            "http://localhost:8301/v1"
        )
        permuted = validate_local_server_urls(
            "http://127.0.0.1:8301/v1,http://localhost:8302/v1,"
            "http://[::1]:8300/v1"
        )
        self.assertEqual(permuted, urls)
        self.assertEqual(
            [
                _episode_server_url(urls, {"server_slot": slots[relative]})
                for relative in pending
            ],
            ["http://localhost:8302/v1", "http://localhost:8302/v1"],
        )

    def test_chat_to_json_fallback_is_parse_only_and_response_audited(self) -> None:
        signature = dspy.Signature("question -> answer")
        lm = SimpleNamespace(provider_attempt_count=0, history=[])
        parse_error = AdapterParseError(
            adapter_name="ChatAdapter", signature=signature,
            lm_response="reasoning only",
        )

        def completed_parse_failure(*_args, **_kwargs):
            lm.provider_attempt_count += 1
            lm.history.append({"complete": True})
            raise parse_error

        json_adapter = Mock()
        def completed_json_repair(*_args, **_kwargs):
            lm.provider_attempt_count += 1
            lm.history.append({"complete": True})
            return [{"answer": "repaired"}]

        json_adapter.side_effect = completed_json_repair
        adapter = ParseOnlyFallbackChatAdapter()
        with patch.object(Adapter, "__call__", side_effect=completed_parse_failure), \
                patch("studybench.react.JSONAdapter", return_value=json_adapter):
            result = adapter(
                lm, {}, signature, [], {"question": "q"}
            )
        self.assertEqual(result, [{"answer": "repaired"}])
        json_adapter.assert_called_once()
        self.assertEqual(adapter.call_audits[0]["provider_calls"], [0, 1])
        self.assertEqual(adapter.call_audits[0]["adapter"], "JSONAdapter")

        with patch.object(Adapter, "__call__", side_effect=RuntimeError("transport")), \
                patch("studybench.react.JSONAdapter") as json_cls:
            with self.assertRaisesRegex(RuntimeError, "transport"):
                ParseOnlyFallbackChatAdapter()(
                    lm, {}, signature, [], {"question": "q"}
                )
            json_cls.assert_not_called()

        with patch.object(Adapter, "__call__", side_effect=parse_error), \
                patch("studybench.react.JSONAdapter") as json_cls:
            with self.assertRaisesRegex(RuntimeError, "complete provider response"):
                ParseOnlyFallbackChatAdapter()(
                    lm, {}, signature, [], {"question": "q"}
                )
            json_cls.assert_not_called()

    def test_chat_success_and_exhausted_json_repair_are_response_audited(self) -> None:
        signature = dspy.Signature("question -> answer")
        lm = SimpleNamespace(provider_attempt_count=0, history=[])

        def chat_success(*_args, **_kwargs):
            lm.provider_attempt_count += 1
            lm.history.append({"complete": True})
            return [{"answer": "answer"}]

        adapter = ParseOnlyFallbackChatAdapter()
        with patch.object(Adapter, "__call__", side_effect=chat_success):
            self.assertEqual(
                adapter(lm, {}, signature, [], {"question": "q"}),
                [{"answer": "answer"}],
            )
        self.assertEqual(adapter.call_audits, [{
            "status": "parsed",
            "adapter": "ChatAdapter",
            "fallback_used": False,
            "provider_calls": [0],
            "provider_call": 0,
            "choice": 0,
            "parsed_outputs": [{"answer": "answer"}],
        }])

        primary_error = AdapterParseError(
            adapter_name="ChatAdapter", signature=signature,
            lm_response="primary",
        )
        fallback_error = AdapterParseError(
            adapter_name="JSONAdapter", signature=signature,
            lm_response="fallback",
            parsed_result={},
        )

        def primary_failure(*_args, **_kwargs):
            lm.provider_attempt_count += 1
            lm.history.append({"complete": True})
            raise primary_error

        def fallback_failure(*_args, **_kwargs):
            lm.provider_attempt_count += 1
            lm.history.append({"complete": True})
            raise fallback_error

        failed_adapter = ParseOnlyFallbackChatAdapter()
        json_adapter = Mock(side_effect=fallback_failure)
        with patch.object(Adapter, "__call__", side_effect=primary_failure), \
                patch("studybench.react.JSONAdapter", return_value=json_adapter):
            with self.assertRaises(AdapterParseError):
                failed_adapter(lm, {}, signature, [], {"question": "q"})
        self.assertEqual(
            failed_adapter.call_audits[0]["provider_calls"], [1, 2]
        )
        self.assertEqual(
            failed_adapter.call_audits[0]["lm_response"], "fallback"
        )

    def test_current_provider_retention_excludes_sdk_and_auth_internals(self) -> None:
        record = self.make_current_call(
            0,
            "[[ ## answer ## ]]\nkept exactly",
            finish_reason="length",
        )
        self.assertEqual(
            set(record["processed_response"]),
            {"id", "model", "system_fingerprint", "choices", "usage"},
        )
        serialized = json.dumps(record, sort_keys=True)
        self.assertNotIn("must-not-survive", serialized)
        self.assertEqual(record["finish_reasons"], ["length"])
        self.assertEqual(record["outputs"], ["[[ ## answer ## ]]\nkept exactly"])
        with self.assertRaisesRegex(ValueError, "retention is invalid"):
            self.make_current_call(
                0,
                "[[ ## answer ## ]]\nanswer",
                choice_index=False,
            )
        for invalid_field in (
            {"reasoning_content": 0},
            {"tool_calls": {}},
        ):
            with self.subTest(invalid_field=invalid_field), self.assertRaisesRegex(
                ValueError, "retention is invalid"
            ):
                self.make_current_call(
                    0,
                    "[[ ## answer ## ]]\nanswer",
                    **invalid_field,
                )

    def test_current_length_answer_remains_evaluable_and_exactly_bound(self) -> None:
        output = "[[ ## answer ## ]]\nkept exactly"
        ledger = [self.make_current_call(0, output, finish_reason="length")]
        episode = self.make_current_episode(
            ledger,
            {
                "status": "parsed",
                "adapter": "ChatAdapter",
                "fallback_used": False,
                "provider_calls": [0],
                "provider_call": 0,
                "choice": 0,
                "parsed_outputs": [{"answer": "kept exactly"}],
            },
            answer="kept exactly",
        )
        validate_episode(episode, {"id": "q1"})
        _validate_final_episode(
            episode,
            {
                "task": "dspy", "qid": "q1", "budget": "direct",
                "rollout": 0, "seed": 17,
            },
            expected_model="model",
            expected_model_revision="revision",
            expected_harness="dspy.ReAct",
            expected_response_model="served-model",
        )

    def test_current_json_fallback_and_parse_non_answer_bind_final_call(self) -> None:
        first = self.make_current_call(0, "not chat fields")
        second = self.make_current_call(1, '{"answer":"repaired"}')
        repaired = self.make_current_episode(
            [first, second],
            {
                "status": "parsed",
                "adapter": "JSONAdapter",
                "fallback_used": True,
                "provider_calls": [0, 1],
                "provider_call": 1,
                "choice": 0,
                "parsed_outputs": [{"answer": "repaired"}],
            },
            answer="repaired",
        )
        validate_episode(repaired, {"id": "q1"})

        failed = self.make_current_episode(
            [first, self.make_current_call(1, "still malformed")],
            {
                "status": "parse_failure",
                "adapter": "JSONAdapter",
                "fallback_used": True,
                "provider_calls": [0, 1],
                "provider_call": 1,
                "choice": 0,
                "lm_response": "still malformed",
                "parsed_result": None,
            },
            answer="",
            status="no_answer",
        )
        validate_episode(failed, {"id": "q1"})

    def test_current_mapping_outputs_bind_success_and_exact_parse_failure_text(self) -> None:
        output = "[[ ## answer ## ]]\nanswer"
        success_call = self.make_current_call(
            0, output, reasoning_content="native reasoning"
        )
        success = self.make_current_episode(
            [success_call],
            {
                "status": "parsed",
                "adapter": "ChatAdapter",
                "fallback_used": False,
                "provider_calls": [0],
                "provider_call": 0,
                "choice": 0,
                "parsed_outputs": [{"answer": "answer"}],
            },
            answer="answer",
        )
        validate_episode(success, {"id": "q1"})

        primary_call = self.make_current_call(0, "not chat fields")
        malformed_call = self.make_current_call(
            1, None, reasoning_content="native reasoning"
        )
        self.assertEqual(
            list(malformed_call["outputs"][0]),
            ["reasoning_content", "text"],
        )
        exact_adapter_value = str({
            "text": None,
            "reasoning_content": "native reasoning",
        })
        failed = self.make_current_episode(
            [primary_call, malformed_call],
            {
                "status": "parse_failure",
                "adapter": "JSONAdapter",
                "fallback_used": True,
                "provider_calls": [0, 1],
                "provider_call": 1,
                "choice": 0,
                "lm_response": exact_adapter_value,
                "parsed_result": None,
            },
            answer="",
            status="no_answer",
        )
        validate_episode(failed, {"id": "q1"})
        self.assertEqual(
            failed["non_answer_audit"]["adapter_lm_response"],
            exact_adapter_value,
        )

    def test_current_tool_call_only_output_remains_a_model_non_answer(self) -> None:
        tool_calls = [{
            "id": "call-1",
            "type": "function",
            "function": {"name": "unexpected", "arguments": "{}"},
        }]
        episode = self.make_current_episode(
            [
                self.make_current_call(0, "not chat fields"),
                self.make_current_call(1, None, tool_calls=tool_calls),
            ],
            {
                "status": "parse_failure",
                "adapter": "JSONAdapter",
                "fallback_used": True,
                "provider_calls": [0, 1],
                "provider_call": 1,
                "choice": 0,
                # The producer captures the actual LiteLLM SDK-object repr;
                # its value is diagnostic, while the provider fields prove the
                # parse failure for signatures without a ToolCalls output.
                "lm_response": "exact runtime SDK repr retained by producer",
                "parsed_result": None,
            },
            answer="",
            status="no_answer",
        )
        validate_episode(episode, {"id": "q1"})
        self.assertEqual(episode["status"], "no_answer")
        self.assertEqual(episode["answer"], "")

    def test_current_retention_and_answer_binding_fail_closed_on_mutation(self) -> None:
        output = "[[ ## answer ## ]]\nanswer"
        ledger = [self.make_current_call(0, output)]
        episode = self.make_current_episode(
            ledger,
            {
                "status": "parsed",
                "adapter": "ChatAdapter",
                "fallback_used": False,
                "provider_calls": [0],
                "provider_call": 0,
                "choice": 0,
                "parsed_outputs": [{"answer": "answer"}],
            },
            answer="answer",
        )
        mutations = (
            lambda value: value["usage_ledger"][0].update(extra="field"),
            lambda value: value["usage_ledger"][0]["processed_response"].update(
                extra="field"
            ),
            lambda value: value["usage_ledger"][0].update(finish_reasons=[""]),
            lambda value: value["usage_ledger"][0]["processed_response"][
                "choices"
            ][0]["message"].update(content="changed"),
            lambda value: value.update(answer="changed"),
            lambda value: value["answer_audit"]["parsed_outputs"][0].update(
                answer="changed"
            ),
        )
        for mutation in mutations:
            invalid = copy.deepcopy(episode)
            mutation(invalid)
            with self.subTest(mutation=mutation), self.assertRaises(
                GradeIntegrityError
            ):
                validate_episode(invalid, {"id": "q1"})

    def test_current_binding_rejects_self_consistent_fabricated_parse_data(self) -> None:
        output = "[[ ## answer ## ]]\nprovider answer"
        ledger = [self.make_current_call(0, output)]
        episode = self.make_current_episode(
            ledger,
            {
                "status": "parsed",
                "adapter": "ChatAdapter",
                "fallback_used": False,
                "provider_calls": [0],
                "provider_call": 0,
                "choice": 0,
                "parsed_outputs": [{"answer": "provider answer"}],
            },
            answer="provider answer",
        )
        fabricated = copy.deepcopy(episode)
        fabricated["answer"] = "invented answer"
        parsed_outputs = [{"answer": "invented answer"}]
        parsed_bytes = canonical_json_bytes(parsed_outputs)
        fabricated["answer_audit"].update({
            "parsed_outputs": parsed_outputs,
            "parsed_outputs_canonical_bytes": len(parsed_bytes),
            "parsed_outputs_sha256": sha256_json(parsed_outputs),
            "answer_sha256": sha256_text("invented answer"),
        })
        with self.assertRaises(GradeIntegrityError):
            validate_episode(fabricated, {"id": "q1"})

    def test_current_binding_rejects_fabricated_parse_failure_and_opposite_audit(self) -> None:
        ledger = [
            self.make_current_call(0, "not chat fields"),
            self.make_current_call(1, "still malformed"),
        ]
        failed = self.make_current_episode(
            ledger,
            {
                "status": "parse_failure",
                "adapter": "JSONAdapter",
                "fallback_used": True,
                "provider_calls": [0, 1],
                "provider_call": 1,
                "choice": 0,
                "lm_response": "still malformed",
                "parsed_result": None,
            },
            answer="",
            status="no_answer",
        )
        fabricated = copy.deepcopy(failed)
        invented = "unrelated invented response"
        fabricated["non_answer_audit"].update({
            "adapter_lm_response": invented,
            "adapter_lm_response_bytes": len(invented.encode("utf-8")),
            "adapter_lm_response_sha256": sha256_text(invented),
        })
        with self.assertRaises(GradeIntegrityError):
            validate_episode(fabricated, {"id": "q1"})

        contradictory = copy.deepcopy(failed)
        contradictory["answer_audit"] = copy.deepcopy(
            contradictory["non_answer_audit"]
        )
        with self.assertRaises(GradeIntegrityError):
            validate_episode(contradictory, {"id": "q1"})

        valid = self.make_current_episode(
            [self.make_current_call(0, "[[ ## answer ## ]]\nanswer")],
            {
                "status": "parsed",
                "adapter": "ChatAdapter",
                "fallback_used": False,
                "provider_calls": [0],
                "provider_call": 0,
                "choice": 0,
                "parsed_outputs": [{"answer": "answer"}],
            },
            answer="answer",
        )
        contradictory = copy.deepcopy(valid)
        contradictory["non_answer_audit"] = copy.deepcopy(
            contradictory["answer_audit"]
        )
        with self.assertRaises(GradeIntegrityError):
            validate_episode(contradictory, {"id": "q1"})

    def test_current_json_failure_replays_recursive_object_extraction(self) -> None:
        completion = 'prefix [1] {"other":"value"} suffix'
        failed = self.make_current_episode(
            [
                self.make_current_call(0, "not chat fields"),
                self.make_current_call(1, completion),
            ],
            {
                "status": "parse_failure",
                "adapter": "JSONAdapter",
                "fallback_used": True,
                "provider_calls": [0, 1],
                "provider_call": 1,
                "choice": 0,
                "lm_response": '{"other":"value"}',
                "parsed_result": {},
            },
            answer="",
            status="no_answer",
        )
        validate_episode(failed, {"id": "q1"})

    def test_current_fallback_cannot_bypass_an_earlier_parsed_output(self) -> None:
        retried = self.make_current_episode(
            [
                self.make_current_call(0, "not chat fields"),
                self.make_current_call(1, "{}"),
                self.make_current_call(2, '{"answer":"final"}'),
            ],
            {
                "status": "parsed",
                "adapter": "JSONAdapter",
                "fallback_used": True,
                "provider_calls": [0, 1, 2],
                "provider_call": 2,
                "choice": 0,
                "parsed_outputs": [{"answer": "final"}],
            },
            answer="final",
        )
        validate_episode(retried, {"id": "q1"})

        omitted_direct_call = self.make_current_episode(
            [
                self.make_current_call(0, "[[ ## answer ## ]]\nomitted"),
                self.make_current_call(1, "not chat fields"),
                self.make_current_call(2, '{"answer":"final"}'),
            ],
            {
                "status": "parsed",
                "adapter": "JSONAdapter",
                "fallback_used": True,
                "provider_calls": [1, 2],
                "provider_call": 2,
                "choice": 0,
                "parsed_outputs": [{"answer": "final"}],
            },
            answer="final",
        )
        with self.assertRaises(GradeIntegrityError):
            validate_episode(omitted_direct_call, {"id": "q1"})

        react_failure = self.make_current_episode(
            [
                self.make_current_call(0, "not chat fields"),
                self.make_current_call(1, "{}"),
            ],
            {
                "status": "parse_failure",
                "adapter": "JSONAdapter",
                "fallback_used": True,
                "provider_calls": [0, 1],
                "provider_call": 1,
                "choice": 0,
                "lm_response": "{}",
                "parsed_result": {},
            },
            answer="",
            status="no_answer",
            stage="react",
            budget="k5",
        )
        validate_episode(react_failure, {"id": "q1"})

        impossible_react_retry = self.make_current_episode(
            [
                self.make_current_call(0, "not chat fields"),
                self.make_current_call(
                    1,
                    '{"next_thought":"t","next_tool_name":"grep",'
                    '"next_tool_args":[]}',
                ),
                self.make_current_call(2, "{}"),
            ],
            {
                "status": "parse_failure",
                "adapter": "JSONAdapter",
                "fallback_used": True,
                "provider_calls": [0, 1, 2],
                "provider_call": 2,
                "choice": 0,
                "lm_response": "{}",
                "parsed_result": {},
            },
            answer="",
            status="no_answer",
            stage="react",
            budget="k5",
        )
        with self.assertRaises(GradeIntegrityError):
            validate_episode(impossible_react_retry, {"id": "q1"})

        bypassed_chat = self.make_current_episode(
            [
                self.make_current_call(0, "[[ ## answer ## ]]\nfirst"),
                self.make_current_call(1, '{"answer":"second"}'),
            ],
            {
                "status": "parsed",
                "adapter": "JSONAdapter",
                "fallback_used": True,
                "provider_calls": [0, 1],
                "provider_call": 1,
                "choice": 0,
                "parsed_outputs": [{"answer": "second"}],
            },
            answer="second",
        )
        with self.assertRaises(GradeIntegrityError):
            validate_episode(bypassed_chat, {"id": "q1"})

        bypassed_json = self.make_current_episode(
            [
                self.make_current_call(0, "not chat fields"),
                self.make_current_call(1, '{"answer":"intermediate"}'),
                self.make_current_call(2, '{"answer":"final"}'),
            ],
            {
                "status": "parsed",
                "adapter": "JSONAdapter",
                "fallback_used": True,
                "provider_calls": [0, 1, 2],
                "provider_call": 2,
                "choice": 0,
                "parsed_outputs": [{"answer": "final"}],
            },
            answer="final",
        )
        with self.assertRaises(GradeIntegrityError):
            validate_episode(bypassed_json, {"id": "q1"})

        impossible_range = self.make_current_episode(
            [
                self.make_current_call(0, "not chat fields"),
                self.make_current_call(1, "{}"),
                self.make_current_call(2, "{}"),
                self.make_current_call(3, '{"answer":"final"}'),
            ],
            {
                "status": "parsed",
                "adapter": "JSONAdapter",
                "fallback_used": True,
                "provider_calls": [0, 1, 2, 3],
                "provider_call": 3,
                "choice": 0,
                "parsed_outputs": [{"answer": "final"}],
            },
            answer="final",
        )
        with self.assertRaises(GradeIntegrityError):
            validate_episode(impossible_range, {"id": "q1"})

    def test_fixed_adapter_replay_matches_pinned_dspy_on_malformed_corpus(self) -> None:
        rt = SimpleNamespace(read_max_lines=READ_MAX_LINES, dispatch=lambda *args: "")
        react = dspy.ReAct(
            "question -> answer", tools=make_tools(rt), max_iters=1
        )
        signatures = {
            "direct": dspy.Signature("question -> answer"),
            "react": react.react.signature,
            "extract": react.extract.predict.signature,
        }
        fixed = [
            "",
            "plain text",
            "null",
            "[]",
            "{}",
            '{"answer":"ok"}',
            '{"answer":',
            'prefix {"answer":"ok"} suffix',
            '[{"answer":"ok"}]',
            '{"reasoning":"r","answer":"a"}',
            '{"next_thought":"t","next_tool_name":"grep",'
            '"next_tool_args":{"pattern":"x"}}',
            "[[ ## answer ## ]]\na",
            "[[ ## reasoning ## ]]\nr\n[[ ## answer ## ]]\na",
            "[[ ## next_thought ## ]]\nt\n[[ ## next_tool_name ## ]]\ngrep\n"
            "[[ ## next_tool_args ## ]]\n{\"pattern\":\"x\"}",
            'prefix [1] {"other":"x"} suffix',
            "[[ ## answer ## ]]\n",
        ]
        rng = random.Random(17)
        alphabet = '{}[],:" abcdef0123#_\n'
        corpus = fixed + [
            "".join(alphabet[rng.randrange(len(alphabet))]
                    for _ in range(rng.randrange(80)))
            for _ in range(1_000)
        ]

        def actual_outcome(adapter, signature, completion):
            try:
                return ("ok", adapter.parse(signature, completion))
            except AdapterParseError as error:
                return ("ape", error.lm_response, error.parsed_result)
            except Exception:
                return ("other",)

        def replay_outcome(replay, contract, completion):
            try:
                return ("ok", replay(completion, contract))
            except _DspyAdapterParseFailure as error:
                return ("ape", error.lm_response, error.parsed_result)
            except Exception:
                return ("other",)

        adapters = (
            (ChatAdapter(), _replay_dspy_chat_parse),
            (JSONAdapter(), _replay_dspy_json_parse),
        )
        for stage, signature in signatures.items():
            contract = _dspy_output_field_contract(stage)
            for adapter, replay in adapters:
                for completion in corpus:
                    with self.subTest(
                        stage=stage,
                        adapter=type(adapter).__name__,
                        completion=completion,
                    ):
                        self.assertEqual(
                            replay_outcome(replay, contract, completion),
                            actual_outcome(adapter, signature, completion),
                        )

    def test_trajectory_recorder_keeps_an_independent_failure_snapshot(self) -> None:
        module = TrajectoryRecordingReAct("question -> answer", tools=[])
        trajectory = {"thought_0": "look", "tool_args_0": {"path": "x"}}
        module._format_trajectory(trajectory)
        trajectory["tool_args_0"]["path"] = "changed"
        self.assertEqual(module.last_trajectory["tool_args_0"]["path"], "x")

    def test_dspy_tool_contract_matches_runtime_and_not_native_schema(self) -> None:
        rt = SimpleNamespace(read_max_lines=READ_MAX_LINES, dispatch=lambda *args: "")
        observed = runtime_dspy_tool_contract(make_tools(rt))
        self.assertEqual(observed, DSPY_REPOSITORY_TOOL_CONTRACT)
        self.assertEqual(READ_MAX_LINES, 200)
        self.assertEqual(NATIVE_READ_MAX_LINES, 500)

    def test_dspy_usage_is_never_invented_from_missing_or_malformed_data(self) -> None:
        for usage in (None, {}, {"prompt_tokens": 1, "completion_tokens": 2}):
            with self.subTest(usage=usage), self.assertRaisesRegex(
                ValueError, "usage"
            ):
                _dspy_usage_record({"usage": usage}, 0)

    def test_completed_study_accepts_only_compatible_launch_nuisance_drift(self) -> None:
        baseline = {
            "model_revision": "revision",
            "slurm_job_id": "1",
            "server_launch_id": "a" * 64,
            "vllm_api_key_sha256": "b" * 64,
            "cuda_visible_devices": "0,1",
            "runner_allocation": {"hostname": "first"},
        }
        retry = {
            **baseline,
            "slurm_job_id": "2",
            "server_launch_id": "c" * 64,
            "vllm_api_key_sha256": "d" * 64,
            "cuda_visible_devices": "6,7",
            "runner_allocation": {"hostname": "second"},
        }
        stored = {"study_id": "study-r1", "environment": baseline, "seed": 7}
        current = {"study_id": "study-r1", "environment": retry, "seed": 7}
        with tempfile.TemporaryDirectory() as directory:
            intent = Path(directory) / "intent.json"
            write_immutable_json(intent, stored)
            self.assertEqual(
                _stored_completed_study_config(intent, current), stored
            )
            with self.assertRaisesRegex(SystemExit, "protocol or source"):
                _stored_completed_study_config(intent, {**current, "seed": 8})
            substantive = {**retry, "model_revision": "other"}
            with self.assertRaisesRegex(SystemExit, "substantive drift"):
                _stored_completed_study_config(
                    intent, {**current, "environment": substantive}
                )

    def make_study(self, root: Path) -> tuple[dict[str, object], Path]:
        config: dict[str, object] = {
            "claim_ready": True,
            "study_id": "study-r1",
            "task": "dspy",
            "corpus": {"commit": "corpus-commit"},
            "study_question_sha256": "q" * 64,
            "model": "model",
            "model_revision": "revision",
            "expected_response_model": "served-model",
            "episode_seed": 17,
            "forced_iterations": 50,
            "repository_tool_scope": "full-pinned-corpus",
        }
        intent = root / "intent.json"
        write_immutable_json(intent, config)
        episode = {
            "status": "ok",
            "answer": "exact note\n",
            "task": "dspy",
            "qid": "cheatsheet",
            "budget": "s50",
            "rollout": 0,
            "study_intent_sha256": sha256_json(config),
            "question_sha256": config["study_question_sha256"],
            "model": config["model"],
            "model_revision": config["model_revision"],
            "harness": "dspy.ReAct",
            "dspy_request_audit_schema": DSPY_REQUEST_AUDIT_LEGACY_SCHEMA_VERSION,
            "seed": config["episode_seed"],
            "started": "start",
            "finished": "finish",
            "turns": [
                {
                    "reasoning": f"reasoning {index}",
                    "tool_calls": [{"name": "read_file", "arguments": "{}"}],
                    "observations": ["source"],
                }
                for index in range(50)
            ],
            "n_react_iters": 50,
            "n_tool_iters": 50,
            "finish_catches": 0,
            "forced_budget_complete": True,
            "n_lm_calls": 51,
            "usage_ledger": [
                {
                    "call": index,
                    "response_model": "served-model",
                    "response_id": f"response-{index}",
                    "system_fingerprint": "fingerprint",
                    "request_messages_sha256": sha256_json([f"request-{index}"]),
                    "outputs_sha256": sha256_json([f"output-{index}"]),
                    "provider_usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                }
                for index in range(51)
            ],
            "prompt_tokens": 51,
            "completion_tokens": 51,
            "total_tokens": 102,
            "gen_tokens": 51,
        }
        episode_path = root / "episode.json"
        write_immutable_json(episode_path, episode)
        note_hash = sha256_text(episode["answer"])
        note_name = f"note-{note_hash}.md"
        (root / note_name).write_text(episode["answer"], encoding="utf-8")
        inventory = _artifact_inventory(root, ("intent.json", "episode.json"))
        manifest = {
            "manifest_schema": 1,
            "manifest_type": "forced-50-cheatsheet",
            "claim_ready": True,
            "study_id": "study-r1",
            "task": "dspy",
            "corpus_commit": "corpus-commit",
            "config": config,
            "note_sha256": note_hash,
            "note_path": note_name,
            "episode_sha256": sha256_json(episode),
            "intent_sha256": sha256_json(config),
            "study_generated_tokens": 51,
            "study_prompt_tokens": 51,
            "study_total_tokens": 102,
            "construction_artifacts": inventory,
            "construction_artifacts_sha256": sha256_json(inventory),
        }
        manifest_path = root / "manifest.json"
        write_immutable_json(manifest_path, manifest)
        return config, manifest_path

    def test_completed_study_revalidates_the_full_dependency_chain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, manifest = self.make_study(root)
            _validate_completed_study(manifest, root / "intent.json", root, config)

    def test_forced_adapter_parse_non_answer_preserves_partial_budget_truth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_study(root)
            episode = json.loads((root / "episode.json").read_text())
        episode.update({
            "qid": "q1",
            "budget": "k20f",
            "status": "no_answer",
            "answer": "",
            "turns": episode["turns"][:7],
            "n_react_iters": 7,
            "n_tool_iters": 7,
            "finish_catches": 0,
            "n_lm_calls": 9,
            "usage_ledger": episode["usage_ledger"][:9],
            "prompt_tokens": 9,
            "completion_tokens": 9,
            "total_tokens": 18,
            "gen_tokens": 9,
            "dspy_request_audit_schema": DSPY_REQUEST_AUDIT_LEGACY_SCHEMA_VERSION,
            "forced_budget_complete": False,
        })
        episode["non_answer_audit"] = {
            "schema_version": DSPY_REQUEST_AUDIT_LEGACY_SCHEMA_VERSION,
            "kind": "adapter_parse_failure",
            "stage": "react",
            "adapter": "JSONAdapter",
            "provider_call": 8,
            "outputs_sha256": episode["usage_ledger"][8]["outputs_sha256"],
        }
        validate_episode(episode, {"id": "q1"})
        _validate_final_episode(
            episode,
            {
                "task": "dspy", "qid": "q1", "budget": "k20f",
                "rollout": 0, "seed": episode["seed"],
            },
            expected_model="model",
            expected_model_revision="revision",
            expected_harness="dspy.ReAct",
            expected_response_model="served-model",
        )

        for mutation in (
            lambda value: value.update(forced_budget_complete=True),
            lambda value: value["non_answer_audit"].update(stage="extract"),
        ):
            invalid = copy.deepcopy(episode)
            mutation(invalid)
            with self.assertRaisesRegex(
                GradeIntegrityError, "forced k20|forced-budget completion"
            ):
                validate_episode(invalid, {"id": "q1"})

    def test_completed_study_rejects_note_or_episode_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, manifest = self.make_study(root)
            note = next(root.glob("note-*.md"))
            note.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "dependency validation"):
                _validate_completed_study(manifest, root / "intent.json", root, config)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, manifest = self.make_study(root)
            (root / "episode.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "dependency validation"):
                _validate_completed_study(manifest, root / "intent.json", root, config)


if __name__ == "__main__":
    unittest.main()
