from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import dspy
from dspy.adapters.base import Adapter
from dspy.utils.exceptions import AdapterParseError

from studybench.integrity import sha256_json, sha256_text, write_immutable_json
from studybench.grade import GradeIntegrityError, validate_episode
from studybench.react import (
    READ_MAX_LINES,
    ParseOnlyFallbackChatAdapter,
    TrajectoryRecordingReAct,
    _artifact_inventory,
    _dspy_usage_record,
    _stored_completed_study_config,
    _validate_completed_study,
    make_tools,
    runtime_dspy_tool_contract,
)
from studybench.rollout import _validate_final_episode
from studybench.study_protocol import (
    DSPY_REPOSITORY_TOOL_CONTRACT,
    DSPY_REQUEST_AUDIT_SCHEMA_VERSION,
)
from studybench.tools import READ_MAX_LINES as NATIVE_READ_MAX_LINES


class ReactStudyIntegrityTests(unittest.TestCase):
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
        json_adapter.return_value = [{"answer": "repaired"}]
        with patch.object(Adapter, "__call__", side_effect=completed_parse_failure), \
                patch("studybench.react.JSONAdapter", return_value=json_adapter):
            result = ParseOnlyFallbackChatAdapter()(
                lm, {}, signature, [], {"question": "q"}
            )
        self.assertEqual(result, [{"answer": "repaired"}])
        json_adapter.assert_called_once()

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
            "dspy_request_audit_schema": DSPY_REQUEST_AUDIT_SCHEMA_VERSION,
            "forced_budget_complete": False,
        })
        episode["non_answer_audit"] = {
            "schema_version": DSPY_REQUEST_AUDIT_SCHEMA_VERSION,
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
