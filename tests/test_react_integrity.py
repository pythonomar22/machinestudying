from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from studybench.integrity import sha256_json, sha256_text, write_immutable_json
from studybench.react import (
    READ_MAX_LINES,
    _artifact_inventory,
    _dspy_usage_record,
    _stored_completed_study_config,
    _validate_completed_study,
    make_tools,
    runtime_dspy_tool_contract,
)
from studybench.study_protocol import DSPY_REPOSITORY_TOOL_CONTRACT
from studybench.tools import READ_MAX_LINES as NATIVE_READ_MAX_LINES


class ReactStudyIntegrityTests(unittest.TestCase):
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
