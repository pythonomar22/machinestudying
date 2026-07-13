from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from studybench.integrity import sha256_json, sha256_text, write_immutable_json
from studybench.react import (
    _artifact_inventory,
    _dspy_usage_record,
    _validate_completed_study,
)


class ReactStudyIntegrityTests(unittest.TestCase):
    def test_dspy_usage_is_never_invented_from_missing_or_malformed_data(self) -> None:
        for usage in (None, {}, {"prompt_tokens": 1, "completion_tokens": 2}):
            with self.subTest(usage=usage), self.assertRaisesRegex(
                ValueError, "usage"
            ):
                _dspy_usage_record({"usage": usage}, 0)

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
