from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from studybench.integrity import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_json,
    sha256_text,
)
from studybench.provenance import _load_note
from studybench.report import validated_note_provenance
from studybench.study_protocol import (
    _DspyAdapterParseFailure,
    _dspy_output_field_contract,
    _replay_dspy_json_parse,
    semantic_dev_exam_terminal,
    semantic_dev_reference_terminal,
    semantic_training_item_terminal,
    DSPY_ADAPTER_NAME,
    DSPY_ADAPTER_POLICY,
    DSPY_SEMANTIC_CHAPTER_SYLLABUS,
    DSPY_REPOSITORY_TOOL_CONTRACT,
    DSPY_REQUEST_AUDIT_SCHEMA_VERSION,
    FORCED50_CONFIG_SCHEMA_VERSION,
    FORCED50_ITERATIONS,
    REACT_SAMPLING,
    SEMANTIC_NOTE_MANIFEST_KEYS,
    SEMANTIC_READINESS_KEYS,
    SEMANTIC_TREATMENT_READINESS_KEYS,
    SEMANTIC_SELFQUIZ_ADAPTER,
    SEMANTIC_SELFQUIZ_ADAPTER_POLICY,
    SEMANTIC_SELFQUIZ_CITATION_POLICY,
    SEMANTIC_SELFQUIZ_METHOD,
    SEMANTIC_SELFQUIZ_NOTE_MANIFEST_TYPE,
    SEMANTIC_SELFQUIZ_TASK_MANIFEST_TYPE,
    SEMANTIC_SELFQUIZ_UNRESOLVED_POLICY,
    STATIC_GRAPH_METHOD,
    STATIC_GRAPH_NOTE_MANIFEST_TYPE,
    STATIC_GRAPH_TASK_MANIFEST_TYPE,
    StudyProtocolError,
    derive_protocol_summary,
    forced50_study_question,
    openbook_attempt_protocol,
    semantic_attempt_protocol,
    validate_construction_protocol,
    validate_forced50_config,
    validate_forced50_episode,
    validate_study_note_archive,
)
from studybench.tools import DSPY_READ_MAX_LINES


def provenance_fields() -> dict:
    readiness = {
        "corpus_pinned_clean": True,
        "source_pinned_clean": True,
        "environment_complete": True,
        "model_revision_pinned": True,
        "server_count_matches_environment": True,
    }
    return {
        "corpus": {
            "name": "fake",
            "commit": "c" * 40,
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
        "environment": {"server_count": "1", "runtime": "pinned"},
        "environment_contract": {"schema_version": 1, "sha256": "e" * 64},
        "server_transport": {
            "scope": "loopback",
            "protocol": "openai-compatible-http",
            "server_count": 1,
            "assignment": (
                "stable_seed(master_seed, owner_id, server) modulo server_count"
            ),
        },
        "provenance_readiness": readiness,
        "automated_provenance_ready": True,
    }


def semantic_task(*, attempt_access: str = "react-corpus") -> dict:
    provenance = provenance_fields()
    provenance["server_transport"] = {
        **provenance["server_transport"],
        "assignment": (
            "stable_seed(master_seed, stochastic_namespace, server) "
            "modulo server_count"
        ),
    }
    return {
        "schema_version": 6,
        "manifest_type": SEMANTIC_SELFQUIZ_TASK_MANIFEST_TYPE,
        "method": SEMANTIC_SELFQUIZ_METHOD,
        "study_id": "study-a",
        "task": "dspy",
        "master_seed": 7,
        "model": "model",
        "model_revision": "revision",
        "sampling": deepcopy(REACT_SAMPLING),
        "corpus_commit": provenance["corpus"]["commit"],
        **provenance,
        "human_audit_protocol": None,
        "config": {
            "chapter_syllabus": list(DSPY_SEMANTIC_CHAPTER_SYLLABUS),
            "chapters_per_round": 4,
            "final_round": 4,
            "questions_per_chapter": 5,
            "attempt_access": attempt_access,
            "adapter": SEMANTIC_SELFQUIZ_ADAPTER,
            "adapter_policy": SEMANTIC_SELFQUIZ_ADAPTER_POLICY,
            "citation_policy": SEMANTIC_SELFQUIZ_CITATION_POLICY,
            "unresolved_policy": SEMANTIC_SELFQUIZ_UNRESOLVED_POLICY,
            "smoke": False,
            "quiz_max_iters": 15,
            "attempt_protocol": semantic_attempt_protocol(attempt_access),
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


def graph_task() -> dict:
    provenance = provenance_fields()
    provenance["corpus"] = {**provenance["corpus"], "name": "dspy"}
    bank_sha256 = "b" * 64
    resolver = {"question_bank_sha256": bank_sha256}
    return {
        "schema_version": 1,
        "manifest_type": STATIC_GRAPH_TASK_MANIFEST_TYPE,
        "study_id": "graph-a",
        "task": "dspy",
        "round": 1,
        "master_seed": 7,
        "source_root": "dspy",
        "model": "model",
        "model_revision": "revision",
        "sampling": deepcopy(REACT_SAMPLING),
        "corpus_commit": provenance["corpus"]["commit"],
        **provenance,
        "resolver_contract": resolver,
        "resolver_contract_sha256": sha256_json(resolver),
        "question_bank_sha256": bank_sha256,
        "question_bank_artifact_sha256": "d" * 64,
        "config": {
            "method": STATIC_GRAPH_METHOD,
            "smoke": False,
            "concurrency": 8,
            "attempt_protocol": openbook_attempt_protocol(),
            "train_input_note_sha256": sha256_text(""),
            "train_question_count": 16,
            "dev_question_count": 4,
            "dev_holdout_targets": [
                "dspy.adapters.utils.parse_value",
                "dspy.signatures.signature._parse_signature",
                "dspy.teleprompt.bootstrap_finetune.all_predictors_have_lms",
                "dspy.teleprompt.utils.eval_candidate_program",
            ],
            "provider_retries": 0,
            "read_max_lines": DSPY_READ_MAX_LINES,
        },
    }


def note_for_task(task: dict, *, graph: bool = False) -> tuple[dict, dict[str, bytes]]:
    task_bytes = canonical_json_bytes(task)
    inventory = {
        "manifest.json": {
            "sha256": sha256_bytes(task_bytes),
            "bytes": len(task_bytes),
        }
    }
    note = {
        "schema_version": 1,
        "manifest_type": (
            STATIC_GRAPH_NOTE_MANIFEST_TYPE
            if graph
            else SEMANTIC_SELFQUIZ_NOTE_MANIFEST_TYPE
        ),
        "method": STATIC_GRAPH_METHOD if graph else SEMANTIC_SELFQUIZ_METHOD,
        "protocol_summary": derive_protocol_summary(task_bytes),
        "study_id": task["study_id"],
        "task": task["task"],
        "round": 1,
        "corpus_commit": task["corpus_commit"],
        "construction_artifacts": inventory,
        "construction_artifacts_sha256": sha256_json(inventory),
    }
    return note, {"manifest.json": task_bytes}


def forced_config() -> dict:
    provenance = provenance_fields()
    question = forced50_study_question("Fake")
    master_seed = 7
    from studybench.integrity import stable_seed

    return {
        "schema_version": FORCED50_CONFIG_SCHEMA_VERSION,
        "study_id": "forced-a",
        "task": "fake",
        "method": "forced-50-cheatsheet",
        "model": "model",
        "model_revision": "revision",
        "expected_response_model": "served-model",
        "sampling": deepcopy(REACT_SAMPLING),
        "adapter": DSPY_ADAPTER_NAME,
        "adapter_fallback_policy": DSPY_ADAPTER_POLICY,
        "dspy_request_audit_schema": DSPY_REQUEST_AUDIT_SCHEMA_VERSION,
        "master_seed": master_seed,
        "episode_seed": stable_seed(master_seed, "cheatsheet", "forced-a", "fake"),
        "study_prompt_sha256": sha256_text(question["question"]),
        "study_question_sha256": sha256_json(question),
        "tool_contract": DSPY_REPOSITORY_TOOL_CONTRACT,
        "tool_schema_sha256": sha256_json(DSPY_REPOSITORY_TOOL_CONTRACT),
        "read_max_lines": DSPY_READ_MAX_LINES,
        "forced_iterations": FORCED50_ITERATIONS,
        "repository_tool_scope": "full-pinned-corpus",
        "corpus": provenance["corpus"],
        "source": provenance["source"],
        "environment": provenance["environment"],
        "claim_ready": True,
        "server_transport": {
            "scope": "loopback",
            "protocol": "openai-compatible-http",
            "available_server_count": 1,
            "selected_server_index": 0,
        },
    }


def forced_episode(config: dict, note: str = "study note") -> dict:
    usage = {
        "prompt_tokens": 90,
        "completion_tokens": 10,
        "total_tokens": 100,
    }
    output = f"[[ ## reasoning ## ]]\nstudied\n\n[[ ## answer ## ]]\n{note}"
    processed_response = {
        "id": "response-1",
        "model": config["expected_response_model"],
        "system_fingerprint": "fingerprint",
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "content": output,
                "reasoning_content": None,
                "tool_calls": None,
            },
        }],
        "usage": usage,
    }
    outputs = [output]
    parsed_outputs = [{"reasoning": "studied", "answer": note}]
    episode = {
        "task": config["task"],
        "qid": "cheatsheet",
        "budget": "s50",
        "rollout": 0,
        "model": config["model"],
        "model_revision": config["model_revision"],
        "harness": "dspy.ReAct",
        "dspy_request_audit_schema": DSPY_REQUEST_AUDIT_SCHEMA_VERSION,
        "seed": config["episode_seed"],
        "study_intent_sha256": sha256_json(config),
        "question_sha256": config["study_question_sha256"],
        "status": "ok",
        "started": "2026-01-01T00:00:00+00:00",
        "finished": "2026-01-01T00:01:00+00:00",
        "answer": note,
        "n_react_iters": 50,
        "n_tool_iters": 50,
        "finish_catches": 0,
        "forced_budget_complete": True,
        "turns": [
            {
                "reasoning": f"step {index}",
                "tool_calls": [{"name": "grep", "arguments": "{}"}],
                "observations": ["evidence"],
            }
            for index in range(50)
        ],
        "prompt_tokens": 90,
        "completion_tokens": 10,
        "total_tokens": 100,
        "gen_tokens": 10,
        "n_lm_calls": 1,
        "usage_ledger": [{
            "call": 0,
            "response_id": "response-1",
            "response_model": config["expected_response_model"],
            "system_fingerprint": "fingerprint",
            "request_messages_sha256": "1" * 64,
            "outputs_sha256": sha256_json(outputs),
            "processed_response": processed_response,
            "processed_response_canonical_bytes": len(
                canonical_json_bytes(processed_response)
            ),
            "processed_response_sha256": sha256_json(processed_response),
            "outputs": outputs,
            "outputs_canonical_bytes": len(canonical_json_bytes(outputs)),
            "finish_reasons": ["stop"],
            "provider_usage": usage,
            "prompt_tokens": 90,
            "completion_tokens": 10,
            "total_tokens": 100,
        }],
    }
    episode["answer_audit"] = {
        "schema_version": DSPY_REQUEST_AUDIT_SCHEMA_VERSION,
        "kind": "parsed_answer",
        "stage": "extract",
        "adapter": "ChatAdapter",
        "fallback_used": False,
        "provider_calls": [0],
        "provider_call": 0,
        "choice": 0,
        "outputs_sha256": sha256_json(outputs),
        "selected_output_canonical_bytes": len(canonical_json_bytes(output)),
        "selected_output_sha256": sha256_json(output),
        "parsed_outputs": parsed_outputs,
        "parsed_outputs_canonical_bytes": len(canonical_json_bytes(parsed_outputs)),
        "parsed_outputs_sha256": sha256_json(parsed_outputs),
        "answer_sha256": sha256_text(note),
    }
    return episode


class StudyProtocolTests(unittest.TestCase):
    def test_main_json_replay_preserves_recursive_object_error_input(self) -> None:
        completion = 'prefix [1] {"other":"value"} suffix'
        with self.assertRaises(_DspyAdapterParseFailure) as caught:
            _replay_dspy_json_parse(
                completion,
                _dspy_output_field_contract("direct"),
            )
        self.assertEqual(caught.exception.lm_response, '{"other":"value"}')
        self.assertEqual(caught.exception.parsed_result, {})

    def test_manifest_only_semantic_archive_cannot_assert_readiness(self) -> None:
        task = semantic_task()
        note, dependencies = note_for_task(task)
        note_bytes = b"fabricated note\n"
        note_hash = sha256_bytes(note_bytes)
        note.update({
            "schema_version": 6,
            "claim_ready": False,
            "publication_claim_ready": False,
            "confirmatory_claim_ready": False,
            "automated_claim_ready": True,
            "automated_readiness": {
                key: True for key in SEMANTIC_READINESS_KEYS
            },
            "automated_treatment_ready": True,
            "automated_treatment_readiness": {
                key: True for key in SEMANTIC_TREATMENT_READINESS_KEYS
            },
            "human_audit": {
                "required": True,
                "status": "not_performed",
                "protocol": "pre-registered blinded verdict and evidence audit",
            },
            "note_sha256": note_hash,
            "note_path": f"by-sha256/{note_hash}.md",
            "input_note_sha256": sha256_text(""),
            "entry_ids": [],
            "entries": [],
            "usage": {},
            "round_usage": {},
            "cumulative_usage": {},
            "round_usage_by_phase": {},
            "cumulative_usage_by_phase": {},
            "round_construction_usage": {},
            "cumulative_construction_usage": {},
            "round_construction_usage_by_phase": {},
            "cumulative_construction_usage_by_phase": {},
            "note_chars": len(note_bytes.decode("utf-8")),
        })
        self.assertEqual(set(note), SEMANTIC_NOTE_MANIFEST_KEYS)
        for label, mutate in {
            "unknown treatment gate": lambda value: value[
                "automated_treatment_readiness"
            ].update(extra=True),
            "drifted treatment flag": lambda value: value.update(
                automated_treatment_ready=False
            ),
        }.items():
            with self.subTest(label=label):
                invalid = deepcopy(note)
                mutate(invalid)
                with self.assertRaisesRegex(StudyProtocolError, "readiness"):
                    validate_study_note_archive(invalid, dependencies, note_bytes)
        with self.assertRaisesRegex(StudyProtocolError, "aliases"):
            validate_study_note_archive(note, dependencies, note_bytes)

    def test_semantic_v4_terminal_states_do_not_weaken_claim_states(self) -> None:
        valid_derivation = {
            "status": "ok",
            "citation_pass": {"status": "ok"},
            "reference_support": {"status": "ok"},
        }
        abstaining_reference = {
            "status": "unresolved",
            "derivations": [
                valid_derivation,
                {
                    "status": "invalid",
                    "citation_pass": {"status": "invalid"},
                    "reference_support": None,
                },
            ],
            "reference_consensus": [],
        }
        abstaining_train = {
            **abstaining_reference,
            "status": "ok",
            "verdict": "unresolved",
            "attempt": {"status": "ok"},
        }
        self.assertTrue(semantic_training_item_terminal(abstaining_train))
        attempt_error = deepcopy(abstaining_train)
        attempt_error["attempt"]["status"] = "error"
        self.assertFalse(semantic_training_item_terminal(attempt_error))
        adjudication_error = deepcopy(abstaining_train)
        adjudication_error["adjudications"] = [{"status": "error"}]
        self.assertFalse(semantic_training_item_terminal(adjudication_error))
        distill_error = deepcopy(abstaining_train)
        distill_error["entry_bounced"] = {
            "reasons": ["distill_error: timeout"]
        }
        self.assertFalse(semantic_training_item_terminal(distill_error))
        citation_error = deepcopy(abstaining_reference)
        citation_error["derivations"][1]["citation_pass"]["status"] = "error"
        self.assertFalse(semantic_dev_reference_terminal(citation_error))
        self.assertTrue(semantic_dev_reference_terminal(abstaining_reference))
        self.assertTrue(semantic_dev_exam_terminal({
            "status": "reference_unresolved",
            "verdicts": {"with_note": "unresolved", "bare": "unresolved"},
        }))
        self.assertTrue(semantic_dev_exam_terminal({
            "status": "ok",
            "verdicts": {"with_note": "correct", "bare": "unresolved"},
            "attempts": {
                "with_note": {"status": "ok"},
                "bare": {"status": "ok"},
            },
            "adjudications": {
                "with_note": [{"status": "ok"}],
                "bare": [{"status": "ok"}],
            },
        }))
        self.assertFalse(semantic_dev_exam_terminal({
            "status": "ok",
            "verdicts": {"with_note": "correct", "bare": "unresolved"},
            "attempts": {
                "with_note": {"status": "ok"},
                "bare": {"status": "ok"},
            },
            "adjudications": {
                "with_note": [{"status": "error"}],
                "bare": [{"status": "ok"}],
            },
        }))
        self.assertFalse(semantic_dev_exam_terminal({
            "status": "reference_unresolved",
            "verdicts": {"with_note": "correct", "bare": "unresolved"},
        }))

    def test_semantic_v4_readiness_schemas_are_exact(self) -> None:
        self.assertEqual(SEMANTIC_READINESS_KEYS, frozenset({
            "non_smoke",
            "provenance_complete",
            "launch_environments_bound",
            "prior_rounds_automated_ready",
            "question_freshness",
            "quiz_episodes_complete",
            "training_complete",
            "dev_references_complete",
            "dev_exam_complete",
            "lineage_clean",
            "evidence_safe",
            "usage_complete",
            "adapter_audit_complete",
            "response_model_homogeneous",
            "response_model_expected",
        }))
        self.assertEqual(SEMANTIC_TREATMENT_READINESS_KEYS, frozenset({
            "non_smoke",
            "provenance_complete",
            "launch_environments_bound",
            "prior_rounds_treatment_ready",
            "question_freshness",
            "quiz_episodes_complete",
            "training_terminal_complete",
            "dev_references_terminal_complete",
            "dev_exam_terminal_complete",
            "lineage_clean",
            "evidence_safe",
            "usage_complete",
            "adapter_audit_complete",
            "response_model_homogeneous",
            "response_model_expected",
        }))

    def test_experiment_freezes_the_executable_openbook_attempt_hash(self) -> None:
        digest = sha256_json(openbook_attempt_protocol())
        document = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "011-full-dspy-selfquiz-ablations.md"
        ).read_text(encoding="utf-8")
        self.assertIn(
            f"| corpus-ReAct ATTEMPT protocol | `{digest}` |",
            document,
        )

    def test_internally_valid_old_source_is_rejected(self) -> None:
        task = semantic_task()
        note, dependencies = note_for_task(task)
        expected_source = {**task["source"], "git_commit": "f" * 40}
        with self.assertRaisesRegex(StudyProtocolError, "source does not match"):
            validate_construction_protocol(
                note,
                dependencies,
                expected_source=expected_source,
            )

    def test_semantic_evaluation_requires_the_final_round(self) -> None:
        task = semantic_task()
        note, dependencies = note_for_task(task)
        with self.assertRaisesRegex(StudyProtocolError, "final construction round"):
            validate_construction_protocol(
                note,
                dependencies,
                require_final_semantic=True,
            )
        note["round"] = 4
        self.assertEqual(
            validate_construction_protocol(
                note,
                dependencies,
                require_final_semantic=True,
            ),
            note["protocol_summary"],
        )

    def test_method_schema_and_attempt_drift_fail_closed(self) -> None:
        task = semantic_task()
        note, _ = note_for_task(task)
        self.assertEqual(SEMANTIC_SELFQUIZ_METHOD, "semantic-selfquiz-v4")
        self.assertEqual(
            SEMANTIC_SELFQUIZ_CITATION_POLICY,
            "uniform-answer-frozen-exact-line-pass-v1",
        )
        self.assertEqual(
            SEMANTIC_SELFQUIZ_UNRESOLVED_POLICY,
            "abstain-no-correction-continue-v1",
        )
        method_drift = deepcopy(task)
        method_drift["method"] = "semantic-selfquiz-v3"
        with self.assertRaises(StudyProtocolError):
            derive_protocol_summary(method_drift)

        version_drift = deepcopy(task)
        version_drift["schema_version"] = 5
        with self.assertRaisesRegex(StudyProtocolError, "method contract"):
            derive_protocol_summary(version_drift)

        schema_drift = semantic_task()
        schema_drift["config"]["focus_chapter"] = "dspy/adapters"
        with self.assertRaisesRegex(StudyProtocolError, "unknown schema"):
            derive_protocol_summary(schema_drift)

        attempt_drift = deepcopy(task)
        attempt_drift["config"]["attempt_protocol"]["max_iters"] = 4
        with self.assertRaisesRegex(StudyProtocolError, "method contract"):
            derive_protocol_summary(attempt_drift)

        for field in ("citation_policy", "unresolved_policy"):
            with self.subTest(field=field):
                policy_drift = deepcopy(task)
                policy_drift["config"][field] += "-drift"
                with self.assertRaisesRegex(
                    StudyProtocolError, "method contract"
                ):
                    derive_protocol_summary(policy_drift)

    def test_semantic_and_graph_have_distinct_report_identity(self) -> None:
        semantic_note, _ = note_for_task(semantic_task())
        graph_note, _ = note_for_task(graph_task(), graph=True)
        semantic = validated_note_provenance({
            "note_manifest": semantic_note,
            "note_protocol_summary": semantic_note["protocol_summary"],
            "note_construction_manifest_sha256": "1" * 64,
        })
        graph = validated_note_provenance({
            "note_manifest": graph_note,
            "note_protocol_summary": graph_note["protocol_summary"],
            "note_construction_manifest_sha256": "2" * 64,
        })
        self.assertEqual(semantic["question_mode"], "semantic")
        self.assertEqual(graph["question_mode"], "static-call-neighborhood")
        self.assertEqual(graph["attempt_access"], "react-corpus")
        self.assertNotEqual(semantic["method"], graph["method"])
        self.assertNotEqual(
            semantic["task_manifest_sha256"], graph["task_manifest_sha256"]
        )

    def test_load_note_rejects_source_drift_before_snapshotting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            study_root = root / "study"
            notes = study_root / "notes"
            notes.mkdir(parents=True)
            task = semantic_task()
            task_bytes = canonical_json_bytes(task)
            (study_root / "manifest.json").write_bytes(task_bytes)
            note_text = "bound note\n"
            note_hash = sha256_text(note_text)
            note_path = notes / "by-sha256" / f"{note_hash}.md"
            note_path.parent.mkdir()
            note_path.write_text(note_text)
            note_manifest, _ = note_for_task(task)
            note_manifest.update({
                "claim_ready": False,
                "publication_claim_ready": False,
                "confirmatory_claim_ready": False,
                "automated_claim_ready": True,
                "automated_readiness": {"complete": True},
                "note_sha256": note_hash,
                "note_path": f"by-sha256/{note_hash}.md",
            })
            note_manifest_path = notes / "note-r1.manifest.json"
            note_manifest_path.write_bytes(canonical_json_bytes(note_manifest))
            run_root = root / "run"
            with self.assertRaisesRegex(ValueError, "source"):
                _load_note(
                    run_root,
                    note_path,
                    note_manifest_path,
                    require_manifest=True,
                    require_claim_ready=False,
                    expected_task="dspy",
                    expected_model="model",
                    expected_model_revision="revision",
                    expected_sampling=REACT_SAMPLING,
                    expected_corpus_commit=task["corpus_commit"],
                    expected_corpus=task["corpus"],
                    expected_source={**task["source"], "git_commit": "f" * 40},
                    expected_environment=task["environment"],
                    expected_corpus_display="Fake",
                )
            self.assertFalse(run_root.exists())

    def test_forced_episode_preflight_rejects_turn_and_token_drift(self) -> None:
        config = forced_config()
        note = "study note"
        episode = forced_episode(config, note)
        validate_forced50_episode(
            canonical_json_bytes(episode),
            config=config,
            expected_note_sha256=sha256_text(note),
        )
        mutations = {
            "empty turns": lambda value: value.update(turns=[]),
            "wrong generated tokens": lambda value: value.update(gen_tokens=9),
            "wrong budget": lambda value: value.update(budget="k20f"),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                invalid = deepcopy(episode)
                mutate(invalid)
                with self.assertRaises(StudyProtocolError):
                    validate_forced50_episode(
                        canonical_json_bytes(invalid),
                        config=config,
                        expected_note_sha256=sha256_text(note),
                    )
        with self.assertRaisesRegex(StudyProtocolError, "canonical"):
            validate_forced50_episode(
                canonical_json_bytes(episode) + b"\n",
                config=config,
                expected_note_sha256=sha256_text(note),
            )

    def test_forced50_is_full_corpus_and_rejects_focus_extension(self) -> None:
        config = forced_config()
        protocol = validate_forced50_config(config, corpus_display="Fake")
        provenance = validated_note_provenance({
            "note_manifest": {
                "study_id": "forced-a",
                "manifest_type": "forced-50-cheatsheet",
            },
            "forced50_protocol": protocol,
            "note_construction_manifest_sha256": "1" * 64,
        })
        self.assertIsNone(provenance["focus_chapter"])
        self.assertEqual(
            provenance["protocol_config_sha256"], sha256_json(config)
        )
        invalid = deepcopy(config)
        invalid["focus_chapter"] = "dspy/adapters"
        with self.assertRaisesRegex(StudyProtocolError, "off protocol"):
            validate_forced50_config(invalid, corpus_display="Fake")


if __name__ == "__main__":
    unittest.main()
