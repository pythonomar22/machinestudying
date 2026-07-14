from contextlib import ExitStack
import json
from pathlib import Path
import platform
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from studybench.graph_study import (
    ATTEMPT_ACCESS,
    DEV_QUESTION_COUNT,
    SCHEMA_VERSION,
    SMOKE_TARGET,
    TRAIN_QUESTION_COUNT,
    _dev_record,
    _question_artifact,
    _require_successful_development,
    _require_successful_training,
    _run_locked,
    _training_record,
    _validate_bank,
    _validate_dev_record,
    _validate_training_record,
    build_correction_entry,
    render_graph_note,
    main,
)
from studybench.dataset import CORPORA
from studybench.integrity import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_json,
    sha256_text,
    write_immutable_json,
)
from studybench.react import MODEL_ID, MODEL_REVISION, READ_MAX_LINES, SAMPLING
from studybench.selfquiz import _attempt_protocol
from studybench.study_protocol import (
    STATIC_GRAPH_METHOD,
    STATIC_GRAPH_NOTE_MANIFEST_TYPE,
    STATIC_GRAPH_TASK_MANIFEST_TYPE,
    StudyProtocolError,
    validate_study_note_archive,
)
from studybench.static_graph import (
    CONTRACT_VERSION,
    FROZEN_DEV_TARGETS,
    FROZEN_TRAIN_TARGETS,
    SOURCE_ROOT,
    contract_sha256,
)


class FakeRepoTools:
    def __init__(self):
        self.text = {
            "dspy/fake.py": "\n".join(
                [f"fake_line_{line}" for line in range(1, 201)]
            ) + "\n",
        }
        self.files = list(self.text)


def question(
    target: str,
    split: str,
    line: int,
    *,
    selection_stage: str,
    rank: int,
    gold_edges=None,
):
    path = "dspy/fake.py"
    if gold_edges is None:
        gold_edges = [{
            "caller": f"dspy.fake.caller_{rank}",
            "callee": target,
            "path": path,
            "line": line + 1,
            "direction": "incoming",
        }]
    return {
        "id": f"static-call-neighborhood::{target}",
        "target": target,
        "question": f"Return strict JSON for {target}",
        "anchors": [path],
        "target_definition": {"path": path, "line": line},
        "gold_edges": gold_edges,
        "split": split,
        "stratum": target.split(".")[1],
        "selection_stage": selection_stage,
        "neighborhood_edge_count": len(gold_edges),
        "stratum_rank": rank,
        "global_rank": rank,
    }


def bank_fixture():
    train = [
        question(
            target,
            "train",
            2 * index + 1,
            selection_stage=(
                "base-stratum-train" if index < 12 else "global-extra-train"
            ),
            rank=index + 1,
        )
        for index, target in enumerate(FROZEN_TRAIN_TARGETS)
    ]
    dev = [
        question(
            target,
            "dev",
            2 * (TRAIN_QUESTION_COUNT + index) + 1,
            selection_stage="global-held-out-dev",
            rank=TRAIN_QUESTION_COUNT + index + 1,
        )
        for index, target in enumerate(FROZEN_DEV_TARGETS)
    ]
    return train + dev


def resolver_fixture(bank):
    excluded = [{
        "caller": "dspy.fake.caller",
        "path": "dspy/fake.py",
        "line": 1,
        "column": 0,
        "callee_syntax": "Name(id='external', ctx=Load())",
        "reason": "unresolved-bare-callee",
    }]
    static_hash = lambda value: sha256_text(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ))
    return {
        "version": CONTRACT_VERSION,
        "source_root": SOURCE_ROOT,
        "contract_sha256": contract_sha256(),
        "source_sha256": static_hash([
            {
                "path": "dspy/fake.py",
                "sha256": "b" * 64,
            }
        ]),
        "source_files": [{
            "path": "dspy/fake.py",
            "sha256": "b" * 64,
            "line_count": 80,
        }],
        "question_bank_sha256": static_hash(bank),
        "python_implementation": "CPython",
        "python_version": platform.python_version(),
        "analyzer_module": "studybench.static_graph",
        "analyzer_source_sha256": "d" * 64,
        "corpus_name": "dspy",
        "corpus_commit": CORPORA["dspy"].commit,
        "target_symbols": [item["target"] for item in bank],
        "target_selection": [{
            key: item[key]
            for key in (
                "target", "stratum", "split", "selection_stage",
                "neighborhood_edge_count", "stratum_rank", "global_rank",
            )
        } for item in bank],
        "target_selection_sha256": static_hash([{
            key: item[key]
            for key in (
                "target", "stratum", "split", "selection_stage",
                "neighborhood_edge_count", "stratum_rank", "global_rank",
            )
        } for item in bank]),
        "candidate_inventory": [{
            "target": item["target"],
            "stratum": item["stratum"],
            "definition_path": item["target_definition"]["path"],
            "definition_line": item["target_definition"]["line"],
            "neighborhood_edge_count": item["neighborhood_edge_count"],
            "eligible": True,
            "exclusion": None,
            "stratum_rank": item["stratum_rank"],
            "global_rank": item["global_rank"],
        } for item in bank],
        "candidate_inventory_sha256": static_hash([{
            "target": item["target"],
            "stratum": item["stratum"],
            "definition_path": item["target_definition"]["path"],
            "definition_line": item["target_definition"]["line"],
            "neighborhood_edge_count": item["neighborhood_edge_count"],
            "eligible": True,
            "exclusion": None,
            "stratum_rank": item["stratum_rank"],
            "global_rank": item["global_rank"],
        } for item in bank]),
        "candidate_count": len(bank),
        "eligible_candidate_count": len(bank),
        "train_question_count": TRAIN_QUESTION_COUNT,
        "dev_question_count": DEV_QUESTION_COUNT,
        "excluded_candidates": excluded,
        "excluded_candidates_sha256": static_hash(excluded),
        "excluded_candidate_count": len(excluded),
    }


def answer_for(item):
    return json.dumps(
        {"edges": item["gold_edges"]},
        sort_keys=True,
        separators=(",", ":"),
    )


def attempt_fixture(seed, answer):
    trajectory = {
        "thought_0": "inspect the source",
        "tool_name_0": "read_file",
        "tool_args_0": {"path": "pkg/a.py"},
        "observation_0": "source",
    }
    return {
        "status": "ok",
        "answer": answer,
        "error": None,
        "seed": seed,
        "trajectory": trajectory,
        "trajectory_sha256": sha256_json(trajectory),
    }


def failed_attempt_fixture(seed, error="test failure"):
    trajectory = {}
    return {
        "status": "error",
        "answer": "",
        "error": error,
        "seed": seed,
        "trajectory": trajectory,
        "trajectory_sha256": sha256_json(trajectory),
    }


def call_fixture(owner_id, phase, seed):
    prompt_tokens, completion_tokens = 5, 3
    return {
        "call_id": f"call-{sha256_text(f'{owner_id}|{phase}|{seed}')[:20]}",
        "owner_id": owner_id,
        "phase": phase,
        "seed": seed,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "response_model": MODEL_ID.removeprefix("openai/"),
        "response_id": f"response-{sha256_text(f'{owner_id}|{phase}|{seed}')[:20]}",
        "system_fingerprint": "test-fingerprint",
        "request_messages_sha256": sha256_json([phase]),
        "request_messages_available": True,
        "outputs_sha256": sha256_json(["answer"]),
        "outputs_available": True,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "usage_reported": True,
        "provider_usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def repository_tools():
    def grep(pattern: str, path: str = ""):
        return ""

    def glob(pattern: str):
        return ""

    def read_file(path: str, start_line: int = 1, end_line: int | None = None):
        return ""

    return [grep, glob, read_file]


class DeterministicCorrectionTests(unittest.TestCase):
    def setUp(self):
        self.rt = FakeRepoTools()
        self.item = bank_fixture()[0]

    def test_correction_is_exact_set_difference_with_source_line_evidence(self):
        from studybench.static_graph import verify_prediction

        first = self.item["gold_edges"][0]
        second = {
            **first,
            "caller": "dspy.fake.other_caller",
            "line": first["line"] + 1,
        }
        item = {**self.item, "gold_edges": [first, second]}
        score = verify_prediction(
            json.dumps({"edges": [first]}), item["gold_edges"], source_view=self.rt
        )
        entry = build_correction_entry(item, score, self.rt)
        self.assertEqual(entry["verdict"], "partial")
        self.assertEqual(entry["missing_edges"], [second])
        self.assertEqual(entry["spurious_edges"], [])
        self.assertEqual(
            [evidence["quote"] for evidence in entry["evidence"]],
            ["fake_line_2", "fake_line_3"],
        )
        self.assertEqual(
            json.loads(entry["correction"]), {"edges": item["gold_edges"]}
        )

        exact = verify_prediction(
            answer_for(item), item["gold_edges"], source_view=self.rt
        )
        self.assertIsNone(build_correction_entry(item, exact, self.rt))

    def test_dev_gold_is_disjoint_and_cannot_construct_or_render_an_entry(self):
        bank = bank_fixture()
        resolver = resolver_fixture(bank)
        _, selected = _validate_bank(bank, resolver, smoke=False)
        self.assertEqual(len(selected), TRAIN_QUESTION_COUNT + DEV_QUESTION_COUNT)
        dev = selected[-1]
        from studybench.static_graph import verify_prediction

        score = verify_prediction("{\"edges\":[]}", dev["gold_edges"], source_view=self.rt)
        with self.assertRaisesRegex(ValueError, "Development|development"):
            build_correction_entry(dev, score, self.rt)

        overlap = json.loads(json.dumps(bank))
        shared_site = {
            "caller": overlap[0]["target"],
            "callee": overlap[-1]["target"],
            "path": "dspy/fake.py",
            "line": 70,
        }
        overlap[0]["gold_edges"] = [{**shared_site, "direction": "outgoing"}]
        overlap[-1]["gold_edges"] = [{**shared_site, "direction": "incoming"}]
        with self.assertRaisesRegex(ValueError, "overlap"):
            _validate_bank(overlap, resolver_fixture(overlap), smoke=False)

        train_score = verify_prediction(
            "{\"edges\":[]}", bank[0]["gold_edges"], source_view=self.rt
        )
        note = render_graph_note(
            self.rt, [build_correction_entry(bank[0], train_score, self.rt)]
        )
        self.assertNotIn(dev["target"], note)


class AttemptAndScoreValidationTests(unittest.TestCase):
    def setUp(self):
        self.rt = FakeRepoTools()
        self.bank = bank_fixture()
        self.resolver = resolver_fixture(self.bank)
        self.environment = {
            "schema_version": 1,
            "sha256": "f" * 64,
            "bytes": 1,
            "snapshot": "r1/environments/environment-test.json",
        }
        self.task_manifest = {"environment": {}}

    @staticmethod
    def fake_attempt(
        question_text, note, tools, url, *, seed, owner_id, phase,
        attempt_access,
    ):
        if attempt_access != ATTEMPT_ACCESS:
            raise AssertionError("graph ATTEMPT access drifted")
        item = next(item for item in bank_fixture() if item["question"] == question_text)
        return (
            attempt_fixture(seed, answer_for(item)),
            [call_fixture(owner_id, phase, seed)],
        )

    def test_training_score_is_recomputed_and_tamper_fails_closed(self):
        item = self.bank[0]
        artifact = _question_artifact(item, self.resolver)
        with patch("studybench.graph_study._attempt", side_effect=self.fake_attempt):
            record = _training_record(
                item,
                artifact,
                self.rt,
                repository_tools(),
                "unused",
                study_id="study-test",
                master_seed=7,
                launch_environment=self.environment,
            )
        with patch("studybench.graph_study._validate_environment_binding"):
            validated = _validate_training_record(
                record,
                item,
                artifact,
                self.rt,
                study_id="study-test",
                master_seed=7,
                sdir=Path("unused"),
                task_manifest=self.task_manifest,
                smoke=False,
            )
            self.assertTrue(validated["score"]["exact"])
            tampered = json.loads(json.dumps(record))
            tampered["score"]["f1"] = 0.25
            with self.assertRaisesRegex(SystemExit, "score"):
                _validate_training_record(
                    tampered,
                    item,
                    artifact,
                    self.rt,
                    study_id="study-test",
                    master_seed=7,
                    sdir=Path("unused"),
                    task_manifest=self.task_manifest,
                    smoke=False,
                )

    def test_failed_training_is_retained_but_never_becomes_a_correction(self):
        item = self.bank[0]
        artifact = _question_artifact(item, self.resolver)

        def fail(
            question_text, note, tools, url, *, seed, owner_id, phase,
            attempt_access,
        ):
            self.assertEqual(attempt_access, ATTEMPT_ACCESS)
            return failed_attempt_fixture(seed), []

        with patch("studybench.graph_study._attempt", side_effect=fail):
            record = _training_record(
                item,
                artifact,
                self.rt,
                repository_tools(),
                "unused",
                study_id="study-test",
                master_seed=7,
                launch_environment=self.environment,
            )
        self.assertEqual(record["attempt"]["status"], "error")
        self.assertIsNone(record["entry"])
        with patch("studybench.graph_study._validate_environment_binding"):
            _validate_training_record(
                record,
                item,
                artifact,
                self.rt,
                study_id="study-test",
                master_seed=7,
                sdir=Path("unused"),
                task_manifest=self.task_manifest,
                smoke=False,
            )
        with self.assertRaisesRegex(SystemExit, "cannot be resumed"):
            _require_successful_training([record])

    def test_successful_smoke_attempt_still_requires_a_model_call(self):
        item = self.bank[0]
        artifact = _question_artifact(item, self.resolver)
        with patch("studybench.graph_study._attempt", side_effect=self.fake_attempt):
            record = _training_record(
                item,
                artifact,
                self.rt,
                repository_tools(),
                "unused",
                study_id="study-test",
                master_seed=7,
                launch_environment=self.environment,
            )
        record["calls"] = []
        with (
            patch("studybench.graph_study._validate_environment_binding"),
            self.assertRaisesRegex(SystemExit, "ledger is missing"),
        ):
            _validate_training_record(
                record,
                item,
                artifact,
                self.rt,
                study_id="study-test",
                master_seed=7,
                sdir=Path("unused"),
                task_manifest=self.task_manifest,
                smoke=True,
            )

    def test_failed_development_arm_burns_the_immutable_study_id(self):
        item = self.bank[-1]
        artifact = _question_artifact(item, self.resolver)

        def fail_one_arm(
            question_text, note, tools, url, *, seed, owner_id, phase,
            attempt_access,
        ):
            self.assertEqual(attempt_access, ATTEMPT_ACCESS)
            if phase.endswith("with_note"):
                return failed_attempt_fixture(seed), []
            return (
                attempt_fixture(seed, answer_for(item)),
                [call_fixture(owner_id, phase, seed)],
            )

        with patch("studybench.graph_study._attempt", side_effect=fail_one_arm):
            record = _dev_record(
                item,
                artifact,
                "learned note",
                self.rt,
                repository_tools(),
                "unused",
                study_id="study-test",
                master_seed=7,
                launch_environment=self.environment,
            )
        with patch("studybench.graph_study._validate_environment_binding"):
            _validate_dev_record(
                record,
                item,
                artifact,
                "learned note",
                self.rt,
                study_id="study-test",
                master_seed=7,
                sdir=Path("unused"),
                task_manifest=self.task_manifest,
            )
        with self.assertRaisesRegex(SystemExit, "with_note.*no treatment result"):
            _require_successful_development([record])

    def test_dev_attempts_share_seed_tools_and_protocol_and_only_note_differs(self):
        item = self.bank[-1]
        artifact = _question_artifact(item, self.resolver)
        observed = []

        def fake_attempt(
            question_text, note, tools, url, *, seed, owner_id, phase,
            attempt_access,
        ):
            observed.append((
                question_text, note, tuple(tools), seed, owner_id, phase,
                attempt_access,
            ))
            return (
                attempt_fixture(seed, answer_for(item)),
                [call_fixture(owner_id, phase, seed)],
            )

        tools = repository_tools()
        with patch("studybench.graph_study._attempt", side_effect=fake_attempt):
            record = _dev_record(
                item,
                artifact,
                "learned note",
                self.rt,
                tools,
                "unused",
                study_id="study-test",
                master_seed=7,
                launch_environment=self.environment,
            )
        self.assertEqual(len(observed), 2)
        self.assertEqual(observed[0][0], observed[1][0])
        self.assertEqual(observed[0][2], observed[1][2])
        self.assertEqual(observed[0][3], observed[1][3])
        self.assertEqual([row[6] for row in observed], [ATTEMPT_ACCESS] * 2)
        self.assertEqual([row[1] for row in observed], ["learned note", ""])
        self.assertEqual(record["paired_seed"], observed[0][3])
        self.assertEqual(
            record["attempt_protocol"],
            {
                **_attempt_protocol(ATTEMPT_ACCESS),
                "paired_seed": record["paired_seed"],
                "only_manipulated_field": "note",
            },
        )
        self.assertEqual(
            record["presented_inputs"]["with_note"]["question_sha256"],
            record["presented_inputs"]["bare"]["question_sha256"],
        )


class EndToEndArtifactTests(unittest.TestCase):
    def _run(self, directory, *, smoke):
        sdir = Path(directory) / "study"
        bank = bank_fixture()
        resolver = resolver_fixture(bank)
        args = SimpleNamespace(
            study_id="graph-test",
            seed=17,
            base_urls="http://localhost:8100/v1",
            concurrency=3,
            smoke=smoke,
            debug=False,
        )

        def write_task(args, target, urls, contract, full_bank):
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "manifest_type": STATIC_GRAPH_TASK_MANIFEST_TYPE,
                "study_id": args.study_id,
                "task": "dspy",
                "round": 1,
                "master_seed": args.seed,
                "source_root": SOURCE_ROOT,
                "model": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "sampling": SAMPLING,
                "corpus_commit": CORPORA["dspy"].commit,
                "corpus": {
                    "name": "dspy",
                    "commit": CORPORA["dspy"].commit,
                },
                "source": {"git_commit": "test-source"},
                "environment": {"server_count": 1},
                "environment_contract": {"schema_version": 1},
                "server_transport": {
                    "scope": "loopback",
                    "protocol": "openai-compatible-http",
                    "server_count": 1,
                    "assignment": (
                        "stable_seed(master_seed, owner_id, server) modulo "
                        "server_count"
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
                "resolver_contract": contract,
                "resolver_contract_sha256": sha256_json(contract),
                "question_bank_sha256": contract["question_bank_sha256"],
                "question_bank_artifact_sha256": sha256_json(full_bank),
                "config": {
                    "method": STATIC_GRAPH_METHOD,
                    "smoke": args.smoke,
                    "concurrency": args.concurrency,
                    "attempt_protocol": _attempt_protocol(ATTEMPT_ACCESS),
                    "train_input_note_sha256": sha256_text(""),
                    "train_question_count": (
                        1 if args.smoke else TRAIN_QUESTION_COUNT
                    ),
                    "dev_question_count": 0 if args.smoke else DEV_QUESTION_COUNT,
                    "dev_holdout_targets": (
                        [] if args.smoke else list(FROZEN_DEV_TARGETS)
                    ),
                    "provider_retries": 0,
                    "read_max_lines": READ_MAX_LINES,
                },
            }
            write_immutable_json(target / "manifest.json", manifest)
            return manifest

        def snapshot(target, task_manifest, *, smoke):
            data = b"{}\n"
            relative = Path(
                f"r1/environments/environment-{sha256_bytes(data)}.json"
            )
            path = target / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_bytes(data)
            return {
                "schema_version": 1,
                "sha256": sha256_bytes(data),
                "bytes": len(data),
                "snapshot": relative.as_posix(),
            }

        def fake_attempt(
            question_text, note, tools, url, *, seed, owner_id, phase,
            attempt_access,
        ):
            if attempt_access != ATTEMPT_ACCESS:
                raise AssertionError("graph ATTEMPT access drifted")
            item = next(item for item in bank if item["question"] == question_text)
            return (
                attempt_fixture(seed, answer_for(item)),
                [call_fixture(owner_id, phase, seed)],
            )

        patches = [
            patch("studybench.graph_study._study_dir", return_value=sdir),
            patch("studybench.graph_study.validate_local_server_urls",
                  return_value=["http://localhost:8100/v1"]),
            patch("studybench.graph_study.RepoTools", return_value=FakeRepoTools()),
            patch("studybench.graph_study.build_question_bank", return_value=bank),
            patch("studybench.graph_study.resolver_contract_record", return_value=resolver),
            patch("studybench.graph_study._write_task_manifest", side_effect=write_task),
            patch("studybench.graph_study._snapshot_launch", side_effect=snapshot),
            patch("studybench.graph_study._validate_environment_binding"),
            patch("studybench.graph_study.environment_contract_is_valid", return_value=True),
            patch("studybench.graph_study.make_tools", return_value=repository_tools()),
            patch("studybench.graph_study._attempt", side_effect=fake_attempt),
        ]
        stack = ExitStack()
        for context in patches:
            stack.enter_context(context)
        try:
            result = _run_locked(args)
        finally:
            stack.close()
        return args, sdir, result, patches

    def test_full_run_manifest_inventory_and_resume_tamper_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            args, sdir, result, patches = self._run(directory, smoke=False)
            manifest = json.loads(result.read_text(encoding="utf-8"))
            self.assertEqual(manifest["manifest_type"], STATIC_GRAPH_NOTE_MANIFEST_TYPE)
            self.assertEqual(manifest["method"], STATIC_GRAPH_METHOD)
            self.assertEqual(
                manifest["protocol_summary"],
                {
                    "schema_version": 1,
                    "task_manifest_sha256": manifest["construction_artifacts"]
                    ["manifest.json"]["sha256"],
                    "method": STATIC_GRAPH_METHOD,
                    "question_mode": "static-call-neighborhood",
                    "focus": None,
                    "attempt_access": "react-corpus",
                    "attempt_protocol_sha256": sha256_json(
                        _attempt_protocol(ATTEMPT_ACCESS)
                    ),
                    "resolver_contract_sha256": sha256_json(resolver_fixture(bank_fixture())),
                    "question_bank_sha256": resolver_fixture(bank_fixture())[
                        "question_bank_sha256"
                    ],
                    "question_bank_artifact_sha256": sha256_json(bank_fixture()),
                },
            )
            self.assertTrue(manifest["automated_claim_ready"])
            self.assertTrue(all(manifest["automated_readiness"].values()))
            self.assertEqual(
                len(manifest["train_question_ids"]), TRAIN_QUESTION_COUNT
            )
            self.assertEqual(
                len(manifest["held_out_dev_question_ids"]), DEV_QUESTION_COUNT
            )
            self.assertTrue((sdir / "r1/dev-exam.jsonl").is_file())
            self.assertEqual(
                len((sdir / "r1/items.jsonl").read_text(encoding="utf-8").splitlines()),
                TRAIN_QUESTION_COUNT,
            )

            dependencies = {
                relative: (sdir / relative).read_bytes()
                for relative in manifest["construction_artifacts"]
            }
            note_bytes = (sdir / "notes/note-r1.md").read_bytes()
            with ExitStack() as stack:
                for context in patches:
                    stack.enter_context(context)
                self.assertEqual(
                    validate_study_note_archive(
                        manifest,
                        dependencies,
                        note_bytes,
                        allow_smoke=True,
                    ),
                    manifest["protocol_summary"],
                )

                # A coherently rehashed, self-consistent stored score is still
                # rejected because downstream validation reruns the deterministic
                # verifier against the pinned source and frozen gold.
                forged = json.loads(json.dumps(manifest))
                forged_dependencies = dict(dependencies)
                item_path = next(
                    path
                    for path in forged_dependencies
                    if path.startswith("r1/items/") and path.endswith(".json")
                )
                item = json.loads(forged_dependencies[item_path])
                item["score"]["f1"] = 0.5
                item["score_sha256"] = sha256_json(item["score"])
                forged_dependencies[item_path] = canonical_json_bytes(item)
                aggregate = [
                    json.loads(line)
                    for line in forged_dependencies["r1/items.jsonl"].splitlines()
                ]
                aggregate = [
                    item if record["question_id"] == item["question_id"] else record
                    for record in aggregate
                ]
                forged_dependencies["r1/items.jsonl"] = b"".join(
                    canonical_json_bytes(record) for record in aggregate
                )
                for path in (item_path, "r1/items.jsonl"):
                    data = forged_dependencies[path]
                    forged["construction_artifacts"][path] = {
                        "sha256": sha256_bytes(data),
                        "bytes": len(data),
                    }
                forged["construction_artifacts_sha256"] = sha256_json(
                    forged["construction_artifacts"]
                )
                with self.assertRaisesRegex(
                    StudyProtocolError, "score|re-attestation"
                ):
                    validate_study_note_archive(
                        forged,
                        forged_dependencies,
                        note_bytes,
                        allow_smoke=True,
                    )

                forged_round = json.loads(json.dumps(manifest))
                forged_round_dependencies = dict(dependencies)
                round_record = json.loads(
                    forged_round_dependencies["r1/manifest.json"]
                )
                round_record["unexpected"] = True
                round_bytes = canonical_json_bytes(round_record)
                forged_round_dependencies["r1/manifest.json"] = round_bytes
                forged_round["construction_artifacts"]["r1/manifest.json"] = {
                    "sha256": sha256_bytes(round_bytes),
                    "bytes": len(round_bytes),
                }
                forged_round["construction_artifacts_sha256"] = sha256_json(
                    forged_round["construction_artifacts"]
                )
                with self.assertRaisesRegex(
                    StudyProtocolError, "round manifest|re-attestation"
                ):
                    validate_study_note_archive(
                        forged_round,
                        forged_round_dependencies,
                        note_bytes,
                        allow_smoke=True,
                    )

            # Rehashing a deliberately truncated inventory must not turn it into
            # a valid archive.  The validator derives the required aggregate
            # independently from the frozen question population.
            truncated = json.loads(json.dumps(manifest))
            truncated_dependencies = dict(dependencies)
            truncated_dependencies.pop("r1/items.jsonl")
            truncated["construction_artifacts"].pop("r1/items.jsonl")
            truncated["construction_artifacts_sha256"] = sha256_json(
                truncated["construction_artifacts"]
            )
            with self.assertRaisesRegex(
                StudyProtocolError, "missing graph training items"
            ):
                validate_study_note_archive(
                    truncated,
                    truncated_dependencies,
                    note_bytes,
                    allow_smoke=True,
                )

            environment = next((sdir / "r1/environments").glob("environment-*.json"))
            environment.write_text("{\"tampered\":true}\n", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "launch-environment|inventory|manifest"):
                self._run(directory, smoke=False)

    def test_round_manifest_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            _, sdir, _, _ = self._run(directory, smoke=False)
            path = sdir / "r1/manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["master_seed"] += 1
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "round manifest drifted"):
                self._run(directory, smoke=False)

    def test_note_protocol_summary_tamper_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            _, sdir, _, _ = self._run(directory, smoke=False)
            path = sdir / "notes/note-r1.manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["protocol_summary"]["method"] = "tampered"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "note manifest|protocol"):
                self._run(directory, smoke=False)

    def test_smoke_runs_first_frozen_target_without_dev_and_is_nonpromotable(self):
        with tempfile.TemporaryDirectory() as directory:
            _, sdir, result, _ = self._run(directory, smoke=True)
            manifest = json.loads(result.read_text(encoding="utf-8"))
            self.assertFalse(manifest["automated_claim_ready"])
            self.assertFalse(manifest["automated_readiness"]["non_smoke"])
            self.assertEqual(len(manifest["train_question_ids"]), 1)
            self.assertEqual(manifest["held_out_dev_question_ids"], [])
            self.assertFalse((sdir / "r1/dev-exam.jsonl").exists())
            records = [
                json.loads(line)
                for line in (sdir / "r1/items.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([record["target"] for record in records], [SMOKE_TARGET])


class CommandLineSafetyTests(unittest.TestCase):
    def test_invalid_study_id_is_rejected_before_log_file_creation(self):
        argv = [
            "graph_study",
            "--study-id", "../escape",
            "--seed", "1",
            "--base-urls", "http://localhost:8100/v1",
        ]
        with (
            patch("sys.argv", argv),
            patch("studybench.graph_study.logging.FileHandler") as file_handler,
            self.assertRaises(SystemExit),
        ):
            main()
        file_handler.assert_not_called()


if __name__ == "__main__":
    unittest.main()
