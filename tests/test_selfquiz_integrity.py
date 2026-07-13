from contextlib import nullcontext
import json
from pathlib import Path, PurePosixPath
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import pydantic

from studybench.integrity import sha256_file, sha256_json, sha256_text, stable_seed
from studybench.human_audit import HUMAN_AUDIT_DECISION_RULE_ID
from studybench.provenance import (
    _load_note,
    environment_contract_record,
    write_environment_snapshot,
)
from studybench.selfquiz import (
    SCHEMA_VERSION,
    _error_rate,
    _bind_launch_environment,
    _derive,
    _launch_environment_inventory,
    _record_id,
    _run_round_locked,
    _schema_error,
    _validate_completed_item,
    _validate_promoted_auditor_identity,
    _study_dir,
    _study_round_lock,
    _load_input_note,
    _prepare_questions,
    _trajectory_hash_valid,
    _validate_artifact_environment,
    _validate_dev_exam,
    _validate_dev_reference,
    _validate_question_provenance,
    _write_note,
    _write_task_manifest,
    collect_dev_questions,
    collect_note_entries,
    dedup,
    eligible_retest,
    freshness_audit,
    freshness_sources_complete,
    is_distillable_item,
    make_retest_item,
    promote_human_audit,
    quote_gate,
    run_dev_item,
    run_round,
    serialize_trajectory,
    usage_by_phase,
    usage_ledger_audit,
    usage_records,
    usage_totals,
    validate_anchor,
    validate_anchors,
    validate_audit_protocol,
    validate_evidence,
    Evidence,
)


class FakeRepoTools:
    def __init__(self):
        self.text = {
            "pkg/a.py": "first = True\nvalue = 1\nreturn value\n",
            "pkg/b.py": "pass\n",
        }
        self.files = list(self.text)


def quiz_item(item_id="q1", *, split="train", round_number=1):
    return {
        "schema_version": SCHEMA_VERSION,
        "item_id": item_id,
        "origin_item_id": item_id,
        "origin_round": round_number,
        "round": round_number,
        "kind": "quiz",
        "split": split,
        "question": f"question {item_id}",
        "qtype": "behavior",
        "anchors": ["pkg/a.py"],
        "chapter": "pkg",
        "writer_sketch": "sketch",
        "status": "ok",
        "verdict": "correct",
        "entry": None,
    }


def model_calls(owner_id: str, phase: str, seed: int) -> list[dict]:
    """Build one complete offline provider record for call-graph tests."""

    lm = SimpleNamespace(history=[{
        "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        "response_model": "served-model",
        "response": SimpleNamespace(
            id=f"response-{owner_id}-{phase}-{seed}", system_fingerprint="fp"),
        "messages": [{"role": "user", "content": phase}],
        "outputs": [f"output-{phase}"],
    }])
    return usage_records(lm, phase=phase, owner_id=owner_id, seed=seed)


def dev_reference_fixture(item: dict, *, study_id: str = "study-a",
                          task: str = "dspy", master_seed: int = 7) -> dict:
    owner_id = _record_id("dev-reference", study_id, task, item["item_id"])
    evidence = {"file": "pkg/a.py", "line": 3, "quote": "return value"}
    derivations, calls = [], []
    for index in range(2):
        seed = stable_seed(master_seed, owner_id, "derive", index)
        support_seed = stable_seed(seed, "reference-support")
        derivations.append({
            "derivation_id": _record_id("derivation", owner_id, index, seed),
            "seed": seed,
            "status": "ok",
            "answer": f"answer {index}",
            "raw_evidence": [evidence],
            "evidence": [evidence],
            "rejected_evidence": [],
            "trajectory": {},
            "trajectory_sha256": sha256_json({}),
            "reference_support": {
                "status": "ok", "seed": support_seed,
                "supported": True, "rationale": "supported",
            },
            "evidence_class": "quote-only",
        })
        calls += model_calls(owner_id, f"derive-{index}", seed)
        calls += model_calls(owner_id, f"reference-support-{index}", support_seed)
    checks = []
    for left, right, direction in (
            (derivations[0], derivations[1], "a-to-b"),
            (derivations[1], derivations[0], "b-to-a")):
        seed = stable_seed(master_seed, owner_id, "reference-consensus", direction)
        checks.append({
            "status": "ok", "seed": seed, "verdict": "correct", "delta": "",
            "reference_id": left["derivation_id"],
            "candidate_id": right["derivation_id"],
        })
        calls += model_calls(owner_id, f"reference-consensus-{direction}", seed)
    return {
        "schema_version": SCHEMA_VERSION,
        "reference_id": owner_id,
        "origin_item_id": item["item_id"],
        "origin_round": item["origin_round"],
        "created_round": item["origin_round"],
        "question": item["question"],
        "qtype": item["qtype"],
        "anchors": item["anchors"],
        "chapter": item["chapter"],
        "status": "ok",
        "derivations": derivations,
        "references": derivations,
        "reference_consensus": checks,
        "reference_ids": [derivation["derivation_id"] for derivation in derivations],
        "calls": calls,
        "usage": usage_totals(calls),
    }


def question_provenance_fixture(chapters: list[str], *, count: int = 3):
    args = SimpleNamespace(
        study_id="study-a", task="dspy", round=1, seed=7,
        smoke=False, questions=count,
    )
    records, episodes = [], []
    for chapter in chapters:
        owner_id = _record_id("quiz", args.study_id, args.task, args.round, chapter)
        seed = stable_seed(
            args.seed, args.study_id, args.task, args.round, "quiz", chapter)
        raw_questions = [{
            "question": f"question {chapter} {ordinal}",
            "qtype": "behavior",
            "anchors": ["pkg/a.py"],
            "writer_sketch": f"sketch {ordinal}",
        } for ordinal in range(count)]
        calls = model_calls(owner_id, "quiz", seed)
        episodes.append({
            "owner_id": owner_id,
            "chapter": chapter,
            "seed": seed,
            "status": "ok",
            "questions": raw_questions,
            "trajectory": {},
            "trajectory_sha256": sha256_json({}),
            "calls": calls,
            "usage": usage_totals(calls),
        })
        for ordinal, raw in enumerate(raw_questions):
            item_id = _record_id(
                "question", args.study_id, args.task, args.round,
                chapter, ordinal, raw["question"])
            records.append({
                "schema_version": SCHEMA_VERSION,
                "item_id": item_id,
                "origin_item_id": item_id,
                "origin_round": args.round,
                "round": args.round,
                "kind": "quiz",
                "split": "dev" if ordinal == 0 else "train",
                **raw,
                "chapter": chapter,
                "quiz_episode_id": owner_id,
                "quiz_ordinal": ordinal,
            })
    return args, records, episodes


def human_audit_fixture(args, *, protocol_extensions=None):
    """Build one complete immutable construction and its exact passing audit."""

    sdir = _study_dir(args)
    protocol = {
        "schema_version": 1,
        "protocol_id": "blind-audit-01",
        "blinding": "condition_and_method_labels_hidden",
        "population": "all_train_dev_verdicts_and_admitted_entries",
        "decision_rule": HUMAN_AUDIT_DECISION_RULE_ID,
    }
    protocol.update(protocol_extensions or {})
    protocol_text = json.dumps(protocol)
    protocol_hash = sha256_text(protocol_text)
    protocol_relative = Path("audit-protocols") / f"{protocol_hash}.json"
    (sdir / protocol_relative).parent.mkdir(parents=True)
    (sdir / protocol_relative).write_text(protocol_text, encoding="utf-8")
    (sdir / "manifest.json").write_text(json.dumps({
        "study_id": args.study_id,
        "task": args.task,
        "master_seed": args.seed,
        "human_audit_protocol": {
            "sha256": protocol_hash,
            "path": str(protocol_relative),
            "protocol_id": "blind-audit-01",
        },
    }), encoding="utf-8")
    record_ids = []
    for round_number in range(1, args.round + 1):
        (sdir / f"r{round_number}").mkdir()
        train_id = f"train-{round_number}"
        dev_id = f"dev-{round_number}"
        record_ids.extend((train_id, dev_id))
        (sdir / f"r{round_number}" / "items.jsonl").write_text(
            json.dumps({"item_id": train_id}) + "\n", encoding="utf-8")
        (sdir / f"r{round_number}" / "dev-exam.jsonl").write_text(
            json.dumps({"item_id": dev_id}) + "\n", encoding="utf-8")
    _write_note(
        args, sdir, "exact note", [], input_note_sha256="input",
        round_calls=[], cumulative_calls=[], round_construction_calls=[],
        cumulative_construction_calls=[], corpus_commit="commit",
        automated_claim_ready=True, automated_readiness={"complete": True})
    construction = sdir / "notes" / f"note-r{args.round}.manifest.json"
    construction_record = json.loads(construction.read_text(encoding="utf-8"))
    audit = {
        "schema_version": 1,
        "study_id": args.study_id,
        "task": args.task,
        "round": args.round,
        "protocol_sha256": protocol_hash,
        "construction_manifest_sha256": sha256_file(construction),
        "note_sha256": construction_record["note_sha256"],
        "blinding_preserved": True,
        "reviewer_independent": True,
        "decision": "pass",
        "auditor_id": "auditor-01",
        "record_reviews": [
            {"record_id": record_id, "verdict_valid": True,
             "evidence_valid": True, "leakage_free": True}
            for record_id in record_ids
        ],
        "entry_reviews": [],
    }
    return sdir, construction, audit


class CitationIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.rt = FakeRepoTools()

    def test_anchor_requires_exact_safe_readable_file(self):
        self.assertEqual(validate_anchor(self.rt, "pkg/a.py"), "pkg/a.py")
        for invalid in ("", "/pkg/a.py", "../pkg/a.py", "pkg\\a.py", " pkg/a.py", "pkg"):
            with self.subTest(invalid=invalid):
                self.assertIsNone(validate_anchor(self.rt, invalid))
        self.assertEqual(validate_anchors(self.rt, ["pkg/a.py", "pkg/a.py"]), ["pkg/a.py"])
        self.assertIsNone(validate_anchors(self.rt, []))
        self.assertIsNone(validate_anchors(self.rt, ["pkg/a.py", "missing.py"]))

    def test_evidence_is_exact_and_canonicalizes_tolerated_line_offset(self):
        evidence = validate_evidence(
            self.rt, {"file": "pkg/a.py", "line": 1, "quote": "`return value`"}
        )
        self.assertEqual(evidence, {"file": "pkg/a.py", "line": 3, "quote": "return value"})
        self.assertTrue(quote_gate(self.rt, "pkg/a.py", 1, "return value"))

    def test_evidence_fails_closed_on_partial_multiline_or_invalid_line(self):
        invalid = [
            {"file": "pkg/a.py", "line": 2, "quote": "value"},
            {"file": "pkg/a.py", "line": 2, "quote": "value = 1\nfabricated"},
            {"file": "pkg/a.py", "line": 0, "quote": "first = True"},
            {"file": "pkg/a.py", "line": -1, "quote": "first = True"},
            {"file": "pkg/a.py", "line": 99, "quote": "return value"},
            {"file": "missing.py", "line": 1, "quote": "pass"},
        ]
        for evidence in invalid:
            with self.subTest(evidence=evidence):
                self.assertIsNone(validate_evidence(self.rt, evidence))

        for invalid_line in (True, "1"):
            with self.subTest(invalid_model_line=invalid_line):
                with self.assertRaises(pydantic.ValidationError):
                    Evidence(file="pkg/a.py", line=invalid_line, quote="first = True")

    def test_evidence_allows_only_explicit_markdown_wrappers(self):
        fenced = "```python\nvalue = 1\n```"
        quoted = "> `value = 1`"
        self.assertEqual(validate_evidence(
            self.rt, {"file": "pkg/a.py", "line": 2, "quote": fenced})["line"], 2)
        self.assertEqual(validate_evidence(
            self.rt, {"file": "pkg/a.py", "line": 2, "quote": quoted})["line"], 2)


class SplitLineageTests(unittest.TestCase):
    def test_only_original_train_quiz_can_distill(self):
        train = quiz_item()
        dev = quiz_item("dev", split="dev")
        missing_schema = {key: value for key, value in train.items() if key != "split"}
        replay = dict(train, round=2)
        self.assertTrue(is_distillable_item(train))
        self.assertFalse(is_distillable_item(dev))
        self.assertFalse(is_distillable_item(missing_schema))
        self.assertFalse(is_distillable_item(replay))

    def test_boolean_schema_and_rounds_never_enter_train_or_dev_sets(self):
        valid = quiz_item()
        invalid = [
            dict(valid, schema_version=True),
            dict(valid, round=True, origin_round=True),
            dict(valid, item_id=None, origin_item_id=None),
        ]
        for item in invalid:
            with self.subTest(item=item):
                self.assertFalse(is_distillable_item(item))
                self.assertIsNotNone(_schema_error(item))

        dev = quiz_item("dev", split="dev")
        self.assertEqual(collect_dev_questions([[dev]]), [dev])
        self.assertEqual(
            collect_dev_questions([[dict(dev, schema_version=True)]]), []
        )
        self.assertEqual(
            collect_dev_questions([[dict(dev, round=True, origin_round=True)]]), []
        )

        completed = dict(valid, input_note_sha256="note", derivations=[], calls=[])
        for field in ("round", "origin_round"):
            with self.subTest(completed_field=field):
                with self.assertRaises(SystemExit):
                    _validate_completed_item(
                        valid,
                        dict(completed, **{field: True}),
                        "note",
                        master_seed=7,
                        rt=FakeRepoTools(),
                    )

    def test_resolved_training_record_requires_its_exact_model_call_graph(self):
        item = quiz_item()
        attempt_seed = stable_seed(7, item["item_id"], "attempt")
        fabricated = {
            **item,
            "input_note_sha256": "note",
            "attempt": "fabricated answer",
            "attempt_seed": attempt_seed,
            "derivations": [],
            "reference_consensus": [],
            "reference_ids": [],
            "calls": [],
            "usage": usage_totals([]),
        }
        with self.assertRaisesRegex(SystemExit, "missing calls for phase attempt"):
            _validate_completed_item(
                item,
                fabricated,
                "note",
                master_seed=7,
                rt=FakeRepoTools(),
            )

    def test_retest_preserves_origin_and_cannot_reenter_note(self):
        origin = quiz_item()
        retest = make_retest_item(origin, task="dspy", study_id="study-a", round_number=2)
        self.assertEqual(retest["origin_item_id"], origin["item_id"])
        self.assertEqual(retest["origin_round"], 1)
        self.assertEqual(retest["retest_of"], origin["item_id"])
        self.assertEqual(retest["split"], "train")
        self.assertFalse(is_distillable_item(retest))
        self.assertFalse(eligible_retest(retest))
        for invalid_round in (True, 0, 1):
            with self.subTest(invalid_round=invalid_round):
                with self.assertRaises(ValueError):
                    make_retest_item(
                        origin, task="dspy", study_id="study-a",
                        round_number=invalid_round,
                    )

    def test_retest_candidates_are_resolved_original_train_items(self):
        correct = quiz_item("correct")
        admitted = quiz_item("admitted")
        admitted.update(verdict="wrong", entry={"entry_id": "e-admitted"})
        bounced = quiz_item("bounced")
        bounced.update(verdict="wrong", entry=None)
        dev = quiz_item("dev", split="dev")
        self.assertTrue(eligible_retest(correct))
        self.assertTrue(eligible_retest(admitted))
        self.assertFalse(eligible_retest(bounced))
        self.assertFalse(eligible_retest(dev))

    def test_note_collection_excludes_dev_retest_and_missing_lineage(self):
        train = quiz_item("train")
        train["entry"] = {"entry_id": "e-train", "chapter": "pkg"}
        dev = quiz_item("dev", split="dev")
        dev["entry"] = {"entry_id": "e-dev", "chapter": "pkg"}
        retest = make_retest_item(quiz_item("origin"), task="dspy",
                                  study_id="study-a", round_number=2)
        retest.update(status="ok", verdict="wrong",
                      entry={"entry_id": "e-retest", "chapter": "pkg"})
        legacy = {"entry": {"entry_id": "e-legacy", "chapter": "pkg"}}
        self.assertEqual([entry["entry_id"] for entry in
                          collect_note_entries([dev, retest, legacy, train])], ["e-train"])

    def test_cumulative_dev_pool_is_unique_across_rounds(self):
        r1 = quiz_item("dev-1", split="dev", round_number=1)
        r2 = quiz_item("dev-2", split="dev", round_number=2)
        train = quiz_item("train")
        pool = collect_dev_questions([[r1, train], [r1, r2]])
        self.assertEqual([item["item_id"] for item in pool], ["dev-1", "dev-2"])

    def test_training_schema_rejects_dev(self):
        self.assertIsNone(_schema_error(quiz_item()))
        self.assertIn("train", _schema_error(quiz_item("dev", split="dev")))


class DeterminismAndArtifactTests(unittest.TestCase):
    def test_round_validates_corpus_before_writing_task_manifest(self):
        args = SimpleNamespace(
            task="dspy",
            study_id="study-ordering",
            round=1,
            base_urls="http://localhost:8100/v1",
            smoke=True,
        )
        with tempfile.TemporaryDirectory() as directory, (
            patch("studybench.selfquiz._study_dir", return_value=Path(directory))
        ), patch(
            "studybench.selfquiz.validate_local_server_urls",
            return_value=["http://localhost:8100/v1"],
        ), patch(
            "studybench.selfquiz.RepoTools",
            side_effect=ValueError("invalid pinned corpus"),
        ), patch("studybench.selfquiz._write_task_manifest") as write_manifest:
            with self.assertRaisesRegex(ValueError, "invalid pinned corpus"):
                _run_round_locked(args)
        write_manifest.assert_not_called()

    def test_stable_seed_depends_on_every_semantic_part(self):
        first = stable_seed(7, "study", "dspy", 1, "quiz", 0)
        self.assertEqual(first, stable_seed(7, "study", "dspy", 1, "quiz", 0))
        self.assertNotEqual(first, stable_seed(7, "study", "dspy", 1, "quiz", 1))
        self.assertNotEqual(first, stable_seed(8, "study", "dspy", 1, "quiz", 0))

    def test_record_ids_are_stable_and_namespaced(self):
        self.assertEqual(_record_id("q", "a", 1), _record_id("q", "a", 1))
        self.assertNotEqual(_record_id("q", "a", 1), _record_id("q", "a", 2))
        self.assertTrue(_record_id("q", "a", 1).startswith("q-"))

    def test_study_id_cannot_escape_output_namespace(self):
        good = _study_dir(SimpleNamespace(study_id="replication-01", task="dspy"))
        self.assertTrue(str(good).endswith("studies/replication-01/dspy"))
        for invalid in ("../escape", "a/b", "", ".hidden", "UPPER", "ab"):
            with self.subTest(invalid=invalid), self.assertRaises(SystemExit):
                _study_dir(SimpleNamespace(study_id=invalid, task="dspy"))

    def test_study_artifact_namespace_rejects_symlinked_parents(self):
        args = SimpleNamespace(study_id="replication-01", task="dspy")
        with tempfile.TemporaryDirectory() as directory, \
                patch("studybench.selfquiz.ROOT", Path(directory)):
            target = Path(directory) / "redirected"
            target.mkdir()
            base = Path(directory) / "study-selfquiz"
            base.mkdir()
            (base / "studies").symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(SystemExit, "symlink"):
                _study_dir(args)

    def test_round_lock_is_nonblocking_and_outside_artifact_tree(self):
        args = SimpleNamespace(study_id="replication-01", task="dspy", round=1)
        with tempfile.TemporaryDirectory() as directory, \
                patch("studybench.selfquiz.ROOT", Path(directory)), \
                patch("studybench.selfquiz._run_round_locked") as inner:
            with _study_round_lock(args) as lock_path:
                self.assertIn(".studybench-locks", lock_path.parts)
                self.assertNotIn("study-selfquiz", lock_path.parts)
                with self.assertRaisesRegex(SystemExit, "already active"):
                    run_round(args)
            inner.assert_not_called()
            with _study_round_lock(args):
                pass

    def test_round_rechecks_final_manifest_inside_lock_before_work(self):
        args = SimpleNamespace(
            study_id="replication-01", task="dspy", round=1, seed=7,
            smoke=False, chapters=4, questions=5, concurrency=8,
            base_urls="http://localhost:8100/v1", audit_protocol=None,
        )
        with tempfile.TemporaryDirectory() as directory, \
                patch("studybench.selfquiz.ROOT", Path(directory)):
            sdir = _study_dir(args)
            sdir.mkdir(parents=True)
            (sdir / "manifest.json").write_text(json.dumps({
                "study_id": args.study_id,
                "task": args.task,
                "master_seed": args.seed,
                "config": {
                    "chapters_per_round": args.chapters,
                    "questions_per_chapter": args.questions,
                    "smoke": args.smoke,
                    "concurrency": args.concurrency,
                },
                "server_transport": {"server_count": 1},
            }))
            _write_note(
                args, sdir, "final note", [], input_note_sha256=sha256_text(""),
                round_calls=[], cumulative_calls=[], round_construction_calls=[],
                cumulative_construction_calls=[], corpus_commit="commit",
                automated_claim_ready=False,
                automated_readiness={"complete": False},
            )
            with patch("studybench.selfquiz._run_round_locked") as inner:
                run_round(args)
            inner.assert_not_called()

    def test_dedup_is_deterministic(self):
        seen = ["How does Foo handle a missing value?"]
        self.assertTrue(dedup("How does Foo handle the missing value?", seen))
        self.assertFalse(dedup("Where is Bar registered?", seen))

    def test_note_manifest_is_immutable_content_addressed_and_complete(self):
        args = SimpleNamespace(study_id="replication-01", task="dspy", round=1)
        entry = {"entry_id": "entry-1", "belief": "b", "correction": "c"}
        calls = [{
            "call_id": "call-1", "owner_id": "q-1", "phase": "distill", "seed": 1,
            "model": "m", "model_revision": "rev", "response_model": "served-model",
            "prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5,
            "usage_reported": True,
            "provider_usage": {"prompt_tokens": 2, "completion_tokens": 3},
        }]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text("{}")
            manifest = _write_note(
                args, root, "exact note", [entry], input_note_sha256="input",
                round_calls=calls, cumulative_calls=calls,
                round_construction_calls=[], cumulative_construction_calls=[],
                corpus_commit="commit", automated_claim_ready=True,
                automated_readiness={"complete": True})
            self.assertEqual(manifest["study_id"], "replication-01")
            self.assertEqual(manifest["task"], "dspy")
            self.assertEqual(manifest["corpus_commit"], "commit")
            self.assertFalse(manifest["claim_ready"])
            self.assertTrue(manifest["automated_claim_ready"])
            self.assertFalse(manifest["publication_claim_ready"])
            self.assertEqual(manifest["human_audit"]["status"], "not_performed")
            self.assertEqual(manifest["entry_ids"], ["entry-1"])
            self.assertEqual(manifest["usage"]["generated_tokens"], 3)
            note_path = root / "notes" / manifest["note_path"]
            self.assertEqual(note_path.read_text(), "exact note")
            self.assertEqual(_load_input_note(root, 2, [entry]), "exact note")
            with self.assertRaises(FileExistsError):
                _write_note(args, root, "drifted note", [entry],
                            input_note_sha256="input", round_calls=calls,
                            cumulative_calls=calls, round_construction_calls=[],
                            cumulative_construction_calls=[], corpus_commit="commit",
                            automated_claim_ready=True,
                            automated_readiness={"complete": True})

    def test_task_manifest_binds_server_transport_retries_and_concurrency(self):
        args = SimpleNamespace(
            study_id="replication-01", task="dspy", round=1, seed=7, smoke=False,
            chapters=4, questions=5, concurrency=11, audit_protocol=None,
        )
        corpus = SimpleNamespace(name="dspy", commit="corpus-commit")
        environment = {"server_count": "2"}
        source = {
            "git_commit": "source-commit", "tree_sha256": "tree", "files": {"f": {}},
            "dirty": False,
        }
        with tempfile.TemporaryDirectory() as directory, \
                patch("studybench.selfquiz.corpus_record", return_value={
                    "name": "dspy", "commit": "corpus-commit", "dirty": False,
                }), \
                patch("studybench.selfquiz.source_record", return_value=source), \
                patch("studybench.selfquiz.environment_record", return_value=environment), \
                patch("studybench.selfquiz._environment_complete", return_value=True):
            manifest, launch_environment = _write_task_manifest(
                args, corpus, Path(directory),
                ["http://localhost:8100/v1", "http://127.0.0.1:8101/v1"],
            )
        self.assertEqual(manifest["server_transport"]["server_count"], 2)
        self.assertEqual(manifest["server_transport"]["scope"], "loopback")
        self.assertEqual(manifest["config"]["concurrency"], 11)
        self.assertEqual(manifest["config"]["provider_retries"], 0)
        self.assertTrue(launch_environment["snapshot"].startswith("r1/environments/"))

    def test_task_manifest_resumes_only_across_compatible_launches(self):
        args = SimpleNamespace(
            study_id="replication-01", task="dspy", round=1, seed=7,
            smoke=False, chapters=4, questions=5, concurrency=11,
            audit_protocol=None,
        )
        corpus = SimpleNamespace(name="dspy", commit="corpus-commit")
        source = {
            "git_commit": "source-commit", "tree_sha256": "tree",
            "files": {"f": {}}, "dirty": False,
        }
        first_environment = {
            "server_count": "2",
            "model_revision": "revision",
            "slurm_job_id": "1",
            "server_launch_id": "a" * 64,
            "vllm_api_key_sha256": "a" * 64,
            "cuda_visible_devices": "0,1",
            "runner_allocation": {"hostname": "first"},
        }
        retry_environment = {
            **first_environment,
            "slurm_job_id": "2",
            "server_launch_id": "b" * 64,
            "vllm_api_key_sha256": "b" * 64,
            "cuda_visible_devices": "6,7",
            "runner_allocation": {"hostname": "retry"},
        }
        with tempfile.TemporaryDirectory() as directory, \
                patch("studybench.selfquiz.corpus_record", return_value={
                    "name": "dspy", "commit": "corpus-commit", "dirty": False,
                }), \
                patch("studybench.selfquiz.source_record", return_value=source), \
                patch("studybench.selfquiz._environment_complete", return_value=True):
            root = Path(directory)
            with patch(
                "studybench.selfquiz.environment_record",
                return_value=first_environment,
            ):
                manifest, first_launch = _write_task_manifest(
                    args,
                    corpus,
                    root,
                    ["http://localhost:8100/v1", "http://localhost:8101/v1"],
                )
            with patch(
                "studybench.selfquiz.environment_record",
                return_value=retry_environment,
            ):
                resumed, retry_launch = _write_task_manifest(
                    args,
                    corpus,
                    root,
                    ["http://localhost:8100/v1", "http://localhost:8101/v1"],
                )
            self.assertEqual(resumed, manifest)
            self.assertEqual(resumed["environment"], first_environment)
            self.assertNotEqual(first_launch, retry_launch)
            self.assertTrue((root / first_launch["snapshot"]).is_file())
            self.assertTrue((root / retry_launch["snapshot"]).is_file())

            drifted = {**retry_environment, "model_revision": "other"}
            with patch(
                "studybench.selfquiz.environment_record", return_value=drifted
            ), self.assertRaisesRegex(SystemExit, "substantive drift"):
                _write_task_manifest(
                    args,
                    corpus,
                    root,
                    ["http://localhost:8100/v1", "http://localhost:8101/v1"],
                )

    def test_model_artifacts_bind_their_own_exact_compatible_launch(self):
        baseline = {
            "model_revision": "revision",
            "slurm_job_id": "1",
            "server_launch_id": "a" * 64,
            "vllm_api_key_sha256": "a" * 64,
            "cuda_visible_devices": "0",
            "runner_allocation": {"hostname": "first"},
        }
        retry = {
            **baseline,
            "slurm_job_id": "2",
            "server_launch_id": "b" * 64,
            "vllm_api_key_sha256": "b" * 64,
            "cuda_visible_devices": "7",
            "runner_allocation": {"hostname": "retry"},
        }
        manifest = {
            "environment": baseline,
            "environment_contract": environment_contract_record(baseline),
            "provenance_readiness": {"environment_complete": False},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_snapshot = write_environment_snapshot(
                root, PurePosixPath("r1/environments"), baseline
            )
            retry_snapshot = write_environment_snapshot(
                root, PurePosixPath("r1/environments"), retry
            )
            first = _bind_launch_environment({"status": "ok"}, first_snapshot)
            resumed = _bind_launch_environment({"status": "ok"}, retry_snapshot)
            _validate_artifact_environment(
                root, manifest, first, label="first artifact"
            )
            _validate_artifact_environment(
                root, manifest, resumed, label="resumed artifact"
            )
            self.assertEqual(
                len(_launch_environment_inventory(root, 1, manifest)), 2
            )

            drifted = {**retry, "model_revision": "other"}
            drifted_snapshot = write_environment_snapshot(
                root, PurePosixPath("r1/environments"), drifted
            )
            with self.assertRaisesRegex(SystemExit, "launch-environment"):
                _validate_artifact_environment(
                    root,
                    manifest,
                    _bind_launch_environment({"status": "ok"}, drifted_snapshot),
                    label="drifted artifact",
                )

    def test_quiz_protocol_rejects_variable_raw_or_valid_question_counts(self):
        args = SimpleNamespace(
            study_id="replication-01", task="dspy", round=1, seed=7,
            smoke=True, questions=5,
        )

        def episode_with(questions):
            def fake_run(chapter, tools, url, n, *, seed, owner_id):
                return {
                    "owner_id": owner_id, "chapter": chapter, "seed": seed,
                    "status": "ok", "questions": questions, "trajectory": {},
                    "trajectory_sha256": sha256_json({}), "calls": [], "usage": usage_totals([]),
                }
            return fake_run

        valid = lambda index: {
            "question": f"Unique question {index}?", "qtype": "behavior",
            "anchors": ["pkg/a.py"], "writer_sketch": "sketch",
        }
        with tempfile.TemporaryDirectory() as directory, \
                patch("studybench.selfquiz.run_quiz", side_effect=episode_with(
                    [valid(0), valid(1)])), \
                patch("studybench.selfquiz._validate_artifact_environment"):
            with self.assertRaisesRegex(SystemExit, "exactly 3 were requested"):
                _prepare_questions(
                    args, FakeRepoTools(), [], ["http://localhost:8100/v1"],
                    Path(directory), ["pkg"], [], [], task_manifest={},
                    launch_environment={"snapshot": "test"},
                )

        invalid = [valid(0), valid(1), valid(2)]
        invalid[2] = {**invalid[2], "anchors": ["missing.py"]}
        with tempfile.TemporaryDirectory() as directory, \
                patch("studybench.selfquiz.run_quiz", side_effect=episode_with(invalid)), \
                patch("studybench.selfquiz._validate_artifact_environment"):
            rdir = Path(directory)
            with self.assertRaisesRegex(SystemExit, "exactly 3 are required"):
                _prepare_questions(
                    args, FakeRepoTools(), [], ["http://localhost:8100/v1"],
                    rdir, ["pkg"], [], [], task_manifest={},
                    launch_environment={"snapshot": "test"},
                )
            self.assertTrue((rdir / "rejected-questions.jsonl").is_file())

    def test_question_provenance_requires_the_complete_planned_chapter_set(self):
        args, records, episodes = question_provenance_fixture(["chapter-a", "chapter-b"])
        _validate_question_provenance(
            args,
            records,
            episodes,
            FakeRepoTools(),
            expected_chapters=["chapter-a", "chapter-b"],
            expected_question_count=3,
        )
        partial_records = [record for record in records
                           if record["chapter"] == "chapter-a"]
        partial_episodes = [episode for episode in episodes
                            if episode["chapter"] == "chapter-a"]
        with self.assertRaisesRegex(SystemExit, "exact chapter plan"):
            _validate_question_provenance(
                args,
                partial_records,
                partial_episodes,
                FakeRepoTools(),
                expected_chapters=["chapter-a", "chapter-b"],
                expected_question_count=3,
            )

    def test_question_provenance_requires_raw_ordinals_zero_through_n_minus_one(self):
        args, records, episodes = question_provenance_fixture(["chapter-a"])
        records[1]["quiz_ordinal"] = 2
        with self.assertRaisesRegex(SystemExit, "raw ordinals"):
            _validate_question_provenance(
                args,
                records,
                episodes,
                FakeRepoTools(),
                expected_chapters=["chapter-a"],
                expected_question_count=3,
            )

    def test_cross_study_exact_reuse_fails_freshness(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "study-selfquiz" / "dspy" / "r1" / "questions.jsonl"
            legacy.parent.mkdir(parents=True)
            legacy.write_text(json.dumps({"question": "How does Foo cache values?"}) + "\n")
            current_dir = root / "study-selfquiz" / "studies" / "new-study" / "dspy"
            record = quiz_item("new")
            record["question"] = "How does Foo cache values?"
            audit = freshness_audit(
                [record], task="dspy", study_dir=current_dir, root=root)
            self.assertFalse(audit["fresh"])
            self.assertEqual(audit["exact_overlaps"], 1)

    def test_freshness_scans_all_legacy_selfquiz_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "study-selfquiz-run1" / "dspy" / "r3" / "questions.jsonl"
            legacy.parent.mkdir(parents=True)
            legacy.write_text(json.dumps({"question": "How does Foo cache values?"}) + "\n")
            current_dir = root / "study-selfquiz" / "studies" / "new-study" / "dspy"
            record = quiz_item("new")
            record["question"] = "How does Foo cache values?"
            audit = freshness_audit(
                [record], task="dspy", study_dir=current_dir, root=root)
            self.assertFalse(audit["fresh"])
            self.assertEqual(
                audit["matches"][0]["prior_path"],
                "study-selfquiz-run1/dspy/r3/questions.jsonl",
            )

    def test_freshness_compares_prior_rounds_in_the_same_study(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current_dir = root / "study-selfquiz" / "studies" / "same-study" / "dspy"
            prior = current_dir / "r1" / "questions.jsonl"
            prior.parent.mkdir(parents=True)
            prior.write_text(json.dumps({"question": "How does Foo cache values?"}) + "\n")
            record = quiz_item("new", round_number=2)
            record["question"] = "How does Foo cache values?"
            audit = freshness_audit(
                [record], task="dspy", study_dir=current_dir, root=root,
                snapshot_dir=current_dir / "r2" / "freshness-sources",
            )
            self.assertFalse(audit["fresh"])
            self.assertEqual(audit["exact_overlaps"], 1)
            self.assertEqual(
                audit["matches"][0]["prior_path"],
                "study-selfquiz/studies/same-study/dspy/r1/questions.jsonl",
            )
            self.assertTrue(freshness_sources_complete(current_dir / "r2", audit))

    def test_distinct_cross_study_questions_are_fresh(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "study-selfquiz" / "dspy" / "r1" / "questions.jsonl"
            legacy.parent.mkdir(parents=True)
            legacy.write_text(json.dumps({"question": "Where is Foo registered?"}) + "\n")
            current_dir = root / "study-selfquiz" / "studies" / "new-study" / "dspy"
            record = quiz_item("new")
            record["question"] = "Why does the optimizer reject an empty metric trace?"
            audit = freshness_audit(
                [record], task="dspy", study_dir=current_dir, root=root)
            self.assertTrue(audit["fresh"])

    def test_human_audit_promotion_is_separate_exact_and_blinded(self):
        args = SimpleNamespace(study_id="replication-01", task="dspy", round=2, seed=7)
        with tempfile.TemporaryDirectory() as directory, \
                patch("studybench.selfquiz.ROOT", Path(directory)):
            sdir, construction, audit = human_audit_fixture(args)
            audit_path = sdir / "audit-result.json"
            audit_path.write_text(json.dumps(audit))

            duplicate_path = sdir / "audit-duplicate.json"
            duplicate_path.write_text(
                json.dumps(audit).replace(
                    '"decision": "pass"',
                    '"decision": "fail", "decision": "pass"',
                )
            )
            with self.assertRaisesRegex(SystemExit, "duplicate key"):
                promote_human_audit(args, duplicate_path)

            artifact_path = sdir / "r1" / "items.jsonl"
            artifact_text = artifact_path.read_text()
            artifact_path.write_text(artifact_text + "tampered\n")
            with self.assertRaisesRegex(SystemExit, "construction artifact"):
                promote_human_audit(args, audit_path)
            artifact_path.write_text(artifact_text)

            construction_record = json.loads(construction.read_text())
            note_path = construction.parent / construction_record["note_path"]
            note_text = note_path.read_text()
            note_path.write_text(note_text + "tampered")
            with self.assertRaisesRegex(SystemExit, "construction artifact|construction note"):
                promote_human_audit(args, audit_path)
            note_path.write_text(note_text)

            promoted_path = promote_human_audit(args, audit_path)
            promoted = json.loads(promoted_path.read_text())
            self.assertTrue(promoted["claim_ready"])
            self.assertEqual(promoted["human_audit"]["status"], "passed")
            self.assertEqual(
                promoted["human_audit"]["auditor_id"], audit["auditor_id"])
            run_root = Path(directory) / "run-snapshot"
            _, note_record = _load_note(
                run_root,
                sdir / "notes" / promoted["note_path"],
                promoted_path,
                require_manifest=True,
                expected_task="dspy",
                expected_corpus_commit="commit",
            )
            self.assertEqual(
                set(note_record["provenance_bundle"]["artifacts"]),
                {"construction_manifest", "human_audit_result", "human_audit_protocol"},
            )
            bundled_dependencies = note_record["provenance_bundle"][
                "construction_artifacts"
            ]["artifacts"]
            self.assertIn("r1/items.jsonl", bundled_dependencies)
            dependency_snapshot = run_root / bundled_dependencies[
                "r1/items.jsonl"
            ]["snapshot"]
            self.assertEqual(
                sha256_file(dependency_snapshot),
                bundled_dependencies["r1/items.jsonl"]["sha256"],
            )

            forged = json.loads(json.dumps(promoted))
            forged["human_audit"]["auditor_id"] = "different-auditor"
            forged_path = sdir / "notes" / "forged-auditor.manifest.json"
            forged_path.write_text(json.dumps(forged), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not bind"):
                _load_note(
                    Path(directory) / "forged-auditor-run",
                    sdir / "notes" / promoted["note_path"],
                    forged_path,
                    require_manifest=True,
                    expected_task="dspy",
                    expected_corpus_commit="commit",
                )

            artifact_path.write_text(artifact_text + "post-promotion tamper\n")
            with self.assertRaisesRegex(ValueError, "construction dependency changed"):
                _load_note(
                    Path(directory) / "tampered-run-snapshot",
                    sdir / "notes" / promoted["note_path"],
                    promoted_path,
                    require_manifest=True,
                    expected_task="dspy",
                    expected_corpus_commit="commit",
                )
            artifact_path.write_text(artifact_text)

            audit["record_reviews"] = audit["record_reviews"][:-1]
            incomplete_path = sdir / "audit-incomplete.json"
            incomplete_path.write_text(json.dumps(audit))
            with self.assertRaises(SystemExit):
                promote_human_audit(args, incomplete_path)

    def test_human_audit_promotion_rejects_unenforced_protocol_clauses(self):
        args = SimpleNamespace(study_id="replication-01", task="dspy", round=1, seed=7)
        extensions = {
            "assignment": {"minimum_reviewers": 2},
            "escalation": "resolve_disagreements",
            "adjudication": {"required": True},
        }
        with tempfile.TemporaryDirectory() as directory, \
                patch("studybench.selfquiz.ROOT", Path(directory)):
            sdir, _, audit = human_audit_fixture(
                args, protocol_extensions=extensions)
            audit_path = sdir / "audit-result.json"
            audit_path.write_text(json.dumps(audit), encoding="utf-8")

            with self.assertRaisesRegex(SystemExit, "unsupported fields"):
                promote_human_audit(args, audit_path)
            self.assertFalse(
                (sdir / "notes" / "note-r1.audited.manifest.json").exists())

    def test_human_audit_integer_fields_reject_booleans_without_archiving(self):
        args = SimpleNamespace(study_id="replication-01", task="dspy", round=1, seed=7)
        with tempfile.TemporaryDirectory() as directory, \
                patch("studybench.selfquiz.ROOT", Path(directory)):
            sdir, _, audit = human_audit_fixture(args)
            for field in ("schema_version", "round"):
                with self.subTest(field=field):
                    candidate = json.loads(json.dumps(audit))
                    candidate[field] = True
                    path = sdir / f"audit-boolean-{field}.json"
                    path.write_text(json.dumps(candidate), encoding="utf-8")
                    with self.assertRaisesRegex(SystemExit, "does not match"):
                        promote_human_audit(args, path)
            self.assertFalse((sdir / "notes" / "audits" / "failed").exists())
            self.assertFalse((sdir / "notes" / "note-r1.audited.manifest.json").exists())

    def test_bound_failed_human_audit_is_archived_without_promotion(self):
        args = SimpleNamespace(study_id="replication-01", task="dspy", round=2, seed=7)
        with tempfile.TemporaryDirectory() as directory, \
                patch("studybench.selfquiz.ROOT", Path(directory)):
            sdir, construction, audit = human_audit_fixture(args)
            failed_root = sdir / "notes" / "audits" / "failed" / "by-sha256"

            inconsistent = json.loads(json.dumps(audit))
            inconsistent["decision"] = "fail"
            inconsistent_path = sdir / "audit-inconsistent-fail.json"
            inconsistent_path.write_text(json.dumps(inconsistent), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "decision must be 'pass'"):
                promote_human_audit(args, inconsistent_path)
            self.assertFalse(failed_root.exists())

            unbound = json.loads(json.dumps(audit))
            unbound["decision"] = "fail"
            unbound["record_reviews"][0]["evidence_valid"] = False
            unbound["construction_manifest_sha256"] = "0" * 64
            unbound_path = sdir / "audit-unbound-fail.json"
            unbound_path.write_text(json.dumps(unbound), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "does not match"):
                promote_human_audit(args, unbound_path)
            self.assertFalse(failed_root.exists())

            failed = json.loads(json.dumps(audit))
            failed["decision"] = "fail"
            failed["record_reviews"][0]["evidence_valid"] = False
            failed_text = json.dumps(failed, indent=2)
            failed_path = sdir / "audit-valid-fail.json"
            failed_path.write_text(failed_text, encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "archived exact result"):
                promote_human_audit(args, failed_path)

            archived = failed_root / f"{sha256_text(failed_text)}.json"
            self.assertEqual(archived.read_text(encoding="utf-8"), failed_text)
            self.assertFalse((sdir / "notes" / "note-r2.audited.manifest.json").exists())
            self.assertFalse(json.loads(construction.read_text())["claim_ready"])

            # Retrying the same exact failed result is idempotent and still cannot promote.
            with self.assertRaisesRegex(SystemExit, "archived exact result"):
                promote_human_audit(args, failed_path)
            self.assertEqual(archived.read_text(encoding="utf-8"), failed_text)

            later_pass_path = sdir / "audit-later-pass.json"
            later_pass_path.write_text(json.dumps(audit), encoding="utf-8")
            with self.assertRaisesRegex(
                    SystemExit, "previously archived valid failing human audit"):
                promote_human_audit(args, later_pass_path)
            self.assertFalse(
                (sdir / "notes" / "note-r2.audited.manifest.json").exists())

    def test_invalid_and_unrelated_failures_do_not_block_but_symlinks_fail_closed(self):
        args = SimpleNamespace(study_id="replication-01", task="dspy", round=1, seed=7)
        with tempfile.TemporaryDirectory() as directory, \
                patch("studybench.selfquiz.ROOT", Path(directory)):
            sdir, _, audit = human_audit_fixture(args)
            failed_root = sdir / "notes" / "audits" / "failed" / "by-sha256"
            failed_root.mkdir(parents=True)

            invalid_bytes = b'{"decision":"fail","decision":"pass"}\n'
            (failed_root / f"{sha256_text(invalid_bytes.decode())}.json").write_bytes(
                invalid_bytes)
            unrelated = json.loads(json.dumps(audit))
            unrelated["study_id"] = "different-study"
            unrelated["decision"] = "fail"
            unrelated["record_reviews"][0]["evidence_valid"] = False
            unrelated_text = json.dumps(unrelated)
            (failed_root / f"{sha256_text(unrelated_text)}.json").write_text(
                unrelated_text, encoding="utf-8")

            audit_path = sdir / "audit-pass.json"
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            promoted = promote_human_audit(args, audit_path)
            self.assertTrue(promoted.exists())

        with tempfile.TemporaryDirectory() as directory, \
                patch("studybench.selfquiz.ROOT", Path(directory)):
            sdir, _, audit = human_audit_fixture(args)
            failed_root = sdir / "notes" / "audits" / "failed" / "by-sha256"
            failed_root.mkdir(parents=True)
            target = sdir / "unrelated.json"
            target.write_text("{}", encoding="utf-8")
            (failed_root / f"{'0' * 64}.json").symlink_to(target)
            audit_path = sdir / "audit-pass.json"
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "must not contain symlinks"):
                promote_human_audit(args, audit_path)
            self.assertFalse(
                (sdir / "notes" / "note-r1.audited.manifest.json").exists())

    def test_human_audit_protocol_rejects_ambiguous_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            path.write_text(
                '{"schema_version":1,"protocol_id":"blind-audit-01",'
                '"blinding":"wrong",'
                '"blinding":"condition_and_method_labels_hidden",'
                '"population":"all_train_dev_verdicts_and_admitted_entries",'
                f'"decision_rule":"{HUMAN_AUDIT_DECISION_RULE_ID}"}}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SystemExit, "duplicate key"):
                validate_audit_protocol(path)

            boolean_version = Path(directory) / "boolean-version.json"
            boolean_version.write_text(json.dumps({
                "schema_version": True,
                "protocol_id": "blind-audit-01",
                "blinding": "condition_and_method_labels_hidden",
                "population": "all_train_dev_verdicts_and_admitted_entries",
                "decision_rule": HUMAN_AUDIT_DECISION_RULE_ID,
            }), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "schema_version=1"):
                validate_audit_protocol(boolean_version)

            wrong_rule = Path(directory) / "wrong-rule.json"
            wrong_rule.write_text(json.dumps({
                "schema_version": 1,
                "protocol_id": "blind-audit-01",
                "blinding": "condition_and_method_labels_hidden",
                "population": "all_train_dev_verdicts_and_admitted_entries",
                "decision_rule": "always-pass",
            }), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "decision_rule"):
                validate_audit_protocol(wrong_rule)

    def test_promoted_auditor_identity_must_match_validated_result(self):
        audit = {"auditor_id": "auditor-01"}
        promoted = {"human_audit": {"auditor_id": "auditor-01"}}
        _validate_promoted_auditor_identity(promoted, audit)
        for candidate in (
            {"human_audit": {"auditor_id": "different-auditor"}},
            {"human_audit": {"auditor_id": "../unsafe"}},
        ):
            with self.subTest(candidate=candidate), self.assertRaisesRegex(
                SystemExit, "auditor identity"
            ):
                _validate_promoted_auditor_identity(candidate, audit)


class TrajectoryAndUsageTests(unittest.TestCase):
    def test_reference_support_rejects_truthy_non_boolean_outputs(self):
        derivation_prediction = SimpleNamespace(
            answer="The value is returned.",
            evidence=[Evidence(file="pkg/a.py", line=3, quote="return value")],
            trajectory={},
        )
        item = {"question": "What is returned?", "_rt": FakeRepoTools()}
        for malformed in ("false", 1):
            with self.subTest(malformed=malformed), patch(
                "studybench.selfquiz.fresh_lm",
                side_effect=[SimpleNamespace(history=[]), SimpleNamespace(history=[])],
            ), patch(
                "studybench.selfquiz.dspy.context",
                side_effect=lambda **_kwargs: nullcontext(),
            ), patch(
                "studybench.selfquiz.dspy.ReAct",
                return_value=lambda **_kwargs: derivation_prediction,
            ), patch(
                "studybench.selfquiz.dspy.Predict",
                return_value=lambda **_kwargs: SimpleNamespace(
                    supported=malformed, rationale="malformed"
                ),
            ):
                result, _calls = _derive(
                    item, [], "unused", seed=11, owner_id="owner", index=0)
            self.assertEqual(result["status"], "invalid")
            self.assertEqual(result["reference_support"]["status"], "error")
            self.assertIs(result["reference_support"]["supported"], False)
            self.assertIn("exact boolean", result["reference_support"]["error"])

    def test_trajectory_serialization_preserves_all_tool_turns(self):
        trajectory = {
            "tool_name_0": "read_file",
            "tool_args_0": {"path": "pkg/a.py"},
            "observation_0": "source",
            "tool_name_1": "read_file",
            "tool_args_1": {"path": "pkg/b.py"},
            "observation_1": "more source",
        }
        serialized = serialize_trajectory(trajectory)
        self.assertEqual(serialized["tool_args_0"], {"path": "pkg/a.py"})
        self.assertEqual(serialized["observation_1"], "more source")
        self.assertTrue(_trajectory_hash_valid({
            "trajectory": serialized,
            "trajectory_sha256": sha256_json(serialized),
        }))
        self.assertFalse(_trajectory_hash_valid({
            "trajectory": serialized,
            "trajectory_sha256": sha256_json({}),
        }))

    def test_usage_ledger_preserves_raw_usage_and_exact_totals(self):
        lm = SimpleNamespace(history=[
            {"usage": {"prompt_tokens": 10, "completion_tokens": 3,
                       "total_tokens": 13, "cached_tokens": 2},
             "response_model": "served-model",
             "response": SimpleNamespace(id="response-1", system_fingerprint="fp"),
             "messages": [{"role": "user", "content": "one"}],
             "outputs": ["answer one"]},
            {"usage": {"prompt_tokens": 20, "completion_tokens": 5},
             "response_model": "served-model",
             "response": SimpleNamespace(id="response-2", system_fingerprint=None),
             "messages": [{"role": "user", "content": "two"}],
             "outputs": ["answer two"]},
        ])
        records = usage_records(lm, phase="quiz", owner_id="owner", seed=11)
        self.assertEqual(usage_totals(records), {
            "calls": 2, "prompt_tokens": 30, "generated_tokens": 8, "total_tokens": 38})
        self.assertEqual(records[0]["provider_usage"]["cached_tokens"], 2)
        self.assertEqual(records[0]["response_model"], "served-model")
        self.assertEqual(records[0]["response_id"], "response-1")
        self.assertEqual(records[0]["request_messages_sha256"], sha256_json(
            [{"role": "user", "content": "one"}]))
        self.assertNotEqual(records[0]["call_id"], records[1]["call_id"])
        self.assertEqual(usage_by_phase(records)["quiz"]["calls"], 2)

    def test_usage_audit_rejects_missing_duplicate_or_unreported_calls(self):
        lm = SimpleNamespace(history=[{
            "usage": {"prompt_tokens": 2, "completion_tokens": 3},
            "response_model": "served-model",
            "response": SimpleNamespace(id="response-1", system_fingerprint=None),
            "messages": [{"role": "user", "content": "question"}],
            "outputs": ["answer"],
        }])
        calls = usage_records(lm, phase="quiz", owner_id="owner", seed=11)
        self.assertTrue(usage_ledger_audit(calls, calls)["complete"])
        self.assertFalse(usage_ledger_audit(calls, [])["complete"])
        self.assertFalse(usage_ledger_audit(calls, calls + calls)["complete"])
        unreported = [dict(calls[0], usage_reported=False)]
        self.assertFalse(usage_ledger_audit(unreported, unreported)["complete"])
        unidentified = [dict(calls[0], response_id=None)]
        self.assertFalse(usage_ledger_audit(unidentified, unidentified)["complete"])
        unhashed = [dict(calls[0], request_messages_available=False)]
        self.assertFalse(usage_ledger_audit(unhashed, unhashed)["complete"])

    def test_dev_reference_requires_distinct_derivations_and_reciprocal_consensus(self):
        item = quiz_item("dev", split="dev")
        record = dev_reference_fixture(item)
        _validate_dev_reference(
            item,
            record,
            study_id="study-a",
            task="dspy",
            master_seed=7,
            rt=FakeRepoTools(),
        )

        duplicated = json.loads(json.dumps(record))
        duplicated["references"] = [
            duplicated["derivations"][0], duplicated["derivations"][0]]
        duplicated["reference_ids"] = [
            duplicated["derivations"][0]["derivation_id"],
            duplicated["derivations"][0]["derivation_id"],
        ]
        with self.assertRaisesRegex(SystemExit, "consensus reference"):
            _validate_dev_reference(
                item,
                duplicated,
                study_id="study-a",
                task="dspy",
                master_seed=7,
                rt=FakeRepoTools(),
            )

        nonreciprocal = json.loads(json.dumps(record))
        nonreciprocal["reference_consensus"][1]["candidate_id"] = \
            nonreciprocal["derivations"][1]["derivation_id"]
        with self.assertRaisesRegex(SystemExit, "drifted adjudication"):
            _validate_dev_reference(
                item,
                nonreciprocal,
                study_id="study-a",
                task="dspy",
                master_seed=7,
                rt=FakeRepoTools(),
            )

        float_seed = json.loads(json.dumps(record))
        float_seed["derivations"][0]["seed"] = float(
            float_seed["derivations"][0]["seed"])
        with self.assertRaisesRegex(SystemExit, "drifted derivation"):
            _validate_dev_reference(
                item,
                float_seed,
                study_id="study-a",
                task="dspy",
                master_seed=7,
                rt=FakeRepoTools(),
            )

    def test_dev_arms_share_signature_seed_and_fixed_reference(self):
        item = quiz_item("dev", split="dev")
        reference = {
            "reference_id": "dev-reference-fixed",
            "references": [
                {"derivation_id": "r1", "answer": "a", "evidence": []},
                {"derivation_id": "r2", "answer": "a", "evidence": []},
            ],
            "status": "ok",
        }
        attempts = []
        judges = []

        def fake_attempt(question, note, url, *, seed, owner_id, phase):
            attempts.append((question, note, seed, owner_id, phase))
            return phase, model_calls(owner_id, phase, seed), None

        def fake_judge(*args, **kwargs):
            judges.append(kwargs)
            owner_id = kwargs["owner_id"]
            calls = []
            checks = [
                {
                    "status": "ok",
                    "verdict": "correct",
                    "delta": "",
                    "seed": stable_seed(
                        7, kwargs["seed_namespace"], kwargs["seed_phase"], index),
                    "reference_id": candidate["derivation_id"],
                    "seed_phase": kwargs["seed_phase"],
                    "audit_phase": f"{kwargs['phase']}-{index}",
                }
                for index, candidate in enumerate(args[2])
            ]
            for index, check in enumerate(checks):
                calls += model_calls(owner_id, check["audit_phase"], check["seed"])
            return "correct", "", checks, calls

        with patch("studybench.selfquiz._attempt", side_effect=fake_attempt), \
                patch("studybench.selfquiz._judge_attempt", side_effect=fake_judge):
            record = run_dev_item(
                item, "the note", reference, "unused", master_seed=7, exam_round=2)
        self.assertEqual(
            {attempt[2] for attempt in attempts},
            {record["attempt_protocol"]["paired_seed"]},
        )
        self.assertEqual([attempt[1] for attempt in attempts], ["the note", ""])
        self.assertEqual(record["attempt_protocol"]["signature"], "note, question -> answer")
        self.assertEqual(record["reference_sha256"], sha256_json(reference))
        self.assertEqual({judge["seed_phase"] for judge in judges}, {
            "dev-paired-adjudication"
        })
        self.assertEqual(
            {judge["phase"] for judge in judges},
            {"dev-adjudication-with_note", "dev-adjudication-bare"},
        )
        self.assertEqual(
            record["adjudication_protocol"]["paired_seeds"],
            [
                stable_seed(7, "dev-reference-fixed", "dev-paired-adjudication", index)
                for index in range(2)
            ],
        )
        self.assertEqual(
            record["adjudication_protocol"]["only_manipulated_field"], "attempt"
        )
        _validate_dev_exam(
            item,
            record,
            reference,
            note_sha256=sha256_text("the note"),
            master_seed=7,
            exam_round=2,
        )

        float_seed = json.loads(json.dumps(record))
        float_seed["attempts"]["bare"]["seed"] = float(
            float_seed["attempts"]["bare"]["seed"])
        with self.assertRaisesRegex(SystemExit, "attempt lineage"):
            _validate_dev_exam(
                item,
                float_seed,
                reference,
                note_sha256=sha256_text("the note"),
                master_seed=7,
                exam_round=2,
            )

        fabricated_verdict = json.loads(json.dumps(record))
        fabricated_verdict["verdicts"]["bare"] = "wrong"
        with self.assertRaisesRegex(SystemExit, "verdicts drifted"):
            _validate_dev_exam(
                item,
                fabricated_verdict,
                reference,
                note_sha256=sha256_text("the note"),
                master_seed=7,
                exam_round=2,
            )

        record["adjudications"]["bare"][0]["seed"] += 1
        with self.assertRaisesRegex(SystemExit, "adjudication"):
            _validate_dev_exam(
                item,
                record,
                reference,
                note_sha256=sha256_text("the note"),
                master_seed=7,
                exam_round=2,
            )

    def test_empty_metrics_are_null_not_fake_zero(self):
        self.assertEqual(_error_rate([]), {"resolved": 0, "errors": 0, "error_rate": None})
        self.assertEqual(_error_rate(["correct", "wrong", "unresolved"]),
                         {"resolved": 2, "errors": 1, "error_rate": 0.5})


if __name__ == "__main__":
    unittest.main()
