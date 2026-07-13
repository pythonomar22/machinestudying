from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from studybench.dataset import (
    CORPORA,
    Corpus,
    load_questions,
    read_pinned_code_bytes,
    validate_corpus_snapshot,
    validate_questions,
)
from studybench.tools import MAX_LINE_NUMBER, RepoTools


class CorpusFixture(unittest.TestCase):
    def make_corpus(
        self,
        files: dict[str, str | bytes],
        *,
        symlinks: dict[str, Path] | None = None,
        object_format: str = "sha1",
    ) -> Corpus:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        repo = Path(temporary.name) / "repo"
        repo.mkdir()
        init = ["git", "init", "-q"]
        if object_format != "sha1":
            init.append(f"--object-format={object_format}")
        subprocess.run([*init, str(repo)], check=True)
        for relative, content in files.items():
            path = repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                path.write_bytes(content)
            else:
                path.write_text(content, encoding="utf-8")
        for relative, target in (symlinks or {}).items():
            path = repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.symlink_to(target)
        subprocess.run(["git", "-C", str(repo), "add", "--all"], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "-c",
                "user.name=StudyBench Test",
                "-c",
                "user.email=studybench@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "-qm",
                "fixture",
            ],
            check=True,
        )
        commit = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        return Corpus(
            name="fixture",
            display="Fixture",
            repo=repo,
            roots=("src",),
            language="python",
            commit=commit,
            code_suffixes=(".py",),
            dataset_sha256="",
            question_count=1,
        )


class DatasetIntegrityTests(CorpusFixture):
    def valid_record(self) -> dict:
        return {
            "id": "fixture_q1",
            "topic": "assignment",
            "question": "What is assigned?",
            "gold_answer": "The value one.",
            "rubric": [
                {
                    "claim_id": "c1",
                    "claim_type": "core",
                    "weight": 100,
                    "statement": "The answer identifies the assignment.",
                    "span_ids": ["s1"],
                }
            ],
            "evidence": [
                {
                    "span_id": "s1",
                    "path": "src/example.py",
                    "start_line": 1,
                    "end_line": 1,
                    "excerpt": "0001: value = 1",
                }
            ],
        }

    def write_dataset(self, record: dict) -> tuple[Path, str]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        raw = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
        path = Path(temporary.name) / "questions.jsonl"
        path.write_bytes(raw)
        return path, hashlib.sha256(raw).hexdigest()

    def test_checked_in_bundles_match_pinned_sources_exactly(self) -> None:
        self.assertEqual(len(load_questions("dspy")), CORPORA["dspy"].question_count)
        self.assertEqual(
            len(load_questions("openclaw")), CORPORA["openclaw"].question_count
        )

    def test_valid_bundle_checks_hash_schema_and_exact_excerpt(self) -> None:
        corpus = self.make_corpus({"src/example.py": "value = 1\nother = 2\n"})
        path, digest = self.write_dataset(self.valid_record())
        rows = validate_questions(
            corpus, path, expected_sha256=digest, expected_count=1
        )
        self.assertEqual(rows[0]["id"], "fixture_q1")

    def test_bundle_rejects_excerpt_hash_and_schema_drift(self) -> None:
        corpus = self.make_corpus({"src/example.py": "value = 1\n"})
        record = self.valid_record()
        record["evidence"][0]["excerpt"] = "0001: value = 2"
        path, digest = self.write_dataset(record)
        with self.assertRaisesRegex(ValueError, "excerpt does not match"):
            validate_questions(corpus, path, expected_sha256=digest, expected_count=1)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_questions(corpus, path, expected_sha256="0" * 64, expected_count=1)

        record = self.valid_record()
        record["unreviewed"] = True
        path, digest = self.write_dataset(record)
        with self.assertRaisesRegex(ValueError, "fields"):
            validate_questions(corpus, path, expected_sha256=digest, expected_count=1)

    def test_unknown_task_cannot_select_an_arbitrary_file(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown StudyBench task"):
            load_questions("../../.env")


class RepositoryBoundaryTests(CorpusFixture):
    def test_snapshot_must_be_exact_and_clean(self) -> None:
        corpus = self.make_corpus({"src/example.py": "value = 1\n"})
        validate_corpus_snapshot(corpus)
        (corpus.repo / "src/example.py").write_text("value = 2\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "dirty"):
            validate_corpus_snapshot(corpus)

    def test_git_assume_unchanged_cannot_hide_corpus_byte_drift(self) -> None:
        corpus = self.make_corpus({"src/example.py": "value = 1\n"})
        subprocess.run(
            [
                "git", "-C", str(corpus.repo), "update-index",
                "--assume-unchanged", "src/example.py",
            ],
            check=True,
        )
        (corpus.repo / "src/example.py").write_text("value = 2\n", encoding="utf-8")
        status = subprocess.check_output(
            ["git", "-C", str(corpus.repo), "status", "--porcelain"], text=True
        )
        self.assertEqual(status, "")
        with self.assertRaisesRegex(ValueError, "hidden index flags"):
            validate_corpus_snapshot(corpus)

    def test_git_skip_worktree_cannot_hide_corpus_byte_drift(self) -> None:
        corpus = self.make_corpus({"src/example.py": "value = 1\n"})
        subprocess.run(
            [
                "git", "-C", str(corpus.repo), "update-index",
                "--skip-worktree", "src/example.py",
            ],
            check=True,
        )
        (corpus.repo / "src/example.py").write_text("value = 2\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hidden index flags"):
            validate_corpus_snapshot(corpus)

    def test_each_code_read_rechecks_the_exact_pinned_blob(self) -> None:
        corpus = self.make_corpus({"src/example.py": "value = 1\n"})
        validate_corpus_snapshot(corpus)
        (corpus.repo / "src/example.py").write_text("value = 2\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "differs from pinned commit"):
            read_pinned_code_bytes(corpus, "src/example.py")

    def test_each_code_read_binds_executable_mode_to_the_same_file(self) -> None:
        corpus = self.make_corpus({"src/example.py": "value = 1\n"})
        (corpus.repo / "src/example.py").chmod(0o755)
        with self.assertRaisesRegex(ValueError, "mode differs from pinned commit"):
            read_pinned_code_bytes(corpus, "src/example.py")

    def test_sha256_git_object_format_is_verified_exactly(self) -> None:
        try:
            corpus = self.make_corpus(
                {"src/example.py": "value = 1\n"}, object_format="sha256"
            )
        except subprocess.CalledProcessError:
            self.skipTest("installed Git does not support SHA-256 repositories")
        self.assertEqual(
            read_pinned_code_bytes(corpus, "src/example.py"), b"value = 1\n"
        )

    def test_only_allowed_code_suffixes_are_exposed(self) -> None:
        corpus = self.make_corpus(
            {
                "src/example.py": "value = 1\n",
                "src/notes.md": "not benchmark code\n",
                ".gitignore": "*.ignored.py\n",
                "src/generated.ignored.py": "not_pinned = True\n",
            }
        )
        tools = RepoTools(corpus)
        self.assertEqual(tools.files, ["src/example.py"])

    def test_allowed_code_is_not_silently_dropped_for_size(self) -> None:
        corpus = self.make_corpus({"src/large.py": "x" * 5_000_001})
        tools = RepoTools(corpus)
        self.assertEqual(tools.files, ["src/large.py"])
        self.assertEqual(len(tools.text["src/large.py"]), 5_000_001)

    def test_escaping_symlink_and_invalid_utf8_fail_closed(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        outside = Path(temporary.name) / "outside.py"
        outside.write_text("secret = True\n", encoding="utf-8")
        corpus = self.make_corpus(
            {"src/good.py": "safe = True\n"},
            symlinks={"src/escape.py": outside},
        )
        with self.assertRaisesRegex(ValueError, "escapes"):
            RepoTools(corpus)

        corpus = self.make_corpus({"src/bad.py": b"\xff\xfe"})
        with self.assertRaisesRegex(ValueError, "valid UTF-8"):
            RepoTools(corpus)

        corpus = self.make_corpus(
            {"src/example.py": "safe = True\n"},
            symlinks={"src/alias.py": Path("example.py")},
        )
        with self.assertRaisesRegex(ValueError, "escapes"):
            RepoTools(corpus)

    def test_tool_arguments_preserve_integer_and_path_boundaries(self) -> None:
        corpus = self.make_corpus({"src/example.py": "first\nsecond\n"})
        for invalid_cap in (True, 0, 1.5):
            with self.subTest(read_max_lines=invalid_cap), self.assertRaises(ValueError):
                RepoTools(corpus, read_max_lines=invalid_cap)
        tools = RepoTools(corpus)
        for value in (True, "1", 1.0):
            with self.subTest(value=value):
                output = tools.dispatch(
                    "read_file",
                    json.dumps({"path": "src/example.py", "start_line": value}),
                )
                self.assertIn("must be a positive", output)
        output = tools.dispatch(
            "read_file",
            json.dumps(
                {"path": "src/example.py", "start_line": MAX_LINE_NUMBER + 1}
            ),
        )
        self.assertIn("no greater than", output)
        self.assertIn("non-empty", tools.dispatch("glob", json.dumps({"pattern": ""})))
        self.assertIn(
            "normalized and relative",
            tools.dispatch("read_file", json.dumps({"path": "src//example.py"})),
        )
        self.assertIn(
            "normalized and relative",
            tools.dispatch("read_file", json.dumps({"path": "../outside.py"})),
        )
        self.assertIn(
            "duplicate tool argument",
            tools.dispatch(
                "read_file", '{"path":"src/example.py","path":"src/example.py"}'
            ),
        )
        self.assertIn(
            "non-finite tool argument",
            tools.dispatch("read_file", '{"path":"src/example.py","start_line":NaN}'),
        )


if __name__ == "__main__":
    unittest.main()
