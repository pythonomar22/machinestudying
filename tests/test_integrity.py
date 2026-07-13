from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import studybench.integrity as integrity

from studybench.integrity import (
    assert_record,
    atomic_write_text,
    atomic_write_json,
    canonical_json_bytes,
    exclusive_process_lock,
    file_record,
    load_json_artifact,
    read_artifact_bytes,
    read_artifact_bytes_with_mode,
    sha256_file,
    sha256_text,
    stable_seed,
    write_immutable_json,
    write_immutable_text,
)


class IntegrityTests(unittest.TestCase):
    def test_process_lock_rejects_overlapping_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "locks" / "episode.lock"
            with exclusive_process_lock(path):
                with self.assertRaisesRegex(RuntimeError, "already working"):
                    with exclusive_process_lock(path):
                        self.fail("overlapping lock unexpectedly succeeded")
            with exclusive_process_lock(path):
                self.assertTrue(path.is_file())

    def test_process_lock_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.write_text("not a lock", encoding="utf-8")
            path = root / "lock"
            path.symlink_to(target)
            with self.assertRaises(OSError):
                with exclusive_process_lock(path):
                    self.fail("symlink lock unexpectedly succeeded")

    def test_process_lock_retains_owner_only_mode_requirement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lock"
            path.touch(mode=0o600)
            path.chmod(0o640)
            with self.assertRaisesRegex(RuntimeError, "mode 0600"):
                with exclusive_process_lock(path):
                    self.fail("wrong-mode lock unexpectedly succeeded")

    def test_json_is_canonical_and_rejects_nonfinite_values(self) -> None:
        self.assertEqual(canonical_json_bytes({"b": 1, "a": "é"}), b'{"a":"\xc3\xa9","b":1}\n')
        with self.assertRaises(ValueError):
            canonical_json_bytes({"bad": float("nan")})

    def test_strict_artifact_json_rejects_duplicates_and_nonfinite_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            for invalid in ('{"same":1,"same":2}', '{"bad":NaN}', b"\xff"):
                with self.subTest(invalid=invalid):
                    path.write_bytes(invalid if isinstance(invalid, bytes) else invalid.encode())
                    with self.assertRaises(ValueError):
                        load_json_artifact(path)

    def test_seed_is_stable_across_processes_and_sensitive_to_parts(self) -> None:
        expected = stable_seed(17, "dspy", 2, "derive", 4)
        code = "from studybench.integrity import stable_seed; print(stable_seed(17, 'dspy', 2, 'derive', 4))"
        observed = int(subprocess.check_output([sys.executable, "-c", code], text=True).strip())
        self.assertEqual(observed, expected)
        self.assertNotEqual(stable_seed(17, "dspy", 3, "derive", 4), expected)

    def test_atomic_json_replaces_mutable_state_canonically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            atomic_write_json(path, {"z": 1, "a": 2})
            self.assertEqual(json.loads(path.read_text()), {"a": 2, "z": 1})
            atomic_write_json(path, {"next": True})
            self.assertEqual(json.loads(path.read_text()), {"next": True})

    def test_immutable_writers_accept_identity_and_reject_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.txt"
            write_immutable_text(path, "same\n")
            write_immutable_text(path, "same\n")
            with self.assertRaises(FileExistsError):
                write_immutable_text(path, "different\n")

            manifest = Path(directory) / "manifest.json"
            write_immutable_json(manifest, {"hash": sha256_text("same\n")})
            write_immutable_json(manifest, {"hash": sha256_text("same\n")})
            with self.assertRaises(FileExistsError):
                write_immutable_json(manifest, {"hash": "wrong"})

    def test_artifact_writers_and_records_reject_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.txt"
            target.write_text("same\n", encoding="utf-8")
            linked = root / "linked.txt"
            linked.symlink_to(target)
            with self.assertRaises(FileExistsError):
                write_immutable_text(linked, "same\n")
            with self.assertRaises(ValueError):
                atomic_write_json(linked, {"replacement": True})
            with self.assertRaises(ValueError):
                file_record(linked)
            with self.assertRaises(ValueError):
                sha256_file(linked)

            real_parent = root / "real-parent"
            real_parent.mkdir()
            linked_parent = root / "linked-parent"
            linked_parent.symlink_to(real_parent, target_is_directory=True)
            with self.assertRaises(ValueError):
                write_immutable_text(linked_parent / "nested" / "new.txt", "content")
            nested = real_parent / "nested.txt"
            nested.write_text("content", encoding="utf-8")
            with self.assertRaises(ValueError):
                sha256_file(linked_parent / "nested.txt")
            self.assertFalse((real_parent / "nested").exists())

            dangling = root / "dangling"
            dangling.symlink_to(root / "missing")
            with self.assertRaises(FileExistsError):
                write_immutable_text(dangling, "content")

    def test_artifact_io_rejects_non_regular_leaf_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            directory_leaf = root / "artifact"
            directory_leaf.mkdir()
            with self.assertRaisesRegex(ValueError, "not a regular file"):
                read_artifact_bytes(directory_leaf)
            with self.assertRaisesRegex(ValueError, "non-regular artifact"):
                atomic_write_text(directory_leaf, "replacement")
            with self.assertRaises(FileExistsError):
                write_immutable_text(directory_leaf, "replacement")

            fifo = root / "fifo"
            os.mkfifo(fifo)
            with self.assertRaisesRegex(ValueError, "not a regular file"):
                read_artifact_bytes(fifo)

    def test_bytes_and_mode_come_from_one_stable_file_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "executable"
            path.write_bytes(b"exact")
            path.chmod(0o751)
            data, mode = read_artifact_bytes_with_mode(path)
            self.assertEqual(data, b"exact")
            self.assertEqual(mode, 0o751)

    def test_read_is_anchored_when_parent_path_is_swapped_for_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "parent"
            parent.mkdir()
            path = parent / "artifact.txt"
            path.write_bytes(b"validated")
            attacker = root / "attacker"
            attacker.mkdir()
            (attacker / path.name).write_bytes(b"redirected")
            detached = root / "detached"
            original_read = integrity._read_regular_bytes_at

            def swap_then_read(parent_fd: int, leaf: str, display: Path) -> bytes:
                parent.rename(detached)
                parent.symlink_to(attacker, target_is_directory=True)
                return original_read(parent_fd, leaf, display)

            with patch.object(
                integrity, "_read_regular_bytes_at", side_effect=swap_then_read
            ):
                self.assertEqual(read_artifact_bytes(path), b"validated")

    def test_atomic_write_is_anchored_when_parent_is_swapped_for_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "parent"
            parent.mkdir()
            attacker = root / "attacker"
            attacker.mkdir()
            detached = root / "detached"
            path = parent / "artifact.txt"
            original_replace = os.replace

            def swap_then_replace(
                source: str,
                destination: str,
                *,
                src_dir_fd: int | None = None,
                dst_dir_fd: int | None = None,
            ) -> None:
                parent.rename(detached)
                parent.symlink_to(attacker, target_is_directory=True)
                original_replace(
                    source,
                    destination,
                    src_dir_fd=src_dir_fd,
                    dst_dir_fd=dst_dir_fd,
                )

            with patch.object(integrity.os, "replace", side_effect=swap_then_replace):
                atomic_write_text(path, "validated")

            self.assertEqual((detached / path.name).read_text(), "validated")
            self.assertFalse((attacker / path.name).exists())

    def test_immutable_write_is_anchored_when_parent_is_swapped_for_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "parent"
            parent.mkdir()
            attacker = root / "attacker"
            attacker.mkdir()
            detached = root / "detached"
            path = parent / "artifact.txt"
            original_link = os.link

            def swap_then_link(
                source: str,
                destination: str,
                *,
                src_dir_fd: int | None = None,
                dst_dir_fd: int | None = None,
                follow_symlinks: bool = True,
            ) -> None:
                parent.rename(detached)
                parent.symlink_to(attacker, target_is_directory=True)
                original_link(
                    source,
                    destination,
                    src_dir_fd=src_dir_fd,
                    dst_dir_fd=dst_dir_fd,
                    follow_symlinks=follow_symlinks,
                )

            with patch.object(integrity.os, "link", side_effect=swap_then_link):
                write_immutable_text(path, "validated")

            self.assertEqual((detached / path.name).read_text(), "validated")
            self.assertFalse((attacker / path.name).exists())

    def test_file_records_detect_byte_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "note.md"
            path.write_text("v1", encoding="utf-8")
            record = file_record(path)
            assert_record(path, record)
            path.write_text("v2", encoding="utf-8")
            with self.assertRaises(ValueError):
                assert_record(path, record)


if __name__ == "__main__":
    unittest.main()
