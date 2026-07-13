import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from studybench.env import load_private_env


class PrivateEnvTests(unittest.TestCase):
    def test_missing_file_is_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            load_private_env(Path(directory) / "missing")

    def test_loads_without_overwriting_and_never_evaluates_shell(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("# comment\nSB_ENV_TEST=new\nLITERAL=$(not-a-command)\n")
            path.chmod(0o600)
            with patch.dict(os.environ, {"SB_ENV_TEST": "old"}, clear=False):
                load_private_env(path)
                self.assertEqual(os.environ["SB_ENV_TEST"], "old")
                self.assertEqual(os.environ["LITERAL"], "$(not-a-command)")
                del os.environ["LITERAL"]

    def test_rejects_broad_permissions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("KEY=value\n")
            path.chmod(0o644)
            with self.assertRaisesRegex(RuntimeError, "mode 0600"):
                load_private_env(path)

    def test_rejects_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "target"
            target.write_text("KEY=value\n")
            target.chmod(0o600)
            link = Path(directory) / ".env"
            link.symlink_to(target)
            with self.assertRaisesRegex(RuntimeError, "unsafe environment file"):
                load_private_env(link)

    def test_fails_closed_without_nofollow_primitive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("SB_ENV_PRIMITIVE_TEST=secret\n", encoding="utf-8")
            path.chmod(0o600)
            for unavailable in (None, 0):
                with self.subTest(unavailable=unavailable), patch.object(
                    os, "O_NOFOLLOW", unavailable
                ), patch.dict(os.environ, {}, clear=True):
                    with self.assertRaisesRegex(
                        RuntimeError, "requires the POSIX O_NOFOLLOW primitive"
                    ):
                        load_private_env(path)
                    self.assertNotIn("SB_ENV_PRIMITIVE_TEST", os.environ)

    def test_rejects_partial_file_before_mutating_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("FIRST=secret\nthis is invalid\n")
            path.chmod(0o600)
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "invalid environment entry"):
                    load_private_env(path)
                self.assertNotIn("FIRST", os.environ)

    def test_rejects_duplicate_keys_before_mutating_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("FIRST=one\nFIRST=two\n", encoding="utf-8")
            path.chmod(0o600)
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "duplicate"):
                    load_private_env(path)
                self.assertNotIn("FIRST", os.environ)

    def test_rejects_nul_before_mutating_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_bytes(b"FIRST=one\nSECOND=bad\x00value\n")
            path.chmod(0o600)
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "NUL"):
                    load_private_env(path)
                self.assertNotIn("FIRST", os.environ)


if __name__ == "__main__":
    unittest.main()
