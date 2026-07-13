from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
import unittest

from studybench.model_cache import (
    ATTESTATION_POLICY,
    ModelCacheIntegrityError,
    build_model_cache_inventory,
    canonical_json_bytes,
    verify_model_cache_inventory,
)


MODEL = "Example/Model"
REVISION = "1" * 40


class ModelCacheIntegrityTests(unittest.TestCase):
    def cache(self, root: Path) -> tuple[Path, Path, Path]:
        hub = root / "hub"
        model = hub / "models--Example--Model"
        blobs = model / "blobs"
        snapshot = model / "snapshots" / REVISION
        blobs.mkdir(parents=True)
        snapshot.mkdir(parents=True)
        (blobs / "config").write_bytes(b'{"model_type":"example"}\n')
        (blobs / "weights").write_bytes(b"weights")
        (snapshot / "config.json").symlink_to("../../blobs/config")
        (snapshot / "model.safetensors").symlink_to("../../blobs/weights")
        return hub, blobs, snapshot

    def test_valid_hugging_face_file_links_are_attested_and_revalidated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hub, blobs, _ = self.cache(root)
            inventory = build_model_cache_inventory(MODEL, REVISION, hub)
            self.assertEqual(inventory["attestation_policy"], ATTESTATION_POLICY)
            self.assertEqual(
                [row["path"] for row in inventory["files"]],
                ["config.json", "model.safetensors"],
            )
            self.assertEqual(
                [row["storage_path"] for row in inventory["files"]],
                [
                    "models--Example--Model/blobs/config",
                    "models--Example--Model/blobs/weights",
                ],
            )
            manifest = root / "model-cache.json"
            manifest_bytes = canonical_json_bytes(inventory)
            manifest.write_bytes(manifest_bytes)
            fingerprint = hashlib.sha256(manifest_bytes).hexdigest()
            verify_model_cache_inventory(MODEL, REVISION, manifest, fingerprint)

            (blobs / "weights").write_bytes(b"drifted")
            with self.assertRaisesRegex(
                ModelCacheIntegrityError, "differs from its prelaunch"
            ):
                verify_model_cache_inventory(MODEL, REVISION, manifest, fingerprint)

    def test_snapshot_escape_symlink_and_special_file_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hub, _, snapshot = self.cache(root)
            (snapshot / "config.json").unlink()
            (snapshot / "config.json").symlink_to("../../../../outside")
            with self.assertRaisesRegex(ModelCacheIntegrityError, "escapes the cache"):
                build_model_cache_inventory(MODEL, REVISION, hub)

            (snapshot / "config.json").unlink()
            os.mkfifo(snapshot / "config.json")
            with self.assertRaisesRegex(ModelCacheIntegrityError, "special file"):
                build_model_cache_inventory(MODEL, REVISION, hub)

    def test_resolved_blob_must_not_itself_be_a_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hub, blobs, snapshot = self.cache(root)
            (snapshot / "config.json").unlink()
            (blobs / "alias").symlink_to("config")
            (snapshot / "config.json").symlink_to("../../blobs/alias")
            with self.assertRaisesRegex(
                ModelCacheIntegrityError, "missing, special, or symlinked"
            ):
                build_model_cache_inventory(MODEL, REVISION, hub)


if __name__ == "__main__":
    unittest.main()
