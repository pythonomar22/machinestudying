from __future__ import annotations

import unittest

from studybench.gpu_identity import normalize_cuda_uuid


class FakeCudaUuid:
    """Match the non-string UUID interface exposed by PyTorch 2.11."""

    def __str__(self) -> str:
        return "352dfe17-0851-a4bf-90ea-5b008d6820d1"


class GpuIdentityTests(unittest.TestCase):
    def test_normalizes_new_pytorch_cuda_uuid_object(self) -> None:
        self.assertEqual(
            normalize_cuda_uuid(FakeCudaUuid()),
            "GPU-352dfe17-0851-a4bf-90ea-5b008d6820d1",
        )

    def test_preserves_safe_prefixed_selectors(self) -> None:
        for value in (
            "GPU-352dfe17-0851-a4bf-90ea-5b008d6820d1",
            "MIG-352dfe17-0851-a4bf-90ea-5b008d6820d1",
        ):
            with self.subTest(value=value):
                self.assertEqual(normalize_cuda_uuid(value), value)

    def test_rejects_unsafe_or_ambiguous_values(self) -> None:
        for value in ("", "352dfe17", "GPU-safe\nsecond-row", object()):
            with self.subTest(value=repr(value)):
                with self.assertRaises(ValueError):
                    normalize_cuda_uuid(value)


if __name__ == "__main__":
    unittest.main()
