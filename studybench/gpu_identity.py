"""Small, dependency-free helpers for CUDA device provenance."""

from __future__ import annotations

import re
from typing import Any


_BARE_GPU_UUID = re.compile(
    r"[0-9A-Fa-f]{8}-(?:[0-9A-Fa-f]{4}-){3}[0-9A-Fa-f]{12}"
)
_NVIDIA_DEVICE_UUID = re.compile(r"(?:GPU|MIG)-[A-Za-z0-9-]+")


def normalize_cuda_uuid(value: Any) -> str:
    """Return a safe NVIDIA selector from a PyTorch CUDA UUID value.

    Older PyTorch releases exposed ``device_properties.uuid`` as a string.
    PyTorch 2.11 exposes a private ``_CUuuid`` value whose stable string form
    is the same canonical UUID without NVIDIA's ``GPU-`` selector prefix.
    Accept only those two documented shapes (plus an already-prefixed MIG
    selector) so an opaque object's representation can never reach a shell or
    provenance artifact unchecked.
    """

    try:
        uuid = value if isinstance(value, str) else str(value)
    except Exception as exc:  # pragma: no cover - defensive foreign-object path
        raise ValueError("CUDA UUID is not string-convertible") from exc
    if _BARE_GPU_UUID.fullmatch(uuid):
        return f"GPU-{uuid.lower()}"
    if _NVIDIA_DEVICE_UUID.fullmatch(uuid):
        return uuid
    raise ValueError("CUDA UUID has an unsupported or unsafe representation")
