"""Process-independent integer seeds derived from arbitrary keys."""
from __future__ import annotations

import hashlib


def stable_seed(*parts) -> int:
    """32-bit seed from `parts`, identical across processes. Built-in hash() is salted per
    process for str/bytes (PYTHONHASHSEED), so it must not be used to seed experiments."""
    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:4], "little")
