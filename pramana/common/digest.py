"""Digests and canonical serialisation.

Everything PRAMANA commits to -- batteries, shards, weights, calibration sets, scale
tables, reports -- is identified by a SHA-256 over a *canonical* byte sequence. If two
parties cannot independently recompute the same digest from the same logical object,
the commitment is worthless, so canonicalisation lives here and nowhere else.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_CHUNK = 1 << 20  # 1 MiB


def sha256_bytes(data: bytes) -> str:
    """Digest raw bytes. Returns the ``sha256:<hex>`` form used throughout the schema."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Digest a file without loading it into memory.

    Used for model weights and dataset archives, which are routinely larger than RAM
    on the 8 GB development box.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def canonical_json(obj: Any) -> bytes:
    """Canonical JSON: sorted keys, no insignificant whitespace, UTF-8, no NaN.

    This is the serialisation every signature and every chain link is computed over.
    ``allow_nan=False`` is deliberate -- a NaN that serialises to the non-standard
    ``NaN`` token would produce a digest no other JSON implementation can reproduce,
    which defeats the point of a commitment.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(obj: Any) -> str:
    """Digest a JSON-serialisable object canonically."""
    return sha256_bytes(canonical_json(obj))


def short(digest: str, head: int = 4, tail: int = 4) -> str:
    """Abbreviate a digest for display only.

    Never use the result for comparison. Reports carry full digests; consoles and log
    lines carry this.
    """
    if not digest.startswith("sha256:"):
        return digest
    hexpart = digest[7:]
    if len(hexpart) <= head + tail:
        return digest
    return f"sha256:{hexpart[:head]}…{hexpart[-tail:]}"
