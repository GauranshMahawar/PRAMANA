"""Air-gap enforcement (clause 2.2.6).

Every tool in this stack violates the air gap by default: ``pip install`` reaches out,
``pretrained=True`` downloads weights, dataset loaders fetch archives, ONNX Runtime may
pull execution-provider binaries. *"The internet, at runtime"* is not an admissible
answer to **where did those weights come from?**

Two mechanisms live here:

* :func:`verify_manifest` recomputes the digest of every staged artefact before an
  assessment runs. A manifest is only worth something if somebody checks it.
* :func:`assert_no_network` fails loudly if a code path tries to open a socket. CI runs
  the whole suite under it, so the claim stays true as six people commit for two weeks
  rather than being true on the day it was written.
"""

from __future__ import annotations

import socket
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import yaml

from pramana.common.digest import sha256_file
from pramana.common.errors import OfflineViolation

__all__ = ["verify_manifest", "assert_no_network", "no_network", "manifest_summary"]


def verify_manifest(path: str | Path = "offline-manifest.yaml") -> dict[str, Any]:
    """Recompute every digest in the offline manifest.

    Three outcomes per entry, and they are deliberately different:

    * ``ok`` -- staged and the digest matches.
    * ``missing`` -- not staged yet. Not an error: the manifest describes what an
      air-gapped install *needs*, and a development box legitimately has a subset.
    * ``mismatch`` -- staged and the digest does **not** match. This fails the command,
      because an artefact whose bytes differ from the ones we recorded is an artefact
      nobody has assessed.
    """
    p = Path(path)
    if not p.exists():
        raise OfflineViolation(
            f"{p} is missing. An air-gapped install without a manifest is an install "
            f"whose provenance nobody can check."
        )

    doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    root = p.parent
    entries: list[dict[str, Any]] = []
    ok = missing = mismatch = 0

    for group_name, group in (doc.get("artefacts") or {}).items():
        for item in group or []:
            target = root / item["path"]
            record = {
                "group": group_name,
                "path": item["path"],
                "licence": item.get("licence"),
                "source": item.get("source"),
                "expected": item.get("sha256"),
            }
            if not target.exists():
                record["status"] = "missing"
                missing += 1
            elif not item.get("sha256") or item["sha256"].endswith("PENDING"):
                record["status"] = "missing"
                record["note"] = "digest not yet recorded; run `make stage` after staging"
                missing += 1
            else:
                actual = sha256_file(target)
                record["actual"] = actual
                if actual == item["sha256"]:
                    record["status"] = "ok"
                    ok += 1
                else:
                    record["status"] = "mismatch"
                    mismatch += 1
            entries.append(record)

    return {
        "manifest": str(p),
        "entries": entries,
        "ok_count": ok,
        "missing_count": missing,
        "mismatch_count": mismatch,
        "ok": mismatch == 0,
    }


def manifest_summary(path: str | Path = "offline-manifest.yaml") -> dict[str, Any]:
    """Licence roll-up over the manifest -- the data half of the bill of materials.

    Flags NonCommercial and source-available licences explicitly, because those are the
    two that a defence assurance instrument cannot carry and the two that are easiest
    to acquire by accident.
    """
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    licences: dict[str, int] = {}
    flagged: list[dict[str, str]] = []

    for group in (doc.get("artefacts") or {}).values():
        for item in group or []:
            lic = item.get("licence", "UNKNOWN")
            licences[lic] = licences.get(lic, 0) + 1
            upper = lic.upper()
            if "NC" in upper.split("-") or "NONCOMMERCIAL" in upper.replace(" ", ""):
                flagged.append({"path": item["path"], "licence": lic, "why": "NonCommercial"})
            elif any(bad in upper for bad in ("SSPL", "RSAL", "BUSL", "ELASTIC")):
                flagged.append({"path": item["path"], "licence": lic, "why": "source-available"})
            elif lic == "UNKNOWN":
                flagged.append({"path": item["path"], "licence": lic, "why": "undeclared"})

    return {"licences": licences, "flagged": flagged, "clean": not flagged}


class _BlockedSocket(socket.socket):
    def connect(self, *args, **kwargs):  # type: ignore[override]
        raise OfflineViolation(
            "A code path attempted a network connection. Clause 2.2.6 mandates "
            "air-gapped operation, so egress is a compliance failure rather than an "
            "inconvenience. Stage the artefact into offline-manifest.yaml instead."
        )

    def connect_ex(self, *args, **kwargs):  # type: ignore[override]
        raise OfflineViolation("A code path attempted a network connection (connect_ex).")


@contextmanager
def no_network() -> Iterator[None]:
    """Block outbound sockets for the duration of the block.

    Used by the test suite. It is not a sandbox and does not pretend to be one -- a
    determined subprocess escapes it. What it catches is the realistic failure: a
    library that quietly fetches weights or a dataset the first time it is called.
    """
    original = socket.socket
    original_create = socket.create_connection

    def _blocked_create(*args, **kwargs):
        raise OfflineViolation("create_connection blocked: this suite runs offline.")

    socket.socket = _BlockedSocket  # type: ignore[assignment,misc]
    socket.create_connection = _blocked_create  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket = original  # type: ignore[assignment,misc]
        socket.create_connection = original_create  # type: ignore[assignment]


def assert_no_network() -> None:
    """Install the block for the remainder of the process."""
    socket.socket = _BlockedSocket  # type: ignore[assignment,misc]
