"""Where live jobs and the ledger persist.

Render's free instances lose their disk on every restart, so nothing that must survive
lives there. With ``FIREBASE_SERVICE_ACCOUNT`` set (the service-account JSON, as a secret),
jobs and the hash-chained ledger go to Firestore; without it, an in-memory store is used,
which is what local development and the tests want.

The ledger is one chain for the whole service. Appends are serialised by the single
assessment worker, so the head can be read and extended without a transaction.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import UTC, datetime
from typing import Any

from pramana.common.digest import canonical_json, sha256_bytes
from pramana.ledger.chain import EVENT_TYPES, GENESIS
from pramana.ledger.signing import Signer

JOBS = "live_jobs"
LEDGER = "live_ledger"
META = "live_meta"


def _now() -> str:
    return datetime.now(UTC).isoformat()


class Store:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._ledger: list[dict[str, Any]] = []
        self._db = None
        raw = os.environ.get("FIREBASE_SERVICE_ACCOUNT", "").strip()
        if raw:
            import firebase_admin
            from firebase_admin import credentials, firestore

            info = json.loads(raw)
            app = firebase_admin.initialize_app(credentials.Certificate(info))
            self._db = firestore.client(app)

    @property
    def backend(self) -> str:
        return "firestore" if self._db is not None else "memory"

    # -- jobs ---------------------------------------------------------------------

    def put_job(self, job: dict[str, Any]) -> None:
        with self._lock:
            self._jobs[job["id"]] = job
        if self._db is not None:
            self._db.collection(JOBS).document(job["id"]).set(job)

    def update_job(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            job = self._jobs.setdefault(job_id, {"id": job_id})
            job.update(fields)
        if self._db is not None:
            self._db.collection(JOBS).document(job_id).set(fields, merge=True)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            if job_id in self._jobs:
                return dict(self._jobs[job_id])
        if self._db is not None:
            snap = self._db.collection(JOBS).document(job_id).get()
            return snap.to_dict() if snap.exists else None
        return None

    # -- ledger ---------------------------------------------------------------------

    def ledger(self) -> list[dict[str, Any]]:
        if self._db is not None and not self._ledger:
            docs = self._db.collection(LEDGER).order_by("index").stream()
            self._ledger = [d.to_dict() for d in docs]
        return list(self._ledger)

    def append(self, event_type: str, payload: dict[str, Any], signer: Signer) -> dict[str, Any]:
        """Append one signed row. Payload values are strings and integers only, so the
        browser's canonical JSON reproduces every hash exactly."""
        if event_type not in EVENT_TYPES:
            raise ValueError(f"{event_type!r} is not a declared ledger event type")
        rows = self.ledger()
        prev = rows[-1]["entry_hash"] if rows else GENESIS
        body = {"index": len(rows), "timestamp_utc": _now(), "event_type": event_type,
                "payload": payload, "prev_hash": prev}
        entry_hash = sha256_bytes(canonical_json(body))
        row = {**body, "entry_hash": entry_hash, "signer_id": signer.signer_id,
               "signature": signer.sign(entry_hash.encode("utf-8"))}
        self._ledger.append(row)
        if self._db is not None:
            self._db.collection(LEDGER).document(f"{row['index']:08d}").set(row)
        return row
