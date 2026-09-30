"""C1 -- the hash-chained, Ed25519-signed, append-only ledger.

Not a blockchain, and the reasoning is a strength rather than a concession: there is a
single certifying authority, so the problem is tamper-*evidence*, not decentralised
consensus among mutually distrustful parties. A hash chain gives tamper-evidence at
microseconds per append with no consensus overhead, no network dependency and no
external infrastructure -- which matters when the deployment is air-gapped.

What a hash chain held by the certifying authority does NOT give is evidence *against*
its holder. That gap is closed, as far as it can be, by :mod:`pramana.ledger.anchor`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from pramana.common.digest import canonical_json, sha256_bytes
from pramana.common.errors import LedgerTampered, SignatureInvalid
from pramana.ledger.signing import Signer, verify

GENESIS = "sha256:" + "0" * 64


@dataclass(frozen=True)
class LedgerEntry:
    """One append-only row.

    ``entry_hash`` covers ``prev_hash`` as well as the payload, so altering any row
    invalidates every row after it. ``signature`` covers ``entry_hash``, so a row
    cannot be re-signed by anyone without the key.
    """

    index: int
    timestamp_utc: str
    event_type: str
    payload: dict[str, Any]
    prev_hash: str
    entry_hash: str
    signer_id: str
    signature: str

    def signed_body(self) -> dict[str, Any]:
        """The exact object the hash is computed over. Field order is irrelevant --
        :func:`canonical_json` sorts keys -- but membership is not."""
        return {
            "index": self.index,
            "timestamp_utc": self.timestamp_utc,
            "event_type": self.event_type,
            "payload": self.payload,
            "prev_hash": self.prev_hash,
        }

    def recompute_hash(self) -> str:
        return sha256_bytes(canonical_json(self.signed_body()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "timestamp_utc": self.timestamp_utc,
            "event_type": self.event_type,
            "payload": self.payload,
            "prev_hash": self.prev_hash,
            "entry_hash": self.entry_hash,
            "signer_id": self.signer_id,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LedgerEntry:
        return cls(
            index=d["index"],
            timestamp_utc=d["timestamp_utc"],
            event_type=d["event_type"],
            payload=d["payload"],
            prev_hash=d["prev_hash"],
            entry_hash=d["entry_hash"],
            signer_id=d["signer_id"],
            signature=d["signature"],
        )


#: Event vocabulary. Closed, because D4.a draws the sequential test's candidate change
#: points from exactly these -- an alarm either names the signer whose act preceded it
#: or is reported as `unexplained_epoch`. A free-text event type would let the index
#: set grow after an alarm, which is the objection that construction has to survive.
EVENT_TYPES = frozenset(
    {
        "battery_committed",
        "taxonomy_committed",
        "threshold_file_committed",
        "shard_admitted",
        "lot_admitted",
        "training_manifest_signed",
        "converter_version_change",
        "calibration_set_change",
        "assessment_preregistered",
        "assessment_completed",
        "certificate_issued",
        "certificate_refused",
        "disposition_set",
        "risk_acceptance_granted",
        "risk_acceptance_revoked",
        "reassessment_triggered",
        "ioc_released",
        "contract_relet",
        "anchor_submitted",
        "anchor_confirmed",
    }
)


class HashChain:
    """An append-only chain persisted as JSON Lines.

    JSONL rather than a database for the reference implementation: it is inspectable
    with ``cat``, diffable, trivially exportable for the "reproducible audit log"
    deliverable, and it carries no server dependency into the air gap. The PostgreSQL
    backend in :mod:`pramana.ledger.store` implements the same interface for the
    deployed configuration.
    """

    def __init__(self, path: str | Path, signer: Signer | None = None):
        self.path = Path(path)
        self.signer = signer
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    # -- reading -----------------------------------------------------------

    def __iter__(self) -> Iterator[LedgerEntry]:
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield LedgerEntry.from_dict(json.loads(line))

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def entries(self) -> list[LedgerEntry]:
        return list(self)

    def head(self) -> str:
        last = None
        for last in self:  # noqa: B007 -- we want the final value
            pass
        return last.entry_hash if last else GENESIS

    # -- writing -----------------------------------------------------------

    def append(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        signer: Signer | None = None,
        timestamp: datetime | None = None,
    ) -> LedgerEntry:
        """Append one row and return it.

        Raises if ``event_type`` is outside the closed vocabulary: see
        :data:`EVENT_TYPES` for why that matters to D4.a.
        """
        if event_type not in EVENT_TYPES:
            raise ValueError(
                f"{event_type!r} is not a declared ledger event type. The candidate "
                f"epoch set for sequential monitoring is drawn from this vocabulary, so "
                f"it cannot grow ad hoc. Add it to EVENT_TYPES deliberately."
            )
        sgn = signer or self.signer
        if sgn is None:
            raise SignatureInvalid("No signer available; an unsigned ledger row is not evidence.")

        prev = self.head()
        idx = len(self)
        ts = (timestamp or datetime.now(UTC)).isoformat()

        body = {
            "index": idx,
            "timestamp_utc": ts,
            "event_type": event_type,
            "payload": payload,
            "prev_hash": prev,
        }
        entry_hash = sha256_bytes(canonical_json(body))
        signature = sgn.sign(entry_hash.encode("utf-8"))

        entry = LedgerEntry(
            index=idx,
            timestamp_utc=ts,
            event_type=event_type,
            payload=payload,
            prev_hash=prev,
            entry_hash=entry_hash,
            signer_id=sgn.signer_id,
            signature=signature,
        )
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry.to_dict(), sort_keys=True, ensure_ascii=False) + "\n")
        return entry

    # -- verification ------------------------------------------------------

    def verify(self, public_keys: dict[str, bytes] | None = None) -> bool:
        """Verify the whole chain. Raises :class:`LedgerTampered` at the first break.

        Demo beat 12 edits one row and calls this. It fails at that row and names it.
        """
        prev = GENESIS
        for i, entry in enumerate(self):
            if entry.index != i:
                raise LedgerTampered(i, str(i), str(entry.index))
            if entry.prev_hash != prev:
                raise LedgerTampered(i, prev, entry.prev_hash)
            recomputed = entry.recompute_hash()
            if recomputed != entry.entry_hash:
                raise LedgerTampered(i, entry.entry_hash, recomputed)
            if public_keys is not None:
                pk = public_keys.get(entry.signer_id)
                if pk is None:
                    raise SignatureInvalid(
                        f"row {i}: no public key registered for signer {entry.signer_id!r}"
                    )
                if not verify(pk, entry.entry_hash.encode("utf-8"), entry.signature):
                    raise SignatureInvalid(f"row {i}: Ed25519 signature does not verify")
            prev = entry.entry_hash
        return True

    def merkle_root(self) -> str:
        """Root over every ``entry_hash``, for external anchoring.

        Duplicating the last leaf on odd levels is the conventional choice and is
        stated because it is a detail an independent verifier must match exactly.
        """
        leaves = [e.entry_hash for e in self]
        if not leaves:
            return GENESIS
        level = leaves
        while len(level) > 1:
            if len(level) % 2:
                level.append(level[-1])
            level = [
                sha256_bytes((level[i] + level[i + 1]).encode("utf-8"))
                for i in range(0, len(level), 2)
            ]
        return level[0]

    # -- queries the monitor needs ----------------------------------------

    def events_of(self, *types: str) -> list[LedgerEntry]:
        """Rows matching the given event types, in order.

        D4.a uses this: the sequential monitor's candidate start times are exactly the
        signed supply-chain acts, not every index.
        """
        wanted = set(types)
        return [e for e in self if e.event_type in wanted]


@dataclass
class InMemoryChain(HashChain):
    """Chain that never touches disk. Used by tests and by the conformance fixtures."""

    _rows: list[LedgerEntry] = field(default_factory=list)

    def __init__(self, signer: Signer | None = None):  # noqa: D107
        self.signer = signer
        self._rows = []

    def __iter__(self) -> Iterator[LedgerEntry]:
        return iter(self._rows)

    def __len__(self) -> int:
        return len(self._rows)

    def append(self, event_type, payload, *, signer=None, timestamp=None):  # type: ignore[override]
        if event_type not in EVENT_TYPES:
            raise ValueError(f"{event_type!r} is not a declared ledger event type.")
        sgn = signer or self.signer
        if sgn is None:
            raise SignatureInvalid("No signer available.")
        prev = self.head()
        idx = len(self._rows)
        ts = (timestamp or datetime.now(UTC)).isoformat()
        body = {
            "index": idx,
            "timestamp_utc": ts,
            "event_type": event_type,
            "payload": payload,
            "prev_hash": prev,
        }
        entry_hash = sha256_bytes(canonical_json(body))
        entry = LedgerEntry(
            index=idx,
            timestamp_utc=ts,
            event_type=event_type,
            payload=payload,
            prev_hash=prev,
            entry_hash=entry_hash,
            signer_id=sgn.signer_id,
            signature=sgn.sign(entry_hash.encode("utf-8")),
        )
        self._rows.append(entry)
        return entry

    def tamper(self, index: int, **payload_changes: Any) -> None:
        """The CLUMSY tamper: alter a payload and leave the stale hash in place.

        Local verification catches this immediately, so it is the weaker of the two
        demonstrations. It is here because it is the attack an outsider mounts.
        """
        old = self._rows[index]
        self._rows[index] = LedgerEntry(
            index=old.index,
            timestamp_utc=old.timestamp_utc,
            event_type=old.event_type,
            payload={**old.payload, **payload_changes},
            prev_hash=old.prev_hash,
            entry_hash=old.entry_hash,  # deliberately stale
            signer_id=old.signer_id,
            signature=old.signature,
        )

    def rewrite_history(self, index: int, signer: Signer, **payload_changes: Any) -> None:
        """The COMPETENT tamper -- adversary A7, the insider who holds our keys.

        Alter a payload, recompute that row's hash, re-sign it, then re-chain and
        re-sign every row after it. Afterwards :meth:`verify` **passes**: the chain is
        internally consistent, every signature is valid, and nothing inside the
        instrument can tell that a verdict was changed.

        This is the demonstration that matters, and it is the honest one, because it
        shows what a hash chain held by the certifying authority actually buys: the
        local ledger is tamper-evident *to its holder* and is not evidence *against*
        its holder. What changes is the Merkle root -- so the externally anchored
        token, on a digest held by a third party we cannot reach, no longer matches.

        Demo beat 12 calls this, not :meth:`tamper`: "the local chain still verifies,
        and the external check fails anyway."
        """
        if not 0 <= index < len(self._rows):
            raise IndexError(f"no ledger row at index {index}")

        old = self._rows[index]
        rewritten = {**old.payload, **payload_changes}
        prev = old.prev_hash

        for i in range(index, len(self._rows)):
            row = self._rows[i]
            payload = rewritten if i == index else row.payload
            body = {
                "index": row.index,
                "timestamp_utc": row.timestamp_utc,
                "event_type": row.event_type,
                "payload": payload,
                "prev_hash": prev,
            }
            entry_hash = sha256_bytes(canonical_json(body))
            self._rows[i] = LedgerEntry(
                index=row.index,
                timestamp_utc=row.timestamp_utc,
                event_type=row.event_type,
                payload=payload,
                prev_hash=prev,
                entry_hash=entry_hash,
                signer_id=signer.signer_id,
                signature=signer.sign(entry_hash.encode("utf-8")),
            )
            prev = entry_hash
