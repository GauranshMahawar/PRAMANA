"""C6 -- signed inference receipts (clause 2.2.3).

The clause asks for a verifiable cryptographic binding among the input image, the model
identifier or weight digest, the preprocessing and inference configuration, and the
resulting output -- and for post-hoc alteration, substitution or **replay** to be
detectable through hashes, signatures and appropriate sequence, timestamp or nonce
controls.

All three anti-replay controls are present and they do different jobs:

* **nonce** -- a fresh random value per receipt, so two identical inferences produce
  two distinct receipts and neither can stand in for the other.
* **sequence** -- a monotonic counter per emitter, so a *deleted* receipt is detectable
  as a gap. A timestamp alone cannot do this.
* **chain position** -- the previous receipt's digest, so the stream is append-only and
  a receipt cannot be re-ordered.

The field that does not appear in other designs is ``rung``. A receipt that names the
model digest but not which *build* produced the inference cannot distinguish the FP32
artefact everyone tested from the INT8 artefact that actually ran, which is the entire
premise of D1.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from pramana.common.digest import canonical_json, sha256_bytes
from pramana.ledger.signing import Signer, verify

__all__ = ["Receipt", "ReceiptEmitter", "verify_stream"]

GENESIS = "sha256:" + "0" * 64


@dataclass(frozen=True)
class Receipt:
    receipt_id: str
    emitter_id: str
    sequence: int
    nonce: str
    timestamp_utc: str
    input_digest: str
    model_digest: str
    rung: str
    preprocessing_chain_digest: str
    inference_config_digest: str
    output: dict[str, Any]
    prev_receipt_digest: str
    receipt_digest: str
    signature: str
    per_prediction_certificate: dict[str, Any] | None = None

    def signed_body(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "emitter_id": self.emitter_id,
            "sequence": self.sequence,
            "nonce": self.nonce,
            "timestamp_utc": self.timestamp_utc,
            "input_digest": self.input_digest,
            "model_digest": self.model_digest,
            "rung": self.rung,
            "preprocessing_chain_digest": self.preprocessing_chain_digest,
            "inference_config_digest": self.inference_config_digest,
            "output": self.output,
            "prev_receipt_digest": self.prev_receipt_digest,
            "per_prediction_certificate": self.per_prediction_certificate,
        }

    def recompute_digest(self) -> str:
        return sha256_bytes(canonical_json(self.signed_body()))

    def to_dict(self) -> dict[str, Any]:
        d = self.signed_body()
        d["receipt_digest"] = self.receipt_digest
        d["signature"] = self.signature
        return d


@dataclass
class ReceiptEmitter:
    """Emits the signed receipt stream from an edge device.

    The realistic edge target is ONNX Runtime on a weak device, and that is deliberate:
    the artefact that actually ships is the converted one, which is why the receipt
    names its rung.
    """

    emitter_id: str
    signer: Signer
    _sequence: int = 0
    _prev: str = GENESIS
    _seen_nonces: set[str] = field(default_factory=set)

    def emit(
        self,
        *,
        input_digest: str,
        model_digest: str,
        rung: str,
        preprocessing_chain_digest: str,
        inference_config_digest: str,
        output: dict[str, Any],
        per_prediction_certificate: dict[str, Any] | None = None,
        timestamp: datetime | None = None,
    ) -> Receipt:
        nonce = secrets.token_hex(16)
        ts = (timestamp or datetime.now(UTC)).isoformat()
        body = {
            "receipt_id": f"{self.emitter_id}-{self._sequence:08d}",
            "emitter_id": self.emitter_id,
            "sequence": self._sequence,
            "nonce": nonce,
            "timestamp_utc": ts,
            "input_digest": input_digest,
            "model_digest": model_digest,
            "rung": rung,
            "preprocessing_chain_digest": preprocessing_chain_digest,
            "inference_config_digest": inference_config_digest,
            "output": output,
            "prev_receipt_digest": self._prev,
            "per_prediction_certificate": per_prediction_certificate,
        }
        digest = sha256_bytes(canonical_json(body))
        receipt = Receipt(
            **body,
            receipt_digest=digest,
            signature=self.signer.sign(digest.encode("utf-8")),
        )
        self._sequence += 1
        self._prev = digest
        self._seen_nonces.add(nonce)
        return receipt


def verify_stream(
    receipts: list[Receipt],
    public_key: bytes | str,
) -> dict[str, Any]:
    """Verify a receipt stream for alteration, substitution, replay and deletion.

    Returns a structured result rather than raising, because "which control caught it"
    is the interesting part -- a replayed receipt and a deleted one fail differently,
    and an operator needs to know which.
    """
    problems: list[dict[str, Any]] = []
    seen_nonces: set[str] = set()
    seen_digests: set[str] = set()
    prev = GENESIS
    expected_seq = 0

    for r in receipts:
        if r.recompute_digest() != r.receipt_digest:
            problems.append({"receipt_id": r.receipt_id, "control": "hash", "issue": "altered"})
        if not verify(public_key, r.receipt_digest.encode("utf-8"), r.signature):
            problems.append({"receipt_id": r.receipt_id, "control": "signature", "issue": "invalid"})
        if r.nonce in seen_nonces or r.receipt_digest in seen_digests:
            problems.append({"receipt_id": r.receipt_id, "control": "nonce", "issue": "replay"})
        if r.prev_receipt_digest != prev:
            problems.append(
                {"receipt_id": r.receipt_id, "control": "chain_position", "issue": "reordered_or_substituted"}
            )
        if r.sequence != expected_seq:
            problems.append(
                {
                    "receipt_id": r.receipt_id,
                    "control": "sequence",
                    "issue": f"gap: expected {expected_seq}, got {r.sequence}",
                }
            )
            expected_seq = r.sequence

        seen_nonces.add(r.nonce)
        seen_digests.add(r.receipt_digest)
        prev = r.receipt_digest
        expected_seq += 1

    return {
        "ok": not problems,
        "n_receipts": len(receipts),
        "problems": problems,
        "controls_checked": ["hash", "signature", "nonce", "sequence", "chain_position"],
    }
