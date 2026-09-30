"""The external anchor -- turning "trust us" into "trust us or catch us".

A hash chain held by the certifying authority is tamper-evident *to its holder*. It is
not evidence *against* its holder. That sits badly beside a claim to produce evidence
admissible against the party it accuses, and closing it is the only thing this module
does.

Be exact about what it buys: it does **not** stop a compromised certifier. An insider
with our keys can still issue a false certificate and anchor it. What it removes is the
ability to alter the record *afterwards* -- to change a verdict, back-date a battery or
delete a refusal -- because the digest of the original is outside their reach.

Three options were considered and the trade is recorded in the technical document:

* **RFC 3161 timestamping** -- adopted as the default, because its token verifies
  offline against the TSA certificate and that is the only property compatible with
  clause 2.2.6's air gap.
* **Sigstore Rekor** -- for a public instance. Entries are public by construction,
  which is acceptable for digests and disqualifying for anything else.
* **A second agency's cross-signed ledger** -- the preferred end state, recorded as a
  capability we do not have because it needs an MoU and a key ceremony.

We publish three digests and nothing else: the pre-committed battery digest, the
pre-registration record for each assessment, and the hash of each issued certificate or
refusal. No imagery, no weights, no supplier identity, no verdict text.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pramana.common.digest import sha256_bytes, sha256_file
from pramana.report.schema import ExternalAnchor

#: Only these three classes of digest are ever submitted externally.
ANCHORABLE = frozenset({"battery_digest", "preregistration_digest", "certificate_digest"})


@dataclass
class AnchorToken:
    """An RFC 3161 timestamp token, or a local stand-in for one.

    ``is_stand_in`` is not a convenience flag. An anchor that claims upstream
    confirmation it does not have is worse than no anchor, so the distinction travels
    with the token and into the report.
    """

    merkle_root: str
    token_bytes: bytes
    obtained_at_utc: str
    tsa_url: str | None
    is_stand_in: bool

    @property
    def token_digest(self) -> str:
        return sha256_bytes(self.token_bytes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "merkle_root": self.merkle_root,
            "token_digest": self.token_digest,
            "obtained_at_utc": self.obtained_at_utc,
            "tsa_url": self.tsa_url,
            "is_stand_in": self.is_stand_in,
        }


class Anchorer:
    """Batches digests and anchors their Merkle root on each authorised window.

    The assessment host is air-gapped, so anchoring is **batched**: digests accumulate
    locally and, on each authorised connection window, the accumulated root is
    submitted and the token carried back. The window is a declared parameter and the
    measured lag is printed in every report -- because an anchor with an undeclared lag
    is an anchor whose gap an insider can sit inside.

    We found no offline or batch-submission procedure documented upstream, so the
    batching is ours and the report labels it ``anchor_procedure: local`` rather than
    implying an upstream feature.
    """

    def __init__(
        self,
        store_dir: str | Path,
        *,
        batch_interval_s: int = 86_400,
        tsa_url: str | None = None,
    ):
        self.store = Path(store_dir)
        self.store.mkdir(parents=True, exist_ok=True)
        self.batch_interval_s = batch_interval_s
        self.tsa_url = tsa_url
        self._pending: list[tuple[str, str]] = []

    def submit(self, kind: str, digest: str) -> None:
        """Queue one digest for the next window."""
        if kind not in ANCHORABLE:
            raise ValueError(
                f"{kind!r} is not anchorable. Only {sorted(ANCHORABLE)} leave the air "
                f"gap; a digest is a commitment and a commitment is all a third party "
                f"needs."
            )
        self._pending.append((kind, digest))

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def pending_root(self) -> str:
        """Merkle root over the queued digests, computed the same way as the chain's."""
        leaves = [d for _, d in self._pending]
        if not leaves:
            return "sha256:" + "0" * 64
        level = leaves
        while len(level) > 1:
            if len(level) % 2:
                level.append(level[-1])
            level = [
                sha256_bytes((level[i] + level[i + 1]).encode("utf-8"))
                for i in range(0, len(level), 2)
            ]
        return level[0]

    def anchor_offline(self, merkle_root: str | None = None) -> AnchorToken:
        """Produce a locally-held stand-in token.

        This is what runs inside the air gap. It records the root and the moment, and
        it is honest in the report that no third party has yet seen it:
        ``anchor_state: pending``. The demo fetches a real token *before* going
        offline; this path exists so that the system is never silently unanchored.
        """
        root = merkle_root or self.pending_root()
        now = datetime.now(UTC).isoformat()
        body = json.dumps({"merkle_root": root, "obtained_at_utc": now}, sort_keys=True)
        token = AnchorToken(
            merkle_root=root,
            token_bytes=body.encode("utf-8"),
            obtained_at_utc=now,
            tsa_url=None,
            is_stand_in=True,
        )
        self._persist(token)
        self._pending.clear()
        return token

    def load_token(self, path: str | Path) -> AnchorToken:
        """Load a real RFC 3161 token fetched during an authorised window.

        Demo beat 12: fetch this before the cable comes out, tamper with a ledger row,
        then re-verify against a digest we hold no key to.
        """
        p = Path(path)
        raw = p.read_bytes()
        meta_path = p.with_suffix(p.suffix + ".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        return AnchorToken(
            merkle_root=meta.get("merkle_root", ""),
            token_bytes=raw,
            obtained_at_utc=meta.get("obtained_at_utc", ""),
            tsa_url=meta.get("tsa_url", self.tsa_url),
            is_stand_in=False,
        )

    def _persist(self, token: AnchorToken) -> None:
        stem = token.merkle_root.replace("sha256:", "")[:16]
        (self.store / f"{stem}.tst").write_bytes(token.token_bytes)
        (self.store / f"{stem}.tst.meta.json").write_text(
            json.dumps(token.to_dict(), indent=2), encoding="utf-8"
        )

    # -- verification ------------------------------------------------------

    @staticmethod
    def verify_against(token: AnchorToken, current_root: str) -> bool:
        """Does the chain still hash to the root the token committed to?

        This is the check that fails in demo beat 12 while the local chain would still
        verify if only the row's own hash had been updated. It is the only moment in
        eight minutes where the evidence does not rest on our good faith.
        """
        return token.merkle_root == current_root

    def report_block(
        self,
        token: AnchorToken | None,
        *,
        current_root: str | None = None,
        lag_s: float | None = None,
    ) -> ExternalAnchor:
        """Render the anchor state for the assurance report."""
        if token is None:
            return ExternalAnchor(
                anchor_procedure="local",
                anchor_type="none",
                anchor_state="unavailable",
                anchor_batch_interval_s=self.batch_interval_s,
            )
        state = "anchored"
        if token.is_stand_in:
            state = "pending"
        elif current_root is not None and not self.verify_against(token, current_root):
            state = "unavailable"
        return ExternalAnchor(
            anchor_procedure="local",
            anchor_type="rfc3161",
            merkle_root=token.merkle_root or None,
            anchor_state=state,  # type: ignore[arg-type]
            anchor_batch_interval_s=self.batch_interval_s,
            anchor_lag_s=lag_s,
            token_digest=token.token_digest,
        )


def digest_of_report(path: str | Path) -> str:
    """Certificate digest for anchoring. Only the digest leaves the air gap."""
    return sha256_file(path)
