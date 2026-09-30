"""D2 -- provenance-aligned partition and the volume-inequality block.

Every certified poisoning defence in the literature bounds robustness in units of
*poisoned samples*. A procurement authority signs contracts with *suppliers*. So the
corpus is partitioned along **contract-lot** boundaries, a voting ensemble is trained
over that partition, and the guarantee is measured in the unit the contract is written
in.

**Read the subject twice.** The bound is a property of the ensemble PRAMANA builds for
the measurement. It is never a property of the artefact the Army fields, and the two
claims do not compose. The schema enforces this; so does every docstring here.

FLCert (arXiv:2210.00584, IEEE TIFS) owns the client-denominated certificate in
federated learning, and FLCert-D's deterministic disjoint groups are algorithmically
this partition. We cite it rather than claim it. The port is not cosmetic: in FL the
partition is *given* by who holds the data, so k is a fact to be measured; in
procurement it is *chosen* by how the contract is written, so **k is a decision
variable** -- which is what makes lot restructuring analysable and what FLCert has no
reason to consider.

Two attacks on the unit itself are answered here (technical doc 8.6):

* **Splitting** -- one supplier registers as eight small lots, inflating k without
  changing adversarial capability. Answered by lots being drawn from the procuring
  authority's contract register rather than self-declared, and by the entity map.
* **Merging** -- a supplier consolidates so its volume sits in one lot, making
  leave-one-lot-out so destructive the analysis is never run. Answered by the
  degeneracy refusal: past a 0.5 single-lot share, **no count is issued at all**.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pramana.common import constants as K

__all__ = ["Lot", "LotRegister", "volume_inequality_block", "DegenerateScope"]


class DegenerateScope(Exception):
    """Raised when the corpus is too concentrated for a contributor-denominated bound.

    A caveat would have been read past; a refusal cannot be. The repair is to re-let
    the contract, not to compute a better statistic -- which is the differentiator,
    not the caveat.
    """

    def __init__(self, largest_share: float, lot_id: str):
        self.largest_share = largest_share
        self.lot_id = lot_id
        super().__init__(
            f"certificate_scope_degenerate: single_lot_majority -- lot {lot_id!r} holds "
            f"{largest_share:.1%} of the corpus, above the declared "
            f"{K.SINGLE_LOT_MAJORITY_THRESHOLD:.0%} threshold. No floor is issued."
        )


@dataclass(frozen=True)
class Lot:
    """A contract object, not a self-declared shard.

    ``lot_id``, ``purchase_order`` and ``signer_id`` come from the procuring
    authority's register. That is exactly why denominating in lots is defensible where
    denominating in shards is not: a shard is a number we chose, a lot is a number an
    auditor can subpoena. Splitting a lot requires forging the contract register.

    ``entity_id`` is D4.b: twelve lots can be four companies. Where the contract does
    not disclose the map, the field is ``None`` and the report emits
    ``entity_map_incomplete`` naming the clause that would have supplied it -- an
    assurance gap turned into a procurement action.
    """

    lot_id: str
    n_samples: int
    purchase_order: str | None = None
    signer_id: str | None = None
    vendor_code: str | None = None
    entity_id: str | None = None
    admitted_at_utc: str | None = None
    shard_digest: str | None = None


@dataclass
class LotRegister:
    """The set of contracted lots making up one training corpus."""

    lots: list[Lot] = field(default_factory=list)

    def add(self, lot: Lot) -> None:
        if any(existing.lot_id == lot.lot_id for existing in self.lots):
            raise ValueError(f"lot {lot.lot_id!r} is already registered")
        self.lots.append(lot)

    @property
    def m(self) -> int:
        return len(self.lots)

    @property
    def total_samples(self) -> int:
        return sum(lot.n_samples for lot in self.lots)

    def shares(self) -> dict[str, float]:
        total = self.total_samples
        if total == 0:
            return {}
        return {lot.lot_id: lot.n_samples / total for lot in self.lots}

    def largest(self) -> tuple[str, float]:
        shares = self.shares()
        lot_id = max(shares, key=shares.__getitem__)
        return lot_id, shares[lot_id]

    def share_of_largest_k(self, k: int) -> float:
        """Volume held by the k largest lots -- COMPARATOR PAIR 6.

        Printed beside every certified k, because three of twelve lots can be half the
        corpus and a count alone would read as a strong guarantee.
        """
        if k <= 0:
            return 0.0
        ordered = sorted(self.shares().values(), reverse=True)
        return float(sum(ordered[:k]))

    def check_scope(self) -> None:
        """Raise :class:`DegenerateScope` if one lot holds more than half the corpus."""
        if not self.lots:
            return
        lot_id, share = self.largest()
        if share > K.SINGLE_LOT_MAJORITY_THRESHOLD:
            raise DegenerateScope(share, lot_id)

    # -- D4.b: entities, not lots -----------------------------------------

    def entity_map(self) -> dict[str, list[str]] | None:
        """Group lots by legal entity. ``None`` when the contract did not disclose it."""
        if any(lot.entity_id is None for lot in self.lots):
            return None
        grouped: dict[str, list[str]] = {}
        for lot in self.lots:
            grouped.setdefault(lot.entity_id, []).append(lot.lot_id)  # type: ignore[arg-type]
        return grouped

    def entity_shares(self) -> dict[str, float] | None:
        mapping = self.entity_map()
        if mapping is None:
            return None
        shares = self.shares()
        return {ent: sum(shares[lid] for lid in lids) for ent, lids in mapping.items()}

    def entity_map_status(self) -> dict[str, object]:
        """The `entity_map_incomplete` finding, with the clause that would retire it.

        No instrument we could read obliges per-lot disclosure of ultimate beneficial
        ownership across the lots of a single award: GeM GTC clause 29 bars sister
        concerns at *bid* time, clause 26's beneficial-ownership declaration triggers
        on land-border jurisdiction, and DAP 2020's prime/Tier-I vendor notion lives in
        offsets discharge rather than the Standard Contract Document. That absence is a
        finding about the instruments, not a gap in our modelling.
        """
        mapping = self.entity_map()
        if mapping is not None:
            return {
                "entity_map_incomplete": None,
                "n_entities": len(mapping),
                "lots_per_entity": {k: len(v) for k, v in mapping.items()},
            }
        undisclosed = [lot.lot_id for lot in self.lots if lot.entity_id is None]
        return {
            "entity_map_incomplete": "no_per_lot_beneficial_ownership_disclosure",
            "undisclosed_lots": undisclosed,
            "clause_that_would_retire_it": (
                "Standard Contract Document (DAP 2020 Ch. VI): add a per-lot declaration "
                "of ultimate beneficial owner and named Tier-I sub-vendors, at award, "
                "refreshed on any change of control."
            ),
            "fallback": (
                "certified_floor_k_lots still stands -- lots are always in the contract. "
                "Where the map is partial the entity floor is reported over the entities "
                "we can separate and labelled a lower bound."
            ),
        }


def volume_inequality_block(register: LotRegister, k: int) -> dict[str, object]:
    """The ``volume_inequality`` report block.

    ``k_is_a_count_not_a_volume`` is a literal field because k alone overstates the
    guarantee exactly as a p-value without its floor does.
    """
    shares = register.shares()
    total = sum(shares.values())
    lot_id, largest = register.largest() if shares else ("", 0.0)
    return {
        "lot_volume_shares": {lid: round(s, 4) for lid, s in shares.items()},
        "shares_sum_check": round(total, 6),
        "largest_single_share": round(largest, 4),
        "largest_lot_id": lot_id,
        "volume_share_of_largest_k": round(register.share_of_largest_k(k), 4),
        "k_is_a_count_not_a_volume": True,
    }


def unequal_partition(
    n_samples: int,
    shares: list[float],
    *,
    seed: int = 0,
    labels: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Partition sample indices into lots with **deliberately unequal** volumes.

    Equal shards are the easy case and they do not test the mechanism. A poisoned 21%
    lot and a poisoned 1% lot are different experiments: 1% of the small lot is 0.01%
    of the corpus and may be undetectable, while 1% of the large lot is 0.21% and
    should be found. Stratifying by label when ``labels`` is given keeps each lot
    class-representative, so a detector cannot succeed merely by noticing that one lot
    is missing a class.
    """
    total = sum(shares)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"shares sum to {total}, not 1.0")

    rng = np.random.default_rng(seed)
    idx = np.arange(n_samples)

    if labels is None:
        rng.shuffle(idx)
        out, start = {}, 0
        for i, s in enumerate(shares):
            take = int(round(s * n_samples))
            end = min(start + take, n_samples) if i < len(shares) - 1 else n_samples
            out[f"lot-{i:02d}"] = idx[start:end]
            start = end
        return out

    out = {f"lot-{i:02d}": [] for i in range(len(shares))}
    for cls in np.unique(labels):
        members = idx[labels == cls]
        rng.shuffle(members)
        start = 0
        for i, s in enumerate(shares):
            take = int(round(s * members.size))
            end = min(start + take, members.size) if i < len(shares) - 1 else members.size
            out[f"lot-{i:02d}"].extend(members[start:end].tolist())
            start = end
    return {k: np.array(sorted(v)) for k, v in out.items()}


#: The example share vector: twelve lots running from 21% down to 1%.
EXAMPLE_SHARES = [0.21, 0.17, 0.13, 0.11, 0.09, 0.08, 0.07, 0.05, 0.04, 0.03, 0.01, 0.01]
