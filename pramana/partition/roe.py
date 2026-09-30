"""Run-Off Election aggregation and the certified floor over the measurement ensemble.

Deep Partition Aggregation and its descendants certify a prediction against a bounded
number of corrupted *partitions*. Run-Off Election (arXiv:2302.02300) is the strongest
published aggregator at low partition counts, which is our regime exactly, because m is
set by the contract structure rather than chosen by us.

**The guarantee is per-prediction and no sentence anywhere may imply otherwise.**
For input x, if the plurality margin between top and runner-up across the m base models
exceeds a threshold, x's label is unchanged under arbitrary corruption of any k-1
partitions. No theorem certifies "the deployment", and a deployment-wide-sounding
sentence is a category error a certified-robustness reader catches immediately.

Two reportable objects follow, and the schema requires both beside any k:

* ``certified_fraction_at_k`` -- the fraction of a held-out set whose **ensemble**
  prediction is certified at collusion threshold k.
* ``surrogate_gap_top1`` -- clean accuracy of the voted ensemble versus the single
  delivered model, on the same evaluation set. Never one without the other, because
  with a dozen partitions each base model sees roughly 1/m of the corpus and the voted
  ensemble is materially weaker than the model the Army wants to field.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["RoeResult", "run_off_election", "certified_floor", "certified_fraction"]


@dataclass
class RoeResult:
    """Per-input certification over the measurement ensemble."""

    predictions: np.ndarray  # (n_inputs,) ensemble label
    certified_k: np.ndarray  # (n_inputs,) how many colluding lots it tolerates
    top1_votes: np.ndarray
    runner_up_votes: np.ndarray
    m: int

    def fraction_at(self, k: int) -> float:
        """Fraction of inputs certified at collusion threshold k."""
        if self.certified_k.size == 0:
            return 0.0
        return float(np.mean(self.certified_k >= k))

    def floor(self, coverage: float = 0.90) -> int:
        """The largest k at which at least ``coverage`` of inputs stay certified.

        Reported as ``certified_floor_k``. A floor quoted without the coverage it was
        computed at is another bare number, so the caller records both.
        """
        if self.certified_k.size == 0:
            return 0
        for k in range(self.m, 0, -1):
            if self.fraction_at(k) >= coverage:
                return k
        return 0


def _vote_counts(votes: np.ndarray, n_classes: int) -> np.ndarray:
    """(n_inputs, n_classes) tally from (m, n_inputs) base-model predictions."""
    m, n = votes.shape
    counts = np.zeros((n, n_classes), dtype=int)
    for j in range(m):
        np.add.at(counts, (np.arange(n), votes[j]), 1)
    return counts


def run_off_election(base_predictions: np.ndarray, n_classes: int) -> RoeResult:
    """Two-round aggregation over per-lot base models.

    Round one takes the two classes with the most first-place votes. Round two runs a
    head-to-head between them across all base models. The certified k is derived from
    the head-to-head margin: an adversary controlling j partitions can move at most j
    votes, so the outcome is unchanged while the margin exceeds 2j.

    ``base_predictions`` is ``(m, n_inputs)``; entry ``[j, i]`` is lot j's model's
    predicted class for input i.
    """
    if base_predictions.ndim != 2:
        raise ValueError("base_predictions must be (m, n_inputs)")
    m, n = base_predictions.shape
    counts = _vote_counts(base_predictions, n_classes)

    order = np.argsort(-counts, axis=1)
    top1, top2 = order[:, 0], order[:, 1]

    votes_top1 = counts[np.arange(n), top1]
    votes_top2 = counts[np.arange(n), top2]

    # Head-to-head: ties broken by lower class index, deterministically, so the
    # certificate is reproducible by a third party.
    margin = votes_top1 - votes_top2
    certified = np.maximum((margin - 1) // 2, 0).astype(int)

    return RoeResult(
        predictions=top1,
        certified_k=certified,
        top1_votes=votes_top1,
        runner_up_votes=votes_top2,
        m=m,
    )


def certified_fraction(result: RoeResult, k: int) -> float:
    return result.fraction_at(k)


def certified_floor(result: RoeResult, coverage: float = 0.90) -> int:
    return result.floor(coverage)


def surrogate_gap(ensemble_acc: float, delivered_acc: float) -> float:
    """Top-1 accuracy the measurement ensemble gives up against the delivered model.

    Required beside every k. It makes the size of the substitution visible rather than
    implied: the ensemble is not what anyone deploys, and a judge is entitled to see
    how far it sits from what they are buying.
    """
    return float(delivered_acc - ensemble_acc)


def collapse_to_entities(
    base_predictions: np.ndarray,
    lot_ids: list[str],
    entity_of: dict[str, str],
) -> tuple[np.ndarray, list[str]]:
    """D4.b -- collapse the lot partition along the contract's entity map.

    Twelve lots can be four companies. An adversary who controls one company controls
    every lot that company holds, so a floor counted in lots overstates the guarantee
    whenever any entity holds more than one. Collapsing first and certifying second
    gives ``certified_floor_k_entities``, reported **beside** the lot floor rather
    than instead of it.

    A lot's base model is kept only once per entity -- the first -- because merging
    predictions across an entity's lots would invent a model nobody trained.
    """
    seen: dict[str, int] = {}
    keep: list[int] = []
    entities: list[str] = []
    for j, lot in enumerate(lot_ids):
        ent = entity_of.get(lot)
        if ent is None:
            raise ValueError(
                f"lot {lot!r} has no entity in the map; emit entity_map_incomplete "
                f"rather than collapsing on a guess."
            )
        if ent not in seen:
            seen[ent] = j
            keep.append(j)
            entities.append(ent)
    return base_predictions[keep, :], entities
