"""C9 -- drift versus manipulation (clause 2.2.4).

An earlier design answered this clause with two statistics -- subset-concentration Gini
and a spectral band ratio -- and called it closed. Neither measures the thing the clause
asks about, and both fail in *both* directions:

* **Subset Gini** discriminates whether anomaly is concentrated in few *subsets*. A
  season, a sensor swap or a single deployment theatre **is** a subset, so seasonal
  drift is maximally concentrated by construction and scores like manipulation.
* **Spectral band ratio** discriminates high-frequency content. JPEG re-encoding, a new
  camera and sharpening in a vendor's pipeline all move the spectrum benignly, while
  blended, low-frequency, warping and clean-label triggers carry no high-frequency
  signature at all.

Both are demoted to **corroborating features**: retained because they are cheap and
sometimes informative, but they may no longer carry a disposition on their own.

The replacement, in the order they are trusted:

1. **Contributor concentration.** Benign drift is time-, terrain- or sensor-concentrated
   but **spreads across contributors** -- every vendor collecting in winter sees winter.
   Manipulation is **contributor-concentrated**: it enters through whoever was
   compromised. This is the one discriminator the field does not have, because it
   requires the contributor as a first-class unit. It is also the cheapest: it reuses
   D3's per-source e-values with no new statistic.
2. **Covariate versus concept decomposition.** Drift moves P(x); a trigger moves
   P(y|x) on a small input-conditional region. Not ours -- WATCH (arXiv:2505.04608,
   ICML 2025) does exactly this and we cite it rather than reinvent it.
3. **Input-conditionality.** A trigger's effect survives conditioning on the
   input-space neighbourhood: near-identical clean and triggered inputs diverge in
   output. Drift's effect does not -- it moves with the neighbourhood. This probes the
   definitional property of a backdoor rather than a correlate of one.

Clause 2.2.4's own hedge -- *"where the available evidence supports such a
distinction"* -- is permission to return AMBIGUOUS, and this module uses it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pramana.ladder.concentration import gini

__all__ = ["DriftVerdict", "contributor_concentration", "input_conditionality", "discriminate"]


@dataclass
class DriftVerdict:
    verdict: str  # DRIFT | MANIPULATION | AMBIGUOUS
    primary: str
    scores: dict[str, float] = field(default_factory=dict)
    corroborating: dict[str, float] = field(default_factory=dict)
    rationale: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "verdict": self.verdict,
            "primary_discriminator": self.primary,
            "discriminator_scores": {k: round(v, 4) for k, v in self.scores.items()},
            "corroborating": {k: round(v, 4) for k, v in self.corroborating.items()},
            "verdict_note": self.rationale,
        }


def contributor_concentration(per_source_anomaly: dict[str, float]) -> float:
    """Discriminator 1 -- how concentrated the anomaly is *across contributors*.

    Near 0: every contributor is equally affected, which is what a season or a sensor
    generation change looks like. Near 1: one contributor carries it, which is what a
    compromised supplier looks like.

    Note this is a **different quantity** from D1's within-artefact divergence
    concentration over classes, and the two must not share an implementation -- doing
    so would propagate this statistic's demotion into D1's headline. They are named
    differently on purpose.
    """
    if not per_source_anomaly:
        return 0.0
    return gini(np.array(list(per_source_anomaly.values()), dtype=float))


def covariate_concept_split(
    p_x_shift: float,
    p_y_given_x_shift: float,
) -> tuple[str, float]:
    """Discriminator 2 -- WATCH-style decomposition. Cited, not claimed.

    Returns the dominant component and the fraction of total shift it accounts for.
    Covariate-dominant shift is consistent with drift; concept-dominant shift on a
    small region is what a conditional trigger produces.
    """
    total = p_x_shift + p_y_given_x_shift
    if total <= 0:
        return "none", 0.0
    if p_y_given_x_shift > p_x_shift:
        return "concept", float(p_y_given_x_shift / total)
    return "covariate", float(p_x_shift / total)


def input_conditionality(
    paired_outputs_clean: np.ndarray,
    paired_outputs_probe: np.ndarray,
    neighbourhood_baseline: np.ndarray,
) -> float:
    """Discriminator 3 -- does the effect survive conditioning on the neighbourhood?

    ``paired_outputs_clean`` and ``paired_outputs_probe`` are outputs for near-identical
    inputs differing only by the suspected perturbation. ``neighbourhood_baseline`` is
    the output spread over genuinely neighbouring clean inputs.

    A ratio near 1 means the observed change is what the neighbourhood does anyway --
    drift. A large ratio means near-identical inputs diverge sharply, which is the
    definitional property of a backdoor rather than a correlate of one.
    """
    paired = np.linalg.norm(
        np.asarray(paired_outputs_probe, dtype=float)
        - np.asarray(paired_outputs_clean, dtype=float),
        axis=-1,
    )
    baseline = float(np.median(np.asarray(neighbourhood_baseline, dtype=float))) or 1e-12
    return float(np.median(paired) / baseline)


def discriminate(
    per_source_anomaly: dict[str, float],
    *,
    covariate_shift: float | None = None,
    concept_shift: float | None = None,
    conditionality_ratio: float | None = None,
    shard_gini: float | None = None,
    spectral_band_ratio: float | None = None,
    contributor_threshold: float = 0.6,
    conditionality_threshold: float = 3.0,
) -> DriftVerdict:
    """Combine the three discriminators, in the order they are trusted.

    Returns AMBIGUOUS rather than guessing when the evidence does not support a
    distinction. The clause explicitly permits that, and a system that always answers
    is a system whose answers mean less.
    """
    scores: dict[str, float] = {}
    corroborating: dict[str, float] = {}

    cc = contributor_concentration(per_source_anomaly)
    scores["contributor_concentration"] = cc

    component = None
    if covariate_shift is not None and concept_shift is not None:
        component, share = covariate_concept_split(covariate_shift, concept_shift)
        scores[f"{component}_dominant_share"] = share

    if conditionality_ratio is not None:
        scores["input_conditionality_ratio"] = conditionality_ratio

    if shard_gini is not None:
        corroborating["shard_gini"] = shard_gini
    if spectral_band_ratio is not None:
        corroborating["spectral_band_ratio"] = spectral_band_ratio

    votes: list[str] = []
    votes.append("MANIPULATION" if cc >= contributor_threshold else "DRIFT")
    if component is not None:
        votes.append("MANIPULATION" if component == "concept" else "DRIFT")
    if conditionality_ratio is not None:
        votes.append(
            "MANIPULATION" if conditionality_ratio >= conditionality_threshold else "DRIFT"
        )

    manip = votes.count("MANIPULATION")
    drift = votes.count("DRIFT")

    if manip and not drift:
        verdict, why = "MANIPULATION", "All available discriminators agree."
    elif drift and not manip:
        verdict, why = "DRIFT", "All available discriminators agree."
    else:
        verdict, why = (
            "AMBIGUOUS",
            f"Discriminators disagree ({manip} manipulation, {drift} drift). Clause 2.2.4 "
            f"asks for the distinction only where the evidence supports it, and here it "
            f"does not.",
        )

    if len(votes) == 1:
        why += (
            " Only discriminator 1 was available; the verdict rests on contributor "
            "concentration alone and is reported at that strength."
        )

    return DriftVerdict(
        verdict=verdict,
        primary="contributor_concentration",
        scores=scores,
        corroborating=corroborating,
        rationale=why,
    )
