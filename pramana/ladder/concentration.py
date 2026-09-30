"""The concentration condition -- the part of D1 that is load-bearing.

The unifying thesis, in one line: **the distinction is not magnitude, it is shape.**

We used to say benign change is *small* and adversarial change is *large*.
Quantisation-conditioned backdoors killed that, because a backdoor can be created by
the conversion and is therefore arbitrarily small in any magnitude metric. What
survives is that benign change is **diffuse** -- INT8 conversion perturbs many classes
slightly -- while adversarial change is **concentrated**: a trigger carves a few
specific places.

This is also the answer to DiverGet. Search-based methods manufacture cross-precision
disagreement in clean models, but that disagreement is *diffuse* -- they find it
wherever the decision surfaces are closest, which is spread across the class space.
Concentration is what search does not naturally manufacture.

**The operating point is not measured.** ``gini >= 0.70`` over ``<= 3`` classes is an
assumption that was once written as a finding. Until experiment E10 sweeps the attack
across 1, 2, 3, 4, 6 and all classes at graded amplitudes, every report quoting it
carries ``concentration_operating_point_status: "predicted"`` and no disposition may
rest on it alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pramana.common import constants as K

__all__ = ["ConcentrationResult", "gini", "effective_classes", "assess_concentration"]


def gini(values: np.ndarray) -> float:
    """Gini coefficient of a non-negative mass vector.

    0 means the mass is spread perfectly evenly across classes (diffuse -- what benign
    quantisation error looks like); 1 means all of it sits on one class (concentrated
    -- what a targeted trigger looks like).

    Computed from the sorted cumulative form, which is exact rather than the
    pairwise-difference approximation, because this number decides dispositions.
    """
    v = np.asarray(values, dtype=float).ravel()
    if v.size == 0:
        return 0.0
    if np.any(v < 0):
        raise ValueError("Gini is undefined for negative mass.")
    total = v.sum()
    if total <= 0:
        return 0.0
    v = np.sort(v)
    n = v.size
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * v)) / (n * total) - (n + 1.0) / n)


def effective_classes(values: np.ndarray, coverage: float = 0.90) -> int:
    """How many classes hold ``coverage`` of the divergence mass.

    Reported beside the Gini because the two answer different questions and a judge
    will ask both. Gini says *how unequal*; this says *how many*. A Gini of 0.9 over
    43 classes where the mass sits on two is a very different artefact from one where
    it sits on eleven.
    """
    v = np.asarray(values, dtype=float).ravel()
    total = v.sum()
    if total <= 0:
        return 0
    ordered = np.sort(v)[::-1]
    cumulative = np.cumsum(ordered) / total
    return int(np.searchsorted(cumulative, coverage) + 1)


@dataclass
class ConcentrationResult:
    """Everything the report needs about the shape of the divergence."""

    gini: float
    classes_at_90pct: int
    dominant_class: int | None
    dominant_share: float
    threshold_gini: float
    threshold_max_classes: int
    condition_met: bool
    status: str
    source: str
    fpr_vs_null: float | None = None

    def as_report_fields(self) -> dict[str, object]:
        return {
            "divergence_concentration_gini": round(self.gini, 4),
            "divergence_concentration_classes": self.classes_at_90pct,
            "dominant_class": self.dominant_class,
            "dominant_class_share": round(self.dominant_share, 4),
            "concentration_operating_point": (
                f"gini >= {self.threshold_gini} over <= {self.threshold_max_classes} classes"
            ),
            "concentration_operating_point_status": self.status,
            "concentration_operating_point_source": self.source,
            "concentration_operating_point_fpr_vs_null": self.fpr_vs_null,
            "concentration_condition_met": self.condition_met,
            "concentration_is_within_artefact": True,
        }


def assess_concentration(
    per_class_mass: np.ndarray,
    *,
    threshold_gini: float = K.CONCENTRATION_GINI_THRESHOLD_PREDICTED,
    max_classes: int = K.CONCENTRATION_MAX_CLASSES_PREDICTED,
    status: str = K.CONCENTRATION_OPERATING_POINT_STATUS_DEFAULT,
    source: str = "E10 (classes affected x trigger amplitude) surface",
    fpr_vs_null: float | None = None,
) -> ConcentrationResult:
    """Apply the concentration condition to a per-class divergence mass vector.

    ``status`` defaults to ``"predicted"`` on purpose. A caller that wants to claim a
    measured operating point has to pass it explicitly, which means somebody had to
    decide to say so -- and E10 has to have run.
    """
    v = np.asarray(per_class_mass, dtype=float).ravel()
    g = gini(v)
    n_eff = effective_classes(v)
    total = v.sum()

    dominant = int(np.argmax(v)) if total > 0 else None
    dominant_share = float(v.max() / total) if total > 0 else 0.0

    met = bool(g >= threshold_gini and 0 < n_eff <= max_classes)

    return ConcentrationResult(
        gini=g,
        classes_at_90pct=n_eff,
        dominant_class=dominant,
        dominant_share=dominant_share,
        threshold_gini=threshold_gini,
        threshold_max_classes=max_classes,
        condition_met=met,
        status=status,
        source=source,
        fpr_vs_null=fpr_vs_null,
    )


def interpret(condition_met: bool, detection_fires: bool) -> str:
    """The ``interpretation`` field.

    Both halves must hold. Divergence alone is DiverGet; concentration alone could be a
    class the converter simply handles badly. The finding is the conjunction.
    """
    if detection_fires and condition_met:
        return "concentrated_divergence_qcb_candidate"
    if detection_fires and not condition_met:
        return "diffuse_divergence_consistent_with_ordinary_conversion"
    if not detection_fires and condition_met:
        return "concentrated_but_within_null_no_finding"
    return "no_finding"
