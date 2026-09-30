"""D1 -- cross-rung divergence.

The measurement is deliberately **not** "how far did the fingerprint move". That was
the original claim and it was inverted: a quantisation-conditioned backdoor is created
*by* the conversion, so it is arbitrarily small in any magnitude metric, and "small
distance under quantisation implies benign" is exactly what such an attacker engineers
for.

What survives is a different question -- *which rungs disagree, on which probes, in
which direction* -- scored against a null fitted from clean models, with a
**concentration** condition. Cross-precision differential testing is prior art
(DiffChaser IJCAI'19, DiverGet EMSE'22, PrecisionDiff). Using the disagreement as
evidence of an implanted modification, with a fitted null and a concentration test,
is the part our sweeps found unoccupied.

DiverGet's own result is the reason the concentration condition exists: search can
*manufacture* cross-precision disagreement in a model with no backdoor in it. So
divergence per se proves nothing, and only the null and the concentration structure
carry the finding.

Everything here is NumPy. Model execution lives in :mod:`pramana.ladder.runner`, so
the statistics can be tested without a GPU, without torch, and inside CI.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "DivergenceProfile",
    "probe_divergence",
    "per_class_divergence",
    "margin",
    "top1_disagreement",
]


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def margin(logits: np.ndarray) -> np.ndarray:
    """Top-1 minus top-2 logit, per probe.

    Chosen because arXiv:2204.04220 measures FP32-vs-quantised disagreement across 42
    shifted sets and finds *margin* the best disagreement indicator. Probes near a
    decision boundary have small margin and are where conversion flips a label, so
    this is also how Battery A selects probes.
    """
    part = np.partition(logits, -2, axis=-1)
    return part[..., -1] - part[..., -2]


def top1_disagreement(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Boolean per-probe: did the argmax change between two rungs?"""
    return a.argmax(axis=-1) != b.argmax(axis=-1)


@dataclass
class DivergenceProfile:
    """Per-probe, per-class divergence between two rungs of one artefact.

    This object is what the null is fitted over and what the concentration statistic
    is computed from. It compares an artefact **against itself** at another precision,
    which is why D1 is the one statistic immune to the null-transfer inversion: no
    external null is needed to establish that two rungs disagree, only to grade how
    surprising the magnitude is.
    """

    pair: str
    per_probe: np.ndarray  # (n_probes,)  scalar divergence per probe
    flipped: np.ndarray  # (n_probes,)  bool, argmax changed
    per_class: np.ndarray  # (n_classes,) EXCESS divergence mass per class
    per_class_raw: np.ndarray  # (n_classes,) total mass, kept for diagnostics only
    baseline: float  # within-artefact median per-probe divergence
    n_probes: int
    n_classes: int

    @property
    def probes_diverging(self) -> int:
        return int(self.flipped.sum())

    @property
    def total_mass(self) -> float:
        return float(self.per_class.sum())

    @property
    def mean_divergence(self) -> float:
        return float(self.per_probe.mean()) if self.n_probes else 0.0

    def summary(self) -> dict[str, float | int]:
        return {
            "pair": self.pair,
            "probes_diverging": self.probes_diverging,
            "probes_total": self.n_probes,
            "mean_divergence": self.mean_divergence,
            "total_mass": self.total_mass,
        }


def probe_divergence(
    logits_a: np.ndarray,
    logits_b: np.ndarray,
    *,
    pair: str,
    metric: str = "js",
) -> DivergenceProfile:
    """Divergence between two rungs over the Battery A probe bank.

    Parameters
    ----------
    logits_a, logits_b
        ``(n_probes, n_classes)`` logit matrices for the two rungs. One forward pass
        per probe per rung; no gradients, no retraining -- a T1 mechanism by
        construction, which is what satisfies clause 2.2.6.
    metric
        ``js`` (Jensen-Shannon over softmax, bounded and symmetric -- the default),
        ``l1``, or ``cosine``. The metric is recorded in the report because changing it
        moves every threshold fitted under the old one.

    Class attribution -- and why it is the EXCESS, not the total
    ------------------------------------------------------------
    A probe contributes to the class it moved **towards** at rung ``b``, because a
    quantisation-conditioned backdoor concentrates on the target class.

    The mass attributed is the probe's divergence **above the artefact's own median**,
    clipped at zero -- not its total divergence. The first implementation used the
    total and it did not work: every probe carries some conversion noise, that noise
    spreads across all classes, and the resulting floor swamped the signal. A QCB
    against 200 probes and 43 classes still measured a Gini of 0.57 and 28 effective
    classes, which would have failed the concentration condition on a genuinely
    poisoned artefact.

    Subtracting the within-artefact median fixes it and keeps the property that makes
    D1 valuable: the baseline is computed from the artefact against **itself**, so no
    external null is needed to establish concentration -- only to grade magnitude. The
    median is used rather than the mean because a successful QCB is exactly the case
    where the mean is dragged by the tail it is supposed to detect.
    """
    if logits_a.shape != logits_b.shape:
        raise ValueError(f"rung shape mismatch: {logits_a.shape} vs {logits_b.shape}")
    if logits_a.ndim != 2:
        raise ValueError("expected (n_probes, n_classes) logit matrices")

    n_probes, n_classes = logits_a.shape
    pa, pb = _softmax(logits_a), _softmax(logits_b)

    if metric == "js":
        m = 0.5 * (pa + pb)
        eps = 1e-12
        kl_a = np.sum(pa * (np.log(pa + eps) - np.log(m + eps)), axis=-1)
        kl_b = np.sum(pb * (np.log(pb + eps) - np.log(m + eps)), axis=-1)
        per_probe = 0.5 * (kl_a + kl_b)
    elif metric == "l1":
        per_probe = np.abs(pa - pb).sum(axis=-1)
    elif metric == "cosine":
        num = (logits_a * logits_b).sum(axis=-1)
        den = np.linalg.norm(logits_a, axis=-1) * np.linalg.norm(logits_b, axis=-1) + 1e-12
        per_probe = 1.0 - num / den
    else:
        raise ValueError(f"unknown divergence metric {metric!r}")

    flipped = top1_disagreement(logits_a, logits_b)
    target = logits_b.argmax(axis=-1)

    baseline = float(np.median(per_probe))
    excess = np.clip(per_probe - baseline, 0.0, None)

    per_class = np.zeros(n_classes, dtype=float)
    np.add.at(per_class, target, excess)

    per_class_raw = np.zeros(n_classes, dtype=float)
    np.add.at(per_class_raw, target, per_probe)

    return DivergenceProfile(
        pair=pair,
        per_probe=per_probe,
        flipped=flipped,
        per_class=per_class,
        per_class_raw=per_class_raw,
        baseline=baseline,
        n_probes=n_probes,
        n_classes=n_classes,
    )


def per_class_divergence(profile: DivergenceProfile, normalise: bool = True) -> np.ndarray:
    """Class-wise divergence mass, optionally normalised to sum to one."""
    v = profile.per_class
    if normalise:
        total = v.sum()
        return v / total if total > 0 else v
    return v


def ladder_profiles(
    rung_logits: dict[str, np.ndarray],
    *,
    reference: str = "fp32",
    metric: str = "js",
) -> list[DivergenceProfile]:
    """Divergence of every rung against the reference rung.

    The reference defaults to FP32 because that is the build everyone else tests. The
    finding is not that a rung differs from FP32 -- ordinary conversion does -- but
    that the difference is *concentrated* and sits outside the fitted null.
    """
    if reference not in rung_logits:
        raise ValueError(f"reference rung {reference!r} absent from {sorted(rung_logits)}")
    ref = rung_logits[reference]
    return [
        probe_divergence(ref, logits, pair=f"{reference}→{rung}", metric=metric)
        for rung, logits in rung_logits.items()
        if rung != reference
    ]
