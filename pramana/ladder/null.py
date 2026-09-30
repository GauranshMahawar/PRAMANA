"""The fitted null -- where every threshold in this system comes from.

The original design used hard-coded constants (MAD > 2, cosine 0.02). They came from
nowhere, which is why they are gone. Every threshold is now an **empirical quantile of
a population of independently-trained clean models**, and the instrument's resolution
is bounded by that population's size:

* 64 null models + the artefact under test = 65 draws, so an L1 detection p-value
  cannot go below **1/65 = 0.01538**. The report prints that floor beside every one.
* Class localisation runs against the **pooled** (model x class) null, 64 x 43 = 2752
  draws, floor 1/2753 = 0.00036, against a BH rank-1 critical value of alpha/43 =
  0.00116. The floor sits 3.2x below the critical value, and that headroom is what
  makes class localisation possible at all.

Two levels, and localisation is **gated on detection**. Running 43 class tests
unconditionally would have inflated the error rate invisibly.

The null is indexed by **(architecture family x training corpus x converter)**, not by
family alone. Applying a GTSRB-fitted null to a model trained elsewhere is a transfer
assumption, and this module makes it a declared one or a refusal -- never a silent
default.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pramana.common import constants as K
from pramana.common.errors import AssessmentUnavailable

__all__ = ["NullIndex", "FittedNull", "TransferReport", "fit_null", "load_null"]


@dataclass(frozen=True)
class NullIndex:
    """What a null is valid for. All three axes, because two is how you get it wrong."""

    family: str
    corpus: str
    converter: str = "unspecified"

    def key(self) -> str:
        return f"{self.family}/{self.corpus}/{self.converter}"

    def matches(self, other: NullIndex) -> bool:
        return (
            self.family == other.family
            and self.corpus == other.corpus
            and (self.converter == other.converter or "unspecified" in (self.converter, other.converter))
        )


@dataclass
class TransferReport:
    """How far a null moves when you change one factor. Measured, not assumed."""

    contrast: str
    factor: str
    median_shift_over_iqr: float
    bootstrap_ci_median_shift: tuple[float, float]
    alpha_tail_shift_over_iqr: float | None = None
    quantile_resolution: str = K.QUANTILE_RESOLUTION_LABEL_AT_N32

    @property
    def exceeds_ceiling(self) -> bool:
        return self.median_shift_over_iqr > K.TRANSFER_CEILING_MEDIAN_SHIFT_OVER_IQR

    def as_dict(self) -> dict[str, Any]:
        return {
            "contrast": self.contrast,
            "factor": self.factor,
            "median_shift_over_iqr": round(self.median_shift_over_iqr, 4),
            "bootstrap_ci_median_shift": [round(x, 4) for x in self.bootstrap_ci_median_shift],
            "alpha_tail_shift_over_iqr": (
                round(self.alpha_tail_shift_over_iqr, 4)
                if self.alpha_tail_shift_over_iqr is not None
                else None
            ),
            "quantile_resolution": self.quantile_resolution,
        }


@dataclass
class FittedNull:
    """An empirical null over a clean-model population.

    ``draws`` holds one statistic per clean model (L1) or one per (model, class) pair
    (L2). Nothing is assumed about its shape -- no Gaussian, no MAD, no z-score. A
    p-value is a rank, and a rank cannot be finer than the population allows.
    """

    index: NullIndex
    level: str  # "L1_detection" | "L2_localisation"
    draws: np.ndarray
    n_models: int
    role: str = "operational_null"
    metric: str = "js"
    population_name: str = ""
    corpus_transfer: TransferReport | None = None
    family_transfer: TransferReport | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    # -- resolution --------------------------------------------------------

    @property
    def n_draws(self) -> int:
        return int(self.draws.size)

    @property
    def p_floor(self) -> float:
        """The finest p-value this null can resolve: 1/(draws + 1)."""
        return 1.0 / (self.n_draws + 1)

    @property
    def p_floor_basis(self) -> str:
        if self.level == "L1_detection":
            return f"1/({self.n_draws}+1); one unmultiplied test at alpha={K.DECLARED_FDR}"
        return f"1/({self.n_draws}+1) = 1/{self.n_draws + 1}"

    @property
    def iqr(self) -> float:
        q75, q25 = np.percentile(self.draws, [75, 25])
        return float(q75 - q25)

    # -- scoring -----------------------------------------------------------

    def p_value(self, statistic: float) -> tuple[float, bool]:
        """Right-tailed empirical p-value, and whether it sits at the floor.

        ``(1 + #{draws >= s}) / (n + 1)`` -- the conservative plus-one form. It never
        returns zero, which is the point: a p-value of zero would be a claim finer
        than any finite population can support.
        """
        exceed = int(np.sum(self.draws >= statistic))
        p = (1.0 + exceed) / (self.n_draws + 1.0)
        return float(p), bool(abs(p - self.p_floor) < 1e-12)

    def quantile(self, q: float) -> float:
        return float(np.quantile(self.draws, q))

    def standardise(self, statistic: float) -> float:
        """Robust standardisation against the null's own median and IQR."""
        med = float(np.median(self.draws))
        spread = self.iqr or 1e-12
        return (statistic - med) / spread

    # -- transfer ----------------------------------------------------------

    def check_applicable(self, target: NullIndex) -> None:
        """Refuse rather than guess when the null does not cover the artefact.

        This is the mechanism behind "we would rather ship a system that declines to
        score than one that scores everything with an unvalidated constant."
        """
        if self.index.matches(target):
            return
        delta = None
        if self.index.family != target.family and self.family_transfer:
            delta = self.family_transfer
        elif self.index.corpus != target.corpus and self.corpus_transfer:
            delta = self.corpus_transfer

        if delta is None or delta.exceeds_ceiling:
            raise AssessmentUnavailable(
                "no_fitted_null",
                f"null is indexed {self.index.key()}, artefact is {target.key()}"
                + (
                    f"; measured median shift {delta.median_shift_over_iqr:.2f} exceeds the "
                    f"declared ceiling {K.TRANSFER_CEILING_MEDIAN_SHIFT_OVER_IQR}"
                    if delta
                    else "; no transfer delta has been measured for this contrast"
                ),
            )

    # -- persistence -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """The publishable artefact.

        This -- not the 64 sets of weights -- is the scientific object. It is small,
        it goes in Git, and a third party can check every p-value in every report
        against it.
        """
        return {
            "index": {
                "family": self.index.family,
                "corpus": self.index.corpus,
                "converter": self.index.converter,
            },
            "level": self.level,
            "role": self.role,
            "metric": self.metric,
            "population_name": self.population_name,
            "n_models": self.n_models,
            "n_draws": self.n_draws,
            "p_floor": self.p_floor,
            "p_floor_basis": self.p_floor_basis,
            "quantiles": {
                str(q): round(self.quantile(q), 8) for q in (0.5, 0.75, 0.9, 0.95, 0.99)
            },
            "median": float(np.median(self.draws)),
            "iqr": self.iqr,
            "draws": [round(float(x), 8) for x in self.draws],
            "corpus_transfer": self.corpus_transfer.as_dict() if self.corpus_transfer else None,
            "family_transfer": self.family_transfer.as_dict() if self.family_transfer else None,
            "meta": self.meta,
        }

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def load_null(path: str | Path) -> FittedNull:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    idx = NullIndex(**d["index"])
    n = FittedNull(
        index=idx,
        level=d["level"],
        draws=np.asarray(d["draws"], dtype=float),
        n_models=d["n_models"],
        role=d.get("role", "operational_null"),
        metric=d.get("metric", "js"),
        population_name=d.get("population_name", ""),
        meta=d.get("meta", {}),
    )
    for key, attr in (("corpus_transfer", "corpus_transfer"), ("family_transfer", "family_transfer")):
        raw = d.get(key)
        if raw:
            setattr(
                n,
                attr,
                TransferReport(
                    contrast=raw["contrast"],
                    factor=raw["factor"],
                    median_shift_over_iqr=raw["median_shift_over_iqr"],
                    bootstrap_ci_median_shift=tuple(raw["bootstrap_ci_median_shift"]),
                    alpha_tail_shift_over_iqr=raw.get("alpha_tail_shift_over_iqr"),
                    quantile_resolution=raw.get("quantile_resolution", K.QUANTILE_RESOLUTION_LABEL_AT_N32),
                ),
            )
    return n


def fit_null(
    statistics: np.ndarray,
    index: NullIndex,
    *,
    level: str = "L1_detection",
    n_models: int | None = None,
    role: str = "operational_null",
    metric: str = "js",
    population_name: str = "",
) -> FittedNull:
    """Fit an empirical null from clean-model statistics.

    For L1 pass one statistic per clean model. For L2 pass the **per-class
    standardised** statistics pooled across models and classes -- standardising before
    pooling matters, because classes differ in difficulty and pooling raw values would
    let an easy class dominate the tail.
    """
    draws = np.asarray(statistics, dtype=float).ravel()
    if draws.size == 0:
        raise ValueError("cannot fit a null from an empty population")
    return FittedNull(
        index=index,
        level=level,
        draws=draws,
        n_models=n_models if n_models is not None else int(draws.size),
        role=role,
        metric=metric,
        population_name=population_name or f"{index.family}-{index.corpus}-clean-n{draws.size}",
    )


def measure_transfer(
    null_a: FittedNull,
    null_b: FittedNull,
    *,
    factor: str,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> TransferReport:
    """Measure how far a null moves between two populations, in IQR units.

    Quoted at the **median**, with a bootstrap interval. At n=32 per contrast arm the
    alpha-tail is the ~1.6th order statistic, so it is reported only with its interval
    and labelled ``under_resolved_at_n32``. A *large* delta will be visible and a
    *small* one will not be provable -- an asymmetry that runs conservatively.
    """
    rng = np.random.default_rng(seed)
    a, b = null_a.draws, null_b.draws
    scale = null_a.iqr or 1e-12

    observed = float((np.median(b) - np.median(a)) / scale)

    boot = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sa = rng.choice(a, size=a.size, replace=True)
        sb = rng.choice(b, size=b.size, replace=True)
        boot[i] = (np.median(sb) - np.median(sa)) / (np.percentile(sa, 75) - np.percentile(sa, 25) or 1e-12)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    alpha_tail = None
    if a.size >= 20 and b.size >= 20:
        alpha_tail = float((np.percentile(b, 95) - np.percentile(a, 95)) / scale)

    return TransferReport(
        contrast=f"{null_a.index.key()} vs {null_b.index.key()}",
        factor=factor,
        median_shift_over_iqr=observed,
        bootstrap_ci_median_shift=(float(lo), float(hi)),
        alpha_tail_shift_over_iqr=alpha_tail,
    )


def validate_null_quality(null: FittedNull, *, seed: int = 0) -> dict[str, Any]:
    """The three criteria a fitted null must pass before any p-value is issued.

    Checked because a null that has not stabilised silently invalidates every
    threshold derived from it, and the failure is invisible in the output.
    """
    rng = np.random.default_rng(seed)
    draws = null.draws
    n = draws.size

    boot = np.array(
        [np.quantile(rng.choice(draws, n, replace=True), 0.95) for _ in range(2000)]
    )
    half = float((np.percentile(boot, 97.5) - np.percentile(boot, 2.5)) / 2.0)
    ratio = half / (null.iqr or 1e-12)

    perm = rng.permutation(draws)
    h1, h2 = perm[: n // 2], perm[n // 2 :]
    est = [float(np.quantile(h1, 0.95)), float(np.quantile(h2, 0.95))]
    within = abs(est[0] - est[1]) <= 2 * half

    fit_n = int(n * 0.75)
    fit, hold = perm[:fit_n], perm[fit_n:]
    thr = float(np.quantile(fit, 0.95))
    exceeding = int(np.sum(hold > thr))
    fail_at = max(1, int(np.ceil(0.25 * hold.size)))

    return {
        "bootstrap_ci_halfwidth_over_iqr": round(ratio, 4),
        "criterion_a": "<= 0.25",
        "criterion_a_pass": ratio <= 0.25,
        "disjoint_halves": {
            "n": [int(h1.size), int(h2.size)],
            "estimates": [round(e, 4) for e in est],
            "within_ci": within,
            "pass": within,
        },
        "holdout_coverage": {
            "fit_n": fit_n,
            "holdout_n": int(hold.size),
            "exceeding_l1_threshold": exceeding,
            "fail_at": fail_at,
            "pass": exceeding < fail_at,
        },
    }
