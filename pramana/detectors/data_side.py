"""C2' -- the four data-side detectors named in clause 2.2.1.

Trigger injection, label flipping, near-duplicate flooding, out-of-distribution
insertion. Each emits a **per-sample e-value** against a calibrated null, and all four
roll up to the source through one aggregation rule rather than four ad-hoc ones -- the
clause asks for source-level aggregation, and one rule for four classes is what makes
that a mechanism rather than a coincidence.

An e-value here is the likelihood ratio of "this sample is anomalous" against the
clean-population null, floored at a small positive number so that a single benign
sample can never drive a merged average to zero.

None of these detectors is claimed as ours. Activation clustering is IBM ART's,
perceptual hashing is standard, Mahalanobis OOD scoring is standard. What the project
claims is the *denomination*: the roll-up to a contributor under a declared
false-discovery rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "DetectorOutput",
    "trigger_injection_score",
    "label_consistency_score",
    "duplicate_flood_score",
    "ood_mahalanobis_score",
    "run_all",
]

_EPS = 1e-6


def _to_evalues(scores: np.ndarray, null_scores: np.ndarray) -> np.ndarray:
    """Convert raw anomaly scores to e-values against an empirical null.

    ``e_i = 1 / max(p_i, 1/(n+1))`` where ``p_i`` is the right-tailed empirical
    p-value. The floor on ``p`` is the null's own resolution limit, so an e-value can
    never exceed ``n+1`` -- the instrument does not manufacture evidence finer than
    the population it was calibrated on.
    """
    null = np.asarray(null_scores, dtype=float).ravel()
    n = null.size
    if n == 0:
        return np.ones_like(scores, dtype=float)
    floor_p = 1.0 / (n + 1.0)
    exceed = (null[None, :] >= np.asarray(scores, dtype=float)[:, None]).sum(axis=1)
    p = np.maximum((1.0 + exceed) / (n + 1.0), floor_p)
    return 1.0 / p


@dataclass
class DetectorOutput:
    name: str
    e_values: np.ndarray
    raw_scores: np.ndarray
    null_size: int
    max_possible_evalue: float
    note: str = ""

    def summary(self) -> dict[str, object]:
        return {
            "detector": self.name,
            "n_samples": int(self.e_values.size),
            "mean_e_value": round(float(self.e_values.mean()), 4),
            "max_e_value": round(float(self.e_values.max()), 4),
            "max_possible_e_value": round(self.max_possible_evalue, 4),
            "null_size": self.null_size,
            "note": self.note,
        }


def trigger_injection_score(
    features: np.ndarray,
    null_features: np.ndarray,
    *,
    n_components: int = 8,
) -> DetectorOutput:
    """Spectral-signature style score for trigger-carrying samples.

    Poisoned samples carrying a common trigger share a direction in feature space, so
    they project unusually far along the leading singular vectors of the centred
    feature matrix. Cheap, no gradients, no retraining -- a T1/T2 mechanism.
    """
    X = np.asarray(features, dtype=float)
    Xc = X - X.mean(axis=0, keepdims=True)
    k = min(n_components, min(Xc.shape) - 1) or 1
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    scores = np.linalg.norm(Xc @ vt[:k].T, axis=1)

    Nc = np.asarray(null_features, dtype=float)
    Nc = Nc - Nc.mean(axis=0, keepdims=True)
    null_scores = np.linalg.norm(Nc @ vt[:k].T, axis=1)

    return DetectorOutput(
        name="trigger_injection",
        e_values=_to_evalues(scores, null_scores),
        raw_scores=scores,
        null_size=null_scores.size,
        max_possible_evalue=null_scores.size + 1.0,
        note="Spectral signature over the leading singular directions.",
    )


def label_consistency_score(
    features: np.ndarray,
    labels: np.ndarray,
    class_centroids: dict[int, np.ndarray],
    null_distances: np.ndarray,
) -> DetectorOutput:
    """Distance from a sample's *claimed* class centroid.

    Detects label flipping and systematic mislabelling alike. The two are the same
    signal and differ only in intent -- which is why this detector also underwrites
    the annotation-bias reading in ``taxonomy.yaml``, at AMBER strength, with its own
    pre-registered falsifier in ``experiments/e13_bias_transfer.py``.
    """
    X = np.asarray(features, dtype=float)
    y = np.asarray(labels)
    scores = np.empty(X.shape[0], dtype=float)
    for i in range(X.shape[0]):
        centroid = class_centroids.get(int(y[i]))
        scores[i] = np.linalg.norm(X[i] - centroid) if centroid is not None else 0.0
    return DetectorOutput(
        name="label_consistency",
        e_values=_to_evalues(scores, null_distances),
        raw_scores=scores,
        null_size=np.asarray(null_distances).size,
        max_possible_evalue=np.asarray(null_distances).size + 1.0,
        note="Distance to the claimed class centroid; flags flips and systematic mislabelling alike.",
    )


def duplicate_flood_score(
    hashes: np.ndarray,
    *,
    hamming_threshold: int = 8,
    null_cluster_sizes: np.ndarray | None = None,
) -> DetectorOutput:
    """Near-duplicate flooding, scored by the size of a sample's duplicate cluster.

    ``hashes`` is ``(n_samples, n_bits)`` of 0/1 perceptual-hash bits. A sample in a
    large near-duplicate cluster scores high. Note the interaction recorded in the
    taxonomy: the ``partition_purity`` rule assigns cross-lot near-duplicates to a
    single shard, trading source resolution for a sound partition, so this detector's
    attribution ceiling is ``statistically_attributed`` and cannot be raised.
    """
    H = np.asarray(hashes, dtype=np.uint8)
    n = H.shape[0]
    scores = np.zeros(n, dtype=float)
    for i in range(n):
        dist = np.count_nonzero(H != H[i], axis=1)
        scores[i] = float(np.count_nonzero(dist <= hamming_threshold) - 1)

    null = (
        np.asarray(null_cluster_sizes, dtype=float)
        if null_cluster_sizes is not None
        else np.zeros(max(n, 32), dtype=float)
    )
    return DetectorOutput(
        name="duplicate_flood",
        e_values=_to_evalues(scores, null),
        raw_scores=scores,
        null_size=null.size,
        max_possible_evalue=null.size + 1.0,
        note=f"Near-duplicate cluster size at Hamming <= {hamming_threshold}.",
    )


def ood_mahalanobis_score(
    features: np.ndarray,
    reference_mean: np.ndarray,
    reference_cov_inv: np.ndarray,
    null_distances: np.ndarray,
) -> DetectorOutput:
    """Out-of-distribution insertion, scored by Mahalanobis distance.

    **The reference must be disjoint from the shards under assessment.** Train the
    backbone on the procurement corpus and a flooded shard becomes part of the
    reference the detector measures deviation *from*: flood hard enough and the flood
    *becomes* the mode, so the clean data is reported as the outlier and detector power
    falls as the attack succeeds. The caller asserts disjointness in
    ``declared_assumptions`` and the admission gate refuses with
    ``backbone_corpus_contaminated`` above a pre-committed overlap ceiling.
    """
    X = np.asarray(features, dtype=float) - np.asarray(reference_mean, dtype=float)
    scores = np.sqrt(np.einsum("ij,jk,ik->i", X, reference_cov_inv, X))
    return DetectorOutput(
        name="ood_mahalanobis",
        e_values=_to_evalues(scores, null_distances),
        raw_scores=scores,
        null_size=np.asarray(null_distances).size,
        max_possible_evalue=np.asarray(null_distances).size + 1.0,
        note="Mahalanobis distance in a frozen, pre-procurement in-house backbone.",
    )


def run_all(
    features: np.ndarray,
    labels: np.ndarray,
    hashes: np.ndarray,
    *,
    null_features: np.ndarray,
    class_centroids: dict[int, np.ndarray],
    null_label_distances: np.ndarray,
    reference_mean: np.ndarray,
    reference_cov_inv: np.ndarray,
    null_ood_distances: np.ndarray,
    null_cluster_sizes: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Run all four detectors over one shard and return e-values keyed by detector.

    The return shape is exactly what :func:`pramana.aggregate.evalues.aggregate_sources`
    consumes, so there is one path from sample to supplier verdict and no opportunity
    for a detector to be merged by a different rule than the others.
    """
    return {
        "trigger": trigger_injection_score(features, null_features).e_values,
        "label_flip": label_consistency_score(
            features, labels, class_centroids, null_label_distances
        ).e_values,
        "dup_flood": duplicate_flood_score(hashes, null_cluster_sizes=null_cluster_sizes).e_values,
        "ood": ood_mahalanobis_score(
            features, reference_mean, reference_cov_inv, null_ood_distances
        ).e_values,
    }
