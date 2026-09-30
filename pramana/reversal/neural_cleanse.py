"""C3.b -- trigger reversal, in the Neural-Cleanse family.

Explicitly **not claimed as ours**. Clause 2.2.2 names "trigger search or
reconstruction" as a required capability, and the mechanism is mature literature. It is
retained as one of three independent tests, not as a contribution, and a diffuse or
input-dependent trigger defeats it -- which is why ``taxonomy.yaml`` declares warping
and input-aware families unsupported with a reason rather than implying coverage.

Three parameterisations, because a single one is a single assumption:

* **patch_l1** -- a small localised patch; L1 mass penalty recovers a compact mask.
* **blend_linf** -- a full-frame blend at low amplitude; Linf bounds the perturbation.
* **dct_band** -- a frequency-band perturbation, for triggers with no spatial locality.

The per-class result is scored against the **pooled (model x class) null**, with
Benjamini-Hochberg across classes -- not a MAD > 2 rule of thumb. And localisation only
runs when L1 detection has already fired: running 43 class tests unconditionally would
inflate the error rate invisibly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

try:  # torch is an optional extra; the statistics above it are not
    import torch
    import torch.nn.functional as F

    _TORCH = True
except ImportError:  # pragma: no cover - exercised on the core install
    _TORCH = False

__all__ = ["ReversalResult", "reverse_trigger", "reverse_all_classes", "bh_select"]

PARAMETERISATIONS = ("patch_l1", "blend_linf", "dct_band")


@dataclass
class ReversalResult:
    """One class's recovered trigger and its norm."""

    target_class: int
    parameterisation: str
    mask: np.ndarray | None
    pattern: np.ndarray | None
    norm: float
    attack_success_rate: float
    converged: bool
    steps: int

    def as_dict(self) -> dict[str, object]:
        return {
            "class": self.target_class,
            "parameterisation": self.parameterisation,
            "perturbation_norm": round(self.norm, 6),
            "attack_success_rate": round(self.attack_success_rate, 4),
            "converged": self.converged,
            "steps": self.steps,
        }


def _require_torch() -> None:
    if not _TORCH:
        raise ImportError(
            "Trigger reversal needs the 'ml' extra: pip install 'pramana[ml]'. The T1 "
            "statistics, ledger, coverage and report layers run without it by design, "
            "because the review box never sees a model."
        )


def reverse_trigger(
    model: "torch.nn.Module",
    data_loader,
    target_class: int,
    *,
    parameterisation: str = "patch_l1",
    steps: int = 500,
    lr: float = 0.1,
    mask_weight: float = 0.01,
    image_shape: tuple[int, int, int] = (3, 32, 32),
    device: str = "cpu",
) -> ReversalResult:
    """Solve for the minimal perturbation that drives inputs to ``target_class``.

    Optimises a mask and a pattern so that ``x' = (1 - m) * x + m * p`` is classified
    as the target, penalising the mask's size. A class holding an implanted trigger
    admits an anomalously *small* solution -- that is the signal, and it is why the
    norm rather than the attack success rate is the statistic scored against the null.

    Gradients are taken with respect to the **input**, never the weights: nothing here
    retrains the contributed model, which is what keeps reversal inside clause 2.2.6's
    no-retraining baseline.
    """
    _require_torch()
    if parameterisation not in PARAMETERISATIONS:
        raise ValueError(f"unknown parameterisation {parameterisation!r}")

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    c, h, w = image_shape
    mask_raw = torch.zeros(1, 1, h, w, device=device, requires_grad=True)
    pattern_raw = torch.zeros(1, c, h, w, device=device, requires_grad=True)
    opt = torch.optim.Adam([mask_raw, pattern_raw], lr=lr)

    target = torch.tensor([target_class], device=device)
    last_asr, converged = 0.0, False

    for step in range(steps):
        total_loss, hits, seen = 0.0, 0, 0
        for batch in data_loader:
            x = (batch[0] if isinstance(batch, (tuple, list)) else batch).to(device)
            m = torch.sigmoid(mask_raw)
            p = torch.tanh(pattern_raw) * 0.5 + 0.5

            if parameterisation == "blend_linf":
                m = m * 0.2  # a blend is low-amplitude and full-frame by construction
            elif parameterisation == "dct_band":
                p = _dct_band_project(p)

            x_adv = (1 - m) * x + m * p
            logits = model(x_adv)
            ce = F.cross_entropy(logits, target.expand(x.shape[0]))

            reg = m.abs().sum() if parameterisation != "blend_linf" else m.abs().amax()
            loss = ce + mask_weight * reg

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += float(loss.detach())
            hits += int((logits.argmax(1) == target).sum())
            seen += x.shape[0]

        last_asr = hits / max(seen, 1)
        if last_asr > 0.99 and step > 20:
            converged = True
            break

    with torch.no_grad():
        m = torch.sigmoid(mask_raw)
        if parameterisation == "blend_linf":
            m = m * 0.2
        p = torch.tanh(pattern_raw) * 0.5 + 0.5
        norm = float(m.abs().sum()) if parameterisation != "blend_linf" else float(m.abs().amax())

    return ReversalResult(
        target_class=target_class,
        parameterisation=parameterisation,
        mask=m.detach().cpu().numpy(),
        pattern=p.detach().cpu().numpy(),
        norm=norm,
        attack_success_rate=last_asr,
        converged=converged,
        steps=step + 1,
    )


def _dct_band_project(pattern: "torch.Tensor") -> "torch.Tensor":
    """Keep only a mid-frequency band, so the recovered trigger has no spatial locality."""
    freq = torch.fft.rfft2(pattern, norm="ortho")
    h, w = freq.shape[-2:]
    mask = torch.zeros_like(freq.real)
    lo_h, hi_h = h // 8, h // 2
    lo_w, hi_w = w // 8, w // 2
    mask[..., lo_h:hi_h, lo_w:hi_w] = 1.0
    return torch.fft.irfft2(freq * mask, s=pattern.shape[-2:], norm="ortho").clamp(0, 1)


def reverse_all_classes(
    model,
    data_loader,
    n_classes: int,
    *,
    parameterisation: str = "patch_l1",
    top_k: int | None = None,
    cheap_scores: np.ndarray | None = None,
    **kwargs,
) -> list[ReversalResult]:
    """Reverse every class, or the ``top_k`` most suspicious by a cheap pre-score.

    43 classes at roughly 25-40 minutes each does not fit a triage budget, so reversal
    is capped to the top-k by the cheap statistics -- and **the classes we skipped are
    listed in the report** rather than implied. An assessment that quietly tested six
    classes and reported as though it had tested forty-three is the failure this
    parameter exists to make visible.
    """
    classes = list(range(n_classes))
    if top_k is not None and cheap_scores is not None:
        classes = list(np.argsort(-np.asarray(cheap_scores))[:top_k])
    return [
        reverse_trigger(model, data_loader, c, parameterisation=parameterisation, **kwargs)
        for c in classes
    ]


def bh_select(
    p_values: dict[int, float],
    *,
    fdr: float = 0.05,
) -> dict[str, object]:
    """Benjamini-Hochberg across classes, at a declared false-discovery rate.

    Replaces the MAD > 2 rule the original design used. The rank-1 critical value is
    ``alpha / m``, and the report prints it beside the p-value -- comparator pair 2.
    """
    if not p_values:
        return {"rejected": [], "critical_value_at_rank_1": None, "classes_tested": 0}

    m = len(p_values)
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    rejected: list[int] = []
    for i in range(m, 0, -1):
        if ordered[i - 1][1] <= (i / m) * fdr:
            rejected = [cls for cls, _ in ordered[:i]]
            break

    adjusted: dict[int, float] = {}
    running = 1.0
    for i in range(m, 0, -1):
        cls, p = ordered[i - 1]
        running = min(running, p * m / i)
        adjusted[cls] = running

    return {
        "rejected": rejected,
        "critical_value_at_rank_1": fdr / m,
        "classes_tested": m,
        "fdr": fdr,
        "adjusted_q": {c: round(q, 6) for c, q in adjusted.items()},
    }
