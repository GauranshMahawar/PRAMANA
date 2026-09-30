"""The fake-quant surrogate, and the gate that stops it becoming the evidence.

Reversal cannot run against the INT8 build. It is gradient-based, and a quantised
ONNX or TorchScript graph exposes no input gradients: ``QuantizeLinear`` and
``DequantizeLinear`` are not differentiable, and ONNX Runtime carries no autograd. A
defence that claims to optimise through a non-differentiable graph has not been
implemented.

So three steps, and **which step produces the evidence is the whole point**:

a. Rebuild the deployed quantisation in PyTorch as **fake-quant with straight-through
   estimators**, from the *delivered* per-layer scales and zero-points -- never
   re-calibrated ones. Re-calibrating would produce a surrogate for a conversion
   nobody performed.
b. **Certify the surrogate.** ``surrogate_exact_agreement`` and
   ``surrogate_mean_logit_distance`` are measured against the real INT8 artefact over
   the Battery A probe bank and printed beside their pre-committed floors on every
   report containing a reversal result. Below the floor, reversal does not run at all:
   ``assessment_unavailable: surrogate_not_faithful``.
c. Reverse on the surrogate, then **verify the recovered trigger by measuring attack
   success on the real INT8 artefact** -- forward-pass only, and therefore always
   available.

The evidence that reaches the report is (c), on the deployed binary. Step (a) is
search, and **search is not evidence.**
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pramana.common import constants as K
from pramana.common.errors import AssessmentUnavailable

try:
    import torch
    import torch.nn as nn

    _TORCH = True
except ImportError:  # pragma: no cover
    _TORCH = False

__all__ = ["ScaleTable", "FakeQuant", "SurrogateCertificate", "certify_surrogate"]


@dataclass
class ScaleTable:
    """Per-layer scales and zero-points, **as delivered**.

    This is a contracted artefact in its own right: ``scale_table_digest`` travels in
    the report, because PTQ has two inputs -- the FP32 weights and a calibration set --
    and only the first is normally contracted for. Shaping the calibration set shapes
    this table, and the table decides which behaviours survive conversion. That is
    adversary A9.
    """

    scales: dict[str, float]
    zero_points: dict[str, int]
    calibration_set_digest: str | None = None
    quantiser: str = "unspecified"
    n_bits: int = 8

    @property
    def qmin(self) -> int:
        return -(2 ** (self.n_bits - 1))

    @property
    def qmax(self) -> int:
        return 2 ** (self.n_bits - 1) - 1


class _FakeQuantSTE(torch.autograd.Function if _TORCH else object):  # type: ignore[misc]
    """Quantise-dequantise in the forward pass; identity in the backward pass.

    The straight-through estimator is what makes the surrogate differentiable. It is an
    approximation of the real quantiser's gradient -- which is exactly why step (b)
    exists and why the recovered trigger is re-measured on the real binary in step (c).
    """

    @staticmethod
    def forward(ctx, x, scale, zero_point, qmin, qmax):  # type: ignore[override]
        q = torch.clamp(torch.round(x / scale) + zero_point, qmin, qmax)
        return (q - zero_point) * scale

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore[override]
        return grad_output, None, None, None, None


class FakeQuant(nn.Module if _TORCH else object):  # type: ignore[misc]
    """Wraps a module so its output is fake-quantised at the delivered scale."""

    def __init__(self, inner, scale: float, zero_point: int, n_bits: int = 8):
        if not _TORCH:
            raise ImportError("FakeQuant needs the 'ml' extra: pip install 'pramana[ml]'")
        super().__init__()
        self.inner = inner
        self.register_buffer("scale", torch.tensor(float(scale)))
        self.register_buffer("zero_point", torch.tensor(int(zero_point)))
        self.qmin = -(2 ** (n_bits - 1))
        self.qmax = 2 ** (n_bits - 1) - 1

    def forward(self, x):
        return _FakeQuantSTE.apply(self.inner(x), self.scale, self.zero_point, self.qmin, self.qmax)


def build_surrogate(model, table: ScaleTable, *, layer_types=(None,)):
    """Wrap every named layer of ``model`` in a :class:`FakeQuant` at its delivered scale.

    Layers absent from the table are left in full precision and **named in the return
    value**, because a surrogate that silently skipped half the graph would pass the
    faithfulness gate by being wrong in a way the gate cannot see.
    """
    if not _TORCH:
        raise ImportError("build_surrogate needs the 'ml' extra: pip install 'pramana[ml]'")

    import copy

    surrogate = copy.deepcopy(model)
    wrapped: list[str] = []
    skipped: list[str] = []

    for name, module in list(surrogate.named_modules()):
        if not name or "." in name and name not in table.scales:
            continue
        if name in table.scales:
            parent_path, _, attr = name.rpartition(".")
            parent = surrogate.get_submodule(parent_path) if parent_path else surrogate
            setattr(
                parent,
                attr,
                FakeQuant(module, table.scales[name], table.zero_points.get(name, 0), table.n_bits),
            )
            wrapped.append(name)
        elif isinstance(module, (nn.Conv2d, nn.Linear)):
            skipped.append(name)

    return surrogate, {"wrapped": wrapped, "skipped_no_scale": skipped}


@dataclass
class SurrogateCertificate:
    """Whether the surrogate is faithful enough for its search to mean anything."""

    exact_agreement: float
    mean_logit_distance: float
    agreement_floor: float = K.SURROGATE_EXACT_AGREEMENT_FLOOR
    distance_ceiling: float = K.SURROGATE_MEAN_LOGIT_DISTANCE_CEILING
    n_probes: int = 0
    scale_table_digest: str | None = None

    @property
    def faithful(self) -> bool:
        return (
            self.exact_agreement >= self.agreement_floor
            and self.mean_logit_distance <= self.distance_ceiling
        )

    def raise_if_unfaithful(self) -> None:
        if not self.faithful:
            raise AssessmentUnavailable(
                "surrogate_not_faithful",
                f"exact_agreement={self.exact_agreement:.4f} (floor "
                f"{self.agreement_floor}), mean_logit_distance="
                f"{self.mean_logit_distance:.4f} (ceiling {self.distance_ceiling}). "
                f"Reversal does not run: a search on an unfaithful surrogate would "
                f"produce a trigger for a model that does not exist.",
            )

    def as_report_fields(self, *, asr_on_real: float, path: str) -> dict[str, object]:
        return {
            "path": path,
            "scale_table_digest": self.scale_table_digest,
            "surrogate_exact_agreement": round(self.exact_agreement, 4),
            "surrogate_exact_agreement_floor": self.agreement_floor,
            "surrogate_mean_logit_distance": round(self.mean_logit_distance, 4),
            "surrogate_mean_logit_distance_ceiling": self.distance_ceiling,
            "search_ran_on": "surrogate",
            "evidence_ran_on": "delivered_int8_binary",
            "forward_pass_asr_on_delivered_int8": round(asr_on_real, 4),
        }


def certify_surrogate(
    surrogate_logits: np.ndarray,
    real_int8_logits: np.ndarray,
    *,
    scale_table_digest: str | None = None,
) -> SurrogateCertificate:
    """Measure agreement between the fake-quant surrogate and the delivered binary.

    Both matrices are ``(n_probes, n_classes)`` over the Battery A probe bank -- the
    same probes the ladder uses, so the surrogate is certified on exactly the inputs
    the search will explore.
    """
    a = np.asarray(surrogate_logits, dtype=float)
    b = np.asarray(real_int8_logits, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: surrogate {a.shape} vs real {b.shape}")

    agreement = float(np.mean(a.argmax(axis=1) == b.argmax(axis=1)))
    distance = float(np.mean(np.linalg.norm(a - b, axis=1)))

    return SurrogateCertificate(
        exact_agreement=agreement,
        mean_logit_distance=distance,
        n_probes=int(a.shape[0]),
        scale_table_digest=scale_table_digest,
    )


def measure_asr_on_real(
    real_model_fn,
    triggered_inputs: np.ndarray,
    target_class: int,
) -> float:
    """Attack success rate of a recovered trigger on the **delivered** artefact.

    Forward passes only, so it is available at every access tier including A0. This is
    step (c), and it is the only number from the reversal path that reaches the report
    as evidence.
    """
    preds = np.asarray(real_model_fn(triggered_inputs))
    if preds.ndim > 1:
        preds = preds.argmax(axis=-1)
    return float(np.mean(preds == target_class))
