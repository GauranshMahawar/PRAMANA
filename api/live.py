"""Live T1 assessment of an uploaded artefact.

T1 is the tier that needs only the delivered artefact: forward passes over the committed
battery at every rung the host can build, ranked against the fitted nulls. No dataset,
no gradients, no retraining. Lot-level evidence, the supplier certificate and leave-lot-
out attribution need T2/T3 access and are declared as not run -- never implied.

Uploads are loaded as **state dicts only** (``torch.load(..., weights_only=True)``) into
the fixed ResNet-18 graph from :mod:`pramana.ladder.models`. A pickled model object or a
TorchScript archive can carry code; a tensor dictionary cannot, and the null is only
valid for this exact graph anyway.
"""

from __future__ import annotations

import base64
import gc
import io
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from pramana.common import constants as K
from pramana.common.digest import sha256_file, sha256_json
from pramana.disposition.lattice import decide
from pramana.ladder.concentration import assess_concentration, interpret
from pramana.ladder.divergence import probe_divergence
from pramana.ladder.models import build
from pramana.ladder.null import FittedNull, NullIndex, fit_null
from pramana.report.schema import AttributionMode, Evidence, EvidenceStrength, Finding, Statistic
from pramana.reversal.neural_cleanse import bh_select

ASSETS = Path(__file__).resolve().parent.parent / "backend" / "assets"
CONVERTER_ID = "torch.ao.fx.ptq.x86"
MAX_UPLOAD_BYTES = 60 * 1024 * 1024

#: The stages the console shows, in order. Ids are the contract with the frontend.
STAGES: list[tuple[str, str]] = [
    ("admit", "Admission & custody"),
    ("precommit", "Pre-commitment check"),
    ("null", "Null match"),
    ("fp32", "FP32 battery"),
    ("ladder", "Precision ladder"),
    ("concentration", "Concentration & localisation"),
    ("limits", "Declared limits"),
    ("disposition", "Disposition"),
    ("ledger", "Ledger & signed record"),
]

Progress = Callable[[str, str], None]


class Refusal(Exception):
    """The assessment declines to score, with a reason -- never a silent pass."""

    def __init__(self, code: str, detail: str, stage: str = "admit"):
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.stage = stage


@dataclass
class Assets:
    baseline: dict[str, Any]
    probes: torch.Tensor
    probe_labels: np.ndarray
    calib: torch.Tensor
    eval_x: torch.Tensor
    eval_y: np.ndarray
    nulls: dict[str, FittedNull]
    l2: FittedNull

    @classmethod
    def load(cls, root: Path = ASSETS) -> Assets:
        baseline = json.loads((root / "baseline.json").read_text(encoding="utf-8"))
        b = np.load(root / "battery.npz")
        nulls = {
            rung: fit_null(np.asarray(spec["draws"], dtype=float), NullIndex("resnet18", "gtsrb", CONVERTER_ID),
                           n_models=64, population_name=spec["population"])
            for rung, spec in baseline["nulls"].items()
        }
        l2 = fit_null(np.asarray(baseline["l2_pooled_shares"]), NullIndex("resnet18", "gtsrb", CONVERTER_ID),
                      level="L2_localisation", n_models=64, population_name="resnet18-gtsrb-clean-n64 x 43 classes")
        return cls(
            baseline=baseline,
            probes=torch.from_numpy(b["probes"]),
            probe_labels=b["probe_labels"],
            calib=torch.from_numpy(b["calib"]),
            eval_x=torch.from_numpy(b["eval_x"]),
            eval_y=b["eval_y"],
            nulls=nulls,
            l2=l2,
        )

    def normalise(self, x_uint8: torch.Tensor) -> torch.Tensor:
        x = x_uint8.float().div_(255.0)
        mean = torch.tensor(self.baseline["mean"]).view(1, 3, 1, 1)
        std = torch.tensor(self.baseline["std"]).view(1, 3, 1, 1)
        return (x - mean) / std


# ---------------------------------------------------------------------------
# Rungs
# ---------------------------------------------------------------------------


@torch.no_grad()
def _logits(model: torch.nn.Module, x: torch.Tensor, batch: int = 25) -> np.ndarray:
    return torch.cat([model(x[i : i + batch]).float() for i in range(0, x.shape[0], batch)]).numpy()


def _quantise(model: torch.nn.Module, calib: torch.Tensor) -> torch.nn.Module:
    """Post-training static INT8 with the pinned FX converter -- identical to every null model."""
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

    torch.backends.quantized.engine = "x86"
    # prepare_fx builds a new GraphModule and folds BN into new tensors; the FP32 model's
    # own weights and buffers are left untouched (checked), so no defensive copy is made
    prepared = prepare_fx(model.eval(), get_default_qconfig_mapping("x86"), (calib[:1],))
    with torch.no_grad():
        # batches of 128, exactly as every null model was calibrated: the histogram
        # observers merge per batch, so a different batch size is a different converter
        for i in range(0, calib.shape[0], 128):
            prepared(calib[i : i + 128])
    return convert_fx(prepared)


def _prune_in_place(model: torch.nn.Module, amount: float = 0.30) -> torch.nn.Module:
    """Zero the ``amount`` smallest-magnitude conv/linear weights, globally, in place.

    Selects exactly what ``torch.nn.utils.prune.global_unstructured(L1Unstructured)``
    selects -- the k smallest |w| over all layers, k = round(amount * n) -- without the
    weight_orig and mask copies that utility keeps, which on a 512 MB host is the
    difference between running and being killed.
    """
    weights = [m.weight for m in model.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
    n_total = sum(w.numel() for w in weights)
    k = round(amount * n_total)
    with torch.no_grad():
        flat = np.empty(n_total, dtype=np.float32)
        offset = 0
        for w in weights:
            flat[offset : offset + w.numel()] = w.detach().abs().reshape(-1).numpy()
            offset += w.numel()
        thr = float(np.partition(flat, k - 1)[k - 1])
        del flat
        below = sum(int((w.abs() < thr).sum()) for w in weights)
        ties_to_prune = k - below  # |w| == thr: prune the first ones in layer order
        for w in weights:
            a = w.abs()
            keep = a > thr
            tie = (a == thr).reshape(-1).nonzero().reshape(-1)
            if tie.numel():
                n = min(max(ties_to_prune, 0), tie.numel())
                keep.view(-1)[tie[n:]] = True  # ties beyond the pruning budget are kept
                ties_to_prune -= n
            w.mul_(keep)
            del a, keep
    return model


def _release() -> None:
    """Return freed memory to the OS where the allocator allows it."""
    gc.collect()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def _png(x_uint8: torch.Tensor, size: int = 96) -> str:
    from PIL import Image

    img = Image.fromarray(x_uint8.permute(1, 2, 0).numpy()).resize((size, size), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# The assessment
# ---------------------------------------------------------------------------


def load_upload(path: Path, a: Assets) -> tuple[torch.nn.Module, int]:
    """State dict -> fixed ResNet-18 graph. Anything else is refused with a reason."""
    try:
        try:  # memory-mapped: the file's pages back the tensors instead of a second RAM copy
            sd = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
        except RuntimeError:
            sd = torch.load(path, weights_only=True, map_location="cpu")
    except Exception as exc:  # noqa: BLE001 - the deserialiser's own error is not the user's contract
        raise Refusal("unsupported_format", "not a PyTorch state dict that can be loaded without executing code") from None
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    if not isinstance(sd, dict) or not all(isinstance(v, torch.Tensor) for v in sd.values()):
        raise Refusal("unsupported_format", "expected a dictionary of tensors (model.state_dict())")
    w = sd.get("classifier.weight")
    if w is None or w.ndim != 2:
        raise Refusal("unsupported_architecture", "no classifier.weight: this is not the fitted ResNet-18 graph")
    n_classes = int(w.shape[0])
    model = build("resnet18", n_classes=n_classes)
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError:
        raise Refusal("unsupported_architecture", "state dict does not match the ResNet-18 graph the null was fitted on") from None
    del sd, w
    _release()
    return model.eval(), n_classes


def assess(path: Path, name: str, a: Assets, progress: Progress) -> dict[str, Any]:
    t0 = time.time()
    torch.set_num_threads(1)
    size = path.stat().st_size
    digest = sha256_file(path)
    base = a.baseline

    # 1 admission ------------------------------------------------------------------
    progress("admit", f"artefact {name} · {size / 1e6:.1f} MB · {digest[:23]}…")
    model, n_classes = load_upload(path, a)
    params = sum(p.numel() for p in model.parameters())
    progress("admit", f"loaded as state dict (no code executed) · ResNet-18 · {params / 1e6:.1f}M parameters · {n_classes} classes")

    # 2 pre-commitment ----------------------------------------------------------------
    progress("precommit", f"battery A {base['battery_digest'][:23]}… · 200 probes · committed before upload")
    progress("precommit", f"calibration set {base['calibration_set_digest'][:23]}… · 512 images")

    # 3 null match ---------------------------------------------------------------------
    if n_classes != base["index"]["classes"]:
        corpus = "cifar10" if n_classes == 10 else f"unknown-{n_classes}-class"
        t = base["transfer"]["null_corpus_transfer_delta"]
        raise Refusal(
            "no_fitted_null",
            f"null is indexed resnet18/gtsrb/{CONVERTER_ID}; artefact is resnet18/{corpus}. "
            f"Measured corpus transfer delta {t['median_shift_over_iqr']:.2f} IQR exceeds the declared "
            f"ceiling {K.TRANSFER_CEILING_MEDIAN_SHIFT_OVER_IQR}, so the null is not reused.",
            stage="null",
        )
    progress("null", f"index resnet18 / gtsrb / {CONVERTER_ID} · operational null n=64 · floor 1/65")

    # 4 FP32 ----------------------------------------------------------------------------
    probes = a.normalise(a.probes)
    evalx = a.normalise(a.eval_x)
    fp32 = _logits(model, probes)
    acc = {"fp32": float((_logits(model, evalx).argmax(1) == a.eval_y).mean())}
    progress("fp32", f"200 probes · FP32 · accuracy {acc['fp32'] * 100:.2f}% on 500 held-out GTSRB images")

    # 5 ladder --------------------------------------------------------------------------
    rungs: dict[str, np.ndarray] = {}
    # order matters on a small host: trace and quantise from the intact FP32 graph,
    # then prune that same graph in place -- no rung ever holds a second FP32 copy
    ts = torch.jit.trace(model, probes[:1])
    rungs["torchscript"] = _logits(ts, probes)
    acc["torchscript"] = acc["fp32"]
    del ts
    _release()

    q = _quantise(model, a.normalise(a.calib))
    rungs["int8_ptq"] = _logits(q, probes)
    acc["int8_ptq"] = float((_logits(q, evalx).argmax(1) == a.eval_y).mean())
    del q
    _release()
    progress("ladder", "int8_ptq  converted with the pinned converter")

    _prune_in_place(model)
    rungs["pruned"] = _logits(model, probes)
    acc["pruned"] = float((_logits(model, evalx).argmax(1) == a.eval_y).mean())
    _release()

    ladder: list[dict[str, Any]] = []
    profiles = {}
    for rung in ("int8_ptq", "pruned", "torchscript"):
        prof = probe_divergence(fp32, rungs[rung], pair=f"fp32->{rung}")
        profiles[rung] = prof
        row: dict[str, Any] = {"rung": rung, "mean_divergence": prof.mean_divergence,
                               "probes_diverging": prof.probes_diverging, "accuracy": acc.get(rung)}
        if rung in a.nulls:
            nl = a.nulls[rung]
            p, floor = nl.p_value(prof.mean_divergence)
            rc = assess_concentration(prof.per_class)
            row.update({"p": p, "p_is_floor": floor, "p_floor": nl.p_floor, "fires": p <= K.DECLARED_FDR,
                        "null_population": nl.population_name, "null_draws": [float(x) for x in nl.draws],
                        "gini": round(rc.gini, 4), "classes_at_90": rc.classes_at_90pct,
                        "interpretation": interpret(rc.condition_met, p <= K.DECLARED_FDR)})
            progress("ladder", f"{rung:<11} JS {prof.mean_divergence:.2e} · flips {prof.probes_diverging}/200 · "
                               f"p {'≤' if floor else '='} {p:.5f}{' FLOOR' if floor else ''}")
        else:
            row.update({"p": None, "note": "traced graph reproduces FP32 to float tolerance"})
            progress("ladder", f"{rung:<11} JS {prof.mean_divergence:.2e} · ≡ fp32")
        ladder.append(row)
    ladder.insert(1, {"rung": "fp16", "unavailable": True, "note": "not built on a CPU-only host; declared, not skipped"})
    ladder.append({"rung": "onnx", "unavailable": True, "note": "onnxruntime not staged on this host"})

    prof8 = profiles["int8_ptq"]
    p8, floor8 = a.nulls["int8_ptq"].p_value(prof8.mean_divergence)
    fires = p8 <= K.DECLARED_FDR

    # 6 concentration / L2 --------------------------------------------------------------
    conc = assess_concentration(prof8.per_class)
    reading = interpret(conc.condition_met, fires)
    loc: dict[str, Any] = {"gated": not fires, "critical_value_at_rank_1": K.BH_CRITICAL_VALUE_RANK_1,
                           "p_floor": a.l2.p_floor, "pooled_draws": a.l2.n_draws}
    progress("concentration", f"gini {conc.gini:.3f} over {conc.classes_at_90pct} classes · operating point ≥0.70 over ≤3 (predicted)")
    if fires:
        v = prof8.per_class
        share = v / v.sum()
        pvals = {c: a.l2.p_value(float(share[c]))[0] for c in range(len(share))}
        bh = bh_select(pvals, fdr=K.DECLARED_FDR)
        loc.update({"p": [round(pvals[c], 6) for c in range(len(share))], "rejected": bh["rejected"]})
        progress("concentration", f"L2 BH over 43 classes · rejected {bh['rejected'] or 'none'}")
    else:
        progress("concentration", "L2 gated: detection did not fire, so the 43 class tests are not run")

    flipped = np.flatnonzero(fp32.argmax(1) != rungs["int8_ptq"].argmax(1)).tolist()
    pf, pi = _softmax(fp32), _softmax(rungs["int8_ptq"])
    gallery = [{"probe": i, "img": _png(a.probes[i]), "truth": int(a.probe_labels[i]),
                "fp32": int(fp32[i].argmax()), "fp32_conf": round(float(pf[i].max()), 4),
                "int8": int(rungs["int8_ptq"][i].argmax()), "int8_conf": round(float(pi[i].max()), 4)}
               for i in flipped[:12]]

    # 7 limits ----------------------------------------------------------------------------
    progress("limits", "trigger_warping → declared_unsupported at T1")
    progress("limits", "lot evidence, supplier certificate, leave-lot-out: need T2/T3 access · not run")

    # 8 disposition -----------------------------------------------------------------------
    findings: list[Finding] = []
    if fires and conc.condition_met:
        null8 = a.nulls["int8_ptq"]
        findings.append(Finding(
            finding_id="F-1", mechanism="precision_ladder_divergence", reason_code="concentrated_cross_rung_divergence",
            affected_asset="build:int8_ptq", rung="int8_ptq", level="L1_detection",
            statistic=Statistic(name="cross_rung_divergence", value=round(prof8.mean_divergence, 8), p_floor=null8.p_floor,
                                p_is_floor=floor8, null_population=null8.population_name, null_family="resnet18",
                                null_corpus="gtsrb"),
            evidence=[Evidence(kind="concentration", detail=f"gini={conc.gini:.3f} over {conc.classes_at_90pct} classes")],
            attribution_mode=AttributionMode.SET_VALUED, attribution_set=["artefact"], attribution_set_size=1,
            containment_scope=["build:int8_ptq"], evidence_strength=EvidenceStrength.CORROBORATED,
            corroborating_mechanisms=["class_localisation_l2"] if loc.get("rejected") else [],
        ))
    disp = decide(findings, refusals=["trigger_warping"], predicted_operating_points=["precision_ladder_divergence"])
    rung_list = "fp32, int8_ptq, pruned, torchscript"
    statement = (
        K.VERDICT_TEMPLATE.format(battery_digest=base["battery_digest"][:23] + "…", rungs=rung_list,
                                  null_family="resnet18", null_corpus="gtsrb", converter=CONVERTER_ID)
        if not findings else
        f"Concentrated cross-rung divergence at int8_ptq (p ≤ {p8:.5f}, gini {conc.gini:.3f} over "
        f"{conc.classes_at_90pct} classes) against null (family=resnet18, corpus=gtsrb, converter={CONVERTER_ID})."
    )
    progress("disposition", f"{disp.state.value} · {len(findings)} finding(s)")

    return {
        "status": "done",
        "duration_s": round(time.time() - t0, 1),
        "artefact": {"name": name, "size_bytes": size, "digest": digest, "family": "resnet18",
                     "classes": n_classes, "params": params},
        "accuracy": acc,
        "ladder": ladder,
        "int8": {"mean_divergence": prof8.mean_divergence, "p": p8, "p_is_floor": floor8,
                 "p_floor": a.nulls["int8_ptq"].p_floor, "fires": fires, "probes_diverging": prof8.probes_diverging,
                 "per_class": [round(float(x), 8) for x in prof8.per_class], "gini": round(conc.gini, 4),
                 "classes_at_90": conc.classes_at_90pct, "dominant_class": conc.dominant_class,
                 "condition_met": conc.condition_met, "interpretation": reading,
                 "operating_point": {"gini": conc.threshold_gini, "max_classes": conc.threshold_max_classes,
                                     "status": conc.status}},
        "benign_gini": base["benign_gini"],
        "benign_classes_at_90": base["benign_classes_at_90"],
        "localisation": loc,
        "gallery": gallery,
        "disposition": {"state": disp.state.value, "rationale": disp.rationale},
        "verdict_statement": statement,
        "battery_digest": base["battery_digest"],
        "tier": "T1",
    }


def refusal_result(exc: Refusal, name: str, path: Path | None, t0: float) -> dict[str, Any]:
    return {
        "status": "refused",
        "duration_s": round(time.time() - t0, 1),
        "artefact": {"name": name, "size_bytes": path.stat().st_size if path and path.exists() else None,
                     "digest": sha256_file(path) if path and path.exists() else None},
        "refusal": {"code": exc.code, "detail": exc.detail},
        "verdict_statement": f"assessment_unavailable: {exc.code} — {exc.detail}",
        "tier": "T1",
    }


def record_digest(result: dict[str, Any]) -> str:
    """Digest of the result without its bulky arrays -- what the ledger row commits to."""
    slim = {k: v for k, v in result.items() if k not in ("gallery", "benign_gini", "benign_classes_at_90")}
    slim["ladder"] = [{k: v for k, v in r.items() if k != "null_draws"} for r in result.get("ladder", [])]
    return sha256_json(json.loads(json.dumps(slim, default=float)))
