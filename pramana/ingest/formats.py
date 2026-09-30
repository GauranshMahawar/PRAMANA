"""Clause 2.2.6 ingest -- COCO, YOLO, ONNX, TorchScript.

The clause mandates these formats, so they get a **conformance test rather than a
mention**. ``pramana ingest --selftest`` loads every fixture in ``conformance/`` and,
just as importantly, confirms that the deliberately malformed fixtures are *rejected*.
A loader that accepts anything is not a loader, it is a liability: a silently
mis-parsed annotation file produces a detector result about data nobody delivered.

Model loading is load-bearing rather than plumbing. D1's whole premise is that the
artefact which ships is a converted one, so the ability to read an ONNX or TorchScript
graph is the ability to assess the thing that actually runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pramana.common.digest import sha256_file

__all__ = ["Annotation", "Dataset", "load_coco", "load_yolo", "load_model", "selftest"]


class IngestError(Exception):
    """A dataset or model could not be ingested, with the reason a human needs."""


#: Files that live in a YOLO labels directory and are NOT per-image label files.
#: Parsing ``classes.txt`` as labels is a real failure mode -- it looks like a label
#: file, it has the right extension, and every line fails validation, so the whole
#: shard is rejected for a reason that has nothing to do with the shard.
_YOLO_NON_LABEL_FILES = frozenset(
    {"classes.txt", "names.txt", "obj.names", "train.txt", "val.txt", "test.txt", "readme.txt"}
)


@dataclass
class Annotation:
    image_id: str
    label: int
    bbox: tuple[float, float, float, float] | None = None
    source_lot: str | None = None


@dataclass
class Dataset:
    """A contributed dataset, with the provenance metadata clause 2.2.1 depends on.

    ``lot_of`` is what makes source-level aggregation possible at all. Where it is
    absent the data-side detectors still run, but the roll-up cannot happen and the
    report says ``no_data_locus`` rather than inventing a partition.
    """

    name: str
    format: str
    annotations: list[Annotation] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    lot_of: dict[str, str] = field(default_factory=dict)
    digest: str | None = None
    source_path: str | None = None

    @property
    def n_images(self) -> int:
        return len({a.image_id for a in self.annotations})

    @property
    def n_classes(self) -> int:
        return len(self.classes) or len({a.label for a in self.annotations})

    @property
    def has_provenance(self) -> bool:
        return bool(self.lot_of)

    def lots(self) -> dict[str, list[Annotation]]:
        grouped: dict[str, list[Annotation]] = {}
        for a in self.annotations:
            lot = a.source_lot or self.lot_of.get(a.image_id)
            if lot:
                grouped.setdefault(lot, []).append(a)
        return grouped

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "format": self.format,
            "n_images": self.n_images,
            "n_annotations": len(self.annotations),
            "n_classes": self.n_classes,
            "has_provenance": self.has_provenance,
            "n_lots": len(self.lots()),
            "digest": self.digest,
        }


# ---------------------------------------------------------------------------
# COCO
# ---------------------------------------------------------------------------


def load_coco(path: str | Path, *, lot_field: str = "source_lot") -> Dataset:
    """Load a COCO-format annotation file.

    ``lot_field`` is read from each image record. It is not part of the COCO spec, and
    that is the point: contributor metadata has to come from somewhere, so PRAMANA
    reads an optional extension field and **says so** rather than pretending COCO
    carries provenance natively.
    """
    p = Path(path)
    if not p.exists():
        raise IngestError(f"COCO file not found: {p}")

    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise IngestError(f"{p.name} is not valid JSON: {exc}") from exc

    for key in ("images", "annotations", "categories"):
        if key not in raw:
            raise IngestError(f"{p.name} is not COCO: missing required key {key!r}")
    if not isinstance(raw["images"], list) or not isinstance(raw["annotations"], list):
        raise IngestError(f"{p.name}: 'images' and 'annotations' must be arrays")

    id_to_name = {img["id"]: str(img.get("file_name", img["id"])) for img in raw["images"]}
    lot_of = {
        str(img.get("file_name", img["id"])): img[lot_field]
        for img in raw["images"]
        if lot_field in img
    }
    categories = [c["name"] for c in sorted(raw["categories"], key=lambda c: c["id"])]

    annotations: list[Annotation] = []
    for a in raw["annotations"]:
        if "image_id" not in a or "category_id" not in a:
            raise IngestError(f"{p.name}: annotation missing image_id or category_id: {a}")
        img_name = id_to_name.get(a["image_id"])
        if img_name is None:
            raise IngestError(
                f"{p.name}: annotation references image_id {a['image_id']} that is not "
                f"in 'images'. A dangling reference means the file describes data we "
                f"were not given."
            )
        bbox = tuple(a["bbox"]) if "bbox" in a and len(a.get("bbox", [])) == 4 else None
        annotations.append(
            Annotation(
                image_id=img_name,
                label=int(a["category_id"]),
                bbox=bbox,  # type: ignore[arg-type]
                source_lot=lot_of.get(img_name),
            )
        )

    return Dataset(
        name=p.stem,
        format="coco",
        annotations=annotations,
        classes=categories,
        lot_of=lot_of,
        digest=sha256_file(p),
        source_path=str(p),
    )


# ---------------------------------------------------------------------------
# YOLO
# ---------------------------------------------------------------------------


def load_yolo(
    labels_dir: str | Path,
    *,
    classes_file: str | Path | None = None,
    lot_manifest: str | Path | None = None,
) -> Dataset:
    """Load a YOLO-format label directory (one ``.txt`` per image).

    Each line is ``class cx cy w h`` with normalised coordinates. Values outside
    ``[0, 1]`` are rejected rather than clamped: a coordinate of 1.4 means the file was
    written against a different convention, and silently clamping it would convert a
    format error into a plausible-looking box.
    """
    d = Path(labels_dir)
    if not d.is_dir():
        raise IngestError(f"YOLO labels directory not found: {d}")

    classes: list[str] = []
    if classes_file and Path(classes_file).exists():
        classes = [
            line.strip()
            for line in Path(classes_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    lot_of: dict[str, str] = {}
    if lot_manifest and Path(lot_manifest).exists():
        lot_of = json.loads(Path(lot_manifest).read_text(encoding="utf-8"))

    annotations: list[Annotation] = []
    files = sorted(f for f in d.glob("*.txt") if f.name.lower() not in _YOLO_NON_LABEL_FILES)
    if not files:
        raise IngestError(f"{d} contains no .txt label files")

    for f in files:
        stem = f.stem
        for lineno, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                raise IngestError(
                    f"{f.name}:{lineno}: expected 5 fields (class cx cy w h), got {len(parts)}"
                )
            try:
                cls = int(parts[0])
                cx, cy, w, h = (float(v) for v in parts[1:])
            except ValueError as exc:
                raise IngestError(f"{f.name}:{lineno}: non-numeric field: {exc}") from exc
            if cls < 0:
                raise IngestError(f"{f.name}:{lineno}: negative class id {cls}")
            for nm, v in (("cx", cx), ("cy", cy), ("w", w), ("h", h)):
                if not 0.0 <= v <= 1.0:
                    raise IngestError(
                        f"{f.name}:{lineno}: {nm}={v} is outside [0,1]. YOLO coordinates "
                        f"are normalised; this file was written against another convention."
                    )
            annotations.append(
                Annotation(
                    image_id=stem,
                    label=cls,
                    bbox=(cx, cy, w, h),
                    source_lot=lot_of.get(stem),
                )
            )

    return Dataset(
        name=d.name,
        format="yolo",
        annotations=annotations,
        classes=classes,
        lot_of=lot_of,
        source_path=str(d),
    )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@dataclass
class LoadedModel:
    """A model artefact, with the identity fields every finding needs to name it."""

    path: str
    format: str
    digest: str
    handle: Any = None
    n_parameters: int | None = None
    input_shape: tuple[int, ...] | None = None
    note: str = ""

    def summary(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "format": self.format,
            "digest": self.digest,
            "n_parameters": self.n_parameters,
            "input_shape": list(self.input_shape) if self.input_shape else None,
            "note": self.note,
        }


def load_model(path: str | Path, *, device: str = "cpu") -> LoadedModel:
    """Load an ONNX or TorchScript artefact, detected by extension then by content.

    The digest is computed **before** anything is deserialised, so the identity in the
    report is the identity of the bytes we were handed, not of whatever they became.
    """
    p = Path(path)
    if not p.exists():
        raise IngestError(f"model not found: {p}")
    digest = sha256_file(p)
    suffix = p.suffix.lower()

    if suffix == ".onnx":
        return _load_onnx(p, digest)
    if suffix in (".pt", ".pth", ".torchscript"):
        return _load_torchscript(p, digest, device)
    raise IngestError(
        f"unsupported model extension {suffix!r}. Clause 2.2.6 names ONNX and "
        f"PyTorch/TorchScript; anything else is declared unsupported rather than guessed at."
    )


def _load_onnx(p: Path, digest: str) -> LoadedModel:
    try:
        import onnxruntime as ort
    except ImportError:
        return LoadedModel(
            path=str(p),
            format="onnx",
            digest=digest,
            note=(
                "onnxruntime is not installed; the digest and identity are recorded but "
                "no forward pass is available. Install the 'ml' extra to assess this rung."
            ),
        )
    sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    shape = tuple(d if isinstance(d, int) else -1 for d in inp.shape)
    return LoadedModel(path=str(p), format="onnx", digest=digest, handle=sess, input_shape=shape)


def _load_torchscript(p: Path, digest: str, device: str) -> LoadedModel:
    try:
        import torch
    except ImportError:
        return LoadedModel(
            path=str(p),
            format="torchscript",
            digest=digest,
            note="torch is not installed; identity recorded, no forward pass available.",
        )
    # Try TorchScript first, then an eager checkpoint. If BOTH fail the file is not a
    # model we can read, and that must surface as IngestError -- a caller has to be able
    # to tell "this artefact is malformed" from "PRAMANA has a bug", and a leaked
    # RuntimeError from the deserialiser says nothing about which.
    try:
        module = torch.jit.load(str(p), map_location=device)
        fmt = "torchscript"
    except Exception as ts_exc:
        try:
            module = torch.load(str(p), map_location=device, weights_only=False)
            fmt = "pytorch"
        except Exception as pt_exc:
            raise IngestError(
                f"{p.name} is neither TorchScript nor a loadable PyTorch checkpoint. "
                f"torch.jit.load: {type(ts_exc).__name__}: {ts_exc}. "
                f"torch.load: {type(pt_exc).__name__}: {pt_exc}"
            ) from pt_exc
    n_params = (
        sum(param.numel() for param in module.parameters())
        if hasattr(module, "parameters")
        else None
    )
    return LoadedModel(
        path=str(p), format=fmt, digest=digest, handle=module, n_parameters=n_params
    )


# ---------------------------------------------------------------------------
# Conformance self-test
# ---------------------------------------------------------------------------


def selftest(conformance_dir: str | Path = "conformance") -> dict[str, Any]:
    """Load every fixture and confirm the malformed ones are rejected.

    Returns a structured result. The second half matters as much as the first: a
    format checker that only proves it can read good files has proved nothing about
    what it does with bad ones.
    """
    root = Path(conformance_dir)
    results: list[dict[str, Any]] = []

    for f in sorted((root / "coco").glob("*.json")):
        expect_fail = f.stem.startswith("bad_")
        results.append(_try(f.name, "coco", lambda: load_coco(f), expect_fail))

    for d in sorted((root / "yolo").iterdir()) if (root / "yolo").is_dir() else []:
        if not d.is_dir():
            continue
        expect_fail = d.name.startswith("bad_")
        results.append(_try(d.name, "yolo", lambda: load_yolo(d), expect_fail))

    for f in sorted((root / "models").glob("*")):
        if f.suffix.lower() not in (".onnx", ".pt", ".pth", ".torchscript"):
            continue
        expect_fail = f.stem.startswith("bad_")
        results.append(_try(f.name, "model", lambda: load_model(f), expect_fail))

    passed = sum(1 for r in results if r["pass"])
    return {
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "ok": passed == len(results) and len(results) > 0,
        "results": results,
    }


def _try(name: str, kind: str, fn, expect_fail: bool) -> dict[str, Any]:
    try:
        fn()
        return {
            "fixture": name,
            "kind": kind,
            "expected": "reject" if expect_fail else "load",
            "actual": "loaded",
            "pass": not expect_fail,
            "detail": "loaded a fixture that should have been rejected" if expect_fail else "",
        }
    except IngestError as exc:
        return {
            "fixture": name,
            "kind": kind,
            "expected": "reject" if expect_fail else "load",
            "actual": "rejected",
            "pass": expect_fail,
            "detail": str(exc),
        }
    except Exception as exc:  # noqa: BLE001 - an unexpected type is itself a failure
        return {
            "fixture": name,
            "kind": kind,
            "expected": "reject" if expect_fail else "load",
            "actual": f"raised {type(exc).__name__}",
            "pass": False,
            "detail": f"{exc}. Loaders must raise IngestError, so a caller can tell a "
            f"format problem from a bug.",
        }
