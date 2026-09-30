"""The hosted live-assessment service -- what Render runs.

    uvicorn api.live_app:app --port 8000

One worker thread, one assessment at a time, a short queue in front of it. On a small
CPU a T1 assessment takes minutes, so submission returns a job id at once and the
console polls for stage-by-stage progress.

Environment:
    PRAMANA_ACCESS_CODE        if set, required on every submission
    PRAMANA_SIGNING_SEED       secret; the assessor's Ed25519 key is derived from it
    ALLOWED_ORIGINS            comma-separated origins allowed to call this API
    FIREBASE_SERVICE_ACCOUNT   service-account JSON; enables Firestore persistence
"""

from __future__ import annotations

import hashlib
import hmac
import os
import queue
import tempfile
import threading
import time
import traceback
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from pramana import __version__
from pramana.ledger.signing import Signer

from api import live
from api.live_store import Store

SAMPLES = {
    "dlv-2026-0422": ("DLV-2026-0422 · routine delivery", live.ASSETS / "samples" / "dlv-2026-0422.pt"),
    "cifar10-resnet18": ("ResNet-18 trained on CIFAR-10", live.ASSETS / "samples" / "cifar10-resnet18.pt"),
}
MAX_QUEUE = 4

ACCESS_CODE = os.environ.get("PRAMANA_ACCESS_CODE", "").strip()
ORIGINS = [o.strip() for o in os.environ.get(
    "ALLOWED_ORIGINS",
    "https://pramana-live.web.app,https://pramana-live.firebaseapp.com,https://pramana-f0eb7.web.app,https://pramana-f0eb7.firebaseapp.com,http://localhost:3100,http://127.0.0.1:3100",
).split(",") if o.strip()]
_seed = os.environ.get("PRAMANA_SIGNING_SEED", "local-development-only-not-a-secret")
SIGNER = Signer.from_seed("assurance-cell-live", hashlib.sha256(_seed.encode()).digest())

app = FastAPI(title="PRAMANA live assessment", version=__version__)
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["GET", "POST"], allow_headers=["*"])

store = Store()
jobs: queue.Queue[tuple[str, Path, str, bool]] = queue.Queue()
state: dict[str, Any] = {"assets": None, "error": None, "running": None, "queued": []}


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def _ensure_genesis(a: live.Assets) -> None:
    if store.ledger():
        return
    store.append("taxonomy_committed", {"generation": 7, "classes": 27}, SIGNER)
    store.append("battery_committed", {"battery_a_digest": a.baseline["battery_digest"],
                                       "calibration_set_digest": a.baseline["calibration_set_digest"],
                                       "generation": 7, "probes": 200}, SIGNER)
    store.append("threshold_file_committed", {"file": "constants.py@0.1.0", "null": "resnet18-gtsrb-clean-n64"}, SIGNER)


def _worker() -> None:
    try:
        state["assets"] = live.Assets.load()
        _ensure_genesis(state["assets"])
    except Exception as exc:  # noqa: BLE001
        state["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        return
    while True:
        job_id, path, name, cleanup = jobs.get()
        state["running"] = job_id
        if job_id in state["queued"]:
            state["queued"].remove(job_id)
        _run(job_id, path, name)
        state["running"] = None
        if cleanup:
            path.unlink(missing_ok=True)


def _run(job_id: str, path: Path, name: str) -> None:
    a: live.Assets = state["assets"]
    t0 = time.time()
    log: list[dict[str, Any]] = []
    stage_ids = [s for s, _ in live.STAGES]

    def progress(stage: str, line: str) -> None:
        log.append({"stage": stage, "line": line, "t": round(time.time() - t0, 1)})
        store.update_job(job_id, stage=stage, stage_index=stage_ids.index(stage), log=log)

    store.update_job(job_id, status="running", started_at=_now(), stage="admit", stage_index=0)
    try:
        try:
            result = live.assess(path, name, a, progress)
        except live.Refusal as exc:
            progress(exc.stage, f"refused · {exc.code}: {exc.detail}")
            result = live.refusal_result(exc, name, path, t0)

        digest = live.record_digest(result)
        pre = store.append("assessment_preregistered", {
            "job": job_id, "artefact_digest": str(result["artefact"].get("digest")),
            "battery_digest": a.baseline["battery_digest"], "tier": "T1"}, SIGNER)
        done = store.append("assessment_completed", {
            "job": job_id, "record_digest": digest,
            "disposition": result.get("disposition", {}).get("state", "UNAVAILABLE"),
            "detection_p": f"{result['int8']['p']:.5f}" if "int8" in result else "n/a"}, SIGNER)
        progress("ledger", f"rows {pre['index']}–{done['index']} appended · Ed25519 · head {done['entry_hash'][:23]}…")
        progress("ledger", f"signed record {digest[:23]}…")

        result.update({"record_digest": digest, "record_signature": SIGNER.sign(digest.encode("utf-8")),
                       "signer_id": SIGNER.signer_id, "signer_public_key": SIGNER.public_key_b64,
                       "ledger_rows": [pre["index"], done["index"]]})
        store.update_job(job_id, status=result["status"], stage="ledger", stage_index=len(stage_ids),
                         finished_at=_now(), result=result, log=log)
    except Exception as exc:  # noqa: BLE001 - reported to the caller, never swallowed
        traceback.print_exc()
        store.update_job(job_id, status="failed", finished_at=_now(), error=f"{type(exc).__name__}: {exc}", log=log)


threading.Thread(target=_worker, daemon=True, name="assessment-worker").start()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
@app.get("/api/live/health")
def health() -> dict[str, Any]:
    return {
        "ok": state["error"] is None,
        "ready": state["assets"] is not None,
        "error": state["error"],
        "store": store.backend,
        "running": state["running"] is not None,
        "queued": len(state["queued"]),
        "access_code_required": bool(ACCESS_CODE),
        "version": __version__,
    }


@app.get("/api/live/samples")
def samples() -> list[dict[str, str]]:
    return [{"id": k, "label": v[0]} for k, v in SAMPLES.items()]


@app.post("/api/live/assessments")
async def submit(
    access_code: str = Form(""),
    sample: str = Form(""),
    file: UploadFile | None = File(None),
) -> dict[str, Any]:
    if ACCESS_CODE and not hmac.compare_digest(access_code.strip(), ACCESS_CODE):
        raise HTTPException(403, "access code not accepted")
    if state["assets"] is None:
        raise HTTPException(503, state["error"] or "the service is still starting; try again in a few seconds")
    if len(state["queued"]) >= MAX_QUEUE:
        raise HTTPException(429, "the assessment queue is full; try again shortly")

    if sample:
        if sample not in SAMPLES:
            raise HTTPException(400, f"unknown sample {sample!r}")
        name, path = SAMPLES[sample]
        cleanup = False
    elif file is not None:
        name = Path(file.filename or "upload.pt").name
        fd, tmp = tempfile.mkstemp(suffix=".pt")
        total = 0
        with os.fdopen(fd, "wb") as fh:
            while chunk := await file.read(1 << 20):
                total += len(chunk)
                if total > live.MAX_UPLOAD_BYTES:
                    fh.close()
                    Path(tmp).unlink(missing_ok=True)
                    raise HTTPException(413, f"file exceeds {live.MAX_UPLOAD_BYTES // (1 << 20)} MB")
                fh.write(chunk)
        path, cleanup = Path(tmp), True
    else:
        raise HTTPException(400, "attach a state-dict file or choose a sample")

    job_id = uuid.uuid4().hex[:12]
    store.put_job({"id": job_id, "status": "queued", "name": name, "submitted_at": _now(),
                   "stages": [{"id": s, "title": t} for s, t in live.STAGES], "stage_index": -1, "log": []})
    state["queued"].append(job_id)
    jobs.put((job_id, path, name, cleanup))
    return {"id": job_id, "status": "queued", "queue_position": len(state["queued"]) + (1 if state["running"] else 0)}


@app.get("/api/live/assessments/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "no such assessment")
    if job.get("status") == "queued" and job_id in state["queued"]:
        job["queue_position"] = state["queued"].index(job_id) + (1 if state["running"] else 0)
    elif job.get("status") in ("queued", "running") and job_id not in state["queued"] and state["running"] != job_id:
        # the host restarted underneath this job (free instances sleep and recycle)
        store.update_job(job_id, status="failed", error="the assessment host restarted; please submit again")
        job.update(status="failed", error="the assessment host restarted; please submit again")
    return job


@app.get("/api/live/ledger")
def ledger() -> dict[str, Any]:
    return {"rows": store.ledger(), "public_keys": {SIGNER.signer_id: SIGNER.public_key_b64}}
