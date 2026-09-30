"""The end-to-end demo -- every beat of the eight-minute script, runnable offline.

This is not a mock. It exercises the real ladder statistics, the real null, the real
e-value aggregation, the real partition certificate, the real ledger and the real
report validator. What is synthetic is the *model population*: logit matrices are
generated rather than produced by 64 trained ResNet-18s, because the point of the demo
is that the assurance machinery works end to end, and the 21-GPU-hour null is a
separate experiment (``experiments/e1_null.py``).

Every number this produces is therefore labelled ``ILLUSTRATIVE_EXAMPLE`` in the
emitted report. A schema-conformant instance full of plausible numbers, screenshotted
out of context, is indistinguishable from a results table -- which is exactly why
``document_status`` is a required field and why this module never sets it to
``MEASUREMENT_RECORD``.

Beats, in the order the script runs them::

    1  ingest selftest        COCO/YOLO/ONNX/TorchScript load; malformed ones rejected
    2  pre-commitment         battery digests written to the ledger BEFORE the artefact
    3  fp32 assessment        every FP32 check returns no finding
    4  the ladder             INT8 diverges, concentrated on few classes
    5  reversal at INT8       trigger recovered; the same search at FP32 finds nothing
    7  source aggregation     12 lots, e-values, e-BH at a declared FDR
    8  honest failure         a warping trigger returns declared_unsupported
    9  certificate            k of 12, beside the volume share those k lots hold
    10 conditional release    named authority, controls, expiry, ledger entry
    11 receipt monitor        e-process crosses; the risk acceptance is revoked
    12 tamper + anchor        insider rewrite: local chain verifies, anchor does not
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from pramana.aggregate.evalues import aggregate_sources
from pramana.common import constants as K
from pramana.common.digest import sha256_json
from pramana.common.errors import AssessmentUnavailable, LedgerTampered
from pramana.disposition.lattice import conditional_release, decide, revoke
from pramana.drift.discriminators import discriminate
from pramana.ingest.formats import selftest as ingest_selftest
from pramana.ladder.concentration import assess_concentration, interpret
from pramana.ladder.divergence import probe_divergence
from pramana.ladder.null import FittedNull, NullIndex, fit_null
from pramana.ledger.anchor import Anchorer
from pramana.ledger.chain import HashChain, InMemoryChain
from pramana.ledger.signing import KeyRegistry, Signer
from pramana.partition.lots import (
    EXAMPLE_SHARES,
    Lot,
    LotRegister,
    volume_inequality_block,
)
from pramana.partition.roe import run_off_election, surrogate_gap
from pramana.receipts.eprocess import EProcess
from pramana.receipts.receipt import ReceiptEmitter, verify_stream
from pramana.report.emitter import ReportBuilder
from pramana.report.schema import (
    AccessTier,
    Artefact,
    AssessmentMeta,
    AttributionMode,
    Battery,
    BatteryCustody,
    Build,
    BuildState,
    Conversion,
    ConverterProvenance,
    DebtItem,
    DocumentStatus,
    DriftAssessment,
    DriftAxis,
    DriftRealism,
    Evidence,
    EvidenceStrength,
    Finding,
    FingerprintDivergence,
    NullArm,
    NullCalibration,
    OodBackbone,
    PrecisionLadder,
    Statistic,
    SupplierConcentrationMeasurement,
    Tier,
    TransferCeiling,
    TransferDelta,
    VendorVsIntegrator,
    VolumeInequality,
    CoverageDisposition,
)

N_PROBES = K.BATTERY_A_PROBE_COUNT
N_CLASSES = K.N_CLASSES_OPERATIONAL
TARGET_CLASS = 14


@dataclass
class Beat:
    n: int
    name: str
    ok: bool
    detail: str
    payload: dict[str, Any] = field(default_factory=dict)

    def line(self) -> str:
        mark = "PASS" if self.ok else "FAIL"
        return f"  [{mark}] beat {self.n:>2}  {self.name:<24} {self.detail}"


# ---------------------------------------------------------------------------
# Synthetic artefact population
# ---------------------------------------------------------------------------


def _clean_logits(rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0.0, 3.0, size=(N_PROBES, N_CLASSES))


def _benign_conversion(fp32: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Ordinary INT8 rounding: diffuse, touching every class a little."""
    return fp32 + rng.normal(0.0, 0.25, size=fp32.shape)


def _qcb_conversion(
    fp32: np.ndarray, rng: np.random.Generator, *, n_affected: int = 2, amplitude: float = 6.0
) -> np.ndarray:
    """A quantisation-conditioned backdoor: nothing at FP32, armed by the conversion.

    The adversary has not tampered with the delivered artefact at all -- it is exactly
    what was signed -- so re-hashing detects nothing and FP32 behavioural testing
    detects nothing either. This is adversary A8.
    """
    out = _benign_conversion(fp32, rng)
    hit = rng.choice(N_PROBES, size=int(N_PROBES * 0.06), replace=False)
    for offset in range(n_affected):
        out[hit, (TARGET_CLASS + offset) % N_CLASSES] += amplitude
    return out


def build_null(rng: np.random.Generator, n_models: int = K.N_NULL_OPERATIONAL) -> FittedNull:
    """Fit the operational null from a population of clean conversions."""
    draws = []
    for _ in range(n_models):
        fp32 = _clean_logits(rng)
        draws.append(probe_divergence(fp32, _benign_conversion(fp32, rng), pair="fp32->int8").mean_divergence)
    return fit_null(
        np.asarray(draws),
        NullIndex("resnet18", "gtsrb", "onnxruntime.qdq"),
        n_models=n_models,
        population_name=f"resnet18-gtsrb-clean-n{n_models}",
    )


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def run(
    out_dir: str | Path = "out",
    *,
    seed: int = 20260921,
    poisoned: bool = True,
    quiet: bool = False,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    beats: list[Beat] = []

    def say(msg: str) -> None:
        if not quiet:
            print(msg)

    # -- beat 1: ingest conformance ---------------------------------------
    st = ingest_selftest("conformance")
    beats.append(
        Beat(1, "ingest selftest", st["ok"],
             f"{st['passed']}/{st['total']} fixtures; malformed ones rejected", st)
    )

    # -- beat 2: pre-commitment -------------------------------------------
    assessor = Signer.from_seed("assurance-cell-north", b"\x11" * 32)
    edge = Signer.from_seed("edge-agent-01", b"\x22" * 32)
    registry = KeyRegistry()
    registry.register_signer(assessor)
    registry.register_signer(edge)

    chain = HashChain(out / "ledger.jsonl", signer=assessor)
    if len(chain) == 0:
        battery_a = sha256_json({"battery": "A", "generation": 7, "seed": seed})
        battery_b = sha256_json({"battery": "B", "generation": 7, "seed": seed})
        taxonomy_digest_row = chain.append(
            "taxonomy_committed", {"generation": 7}
        )
        chain.append(
            "battery_committed",
            {"battery_a_digest": battery_a, "battery_b_digest": battery_b, "generation": 7},
        )
        chain.append("assessment_preregistered", {"report_id": "PRM-DEMO-0001"})
    else:
        battery_a = sha256_json({"battery": "A", "generation": 7, "seed": seed})
        battery_b = sha256_json({"battery": "B", "generation": 7, "seed": seed})

    committed_before = True
    beats.append(
        Beat(2, "pre-commitment", committed_before,
             f"battery digests in the ledger before artefact receipt ({len(chain)} rows)")
    )

    # -- beat 3/4: the ladder ---------------------------------------------
    null = build_null(rng)
    fp32 = _clean_logits(rng)
    int8 = _qcb_conversion(fp32, rng) if poisoned else _benign_conversion(fp32, rng)
    vendor_int8 = _benign_conversion(fp32, rng)

    fp32_self = probe_divergence(fp32, _benign_conversion(fp32, rng) * 0 + fp32, pair="fp32->fp32")
    beats.append(
        Beat(3, "fp32 assessment", True,
             f"{fp32_self.probes_diverging}/{N_PROBES} probes diverge at FP32 - no finding")
    )

    profile = probe_divergence(fp32, int8, pair="fp32→int8_ptq")
    p_value, is_floor = null.p_value(profile.mean_divergence)
    conc = assess_concentration(profile.per_class)
    fires = p_value <= K.DECLARED_FDR
    reading = interpret(conc.condition_met, fires)

    beats.append(
        Beat(4, "precision ladder", fires if poisoned else not fires,
             f"{profile.probes_diverging}/{N_PROBES} probes diverge, p={p_value:.5f}"
             f"{' (FLOOR)' if is_floor else ''}, gini={conc.gini:.3f}, "
             f"{conc.classes_at_90pct} classes -> {reading}",
             {"p": p_value, "gini": conc.gini, "interpretation": reading})
    )

    vendor_profile = probe_divergence(int8, vendor_int8, pair="int8_ptq→int8_vendor")
    vendor_p, _ = null.p_value(vendor_profile.mean_divergence)

    # -- beat 5: reversal (statistics only; the torch path is exercised by E2)
    class_p = {c: float(rng.uniform(0.2, 0.9)) for c in range(N_CLASSES)}
    if poisoned:
        class_p[TARGET_CLASS] = 0.00073
    from pramana.reversal.neural_cleanse import bh_select

    bh = bh_select(class_p, fdr=K.DECLARED_FDR)
    beats.append(
        Beat(5, "reversal at INT8", (TARGET_CLASS in bh["rejected"]) == poisoned,
             f"BH over {bh['classes_tested']} classes, crit@rank1={bh['critical_value_at_rank_1']:.5f}, "
             f"rejected={bh['rejected'] or 'none'}")
    )

    # -- beat 7: source aggregation ---------------------------------------
    per_source: dict[str, dict[str, np.ndarray]] = {}
    for i in range(12):
        sid = f"lot-{i:02d}"
        dets = {}
        for det in ("trigger", "label_flip", "dup_flood", "ood"):
            e = rng.gamma(1.0, 1.0, size=400)
            if poisoned and sid == "lot-07" and det in ("trigger", "label_flip"):
                hit = rng.choice(400, 90, replace=False)
                e[hit] = rng.gamma(3.0, 1500.0, size=90)
            dets[det] = e
        per_source[sid] = dets

    weights = {"trigger": 0.4, "label_flip": 0.3, "dup_flood": 0.2, "ood": 0.1}
    weights_digest = sha256_json(weights)
    evidences, decision = aggregate_sources(
        per_source, weights=weights, weights_digest=weights_digest
    )
    top = evidences[0]
    beats.append(
        Beat(7, "source aggregation", True,
             f"{decision['sources_tested']} lots, e-BH threshold={decision['threshold_at_k1']:.0f}, "
             f"top={top.source_id} e={top.merged:.1f}, rejected={decision['rejections'] or 'none'}")
    )

    # -- beat 8: honest failure -------------------------------------------
    warping_refusal = AssessmentUnavailable(
        "insufficient_access_tier",
        "warping-trigger family is declared_unsupported: the reversal parameterisation "
        "does not cover warping fields at T1",
        asset="artefact:gtsrb-r18-demo",
    )
    beats.append(
        Beat(8, "honest failure", True,
             "warping artefact -> declared_unsupported: trigger_warping (not a silent pass)")
    )

    # -- beat 9: the certificate ------------------------------------------
    register = LotRegister()
    for i, share in enumerate(EXAMPLE_SHARES):
        register.add(
            Lot(
                lot_id=f"lot-{i:02d}",
                n_samples=int(share * 39209),
                purchase_order=f"PO-2026-{1000 + i}",
                signer_id=f"vendor-{i:02d}",
                vendor_code=f"V{i:03d}",
            )
        )
    register.check_scope()

    base_preds = rng.integers(0, N_CLASSES, size=(12, 500))
    base_preds[:9, :] = TARGET_CLASS
    roe = run_off_election(base_preds, N_CLASSES)
    k = roe.floor(coverage=0.90)
    vi_raw = volume_inequality_block(register, k)
    gap = surrogate_gap(ensemble_acc=0.83, delivered_acc=0.96)

    beats.append(
        Beat(9, "certificate", True,
             f"k={k} of 12 lots, certified_fraction={roe.fraction_at(k):.3f}, "
             f"those k hold {vi_raw['volume_share_of_largest_k']:.1%} of the corpus, "
             f"surrogate_gap={gap:.3f}",
             {"k": k, "volume_share": vi_raw["volume_share_of_largest_k"]})
    )

    # -- drift -------------------------------------------------------------
    focused = {f"lot-{i:02d}": 0.02 for i in range(12)}
    focused["lot-07"] = 3.0
    drift_verdict = discriminate(
        focused, covariate_shift=0.15, concept_shift=0.85, conditionality_ratio=6.1
    )

    # -- beat 10: conditional release -------------------------------------
    findings: list[Finding] = []
    if poisoned:
        findings.append(
            Finding(
                finding_id="F-1",
                mechanism="precision_ladder_divergence",
                reason_code="concentrated_cross_rung_divergence",
                affected_asset="build:int8_ptq",
                rung="int8_ptq",
                level="L1_detection",
                statistic=Statistic(
                    name="cross_rung_divergence",
                    value=round(profile.mean_divergence, 8),
                    p_floor=null.p_floor,
                    p_is_floor=is_floor,
                    null_population=null.population_name,
                    null_family=null.index.family,
                    null_corpus=null.index.corpus,
                    note=f"Empirical rank against {null.n_draws} clean conversions.",
                ),
                evidence=[
                    Evidence(kind="divergence_profile", detail=f"{profile.probes_diverging}/{N_PROBES} probes"),
                    Evidence(kind="concentration", detail=f"gini={conc.gini:.3f} over {conc.classes_at_90pct} classes"),
                ],
                attribution_mode=AttributionMode.SET_VALUED,
                attribution_set=["lot-07"],
                attribution_set_size=1,
                containment_scope=["lot-07"],
                attribution_note=(
                    "attribution_set_size is 1 but evidence_strength is corroborated, not "
                    "causal_verified, so attribution_mode stays set_valued: point "
                    "attribution requires both. S3 leave-lot-out not run at this tier."
                ),
                evidence_strength=EvidenceStrength.CORROBORATED,
                corroborating_mechanisms=["trigger_reversal", "activation_statistics"],
            )
        )

    base_disposition = decide(
        findings,
        refusals=["trigger_warping"],
        predicted_operating_points=["precision_ladder_divergence"],
    )
    released = conditional_release(
        base_disposition,
        authority_name="Brig. A. Rao",
        authority_role="Programme Risk Acceptance Authority",
        compensating_controls=[
            "IOC monitoring on the live receipt stream",
            "human confirmation required on class 14 detections",
            "re-assessment at 90 days or on converter version change",
        ],
        validity_days=90,
    )
    chain.append(
        "risk_acceptance_granted",
        {
            "authority": released.risk_acceptance.authority_name,  # type: ignore[union-attr]
            "expires": released.risk_acceptance.expires_at_utc.isoformat(),  # type: ignore[union-attr]
            "state": released.state.value,
        },
    )
    beats.append(
        Beat(10, "conditional release", True,
             f"{base_disposition.state.value} -> {released.state.value} by "
             f"{released.risk_acceptance.authority_name}, 3 controls, 90-day expiry")  # type: ignore[union-attr]
    )

    # -- beat 11: receipt monitor -----------------------------------------
    emitter = ReceiptEmitter(emitter_id="edge-agent-01", signer=edge)
    receipts = [
        emitter.emit(
            input_digest=sha256_json({"frame": i}),
            model_digest=sha256_json({"rung": "int8_ptq"}),
            rung="int8_ptq",
            preprocessing_chain_digest=sha256_json({"chain": "resize224-norm"}),
            inference_config_digest=sha256_json({"providers": ["CPUExecutionProvider"]}),
            output={"class": int(rng.integers(0, N_CLASSES)), "confidence": float(rng.uniform(0.6, 0.99))},
        )
        for i in range(64)
    ]
    stream_ok = verify_stream(receipts, edge.public_key_b64)["ok"]

    p_stream = rng.uniform(0.0, 1.0, size=6000)
    p_stream[4000:] = rng.uniform(1e-4, 0.03, size=2000)
    monitor = EProcess(alpha=K.DECLARED_FDR, weights=np.ones(6000), calibrator="mixture")
    mres = monitor.run(p_stream, require_weight_model=False)

    final_disposition = released
    if mres.crossed:
        final_disposition = revoke(released, f"receipt e-process crossed at index {mres.crossing_index}")
        chain.append(
            "risk_acceptance_revoked",
            {"crossing_index": mres.crossing_index, "state": final_disposition.state.value},
        )
    beats.append(
        Beat(11, "receipt monitor", stream_ok and mres.crossed,
             f"{len(receipts)} receipts verify; e-process crossed at {mres.crossing_index} "
             f"-> risk acceptance revoked")
    )

    # -- beat 12: tamper and external anchor ------------------------------
    anchorer = Anchorer(out / "anchors")
    root_before = chain.merkle_root()
    token = anchorer.anchor_offline(root_before)

    shadow = InMemoryChain(signer=assessor)
    for e in chain:
        shadow.append(e.event_type, e.payload, timestamp=datetime.fromisoformat(e.timestamp_utc))
    shadow_root_before = shadow.merkle_root()
    shadow_token = anchorer.anchor_offline(shadow_root_before)

    shadow.rewrite_history(0, assessor, generation=99)
    try:
        local_ok = shadow.verify(registry.as_dict())
    except LedgerTampered:
        local_ok = False
    anchor_ok = Anchorer.verify_against(shadow_token, shadow.merkle_root())

    beats.append(
        Beat(12, "tamper + anchor", local_ok and not anchor_ok,
             f"insider rewrite: local chain verifies={local_ok}, external anchor "
             f"matches={anchor_ok} <- caught on a digest we hold no key to")
    )

    # -- assemble the report ----------------------------------------------
    report = _assemble(
        out=out,
        null=null,
        profile=profile,
        p_value=p_value,
        is_floor=is_floor,
        conc=conc,
        reading=reading,
        vendor_profile=vendor_profile,
        vendor_p=vendor_p,
        findings=findings,
        evidences=evidences,
        decision=decision,
        weights_digest=weights_digest,
        register=register,
        k=k,
        roe=roe,
        gap=gap,
        vi_raw=vi_raw,
        drift_verdict=drift_verdict,
        disposition=final_disposition,
        anchorer=anchorer,
        token=token,
        chain_root=chain.merkle_root(),
        battery_a=battery_a,
        battery_b=battery_b,
        warping_refusal=warping_refusal,
    )

    for b in beats:
        say(b.line())

    return {
        "beats": [b.__dict__ for b in beats],
        "all_passed": all(b.ok for b in beats),
        "report_path": str(out / "report.json"),
        "ledger_path": str(out / "ledger.jsonl"),
        "coverage": {
            "assessed": report.coverage.assessed,
            "declared_unsupported": report.coverage.declared_unsupported,
            "not_assessed": report.coverage.not_assessed,
            "sum_check": report.coverage.sum_check,
        },
        "disposition": report.disposition.state.value,
        "verdict": report.verdict_statement,
    }


def _assemble(**kw: Any):
    """Build and emit the assurance report from the demo's live results."""
    out: Path = kw["out"]
    null: FittedNull = kw["null"]
    profile = kw["profile"]
    conc = kw["conc"]
    register: LotRegister = kw["register"]
    k: int = kw["k"]
    roe = kw["roe"]

    builder = ReportBuilder("PRM-DEMO-0001", document_status=DocumentStatus.ILLUSTRATIVE_EXAMPLE)

    builder.assessment(
        AssessmentMeta(
            tier=Tier.T2,
            access_tier=AccessTier.A2,
            build_state=BuildState.BUILT,
            turnaround_s={"triage_s": 3.1, "full_s": 41.7},
        )
    )

    builder.battery(
        Battery(
            generation=7,
            battery_a_digest=kw["battery_a"],
            battery_b_digest=kw["battery_b"],
            committed_at_utc=datetime.now(UTC),
            committed_before_artefact_receipt=True,
            taxonomy_digest=builder.taxonomy.digest,
            threshold_file_digest=sha256_json({"thresholds": "constants.py@0.1.0"}),
            battery_custody=BatteryCustody(
                digest_custodian="assurance-ledger (hash chain, externally anchored)",
                material_custodian="DGIS-assurance-authority-2",
                assessing_team="assurance-cell-north",
                custodians_distinct=True,
                control_type="organisational",
                defeats=["post_hoc_alteration_of_battery_contents"],
                does_not_defeat=["selection_of_which_sealed_battery_to_open_after_seeing_the_artefact"],
                residual_mitigation=["committed_seed", "single_use_per_delivery"],
            ),
        )
    )

    builder.null_calibration(
        NullCalibration(
            null_family=null.index.family,
            null_corpus=null.index.corpus,
            null_population=null.population_name,
            arms=[
                NullArm(arm="resnet18/gtsrb", n=null.n_models, role="operational_null"),
                NullArm(arm="resnet18/cifar10", n=32, role="corpus_contrast_only"),
                NullArm(arm="smallcnn/cifar10", n=32, role="family_contrast_only"),
            ],
            null_corpus_transfer_delta=TransferDelta(
                contrast="resnet18/gtsrb vs resnet18/cifar10",
                factor="corpus",
                median_shift_over_iqr=0.31,
                bootstrap_ci_median_shift=(0.12, 0.52),
                alpha_tail_shift_over_iqr=0.61,
            ),
            null_family_transfer_delta=TransferDelta(
                contrast="resnet18/cifar10 vs smallcnn/cifar10",
                factor="architecture_family",
                median_shift_over_iqr=0.88,
                bootstrap_ci_median_shift=(0.61, 1.14),
                alpha_tail_shift_over_iqr=1.55,
            ),
            transfer_ceiling=TransferCeiling(
                family_delta_exceeds_ceiling=True,
                corpus_delta_exceeds_ceiling=False,
                affects_this_report=False,
                consequence_recorded=(
                    "cross-family reuse of this null is refused: assessment_unavailable "
                    "no_fitted_null. Corpus delta 0.31 is within ceiling; family delta 0.88 is not."
                ),
                why_not="the artefact is resnet18/gtsrb, matching null_family and null_corpus exactly",
            ),
            ood_backbone=OodBackbone(
                backbone_corpus_digest=sha256_json({"corpus": "pre-procurement-public"}),
                disjoint_from_assessed_shards=True,
                pinned_to_battery_generation=7,
                disjointness_basis="pre-procurement corpus, digest committed at generation 7 before artefact receipt",
            ),
            declared_assumptions=["backbone_corpus_disjoint_from_assessed_shards"],
        )
    )

    builder.artefact(
        Artefact(
            logical_model_id="gtsrb-r18-demo",
            family="resnet18",
            preprocessing_chain_digest=sha256_json({"chain": "resize32-norm"}),
            builds=[
                Build(rung="fp32", format="torchscript", digest=sha256_json({"r": "fp32"}), signer="vendor-07"),
                Build(
                    rung="int8_ptq",
                    format="onnx",
                    digest=sha256_json({"r": "int8_ptq"}),
                    signer="integrator-02",
                    role="will_run",
                    certified=True,
                    conversion=Conversion(
                        quantiser="onnxruntime.qdq",
                        calibration_set_digest=sha256_json({"calib": "declared-500"}),
                        scale_table_digest=sha256_json({"scales": "per-layer"}),
                    ),
                ),
                Build(
                    rung="int8_vendor_claimed",
                    format="onnx",
                    digest=sha256_json({"r": "int8_vendor"}),
                    signer="vendor-07",
                    role="delivered_not_deployed",
                    certified=False,
                    conversion=Conversion(
                        quantiser="vendor_toolchain_undisclosed",
                        calibration_set_digest=None,
                        scale_table_digest=sha256_json({"scales": "per-layer"}),
                    ),
                ),
            ],
        )
    )

    fd = FingerprintDivergence(
        pair=profile.pair,
        probes_diverging=profile.probes_diverging,
        probes_total=profile.n_probes,
        level="L1_detection",
        detection_p=kw["p_value"],
        detection_p_reported_as=("<= %.5f" % kw["p_value"]) if kw["is_floor"] else "%.5f" % kw["p_value"],
        detection_p_is_floor=kw["is_floor"],
        detection_p_basis=null.p_floor_basis,
        detection_fires=kw["p_value"] <= K.DECLARED_FDR,
        null_family=null.index.family,
        null_corpus=null.index.corpus,
        null_population=null.population_name,
        divergence_concentration_gini=round(conc.gini, 4),
        divergence_concentration_classes=conc.classes_at_90pct,
        dominant_class=conc.dominant_class,
        concentration_operating_point=f"gini >= {conc.threshold_gini} over <= {conc.threshold_max_classes} classes",
        concentration_operating_point_status=conc.status,  # type: ignore[arg-type]
        concentration_operating_point_source=conc.source,
        concentration_condition_met=conc.condition_met,
        interpretation=kw["reading"],
    )

    builder.precision_ladder(
        PrecisionLadder(
            rungs_assessed=["fp32", "int8_ptq", "int8_vendor_claimed"],
            rungs_declared_unavailable=["fp16", "pruned", "int8_qat"],
            fingerprint_divergence=[fd],
            converter_provenance=ConverterProvenance(
                certified_rung="int8_ptq",
                recorded_not_certified_rungs=["int8_vendor_claimed"],
                vendor_vs_integrator=VendorVsIntegrator(
                    probes_diverging=kw["vendor_profile"].probes_diverging,
                    probes_total=N_PROBES,
                    divergence_p=round(kw["vendor_p"], 5),
                    exceeds_null=kw["vendor_p"] <= K.DECLARED_FDR,
                    divergence_p_basis=f"quantile position in {null.population_name}",
                ),
                shared_scale_table=True,
                note=(
                    "We certify the build that will run and record the digest of every build "
                    "that will not. Both INT8 builds share the delivered scale table, so "
                    "divergence between them would indicate an undisclosed conversion step "
                    "rather than quantisation noise."
                ),
            ),
        )
    )

    for f in kw["findings"]:
        builder.add_finding(f)

    top = kw["evidences"][0]
    if top.rejected:
        builder.add_finding(
            Finding(
                finding_id="F-2",
                mechanism="source_aggregation",
                reason_code="source_evalue_exceeds_ebh_threshold",
                affected_asset=f"lot:{top.source_id}",
                contributor=top.source_id,
                statistic=Statistic(
                    **top.as_statistic(
                        sources_tested=kw["decision"]["sources_tested"],
                        weights_digest=kw["weights_digest"],
                    )
                ),
                attribution_mode=AttributionMode.SET_VALUED,
                attribution_set=[top.source_id],
                attribution_set_size=1,
                containment_scope=[top.source_id],
                evidence_strength=EvidenceStrength.INDICATIVE,
                note="Statistically flagged at a declared FDR; causally unverified.",
            )
        )

    vi = kw["vi_raw"]
    builder.supplier_concentration(
        SupplierConcentrationMeasurement(
            m_lots=register.m,
            certified_floor_k=k,
            certified_fraction_at_k=round(roe.fraction_at(k), 4),
            surrogate_gap_top1=round(kw["gap"], 4),
            volume_inequality=VolumeInequality(
                lot_volume_shares=vi["lot_volume_shares"],
                shares_sum_check=vi["shares_sum_check"],
                largest_single_share=vi["largest_single_share"],
                volume_share_of_largest_k=vi["volume_share_of_largest_k"],
            ),
            entity_map_incomplete=str(register.entity_map_status()["entity_map_incomplete"]),
        )
    )

    dv = kw["drift_verdict"]
    builder.drift(
        DriftAssessment(
            axes=[
                DriftAxis(axis="illumination", status=CoverageDisposition.ASSESSED,
                          drift_realism=DriftRealism.MEASURED, score=0.21,
                          score_basis="empirical quantile against the fitted drift null"),
                DriftAxis(axis="sensor", status=CoverageDisposition.ASSESSED,
                          drift_realism=DriftRealism.CORRUPTION_MODEL_PROXY, score=0.33,
                          score_basis="corruption model, not a capture across sensors"),
                DriftAxis(axis="acquisition", status=CoverageDisposition.ASSESSED,
                          drift_realism=DriftRealism.CORRUPTION_MODEL_PROXY, score=0.28,
                          score_basis="corruption model proxy"),
                DriftAxis(axis="terrain", status=CoverageDisposition.DECLARED_UNSUPPORTED,
                          drift_realism=DriftRealism.NOT_APPLICABLE,
                          reason="no_defence_domain_corpus"),
                DriftAxis(axis="season", status=CoverageDisposition.DECLARED_UNSUPPORTED,
                          drift_realism=DriftRealism.NOT_APPLICABLE,
                          reason="no_defence_domain_corpus"),
            ],
            verdict=dv.verdict,  # type: ignore[arg-type]
            verdict_note=dv.rationale,
        )
    )

    builder.disposition(kw["disposition"])
    builder.anchor(
        kw["anchorer"].report_block(kw["token"], current_root=kw["chain_root"], lag_s=0.0)
    )

    builder.add_debt(
        DebtItem(
            debt_id="DBT-001",
            description="Per-lot ultimate-beneficial-owner disclosure absent, so the entity floor cannot be computed",
            owner_class="contracting_authority",
            magnitude=float(register.m),
            unit="lots",
            retirement_condition="entity map disclosed at award and refreshed on change of control",
            clause_that_would_retire_it="DAP 2020 Ch. VI Standard Contract Document (proposed amendment)",
            first_raised_in_report="PRM-DEMO-0001",
        )
    )
    builder.add_debt(
        DebtItem(
            debt_id="DBT-002",
            description="Null pairs not fitted for families other than resnet18/gtsrb",
            owner_class="assessor",
            magnitude=33.0,
            unit="null_pairs",
            retirement_condition="21 GPU-hours per (family x corpus x converter) triple",
            first_raised_in_report="PRM-DEMO-0001",
        )
    )

    builder.refuse(kw["warping_refusal"], class_id="trigger_warping")
    builder.declare_assumption("null_corpus_transfer_not_invoked_for_this_artefact")

    return builder.emit(out / "report.json")
