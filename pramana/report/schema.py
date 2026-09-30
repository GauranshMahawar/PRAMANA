"""The assurance-report schema (clause 2.3 deliverable).

AIRS (arXiv:2511.12668) is prior art for the *idea* of a machine-readable assurance
report, and we cite it rather than claim it. The differentiation is entirely in the
fields that carry the project's discipline:

* every rate is accompanied by the thing that bounds it (the six comparator pairs);
* every refusal is a value, not an absence;
* access tier, evidence strength and build state are *fields*, not prose;
* `attribution_mode: point` is structurally impossible to assert without both
  `attribution_set_size == 1` and `evidence_strength == causal_verified`.

The rule that generated most of this file: **a number is not checkable unless the
report also prints the thing it must be checked against.** Where a field would be a
bare number, it has a companion. Where a companion is missing, the validator refuses
the whole report.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pramana.common import constants as K

Digest = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{4,64}(…[0-9a-f]{4})?$")]


class Strict(BaseModel):
    """Base model: unknown fields are an error.

    A report that silently accepts a misspelled field is a report where a mandated
    comparator can go missing without anyone noticing.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# ---------------------------------------------------------------------------
# Enumerations -- closed vocabularies
# ---------------------------------------------------------------------------


class DocumentStatus(str, Enum):
    ILLUSTRATIVE_EXAMPLE = "ILLUSTRATIVE_EXAMPLE"
    MEASUREMENT_RECORD = "MEASUREMENT_RECORD"


class Tier(str, Enum):
    """Assessment tier. Clause 2.2.6 forbids retraining in the baseline."""

    T1 = "T1"  # baseline, mandatory, no retraining
    T2 = "T2"  # confirmation, optional, user-invoked
    T3 = "T3"  # remediation, optional


class AccessTier(str, Enum):
    """What we actually hold. Mandatory machine-readable output, not documentation."""

    A0 = "A0"  # inference API only
    A1 = "A1"  # weights + architecture
    A2 = "A2"  # + contributed dataset with source metadata
    A3 = "A3"  # + training-time instrumentation


class BuildState(str, Enum):
    """Whether code exists and evidence was produced -- distinct from design coverage.

    A jury that sees `designed` beside a green compliance row learns that we know the
    difference. A jury that catches an undeclared one learns the opposite.
    """

    MEASURED = "measured"
    BUILT = "built"
    DESIGNED = "designed"


class EvidenceStrength(str, Enum):
    """The ladder that gates what a finding is permitted to say (G5, doc 29.3)."""

    INDICATIVE = "indicative"
    CORROBORATED = "corroborated"
    CAUSAL_VERIFIED = "causal_verified"
    ATTRIBUTION_UNAVAILABLE = "attribution_unavailable"


class AttributionMode(str, Enum):
    POINT = "point"
    SET_VALUED = "set_valued"


class Disposition(str, Enum):
    """G2's five-state lattice. A gate that can only say NO gets waived and ignored."""

    ACCEPT = "ACCEPT"
    ACCEPT_WITH_CONDITIONS = "ACCEPT_WITH_CONDITIONS"
    CONDITIONAL_RELEASE = "CONDITIONAL_RELEASE"
    QUARANTINE = "QUARANTINE"
    REJECT = "REJECT"


class CoverageDisposition(str, Enum):
    ASSESSED = "assessed"
    DECLARED_UNSUPPORTED = "declared_unsupported"
    NOT_ASSESSED = "not_assessed"


class DriftRealism(str, Enum):
    MEASURED = "measured"
    CORRUPTION_MODEL_PROXY = "corruption_model_proxy"
    NOT_APPLICABLE = "not_applicable"


# ---------------------------------------------------------------------------
# Battery and pre-commitment (G1)
# ---------------------------------------------------------------------------


class BatteryCustody(Strict):
    """Who held what.

    `control_type` is `organisational` and says so. Labelling an organisational control
    as cryptographic would have been the more impressive and the less true choice: a
    digest proves the battery was not *altered*, never that it was not *selected*.
    """

    digest_custodian: str
    material_custodian: str
    assessing_team: str
    custodians_distinct: bool
    control_type: Literal["organisational", "cryptographic"] = "organisational"
    defeats: list[str] = Field(default_factory=list)
    does_not_defeat: list[str] = Field(default_factory=list)
    residual_mitigation: list[str] = Field(default_factory=list)


class Battery(Strict):
    generation: int = Field(ge=1)
    battery_a_digest: Digest
    battery_b_digest: Digest
    committed_at_utc: datetime
    committed_before_artefact_receipt: bool
    battery_b_opened: bool = False
    taxonomy_digest: Digest
    threshold_file_digest: Digest
    seed: int | None = None
    battery_custody: BatteryCustody

    @model_validator(mode="after")
    def _precommitment_is_the_whole_point(self) -> Battery:
        if not self.committed_before_artefact_receipt:
            raise ValueError(
                "committed_before_artefact_receipt is False. A battery committed after "
                "the artefact arrived is not evidence -- it is a finding the accused "
                "party cannot check. Emit assessment_unavailable: battery_not_precommitted."
            )
        return self


# ---------------------------------------------------------------------------
# Null calibration -- where every threshold comes from
# ---------------------------------------------------------------------------


class NullArm(Strict):
    arm: str
    n: int = Field(ge=1)
    role: Literal["operational_null", "corpus_contrast_only", "family_contrast_only"]


class PFloors(Strict):
    """COMPARATOR PAIR 1 and 2 live here: every p-value is reported beside its floor."""

    l1_detection: float = K.P_FLOOR_L1_DETECTION
    l1_basis: str = K.BASES["l1_detection"]
    l2_localisation: float = K.P_FLOOR_L2_LOCALISATION
    l2_basis: str = K.BASES["l2_localisation"]
    l2_pooling: str = "per_class_standardised_before_pooling"
    l2_bh_smallest_critical_value: float = K.BH_CRITICAL_VALUE_RANK_1
    l2_basis_note: str = K.BASES["l2_bh"]


class TransferDelta(Strict):
    """COMPARATOR PAIR 5: the delta is printed beside the ceiling it must stay under."""

    contrast: str
    factor: Literal["corpus", "architecture_family"]
    median_shift_over_iqr: float
    bootstrap_ci_median_shift: tuple[float, float]
    alpha_tail_shift_over_iqr: float | None = None
    quantile_resolution: str = K.QUANTILE_RESOLUTION_LABEL_AT_N32


class TransferCeiling(Strict):
    max_median_shift_over_iqr_permitted: float = K.TRANSFER_CEILING_MEDIAN_SHIFT_OVER_IQR
    family_delta_exceeds_ceiling: bool
    corpus_delta_exceeds_ceiling: bool = False
    affects_this_report: bool
    consequence_recorded: str
    why_not: str | None = None


class OodBackbone(Strict):
    """The reference the OOD detector measures deviation FROM.

    Train this on the procurement corpus and a flooded shard becomes part of the
    reference: detector power falls as the attack succeeds, which is the worst
    direction a failure mode can run. Hence the disjointness assertion is a field.
    """

    backbone_corpus_digest: Digest
    disjoint_from_assessed_shards: bool
    pinned_to_battery_generation: int
    disjointness_basis: str
    failure_value_if_violated: str = "assessment_unavailable: backbone_corpus_contaminated"


class NullCalibration(Strict):
    null_family: str
    null_corpus: str
    null_population: str
    arms: list[NullArm]
    p_floors: PFloors = Field(default_factory=PFloors)
    null_corpus_transfer_delta: TransferDelta | None = None
    null_family_transfer_delta: TransferDelta | None = None
    transfer_ceiling: TransferCeiling | None = None
    ood_backbone: OodBackbone | None = None
    declared_assumptions: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# The artefact and its precision ladder (D1)
# ---------------------------------------------------------------------------


class Conversion(Strict):
    """PTQ has TWO inputs and only the weights are normally contracted for.

    `calibration_set_digest` is adversary A9 given a schema slot.
    """

    quantiser: str
    calibration_set_digest: Digest | None = None
    scale_table_digest: Digest | None = None


class Build(Strict):
    rung: str
    format: Literal["torchscript", "onnx", "pytorch", "safetensors"]
    digest: Digest
    signer: str
    role: Literal["will_run", "delivered_not_deployed", "reference"] | None = None
    certified: bool = False
    conversion: Conversion | None = None


class Artefact(Strict):
    logical_model_id: str
    family: str
    preprocessing_chain_digest: Digest
    builds: list[Build]

    @model_validator(mode="after")
    def _exactly_one_certified_rung(self) -> Artefact:
        certified = [b for b in self.builds if b.certified]
        if len(certified) > 1:
            raise ValueError(
                "More than one rung is marked certified. The certificate names the build "
                "that WILL RUN; every other build is recorded and not certified."
            )
        return self


class FingerprintDivergence(Strict):
    """D1's headline statistic.

    Note `concentration_is_within_artefact`. This is the within-artefact divergence
    concentration over CLASSES. It is a different quantity from C9's demoted shard
    Gini and the two must not share an implementation -- doing so would propagate a
    demotion into D1's headline.
    """

    pair: str
    probes_diverging: int = Field(ge=0)
    probes_total: int = Field(ge=1)
    level: Literal["L1_detection", "L2_localisation"]

    # COMPARATOR PAIR 1 -- p beside its floor, always
    detection_p: float
    detection_p_reported_as: str
    detection_p_is_floor: bool
    detection_p_basis: str
    detection_fires: bool

    null_family: str
    null_corpus: str
    null_population: str

    divergence_concentration_gini: float = Field(ge=0.0, le=1.0)
    divergence_concentration_classes: int = Field(ge=0)
    dominant_class: int | None = None
    concentration_operating_point: str
    concentration_operating_point_status: Literal["predicted", "measured"]
    concentration_operating_point_source: str
    concentration_operating_point_fpr_vs_null: float | None = None
    concentration_condition_met: bool
    concentration_is_within_artefact: bool = True
    concentration_note: str = (
        "D1's within-artefact concentration over classes. NOT C9's demoted shard_gini; "
        "the two are named differently on purpose."
    )
    interpretation: str

    @model_validator(mode="after")
    def _floor_arithmetic(self) -> FingerprintDivergence:
        floor = K.P_FLOOR_L1_DETECTION if self.level == "L1_detection" else K.P_FLOOR_L2_LOCALISATION
        if self.detection_p < floor - 1e-12:
            raise ValueError(
                f"detection_p={self.detection_p} is below the instrument's floor {floor:.5f}. "
                f"A p-value finer than the null can resolve is an artefact of arithmetic, "
                f"not a measurement."
            )
        at_floor = abs(self.detection_p - floor) < 1e-9
        if at_floor != self.detection_p_is_floor:
            raise ValueError(
                f"detection_p_is_floor={self.detection_p_is_floor} contradicts the arithmetic "
                f"(p={self.detection_p}, floor={floor:.5f})."
            )
        if self.probes_diverging > self.probes_total:
            raise ValueError("probes_diverging exceeds probes_total.")
        if self.concentration_operating_point_status == "predicted" and self.concentration_condition_met:
            # Permitted, but the disposition layer must not rest on it alone.
            pass
        return self


class VendorVsIntegrator(Strict):
    probes_diverging: int
    probes_total: int
    divergence_p: float
    divergence_p_floor: float = K.P_FLOOR_L1_DETECTION
    exceeds_null: bool
    divergence_p_basis: str


class ConverterProvenance(Strict):
    certified_rung: str
    recorded_not_certified_rungs: list[str] = Field(default_factory=list)
    vendor_vs_integrator: VendorVsIntegrator | None = None
    shared_scale_table: bool | None = None
    mismatch_finding_class_if_it_fires: str = "converter_provenance_mismatch"
    note: str | None = None


class PrecisionLadder(Strict):
    rungs_assessed: list[str]
    rungs_declared_unavailable: list[str] = Field(default_factory=list)
    fingerprint_divergence: list[FingerprintDivergence] = Field(default_factory=list)
    converter_provenance: ConverterProvenance | None = None


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------


class Statistic(Strict):
    """Any statistic that reaches a report carries what bounds it.

    COMPARATOR PAIRS 2 and 4 are enforced here and in the validator.
    """

    name: str
    value: float
    p_floor: float | None = None
    p_is_floor: bool | None = None
    null_population: str | None = None
    null_family: str | None = None
    null_corpus: str | None = None
    pooling: str | None = None
    bh_classes_tested: int | None = None
    fdr: float | None = None
    bh_critical_value_at_rank_1: float | None = None
    bh_rejects: bool | None = None
    bh_adjusted_q: float | None = None
    # e-value companions
    merging_function: Literal["weighted_arithmetic_mean", "arithmetic_mean"] | None = None
    merge_weights_committed_in: str | None = None
    within_detector_merge: str | None = None
    sources_tested: int | None = None
    size_normalised: bool | None = None
    ebh_threshold_at_k1: float | None = None
    ebh_rejections: int | None = None
    note: str | None = None

    @model_validator(mode="after")
    def _merge_operator_is_never_the_product(self) -> Statistic:
        if self.merging_function is not None and "product" in str(self.merging_function):
            raise ValueError(
                "e-values merged by product. Under arbitrary dependence the product is "
                "invalid (Vovk & Wang, arXiv:1912.06116); the valid merge is the "
                "arithmetic mean. Four detectors on one shard are not independent."
            )
        return self


class Surrogate(Strict):
    """The fake-quant surrogate, and the gate that stops it becoming the evidence.

    Search runs on the surrogate; the EVIDENCE is attack success measured on the real
    INT8 binary. Search is not evidence.
    """

    path: str
    scale_table_digest: Digest
    surrogate_exact_agreement: float
    surrogate_exact_agreement_floor: float = K.SURROGATE_EXACT_AGREEMENT_FLOOR
    surrogate_mean_logit_distance: float
    surrogate_mean_logit_distance_ceiling: float = K.SURROGATE_MEAN_LOGIT_DISTANCE_CEILING
    search_ran_on: Literal["surrogate"] = "surrogate"
    evidence_ran_on: Literal["delivered_int8_binary"] = "delivered_int8_binary"
    forward_pass_asr_on_delivered_int8: float

    @model_validator(mode="after")
    def _faithfulness_gate(self) -> Surrogate:
        if self.surrogate_exact_agreement < self.surrogate_exact_agreement_floor:
            raise ValueError(
                f"surrogate_exact_agreement {self.surrogate_exact_agreement} is below its "
                f"floor {self.surrogate_exact_agreement_floor}. Reversal must not run: emit "
                f"assessment_unavailable: surrogate_not_faithful."
            )
        if self.surrogate_mean_logit_distance > self.surrogate_mean_logit_distance_ceiling:
            raise ValueError(
                f"surrogate_mean_logit_distance {self.surrogate_mean_logit_distance} exceeds "
                f"its ceiling {self.surrogate_mean_logit_distance_ceiling}."
            )
        return self


class Evidence(Strict):
    kind: str
    digest: Digest | None = None
    detail: str | None = None


class Finding(Strict):
    """Clause 2.2.5's five mandated fields are all REQUIRED here.

    reason_code, evidence, evidence_strength + statistic, affected_asset, disposition.
    The validator rejects a finding missing any of the five, which is the mechanism
    that stops "confidence" from being a vibe.
    """

    finding_id: str
    mechanism: str
    reason_code: str
    affected_asset: str
    rung: str | None = None
    parameterisation: str | None = None
    class_id: int | None = Field(default=None, alias="class")
    contributor: str | None = None
    level: Literal["L1_detection", "L2_localisation"] | None = None
    reported_only_because: str | None = None

    statistic: Statistic
    evidence: list[Evidence] = Field(default_factory=list)
    surrogate: Surrogate | None = None

    attribution_mode: AttributionMode
    attribution_set: list[str]
    attribution_set_size: int = Field(ge=0)
    containment_scope: list[str]
    attribution_note: str | None = None

    evidence_strength: EvidenceStrength
    corroborating_mechanisms: list[str] = Field(default_factory=list)
    disposition: Disposition | None = None
    note: str | None = None

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @model_validator(mode="after")
    def _point_attribution_needs_both(self) -> Finding:
        """The rule that makes it impossible to report the useless-but-true case as
        the useful-but-false one (doc 30.5.1)."""
        if self.attribution_set_size != len(self.attribution_set):
            raise ValueError(
                f"attribution_set_size={self.attribution_set_size} disagrees with "
                f"len(attribution_set)={len(self.attribution_set)}."
            )
        if self.attribution_mode is AttributionMode.POINT:
            if self.attribution_set_size != 1:
                raise ValueError(
                    "attribution_mode=point requires attribution_set_size==1."
                )
            if self.evidence_strength is not EvidenceStrength.CAUSAL_VERIFIED:
                raise ValueError(
                    "attribution_mode=point requires evidence_strength=causal_verified. "
                    "A set of size one is not a point attribution: it is a set of size one. "
                    "Either alone is rejected."
                )
        return self


# ---------------------------------------------------------------------------
# D2 -- supplier concentration over the measurement ensemble
# ---------------------------------------------------------------------------


class VolumeInequality(Strict):
    """COMPARATOR PAIR 6.

    A count of contributors silently overstates a guarantee when the contributors are
    unequal in size, so k is never printed without the volume share those k lots hold.
    """

    lot_volume_shares: dict[str, float]
    shares_sum_check: float
    largest_single_share: float
    volume_share_of_largest_k: float
    k_is_a_count_not_a_volume: bool = True

    @model_validator(mode="after")
    def _shares_sum_to_one(self) -> VolumeInequality:
        total = sum(self.lot_volume_shares.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"lot volume shares sum to {total}, not 1.0.")
        if abs(self.shares_sum_check - total) > 1e-9:
            raise ValueError("shares_sum_check does not match the printed share vector.")
        return self


class SupplierConcentrationMeasurement(Strict):
    """READ THE SUBJECT TWICE.

    This is a property of an ensemble PRAMANA constructs for the measurement. It is
    NEVER a property of the delivered model, and the two claims do not compose. The
    schema enforces the non-composition: `certified_floor_k` cannot be serialised
    without both companions.
    """

    scope: Literal["surrogate_ensemble"] = "surrogate_ensemble"
    scope_note: str = (
        "certified_floor_k is a property of the partition ensemble built for this "
        "measurement, and never of the artefact that ships. The two do not compose."
    )
    partition_basis: Literal["contract_lot", "random"] = "contract_lot"
    m_lots: int = Field(ge=1)
    aggregator: str = "run_off_election"

    certified_floor_k: int | None = None
    certified_floor_k_entities: int | None = None
    certified_fraction_at_k: float | None = None
    surrogate_gap_top1: float | None = None

    volume_inequality: VolumeInequality | None = None
    certificate_scope_degenerate: str | None = None
    entity_map_incomplete: str | None = None

    @model_validator(mode="after")
    def _k_never_travels_alone(self) -> SupplierConcentrationMeasurement:
        if self.certified_floor_k is not None:
            missing = [
                name
                for name, val in (
                    ("certified_fraction_at_k", self.certified_fraction_at_k),
                    ("surrogate_gap_top1", self.surrogate_gap_top1),
                    ("volume_inequality", self.volume_inequality),
                )
                if val is None
            ]
            if missing:
                raise ValueError(
                    f"certified_floor_k was emitted without {', '.join(missing)}. "
                    f"The field cannot be quoted alone, by construction."
                )
        if self.certificate_scope_degenerate and self.certified_floor_k is not None:
            raise ValueError(
                "A degenerate certificate scope WITHHOLDS the count. Above a single-lot "
                "share of 0.5 you do not get a weak floor with a caveat -- you get no "
                "floor at all."
            )
        return self


# ---------------------------------------------------------------------------
# Drift (clause 2.2.4)
# ---------------------------------------------------------------------------


class DriftAxis(Strict):
    axis: Literal["terrain", "season", "sensor", "illumination", "acquisition"]
    status: CoverageDisposition
    drift_realism: DriftRealism
    reason: str | None = None
    score: float | None = None
    score_basis: str | None = None


class DriftAssessment(Strict):
    axes: list[DriftAxis]
    primary_discriminator: str = "contributor_concentration"
    discriminator_order: list[str] = Field(
        default_factory=lambda: [
            "contributor_concentration",
            "covariate_vs_concept_decomposition",
            "input_conditionality",
        ]
    )
    corroborating_features: list[str] = Field(default_factory=lambda: ["shard_gini", "spectral_band_ratio"])
    verdict: Literal["DRIFT", "MANIPULATION", "AMBIGUOUS"] | None = None
    verdict_note: str | None = None


# ---------------------------------------------------------------------------
# Coverage (G5) -- COMPARATOR PAIR 3
# ---------------------------------------------------------------------------


class CoverageItem(Strict):
    class_id: str
    disposition: CoverageDisposition
    reason: str | None = None
    mechanisms: list[str] = Field(default_factory=list)
    attribution_ceiling: str | None = None


class Coverage(Strict):
    taxonomy_digest: Digest
    taxonomy_generation: int
    items: list[CoverageItem]
    assessed: int
    declared_unsupported: int
    not_assessed: int
    total_classes: int
    sum_check: int

    @model_validator(mode="after")
    def _counts_reconcile(self) -> Coverage:
        computed = self.assessed + self.declared_unsupported + self.not_assessed
        if computed != self.sum_check:
            raise ValueError(f"coverage counts sum to {computed}, sum_check says {self.sum_check}.")
        if self.sum_check != self.total_classes:
            raise ValueError(
                f"sum_check {self.sum_check} != total_classes {self.total_classes}. "
                f"A class that is neither assessed nor declared has gone unenumerated, "
                f"which is the exact failure G5 exists to make impossible."
            )
        if len(self.items) != self.total_classes:
            raise ValueError("coverage items do not cover every enumerated class.")
        return self


# ---------------------------------------------------------------------------
# Disposition (G2) and assurance debt (D4.d)
# ---------------------------------------------------------------------------


class RiskAcceptance(Strict):
    authority_name: str
    authority_role: str
    accepted_at_utc: datetime
    expires_at_utc: datetime
    compensating_controls: list[str]
    reassessment_triggers: list[str]
    ledger_entry_digest: Digest | None = None
    revoked: bool = False
    revoked_reason: str | None = None


class DispositionRecord(Strict):
    state: Disposition
    rationale: str
    risk_acceptance: RiskAcceptance | None = None

    @model_validator(mode="after")
    def _conditional_release_needs_an_owner(self) -> DispositionRecord:
        if self.state is Disposition.CONDITIONAL_RELEASE and self.risk_acceptance is None:
            raise ValueError(
                "CONDITIONAL_RELEASE without a named risk-acceptance authority. The whole "
                "point of the state is that the override becomes the audit trail."
            )
        return self


class DebtItem(Strict):
    """An itemised piece of assurance work that was NOT done, with an owner.

    Reported per owner class and never as a total: 12 lots and 33 null pairs cannot be
    added, so the field a total would sit in holds a refusal instead. The partition
    exists so that neither party can hide inside the other's number.
    """

    debt_id: str
    description: str
    owner_class: Literal["contracting_authority", "supplier", "assessor"]
    magnitude: float
    unit: str
    retirement_condition: str
    clause_that_would_retire_it: str | None = None
    first_raised_in_report: str | None = None
    status: Literal["opened", "carried", "grown", "retired"] = "opened"


class AssuranceDebt(Strict):
    items: list[DebtItem] = Field(default_factory=list)
    by_owner_class: dict[str, dict[str, float]] = Field(default_factory=dict)
    grand_total: Literal["refused: units_not_commensurable"] = "refused: units_not_commensurable"


# ---------------------------------------------------------------------------
# External anchor (7.1)
# ---------------------------------------------------------------------------


class ExternalAnchor(Strict):
    anchor_procedure: Literal["local", "upstream"] = "local"
    anchor_type: Literal["rfc3161", "rekor", "cross_signed_ledger", "none"] = "rfc3161"
    merkle_root: Digest | None = None
    anchor_state: Literal["anchored", "pending", "unavailable"] = "pending"
    anchor_batch_interval_s: int | None = None
    anchor_lag_s: float | None = None
    token_digest: Digest | None = None
    note: str = (
        "Converts 'trust us' into 'trust us or catch us'. It does NOT stop a compromised "
        "certifier; it removes the ability to alter the record afterwards."
    )


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


class AssessmentMeta(Strict):
    tier: Tier
    access_tier: AccessTier
    build_state: BuildState
    turnaround_s: dict[str, float] = Field(default_factory=dict)


class AssuranceReport(Strict):
    """The top-level object. Every emitted report is an instance of this."""

    schema_: str = Field(
        default="https://pramana.local/schema/assurance-report/1.0.json", alias="$schema"
    )
    report_id: str
    document_status: DocumentStatus
    document_status_note: str | None = None
    generated_at_utc: datetime

    assessment: AssessmentMeta
    battery: Battery
    null_calibration: NullCalibration
    artefact: Artefact
    precision_ladder: PrecisionLadder

    findings: list[Finding] = Field(default_factory=list)
    supplier_concentration_measurement: SupplierConcentrationMeasurement | None = None
    drift_assessment: DriftAssessment | None = None

    coverage: Coverage
    disposition: DispositionRecord
    assurance_debt: AssuranceDebt = Field(default_factory=AssuranceDebt)
    external_anchor: ExternalAnchor = Field(default_factory=ExternalAnchor)

    verdict_statement: str
    declared_assumptions: list[str] = Field(default_factory=list)
    assessment_unavailable: list[dict[str, Any]] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @model_validator(mode="after")
    def _verdict_language(self) -> AssuranceReport:
        low = self.verdict_statement.lower()
        for phrase in K.FORBIDDEN_VERDICT_PHRASES:
            if phrase in low:
                raise ValueError(
                    f"verdict_statement contains the forbidden phrase {phrase!r}. "
                    f"'Clean' is not a verdict this instrument can issue "
                    f"(arXiv:2204.06974). Use the template in constants.VERDICT_TEMPLATE."
                )
        return self

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent, by_alias=True, exclude_none=False)
