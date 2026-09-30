"""Assemble a complete assurance report from the pieces the mechanisms produce.

This module is where the discipline stops being a set of conventions and becomes a
type. Every comparator pair is populated here or the schema refuses to serialise;
every refusal reaches the report as a value rather than a gap; and the verdict is
rendered from a template so that no caller can write "the model is clean" even by
accident.

The order of operations matters and is not arbitrary:

1. Build the blocks.
2. Let Pydantic validate them -- most arithmetic rules fire here.
3. Run the independent validator over the serialised JSON -- which is what CI and a
   judge run, and which must agree with (2).

Step 3 is not redundant. The schema validates the object we built; the validator
validates the *document that ships*, tag-independently, exactly as it would validate a
report produced by anything else.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pramana.common import constants as K
from pramana.common.digest import short
from pramana.common.errors import AssessmentUnavailable
from pramana.coverage.generator import Taxonomy, generate_coverage
from pramana.report import validator as V
from pramana.report.schema import (
    AssessmentMeta,
    AssuranceReport,
    Battery,
    DispositionRecord,
    DocumentStatus,
    DriftAssessment,
    NullCalibration,
    Artefact,
    PrecisionLadder,
    SupplierConcentrationMeasurement,
    AssuranceDebt,
    DebtItem,
    ExternalAnchor,
    Finding,
)

__all__ = ["ReportBuilder", "render_verdict", "summarise_debt"]


def render_verdict(
    *,
    battery_digest: str,
    rungs: list[str],
    null_family: str,
    null_corpus: str,
    converter: str,
) -> str:
    """The only permitted verdict form.

    There is no parameter here that makes it say something stronger, because there is
    no stronger true statement available: arXiv:2204.06974 constructs backdoors no
    efficient black-box behavioural test detects, so absence of evidence is the
    ceiling. The build gate greps for the alternatives.
    """
    return K.VERDICT_TEMPLATE.format(
        battery_digest=short(battery_digest),
        rungs=", ".join(rungs) if rungs else "none",
        null_family=null_family,
        null_corpus=null_corpus,
        converter=converter,
    )


def summarise_debt(items: list[DebtItem]) -> dict[str, dict[str, float]]:
    """Summarise the assurance remainder **per owner class, never as a total**.

    Twelve lots and thirty-three null pairs cannot be added. A debt retired by a
    contract clause and a debt retired by GPU hours are not the same kind of thing, and
    summing them is the same error as quoting a concentration Gini with no population
    beside it. The partition exists so that neither party can hide inside the other's
    number -- which is why the grand-total field on the schema holds a refusal string
    rather than a figure.
    """
    out: dict[str, dict[str, float]] = {}
    for item in items:
        bucket = out.setdefault(item.owner_class, {})
        bucket[item.unit] = round(bucket.get(item.unit, 0.0) + item.magnitude, 4)
    return out


class ReportBuilder:
    """Accumulates blocks, then emits a validated report.

    Usage is deliberately linear: you cannot emit before the coverage object exists,
    and you cannot emit a certified *k* without its companions, because the schema
    rejects both.
    """

    def __init__(
        self,
        report_id: str,
        *,
        document_status: DocumentStatus = DocumentStatus.MEASUREMENT_RECORD,
        taxonomy_path: str | Path = "taxonomy.yaml",
    ):
        self.report_id = report_id
        self.document_status = document_status
        self.taxonomy = Taxonomy(taxonomy_path)

        self._assessment: AssessmentMeta | None = None
        self._battery: Battery | None = None
        self._null: NullCalibration | None = None
        self._artefact: Artefact | None = None
        self._ladder: PrecisionLadder | None = None
        self._findings: list[Finding] = []
        self._supplier: SupplierConcentrationMeasurement | None = None
        self._drift: DriftAssessment | None = None
        self._disposition: DispositionRecord | None = None
        self._anchor: ExternalAnchor = ExternalAnchor()
        self._debt: list[DebtItem] = []
        self._assumptions: list[str] = []
        self._unavailable: list[dict[str, Any]] = []
        self._not_assessed: dict[str, str] = {}

    # -- accumulation ------------------------------------------------------

    def assessment(self, meta: AssessmentMeta) -> ReportBuilder:
        self._assessment = meta
        return self

    def battery(self, block: Battery) -> ReportBuilder:
        self._battery = block
        return self

    def null_calibration(self, block: NullCalibration) -> ReportBuilder:
        self._null = block
        self._assumptions.extend(block.declared_assumptions)
        return self

    def artefact(self, block: Artefact) -> ReportBuilder:
        self._artefact = block
        return self

    def precision_ladder(self, block: PrecisionLadder) -> ReportBuilder:
        self._ladder = block
        return self

    def add_finding(self, finding: Finding) -> ReportBuilder:
        self._findings.append(finding)
        return self

    def supplier_concentration(self, block: SupplierConcentrationMeasurement) -> ReportBuilder:
        self._supplier = block
        return self

    def drift(self, block: DriftAssessment) -> ReportBuilder:
        self._drift = block
        return self

    def disposition(self, record: DispositionRecord) -> ReportBuilder:
        self._disposition = record
        return self

    def anchor(self, block: ExternalAnchor) -> ReportBuilder:
        self._anchor = block
        return self

    def add_debt(self, item: DebtItem) -> ReportBuilder:
        self._debt.append(item)
        return self

    def refuse(self, exc: AssessmentUnavailable, *, class_id: str | None = None) -> ReportBuilder:
        """Record a refusal as an output.

        A refusal that only appears in a log has not been declared. When it downgrades
        an enumerated attack class, the coverage object records the class as
        ``not_assessed`` with this reason, so the count moves and the sum still checks.
        """
        self._unavailable.append(exc.as_field())
        if class_id:
            self._not_assessed[class_id] = f"{exc.reason}: {exc.detail}" if exc.detail else exc.reason
        return self

    def declare_assumption(self, assumption: str) -> ReportBuilder:
        if assumption not in self._assumptions:
            self._assumptions.append(assumption)
        return self

    # -- emission ----------------------------------------------------------

    def build(self) -> AssuranceReport:
        missing = [
            name
            for name, value in (
                ("assessment", self._assessment),
                ("battery", self._battery),
                ("null_calibration", self._null),
                ("artefact", self._artefact),
                ("precision_ladder", self._ladder),
                ("disposition", self._disposition),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                f"cannot emit a report without {', '.join(missing)}. These blocks carry "
                f"the access assumptions, the null the p-values are ranked against, and "
                f"the disposition -- a report missing any of them is not checkable."
            )

        assert self._battery and self._null and self._artefact and self._ladder
        assert self._assessment and self._disposition

        coverage = generate_coverage(self.taxonomy, not_assessed=self._not_assessed)

        debt = AssuranceDebt(items=self._debt, by_owner_class=summarise_debt(self._debt))

        verdict = render_verdict(
            battery_digest=self._battery.battery_a_digest,
            rungs=self._ladder.rungs_assessed,
            null_family=self._null.null_family,
            null_corpus=self._null.null_corpus,
            converter=(
                self._ladder.converter_provenance.certified_rung
                if self._ladder.converter_provenance
                else "unspecified"
            ),
        )

        note = None
        if self.document_status is DocumentStatus.ILLUSTRATIVE_EXAMPLE:
            note = (
                "Schema-conformant instance with synthetic values, published to show field "
                "structure and the arithmetic each field must satisfy. NOT a measurement "
                "record. A screenshot of this out of context is indistinguishable from a "
                "results table, which is why the status is a required field."
            )

        return AssuranceReport(
            report_id=self.report_id,
            document_status=self.document_status,
            document_status_note=note,
            generated_at_utc=datetime.now(UTC),
            assessment=self._assessment,
            battery=self._battery,
            null_calibration=self._null,
            artefact=self._artefact,
            precision_ladder=self._ladder,
            findings=self._findings,
            supplier_concentration_measurement=self._supplier,
            drift_assessment=self._drift,
            coverage=coverage,
            disposition=self._disposition,
            assurance_debt=debt,
            external_anchor=self._anchor,
            verdict_statement=verdict,
            declared_assumptions=self._assumptions,
            assessment_unavailable=self._unavailable,
        )

    def emit(self, path: str | Path, *, validate: bool = True) -> AssuranceReport:
        """Build, write, and run the independent validator over what was written.

        ``validate=False`` exists only for the mutation test, which needs to write a
        deliberately broken report to confirm the validator catches it.
        """
        report = self.build()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(report.to_json(), encoding="utf-8")

        if validate:
            result = V.validate_file(p)
            if not result.ok:
                detail = "\n  ".join(str(v) for v in result.violations)
                raise ValueError(
                    f"The emitted report failed its own validator. This means the schema "
                    f"and the validator disagree, which is a defect in PRAMANA rather "
                    f"than in the report:\n  {detail}"
                )
        return report


def load_report(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
