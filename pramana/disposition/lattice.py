"""G2 -- the disposition lattice, and why CONDITIONAL_RELEASE is the adoption argument.

Every assurance gate in history that could only say NO has been waived under
operational pressure, once, and then ignored forever. So the override is a first-class,
signed, expiring, monitored ledger entry rather than a conversation that happens around
the system.

This turns the hardest hostile question -- *"what if you're wrong and we need the model
tomorrow?"* -- into a feature demonstration: the model ships, under named compensating
controls, with a named risk-acceptance authority, an expiry, and IOC monitoring on the
live receipt stream. **The override is the audit trail.**

The second mechanism here is the evidence ladder. ``evidence_strength`` gates what a
finding is permitted to *say*: a statistically flagged lot may be named as a
containment scope, but no finding may accuse a supplier of cause without a control.
That is a report-level authority rule tying statistical strength to permitted
attribution language, and it is enforced here rather than left to whoever writes the
summary.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from pramana.report.schema import (
    Disposition,
    EvidenceStrength,
    Finding,
    RiskAcceptance,
    DispositionRecord,
)

__all__ = ["decide", "conditional_release", "permitted_language", "REASSESSMENT_TRIGGERS"]

#: Severity ordering. Used to take the worst disposition across findings, because an
#: assessment's disposition is the most serious one it contains, never an average.
_ORDER = {
    Disposition.ACCEPT: 0,
    Disposition.ACCEPT_WITH_CONDITIONS: 1,
    Disposition.CONDITIONAL_RELEASE: 2,
    Disposition.QUARANTINE: 3,
    Disposition.REJECT: 4,
}

#: The events that revoke or re-open a disposition. These are ledger event types, so a
#: trigger is a signed act rather than a note in a spreadsheet.
REASSESSMENT_TRIGGERS = (
    "converter_version_change",
    "calibration_set_change",
    "lot_admitted",
    "contract_relet",
    "receipt_eprocess_crossing",
    "expiry_reached",
)


def permitted_language(strength: EvidenceStrength) -> dict[str, bool | str]:
    """What a finding at this evidence strength may assert.

    The distinction that matters: a *ranked estimate with a causal check* is strictly
    stronger than a score and strictly weaker than a proof, and the report says which.
    Nothing at any strength may say a supplier "causally" did anything -- that phrase
    is on the forbidden list, because our own attribution is a leave-lot-out control
    with seeds, not a proof.
    """
    return {
        EvidenceStrength.INDICATIVE: {
            "may_name_containment_scope": True,
            "may_name_supplier_as_suspect": False,
            "may_assert_cause": False,
            "language": "Statistically flagged at a declared FDR; causally unverified.",
        },
        EvidenceStrength.CORROBORATED: {
            "may_name_containment_scope": True,
            "may_name_supplier_as_suspect": True,
            "may_assert_cause": False,
            "language": "Multiple independent mechanisms concur; no control has been run.",
        },
        EvidenceStrength.CAUSAL_VERIFIED: {
            "may_name_containment_scope": True,
            "may_name_supplier_as_suspect": True,
            "may_assert_cause": False,
            "language": (
                "Removing this lot moves attack success and a size-matched control lot "
                "does not. A ranked estimate with a causal check."
            ),
        },
        EvidenceStrength.ATTRIBUTION_UNAVAILABLE: {
            "may_name_containment_scope": True,
            "may_name_supplier_as_suspect": False,
            "may_assert_cause": False,
            "language": "Containment scope only; no attribution is available at this tier.",
        },
    }[strength]


def _finding_disposition(f: Finding) -> Disposition:
    """Map one finding to a disposition using its evidence strength and scope."""
    if f.evidence_strength is EvidenceStrength.CAUSAL_VERIFIED:
        return Disposition.REJECT
    if f.evidence_strength is EvidenceStrength.CORROBORATED:
        return Disposition.QUARANTINE
    if f.evidence_strength is EvidenceStrength.INDICATIVE:
        return Disposition.ACCEPT_WITH_CONDITIONS
    return Disposition.ACCEPT_WITH_CONDITIONS


def decide(
    findings: list[Finding],
    *,
    refusals: list[str] | None = None,
    predicted_operating_points: list[str] | None = None,
) -> DispositionRecord:
    """Derive the assessment's disposition from its findings.

    Two rules that are easy to get wrong and are therefore explicit:

    * **The worst finding wins.** Dispositions do not average.
    * **A finding resting only on a `predicted` operating point cannot carry a
      disposition on its own.** The concentration threshold is not measured until E10
      runs, so a QUARANTINE justified solely by it is downgraded and the rationale
      says so. Shipping a disposition on an unmeasured threshold is precisely the
      defect the project's own audit register exists to catch.
    """
    refusals = refusals or []
    predicted = set(predicted_operating_points or [])

    if not findings:
        state = Disposition.ACCEPT
        rationale = (
            "No findings were raised. This is an absence of evidence under the battery "
            "and rungs named in the report, not a statement that the artefact is sound."
        )
        if refusals:
            state = Disposition.ACCEPT_WITH_CONDITIONS
            rationale += f" {len(refusals)} assessment(s) were unavailable: {', '.join(refusals)}."
        return DispositionRecord(state=state, rationale=rationale)

    worst = Disposition.ACCEPT
    reasons: list[str] = []
    for f in findings:
        d = _finding_disposition(f)
        rests_on_predicted = f.mechanism in predicted and not f.corroborating_mechanisms
        if rests_on_predicted and _ORDER[d] > _ORDER[Disposition.ACCEPT_WITH_CONDITIONS]:
            d = Disposition.ACCEPT_WITH_CONDITIONS
            reasons.append(
                f"{f.finding_id} downgraded: it rests only on an operating point tagged "
                f"'predicted' and is not corroborated."
            )
        else:
            reasons.append(f"{f.finding_id} ({f.evidence_strength.value}) -> {d.value}")
        if _ORDER[d] > _ORDER[worst]:
            worst = d

    if refusals:
        reasons.append(f"Unavailable assessments: {', '.join(refusals)}.")

    return DispositionRecord(state=worst, rationale=" ".join(reasons))


def conditional_release(
    base: DispositionRecord,
    *,
    authority_name: str,
    authority_role: str,
    compensating_controls: list[str],
    validity_days: int = 90,
    ledger_entry_digest: str | None = None,
    triggers: list[str] | None = None,
) -> DispositionRecord:
    """Override a failing disposition -- on the record.

    This is the mechanism that makes the system survive contact with a real programme.
    It is deliberately impossible to invoke anonymously: the schema rejects a
    CONDITIONAL_RELEASE with no named risk-acceptance authority.
    """
    if not compensating_controls:
        raise ValueError(
            "CONDITIONAL_RELEASE with no compensating controls is an unconditional "
            "release wearing a different name."
        )
    now = datetime.now(UTC)
    return DispositionRecord(
        state=Disposition.CONDITIONAL_RELEASE,
        rationale=(
            f"Overridden from {base.state.value} by a named authority. Original basis: "
            f"{base.rationale}"
        ),
        risk_acceptance=RiskAcceptance(
            authority_name=authority_name,
            authority_role=authority_role,
            accepted_at_utc=now,
            expires_at_utc=now + timedelta(days=validity_days),
            compensating_controls=compensating_controls,
            reassessment_triggers=list(triggers or REASSESSMENT_TRIGGERS),
            ledger_entry_digest=ledger_entry_digest,
        ),
    )


def revoke(record: DispositionRecord, reason: str) -> DispositionRecord:
    """Revoke a risk acceptance -- what an e-process crossing actually *does*.

    The point of the receipt monitor is not the statistic. It is that a signed
    certificate changes state and somebody's acceptance goes with it.
    """
    if record.risk_acceptance is None:
        return record
    ra = record.risk_acceptance
    return DispositionRecord(
        state=Disposition.QUARANTINE,
        rationale=f"Risk acceptance revoked: {reason}. Prior rationale: {record.rationale}",
        risk_acceptance=RiskAcceptance(
            authority_name=ra.authority_name,
            authority_role=ra.authority_role,
            accepted_at_utc=ra.accepted_at_utc,
            expires_at_utc=ra.expires_at_utc,
            compensating_controls=ra.compensating_controls,
            reassessment_triggers=ra.reassessment_triggers,
            ledger_entry_digest=ra.ledger_entry_digest,
            revoked=True,
            revoked_reason=reason,
        ),
    )
