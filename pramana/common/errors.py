"""Refusals and failures.

A refusal is a first-class output in this system, not an exception path. When PRAMANA
cannot make a statement it says which statement it cannot make and why, and that
answer is machine-readable. These exceptions carry the exact `assessment_unavailable`
reason codes the schema permits, so a refusal raised deep in a detector reaches the
report without being reworded on the way up.
"""

from __future__ import annotations


class PramanaError(Exception):
    """Base for everything raised by PRAMANA."""


class AssessmentUnavailable(PramanaError):
    """The assessment cannot be made, and the reason is part of the output.

    This is NOT a bug. It is the mechanism behind "we would rather ship a system that
    declines to score than one that scores everything with an unvalidated constant".
    """

    #: Closed vocabulary. The schema rejects any reason not in this set, because a
    #: free-text refusal is a refusal nobody can aggregate or trend.
    VALID_REASONS = frozenset(
        {
            "no_fitted_null",
            "surrogate_not_faithful",
            "backbone_corpus_contaminated",
            "no_weight_model",
            "converter_uncertified",
            "entity_map_incomplete",
            "insufficient_access_tier",
            "no_data_locus",
            "certificate_scope_degenerate",
            "battery_not_precommitted",
            "rung_unavailable",
        }
    )

    def __init__(self, reason: str, detail: str = "", *, asset: str | None = None):
        if reason not in self.VALID_REASONS:
            raise ValueError(
                f"{reason!r} is not a declared refusal reason. "
                f"Add it to AssessmentUnavailable.VALID_REASONS and to the schema, "
                f"or use an existing one. Free-text refusals are not permitted."
            )
        self.reason = reason
        self.detail = detail
        self.asset = asset
        super().__init__(f"assessment_unavailable: {reason}" + (f" -- {detail}" if detail else ""))

    def as_field(self) -> dict[str, str | None]:
        """Render as the report field a caller should emit in place of a number."""
        return {
            "assessment_unavailable": self.reason,
            "assessment_unavailable_detail": self.detail or None,
            "affected_asset": self.asset,
        }


class ValidationFailure(PramanaError):
    """A report violated an arithmetic or paired-companion rule.

    Raised by the report validator. Every instance names the rule it broke, because a
    validator that says "invalid" teaches nobody anything.
    """

    def __init__(self, rule: str, message: str):
        self.rule = rule
        self.message = message
        super().__init__(f"[{rule}] {message}")


class CoverageIncomplete(PramanaError):
    """An enumerated attack class reached the report with no disposition.

    This FAILS THE BUILD by design (G5). If we forgot an attack class, we want to find
    out from CI rather than from a judge.
    """


class LedgerTampered(PramanaError):
    """Hash-chain verification failed at a named row."""

    def __init__(self, index: int, expected: str, actual: str):
        self.index = index
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"ledger chain broken at row {index}: expected prev_hash {expected}, got {actual}"
        )


class SignatureInvalid(PramanaError):
    """An Ed25519 signature did not verify."""


class OfflineViolation(PramanaError):
    """Something tried to reach the network.

    Clause 2.2.6 mandates air-gapped operation, so egress is a compliance failure
    rather than an inconvenience, and CI runs with networking disabled to prove it.
    """
