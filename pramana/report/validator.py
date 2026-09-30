"""The report validator -- a build gate, not a linter.

The single rule that generated this file:

    A number is not checkable unless the report also prints the thing it must be
    checked against.

Six comparator pairs are mandated. Each is checked here, each names itself when it
fails, and the mutation test in ``tests/test_validator_mutations.py`` plants a
violation of each in turn and asserts the validator fired *for the stated reason* --
because a mechanised check nobody has tried to break is still only a claim.

The JSON extractor is deliberately **tag-independent**: it parses every fenced block
whose body begins ``{`` or ``[``, regardless of the language tag. The block that
ships untagged is exactly the one a tag-keyed check skips.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pramana.common import constants as K
from pramana.common.errors import ValidationFailure

_FENCE = re.compile(r"^[ \t]*(?:```|~~~)[^\n]*\n(.*?)^[ \t]*(?:```|~~~)", re.DOTALL | re.MULTILINE)


@dataclass
class Violation:
    rule: str
    path: str
    message: str

    def __str__(self) -> str:
        return f"[{self.rule}] {self.path}: {self.message}"


@dataclass
class ValidationResult:
    violations: list[Violation] = field(default_factory=list)
    blocks_checked: int = 0
    pairs_checked: int = 0

    @property
    def ok(self) -> bool:
        return not self.violations

    def add(self, rule: str, path: str, message: str) -> None:
        self.violations.append(Violation(rule, path, message))


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_json_blocks(text: str) -> list[Any]:
    """Return every fenced block whose body parses as JSON.

    Tag-independent by design. A block tagged ``json``, ``jsonc``, ``javascript`` or
    nothing at all is treated identically, because the failure mode this guards
    against is a real report shipping with a missing or wrong tag.
    """
    out: list[Any] = []
    for match in _FENCE.finditer(text):
        body = match.group(1).strip()
        if not body or body[0] not in "{[":
            continue
        try:
            out.append(json.loads(body))
        except json.JSONDecodeError:
            continue
    return out


def _walk(node: Any, path: str = "$"):
    """Yield every (path, dict) in a nested structure."""
    if isinstance(node, dict):
        yield path, node
        for key, value in node.items():
            yield from _walk(value, f"{path}.{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from _walk(value, f"{path}[{i}]")


def _close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


# ---------------------------------------------------------------------------
# The six comparator pairs
# ---------------------------------------------------------------------------


def _pair_1_detection_p_beside_floor(path: str, d: dict, r: ValidationResult) -> bool:
    """PAIR 1 -- every detection p-value is printed beside the floor that bounds it."""
    if "detection_p" not in d:
        return False
    r.pairs_checked += 1
    p = d["detection_p"]

    floor = d.get("detection_p_floor")
    if floor is None:
        level = d.get("level", "L1_detection")
        floor = (
            K.P_FLOOR_L1_DETECTION if level == "L1_detection" else K.P_FLOOR_L2_LOCALISATION
        )
        if "detection_p_basis" not in d and "detection_p_reported_as" not in d:
            r.add(
                "PAIR_1_detection_p_floor",
                path,
                "detection_p printed with no floor and no basis beside it. A p-value "
                "without its resolution limit is not evidence.",
            )
            return True

    if p < floor - 1e-12:
        r.add(
            "PAIR_1_detection_p_floor",
            path,
            f"detection_p={p} is below the floor {floor:.6f}. The null cannot resolve it.",
        )
    if "detection_p_is_floor" in d:
        claimed = bool(d["detection_p_is_floor"])
        actual = _close(p, floor)
        if claimed != actual:
            r.add(
                "PAIR_1_detection_p_floor",
                path,
                f"detection_p_is_floor={claimed} contradicts p={p} vs floor={floor:.6f}.",
            )
    return True


def _pair_2_localisation_p_beside_bh(path: str, d: dict, r: ValidationResult) -> bool:
    """PAIR 2 -- a localisation p-value is printed beside its BH critical value."""
    name = d.get("name", "")
    is_localisation = "pooled" in str(name) or d.get("bh_classes_tested") is not None
    if not is_localisation or "value" not in d:
        return False
    r.pairs_checked += 1

    crit = d.get("bh_critical_value_at_rank_1")
    if crit is None:
        r.add(
            "PAIR_2_localisation_bh",
            path,
            "A localisation statistic was printed without bh_critical_value_at_rank_1. "
            "Running 43 class tests without the correction inflates the error rate "
            "invisibly.",
        )
        return True

    n_classes = d.get("bh_classes_tested")
    fdr = d.get("fdr", K.DECLARED_FDR)
    if n_classes:
        expected = fdr / n_classes
        if not _close(crit, expected, 1e-6):
            r.add(
                "PAIR_2_localisation_bh",
                path,
                f"bh_critical_value_at_rank_1={crit} != alpha/m = {fdr}/{n_classes} = {expected:.6f}.",
            )
    if "bh_rejects" in d:
        claimed = bool(d["bh_rejects"])
        actual = d["value"] <= crit + 1e-12
        if claimed != actual:
            r.add(
                "PAIR_2_localisation_bh",
                path,
                f"bh_rejects={claimed} but value={d['value']} vs critical={crit}.",
            )
    floor = d.get("p_floor")
    if floor is not None and d["value"] < floor - 1e-12:
        r.add(
            "PAIR_2_localisation_bh",
            path,
            f"value={d['value']} sits below its own pooled floor {floor}.",
        )
    return True


def _pair_3_coverage_counts_beside_sum(path: str, d: dict, r: ValidationResult) -> bool:
    """PAIR 3 -- coverage counts are printed beside their sum_check."""
    if "assessed" not in d or "declared_unsupported" not in d:
        return False
    r.pairs_checked += 1

    if "sum_check" not in d:
        r.add(
            "PAIR_3_coverage_sum",
            path,
            "coverage counts printed with no sum_check. A class that is neither "
            "assessed nor declared can then go unenumerated.",
        )
        return True

    total = d["assessed"] + d["declared_unsupported"] + d.get("not_assessed", 0)
    if total != d["sum_check"]:
        r.add(
            "PAIR_3_coverage_sum",
            path,
            f"counts sum to {total}, sum_check says {d['sum_check']}.",
        )
    if "total_classes" in d and d["sum_check"] != d["total_classes"]:
        r.add(
            "PAIR_3_coverage_sum",
            path,
            f"sum_check {d['sum_check']} != total_classes {d['total_classes']}.",
        )
    return True


def _pair_4_evalue_beside_threshold(path: str, d: dict, r: ValidationResult) -> bool:
    """PAIR 4 -- a merged e-value is printed beside its e-BH rejection threshold."""
    if d.get("name") != "e_value_merged":
        return False
    r.pairs_checked += 1

    thr = d.get("ebh_threshold_at_k1")
    if thr is None:
        r.add(
            "PAIR_4_evalue_threshold",
            path,
            "e_value_merged printed with no ebh_threshold_at_k1. An e-value without "
            "its threshold is a ranking wearing the costume of a verdict.",
        )
        return True

    m = d.get("sources_tested")
    fdr = d.get("fdr", K.DECLARED_FDR)
    if m:
        expected = m / (fdr * 1)
        if not _close(thr, expected, 1e-6):
            r.add(
                "PAIR_4_evalue_threshold",
                path,
                f"ebh_threshold_at_k1={thr} != m/(alpha*k) = {m}/({fdr}*1) = {expected:.4f}.",
            )
    merge = d.get("merging_function")
    if merge is None:
        r.add(
            "PAIR_4_evalue_threshold",
            path,
            "merging_function is absent. Under arbitrary dependence the product is "
            "invalid and the error is invisible in the output, so the operator is a field.",
        )
    elif "product" in str(merge) or "geometric" in str(merge):
        r.add(
            "PAIR_4_evalue_threshold",
            path,
            f"merging_function={merge!r}. Four detectors on one shard are not "
            f"independent; the valid merge is the arithmetic mean (Vovk & Wang).",
        )
    return True


def _pair_5_transfer_delta_beside_ceiling(path: str, d: dict, r: ValidationResult) -> bool:
    """PAIR 5 -- a null-transfer delta is printed beside the ceiling it must respect."""
    if "median_shift_over_iqr" not in d:
        return False
    r.pairs_checked += 1

    if "quantile_resolution" not in d and "alpha_tail_shift_over_iqr" in d:
        r.add(
            "PAIR_5_transfer_ceiling",
            path,
            "An alpha-tail shift was quoted with no quantile_resolution label. At n=32 "
            "the alpha-tail is the ~1.6th order statistic and cannot carry a claim.",
        )
    if "bootstrap_ci_median_shift" not in d:
        r.add(
            "PAIR_5_transfer_ceiling",
            path,
            "median_shift_over_iqr printed with no bootstrap interval beside it.",
        )
    return True


def _pair_6_k_beside_volume_share(path: str, d: dict, r: ValidationResult) -> bool:
    """PAIR 6 -- certified k is printed beside the volume share those k lots hold.

    Also enforces the paired-companion rule: k never travels without
    certified_fraction_at_k and surrogate_gap_top1.
    """
    if "certified_floor_k" not in d or d.get("certified_floor_k") is None:
        return False
    r.pairs_checked += 1

    for companion in ("certified_fraction_at_k", "surrogate_gap_top1"):
        if d.get(companion) is None:
            r.add(
                "PAIR_6_k_volume_share",
                path,
                f"certified_floor_k emitted without {companion}. The field cannot be "
                f"quoted alone, by construction.",
            )

    vi = d.get("volume_inequality")
    if not isinstance(vi, dict):
        r.add(
            "PAIR_6_k_volume_share",
            path,
            "certified_floor_k emitted without volume_inequality. A count of "
            "contributors silently overstates the guarantee when lots are unequal.",
        )
        return True

    if "volume_share_of_largest_k" not in vi:
        r.add(
            "PAIR_6_k_volume_share",
            path,
            "volume_inequality carries no volume_share_of_largest_k beside k.",
        )
    shares = vi.get("lot_volume_shares")
    if isinstance(shares, dict) and shares:
        total = sum(shares.values())
        if not _close(total, 1.0, 1e-6):
            r.add("PAIR_6_k_volume_share", path, f"lot volume shares sum to {total}, not 1.0.")
        largest = max(shares.values())
        if largest > K.SINGLE_LOT_MAJORITY_THRESHOLD and d.get("certified_floor_k") is not None:
            r.add(
                "PAIR_6_k_volume_share",
                path,
                f"largest lot share {largest:.3f} exceeds "
                f"{K.SINGLE_LOT_MAJORITY_THRESHOLD}, so the certificate must be WITHHELD "
                f"(certificate_scope_degenerate: single_lot_majority), not caveated.",
            )
    if d.get("scope") != "surrogate_ensemble":
        r.add(
            "PAIR_6_k_volume_share",
            path,
            "supplier_concentration_measurement.scope must be 'surrogate_ensemble'. "
            "The floor is a property of the ensemble built for the measurement and "
            "never of the artefact that ships.",
        )
    return True


# ---------------------------------------------------------------------------
# Structural rules that are not comparator pairs
# ---------------------------------------------------------------------------


def _rule_point_attribution(path: str, d: dict, r: ValidationResult) -> None:
    if d.get("attribution_mode") != "point":
        return
    size = d.get("attribution_set_size")
    strength = d.get("evidence_strength")
    if size != 1 or strength != "causal_verified":
        r.add(
            "RULE_point_attribution",
            path,
            f"attribution_mode=point requires attribution_set_size==1 "
            f"(got {size}) AND evidence_strength==causal_verified (got {strength!r}). "
            f"Either alone is rejected.",
        )


def _rule_surrogate_gate(path: str, d: dict, r: ValidationResult) -> None:
    if "surrogate_exact_agreement" not in d:
        return
    agree = d["surrogate_exact_agreement"]
    floor = d.get("surrogate_exact_agreement_floor")
    if floor is None:
        r.add(
            "RULE_surrogate_gate",
            path,
            "surrogate_exact_agreement printed without its pre-committed floor.",
        )
        return
    if agree < floor:
        r.add(
            "RULE_surrogate_gate",
            path,
            f"agreement {agree} is below floor {floor}: reversal must not have run. "
            f"Expected assessment_unavailable: surrogate_not_faithful.",
        )
    if d.get("evidence_ran_on") not in (None, "delivered_int8_binary"):
        r.add(
            "RULE_surrogate_gate",
            path,
            f"evidence_ran_on={d.get('evidence_ran_on')!r}. Search runs on the "
            f"surrogate; the evidence is measured on the delivered binary. Search is "
            f"not evidence.",
        )


def _rule_predicted_labels(path: str, d: dict, r: ValidationResult) -> None:
    if "concentration_operating_point" not in d:
        return
    status = d.get("concentration_operating_point_status")
    if status not in ("predicted", "measured"):
        r.add(
            "RULE_predicted_label",
            path,
            "concentration_operating_point printed with no status. An unlabelled "
            "prediction discredits every other number in the report.",
        )


def _rule_document_status(obj: Any, r: ValidationResult) -> None:
    if not isinstance(obj, dict):
        return
    if "report_id" in obj and "document_status" not in obj:
        r.add(
            "RULE_document_status",
            "$",
            "A report without document_status. A schema-conformant instance full of "
            "plausible numbers, screenshotted out of context, is indistinguishable "
            "from a results table.",
        )


def _rule_forbidden_phrases(text: str, r: ValidationResult) -> None:
    low = text.lower()
    for phrase in K.FORBIDDEN_VERDICT_PHRASES:
        if phrase in low:
            r.add(
                "MUST8_forbidden_phrase",
                "$",
                f"forbidden phrase {phrase!r}. Absence of evidence is the strongest "
                f"true statement available (arXiv:2204.06974).",
            )


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

_PAIR_CHECKS = (
    _pair_1_detection_p_beside_floor,
    _pair_2_localisation_p_beside_bh,
    _pair_3_coverage_counts_beside_sum,
    _pair_4_evalue_beside_threshold,
    _pair_5_transfer_delta_beside_ceiling,
    _pair_6_k_beside_volume_share,
)

_RULES = (_rule_point_attribution, _rule_surrogate_gate, _rule_predicted_labels)

#: Named so the CLI can print "6 comparator pairs checked" rather than a vague pass.
PAIR_NAMES = (
    "PAIR_1_detection_p_floor",
    "PAIR_2_localisation_bh",
    "PAIR_3_coverage_sum",
    "PAIR_4_evalue_threshold",
    "PAIR_5_transfer_ceiling",
    "PAIR_6_k_volume_share",
)


def validate_object(obj: Any, result: ValidationResult | None = None) -> ValidationResult:
    """Validate one parsed JSON object against all six pairs and the structural rules."""
    r = result or ValidationResult()
    _rule_document_status(obj, r)
    for path, node in _walk(obj):
        for check in _PAIR_CHECKS:
            check(path, node, r)
        for rule in _RULES:
            rule(path, node, r)
    return r


def validate_text(text: str) -> ValidationResult:
    """Validate every JSON block embedded in a markdown or text document.

    This is how the gate runs over the design document itself: the populated report
    example in the technical doc is parsed and checked exactly as a live report is.
    """
    r = ValidationResult()
    _rule_forbidden_phrases(text, r)
    for obj in extract_json_blocks(text):
        r.blocks_checked += 1
        validate_object(obj, r)
    return r


def validate_file(path: str | Path) -> ValidationResult:
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    if p.suffix.lower() == ".json":
        r = ValidationResult()
        _rule_forbidden_phrases(text, r)
        obj = json.loads(text)
        r.blocks_checked = 1
        return validate_object(obj, r)
    return validate_text(text)


def assert_valid(path: str | Path) -> None:
    """Raise :class:`ValidationFailure` on the first violation. Used by CI."""
    r = validate_file(path)
    if not r.ok:
        first = r.violations[0]
        raise ValidationFailure(
            first.rule,
            f"{len(r.violations)} violation(s) in {path}; first: {first}",
        )
