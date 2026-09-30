"""G5 -- the coverage object, generated from the taxonomy rather than written.

Coverage statements exist everywhere. What is unusual here is that a build which
*fails* because an enumerated attack class has no disposition is a CI discipline
rather than a documentation practice.

Two properties make this worth the code:

* **Absence is an output.** Every class in ``taxonomy.yaml`` reaches the report with a
  disposition and, where it is not ``assessed``, a reason. "What about clean-label
  attacks?" becomes a field lookup rather than a stumble at the podium.
* **The count is recomputed, never copied.** ``sum_check`` is derived from the items
  on every build, so a stale hand-propagated figure cannot survive a commit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from pramana.common.digest import sha256_file
from pramana.common.errors import CoverageIncomplete
from pramana.report.schema import Coverage, CoverageDisposition, CoverageItem

DEFAULT_TAXONOMY = Path("taxonomy.yaml")

_REQUIRE_REASON = {
    CoverageDisposition.DECLARED_UNSUPPORTED,
    CoverageDisposition.NOT_ASSESSED,
}


class Taxonomy:
    """The enumerated attack-class list, loaded once and treated as authoritative."""

    def __init__(self, path: str | Path = DEFAULT_TAXONOMY):
        self.path = Path(path)
        if not self.path.exists():
            raise CoverageIncomplete(
                f"{self.path} is missing. The coverage object is generated from it, so "
                f"its absence is a build failure rather than a default."
            )
        raw: dict[str, Any] = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        self.generation: int = int(raw["generation"])
        self.classes: list[dict[str, Any]] = list(raw["classes"])
        self.digest: str = sha256_file(self.path)
        self._validate()

    def _validate(self) -> None:
        seen: set[str] = set()
        for entry in self.classes:
            cid = entry.get("id")
            if not cid:
                raise CoverageIncomplete(f"A taxonomy entry has no id: {entry!r}")
            if cid in seen:
                raise CoverageIncomplete(f"Duplicate taxonomy class id {cid!r}.")
            seen.add(cid)
            disp = CoverageDisposition(entry["disposition"])
            if disp in _REQUIRE_REASON and not entry.get("reason"):
                raise CoverageIncomplete(
                    f"Class {cid!r} is {disp.value} with no reason. A declared gap "
                    f"without a reason is a gap we have not actually declared."
                )

    @property
    def ids(self) -> set[str]:
        return {c["id"] for c in self.classes}


def generate_coverage(
    taxonomy: Taxonomy | str | Path = DEFAULT_TAXONOMY,
    *,
    not_assessed: dict[str, str] | None = None,
) -> Coverage:
    """Build the :class:`Coverage` block for one assessment.

    ``not_assessed`` maps a class id to the reason it could not be assessed *in this
    run* -- typically an access tier that does not reach it. It downgrades a class
    that the taxonomy marks ``assessed``; it can never upgrade one.

    Raises :class:`CoverageIncomplete` if a class would reach the report with no
    disposition, which is the whole point of the mechanism.
    """
    tax = taxonomy if isinstance(taxonomy, Taxonomy) else Taxonomy(taxonomy)
    downgrades = dict(not_assessed or {})

    unknown = set(downgrades) - tax.ids
    if unknown:
        raise CoverageIncomplete(
            f"not_assessed names classes absent from the taxonomy: {sorted(unknown)}. "
            f"Either the class was removed and the caller was not updated, or the id is "
            f"misspelled. Both are build failures."
        )

    items: list[CoverageItem] = []
    counts = {d: 0 for d in CoverageDisposition}

    for entry in tax.classes:
        cid = entry["id"]
        declared = CoverageDisposition(entry["disposition"])
        reason = entry.get("reason")

        if cid in downgrades and declared is CoverageDisposition.ASSESSED:
            declared = CoverageDisposition.NOT_ASSESSED
            reason = downgrades[cid]

        if declared in _REQUIRE_REASON and not reason:
            raise CoverageIncomplete(
                f"Class {cid!r} reached the report as {declared.value} with no reason."
            )

        items.append(
            CoverageItem(
                class_id=cid,
                disposition=declared,
                reason=_flatten(reason),
                mechanisms=list(entry.get("mechanisms", [])),
                attribution_ceiling=entry.get("attribution_ceiling"),
            )
        )
        counts[declared] += 1

    total = len(items)
    return Coverage(
        taxonomy_digest=tax.digest,
        taxonomy_generation=tax.generation,
        items=items,
        assessed=counts[CoverageDisposition.ASSESSED],
        declared_unsupported=counts[CoverageDisposition.DECLARED_UNSUPPORTED],
        not_assessed=counts[CoverageDisposition.NOT_ASSESSED],
        total_classes=total,
        sum_check=total,
    )


def _flatten(text: str | None) -> str | None:
    """YAML folded scalars arrive with newlines; reports want one line."""
    if text is None:
        return None
    return " ".join(text.split())


def check_handlers(tax: Taxonomy, implemented: set[str]) -> list[str]:
    """Return classes marked ``assessed`` for which no mechanism is registered.

    This is the other half of the gate. The taxonomy can claim a class is assessed;
    this asks the code whether anything actually assesses it, so a class cannot be
    quietly upgraded in YAML without a handler existing.
    """
    missing: list[str] = []
    for entry in tax.classes:
        if CoverageDisposition(entry["disposition"]) is not CoverageDisposition.ASSESSED:
            continue
        mechs = set(entry.get("mechanisms", []))
        if not mechs:
            missing.append(entry["id"])
        elif not (mechs & implemented):
            missing.append(entry["id"])
    return missing
