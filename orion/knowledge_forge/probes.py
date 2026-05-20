from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from orion.knowledge_forge.models import ClaimV1
from orion.knowledge_forge.store import KnowledgeStore


@dataclass
class ProbeIssue:
    code: str
    message: str


@dataclass
class ProbeReport:
    issues: list[ProbeIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues


def probe_source_coverage(
    store: KnowledgeStore,
    *,
    source_path: Path,
    source_id: str,
    min_keyword: str,
) -> ProbeReport:
    text = source_path.read_text(encoding="utf-8").lower()
    report = ProbeReport()
    if min_keyword.lower() not in text:
        return report

    claim_text = " ".join(
        c.statement.lower()
        for c in store.claims()
        if isinstance(c, ClaimV1) and source_id in c.source_refs
    )
    if min_keyword.lower() not in claim_text:
        report.issues.append(
            ProbeIssue(
                "uncited_keyword",
                f"source mentions {min_keyword!r} but no claim citing {source_id} covers it",
            )
        )
    return report
