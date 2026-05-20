from __future__ import annotations

from dataclasses import dataclass, field

from orion.knowledge_forge.models import ClaimV1, DecisionV1, SpecV1, TypedRelationsV1
from orion.knowledge_forge.store import KnowledgeStore, DocModel

RELATION_FIELDS = tuple(TypedRelationsV1.model_fields.keys())


@dataclass
class LintIssue:
    code: str
    doc_id: str
    message: str


@dataclass
class LintReport:
    issues: list[LintIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues


def lint_corpus(store: KnowledgeStore) -> LintReport:
    report = LintReport()
    for doc_id, doc in store.by_id.items():
        _lint_relations(store, doc_id, doc, report)
        if isinstance(doc, ClaimV1):
            if not doc.source_refs:
                report.issues.append(
                    LintIssue("missing_source_ref", doc_id, "claims must cite at least one source_ref")
                )
            for source_id in doc.source_refs:
                if source_id not in store.by_id:
                    report.issues.append(
                        LintIssue("dangling_ref", doc_id, f"source_refs missing {source_id}")
                    )
        if isinstance(doc, SpecV1):
            for claim_id in doc.source_claims:
                if claim_id not in store.by_id:
                    report.issues.append(
                        LintIssue("dangling_ref", doc_id, f"source_claims missing {claim_id}")
                    )
    return report


def _lint_relations(store: KnowledgeStore, doc_id: str, doc: DocModel, report: LintReport) -> None:
    if not isinstance(doc, (ClaimV1, DecisionV1, SpecV1)):
        return
    for field_name in RELATION_FIELDS:
        targets = getattr(doc, field_name)
        for target_id in targets:
            if target_id not in store.by_id:
                report.issues.append(
                    LintIssue(
                        "dangling_ref",
                        doc_id,
                        f"{field_name} references missing id {target_id}",
                    )
                )
