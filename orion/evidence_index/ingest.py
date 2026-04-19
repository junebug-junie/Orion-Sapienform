from __future__ import annotations

from typing import Any

from orion.evidence_index.adapters.base import EvidenceAdapter
from orion.evidence_index.adapters.collapse_mirror import CollapseMirrorEvidenceAdapter
from orion.evidence_index.adapters.journal import JournalEvidenceAdapter
from orion.evidence_index.adapters.markdown_spec import MarkdownSpecEvidenceAdapter
from orion.evidence_index.adapters.notify_output import NotifyOutputEvidenceAdapter
from orion.evidence_index.adapters.parsed_document import ParsedDocumentEvidenceAdapter
from orion.schemas.evidence_index import EvidenceUnitV1

_ADAPTERS_BY_KIND: dict[str, EvidenceAdapter] = {
    "journal.entry.write.v1": JournalEvidenceAdapter(),
    "collapse.mirror": CollapseMirrorEvidenceAdapter(),
    "collapse.mirror.entry.v2": CollapseMirrorEvidenceAdapter(),
    "document.markdown.spec.v1": MarkdownSpecEvidenceAdapter(),
    "document.parsed.v1": ParsedDocumentEvidenceAdapter(),
    "notify.notification.request.v1": NotifyOutputEvidenceAdapter(),
    "notify.notification.receipt.v1": NotifyOutputEvidenceAdapter(),
}


def build_evidence_units(kind: str, payload: Any, *, correlation_id: str | None = None) -> list[EvidenceUnitV1]:
    adapter = _ADAPTERS_BY_KIND.get(kind)
    if adapter is None:
        return []
    return adapter.to_units(payload, kind=kind, correlation_id=correlation_id)
