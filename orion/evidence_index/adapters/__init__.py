from .base import EvidenceAdapter
from .collapse_mirror import CollapseMirrorEvidenceAdapter
from .journal import JournalEvidenceAdapter
from .markdown_spec import MarkdownSpecEvidenceAdapter
from .notify_output import NotifyOutputEvidenceAdapter
from .parsed_document import ParsedDocumentEvidenceAdapter

__all__ = [
    "EvidenceAdapter",
    "JournalEvidenceAdapter",
    "CollapseMirrorEvidenceAdapter",
    "MarkdownSpecEvidenceAdapter",
    "NotifyOutputEvidenceAdapter",
    "ParsedDocumentEvidenceAdapter",
]
