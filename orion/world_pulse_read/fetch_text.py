"""Retain the text a reading turn actually fetched, content-addressed, for evidence checks.

#2497 "Evidence": a relationship claim from a reading may cite a quote only from text
Hub retained for that read. The harness copies each usable fetch tool_result onto
``SourceFetchEvidenceV1.content_text`` (orion/harness/reading_receipts.py; never model
prose). Stage 1 hands that evidence here before persisting the handoff: the text goes
into the existing content-addressed ``reading_document_snapshot`` table (the one Hub
already uses for document sources), the evidence keeps only ``content_sha256``, and
``content_text`` is dropped. So the reading record (``handoff_json.read_evidence``) links
to the exact retained text by hash, and no stored handoff or Stage 2 prompt carries a page.

What the text IS depends on the tool, and is never upgraded: a document snapshot is the
source text Hub read itself; a web fetch is the fetch tool's digest of the page
(``tool_digest``), not the verbatim page. Reads stored before this module have no
retained text; nothing here backfills or fabricates one.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Iterable, Literal

from orion.schemas.reading import SourceFetchEvidenceV1

from .documents import SNAPSHOT_TOOL

logger = logging.getLogger(__name__)

FetchRepresentationV1 = Literal["source_text", "tool_digest"]

_STORE_SQL = """INSERT INTO reading_document_snapshot (sha256, content, content_chars, first_source)
VALUES ($1, $2, $3, $4) ON CONFLICT (sha256) DO NOTHING"""
_LOAD_SQL = "SELECT sha256, content FROM reading_document_snapshot WHERE sha256 = ANY($1::text[])"


def fetch_representation(evidence: SourceFetchEvidenceV1) -> FetchRepresentationV1:
    """Only Hub's own document snapshot is source text; every tool fetch is a digest."""
    return "source_text" if evidence.tool_name == SNAPSHOT_TOOL else "tool_digest"


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def retain_fetch_texts(
    conn: Any, evidence: Iterable[SourceFetchEvidenceV1]
) -> list[SourceFetchEvidenceV1]:
    """Store each carried text and return the evidence with ``content_sha256`` set and
    ``content_text`` dropped. Hub computes the hash itself. A store failure keeps the
    evidence (the read still happened) without a hash: it simply cannot back a quote."""
    out: list[SourceFetchEvidenceV1] = []
    for item in evidence:
        text = item.content_text
        if not text:
            out.append(item.model_copy(update={"content_text": None}))
            continue
        sha = text_sha256(text)
        try:
            if conn is None:
                raise RuntimeError("no_connection")
            await conn.execute(_STORE_SQL, sha, text, len(text), item.url)
        except Exception as exc:  # noqa: BLE001 - evidence capture must not fail the read
            logger.warning("reading_fetch_text_not_retained url=%s err=%s", item.url, type(exc).__name__)
            out.append(item.model_copy(update={"content_text": None}))
            continue
        out.append(item.model_copy(update={"content_text": None, "content_sha256": sha}))
    return out


@dataclass(frozen=True)
class RetainedTextV1:
    sha256: str
    url: str
    representation: FetchRepresentationV1
    text: str


async def load_retained_texts(
    conn: Any, evidence: Iterable[SourceFetchEvidenceV1]
) -> list[RetainedTextV1]:
    """The retained texts behind a handoff's read evidence, re-verified against their hash.
    A row whose content no longer hashes to its key is not evidence and is dropped."""
    wanted = [e for e in evidence if e.content_sha256]
    if not wanted or conn is None:
        return []
    rows = await conn.fetch(_LOAD_SQL, sorted({e.content_sha256 for e in wanted}))
    by_sha = {str(r["sha256"]): str(r["content"]) for r in rows}
    out: list[RetainedTextV1] = []
    seen: set[str] = set()
    for e in wanted:
        text = by_sha.get(e.content_sha256 or "")
        if text is None or e.content_sha256 in seen:
            continue
        if text_sha256(text) != e.content_sha256 and fetch_representation(e) != "source_text":
            # A document snapshot's key is the sha256 of its raw file bytes (a BOM is
            # stripped from the stored text), so only a fetch digest can be re-hashed.
            logger.warning("reading_fetch_text_hash_mismatch sha=%s", e.content_sha256)
            continue
        seen.add(e.content_sha256 or "")
        out.append(RetainedTextV1(sha256=e.content_sha256 or "", url=e.url,
                                  representation=fetch_representation(e), text=text))
    return out
