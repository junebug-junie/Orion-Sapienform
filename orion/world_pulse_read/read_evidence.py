"""Did a Stage 1 turn actually read its source? Decided from the tool trace.

Live 2026-09-25: seed ``finding:60d59b10...:9b084fc0f1583da0`` was claimed,
debited a Wallet A slot and marked ``done`` although its harness run made zero
tool calls (``harness_turn_trace.step_count=3``: init, assistant, result) and
its own ``what_i_learned`` opened with "Metadata-only extraction; I did not
fetch or read the article body this turn". Schema-valid JSON is not a read.

Evidence is ``SourceFetchEvidenceV1`` rows the harness derives from raw FCC
tool_use/tool_result pairs (orion/harness/reading_receipts.py) and Hub puts on
the final frame as ``harness_source_fetches``. Model prose never counts.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional
from urllib.parse import urlsplit

from orion.schemas.reading import SourceFetchEvidenceV1
from orion.world_pulse_read.documents import SNAPSHOT_TOOL, is_document_ref
from orion.world_pulse_read.fetch_provenance import fetch_http_error_status
from orion.world_pulse_read.url_filters import url_looks_like_section_index

# last_error labels. Plain `no_read_evidence` is terminal (the model ran and
# chose not to / could not fetch the source -- a retry spends another slot on
# the same outcome). The `:harness_unreported` form means the governor that ran
# the turn predates `HarnessRunV1.source_fetches`; that is an infrastructure
# condition, listed as transient in orion/world_pulse_read/retry.py.
NO_READ_EVIDENCE = "no_read_evidence"
NO_READ_EVIDENCE_UNREPORTED = "no_read_evidence:harness_unreported"
# The turn did fetch a page on the seed's site, but every such result was under
# MIN_SOURCE_CONTENT_CHARS. Terminal like plain no_read_evidence; a separate
# label only so the floor can be calibrated from real rows (a WebFetch result
# is the fetch tool's answer to a prompt the reader wrote, so a short one may
# be a narrow question rather than an empty page).
NO_READ_EVIDENCE_THIN = "no_read_evidence:thin_fetch"
# Every fetch of the seed's site came back as the fetch tool's own HTTP error
# notice ("The server returned HTTP 404 Not Found." -- fetch_provenance.py).
# Terminal: the page is gone or refused. Live 2026-10-13: three guessed
# CoreWeave URLs were marked `done` on exactly that 215-char notice.
NO_READ_EVIDENCE_HTTP_ERROR = "no_read_evidence:http_error"

# Floor on the tool_result text the model received. A WebFetch result is the
# fetch tool's own multi-sentence digest of the page; refusal/blocked strings
# ("Request failed with status code 403", "Unable to fetch ...") are well under
# this. Checked 2026-10-13 against the 74 retained live WebFetch texts: size
# does NOT separate pages from errors -- real CoreWeave digests ran 139-208
# chars while WebFetch's HTTP error notice is 215-223 chars. The floor stays as
# a backstop for rows without retained text; the error notice is caught by its
# status line instead (is_http_error_fetch below, fetch_provenance.py).
MIN_SOURCE_CONTENT_CHARS = 200


def _site(url: str) -> str:
    try:
        host = (urlsplit(str(url or "").strip()).hostname or "").lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def source_page_candidates(
    seed_url: str, fetches: Iterable[SourceFetchEvidenceV1]
) -> list[SourceFetchEvidenceV1]:
    """Fetches that could be this seed's page: same site, and not the site's
    homepage or a listing page (a fetch of ``/`` or ``/news`` after the article
    404s is not a read of the article). Exact-path matching is deliberately not
    required -- live reads legitimately reach the content through a sibling
    endpoint (YouTube oembed, the arXiv export API)."""
    return [
        f
        for f in fetches
        if same_site(seed_url, f.url) and not url_looks_like_section_index(f.url)
    ]


def same_site(seed_url: str, fetched_url: str) -> bool:
    """Same host, or one a subdomain of the other (arxiv.org vs
    export.arxiv.org, nvidianews.nvidia.com vs nvidia.com). A different
    publication covering the same story is not a read of this source."""
    a, b = _site(seed_url), _site(fetched_url)
    if not a or not b:
        return False
    return a == b or a.endswith("." + b) or b.endswith("." + a)


def parse_source_fetches(raw: Any) -> Optional[list[SourceFetchEvidenceV1]]:
    """``None`` when the frame carried no report at all; a list otherwise.
    Malformed entries are dropped, never promoted to evidence."""
    if raw is None:
        return None
    if not isinstance(raw, list):
        return []
    out: list[SourceFetchEvidenceV1] = []
    for item in raw:
        try:
            out.append(SourceFetchEvidenceV1.model_validate(item))
        except Exception:  # noqa: BLE001
            continue
    return out


def document_snapshot_evidence(
    seed_url: str, *, content_sha256: str, content_chars: int
) -> SourceFetchEvidenceV1:
    """Hub put this exact snapshot into the reader's bound prompt."""
    return SourceFetchEvidenceV1(
        url=seed_url, tool_name=SNAPSHOT_TOOL,
        content_chars=content_chars, content_sha256=content_sha256,
    )


def source_read_evidence(
    seed_url: str, fetches: Iterable[SourceFetchEvidenceV1]
) -> list[SourceFetchEvidenceV1]:
    """The subset of ``fetches`` that shows this seed's source was read: a
    non-listing page on the seed's site that returned real content. For an
    internal document, only Hub's own snapshot record for that exact
    ``file://...?sha256=`` source counts -- no model tool call can produce one."""
    if is_document_ref(seed_url):
        return [
            f for f in fetches
            if f.tool_name == SNAPSHOT_TOOL and f.url == seed_url
            and f.content_sha256 and seed_url.endswith(f"sha256={f.content_sha256}")
            and f.content_chars > 0
        ]
    return [
        f
        for f in source_page_candidates(seed_url, fetches)
        if f.content_chars >= MIN_SOURCE_CONTENT_CHARS and not is_http_error_fetch(f)
    ]


def is_http_error_fetch(fetch: SourceFetchEvidenceV1) -> bool:
    """The tool_result was the fetch tool's HTTP error notice, not the page.
    Decidable only when the harness carried the text (every reading turn does,
    ``reading_only`` -> ``retain_text``); a row without text is judged by size."""
    return fetch_http_error_status(fetch.tool_name, fetch.content_text) is not None


def no_evidence_reason(
    seed_url: str, fetches: Optional[list[SourceFetchEvidenceV1]]
) -> str:
    """last_error label for a turn with no read evidence."""
    if fetches is None:
        return NO_READ_EVIDENCE_UNREPORTED
    candidates = source_page_candidates(seed_url, fetches)
    if any(is_http_error_fetch(f) for f in candidates):
        return NO_READ_EVIDENCE_HTTP_ERROR
    if candidates:
        return NO_READ_EVIDENCE_THIN
    return NO_READ_EVIDENCE
