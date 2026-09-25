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

# last_error labels. Plain `no_read_evidence` is terminal (the model ran and
# chose not to / could not fetch the source -- a retry spends another slot on
# the same outcome). The `:harness_unreported` form means the governor that ran
# the turn predates `HarnessRunV1.source_fetches`; that is an infrastructure
# condition, listed as transient in orion/world_pulse_read/retry.py.
NO_READ_EVIDENCE = "no_read_evidence"
NO_READ_EVIDENCE_UNREPORTED = "no_read_evidence:harness_unreported"

# Floor on the tool_result text the model received. A WebFetch result is the
# fetch tool's own multi-sentence digest of the page; refusal/blocked strings
# ("Request failed with status code 403", "Unable to fetch ...") are well under
# this. UNVERIFIED against live WebFetch bodies (raw steps are not persisted);
# every accepted/rejected fetch logs its content_chars so this can be checked.
MIN_SOURCE_CONTENT_CHARS = 200


def _site(url: str) -> str:
    try:
        host = (urlsplit(str(url or "").strip()).hostname or "").lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


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


def source_read_evidence(
    seed_url: str, fetches: Iterable[SourceFetchEvidenceV1]
) -> list[SourceFetchEvidenceV1]:
    """The subset of ``fetches`` that shows this seed's source was read."""
    return [
        f
        for f in fetches
        if f.content_chars >= MIN_SOURCE_CONTENT_CHARS and same_site(seed_url, f.url)
    ]
