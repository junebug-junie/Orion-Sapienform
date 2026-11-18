"""Offline behavioral eval: dead pages are not reads, and hops follow fetched links only.

Every case is a frozen live row (2026-10-10..13, world_pulse_read_seed +
reading_document_snapshot + harness_turn_trace). The live replay that found them is
run_reading_dead_url_replay.py; this file pins the verdicts so they cannot regress.
"""
from __future__ import annotations

import pytest

from orion.schemas.reading import SourceFetchEvidenceV1
from orion.world_pulse_read.fetch_provenance import (
    hop_allowlist,
    hop_refusal_reason,
    reached_pages,
)
from orion.world_pulse_read.read_evidence import no_evidence_reason, source_read_evidence

_TAIL = ("\n\nThe response body was not retrieved. If this URL requires authentication, use an "
         "authenticated tool (e.g. `gh` for GitHub, or an MCP-provided fetch tool) instead of WebFetch.")
NOTICE_404 = "The server returned HTTP 404 Not Found." + _TAIL
COREWEAVE_202 = (
    "CoreWeave offers bare-metal NVIDIA GPUs (Blackwell, H200, H100) for AI training/inference. "
    "GB200 NVL72 provides 4x higher training performance vs H100 and supports 130 kW rack power "
    "with liquid cooling."
)
NASA_FLYOVER_EXCERPT = (
    "- **Culmination:** \"MAX POWER,\" a two-day public expo on American air/space innovation, "
    "**Sat Nov. 7 and Sun Nov. 8 (2026)**, on/near Kennedy Space Center, Florida; family-friendly "
    "and framed around America's 250th anniversary; details at nasa.gov/maxpower."
)


def _fetch(url: str, text: str) -> SourceFetchEvidenceV1:
    return SourceFetchEvidenceV1(url=url, tool_name="WebFetch", content_chars=len(text), content_text=text)


@pytest.mark.parametrize("url", [
    "https://www.coreweave.com/infrastructure",   # reading:a8f61bf4-...
    "https://www.coreweave.com/architecture",     # reading:a320c2dd-...
    "https://www.coreweave.com/gpu-cpu-compute",  # reading:e8403367-...
])
def test_live_404_reads_marked_done_would_now_fail(url: str) -> None:
    fetches = [_fetch(url, NOTICE_404)]
    assert source_read_evidence(url, fetches) == []
    assert no_evidence_reason(url, fetches) == "no_read_evidence:http_error"


def test_live_short_real_page_is_still_a_read() -> None:
    url = "https://www.coreweave.com/products/gpu-compute"  # finding:5ecbce82-...
    assert len(source_read_evidence(url, [_fetch(url, COREWEAVE_202)])) == 1


@pytest.mark.parametrize("hop", [
    "https://www.coreweave.com/infrastructure",
    "https://www.coreweave.com/architecture",
    "https://docs.coreweave.com",
])
def test_live_guessed_coreweave_hops_are_refused(hop: str) -> None:
    # Parent finding:5ecbce82-...: its only retained text is the 202-char digest,
    # and its Stage 2 turn's own fetches of these URLs were all 404 notices.
    stage2_404s = [_fetch(hop, NOTICE_404)]
    reached, texts = reached_pages(stage2_404s)  # what Stage 2's _hop_allowlist calls
    assert reached == [] and texts == []  # a 404 is not "reached"
    allow = hop_allowlist([COREWEAVE_202, *texts], reached)
    assert hop_refusal_reason(hop, allow) == "url_not_in_fetched_content"


def test_live_linked_hop_is_followed() -> None:
    # Parent digest_item:d7daaa9b-...: the hop to nasa.gov/maxpower read a real 1786-char page.
    allow = hop_allowlist([NASA_FLYOVER_EXCERPT])
    assert hop_refusal_reason("https://www.nasa.gov/maxpower", allow) is None
