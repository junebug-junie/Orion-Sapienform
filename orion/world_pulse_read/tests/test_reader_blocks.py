"""Sites that refuse the reader's fetch tool are named before a turn is spent."""
from __future__ import annotations

import pytest

from orion.world_pulse_read.reader_blocks import reader_block_reason


@pytest.mark.parametrize(
    "url, site",
    [
        # Live failed seeds (2026-10-13): WebFetch -> "Claude Code is unable to fetch from www.bbc.co.uk".
        ("https://www.bbc.co.uk/news/articles/cmn06w5x9g25o?at_medium=RSS&at_campaign=rss", "bbc.co.uk"),
        ("https://www.bbc.co.uk/news/articles/c9y7lgg2vdggo?at_medium=RSS&at_campaign=rss", "bbc.co.uk"),
        # The reader's own fallback host -> same refusal (16x).
        ("https://www.bbc.com/news/articles/cmn06w5x9g25o", "bbc.com"),
        ("https://feeds.bbci.co.uk/news/world/rss.xml", "bbci.co.uk"),
        # Live failed seeds: WebFetch -> "timeout of 60000ms exceeded" (robots: Claude-User Disallow /).
        ("https://www.npr.org/2026/10/12/nx-s1-5998948/new-white-house-press-secretary-a-devoted-fan-of-trump-and-maga", "npr.org"),
        ("https://www.npr.org/2026/10/06/g-s1-146626/up-first-newsletter-election-day-skydance-christa-pike-russia-plague-scare", "npr.org"),
        ("https://text.npr.org/nx-s1-5998948", "npr.org"),
        ("HTTPS://WWW.NPR.ORG./2026/10/12/x", "npr.org"),
    ],
)
def test_live_refusing_sites_are_named(url: str, site: str) -> None:
    reason = reader_block_reason(url)
    assert reason is not None and reason.startswith(f"source_blocks_reader:{site}:")


@pytest.mark.parametrize(
    "url",
    [
        # Live done reads on other sites must stay readable.
        "https://www.nasa.gov/maxpower",
        "https://www.who.int/news/item/30-07-2025-paying-tribute-to-david-nabarro",
        "https://arxiv.org/pdf/2509.07526.pdf",
        "https://nvidianews.nvidia.com/news/rubin-platform-ai-supercomputer",
        # Lookalike suffixes are other sites.
        "https://notbbc.com/news/1",
        "https://www.bbc.co.uk.example.com/news/1",
        "https://snpr.org/a",
        "",
        "not a url",
        "file:///mnt/scripts/Orion-Sapienform/docs/x.md?sha256=abc",
    ],
)
def test_other_sites_are_not_blocked(url: str) -> None:
    assert reader_block_reason(url) is None
