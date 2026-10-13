"""Replay eval: the publisher-block skip, scored against real seed outcomes.

Fixture: every non-YouTube ``no_read_evidence`` seed 2026-10-05..13 and one
recently *done* read per site, pulled from ``world_pulse_read_seed``. The
skip must catch every BBC/NPR failure (each burned a full turn) and must
never catch a site Orion has actually read.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

from orion.world_pulse_read.reader_blocks import reader_block_reason

FIXTURE = Path(__file__).parent / "fixtures" / "reader_blocks_live_2026-10-13.json"


def _load() -> dict:
    return json.loads(FIXTURE.read_text())


def test_every_bbc_and_npr_failure_is_skipped_before_a_turn() -> None:
    failed = _load()["failed_no_read_evidence"]
    by_site = Counter(
        (urlsplit(r["url"]).hostname or "").removeprefix("www.") for r in failed
    )
    skipped = Counter(
        (urlsplit(r["url"]).hostname or "").removeprefix("www.")
        for r in failed
        if reader_block_reason(r["url"])
    )
    # Before: 18 BBC + 17 NPR turns spent, all no_read_evidence.
    assert by_site["bbc.co.uk"] == 18 and by_site["npr.org"] == 17
    # After: every one is skipped without a turn.
    assert skipped == {"bbc.co.uk": 18, "npr.org": 17}
    # Every BBC/NPR-family host (www., text., feeds., bbc.com, bbci) is caught.
    family = [r["url"] for r in failed if any(
        s in (urlsplit(r["url"]).hostname or "") for s in ("bbc.", "bbci.", "npr.org"))]
    assert family and all(reader_block_reason(u) for u in family)
    # NASA thin_fetch is a reachable site with short digests, not a block.
    assert all(
        reader_block_reason(r["url"]) is None
        for r in failed
        if "nasa.gov" in r["url"]
    )


def test_no_site_orion_has_read_is_skipped() -> None:
    done = _load()["done_reads"]
    assert len(done) >= 10
    assert [u for u in done if reader_block_reason(u)] == []
