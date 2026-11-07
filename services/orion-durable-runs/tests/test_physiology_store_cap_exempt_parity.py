"""The body's attention:juniper line must count the same messages as the Hub's daily cap.

Regression (2026-11-07): Hub's ``count_sent_on`` excluded ``office_reach_out`` from 2026-11-05 (office
visit messages skip the cap), but ``CAP_EXEMPT_SOURCES`` here did not, so the body read each office
message as a cap-counted send on attention:juniper while the Hub's cap never counted it.
"""
from __future__ import annotations

import re
from pathlib import Path

from app.physiology_store import CAP_EXEMPT_SOURCES

HUB_DECISIONS = Path(__file__).resolve().parents[2] / "orion-hub" / "scripts" / "endogenous_outreach_decisions.py"


def test_cap_exempt_sources_match_the_hub_daily_cap() -> None:
    text = HUB_DECISIONS.read_text()
    m = re.search(r"NOT IN \(([^)]*)\)", text)
    assert m, "count_sent_on's NOT IN list moved; update this parity test"
    hub = {s.strip().strip("'\"") for s in m.group(1).split(",") if s.strip()}
    assert hub == set(CAP_EXEMPT_SOURCES)
