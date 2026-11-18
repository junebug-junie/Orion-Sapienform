"""The body's attention:juniper line must count the same messages as the Hub's daily cap.

Regression (2026-11-07): Hub's ``count_sent_on`` excluded ``office_reach_out`` from 2026-11-05, but this
service's own copy of the list did not, so the body read each office message as a cap-counted send while
the Hub's cap never counted it. Since 2026-11-10 there is one list, owned by
``orion/physiology/attention_pacing.py``, and both sides import it: this test holds them to that.
"""
from __future__ import annotations

from pathlib import Path

from app import physiology_store
from orion.physiology import attention_pacing

HUB_DECISIONS = Path(__file__).resolve().parents[2] / "orion-hub" / "scripts" / "endogenous_outreach_decisions.py"


def test_one_exempt_list_read_by_both_sides() -> None:
    assert physiology_store.CAP_EXEMPT_SOURCES is attention_pacing.CAP_EXEMPT_SOURCES
    text = HUB_DECISIONS.read_text()
    assert "from orion.physiology.attention_pacing import" in text and "CAP_EXEMPT_SOURCES" in text
    assert "NOT IN ('curiosity_outreach'" not in text   # no second, hand-kept copy


def test_every_door_counts_except_the_strain_report() -> None:
    assert attention_pacing.CAP_EXEMPT_SOURCES == ("strain_report",)
    for door in ("curiosity_outreach", "reading_finding_outreach", "office_reach_out"):
        assert door not in attention_pacing.CAP_EXEMPT_SOURCES


def test_forced_sends_do_not_count_on_the_body_line() -> None:
    assert "NOT forced" in physiology_store.JUNIPER_SENT_SQL
