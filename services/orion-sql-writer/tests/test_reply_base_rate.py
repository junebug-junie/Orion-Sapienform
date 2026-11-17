"""Juniper's reply base rate for an outreach act's time of day (#3392 option A). Pure, no Postgres."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

import orion.grammar.reply_base_rate as rbr
from orion.grammar.reply_base_rate import ReplyBaseRate, applies, base_rate_from_turns

DENVER = ZoneInfo("America/Denver")
H10 = 36000.0
OPENED = datetime(2026, 11, 15, 22, 0, tzinfo=DENVER)  # 22:00 Denver


def _daily(at_local_hour: int, days: int, *, minute: int = 30):
    """One Juniper turn per day at a local hour, on OPENED's own date and the ``days`` days before it."""
    return [datetime(2026, 11, 15, at_local_hour, minute, tzinfo=DENVER) - timedelta(days=k) for k in range(0, days + 1)]


def test_hour_with_no_history_has_no_base_rate():
    # She has never chatted at all: no rate (the ledger keeps the posterior stamp).
    assert base_rate_from_turns([], opened_at=OPENED, horizon_sec=H10, first_turn_at=None) is None
    # Her first turn ever was 3 days ago: 3 days of history is under the minimum.
    first = OPENED - timedelta(days=3, hours=1)
    assert base_rate_from_turns([first], opened_at=OPENED, horizon_sec=H10, first_turn_at=first) is None


def test_hour_she_never_answers_reads_low_not_zero():
    # Turns only at 09:30 local: the 22:00 window (22:00 -> 08:00) never sees one.
    turns = _daily(9, 40)
    br = base_rate_from_turns(turns, opened_at=OPENED, horizon_sec=H10, first_turn_at=min(turns))
    assert br == ReplyBaseRate(rate=pytest.approx(0.5 / 29), days=28, replied_days=0)


def test_hour_where_she_always_replies_reads_calm_on_a_reply():
    # A turn every morning at 06:30 (inside 22:00 + 10 h even on the DST night, when that ends at 07:00): every 22:00 window has one. A reply is "as usual" (error ~0),
    # a silence is a large miss.
    turns = _daily(6, 40)
    br = base_rate_from_turns(turns, opened_at=OPENED, horizon_sec=H10, first_turn_at=min(turns))
    assert br.days == 28 and br.replied_days == 28
    assert br.rate == pytest.approx(28.5 / 29)
    assert abs(1.0 - br.rate) < 0.02 and (0.0 - br.rate) < -0.95


def test_only_days_before_the_act_count_and_partial_history_is_used():
    # 10 days of history, a turn every day inside the window; a turn after the open must not leak in.
    turns = _daily(6, 9) + [OPENED + timedelta(hours=1)]
    br = base_rate_from_turns(turns, opened_at=OPENED, horizon_sec=H10, first_turn_at=min(turns) - timedelta(hours=12))
    assert br.days == 10 and br.replied_days == 10


def test_window_edges_are_open_then_closed():
    # (lo, lo + horizon]: a turn exactly at lo is not a reply, one exactly at lo + horizon is.
    lo1 = OPENED - timedelta(days=1)
    first = OPENED - timedelta(days=30)
    at_lo = base_rate_from_turns([lo1], opened_at=OPENED, horizon_sec=H10, first_turn_at=first)
    at_hi = base_rate_from_turns([lo1 + timedelta(seconds=H10)], opened_at=OPENED, horizon_sec=H10, first_turn_at=first)
    assert at_lo.replied_days == 0 and at_hi.replied_days == 1


def test_same_local_hour_across_dst():
    # 2026-11-01 is the fall-back day in Denver. A turn every day at 23:00 LOCAL, act opened at 22:00
    # local: a 1 h horizon must see every day's turn on both sides of the change.
    opened = datetime(2026, 11, 10, 22, 0, tzinfo=DENVER)
    turns = [datetime(2026, 11, 10, 22, 59, tzinfo=DENVER) - timedelta(days=k) for k in range(1, 20)]
    br = base_rate_from_turns(turns, opened_at=opened, horizon_sec=3600.0, first_turn_at=min(turns) - timedelta(days=1),
                              days=18)
    assert br.days == 18 and br.replied_days == 18


def test_naive_utc_inputs_are_accepted():
    turns = [t.astimezone(timezone.utc).replace(tzinfo=None) for t in _daily(6, 40)]
    br = base_rate_from_turns(turns, opened_at=OPENED.astimezone(timezone.utc), horizon_sec=H10, first_turn_at=min(turns))
    assert br.replied_days == 28


def test_applies_only_to_outreach_replies_and_obeys_the_switch():
    assert applies("hub.outreach", "juniper_replied")
    assert not applies("hub.chat_reply", "juniper_replied")
    assert not applies("hub.outreach", "artifact_produced")
    try:
        rbr.configure_reply_base_rate(False)
        assert not applies("hub.outreach", "juniper_replied")
    finally:
        rbr.configure_reply_base_rate(True)


def test_flag_ships_on():
    import importlib

    assert importlib.reload(rbr)._ENABLED is True
