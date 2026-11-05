"""Attention as the want competition, patch 2: the market reads its event history; Prime's menu of
funded event wants (GET /physiology/event-wants). Fakes only."""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(Path(__file__).resolve().parents[1])]

from app.event_wants_menu import ASK_LABELS_SQL, LATEST_EVENT_DECISIONS_SQL, READ_LABELS_SQL, menu_from_rows, read_menu  # noqa: E402
from app.physiology_market import EVENT_HISTORY_SQL, Market  # noqa: E402
from orion.physiology.families import load_families  # noqa: E402
from orion.schemas.physiology import PhysiologyConstraintReadingV1 as R, PhysiologyStateV1  # noqa: E402
from orion.schemas.physiology_want import FundedEventWantsV1, WantV1  # noqa: E402

FAMS, _ = load_families(REPO / "config/physiology/families.yaml")
NOW = datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc)


class _Cursor:
    def __init__(self, rows):
        self.rows = rows

    async def fetchall(self):
        return self.rows


class _Pool:
    def __init__(self, by_sql):
        self.by_sql, self.seen = by_sql, []

    def connection(self):
        pool = self

        class _Conn:
            async def execute(self, sql, params=None):
                pool.seen.append((sql, params))
                rows = pool.by_sql.get(sql)
                if rows is None:
                    rows = next((v for k, v in pool.by_sql.items() if k in sql), [])
                return _Cursor(rows)

        class _CM:
            async def __aenter__(self):
                return _Conn()

            async def __aexit__(self, *a):
                return False

        return _CM()


def _row(want_id, ref, *, outcome="funded", net=0.5, decided=NOW - timedelta(minutes=10), family="hub.world_pulse_read",
         latest=None, opened=None):
    return {"want_id": want_id, "family": family, "outcome": outcome, "decided_at": decided, "net": net,
            "value_rank": net, "event_ref": ref, "salience": 0.9,
            "opened_at": (opened or decided).isoformat(), "latest": latest.isoformat() if latest else None}


def test_the_menu_keeps_funded_unexpired_events_highest_net_first():
    rows = [
        _row("a", "world_pulse_read_seed:s1", net=0.2),
        _row("b", "world_pulse_read_seed:s2", net=0.7),
        _row("c", "world_pulse_read_seed:s3", outcome="deferred", net=0.9),
        # Read 4 h ago: past the family's 3 h window.
        _row("d", "world_pulse_read_seed:s4", net=0.8, decided=NOW - timedelta(hours=4)),
        _row("e", "orion_ask:k1", net=0.6, family="vision.ask", latest=NOW + timedelta(days=5)),
        _row("f", "orion_ask:k2", net=0.95, family="vision.ask", latest=NOW + timedelta(days=5)),
    ]
    menu = menu_from_rows(rows, labels={"world_pulse_read_seed:s2": "a read finished: X"}, gone={"orion_ask:k2"},
                          families=FAMS, now=NOW, limit=8)
    assert [w.want_id for w in menu.wants] == ["b", "e", "a"]
    assert menu.considered == 6 and menu.funded_open == 3
    assert menu.wants[0].label == "a read finished: X"
    assert menu.wants[0].expires_at == NOW - timedelta(minutes=10) + timedelta(hours=3)
    assert len(menu_from_rows(rows, labels={}, gone=set(), families=FAMS, now=NOW, limit=1).wants) == 1


def test_read_menu_labels_from_the_source_rows_and_drops_an_answered_ask():
    pool = _Pool({
        LATEST_EVENT_DECISIONS_SQL: [_row("a", "world_pulse_read_seed:s1"),
                                     _row("e", "orion_ask:k1", family="vision.ask", latest=NOW + timedelta(days=5)),
                                     _row("f", "orion_ask:k2", family="vision.ask", latest=NOW + timedelta(days=5))],
        READ_LABELS_SQL: [{"seed_id": "s1", "title": "Bees learn", "url": "https://b"}],
        ASK_LABELS_SQL: [{"ask_id": "k1", "question": "Is this Jacob?", "status": "open"},
                         {"ask_id": "k2", "question": "Whose car?", "status": "answered"}],
    })
    menu = asyncio.run(read_menu(pool, FAMS, now=NOW, hours=24, limit=8))
    labels = {w.event_ref: w.label for w in menu.wants}
    assert labels == {"world_pulse_read_seed:s1": "a read finished: Bees learn",
                      "orion_ask:k1": "an open camera question to Juniper: Is this Jacob?"}


def test_a_cooling_act_want_is_labelled_from_its_own_reading_ref():
    ref = "cabinet_reading:athena:2026-11-05T18:00:00Z"
    pool = _Pool({LATEST_EVENT_DECISIONS_SQL: [_row("c", ref, family="self_regulate.shed",
                                                    latest=NOW + timedelta(minutes=10))]})
    menu = asyncio.run(read_menu(pool, FAMS, now=NOW, hours=24, limit=8))
    assert [w.label for w in menu.wants] == ["the athena cabinet was warming at 2026-11-05T18:00:00Z (Orion's cooling act)"]
    assert FundedEventWantsV1.model_validate_json(menu.model_dump_json()) == menu


def _state():
    return PhysiologyStateV1(tick_id="t", generated_at=NOW, constraints=[
        R(constraint="heat", unit="c", level=24.0, budget=30.0, budget_kind="soft_line",
          price={"value": 0.0, "ceiling": 1.0, "bind_state": "slack", "horizon_sec": 3600.0, "why": ""})])


class _Redis:
    async def get(self, key):
        return _state().model_dump_json()


def _acts():
    return [{"act_id": f"a{k}", "actor": "hub.world_pulse_read", "opened_at": NOW - timedelta(minutes=30 * k + 20),
             "closed_at": NOW - timedelta(minutes=30 * k), "outcome_value": None, "outcome_source": None,
             "signal_id": None, "cost_json": None} for k in range(1, 40)]


def test_the_market_ranks_an_event_want_against_its_event_history():
    history = [{"family": "hub.world_pulse_read", "intensity": v} for v in [0.05] * 30 + [0.9] * 10]
    pool = _Pool({EVENT_HISTORY_SQL: history, "grammar_acts": _acts(), "act_downstream_use": [],
                  "outcome = 'funded'": []})
    published = []

    async def publish(d):
        published.append(d)

    m = Market(families=FAMS, families_error=None, pool_getter=lambda: pool, redis_getter=lambda: _Redis(), publish=publish)
    want = WantV1(want_id="w", family="hub.world_pulse_read", producer="orion-hub.reading_events", verb="look", tier=2,
                  opened_at=NOW, event_ref="world_pulse_read_seed:s", salience=0.9)
    d = asyncio.run(m.bid(want, NOW))
    assert d.value_basis.startswith("event:") and d.outcome == "funded"
    assert any(sql == EVENT_HISTORY_SQL for sql, _ in pool.seen)
    assert json.loads(published[0].model_dump_json())["want"]["salience"] == 0.9


def test_a_failed_event_history_read_ranks_on_intensity_and_never_fails_the_bid():
    class _Bad(_Pool):
        def connection(self):
            cm = super().connection()
            outer = self

            class _CM:
                async def __aenter__(self):
                    conn = await cm.__aenter__()
                    orig = conn.execute

                    async def execute(sql, params=None):
                        if sql == EVENT_HISTORY_SQL:
                            raise RuntimeError("down")
                        return await orig(sql, params)

                    conn.execute = execute
                    return conn

                async def __aexit__(self, *a):
                    return False

            return _CM()

    pool = _Bad({"grammar_acts": _acts(), "act_downstream_use": [], "outcome = 'funded'": []})

    async def publish(d):
        pass

    m = Market(families=FAMS, families_error=None, pool_getter=lambda: pool, redis_getter=lambda: _Redis(), publish=publish)
    want = WantV1(want_id="w", family="hub.world_pulse_read", producer="p", verb="look", tier=2, opened_at=NOW,
                  event_ref="world_pulse_read_seed:s", salience=0.4)
    d = asyncio.run(m.bid(want, NOW))
    # No event history: ranked as any want, against the hunger history (intensity max(hunger 0.63, 0.4)).
    assert d.value_basis == "event_cold:ungraded" and d.outcome == "funded"
    assert d.intensity == pytest.approx(1 - 1 / 2.718281828, abs=1e-3)
