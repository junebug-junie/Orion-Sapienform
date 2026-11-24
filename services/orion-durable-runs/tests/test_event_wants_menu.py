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

from app.event_wants_menu import (  # noqa: E402
    FINISH_LABELS_SQL, finding_label,
    ASK_LABELS_SQL, LATEST_EVENT_DECISIONS_SQL, OFFICE_LABELS_SQL, OFFICE_TABLE_SQL, READ_LABELS_SQL, menu_from_rows,
    OPEN_WINDOW_SEARCH_HOURS, read_menu,
)
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


def test_a_three_day_old_open_camera_ask_stays_on_the_menu_prime_reads():
    """#3558: a camera ask is decided once and stays open 7 days. Prime reads ``hours=24``; an ask decided
    3 days ago whose window is still open is on the menu. An answered one and an expired one are not."""
    three_days = NOW - timedelta(days=3)
    pool = _Pool({
        LATEST_EVENT_DECISIONS_SQL: [
            _row("open", "orion_ask:k1", family="vision.ask", decided=three_days, latest=three_days + timedelta(days=7)),
            _row("answered", "orion_ask:k2", family="vision.ask", decided=three_days,
                 latest=three_days + timedelta(days=7)),
            _row("expired", "orion_ask:k3", family="vision.ask", decided=NOW - timedelta(days=7, hours=1),
                 latest=NOW - timedelta(hours=1)),
        ],
        ASK_LABELS_SQL: [{"ask_id": "k1", "question": "Is this Jacob?", "status": "open"},
                         {"ask_id": "k2", "question": "Whose car?", "status": "answered"},
                         {"ask_id": "k3", "question": "Who is this?", "status": "open"}],
    })
    menu = asyncio.run(read_menu(pool, FAMS, now=NOW, hours=24, limit=8))
    assert [w.want_id for w in menu.wants] == ["open"]
    assert menu.wants[0].label == "an open camera question to Juniper: Is this Jacob?"
    params = pool.seen[0][1]
    # The search reaches back past ``hours`` far enough to find any still-open camera ask (7-day window).
    assert params["since"] == NOW - timedelta(hours=24)
    assert params["search_since"] == NOW - timedelta(hours=OPEN_WINDOW_SEARCH_HOURS)
    assert OPEN_WINDOW_SEARCH_HOURS > 7 * 24
    assert params["now"] == NOW


def test_an_old_want_with_no_known_window_end_is_not_kept_past_hours():
    """Past ``hours`` only a want with a known, still-open window end is kept (no end, no proof it is open).
    Inside ``hours`` the old rule holds: no end reads as open."""
    old = NOW - timedelta(days=2)
    rows = [_row("old", "x_unknown:1", family="no.such.family", decided=old),
            _row("new", "x_unknown:2", family="no.such.family")]
    menu = menu_from_rows(rows, labels={}, gone=set(), families=FAMS, now=NOW, limit=8,
                          since=NOW - timedelta(hours=24))
    assert [w.want_id for w in menu.wants] == ["new"]


def test_the_menu_sql_keeps_open_windows_older_than_hours_after_picking_the_latest_decision():
    sql = " ".join(LATEST_EVENT_DECISIONS_SQL.split())
    # The window filter runs on each want's LATEST decision (outer query), never before DISTINCT ON.
    assert sql.index("DISTINCT ON (want_id)") < sql.index(") latest_decision")
    assert "decided_at >= %(search_since)s" in sql
    assert ("WHERE decided_at >= %(since)s OR (outcome = 'funded' AND CASE WHEN latest ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}' "
            "THEN latest::timestamptz END > %(now)s)") in sql


def test_a_funded_ask_want_whose_card_never_opened_is_not_a_concern():
    """Attention patch 4b: the ask bids before its card is inserted. A funded want naming a card that does not
    exist (the insert lost to an open card for the same subject) is not shown to Prime."""
    pool = _Pool({
        LATEST_EVENT_DECISIONS_SQL: [_row("e", "orion_ask:k1", family="vision.ask", latest=NOW + timedelta(days=5)),
                                     _row("g", "orion_ask:never", family="vision.ask", latest=NOW + timedelta(days=5))],
        ASK_LABELS_SQL: [{"ask_id": "k1", "question": "Is this Jacob?", "status": "open"}],
    })
    menu = asyncio.run(read_menu(pool, FAMS, now=NOW, hours=24, limit=8))
    assert [w.event_ref for w in menu.wants] == ["orion_ask:k1"]


def test_a_cooling_act_want_is_labelled_from_its_own_reading_ref():
    ref = "cabinet_reading:athena:2026-11-05T18:00:00Z"
    pool = _Pool({LATEST_EVENT_DECISIONS_SQL: [_row("c", ref, family="self_regulate.shed",
                                                    latest=NOW + timedelta(minutes=10))]})
    menu = asyncio.run(read_menu(pool, FAMS, now=NOW, hours=24, limit=8))
    assert [w.label for w in menu.wants] == ["the athena cabinet was warming at 2026-11-05T18:00:00Z (Orion's cooling act)"]
    assert FundedEventWantsV1.model_validate_json(menu.model_dump_json()) == menu


def test_an_office_visit_want_is_labelled_from_its_session_row():
    ref = "office_session:cam0-abc"
    rows = [_row("o", ref, family="perception.office")]
    pool = _Pool({LATEST_EVENT_DECISIONS_SQL: rows, OFFICE_TABLE_SQL: [{"present": True}],
                  OFFICE_LABELS_SQL: [{"session_id": "cam0-abc", "who_text": "Juniper (probably)",
                                       "label": "Juniper (probably) in the office: sitting at a desk, opening a box"}]})
    menu = asyncio.run(read_menu(pool, FAMS, now=NOW, hours=24, limit=8))
    assert [w.label for w in menu.wants] == ["Juniper (probably) in the office: sitting at a desk, opening a box"]
    # The family's 2 h window bounds it.
    assert menu.wants[0].expires_at == rows[0]["decided_at"] + timedelta(hours=2)
    # Before the table exists (migration not applied) the want stays on the menu, unlabelled.
    pool = _Pool({LATEST_EVENT_DECISIONS_SQL: rows, OFFICE_TABLE_SQL: [{"present": False}]})
    menu = asyncio.run(read_menu(pool, FAMS, now=NOW, hours=24, limit=8))
    assert [w.label for w in menu.wants] == [""]
    assert not any(sql == OFFICE_LABELS_SQL for sql, _ in pool.seen)


def test_a_curiosity_finding_want_is_labelled_from_its_finish_row():
    ref, ref2 = "curiosity_run:run1", "curiosity_run:run2"
    rows = [_row("f", ref, family="hub.curiosity_finding", net=0.6),
            _row("g", ref2, family="hub.curiosity_finding", net=0.4)]
    pool = _Pool({LATEST_EVENT_DECISIONS_SQL: rows, FINISH_LABELS_SQL: [
        {"run_id": "run1", "detail": {"line": "self_inquiry", "finding_text": "Done for the sitting.",
                                       "lived_answer": {"question_id": "q", "text": "Yes, but what comes back\nis my own loops."}}},
        # json columns can arrive as text; a finish row with only narration falls back to it, headings skipped.
        {"run_id": "run2", "detail": json.dumps({"finding_text": "## What I followed\n\nThe flicker is real."})},
    ]})
    menu = asyncio.run(read_menu(pool, FAMS, now=NOW, hours=24, limit=8))
    assert [w.label for w in menu.wants] == ["a curiosity finding: Yes, but what comes back is my own loops.",
                                             "a curiosity finding: The flicker is real."]
    sql, params = next((q, p) for q, p in pool.seen if q == FINISH_LABELS_SQL)
    assert params == {"ids": ["run1", "run2"]} and "node = 'finish'" in sql
    # The family's 12 h window bounds it.
    assert menu.wants[0].expires_at == rows[0]["decided_at"] + timedelta(hours=12)


def test_finding_label_prefers_the_answer_and_caps_its_length():
    assert finding_label({"self_definition": {"text": "I am one process."}, "finding_text": "narration"}) == \
        "a curiosity finding: I am one process."
    assert finding_label({}) == "" and finding_label(None) == "" and finding_label("not json") == ""
    long = finding_label({"lived_answer": {"text": "x" * 1000}})
    assert len(long) <= 300 and long.endswith("…")


def _state():
    return PhysiologyStateV1(tick_id="t", generated_at=NOW, constraints=[
        R(constraint="heat", unit="c", level=24.0, budget=30.0, budget_kind="soft_line",
          price={"value": 0.0, "ceiling": 1.0, "bind_state": "slack", "horizon_sec": 3600.0, "why": ""}),
        # Reading's lanes (attention patch 3a), idle.
        *[R(constraint=f"lane:{lane}", unit="slot_s", level=0.0, budget=budget, budget_kind="capacity",
            detail={"slots": int(budget // 3600), "held_spans": 0, "window_sec": 3600.0},
            price={"value": 0.0, "ceiling": 1.0, "bind_state": "slack", "horizon_sec": 3600.0, "why": ""})
          for lane, budget in (("agent", 3600.0), ("agent-gpu2", 7200.0))]])


class _Redis:
    async def get(self, key):
        return _state().model_dump_json()


def _acts():
    return [{"act_id": f"a{k}", "actor": "hub.world_pulse_read", "opened_at": NOW - timedelta(minutes=30 * k + 20),
             "closed_at": NOW - timedelta(minutes=30 * k), "outcome_value": None, "outcome_source": None,
             "signal_id": None,
             # A read's measured lane cost (attention patch 3a: reading declares [agent, agent-gpu2]; live
             # median 692 slot-s). With no measured cost a read is priced at one whole window.
             "cost_json": {"gpu_slot_s_by_lane": {"agent-gpu2": 692.0}}} for k in range(1, 40)]


def test_the_market_ranks_an_event_want_against_its_event_history():
    # The fake acts carry no outcome, so the family is ungraded: its past events were decided ungraded too.
    history = [{"family": "hub.world_pulse_read", "intensity": v, "graded": False} for v in [0.05] * 30 + [0.9] * 10]
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


def test_the_market_ranks_only_against_past_events_decided_with_the_same_grading():
    """Issue #3364: the family is ungraded here, so 40 rows decided graded (or with no grading) are not its
    reference; under EVENT_HISTORY_MIN same-grading rows the want ranks cold. With the same rows labelled ungraded
    it ranks against them."""
    rows = [0.05] * 30 + [0.9] * 10

    def bid(graded):
        hist = [{"family": "hub.world_pulse_read", "intensity": v, "graded": graded} for v in rows]
        pool = _Pool({EVENT_HISTORY_SQL: hist, "grammar_acts": _acts(), "act_downstream_use": [],
                      "outcome = 'funded'": []})

        async def publish(d):
            pass

        m = Market(families=FAMS, families_error=None, pool_getter=lambda: pool, redis_getter=lambda: _Redis(),
                   publish=publish)
        want = WantV1(want_id="w", family="hub.world_pulse_read", producer="orion-hub.reading_events", verb="look",
                      tier=2, opened_at=NOW, event_ref="world_pulse_read_seed:s", salience=0.9)
        return asyncio.run(m.bid(want, NOW))

    assert bid(False).value_basis == "event:ungraded"
    assert bid(True).value_basis.startswith("event_cold:")
    assert bid(None).value_basis.startswith("event_cold:")     # grading unknown: not this family's reference


def test_the_event_history_read_carries_each_rows_grading():
    from app.physiology_market import read_event_history

    pool = _Pool({EVENT_HISTORY_SQL: [{"family": "f", "intensity": 0.5, "graded": True},
                                      {"family": "f", "intensity": 0.7, "graded": False},
                                      {"family": "f", "intensity": 0.9, "graded": None},
                                      {"family": "f", "intensity": None, "graded": True}]})
    assert asyncio.run(read_event_history(pool, NOW)) == {"f": [(0.5, True), (0.7, False), (0.9, None)]}
    sql = " ".join(EVENT_HISTORY_SQL.split())
    assert "CASE jsonb_typeof(decision->'graded') WHEN 'boolean' THEN (decision->>'graded')::boolean END AS graded FROM want_ledger" in sql


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


def _care(nwc):
    return ["personality:arm=care", "personality:care=0.5000:tendency=d1:cos=0.9000",
            f"personality:net_without_care={nwc:.6f}"]


def test_care_promoted_marks_a_want_on_the_menu_only_through_care():
    """Personality rev 4 runaway guard for Prime's concern source: a funded event want inside the top
    ``limit`` by net that falls outside it by net without care, or that would not be funded at all
    without care, is ``care_promoted``. Prime sets the flag on a choice of its concern."""
    rows = [
        dict(_row("a", "world_pulse_read_seed:s1", net=0.50), reason="net_positive", warnings=_care(0.10)),
        dict(_row("b", "world_pulse_read_seed:s2", net=0.40), reason="net_positive", warnings=[]),
        dict(_row("c", "world_pulse_read_seed:s3", net=0.30), reason="net_positive", warnings=_care(0.30)),
        # Funded only because care lifted its net above 0: care-promoted wherever it ranks.
        dict(_row("d", "world_pulse_read_seed:s4", net=0.05), reason="net_positive", warnings=_care(-0.02)),
    ]
    menu = menu_from_rows(rows, labels={}, gone=set(), families=FAMS, now=NOW, limit=2)
    # Shown: a (0.50) and b (0.40). Without care a (0.10) drops below b and c: a is care-promoted.
    assert [(w.want_id, w.care_promoted) for w in menu.wants] == [("a", True), ("b", False)]
    menu = menu_from_rows(rows, labels={}, gone=set(), families=FAMS, now=NOW, limit=8)
    assert {w.want_id: w.care_promoted for w in menu.wants} == {"a": False, "b": False, "c": False, "d": True}
    # No personality trace (flag off, old rows): nothing is care-promoted.
    plain = [{k: v for k, v in r.items() if k != "warnings"} for r in rows]
    assert not any(w.care_promoted for w in menu_from_rows(plain, labels={}, gone=set(), families=FAMS,
                                                           now=NOW, limit=2).wants)
    # jsonb as text, and the contract round-trips.
    as_text = [dict(r, warnings=json.dumps(r["warnings"])) for r in rows]
    menu = menu_from_rows(as_text, labels={}, gone=set(), families=FAMS, now=NOW, limit=2)
    assert [w.care_promoted for w in menu.wants] == [True, False]
    assert FundedEventWantsV1.model_validate_json(menu.model_dump_json()) == menu


def test_read_menu_passes_reason_and_jsonb_warnings_through_to_care_promoted():
    rows = [
        dict(_row("a", "world_pulse_read_seed:s1", net=0.50), reason="net_positive",
             warnings=json.dumps(_care(0.10))),
        dict(_row("b", "world_pulse_read_seed:s2", net=0.40), reason="net_positive", warnings=None),
        dict(_row("c", "world_pulse_read_seed:s3", net=0.30), reason="net_positive", warnings=_care(0.30)),
    ]
    pool = _Pool({LATEST_EVENT_DECISIONS_SQL: rows})
    menu = asyncio.run(read_menu(pool, FAMS, now=NOW, hours=24, limit=2))
    assert "decision->>'reason' AS reason" in pool.seen[0][0] and "decision->'warnings' AS warnings" in pool.seen[0][0]
    assert [(w.want_id, w.care_promoted) for w in menu.wants] == [("a", True), ("b", False)]


def test_the_payload_leaves_care_promoted_out_unless_asked():
    """``FundedEventWantV1`` is extra="forbid": an orion-mind built before the field must still validate the
    list, whatever the deploy order. Prime asks with ``care=1``."""
    from app.event_wants_menu import menu_payload

    rows = [dict(_row("a", "world_pulse_read_seed:s1", net=0.05), reason="net_positive", warnings=_care(-0.02))]
    menu = menu_from_rows(rows, labels={}, gone=set(), families=FAMS, now=NOW, limit=8)
    plain = menu_payload(menu, include_care=False)
    assert "care_promoted" not in plain["wants"][0]
    asked = menu_payload(menu, include_care=True)
    assert asked["wants"][0]["care_promoted"] is True
    assert FundedEventWantsV1.model_validate(plain).wants[0].care_promoted is False
