"""Orion plays Zelda, section F: one deterministic memory per finished Retro session.

Pins:
- the summary is facts and counts only, the same every time, and reads like the spec's example;
- control share: a human driving MORE than half of the frames is "played together"; exactly half is not;
- a journal with no observed start or end is not summarized (no empty-shell memory);
- the card has provenance and trust_source `retro_play`, is active (recall reads active cards), and
  links session and lineage;
- export writes the session row first (the claim), then one card, then links them; a replayed session
  end writes no second card; the switch off writes nothing; a leisure dispatch's `play` act is closed
  1 only when the session had a real planner decision;
- the lazy DDL and the migration declare the same columns.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

import pytest

from orion.schemas.grammar import GrammarEventV1
from scripts import retro_memory_export as rx

T0 = datetime(2026, 10, 22, 20, 2, tzinfo=timezone.utc)
MIGRATION = Path(__file__).resolve().parents[2] / "orion-sql-db" / "manual_migration_retro_play_sessions_v1.sql"


def _journal(*, minutes=22, human=0, orion=79200, met=14, refuted=12, unknown=5, end=True):
    out = [{"kind": "session_started", "at": T0.isoformat()},
           {"kind": "control", "port": 0, "controller": "orion", "frames": orion},
           {"kind": "control", "port": 0, "controller": "human", "frames": human}]
    verdicts = ["met"] * met + ["refuted"] * refuted + ["unknown"] * unknown
    out += [{"kind": "goal_finished", "decision_id": f"d{i}", "verdict": v, "human_intervention": False}
            for i, v in enumerate(verdicts)]
    if end:
        out.append({"kind": "session_ended", "at": (T0 + timedelta(minutes=minutes)).isoformat(), "reason": "budget"})
    return out


DELTA = {
    "rooms_new": [f"0:{r}" for r in (0x77, 0x78, 0x79, 0x67, 0x68, 0x69, 0x57, 0x58, 0x59)],
    "caves_new": ["0:77:c1", "0:67:c1", "0:59:c1"],
    "items_gained": [{"item": "sword", "value": 1, "room": "0:77"}],
    "deaths": 2,
    "death_rooms": ["0:79"],
}


def _summary(**kw):
    journal_kw = {k: kw.pop(k) for k in list(kw) if k in ("minutes", "human", "orion", "met", "refuted", "unknown", "end")}
    return rx.build_session_summary("sess-1", _journal(**journal_kw), kw.pop("delta", DELTA),
                                    kind=kw.pop("kind", "leisure"), game_id="zelda-prg0", lineage_id="lin-1", **kw)


def test_summary_is_facts_and_counts_like_the_spec_example() -> None:
    s = _summary()
    assert s.summary == ("Played The Legend of Zelda solo for 22 min, 9 new rooms (3 caves), got the wooden sword, "
                         "died twice, 14 of 31 predictions met, 5 unknown.")
    assert (s.rooms_new, s.caves_new, s.deaths, s.death_rooms) == (9, 3, 2, ["0:79"])
    assert (s.decisions, s.predictions_met, s.predictions_refuted, s.predictions_unknown) == (31, 14, 12, 5)
    assert s.human_share == 0.0 and not s.played_together and s.control_basis == "frames"
    assert _summary() == s  # deterministic


def test_summary_never_claims_fun_or_learning() -> None:
    for human in (0, 30000, 60000, 79200):
        text = _summary(human=human).summary.lower()
        assert not re.search(r"\b(fun|enjoy|learn|learned|learnt|loved)\b", text), text


def test_a_human_driving_most_frames_is_played_together() -> None:
    s = _summary(kind="co_play", human=60000, orion=40000)
    assert s.played_together and s.human_share == 0.6
    assert s.summary.startswith("Played The Legend of Zelda together for 22 min (a human drove 60% of frames)")
    half = _summary(kind="co_play", human=50000, orion=50000)
    assert not half.played_together  # "more than 50%", not 50%
    assert half.summary.startswith("Played The Legend of Zelda for 22 min (Orion drove 50% of frames)")


def test_control_share_is_per_port_and_summed() -> None:
    journal = _journal(human=0, orion=0) + [
        {"kind": "control", "port": 0, "controller": "orion", "frames": 100},
        {"kind": "control", "port": 1, "controller": "human", "frames": 300},
        {"kind": "control", "port": 0, "controller": "human", "frames": 50}]
    s = rx.build_session_summary("s", journal, DELTA, kind="co_play", game_id="zelda-prg0")
    assert [(c.port, c.human_frames, c.orion_frames) for c in s.control] == [(0, 50, 100), (1, 300, 0)]
    assert s.human_share == pytest.approx(350 / 450) and s.played_together


def test_no_frame_lines_leisure_is_orions_by_the_sponsor_rule_co_play_is_unknown() -> None:
    s = _summary(human=0, orion=0)
    assert (s.control_basis, s.human_share, s.played_together) == ("leisure_sponsor", 0.0, False)
    c = _summary(human=0, orion=0, kind="co_play")
    assert (c.control_basis, c.human_share, c.played_together) == ("unknown", None, False)
    assert c.summary.startswith("Played The Legend of Zelda with a human for 22 min (control share not recorded)")


def test_unrecorded_atlas_counts_are_left_out_never_zero() -> None:
    s = _summary(delta={})
    assert (s.rooms_new, s.caves_new, s.deaths) == (None, None, None)
    assert s.summary == "Played The Legend of Zelda solo for 22 min, 14 of 31 predictions met, 5 unknown."


def test_verdict_counts_stand_in_when_the_journal_has_no_goal_lines() -> None:
    journal = [e for e in _journal() if e["kind"] != "goal_finished"]
    s = rx.build_session_summary("s", journal, {"rooms_new": 4, "deaths": 1}, kind="leisure", game_id="zelda",
                                 verdict_counts={"met": 5, "refuted": 2, "unknown": 1}, decisions=9)
    assert (s.decisions, s.predictions_met, s.predictions_refuted, s.predictions_unknown) == (9, 5, 2, 2)
    assert s.summary.endswith("4 new rooms, died once, 5 of 9 predictions met, 2 unknown.")


def _ended(**kw):
    """Lane B's real record (scripts/retro_session_end.py)."""
    from scripts.retro_session_end import RetroSessionEnded

    base = dict(session_id="retro-1", lineage_id="lin-1", leisure_id="l1", game_id="the-legend-of-zelda",
                started_at=1792699320.0, ended_at=1792700640.0, end_reason="session_cap", act_id="d:1",
                decisions_planned=31, verdicts={"met": 14, "refuted": 12, "unknown": 3}, rooms_new=9, deaths=2,
                controller_frames={"human": 0, "script": 0, "leisure": 79200, "none": 300})
    base.update(kw)
    return RetroSessionEnded(**base)


def test_a_leisure_session_end_maps_onto_the_builder() -> None:
    kw = rx.session_end_inputs(_ended())
    assert (kw["kind"], kw["session_id"], kw["dispatch_id"], kw["atlas_delta"]) == (
        "leisure", "retro-1", "d:1", {"rooms_new": 9, "deaths": 2})
    s = rx.build_session_summary(**kw)
    assert s.summary == ("Played The Legend of Zelda solo for 22 min, 9 new rooms, died twice, "
                         "14 of 31 predictions met, 5 unknown.")
    assert (s.control_basis, s.human_share, s.orion_frames) == ("frames", 0.0, 79200)  # 'none' drove nobody
    assert (s.started_at.isoformat(), s.end_reason) == ("2026-10-22T20:02:00+00:00", "session_cap")
    # Unknown counts are left out, never 0.
    bare = rx.build_session_summary(**rx.session_end_inputs(_ended(rooms_new=None, deaths=None)))
    assert bare.rooms_new is None and "rooms" not in bare.summary and "died" not in bare.summary


def test_a_human_who_may_have_played_is_never_orions_alone() -> None:
    held = rx.build_session_summary(**rx.session_end_inputs(_ended(human_may_be_in_control=True,
                                                                   controller_frames=None)))
    assert (held.control_basis, held.human_share, held.played_together) == ("unknown", None, True)
    assert held.summary.startswith("Played The Legend of Zelda together for 22 min (control share not recorded)")
    # Counted frames decide when Retro sent them, flag or not.
    counted = rx.build_session_summary(**rx.session_end_inputs(_ended(
        human_may_be_in_control=True, controller_frames={"human": 700, "script": 200, "leisure": 100, "none": 50})))
    assert (counted.human_share, counted.played_together) == (0.7, True)
    assert counted.summary.startswith("Played The Legend of Zelda together for 22 min (a human drove 70% of frames)")


def test_flagged_leisure_with_zero_counted_human_frames_is_never_solo() -> None:
    """Lane B sets the flag when a human acted or presence was unreadable; Retro can still count 0."""
    s = rx.build_session_summary(**rx.session_end_inputs(_ended(human_may_be_in_control=True)))
    assert "solo" not in s.summary and s.human_may_be_in_control is True
    assert s.summary.startswith("Played The Legend of Zelda for 22 min (Orion drove 100% of frames)")


def test_a_few_human_frames_never_read_as_orion_100_percent() -> None:
    s = rx.build_session_summary(**rx.session_end_inputs(_ended(
        controller="co_play", leisure_id=None, act_id=None,
        controller_frames={"human": 3, "script": 997, "leisure": 0, "none": 0})))
    assert "(Orion drove 99% of frames)" in s.summary


def test_a_co_play_handoff_is_its_own_row_with_no_dispatch() -> None:
    e = _ended(controller="co_play", leisure_id=None, act_id=None,
               controller_frames={"human": 300, "script": 700, "leisure": 0, "none": 0})
    kw = rx.session_end_inputs(e)
    assert (kw["kind"], kw["session_id"], kw["retro_session_id"], kw["dispatch_id"]) == (
        "co_play", "retro-1:1792699320.000", "retro-1", None)
    s = rx.build_session_summary(**kw)
    assert s.summary.startswith("Played The Legend of Zelda for 22 min (Orion drove 70% of frames)")
    card = rx.memory_card_for(s)
    assert card.anchors[0] == "retro_session:retro-1"


def test_no_decision_count_is_not_claimed() -> None:
    none = rx.build_session_summary(**rx.session_end_inputs({
        "session_id": "x", "game_id": "zelda", "started_at": 1792699320.0, "ended_at": 1792700640.0,
        "end_reason": "stop"}))
    assert none.decisions is None and none.summary == "Played The Legend of Zelda solo for 22 min."


@pytest.mark.asyncio
async def test_the_co_play_sinks_open_and_close_one_play_act(cards) -> None:
    # Regression (2026-10-23): co-play closed no act ("co-play has no play act"), and the gateway
    # that runs it had no bus, so 7 played handoffs left 0 rows in grammar_acts.
    pool, bus = _Pool(), _Bus()
    start = rx.make_session_start_sink(bus_provider=lambda: bus)
    sink = rx.make_session_end_sink(pool_provider=lambda: pool, bus_provider=lambda: bus)
    assert await start("retro-1", 1792699320.0) is True
    out = await sink(_ended(controller="co_play", leisure_id=None, act_id=None))
    assert out.state == "exported" and list(pool.rows) == ["retro-1:1792699320.000"]
    opened, closed = bus.events()
    assert (opened.event_kind, closed.event_kind) == ("act_opened", "act_closed")
    assert opened.act.act_id == closed.act.act_id
    assert (closed.act.actor, closed.act.verb, opened.act.expect.signal_id) == ("hub.retro_coplay", "play", "artifact_produced")
    assert opened.correlation_id == closed.correlation_id == "retro-1:1792699320.000"   # the row key
    assert opened.observed_at == datetime.fromtimestamp(1792699320.0, tz=timezone.utc)  # the handoff start
    assert closed.act.object_ref == "retro_session:retro-1"
    assert closed.act.outcome.value == 1.0 and closed.act.outcome.note == "decisions=31 met=14"


def test_game_title_by_id_segment() -> None:
    assert rx.game_title("the-legend-of-zelda") == rx.game_title("zelda-prg0") == "The Legend of Zelda"
    assert rx.game_title("nova-1") == "nova-1"


def test_a_replayed_goal_line_counts_once() -> None:
    journal = _journal(met=1, refuted=0, unknown=0) + [
        {"kind": "goal_finished", "decision_id": "d0", "verdict": "met"}]
    s = rx.build_session_summary("s", journal, DELTA, kind="leisure", game_id="zelda-prg0")
    assert (s.decisions, s.predictions_met) == (1, 1)


def test_unbounded_journal_is_not_summarized() -> None:
    with pytest.raises(ValueError, match="journal_incomplete"):
        _summary(end=False)


def test_no_decisions_and_unknown_game_render_plainly() -> None:
    s = rx.build_session_summary("s", _journal(met=0, refuted=0, unknown=0, minutes=0), {}, kind="leisure",
                                 game_id="nova-1", decisions=0)
    assert s.summary == "Played nova-1 solo for under a minute, no planner decisions."
    unrecorded = rx.build_session_summary("s", _journal(met=0, refuted=0, unknown=0, minutes=0), {},
                                          kind="leisure", game_id="nova-1")
    assert unrecorded.decisions is None and unrecorded.summary == "Played nova-1 solo for under a minute."


def test_card_is_an_active_retro_play_memory_linked_to_session_and_lineage() -> None:
    card = rx.memory_card_for(_summary())
    assert (card.provenance, card.trust_source, card.status, card.confidence) == (
        "retro_play", "retro_play", "active", "certain")
    assert rx.memory_card_for(_summary(human=0, orion=0)).confidence == "likely"  # share inferred, not counted
    assert card.anchors == ["retro_session:sess-1", "retro_lineage:lin-1"]
    assert card.title == "Played The Legend of Zelda on 2026-10-22"
    assert card.summary == _summary().summary and re.fullmatch(r"retro-play-sess-1-[0-9a-f]{8}", card.slug)
    assert card.subschema["retro_play"]["human_share"] == 0.0
    weird = rx.memory_card_for(rx.build_session_summary("A/B c:1", _journal(), DELTA, kind="leisure", game_id="zelda"))
    assert weird.slug.startswith("retro-play-a-b-c-1-")
    twin = rx.memory_card_for(rx.build_session_summary("a b/c 1", _journal(), DELTA, kind="leisure", game_id="zelda"))
    assert twin.slug != weird.slug  # different ids never share a slug


def test_lazy_ddl_and_migration_declare_the_same_columns() -> None:
    def columns(sql: str) -> list[str]:
        body = sql.split("(", 1)[1].rsplit(");", 1)[0] if ");" in sql else sql.split("(", 1)[1].rsplit(")", 1)[0]
        body = body.split("CREATE INDEX")[0]
        cols = []
        for part in re.split(r",\s*(?![^()]*\))", body):
            word = part.strip().split()[0] if part.strip() else ""
            if word and not word.startswith("--"):
                cols.append(word)
        return cols

    migration = "\n".join(l for l in MIGRATION.read_text().splitlines() if not l.startswith("--"))
    mig_cols = columns(migration)
    ddl_cols = columns(rx.DDL.split("CREATE INDEX")[0])
    assert len(mig_cols) == 32 and mig_cols[0] == "session_id" and mig_cols[-1] == "created_at"
    assert mig_cols == ddl_cols
    assert list(rx._COLUMNS) == [c for c in mig_cols if c not in ("memory_card_id", "card_error", "created_at")]
    assert "-- ORION-MIGRATION-REQUIRED-BY: orion-hub, orion-durable-runs" in MIGRATION.read_text()


# --- export ----------------------------------------------------------------------------------------

class _Bus:
    def __init__(self):
        self.sent = []

    async def publish(self, channel, envelope):
        self.sent.append((channel, envelope))

    def events(self):
        return [GrammarEventV1.model_validate(e.payload) for ch, e in self.sent if ch == "orion:grammar:event"]


class _Conn:
    def __init__(self, store):
        self.store = store

    def transaction(self):
        conn = self

        class _Tx:
            async def __aenter__(self):
                return conn

            async def __aexit__(self, *a):
                return False

        return _Tx()

    async def execute(self, sql, *args):
        self.store.executed.append((sql.strip().split()[0], args))

    async def fetchval(self, sql, *args):
        assert sql == rx.INSERT_SQL
        if args[0] in self.store.rows:
            return None
        self.store.rows[args[0]] = args
        return args[0]


class _Pool:
    def __init__(self):
        self.rows, self.executed = {}, []

    def acquire(self):
        pool = self

        class _Acq:
            async def __aenter__(self):
                return _Conn(pool)

            async def __aexit__(self, *a):
                return False

        return _Acq()


@pytest.fixture(autouse=True)
def _flags(monkeypatch):
    from scripts.settings import settings

    monkeypatch.setattr(settings, "HUB_ACT_LEDGER_ENABLED", True, raising=False)
    monkeypatch.setattr(settings, "HUB_RETRO_MEMORY_EXPORT_ENABLED", True, raising=False)
    monkeypatch.setattr(rx, "_table_ready", False)


@pytest.fixture
def cards(monkeypatch):
    from orion.core.storage import memory_cards as mc_dal

    made = []

    async def insert_card(pool, card, *, actor, op="create"):
        made.append((card, actor))
        return UUID("00000000-0000-4000-8000-000000000001")

    monkeypatch.setattr(mc_dal, "insert_card", insert_card)
    return made


async def _export(pool, bus, **kw):
    base = dict(pool=pool, bus=bus, session_id="sess-1", journal=_journal(), atlas_delta=DELTA, kind="leisure",
                game_id="zelda-prg0", lineage_id="lin-1", dispatch_id="dispatch:proposal:retro_leisure_session:t1")
    base.update(kw)
    return await rx.export_session(**base)


@pytest.mark.asyncio
async def test_export_claims_the_row_writes_one_card_links_it_and_closes_the_act(cards) -> None:
    pool, bus = _Pool(), _Bus()
    out = await _export(pool, bus)
    assert out.state == "exported" and out.card_id == "00000000-0000-4000-8000-000000000001"
    assert list(pool.rows) == ["sess-1"]
    assert [(c.provenance, a) for c, a in cards] == [("retro_play", "retro_memory_export")]
    assert [e for e in pool.executed if e[0] == "UPDATE"] == [
        ("UPDATE", ("sess-1", UUID("00000000-0000-4000-8000-000000000001"), None))]
    (ev,) = bus.events()
    assert ev.event_kind == "act_closed" and ev.act.verb == "play" and ev.act.actor == "hub.retro_leisure"
    assert ev.act.outcome.value == 1.0 and ev.act.test_before == "retro_leisure_dispatch:dispatch:proposal:retro_leisure_session:t1"
    # A replayed session end: no second card, no second close.
    again = await _export(pool, bus)
    assert again.state == "duplicate" and len(cards) == 1
    # The close is repeated with the same act id (the first may not have been delivered).
    assert [e.act.act_id for e in bus.events()] == [ev.act.act_id] * 2


@pytest.mark.asyncio
async def test_a_session_with_no_planner_decision_closes_its_act_at_zero(cards) -> None:
    bus = _Bus()
    out = await _export(_Pool(), bus, journal=_journal(met=0, refuted=0, unknown=0), decisions=0)
    assert out.state == "exported" and len(cards) == 1
    (ev,) = bus.events()
    assert ev.act.outcome.value == 0.0 and ev.act.outcome.note == "no_planner_decisions"


@pytest.mark.asyncio
async def test_co_play_closes_its_own_act_keyed_by_the_row(cards) -> None:
    bus = _Bus()
    out = await _export(_Pool(), bus, kind="co_play", dispatch_id=None, session_id="r:1.000", retro_session_id="r",
                        journal=_journal(human=60000, orion=40000))
    assert out.state == "exported" and out.summary.played_together
    (ev,) = bus.events()
    assert ev.act.actor == "hub.retro_coplay" and ev.correlation_id == "r:1.000"
    # The disabled switch still closes the co-play act at 0, like leisure.
    bus2 = _Bus()
    off = await _export(_Pool(), bus2, kind="co_play", dispatch_id=None, session_id="r:1.000", enabled=False)
    assert off.state == "disabled" and [e.act.outcome.value for e in bus2.events()] == [0.0]


@pytest.mark.asyncio
async def test_switch_off_writes_nothing_and_closes_the_act_at_zero(cards, monkeypatch) -> None:
    from scripts.settings import settings

    monkeypatch.setattr(settings, "HUB_RETRO_MEMORY_EXPORT_ENABLED", False, raising=False)
    pool, bus = _Pool(), _Bus()
    out = await _export(pool, bus)
    assert out.state == "disabled" and pool.rows == {} and cards == []
    (ev,) = bus.events()
    assert ev.act.outcome.value == 0.0 and ev.act.outcome.note == "export_disabled"


@pytest.mark.asyncio
async def test_an_incomplete_journal_writes_nothing(cards) -> None:
    pool, bus = _Pool(), _Bus()
    out = await _export(pool, bus, journal=_journal(end=False))
    assert out.state == "skipped" and pool.rows == {} and cards == []
    assert bus.events()[0].act.outcome.value == 0.0


@pytest.mark.asyncio
async def test_a_failed_card_is_recorded_on_the_row(monkeypatch) -> None:
    from orion.core.storage import memory_cards as mc_dal

    async def broken(*a, **k):
        raise RuntimeError("pg down")

    monkeypatch.setattr(mc_dal, "insert_card", broken)
    pool = _Pool()
    out = await _export(pool, _Bus())
    assert out.state == "exported" and out.card_id is None and "pg down" in out.reason
    (update,) = [e for e in pool.executed if e[0] == "UPDATE"]
    assert update[1][1] is None and "pg down" in update[1][2]
