"""Act ledger wiring in the grammar persist path (no Postgres).

Only freshly inserted act events reach the act ledger (a replayed batch must
not close or grade an act twice), the kill switch stops it, and the flags ship ON.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

import orion.grammar.act_ledger as act_ledger
import orion.grammar.ledger as ledger
from orion.grammar.acts import act_closed_event, act_opened_event, build_act


class _Session:
    def flush(self):
        pass


@pytest.fixture
def captured(monkeypatch):
    seen: list = []
    inserted: dict = {"ids": None}
    monkeypatch.setattr(ledger, "_upsert_trace", lambda s, e: None)
    monkeypatch.setattr(ledger, "_bulk_insert_derived", lambda s, e: None)
    monkeypatch.setattr(ledger, "_bulk_insert_events",
                        lambda s, evs: [e.event_id for e in evs] if inserted["ids"] is None else inserted["ids"])
    monkeypatch.setattr(act_ledger, "apply_act_events", lambda s, evs: seen.extend(evs) or len(evs))
    yield seen, inserted
    ledger.configure_act_ledger(True)


def _events():
    act = build_act(actor="thought.painting", verb="make", key="r", signal_id="artifact_produced",
                    horizon_sec=5400.0, basis="deadline")
    at = datetime(2026, 10, 16, tzinfo=timezone.utc)
    return [act_opened_event(act, source_service="orion-thought", source_component="t", at=at),
            act_closed_event(act, produced=True, source_service="orion-thought", source_component="t", at=at)]


def test_fresh_act_events_reach_the_act_ledger(captured):
    seen, _ = captured
    evs = _events()
    assert ledger.apply_grammar_trace_batch(_Session(), evs) == 2
    assert [e.event_kind for e in seen] == ["act_opened", "act_closed"]


def test_deduped_act_events_do_not(captured):
    seen, inserted = captured
    evs = _events()
    inserted["ids"] = [evs[0].event_id]  # the close was already persisted once
    ledger.apply_grammar_trace_batch(_Session(), evs)
    assert [e.event_kind for e in seen] == ["act_opened"]


def test_kill_switch_stops_materialisation(captured):
    seen, _ = captured
    ledger.configure_act_ledger(False)
    ledger.apply_grammar_trace_batch(_Session(), _events())
    assert seen == []


def test_flags_ship_on():
    from app.settings import Settings

    fields = Settings.model_fields
    assert fields["sql_writer_act_ledger_enabled"].default is True
    assert fields["sql_writer_act_resolver_interval_sec"].default > 0


def test_grammar_acts_table_is_registered_for_create_all():
    from app.db import Base
    from app.models import GrammarActSQL

    assert Base.metadata.tables["grammar_acts"] is GrammarActSQL.__table__
