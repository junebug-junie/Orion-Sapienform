"""Promise acts are on the act ledger in shadow: the self-monitor builds no bet from them (until #3531).

And a void close (censored: no reply existed to keep the promise in) is never a bet, live or shadow.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.grammar.promise_acts import VOID_NOTE
from orion.self_monitor import adapters
from orion.self_monitor.collect import SOURCES

NOW = datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc)


def _row(**kw):
    row = {"act_id": "act_p", "actor": "hub.promise", "verb": "say", "signal_id": "promise_kept",
           "object_ref": "juniper", "predicted": 0.5, "opened_at": NOW - timedelta(hours=80),
           "closed_at": NOW - timedelta(hours=8), "outcome_value": 0.0, "outcome_source": "expired",
           "outcome_note": "no keep within horizon"}
    row.update(kw)
    return row


def test_grammar_acts_source_excludes_shadow_signals_in_sql():
    sql = next(sql for name, sql, _ in SOURCES if name == "grammar_acts")
    assert "COALESCE(signal_id, '') <> ALL (ARRAY['promise_kept'])" in sql


def test_adapter_builds_no_bet_from_a_shadow_act_or_a_void_close():
    assert adapters.grammar_act(_row()) is None
    assert adapters.grammar_act(_row(signal_id="artifact_produced", actor="thought.painting",
                                     outcome_note=VOID_NOTE)) is None
    # Any other act still settles as before.
    bet = adapters.grammar_act(_row(signal_id="artifact_produced", actor="thought.painting", outcome_source="producer",
                                    outcome_value=1.0, outcome_note=""))
    assert bet is not None and bet.outcome == "upheld"
