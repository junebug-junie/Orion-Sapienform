"""#3595 (Orion's 11-17 round-up on PR #3532): one 72 h clock for a promise, and one settled bet per promise.

Three readers used to carry their own 72 h: the promise act (orion/grammar/promise_acts.py), Prime's promise concern
life (orion/prime/state.py) and the carry turn's horizon (orion/prime/intention_sources/turn_promise.py). They now
import one constant (orion/schemas/promise_horizon.py); these tests fail if any of them drifts or grows its own
literal again.

One bet: a promise lives in two records, Prime's concern and the ledger's act. Only the act can become a settled bet
in the self-monitor (and only once the ledger leaves shadow). Prime's turn-opened concern places no bet
(``no_bet_concern_ids``), so the concern closing and the act closing never count the same promise twice.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import orion.grammar.promise_acts as pa
import orion.prime.handback as hb
import orion.self_monitor.adapters as sm
from orion.prime.intention_sources import turn_promise
from orion.prime.state import PROMISE_CONCERN_TTL, PrimeStateV1
from orion.schemas.cockpit_sighting import CockpitHopV1
from orion.schemas.intention import IntentionSourceV1
from orion.schemas.promise_horizon import PROMISE_HORIZON_SEC

REPO = Path(__file__).resolve().parents[3]
T0 = datetime(2026, 11, 17, 12, 0, tzinfo=timezone.utc)


def test_the_three_promise_clocks_are_one_constant():
    assert PROMISE_HORIZON_SEC == 72 * 3600.0
    assert pa.PROMISE_HORIZON_SEC is PROMISE_HORIZON_SEC
    assert turn_promise.HORIZON_SEC is PROMISE_HORIZON_SEC
    assert PROMISE_CONCERN_TTL == timedelta(seconds=PROMISE_HORIZON_SEC)
    act = pa.build_promise_act(pa.promise_ref("c1", 0), parent_act_id=None)
    assert act.expect.horizon_sec == PROMISE_HORIZON_SEC


def test_the_carry_horizon_and_the_act_horizon_end_at_the_same_instant():
    corr = "a129b72f-e698-425a-a435-3388cd7cd759"
    row = {"promised_at": T0, "promises_made": [{"quote": "I'll read the three proposals closely.", "offer": False}],
           "session_id": "9c91d646", "prompt": "hi", "response": "I'll read the three proposals closely."}
    src = turn_promise.adapt(f"turn:{corr}", row, why="it is due")
    assert isinstance(src, IntentionSourceV1)
    assert src.horizon_at == T0 + timedelta(seconds=pa.build_promise_act(src.quote_ref, parent_act_id=None)
                                            .expect.horizon_sec)


def test_no_reader_keeps_its_own_72h_literal():
    pattern = re.compile(r"72(?:\.0+)?\s*\*\s*(?:3600|60\s*\*\s*60)|hours\s*=\s*72(?:\.0+)?\b"
                         r"|days\s*=\s*3(?:\.0+)?\b|259200|4320\s*\*\s*60")
    for rel in ("orion/grammar/promise_acts.py", "orion/prime/state.py",
                "orion/prime/intention_sources/turn_promise.py", "orion/harness/evals/promise_ledger_replay_eval.py",
                "scripts/report_promise_ledger.py"):
        text = (REPO / rel).read_text()
        assert not pattern.search(text), f"{rel} carries its own 72 h; import PROMISE_HORIZON_SEC instead"


def _hops(corr: str) -> list[dict]:
    ingress = {"session_id": "9c91d646", "user_message": "read them when they land",
               "mode": "orion", "surface_context": {"surface": "hub_desktop", "input_modality": "typed"}}
    fin = {"final_text": "I'll read the three proposals closely when they appear.", "compliance_verdict": "completed",
           "reflection": {"promises_made": [{"offer": False,
                                             "quote": "I'll read the three proposals closely when they appear."}],
                          "promises_kept": []}}
    return [CockpitHopV1(correlation_id=corr, seq=0, ts=T0, stage="ingress", visor_line="ingress", status="ok",
                         summary={}, raw=ingress, producer="orion-hub").model_dump(mode="json"),
            CockpitHopV1(correlation_id=corr, seq=9, ts=T0 + timedelta(minutes=1), stage="finalize",
                         visor_line="finalize", status="ok", summary={}, raw=fin,
                         producer="orion-hub").model_dump(mode="json")]


def _act_close_row(act_id: str) -> dict:
    return {"act_id": act_id, "actor": pa.PROMISE_ACTOR, "verb": pa.PROMISE_VERB, "object_ref": pa.PROMISE_OBJECT_REF,
            "signal_id": pa.PROMISE_SIGNAL, "opened_at": T0, "closed_at": T0 + timedelta(hours=30),
            "outcome_source": "producer", "outcome_value": 1.0, "outcome_note": "kept_in:c2 evidence=reply",
            "predicted": None}


def test_one_promise_closing_both_ways_yields_at_most_one_settled_bet(monkeypatch):
    corr = "9f846e8e-0000-4000-8000-000000000001"
    signals = [s for s in (hb.signal_from_hop(h) for h in _hops(corr)) if s is not None]
    state, turns, folded = hb.fold_turn_signals(PrimeStateV1(), hb.PrimeTurnsStateV1(), signals,
                                                T0 + timedelta(minutes=2))
    (turn,) = folded
    cid = turn.opened_concern_id
    assert cid is not None
    # Prime's record of the promise is never a bet: the runtime places none for a turn-opened concern.
    assert cid in turns.no_bet_concern_ids
    # Both records name the same promise by the same ref.
    assert turn.promise_refs == [pa.promise_ref(corr, 0)]
    act_id = pa.promise_act_id(turn.promise_refs[0])
    # In shadow the act is no bet either.
    assert sm.grammar_act(_act_close_row(act_id)) is None
    # After go-live (the shadow set emptied) the act is exactly one bet, the only one for this promise.
    monkeypatch.setattr(pa, "SHADOW_ACT_SIGNALS", frozenset())
    bet = sm.grammar_act(_act_close_row(act_id))
    assert bet is not None and bet.bet_id == f"grammar_acts:{act_id}"


def test_the_literal_guard_catches_the_usual_spellings():
    pattern = re.compile(r"72(?:\.0+)?\s*\*\s*(?:3600|60\s*\*\s*60)|hours\s*=\s*72(?:\.0+)?\b"
                         r"|days\s*=\s*3(?:\.0+)?\b|259200|4320\s*\*\s*60")
    for spelled in ("72 * 3600.0", "72.0 * 3600", "72 * 60 * 60", "timedelta(hours=72)", "timedelta(hours=72.0)",
                    "timedelta(days=3)", "259200.0", "4320 * 60"):
        assert pattern.search(spelled), spelled
    assert not pattern.search("timedelta(hours=24)") and not pattern.search("days=30")
