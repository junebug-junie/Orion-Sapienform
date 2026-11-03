"""Felt state: binding arcs from the physiology node's body_binding rows (adapter, reducer, stance
cue, Day letter, recent arcs, citation truth).

T0 is 09:00 local (America/Denver, MDT, UTC-6) on 2026-10-09; the local day runs 06:00Z 10-09 to
06:00Z 10-10. Rows are shaped exactly as ``services/orion-durable-runs/app/body_binding.py``
``binding_row`` writes them (``services/orion-durable-runs/tests/test_body_binding.py`` checks that
the producer's rows go through this adapter).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from orion.schemas.temporal_self import STANCE_CUE_MAX_CHARS
from orion.temporal_self import sources as S
from orion.temporal_self.arcs import advance_clock, drain_closed_days, fold, initial_state
from orion.temporal_self.day_block import render_closed_day_block
from orion.temporal_self.frame import build_frame
from orion.temporal_self.recent_arcs import project_recent_arcs
from orion.temporal_self.stance_cue import project_stance_cue
from orion.temporal_self.tests.fixtures import T0, at

HEAT = dict(unit="c", level=32.4, line=31.0, ceiling=1.0)
LANE = dict(unit="slot_s", level=4100.0, line=3600.0, ceiling=1.0)


def row(kind: str, constraint: str, t: datetime, began: datetime, state: str = "binding", price: float = 0.5,
        slack_since: datetime | None = None, **reading) -> dict:
    r = {**(HEAT if constraint == "heat" else LANE), **reading}
    payload = {"row": kind, "constraint": constraint, "bind_state": state, "began_at": began.isoformat(),
               "slack_since": slack_since.isoformat() if slack_since else None, "unit": r["unit"],
               "level": r["level"], "line": r["line"], "price": price, "ceiling": r["ceiling"],
               "why": f"{constraint} why", "self_spent": constraint == "attention:juniper"}
    return {"event_id": f"body_binding:{constraint}:{began.isoformat()}:{kind}:{t.isoformat()}", "occurred_at": t,
            "payload_json": json.dumps(payload)}


def evs(rows):
    return S.adapt("body_binding", rows)


def bindings(state, constraint=None):
    return sorted((a for a in state.arcs.values() if a.kind == "binding"
                   and (constraint is None or a.subject_label == constraint)), key=lambda a: (a.began_at, a.arc_id))


def _frame(rows, now):
    return build_frame(advance_clock(fold(initial_state(), events=evs(rows)), now), now)


# --------------------------------------------------------------------------- adapter


def test_adapter_keeps_the_rows_own_event_id_and_rejects_bad_rows():
    r = row("onset", "lane:agent", at(4), at(0))
    e = S.body_binding(r)
    # The chronicle stores folded events in the same table: a new id would duplicate the row forever.
    assert e.event_id == r["event_id"]
    assert e.source_kind == "body_binding" and e.source_table == "physiology"
    assert e.subject_ref == "orion:body:lane:agent" and e.label == "lane:agent" and e.verdict == "onset"
    assert e.payload["began_at"] == at(0).isoformat() and e.payload["level"] == 4100.0
    assert S.body_binding(dict(r, event_id="arousal_transition:x")) is None
    bad_state = json.loads(r["payload_json"]) | {"bind_state": "slack"}
    assert S.body_binding(dict(r, payload_json=json.dumps(bad_state))) is None   # only a release may be slack
    bad_kind = json.loads(r["payload_json"]) | {"row": "maybe"}
    assert S.body_binding(dict(r, payload_json=json.dumps(bad_kind))) is None
    release = row("release", "lane:agent", at(30), at(0), state="slack", slack_since=at(26))
    assert S.body_binding(release).payload["slack_since"] == at(26).isoformat()


# --------------------------------------------------------------------------- reducer


def test_one_arc_per_constraint_several_open_at_once():
    s = fold(initial_state(), events=evs([row("onset", "heat", at(4), at(0)),
                                          row("onset", "lane:agent", at(10), at(6))]))
    (heat,) = bindings(s, "heat")
    (lane,) = bindings(s, "lane:agent")
    assert heat.status == lane.status == "open" and heat.arc_id != lane.arc_id
    assert heat.episode_began_at == at(0) and heat.began_at == at(4)
    assert heat.binding.level == 32.4 and heat.binding.line == 31.0 and heat.binding.bind_state == "binding"
    assert s.active["binding:heat"] == heat.arc_id and s.active["binding:lane:agent"] == lane.arc_id
    frame = build_frame(s, at(12))
    assert frame.active_by_kind["binding:heat"].binding.constraint == "heat"


def test_change_and_keepalive_refresh_release_closes_at_the_first_slack_reading():
    rows = [row("onset", "heat", at(4), at(0)),
            row("change", "heat", at(20), at(0), state="stuck_at_ceiling", price=1.0, level=33.1),
            row("keepalive", "heat", at(50), at(0), state="stuck_at_ceiling", price=1.0),
            row("release", "heat", at(70), at(0), state="slack", slack_since=at(66))]
    s = fold(initial_state(), events=evs(rows[:2]))
    (arc,) = bindings(s)
    assert arc.binding.bind_state == "stuck_at_ceiling" and arc.binding.level == 33.1
    s = advance_clock(fold(s, events=evs(rows[2:])), at(80))
    (arc,) = bindings(s)
    assert arc.status == "closed" and arc.closed_reason == "process_ended" and arc.ended_at == at(66)
    assert len(arc.evidence_refs) == 4 and "binding:heat" not in s.active


def test_silence_closes_at_the_last_row_and_a_stray_release_does_nothing():
    s = fold(initial_state(), events=evs([row("release", "heat", at(0), at(-30), state="slack", slack_since=at(-4))]))
    assert bindings(s) == []
    rows = [row("onset", "heat", at(4), at(0)), row("keepalive", "heat", at(34), at(0))]
    s = advance_clock(fold(initial_state(), events=evs(rows)), at(34 + 74))
    (arc,) = bindings(s)
    assert arc.status == "open"
    s = advance_clock(s, at(34 + 76))     # no_reading for 76 min: no row past 4,500 s
    (arc,) = bindings(s)
    assert arc.closed_reason == "source_stale" and arc.ended_at == at(34) and "binding:heat" not in s.active


def test_a_row_from_a_new_episode_closes_the_open_arc_at_its_last_row():
    rows = [row("onset", "heat", at(4), at(0)), row("keepalive", "heat", at(34), at(0)),
            row("onset", "heat", at(200), at(196))]
    s = fold(initial_state(), events=evs(rows))
    old, new = bindings(s)
    assert old.closed_reason == "source_stale" and old.ended_at == at(34)
    assert new.status == "open" and new.episode_began_at == at(196)


def test_binding_arcs_bind_no_context():
    from orion.temporal_self.arcs import ReducerConfig, _covers

    s = fold(initial_state(), events=evs([row("onset", "heat", at(4), at(0)), row("keepalive", "heat", at(34), at(0))]))
    (arc,) = bindings(s)
    assert arc.segments[-1].began_at <= at(10) <= arc.segments[-1].ended_at
    assert not _covers(s, arc, at(10), ReducerConfig())


def test_replay_identity_fold_twice_and_chunked():
    rows = [row("onset", "heat", at(4), at(0)), row("onset", "lane:agent", at(9), at(5)),
            row("change", "heat", at(20), at(0), state="exogenous_bind"),
            row("release", "lane:agent", at(40), at(5), state="slack", slack_since=at(36)),
            row("keepalive", "heat", at(50), at(0), state="exogenous_bind")]
    e = evs(rows)
    end = at(120)
    whole = advance_clock(fold(initial_state(), events=e), end)
    again = advance_clock(fold(fold(initial_state(), events=e), events=e), end)
    assert again.arcs == whole.arcs
    s = initial_state()
    for lo in range(0, 120, 15):
        s = advance_clock(fold(s, events=[x for x in e if at(lo) <= x.occurred_at < at(lo + 15)]), at(lo + 15))
    assert s.arcs == whole.arcs and build_frame(s, end) == build_frame(whole, end)


def test_episode_carries_over_midnight_with_its_real_start_and_state():
    late = datetime(2026, 10, 10, 5, 0, tzinfo=timezone.utc)  # 23:00 local 10-09
    midnight = datetime(2026, 10, 10, 6, 0, tzinfo=timezone.utc)
    rows = [row("onset", "lane:agent", late, late - timedelta(minutes=4)),
            row("keepalive", "lane:agent", late + timedelta(minutes=30), late - timedelta(minutes=4)),
            row("keepalive", "lane:agent", midnight + timedelta(minutes=1), late - timedelta(minutes=4))]
    s = advance_clock(fold(initial_state(), events=evs(rows)), midnight + timedelta(minutes=2))
    s, days = drain_closed_days(s)
    (old,) = [a for a in days[0].arcs if a.kind == "binding"]
    assert old.closed_reason == "day_boundary" and old.ended_at == midnight
    (cont,) = bindings(s)
    assert cont.status == "open" and cont.carried_from_arc_id == old.arc_id
    assert cont.episode_began_at == late - timedelta(minutes=4) and cont.binding.constraint == "lane:agent"
    assert s.active["binding:lane:agent"] == cont.arc_id
    block = render_closed_day_block(days[0])
    assert "Body limits binding: my agent GPU lane full 22:56 still binding at midnight [arc:" in block.text


# --------------------------------------------------------------------------- stance cue


def test_cue_names_open_constraints_by_price_with_level_line_and_citation():
    now = at(30)
    rows = [row("onset", "lane:agent", at(10), at(6), price=0.3), row("onset", "heat", at(4), at(0), price=0.8)]
    cue = project_stance_cue(_frame(rows, now), now)
    heat = [a for a in cue.arc_ids][-2]
    assert (f"Binding now: cabinet running hot (32.4 °C, line 31 °C) since 09:00 [arc:{heat}]; "
            "my agent GPU lane full since 09:06 [arc:") in cue.text
    assert len(cue.text) <= STANCE_CUE_MAX_CHARS
    assert cue.labels == {}  # constraint ids are not in the text, so they are no match label


def test_cue_names_at_most_two_and_out_of_hands_trails():
    now = at(30)
    rows = [row("onset", "heat", at(4), at(0), price=0.9), row("onset", "lane:agent", at(5), at(1), price=0.6),
            row("onset", "lane:chat", at(6), at(2), price=0.2),
            row("onset", "claude", at(7), at(3), state="exogenous_bind", price=1.0, unit="limit_events", level=1.0,
                line=0.0),
            row("onset", "energy", at(8), at(3), state="exogenous_bind", price=1.0, unit="usd", level=8.4, line=100.0)]
    cue = project_stance_cue(_frame(rows, now), now, max_chars=2000)
    clause = cue.text[cue.text.index("Binding now"):]
    assert clause.startswith("Binding now: cabinet running hot")
    assert "my agent GPU lane full" in clause and "lane:chat" not in clause and "chat GPU lane" not in clause
    assert "; and 2 more (out of my hands: Claude limit reached [arc:" in clause
    # Energy's exogenous bind is the always-on body (Juniper, 10-30): never "out of my hands".
    assert "electricity" not in clause.split("out of my hands")[1]


def test_energy_alone_is_named_as_my_always_on_body():
    now = at(30)
    cue = project_stance_cue(_frame([row("onset", "energy", at(4), at(0), state="exogenous_bind", price=1.0,
                                         unit="usd", level=8.4, line=100.0)], now), now)
    assert "Binding now: my always-on body alone over this month's electricity allowance since 09:00 [arc:" in cue.text
    assert "out of my hands" not in cue.text


def test_cue_past_tense_within_6_h_then_nothing():
    rows = [row("onset", "heat", at(4), at(0)), row("release", "heat", at(44), at(0), state="slack", slack_since=at(40))]
    now = at(40 + 5 * 60)
    cue = project_stance_cue(_frame(rows, now), now)
    assert "Was binding 09:00 to 09:40: cabinet running hot [arc:" in cue.text
    now = at(40 + 6 * 60 + 1)
    cue = project_stance_cue(_frame(rows, now), now)
    assert "inding" not in cue.text
    cue = project_stance_cue(_frame([], at(30)), at(30))
    assert "inding" not in cue.text


def test_binding_clause_checks_as_true_against_its_own_arc():
    from orion.temporal_self.citation_truth import ArcFacts, check_decision

    now = at(30)
    s = advance_clock(fold(initial_state(), events=evs([row("onset", "heat", at(4), at(0))])), now)
    (arc,) = bindings(s)
    facts = {arc.arc_id: ArcFacts.from_row(arc.model_dump(mode="json"))}
    good = f"Binding now: cabinet running hot since 09:00 [arc:{arc.arc_id}]."
    checks, _ = check_decision(at=now, cited_valid=[], texts={"reply": good}, arcs=facts)
    assert [c.outcome for c in checks] == ["match"]
    wrong = f"Binding now: cabinet running hot since 07:30 [arc:{arc.arc_id}]."
    checks, _ = check_decision(at=now, cited_valid=[], texts={"reply": wrong}, arcs=facts)
    assert [c.outcome for c in checks] == ["mismatch"]


def test_citation_phrase_ignores_ordinary_binding():
    from orion.temporal_self.citation_truth import KIND_PHRASES

    pats = [p for p, k in KIND_PHRASES if k == "binding"]
    assert any(p.search("Binding now: cabinet running hot") for p in pats)
    assert any(p.search("Was binding 09:00 to 09:40: cabinet running hot") for p in pats)
    assert not any(p.search("a binding contract for the lane") for p in pats)


def test_recent_arcs_names_a_closed_binding():
    rows = [row("onset", "lane:agent", at(4), at(0)),
            row("release", "lane:agent", at(44), at(0), state="slack", slack_since=at(40))]
    now = at(60)
    assert "binding: my agent GPU lane full" in project_recent_arcs(_frame(rows, now), now).text


def test_juniper_words_say_pace_until_the_cap_is_reached():
    from orion.schemas.temporal_self import ArcBindingV1
    from orion.temporal_self.stance_cue import binding_words

    b = ArcBindingV1(constraint="attention:juniper", bind_state="binding", unit="messages", level=1.0, line=4.0,
                     self_spent=True)
    assert binding_words(b) == "messages to Juniper over today's pace (1 of 4 sent)"
    assert binding_words(b.model_copy(update={"level": 4.0})) == "used today's messages to Juniper (4 of 4 sent)"
