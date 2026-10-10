"""Arc rules per lane (spec "Arc rules, stated so a test can pin them")."""

from __future__ import annotations

from datetime import timedelta

from orion.temporal_self import ReducerConfig, advance_clock, build_frame, fold, initial_state
from orion.temporal_self.tests.fixtures import T0, at, attention_row, chat, ev, metacog, run, ticks

CFG = ReducerConfig()
A, B = "node:substrate.bus_synaptic", "node:substrate.biometrics"


def arcs_of(state, kind):
    return sorted((a for a in state.arcs.values() if a.kind == kind), key=lambda a: a.began_at)


# ---------------------------------------------------------------- attention (broadcast)


def test_k_ticks_open_an_attention_arc_and_fewer_do_not():
    s = fold(initial_state(), ticks([A, A]), cfg=CFG)
    assert arcs_of(s, "attention") == []
    s = fold(initial_state(), ticks([A, A, A]), cfg=CFG)
    (arc,) = arcs_of(s, "attention")
    assert arc.subject_ref == A and arc.status == "open"
    assert arc.began_at == T0  # backdated to the first tick of the streak
    assert arc.cumulative_dwell_sec == 60.0
    assert len(arc.evidence_refs) == 3


def test_flicker_below_k_does_not_suspend_and_is_not_evidence():
    s = fold(initial_state(), ticks([A, A, A, B, B, A, A]), cfg=CFG)
    (arc,) = arcs_of(s, "attention")
    assert arc.status == "open" and arc.attention_returns == 0
    assert all(r.endswith(("L00000", "L00001", "L00002", "L00005", "L00006")) for r in arc.evidence_refs)
    # Dwell counts only time actually held: 0->60 s, then 150->180 s (B's ticks are not A's).
    assert arc.cumulative_dwell_sec == 90.0


def test_switch_suspends_records_interruption_and_return_within_r_resumes():
    s = fold(initial_state(), ticks([A] * 4 + [B] * 4 + [A] * 3), cfg=CFG)
    a_arc, b_arc = arcs_of(s, "attention")
    assert a_arc.subject_ref == A and b_arc.subject_ref == B
    assert a_arc.status == "open" and a_arc.attention_returns == 1
    assert a_arc.interruptions == [b_arc.arc_id]
    assert b_arc.status == "suspended" and b_arc.interruptions == [a_arc.arc_id]
    assert len(a_arc.segments) == 2
    assert a_arc.cumulative_dwell_sec == 90.0 + 60.0


def test_suspended_arc_closes_after_return_window():
    s = fold(initial_state(), ticks([A] * 3 + [B] * 3), cfg=CFG)
    a_arc = arcs_of(s, "attention")[0]
    s = advance_clock(s, a_arc.last_seen_at + timedelta(seconds=CFG.return_window_sec + 1), CFG)
    a_arc = s.arcs[a_arc.arc_id]
    assert a_arc.status == "closed" and a_arc.closed_reason == "return_window_expired"
    assert a_arc.ended_at == a_arc.last_seen_at
    # A later win opens a NEW arc, not a return.
    late = ticks([A] * 3, start=a_arc.last_seen_at + timedelta(hours=1), prefix="M")
    s = fold(s, late, cfg=CFG)
    assert len(arcs_of(s, "attention")) == 3


def test_no_winner_ticks_suspend_to_a_rest_state():
    s = fold(initial_state(), ticks([A] * 3 + [None] * 3), cfg=CFG)
    (arc,) = arcs_of(s, "attention")
    assert arc.status == "suspended" and arc.interruptions == []
    frame = build_frame(s, at(5), CFG)
    assert frame.active_arc is None
    assert frame.model_validate(frame.model_dump()) == frame


def test_quiet_window_yields_rest_frame_that_still_validates_and_moves_cursors():
    s = fold(initial_state(), ticks([None] * 20), cfg=CFG)
    frame = build_frame(s, at(15), CFG)
    assert frame.active_arc is None and frame.arcs_today == []
    assert "broadcast_tick" in frame.source_cursors
    assert all(a.attention_returns == 0 for a in s.arcs.values())


def test_unchanged_winner_all_day_is_one_arc_with_a_warning():
    s = fold(initial_state(), ticks([A] * 150), cfg=CFG)
    assert len(arcs_of(s, "attention")) == 1
    assert any("unchanged all day" in w for w in build_frame(s, at(80), CFG).warnings)


def test_label_change_on_the_same_ref_does_not_split_the_arc():
    tk = ticks([A] * 6)
    relabelled = [t.__class__(t.log_id, t.generated_at, t.ref, label=f"label {i}") for i, t in enumerate(tk)]
    s = fold(initial_state(), relabelled, cfg=CFG)
    assert len(arcs_of(s, "attention")) == 1


def test_broadcast_gap_suspends_without_accruing_dwell_and_resume_is_one_arc():
    before = ticks([A] * 4)
    after = ticks([A] * 3, start=before[-1].generated_at + timedelta(minutes=10), prefix="R")
    s = fold(initial_state(), before + after, cfg=CFG)
    (arc,) = arcs_of(s, "attention")
    assert s.tick_source_gaps_today == 1
    assert arc.attention_returns == 1
    assert arc.cumulative_dwell_sec == 90.0 + 60.0  # the 10 minute outage is not dwell


# ---------------------------------------------------------------- interoception


def test_run_below_minimum_streak_is_ignored():
    s = fold(initial_state(), events=[run("r1", A, T0, 1, ticks_=2, bar=3)], cfg=CFG)
    assert arcs_of(s, "interoception") == []


def test_interoception_returns_and_dwell_sum_over_runs():
    evs = [run("r1", A, at(0), 5), run("r2", B, at(5), 5), run("r3", A, at(10), 5)]
    s = fold(initial_state(), events=evs, cfg=CFG)
    a_arc, b_arc = arcs_of(s, "interoception")
    assert a_arc.subject_ref == A and a_arc.attention_returns == 1
    assert a_arc.cumulative_dwell_sec == 600.0
    assert a_arc.evidence_refs == ["field_dominance_run:r1", "field_dominance_run:r3"]
    assert a_arc.interruptions == [b_arc.arc_id]


def test_nine_hour_single_run_is_one_arc_with_stuck_warning():
    s = fold(initial_state(), events=[run("r1", A, at(0), 9 * 60, ticks_=16000)], cfg=CFG)
    (arc,) = arcs_of(s, "interoception")
    assert arc.warnings and "stuck" in arc.warnings[0]
    assert any("stuck" in w for w in build_frame(s, at(9 * 60), CFG).warnings)


# ---------------------------------------------------------------- conversation


def test_conversation_idle_suspends_and_return_within_three_hours_resumes():
    evs = [chat("1", "s1", at(0)), chat("2", "s1", at(5)), chat("3", "s1", at(120))]
    s = fold(initial_state(), events=evs, cfg=CFG)
    (arc,) = arcs_of(s, "conversation")
    assert arc.attention_returns == 1 and arc.cumulative_dwell_sec == 300.0
    assert len(arc.evidence_refs) == 3


def test_conversation_after_return_window_is_a_new_arc():
    s = fold(initial_state(), events=[chat("1", "s1", at(0)), chat("2", "s1", at(4 * 60))], cfg=CFG)
    first, second = arcs_of(s, "conversation")
    assert first.closed_reason == "return_window_expired" and second.attention_returns == 0


def test_chat_turn_without_session_opens_nothing():
    e = chat("1", "s1", at(0)).model_copy(update={"subject_ref": None})
    assert fold(initial_state(), events=[e], cfg=CFG).arcs == {}


# ---------------------------------------------------------------- concern


def test_concern_opens_on_chat_raise_and_closes_on_verdict():
    raised = ev("attention_loop_raised", "t1", at(0), subject="loop-1", table="attention_salience_trace", label="x")
    verdict = ev("attention_loop_verdict", "o1", at(30), subject="loop-1", table="attention_loop_outcome", verdict="resolved")
    s = fold(initial_state(), events=[raised, verdict], cfg=CFG)
    (arc,) = arcs_of(s, "concern")
    assert arc.status == "closed" and arc.closed_reason == "verdict"
    assert arc.evidence_refs == ["attention_salience_trace:t1", "attention_loop_outcome:o1"]


def test_verdict_for_a_loop_never_raised_in_chat_makes_no_arc():
    verdict = ev("attention_loop_verdict", "o1", at(30), subject="loop-9", table="attention_loop_outcome", verdict="decayed_unattended")
    assert fold(initial_state(), events=[verdict], cfg=CFG).arcs == {}


def test_concern_open_thread_in_frame():
    raised = ev("attention_loop_raised", "t1", at(0), subject="loop-1", table="attention_salience_trace", label="a worry")
    s = fold(initial_state(), events=[raised], cfg=CFG)
    (thread,) = build_frame(s, at(10), CFG).open_threads
    assert thread.subject_ref == "loop-1" and thread.subject_label == "a worry"


# ---------------------------------------------------------------- process arcs and binding


def test_process_arcs_are_born_closed_and_bind_context_inside_them():
    m = metacog("m1", at(10))
    cur = ev("curiosity_run", "run1", at(0), ended=at(30), subject="run1", table="curiosity_run_outcomes",
             related_refs=["prior:a"])
    s = fold(initial_state(), events=[m, cur], cfg=CFG)
    (arc,) = arcs_of(s, "curiosity")
    assert arc.status == "closed" and arc.closed_reason == "process_ended"
    assert arc.related_refs == ["prior:a"]
    assert arc.context_event_ids == ["metacog_observation:m1"]
    assert arc.cumulative_dwell_sec == 1800.0


def test_long_process_is_flagged():
    cur = ev("curiosity_run", "run1", at(0), ended=at(20 * 60), subject="run1", table="curiosity_run_outcomes")
    (arc,) = arcs_of(fold(initial_state(), events=[cur], cfg=CFG), "curiosity")
    assert arc.warnings


def test_context_outside_every_arc_lands_in_the_day():
    s = fold(initial_state(), events=[metacog("m1", at(0))], cfg=CFG)
    s = advance_clock(s, at(7 * 60), CFG)
    assert "metacog_observation:m1" in s.day_context_event_ids
    assert "metacog_observation:m1" in build_frame(s, at(7 * 60), CFG).unbound_context_event_ids


def test_context_binds_to_every_lane_open_at_that_time():
    s = fold(initial_state(), ticks([A] * 4), [chat("1", "s1", at(0)), metacog("m1", at(1))], cfg=CFG)
    for kind in ("attention", "conversation"):
        assert arcs_of(s, kind)[0].context_event_ids == ["metacog_observation:m1"]


def test_context_during_a_broadcast_outage_does_not_bind_to_the_silent_arc():
    s = fold(initial_state(), ticks([A] * 4), [metacog("m1", at(30))], cfg=CFG)
    assert arcs_of(s, "attention")[0].context_event_ids == []


def test_attention_rows_fold_into_lane_counts_not_context():
    rows = [attention_row(f"a{i}", at(0.5 + i * 0.1)) for i in range(3)]
    s = fold(initial_state(), ticks([A] * 5), rows, cfg=CFG)
    (arc,) = arcs_of(s, "attention")
    assert arc.attention.rows_by_lane == {"substrate_attention": 3}
    assert arc.attention.reasons_by_lane == {"substrate_attention": ["bottom_up_salience"]}
    assert arc.context_event_ids == []


def test_reverie_attention_rows_bind_by_correlation_not_by_time():
    own = attention_row("a1", at(1), process="reverie", reason="self_generated", corr="thought-corr-1")
    other = attention_row("a2", at(1), process="reverie", reason="self_generated", corr="someone-else")
    chain = ev("reverie_chain", "ch1", at(0), ended=at(2), subject="ch1", table="substrate_reverie_chain",
               related_refs=["thought-corr-1"], payload={"thought_ids": ["t1"]})
    s = fold(initial_state(), events=[own, other, chain], cfg=CFG)
    (arc,) = arcs_of(s, "reverie")
    assert arc.attention.rows_by_lane == {"reverie": 1}
    assert arc.evidence_refs == ["substrate_reverie_chain:ch1", "substrate_reverie_thought:t1"]


def test_retro_and_forward_binding_never_double_count():
    rows = [attention_row("a1", at(1)), attention_row("a2", at(4))]
    evs = rows + [run("r1", A, at(0), 2), run("r2", A, at(2), 3)]
    (arc,) = arcs_of(fold(initial_state(), events=evs, cfg=CFG), "interoception")
    assert arc.attention.rows_by_lane == {"substrate_attention": 2}


# ---------------------------------------------------------------- late evidence


def test_late_reverie_verdict_attaches_by_id_and_does_not_reopen():
    chain = ev("reverie_chain", "ch1", at(0), ended=at(1), subject="ch1", table="substrate_reverie_chain",
               payload={"thought_ids": ["t1"]})
    verdict = ev("expectation_verdict", "t1", at(45), table="substrate_reverie_thought", verdict="confirmed",
                 related_refs=["ch1"], payload={"committed_at": at(0).isoformat()})
    s = fold(initial_state(), events=[chain, verdict], cfg=CFG)
    (arc,) = arcs_of(s, "reverie")
    assert arc.status == "closed" and arc.ended_at == at(1)
    assert arc.expectation_event_ids == ["expectation_verdict:t1"]
    frame = build_frame(s, at(50), CFG)
    assert [x.verdict for x in frame.expectations_resolved_today] == ["confirmed"]
    assert "expectation_verdict:t1" in frame.self_change_event_ids


def test_dream_hypothesis_attaches_to_its_sleep_arc_by_cycle_and_is_pending_until_expiry():
    cycle = ev("dream_cycle", "cy1", at(0), ended=at(1), subject="cy1", table="dream_cycle")
    hyp = ev("dream_hypothesis", "h1", at(1), table="dream_hypothesis", related_refs=["cy1"],
             payload={"expires_at": at(72 * 60).isoformat()})
    s = fold(initial_state(), events=[cycle, hyp], cfg=CFG)
    (arc,) = arcs_of(s, "sleep")
    assert arc.expectation_event_ids == ["dream_hypothesis:h1"]
    frame = build_frame(s, at(2), CFG)
    assert [x.event_id for x in frame.expectations_pending] == ["dream_hypothesis:h1"]
    assert frame.sleep_arc_ids == [arc.arc_id]


def test_constraints_reach_the_frame_and_the_arc_open_at_the_time():
    wait = ev("gpu_wait", "g1", at(1), table="gpu_pool_events", verdict="granted", payload={"waited_ms": 900.0})
    s = fold(initial_state(), ticks([A] * 4), [wait], cfg=CFG)
    assert arcs_of(s, "attention")[0].constraint_event_ids == ["gpu_wait:g1"]
    assert build_frame(s, at(2), CFG).constraint_event_ids == ["gpu_wait:g1"]


def test_evidence_and_context_caps_count_overflow():
    s = fold(initial_state(), ticks([A] * 300), [metacog(f"m{i}", at(0.1 + i * 0.01)) for i in range(70)], cfg=CFG)
    (arc,) = arcs_of(s, "attention")
    assert len(arc.evidence_refs) == 256 and arc.evidence_overflow == 44
    assert len(arc.context_event_ids) == 64 and arc.context_overflow == 6


def test_frame_active_and_previous_arcs():
    s = fold(initial_state(), ticks([A] * 3 + [B] * 3), [chat("1", "s1", at(0))], cfg=CFG)
    frame = build_frame(s, at(3), CFG)
    assert frame.active_arc.subject_ref == B
    assert frame.previous_arc.subject_ref == A
    assert set(frame.active_by_kind) == {"attention", "conversation"}


def test_unscored_reverie_verdict_is_resolved_but_not_a_self_change():
    v = ev("expectation_verdict", "t9", at(5), table="substrate_reverie_thought", verdict="unscored",
           payload={"committed_at": at(0).isoformat()})
    frame = build_frame(fold(initial_state(), events=[v], cfg=CFG), at(6), CFG)
    assert [x.verdict for x in frame.expectations_resolved_today] == ["unscored"]
    assert frame.self_change_event_ids == [] and frame.expectations_resolved_total == 1
