"""Source adapters: filters, subjects, casts, and no prose."""

from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path

import pytest

from orion.schemas.registry import resolve
from orion.schemas.temporal_self import TemporalSelfEventV1, TemporalSelfFrameV1
from orion.temporal_self import sources as S
from orion.temporal_self.body import summarize_body
from orion.temporal_self.broadcast import tick_from_log_row

T = datetime(2026, 10, 9, 15, 0, tzinfo=timezone.utc)


def test_gpu_wait_uses_the_admission_cue_predicate_exactly():
    base = {"event_id": "g", "generated_at": T, "priority": "background", "holder": "orion-mind"}
    assert S.gpu_wait({**base, "event": "granted", "waited_ms": 500.0}) is not None
    assert S.gpu_wait({**base, "event": "granted", "waited_ms": 499.9}) is None
    assert S.gpu_wait({**base, "event": "unavailable", "waited_ms": None}).verdict == "unavailable"
    assert S.gpu_wait({**base, "event": "granted", "waited_ms": 900.0, "holder": "http:town"}) is None
    assert S.gpu_wait({**base, "event": "granted", "waited_ms": 900.0, "priority": "interactive"}) is None


def test_metacog_only_degraded_or_critical():
    assert S.metacog_observation({"id": "m", "severity": "nominal", "timestamp": T.isoformat()}) is None
    e = S.metacog_observation({"id": "m", "severity": "critical", "timestamp": T.isoformat(), "trigger_kind": "transport",
                               "summary": "PROSE", "mantra": "PROSE"})
    assert e.verdict == "critical" and "PROSE" not in e.model_dump_json()


def test_concern_raise_requires_a_real_chat_turn():
    row = {"trace_id": "t", "loop_id": "l", "scope": "chat", "created_at": T, "description": "her words"}
    assert S.attention_loop_raised(row) is None  # scorer re-emission, not a raise
    e = S.attention_loop_raised({**row, "chat_turn": True})
    assert e.subject_ref == "l" and e.privacy_class == "juniper_chat"
    assert S.attention_loop_raised({**row, "chat_turn": True, "scope": "reverie"}) is None


def test_chat_turn_copies_no_text():
    e = S.chat_turn({"id": "1", "session_id": "s", "created_at": T, "prompt": "SECRET", "response": "SECRET",
                     "correlation_id": "c"})
    assert e.subject_ref == "s" and "SECRET" not in e.model_dump_json()


def test_incomplete_processes_are_not_emitted():
    assert S.curiosity_run({"run_id": "r", "turn_started_at": T, "completed_at": None}) is None
    assert S.dream_cycle({"cycle_id": "c", "status": "running", "started_at": T, "ended_at": None}) is None
    assert S.reverie_chain({"chain_id": "c", "created_at": T, "thoughts": [], "terminal_reason": "x"}) is None


def test_curiosity_run_carries_offered_priors_and_its_interval():
    e = S.curiosity_run({"run_id": "r", "turn_started_at": T, "completed_at": T.replace(hour=16),
                         "offered": [{"prior_id": "b"}, {"prior_id": "a"}]})
    assert e.related_refs == ["a", "b"] and e.ended_at.hour == 16 and e.subject_ref == "r"


def test_reverie_chain_window_and_thought_correlations():
    row = {"chain_id": "c", "created_at": T, "theme_key": "open-loop-x",
           "thoughts": [{"thought_id": "t2", "created_at": T.replace(minute=2), "correlation_id": "k2"},
                        {"thought_id": "t1", "created_at": T.replace(minute=1), "correlation_id": "k1"}]}
    assert S.reverie_chain(row) is None  # still running: no terminal_reason yet
    e = S.reverie_chain({**row, "terminal_reason": "max_steps"})
    assert e.occurred_at.minute == 1 and e.ended_at.minute == 2
    assert e.payload["thought_ids"] == ["t1", "t2"] and e.related_refs == ["k1", "k2"]


def test_visual_deferral_keeps_the_sources_own_word():
    assert S.visual_deferral({"attempt_id": "a", "started_at": T, "outcome": "produced"}) is None
    e = S.visual_deferral({"attempt_id": "a", "started_at": T, "outcome": "abandoned",
                           "result_json": {"reason": "run_abandoned", "detail": {"state": "normal"}}})
    assert e.verdict == "abandoned" and e.payload["reason"] == "run_abandoned"


def test_vision_percept_room_only_entities_only_no_narrative():
    row = {"event_id": "v", "stream_id": "cam0", "created_at": T, "entities": ["person"], "narrative": "PROSE"}
    assert S.vision_percept(row).payload["entities"] == ["person"]
    assert "PROSE" not in S.vision_percept(row).model_dump_json()
    assert S.vision_percept({**row, "entities": []}) is None
    assert S.vision_percept({**row, "stream_id": "walkway"}) is None


def test_dream_hypothesis_and_episode_copy_no_prose():
    h = S.dream_hypothesis({"hypothesis_id": "h", "cycle_id": "c", "created_at": T, "claim": "PROSE", "why": "PROSE",
                            "expires_at": T})
    m = S.memory_episode({"memory_id": "m", "episode_id": "e", "occurred_at": T, "purpose": "happened", "statement": "PROSE"})
    assert "PROSE" not in h.model_dump_json() + m.model_dump_json()
    assert h.related_refs == ["c"]


def test_action_outcome_verdict_words():
    assert S.action_outcome({"id": 1, "observed_at": T, "claim_upheld": None}).verdict == "claim_upheld=null"
    assert S.action_outcome({"id": 2, "observed_at": T, "claim_upheld": False}).verdict == "claim_upheld=false"


def test_event_ids_are_deterministic_and_validated():
    e = S.field_dominance_run({"run_id": "r", "target_id": "x", "started_at": T, "ended_at": T, "tick_count": 3,
                               "min_streak_at_run": 3})
    assert e.event_id == "field_dominance_run:r"
    with pytest.raises(ValueError):
        TemporalSelfEventV1(event_id="wrong", day_id="2026-10-09", occurred_at=T, source_kind="gpu_wait",
                            source_table="gpu_pool_events", source_ref="g")


def test_broadcast_tick_subject_is_the_selected_loops_first_source_ref():
    row = {"log_id": "l", "generated_at": T, "projection_json": {
        "selected_open_loop_id": "open-loop-1",
        "attended_node_ids": ["IGNORED"], "dwell_ticks": 99,
        "frame": {"open_loops": [{"id": "open-loop-0", "source_refs": ["nope"]},
                                 {"id": "open-loop-1", "source_refs": ["node:substrate.chat", "x"], "description": "Chat"}]}}}
    tick = tick_from_log_row(row)
    assert tick.ref == "node:substrate.chat" and tick.label == "Chat"
    none = tick_from_log_row({"log_id": "m", "generated_at": T, "projection_json": '{"frame": {"open_loops": []}}'})
    assert none.ref is None  # a no-winner frame is never invented into a subject


def test_body_summary_from_rows_and_rest_values():
    hot = S.visual_deferral({"attempt_id": "a", "started_at": T, "outcome": "deferred_thermal",
                             "result_json": {"detail": {"state": "hot"}, "refused": True}})
    busy = S.visual_deferral({"attempt_id": "b", "started_at": T, "outcome": "deferred_resource",
                              "result_json": {"detail": {"state": "elevated"}}})
    b = summarize_body(
        [{"chassis_watts": 100, "peak_pressure": 0.2}, {"chassis_watts": 300, "peak_pressure": 1.0}],
        [{"cabinet_temp_c": 27.5}, {"cabinet_temp_c": 29.0}],
        [{}, {}],
        [hot, busy],
    )
    assert (b.chassis_watts_mean, b.cabinet_temp_c_min, b.cabinet_temp_c_max) == (200.0, 27.5, 29.0)
    assert (b.ambient_spike_count, b.thermal_refusals, b.cluster_sample_count) == (2, 1, 2)
    empty = summarize_body()
    assert empty.cluster_sample_count == 0 and empty.chassis_watts_mean is None and empty.thermal_refusals == 0
    # Gate failures are not fields: they cannot be filled by accident.
    assert "peak_pressure_max" not in type(b).model_fields and "cooling_switch_changes" not in type(b).model_fields


def test_schemas_resolve_through_the_registry():
    for name in ("TemporalSelfEventV1", "TemporalSelfArcV1", "TemporalSelfFrameV1", "TemporalSelfDayV1",
                 "TemporalSelfStateV1", "ArcSummaryV1", "OpenThreadV1", "ExpectationRefV1",
                 "ArcAttentionSummaryV1", "ArcBodySummaryV1"):
        assert resolve(name).__module__ == "orion.schemas.temporal_self"
    assert resolve("TemporalSelfFrameV1") is TemporalSelfFrameV1


def test_every_source_kind_has_an_adapter_and_no_dead_kind_is_declared():
    from typing import get_args

    from orion.schemas.temporal_self import TemporalSourceKind

    assert set(get_args(TemporalSourceKind)) == set(S.ADAPTERS)
    for dead in ("town_exchange", "presence_transition", "prior_revision", "peer_ask", "situation_revision"):
        assert dead not in S.ADAPTERS


def test_reducer_package_does_no_io_and_names_no_observational_question():
    root = Path(__file__).resolve().parents[1]
    banned_imports = {"sqlalchemy", "asyncpg", "psycopg", "psycopg2", "redis", "requests", "httpx", "socket", "urllib",
                      "subprocess", "os", "shutil", "pathlib"}
    observational = ("deliberation_need", "reverie_fit", "attention_interrupt")
    for path in root.glob("*.py"):
        text = path.read_text()
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [(node.module or "").split(".")[0]]
            else:
                continue
            assert not banned_imports.intersection(names), f"{path.name} imports I/O: {names}"
        assert not any(q in text for q in observational), path.name
        assert "open(" not in text, f"{path.name} opens a file"
