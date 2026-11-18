"""R3 reader 6: no self-modification proposal while Orion's arousal reads strained.

Must-not: arousal never makes a below-threshold or cooling pressure ready, is read only at the
decision point (once per cycle, only when a pressure is otherwise ready), does not cool the
held pressure, and a worker without a reader behaves exactly as before.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orion.core.schemas.substrate_mutation import MutationPressureV1
from orion.substrate.mutation_pressure import AROUSAL_STRAINED_REASON, PressureAccumulator, PressurePolicy
from orion.substrate.mutation_queue import SubstrateMutationStore
from orion.substrate.tests.test_mutation_worker_extra_signals import _build_worker, _continuity_pressure_signal

NOW = datetime(2026, 10, 13, 15, 0, tzinfo=timezone.utc)
LEVELS = ["engaged", "idle", "strained", None]


class Probe:
    def __init__(self, level):
        self.level, self.calls = level, 0

    def __call__(self):
        self.calls += 1
        return self.level


def _pressure(score=10.0, cooldown_until=None):
    return MutationPressureV1(
        pressure_id="p-1", anchor_scope="orion", subject_ref="entity:orion",
        target_surface="graph_consolidation", target_zone="world_ontology", pressure_kind="k",
        pressure_score=score, cooldown_until=cooldown_until, updated_at=NOW,
    )


def test_strained_withholds_a_ready_pressure():
    acc = PressureAccumulator()
    assert acc.ready_for_proposal(_pressure(), now=NOW) is True
    assert acc.ready_for_proposal(_pressure(), now=NOW, arousal_level=Probe("strained")) is False
    assert acc.readiness(_pressure(), now=NOW, arousal_level=Probe("strained")) == AROUSAL_STRAINED_REASON


@pytest.mark.parametrize("level", ["engaged", "idle", None])
def test_other_levels_and_unknown_are_today(level):
    assert PressureAccumulator().ready_for_proposal(_pressure(), now=NOW, arousal_level=Probe(level)) is True


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("pressure,why", [
    (_pressure(score=1.0), "below_threshold"),
    (_pressure(cooldown_until=NOW + timedelta(seconds=60)), "cooldown"),
])
def test_must_not_arousal_never_turns_a_hold_into_ready(level, pressure, why):
    probe = Probe(level)
    acc = PressureAccumulator(policy=PressurePolicy())
    assert acc.ready_for_proposal(pressure, now=NOW, arousal_level=probe) is False
    assert acc.readiness(pressure, now=NOW, arousal_level=probe) == why
    assert probe.calls == 0  # never consulted unless otherwise ready


def test_worker_holds_and_traces_without_cooling(monkeypatch):
    monkeypatch.setenv("SUBSTRATE_MUTATION_AUTONOMY_ENABLED", "true")
    store = SubstrateMutationStore()
    worker = _build_worker(store)
    traces = []
    probe = Probe("strained")
    worker.trace_logger = traces.append
    worker.arousal_level = probe
    signal = _continuity_pressure_signal(0.9)

    result = worker.run_cycle(telemetry=[], measured_metrics_by_proposal={}, extra_signals=[signal], now=NOW)
    assert result["proposals"] == 0
    assert result["proposals_held_arousal"] == 1
    assert "proposals_held:arousal_strained=1" in result["notes"]
    assert probe.calls == 1  # one read per cycle, at the decision point
    held = [t for t in traces if t.get("event") == "mutation_proposal_held"]
    assert len(held) == 1 and "reason=arousal_strained" in held[0]["notes"]
    assert not [t for t in traces if t.get("event") == "mutation_proposal_enqueued"]
    (pressure,) = store._pressures.values()
    assert pressure.cooldown_until is None  # kept, not cooled

    # Strain clears: the same pressure is proposed on the next cycle with no new signal.
    worker.arousal_level = Probe("idle")
    result2 = worker.run_cycle(telemetry=[], measured_metrics_by_proposal={}, now=NOW + timedelta(seconds=30))
    assert result2["proposals"] == 1 and result2["proposals_held_arousal"] == 0


@pytest.mark.parametrize("level", ["engaged", "idle", None])
def test_worker_other_levels_match_no_reader(monkeypatch, level):
    monkeypatch.setenv("SUBSTRATE_MUTATION_AUTONOMY_ENABLED", "true")
    signal = _continuity_pressure_signal(0.9)
    w_reader = _build_worker(SubstrateMutationStore())
    w_reader.arousal_level = Probe(level)
    w_plain = _build_worker(SubstrateMutationStore())
    r1 = w_reader.run_cycle(telemetry=[], measured_metrics_by_proposal={}, extra_signals=[signal], now=NOW)
    r2 = w_plain.run_cycle(telemetry=[], measured_metrics_by_proposal={}, extra_signals=[signal], now=NOW)
    assert r1["proposals"] == r2["proposals"] == 1
    assert r1["proposals_held_arousal"] == r2["proposals_held_arousal"] == 0


def test_worker_reader_error_is_unknown(monkeypatch):
    monkeypatch.setenv("SUBSTRATE_MUTATION_AUTONOMY_ENABLED", "true")
    worker = _build_worker(SubstrateMutationStore())

    def boom():
        raise ConnectionError("redis down")

    worker.arousal_level = boom
    r = worker.run_cycle(telemetry=[], measured_metrics_by_proposal={},
                         extra_signals=[_continuity_pressure_signal(0.9)], now=NOW)
    assert r["proposals"] == 1


def test_worker_below_threshold_never_reads_arousal(monkeypatch):
    monkeypatch.setenv("SUBSTRATE_MUTATION_AUTONOMY_ENABLED", "true")
    worker = _build_worker(SubstrateMutationStore())
    probe = Probe("strained")
    worker.arousal_level = probe
    worker.run_cycle(telemetry=[], measured_metrics_by_proposal={},
                     extra_signals=[_continuity_pressure_signal(0.1)], now=NOW)
    assert probe.calls == 0


def test_must_not_kill_switch_wins_over_any_level(monkeypatch):
    monkeypatch.setenv("SUBSTRATE_MUTATION_AUTONOMY_ENABLED", "false")
    worker = _build_worker(SubstrateMutationStore())
    probe = Probe("idle")
    worker.arousal_level = probe
    r = worker.run_cycle(telemetry=[], measured_metrics_by_proposal={},
                         extra_signals=[_continuity_pressure_signal(0.9)], now=NOW)
    assert r["notes"] == ["autonomy_kill_switch_disabled"] and probe.calls == 0
