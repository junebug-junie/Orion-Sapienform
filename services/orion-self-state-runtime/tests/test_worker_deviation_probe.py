"""Unit tests for the Phase 2 deviation-probe logging hook (2026-07-12,
measurement-only, no schema field yet).

Mirrors the __new__-and-hand-wire pattern in test_worker_prune.py /
test_embodiment_perception_input.py.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from orion.autonomy.deviation_gate import DeviationGate
from orion.schemas.self_state import SelfStateDimensionV1, SelfStateV1
from orion.self_state.policy import load_self_state_policy

from app.worker import SelfStateRuntimeWorker

REPO = Path(__file__).resolve().parents[3]
POLICY = load_self_state_policy(REPO / "config" / "self_state" / "self_state_policy.v1.yaml")
NOW = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)


def _make_worker(monkeypatch, *, probe_enabled: bool = True) -> SelfStateRuntimeWorker:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    monkeypatch.setenv("SELF_STATE_DEVIATION_PROBE_ENABLED", "true" if probe_enabled else "false")
    import app.settings as settings_mod

    settings_mod._settings = None

    worker = SelfStateRuntimeWorker.__new__(SelfStateRuntimeWorker)
    worker._settings = settings_mod.get_settings()
    worker._policy = POLICY
    worker._deviation_gate = DeviationGate()
    return worker


def _state() -> SelfStateV1:
    dims = {
        "resource_pressure": SelfStateDimensionV1(
            dimension_id="resource_pressure", score=0.6, confidence=0.7
        ),
        "coherence": SelfStateDimensionV1(dimension_id="coherence", score=0.8, confidence=0.9),
    }
    return SelfStateV1(
        self_state_id="self.state:probe_test",
        generated_at=NOW,
        source_field_tick_id="tick",
        source_field_generated_at=NOW,
        source_attention_frame_id="frame",
        source_attention_generated_at=NOW,
        overall_condition="steady",
        overall_intensity=0.4,
        overall_confidence=0.8,
        dimensions=dims,
    )


def test_log_deviation_probe_logs_impulses_and_confidence(monkeypatch, caplog) -> None:
    worker = _make_worker(monkeypatch)
    with caplog.at_level(logging.INFO, logger="orion.self_state.runtime"):
        worker._log_deviation_probe(_state())

    records = [r for r in caplog.records if r.message.startswith("self_state_deviation_probe")]
    assert len(records) == 1
    msg = records[0].getMessage()
    assert "self.state:probe_test" in msg
    assert "resource_pressure" in msg
    assert "coherence" in msg


def test_log_deviation_probe_disabled_via_settings_does_not_log(monkeypatch, caplog) -> None:
    # 2026-07-12 review finding: no way to disable a per-tick log without a
    # redeploy -- SELF_STATE_DEVIATION_PROBE_ENABLED=false must be a real
    # no-op, not just a documented intent.
    worker = _make_worker(monkeypatch, probe_enabled=False)
    with caplog.at_level(logging.INFO, logger="orion.self_state.runtime"):
        worker._log_deviation_probe(_state())

    assert not any(r.message.startswith("self_state_deviation_probe") for r in caplog.records)


def test_log_deviation_probe_never_raises_on_gate_failure(monkeypatch, caplog) -> None:
    worker = _make_worker(monkeypatch)

    class _ExplodingGate:
        def observe(self, *_a, **_kw):
            raise RuntimeError("boom")

    worker._deviation_gate = _ExplodingGate()
    with caplog.at_level(logging.ERROR, logger="orion.self_state.runtime"):
        # Must not raise -- a probe failure must never block persisting self_state.
        worker._log_deviation_probe(_state())

    assert any("self_state_deviation_probe_failed" in r.message for r in caplog.records)


def test_log_deviation_probe_carries_state_across_ticks(monkeypatch) -> None:
    # Same worker instance (same gate) across two ticks should build up a
    # baseline -- proving the gate is a per-process, cross-tick object, not
    # reconstructed fresh every tick (which would defeat its whole purpose).
    worker = _make_worker(monkeypatch)
    stable = _state()
    for _ in range(6):
        worker._log_deviation_probe(stable)

    assert worker._deviation_gate.baseline_count() == 2  # resource_pressure + coherence
