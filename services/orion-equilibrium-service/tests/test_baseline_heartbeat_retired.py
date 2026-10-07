"""The scheduled baseline heartbeat and the substrate dense/pulse gate were
retired 2026-09-29. Baseline carried no evidence (empty upstream), so once its
rows could publish they were the same "stable system state" sentence every hour;
dense/pulse could never fire (eventfulness max 0.25 < both thresholds).
Kill means kill: no timer-driven metacog trigger comes back through this service.
"""
from __future__ import annotations

import re
from pathlib import Path

from app.service import EquilibriumService

SERVICE_SRC = (Path(__file__).resolve().parents[1] / "app" / "service.py").read_text()


def test_no_baseline_loop_or_substrate_gate():
    assert not hasattr(EquilibriumService, "_metacog_baseline_loop")
    assert not hasattr(EquilibriumService, "_maybe_emit_baseline_metacog_trigger")
    assert not (Path(__file__).resolve().parents[1] / "app" / "substrate_metacog_gate.py").exists()


def test_service_never_builds_a_baseline_dense_or_pulse_trigger():
    assert not re.search(r'trigger_kind\s*=\s*"(baseline|dense|pulse)"', SERVICE_SRC)
