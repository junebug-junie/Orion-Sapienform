"""Wall-clock budget enforcement (loop_budget_exceeded)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_guard_path = Path(__file__).resolve().parent / "_mind_import_guard.py"


def _mind_prep() -> None:
    spec = importlib.util.spec_from_file_location("_mind_guard_lazy", _guard_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ensure_orion_mind_app()


def test_raises_loop_budget_exceeded_when_wall_clock_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate snapshot phase taking longer than policy allows."""
    _mind_prep()
    from app.engine import run_mind_deterministic
    from orion.mind.v1 import MindRunRequestV1, MindRunPolicyV1

    router_dir = Path(__file__).resolve().parents[1] / "app" / "config"
    seq = [0.0, 100.0, 100.0]
    i = 0

    def fake_perf() -> float:
        nonlocal i
        v = seq[i] if i < len(seq) else seq[-1]
        i += 1
        return v

    monkeypatch.setattr("app.engine.time.perf_counter", fake_perf)

    req = MindRunRequestV1(
        correlation_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        snapshot_inputs={"user_text": "x"},
        policy=MindRunPolicyV1(n_loops_max=1, wall_time_ms_max=50, router_profile_id="default"),
    )
    out = run_mind_deterministic(req, router_profiles_dir=router_dir, snapshot_max_bytes=512_000)
    assert out.ok is False
    assert out.error_code == "loop_budget_exceeded"
