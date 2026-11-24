"""The input audit (evals/audit_heartbeat_inputs.py) must flag a rule that the
live data contradicts -- that is what makes it a gate, not a report."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_EVAL = Path(__file__).resolve().parents[1] / "evals" / "audit_heartbeat_inputs.py"


def _load():
    spec = importlib.util.spec_from_file_location("audit_heartbeat_inputs", _EVAL)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


T0 = 1795478400.0  # 2026-11-24T00:00Z


def _clock_rows(service: str, role: str, atom_type: str, salience_fn, every_sec: float = 60.0, hours: int = 24):
    rows = []
    t = T0
    while t < T0 + hours * 3600:
        rows.append([str(t), service, atom_type, role, "0.9", str(salience_fn(t)), "0.1"])
        t += every_sec
    return rows


def _by_role(records):
    return {(r["service"], r["role"]): r for r in records}


def test_floored_measured_gauge_is_flagged_no_rest_and_calm_gauge_is_not() -> None:
    audit = _load()
    rows = _clock_rows("orion-biometrics", "gpu_pressure_signal", "signal", lambda t: 0.0 if int(t) % 120 == 0 else 0.4)
    rows += _clock_rows("orion-biometrics", "thermal_pressure_signal", "signal", lambda t: 0.3 + (int(t) % 7) / 100.0)
    rows.sort(key=lambda r: float(r[0]))
    recs = _by_role(audit.audit(rows))
    assert recs[("orion-biometrics", "gpu_pressure_signal")]["flags"] == []
    assert recs[("orion-biometrics", "gpu_pressure_signal")]["rest_share_night"] > 0.4
    assert "MEASURED_NO_REST" in recs[("orion-biometrics", "thermal_pressure_signal")]["flags"]


def test_flat_measured_and_unlisted_role_and_unnamed_producer_are_flagged() -> None:
    audit = _load()
    rows = _clock_rows("orion-biometrics", "disk_pressure_signal", "signal", lambda t: 0.0)
    rows += _clock_rows("orion-cortex-exec", "brand_new_role", "signal", lambda t: 0.5, every_sec=900)
    rows += _clock_rows("orion-made-up-service", "x", "observation", lambda t: 0.5, every_sec=900)
    rows.sort(key=lambda r: float(r[0]))
    recs = _by_role(audit.audit(rows))
    assert "MEASURED_FLAT" in recs[("orion-biometrics", "disk_pressure_signal")]["flags"]
    assert "UNLISTED_ROLE" in recs[("orion-cortex-exec", "brand_new_role")]["flags"]
    assert "UNNAMED_PRODUCER" in recs[("orion-made-up-service", "x")]["flags"]


def test_presence_input_on_a_fixed_clock_is_flagged() -> None:
    audit = _load()
    rows = _clock_rows("orion-cortex-exec", "exec_step_started", "action_candidate", lambda t: 0.85)
    recs = _by_role(audit.audit(rows))
    assert "PRESENCE_CLOCKWORK" in recs[("orion-cortex-exec", "exec_step_started")]["flags"]
