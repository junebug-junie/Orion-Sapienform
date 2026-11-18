"""Regression: the Hub "Bus Anomaly Detected" notifier stays retired.

Removed 2026-11-10 (prediction-error suite rebuild, retire step 1). It listened for
metacog transport triggers with evidence_source=bus_synaptic_prediction_error, a
trigger the equilibrium service stopped publishing on 2026-09-30 (last live row in
metacog_trigger: 2026-09-30 06:31 UTC). It could never fire again.
"""

from __future__ import annotations

from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def test_notifier_module_is_gone() -> None:
    assert not (_SCRIPTS / "bus_synaptic_trigger_notifier.py").exists()


def test_hub_scripts_do_not_wire_or_emit_the_bus_anomaly_notification() -> None:
    for path in _SCRIPTS.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "bus_synaptic_trigger_notifier" not in text, path.name
        assert "BusSynapticTriggerNotifier" not in text, path.name
        assert "Bus Anomaly Detected" not in text, path.name
