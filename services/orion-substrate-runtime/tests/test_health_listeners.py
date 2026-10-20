"""/health counts bus listeners: a dead listener must not read as healthy (2026-10-20)."""
from __future__ import annotations

import asyncio
from pathlib import Path

from orion.core.bus import resilience

REPO_ROOT = Path(__file__).resolve().parents[3]


def _import_main(monkeypatch):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused:5432/unused")
    monkeypatch.setenv("NODE_CATALOG_PATH", str(REPO_ROOT / "config" / "biometrics" / "node_catalog.yaml"))
    import app.settings as settings_mod

    settings_mod._settings = None
    import app.main as main

    return main


def _health(main, monkeypatch, grammar_ok: bool) -> dict:
    monkeypatch.setattr(
        main, "build_substrate_grammar_truth", lambda store: {"ok": grammar_ok, "degraded": not grammar_ok}
    )
    return asyncio.run(main.health())


def test_health_degrades_when_a_listener_is_down(monkeypatch):
    main = _import_main(monkeypatch)
    monkeypatch.setattr(resilience, "_SUBSCRIPTION_STATES", {})
    resilience._SUBSCRIPTION_STATES["finalize_appraisal listener"] = resilience.SubscriptionState(
        name="finalize_appraisal listener", channel="orion:substrate:finalize_appraisal:request", subscribed=False
    )
    out = _health(main, monkeypatch, grammar_ok=True)
    assert out["ok"] is False and out["status"] == "degraded" and out["degraded"] is True
    assert out["grammar_ok"] is True and out["listeners_ok"] is False
    assert out["listeners_down"] == ["finalize_appraisal listener"]


def test_health_ok_when_grammar_and_listeners_are_ok(monkeypatch):
    main = _import_main(monkeypatch)
    monkeypatch.setattr(resilience, "_SUBSCRIPTION_STATES", {})
    resilience._SUBSCRIPTION_STATES["x"] = resilience.SubscriptionState(name="x", channel="c", subscribed=True)
    out = _health(main, monkeypatch, grammar_ok=True)
    assert out["ok"] is True and out["status"] == "ok" and out["listeners_down"] == []
    assert out["listeners"][0]["channel"] == "c"


def test_health_monitor_flags_a_down_listener(monkeypatch):
    from types import SimpleNamespace

    import app.health_monitor as hm

    monkeypatch.setattr(hm, "build_substrate_grammar_truth", lambda store: {"degraded": False})
    monkeypatch.setattr(resilience, "_SUBSCRIPTION_STATES", {})
    resilience._SUBSCRIPTION_STATES["finalize_appraisal listener"] = resilience.SubscriptionState(
        name="finalize_appraisal listener", channel="c", subscribed=False
    )
    settings = SimpleNamespace(enable_biometrics_pressure_organ=False)
    check = next(c for c in hm.run_checks(store=None, settings=settings) if c.key == "substrate_listeners_down")
    assert check.healthy is False and check.severity == "critical"
    assert "finalize_appraisal listener" in check.message

    resilience._SUBSCRIPTION_STATES["finalize_appraisal listener"].subscribed = True
    check = next(c for c in hm.run_checks(store=None, settings=settings) if c.key == "substrate_listeners_down")
    assert check.healthy is True and check.message == ""
