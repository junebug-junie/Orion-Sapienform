"""R3 reader 4 wiring: the worker's arousal callable honours its flag and maps unknown to None."""
from __future__ import annotations

from types import SimpleNamespace

from orion.regulation.arousal_view import ArousalView, UNKNOWN_DISABLED


def _worker(monkeypatch, *, enabled, view):
    import app.worker as wm

    calls = []

    def fake_read(url, *, enabled, reader, max_age_sec, **_kw):
        calls.append({"url": url, "enabled": enabled, "reader": reader, "max_age_sec": max_age_sec})
        return view if enabled else UNKNOWN_DISABLED

    monkeypatch.setattr(wm, "read_arousal_sync", fake_read)
    w = wm.ProposalRuntimeWorker.__new__(wm.ProposalRuntimeWorker)
    w._settings = SimpleNamespace(orion_bus_url="redis://bus:6379/0", visual_baseline_arousal_enabled=enabled,
                                  regulation_reader_max_age_sec=360.0)
    return w, calls


def test_strained_is_returned_to_schedule(monkeypatch):
    w, calls = _worker(monkeypatch, enabled=True, view=ArousalView(level="strained", reason="strained"))
    assert w._baseline_arousal_level() == "strained"
    assert calls == [{"url": "redis://bus:6379/0", "enabled": True, "reader": "visual_baseline", "max_age_sec": 360.0}]


def test_flag_off_and_unknown_are_none(monkeypatch):
    w, _ = _worker(monkeypatch, enabled=False, view=ArousalView(level="strained", reason="strained"))
    assert w._baseline_arousal_level() is None
    w, _ = _worker(monkeypatch, enabled=True, view=ArousalView(level="unknown", reason="stale"))
    assert w._baseline_arousal_level() is None


def test_settings_ship_the_flag_on():
    from app.settings import Settings

    s = Settings(POSTGRES_URI="postgresql://x@h/d")
    assert s.visual_baseline_arousal_enabled is True and s.regulation_reader_max_age_sec == 360.0
