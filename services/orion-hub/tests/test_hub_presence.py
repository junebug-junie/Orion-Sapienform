"""Unit tests for hub presence recording and the curiosity focus hint."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
HUB_ROOT = Path(__file__).resolve().parents[1]


def _ensure_hub_scripts_import_path() -> None:
    for key in list(sys.modules):
        if key == "scripts" or key.startswith("scripts."):
            del sys.modules[key]
    for p in (str(REPO_ROOT), str(HUB_ROOT)):
        try:
            sys.path.remove(p)
        except ValueError:
            pass
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(HUB_ROOT))


_ensure_hub_scripts_import_path()

from scripts import curiosity_hint, hub_presence  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_presence():
    hub_presence.reset()
    yield
    hub_presence.reset()


# ── presence_snapshot ────────────────────────────────────────────────


def test_snapshot_none_before_first_turn():
    assert hub_presence.presence_snapshot() is None


def test_snapshot_active_health_and_rate():
    for offset in (0.0, 60.0, 120.0):
        hub_presence.record_turn(now=1000.0 + offset)
    snap = hub_presence.presence_snapshot(now=1150.0)
    assert snap["connection_health"] == "active"
    assert snap["last_turn_age_sec"] == 30.0
    # 3 turns in the trailing 300s window → 0.6 turns/minute.
    assert snap["turns_per_minute"] == 0.6


def test_snapshot_idle_then_dormant():
    hub_presence.record_turn(now=1000.0)
    assert hub_presence.presence_snapshot(now=1000.0 + 300)["connection_health"] == "idle"
    assert hub_presence.presence_snapshot(now=1000.0 + 1000)["connection_health"] == "dormant"


def test_record_turn_never_raises_without_postgres(monkeypatch):
    monkeypatch.delenv("POSTGRES_URI", raising=False)
    hub_presence.record_turn()
    assert hub_presence.presence_snapshot() is not None


def test_record_turn_rate_limits_writes(monkeypatch):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    calls: list[dict] = []

    def fake_writer(snapshot):
        calls.append(snapshot)

    with patch.object(hub_presence, "_write_snapshot_to_postgres", side_effect=fake_writer):
        with patch.object(hub_presence.threading, "Thread") as fake_thread:
            fake_thread.side_effect = lambda target, args, **kw: type(
                "T", (), {"start": lambda self: target(*args)}
            )()
            hub_presence.record_turn(now=1000.0)
            hub_presence.record_turn(now=1001.0)  # within min interval → no write
            hub_presence.record_turn(now=1006.0)  # past min interval → write

    assert len(calls) == 2
    assert calls[-1]["connection_health"] == "active"


# ── curiosity focus hint ─────────────────────────────────────────────


def test_hint_none_without_summaries():
    assert curiosity_hint.format_curiosity_hint([]) is None
    assert curiosity_hint.format_curiosity_hint([{"signal_strength": 0.9}]) is None


def test_hint_ranked_capped_truncated():
    candidates = [
        {"signal_strength": 0.2, "evidence_summary": "weak gap"},
        {"signal_strength": 0.9, "evidence_summary": "x" * 200},
        {"signal_strength": 0.5, "evidence_summary": "medium gap"},
    ]
    hint = curiosity_hint.format_curiosity_hint(candidates)
    assert hint.startswith("[curiosity focus] Self-observed gaps: ")
    # Top-2 by strength only; the strongest is truncated to 120 chars.
    assert "weak gap" not in hint
    assert "medium gap" in hint
    truncated = "x" * 119 + "…"
    assert truncated in hint


def test_apply_hint_prepends_and_degrades():
    with patch.object(curiosity_hint, "_fetch_fresh_candidates", return_value=[
        {"signal_strength": 0.8, "evidence_summary": "open loop in transport"}
    ]):
        out = curiosity_hint.apply_curiosity_hint("what am I missing?")
    assert out.startswith("[curiosity focus] Self-observed gaps: open loop in transport")
    assert out.endswith("\n\nwhat am I missing?")

    with patch.object(curiosity_hint, "_fetch_fresh_candidates", side_effect=RuntimeError("db down")):
        assert curiosity_hint.apply_curiosity_hint("prompt") == "prompt"

    with patch.object(curiosity_hint, "_fetch_fresh_candidates", return_value=[]):
        assert curiosity_hint.apply_curiosity_hint("prompt") == "prompt"
