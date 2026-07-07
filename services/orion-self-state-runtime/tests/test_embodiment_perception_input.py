"""Unit tests for the Orion embodiment perception grounding input.

Verifies the self-state worker caches the latest ``WorldPerceptionV1`` as a
best-effort, age-gated observability input when enabled, ignores it when
disabled, and drops stale perception. Mirrors the observability-input pattern
in ``test_store_observability_loaders.py`` (age gate + graceful None).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import app.worker as worker_mod
from app.worker import SelfStateRuntimeWorker, _PERCEPTION_MAX_AGE_SEC

from orion.schemas.embodiment import WorldPerceptionV1


def _make_worker(monkeypatch, *, enabled: bool) -> SelfStateRuntimeWorker:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    monkeypatch.setenv(
        "EMBODIMENT_PERCEPTION_SELFSTATE_ENABLED", "true" if enabled else "false"
    )
    import app.settings as settings_mod

    settings_mod._settings = None

    worker = SelfStateRuntimeWorker.__new__(SelfStateRuntimeWorker)
    worker._settings = settings_mod.get_settings()
    worker._latest_perception = None
    worker._latest_perception_at = None
    return worker


def _perception() -> WorldPerceptionV1:
    return WorldPerceptionV1(
        player_id="orion",
        position={"x": 1.0, "y": 2.0},
        nearby_players=[{"player_id": "j", "name": "Juniper", "distance": 1.0}],
    )


def test_flag_on_caches_latest_perception_input(monkeypatch):
    worker = _make_worker(monkeypatch, enabled=True)
    now = datetime.now(timezone.utc)

    stored = worker.cache_perception(_perception(), now=now)
    assert stored is True

    payload = worker.perception_input(now=now)
    assert payload is not None
    assert payload["player_id"] == "orion"
    assert payload["nearby_count"] == 1
    assert payload["position"] == {"x": 1.0, "y": 2.0}


def test_flag_off_ignores_perception(monkeypatch):
    worker = _make_worker(monkeypatch, enabled=False)
    now = datetime.now(timezone.utc)

    assert worker.cache_perception(_perception(), now=now) is False
    assert worker.perception_input(now=now) is None


def test_stale_perception_is_age_gated(monkeypatch):
    worker = _make_worker(monkeypatch, enabled=True)
    now = datetime.now(timezone.utc)
    stale = now - timedelta(seconds=_PERCEPTION_MAX_AGE_SEC + 1)

    assert worker.cache_perception(_perception(), now=stale) is True
    assert worker.perception_input(now=now) is None


def test_handle_message_fails_open_on_bad_decode(monkeypatch):
    worker = _make_worker(monkeypatch, enabled=True)
    bus = MagicMock()
    bus.codec.decode.return_value = MagicMock(ok=False, error="boom")
    monkeypatch.setattr(worker_mod, "_bus", bus)

    # Must not raise and must not cache anything.
    worker._handle_perception_message({"data": b"garbage"})
    assert worker._latest_perception is None
    assert worker.perception_input(now=datetime.now(timezone.utc)) is None
