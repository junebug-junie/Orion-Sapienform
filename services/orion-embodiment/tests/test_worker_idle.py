from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.worker import EmbodimentWorker


def _worker(idle_sec: float, last_move, last_wander=None):
    w = EmbodimentWorker.__new__(EmbodimentWorker)
    w._settings = SimpleNamespace(idle_heartbeat_sec=idle_sec)
    w._orion_player_id = "orion"
    w._last_move_at = last_move
    w._last_idle_wander_at = last_wander
    w._actuate_lock = asyncio.Lock()
    return w


def test_idle_wander_emits_when_idle():
    w = _worker(20.0, None)
    with patch.object(w, "process_intent", return_value="OUT") as pi, \
         patch.object(w, "_publish_outcome", new=AsyncMock()) as pub:
        asyncio.run(w._maybe_idle_wander(now=datetime(2026, 7, 7, tzinfo=timezone.utc)))
    assert pi.called
    intent = pi.call_args.args[0]
    assert intent.kind == "wander" and intent.source == "involuntary"
    pub.assert_awaited_once()


def test_idle_wander_skips_within_window():
    now = datetime(2026, 7, 7, 0, 0, 10, tzinfo=timezone.utc)
    w = _worker(20.0, datetime(2026, 7, 7, 0, 0, 5, tzinfo=timezone.utc))
    with patch.object(w, "process_intent") as pi, \
         patch.object(w, "_publish_outcome", new=AsyncMock()):
        asyncio.run(w._maybe_idle_wander(now=now))
    assert not pi.called


def test_idle_wander_off_when_zero():
    w = _worker(0.0, None)
    with patch.object(w, "process_intent") as pi:
        asyncio.run(w._maybe_idle_wander(now=datetime(2026, 7, 7, tzinfo=timezone.utc)))
    assert not pi.called


def test_idle_wander_does_not_refire_after_non_actuated_outcome():
    # A non-actuated wander (e.g. resolved_noop) must NOT re-fire every tick: the
    # attempt timestamp gates the window even though _last_move_at never advanced.
    w = _worker(20.0, None)
    with patch.object(w, "process_intent", return_value="NOOP") as pi, \
         patch.object(w, "_publish_outcome", new=AsyncMock()):
        asyncio.run(w._maybe_idle_wander(now=datetime(2026, 7, 7, 0, 0, 0, tzinfo=timezone.utc)))
        # 3s later (one perception tick) -> still inside the 20s window -> no re-fire.
        asyncio.run(w._maybe_idle_wander(now=datetime(2026, 7, 7, 0, 0, 3, tzinfo=timezone.utc)))
        # Past the window -> fires again.
        asyncio.run(w._maybe_idle_wander(now=datetime(2026, 7, 7, 0, 0, 25, tzinfo=timezone.utc)))
    assert pi.call_count == 2
