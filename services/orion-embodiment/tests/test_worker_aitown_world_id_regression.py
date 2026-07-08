"""Regression tests for stale AITOWN_WORLD_ID after a fresh AI Town game.

Live incident (2026-07-08): the embodiment worker cached a deleted world id at
boot while ~/.fcc/.env already pointed at the new world. Every perception tick
failed with ``Invalid world ID``, so Orion could not accept invites, move, or
speak until the container was restarted.
"""
from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, patch

from app.settings import get_settings
from app.worker import EmbodimentWorker
from orion.embodiment.aitown_client import AitownClientError


_DELETED_WORLD_ID = "m173cy1e74gt2e06g9js1eq24s8a0b1b"
_LIVE_WORLD_ID = "m174spk0rd4namch9qvt53fs4x8a4f62"


def _worker() -> EmbodimentWorker:
    w = EmbodimentWorker.__new__(EmbodimentWorker)
    w._settings = get_settings()
    w._bus = object()
    w._stop = asyncio.Event()
    w._walkable_loaded = False
    w._walkable = None
    return w


def test_load_fcc_env_overwrites_stale_aitown_world_id_in_process_env(
    tmp_path, monkeypatch,
) -> None:
    """AITOWN_* must not use setdefault — a stale process env must lose to ~/.fcc/.env."""
    monkeypatch.setenv("AITOWN_WORLD_ID", _DELETED_WORLD_ID)
    fcc = tmp_path / ".fcc.env"
    fcc.write_text(f"AITOWN_WORLD_ID={_LIVE_WORLD_ID}\n", encoding="utf-8")

    w = _worker()
    w._settings.fcc_env_path = str(fcc)
    w._load_fcc_env()

    assert os.environ["AITOWN_WORLD_ID"] == _LIVE_WORLD_ID


def test_stale_boot_world_id_recovers_after_fresh_game_wipe_without_restart(
    tmp_path, monkeypatch,
) -> None:
    """After bootstrap writes a new world id, perception must query the live world."""
    monkeypatch.setenv("AITOWN_WORLD_ID", _DELETED_WORLD_ID)
    monkeypatch.setenv("AITOWN_ORION_PLAYER_ID", "p:24")

    fcc = tmp_path / ".fcc.env"
    fcc.write_text(
        f"AITOWN_WORLD_ID={_LIVE_WORLD_ID}\nAITOWN_ORION_PLAYER_ID=p:24\n",
        encoding="utf-8",
    )

    w = _worker()
    w._settings.fcc_env_path = str(fcc)
    w._world_id = _DELETED_WORLD_ID
    w._orion_player_id = "p:24"

    queried: list[str | None] = []

    def list_players(*, world_id: str | None = None):
        queried.append(world_id)
        if world_id == _DELETED_WORLD_ID:
            raise AitownClientError(f"Invalid world ID: {world_id}")
        return [{"id": "p:24", "position": {"x": 63.0, "y": 27.0}}]

    with patch("app.worker.aitown_client.list_players", side_effect=list_players), \
         patch("app.worker.aitown_client.list_conversations", return_value=[]), \
         patch("app.worker.publish_with_reconnect", new=AsyncMock()):
        stale_perception = asyncio.run(w._emit_perception_once())

    assert stale_perception is None
    assert queried == [_DELETED_WORLD_ID]

    queried.clear()
    w._refresh_aitown_runtime()

    with patch("app.worker.aitown_client.list_players", side_effect=list_players), \
         patch("app.worker.aitown_client.list_conversations", return_value=[]), \
         patch("app.worker.publish_with_reconnect", new=AsyncMock()):
        live_perception = asyncio.run(w._emit_perception_once())

    assert live_perception is not None
    assert live_perception.player_id == "p:24"
    assert queried == [_LIVE_WORLD_ID]


def test_perception_loop_refreshes_aitown_runtime_before_each_emit(
    tmp_path, monkeypatch,
) -> None:
    """The live loop must re-read ~/.fcc/.env every tick, not only at container boot."""
    monkeypatch.setenv("AITOWN_WORLD_ID", _DELETED_WORLD_ID)
    monkeypatch.setenv("AITOWN_ORION_PLAYER_ID", "p:24")

    fcc = tmp_path / ".fcc.env"
    fcc.write_text(
        f"AITOWN_WORLD_ID={_LIVE_WORLD_ID}\nAITOWN_ORION_PLAYER_ID=p:24\n",
        encoding="utf-8",
    )

    w = _worker()
    w._settings.fcc_env_path = str(fcc)
    w._settings.perception_interval_sec = 0.01
    w._world_id = _DELETED_WORLD_ID
    w._orion_player_id = "p:24"
    w._last_heartbeat_log_at = None

    refresh_calls = 0
    emit_world_ids: list[str] = []

    real_refresh = w._refresh_aitown_runtime

    def counting_refresh() -> None:
        nonlocal refresh_calls
        refresh_calls += 1
        real_refresh()

    async def capture_emit():
        emit_world_ids.append(w._world_id)
        w._stop.set()
        return None

    async def run_loop() -> None:
        with patch.object(w, "_refresh_aitown_runtime", side_effect=counting_refresh), \
             patch.object(w, "_emit_perception_once", side_effect=capture_emit), \
             patch.object(w, "_maybe_log_heartbeat"):
            loop_task = asyncio.create_task(w._perception_loop())
            await asyncio.wait_for(w._stop.wait(), timeout=2.0)
            await asyncio.wait_for(loop_task, timeout=2.0)

    asyncio.run(run_loop())

    assert refresh_calls >= 1
    assert emit_world_ids == [_LIVE_WORLD_ID]
