from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from app.settings import get_settings
from app.worker import EmbodimentWorker
from orion.schemas.embodiment import EMBODIMENT_PERCEPTION_KIND


def _worker() -> EmbodimentWorker:
    w = EmbodimentWorker.__new__(EmbodimentWorker)
    w._settings = get_settings()
    w._orion_player_id = "orion"
    w._world_id = "w1"
    w._bus = object()
    return w


def test_emit_perception_once_publishes_built_perception():
    w = _worker()
    players = [
        {"id": "orion", "position": {"x": 0.0, "y": 0.0}},
        {"id": "j", "name": "Juniper", "position": {"x": 3.0, "y": 4.0}},
    ]
    with patch("app.worker.aitown_client.list_players", return_value=players), \
         patch("app.worker.publish_with_reconnect", new=AsyncMock()) as pub:
        perc = asyncio.run(w._emit_perception_once())

    assert perc is not None
    assert perc.player_id == "orion"
    assert perc.nearby_players[0]["distance"] == 5.0
    pub.assert_awaited_once()
    args, kwargs = pub.call_args
    assert args[1] == w._settings.channel_perception
    env = args[2]
    assert env.kind == EMBODIMENT_PERCEPTION_KIND
    assert env.payload["player_id"] == "orion"


def test_emit_perception_once_none_when_orion_absent():
    w = _worker()
    players = [{"id": "x", "position": {"x": 1.0, "y": 1.0}}]
    with patch("app.worker.aitown_client.list_players", return_value=players), \
         patch("app.worker.publish_with_reconnect", new=AsyncMock()) as pub:
        perc = asyncio.run(w._emit_perception_once())

    assert perc is None
    pub.assert_not_awaited()


def test_emit_perception_once_fail_open_on_list_players_error():
    w = _worker()
    with patch("app.worker.aitown_client.list_players", side_effect=RuntimeError("down")), \
         patch("app.worker.publish_with_reconnect", new=AsyncMock()) as pub:
        perc = asyncio.run(w._emit_perception_once())

    assert perc is None
    pub.assert_not_awaited()


def test_emit_perception_once_fail_open_on_malformed_row():
    w = _worker()
    players = [
        {"id": "orion", "position": {"x": 0.0, "y": 0.0}},
        {"id": "j", "position": {"x": 1.0}},
    ]
    with patch("app.worker.aitown_client.list_players", return_value=players), \
         patch("app.worker.publish_with_reconnect", new=AsyncMock()) as pub:
        perc = asyncio.run(w._emit_perception_once())

    assert perc is None
    pub.assert_not_awaited()


def test_refresh_aitown_runtime_picks_up_updated_fcc_world_id(tmp_path, monkeypatch):
    w = _worker()
    w._walkable_loaded = True
    w._walkable = {(1, 1)}
    fcc = tmp_path / ".fcc.env"
    fcc.write_text(
        "AITOWN_WORLD_ID=new-world\nAITOWN_ORION_PLAYER_ID=p:99\n",
        encoding="utf-8",
    )
    w._settings.fcc_env_path = str(fcc)
    w._world_id = "old-world"
    w._orion_player_id = "p:1"

    w._refresh_aitown_runtime()

    assert w._world_id == "new-world"
    assert w._orion_player_id == "p:99"
    assert w._walkable_loaded is False
    assert w._walkable is None


def test_emit_perception_once_fail_open_on_publish_error():
    w = _worker()
    players = [{"id": "orion", "position": {"x": 0.0, "y": 0.0}}]
    with patch("app.worker.aitown_client.list_players", return_value=players), \
         patch("app.worker.publish_with_reconnect", new=AsyncMock(side_effect=RuntimeError("bus"))):
        perc = asyncio.run(w._emit_perception_once())

    assert perc is None
