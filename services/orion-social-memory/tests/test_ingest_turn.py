from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import yaml
from fastapi.testclient import TestClient

SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for module_name in [name for name in sys.modules if name == "app" or name.startswith("app.")]:
    sys.modules.pop(module_name)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SERVICE_ROOT))

from orion.schemas.social_chat import SocialRoomTurnV1

from app.service import SocialMemoryService
from app.settings import Settings


class _FakeBus:
    enabled = True

    def __init__(self) -> None:
        self.published = []

    async def publish(self, channel, envelope) -> None:
        self.published.append((channel, envelope))

    async def close(self) -> None:
        return None


def _aitown_turn() -> SocialRoomTurnV1:
    return SocialRoomTurnV1(
        prompt="the urn is dying",
        response="I'll pull a spare from the back",
        client_meta={
            "external_room": {
                "platform": "aitown",
                "room_id": "aitown-town",
                "thread_id": "cam-lin--sofia-bell",
            },
            "external_participant": {
                "participant_id": "sofia-bell",
                "participant_name": "Sofia Bell",
                "participant_kind": "npc",
            },
        },
    )


def _aitown_turn_body() -> dict:
    return _aitown_turn().model_dump(mode="json")


def _ingest_service() -> SocialMemoryService:
    return SocialMemoryService(settings=Settings(ORION_BUS_ENABLED=False), bus=_FakeBus())


class _NoopService:
    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def ingest_turn(self, turn: SocialRoomTurnV1) -> None:
        raise AssertionError("ingest_turn must not run on 401")


def _post_ingest(client: TestClient, *, headers: dict | None = None) -> object:
    return client.post("/ingest-turn", headers=headers or {}, json=_aitown_turn_body())


def test_ingest_turn_publishes_social_turn_v1() -> None:
    svc = _ingest_service()
    turn = _aitown_turn()

    asyncio.run(svc.ingest_turn(turn))

    published = [(channel, envelope) for channel, envelope in svc.bus.published]
    assert len(published) == 1
    channel, envelope = published[0]
    assert channel == "orion:chat:social:turn"
    assert envelope.kind == "social.turn.v1"
    assert envelope.payload["prompt"] == "the urn is dying"
    assert envelope.payload["response"] == "I'll pull a spare from the back"
    assert envelope.payload["client_meta"]["external_room"]["thread_id"] == "cam-lin--sofia-bell"


def test_ingest_turn_does_not_call_process_social_turn(monkeypatch) -> None:
    svc = _ingest_service()

    async def _fail(*_args, **_kwargs):
        raise AssertionError("process_social_turn must not be called from ingest_turn")

    monkeypatch.setattr(svc, "process_social_turn", _fail)
    asyncio.run(svc.ingest_turn(_aitown_turn()))


def test_ingest_http_401_without_token() -> None:
    import app.main as main_mod

    original_service = main_mod.service
    main_mod.service = _NoopService()
    try:
        with TestClient(main_mod.app) as client:
            resp = _post_ingest(client)
        assert resp.status_code == 401
    finally:
        main_mod.service = original_service


def test_ingest_http_401_when_server_token_empty() -> None:
    import app.main as main_mod

    original_service = main_mod.service
    original_token = main_mod.settings.social_memory_ingest_token
    main_mod.service = _NoopService()
    main_mod.settings.social_memory_ingest_token = ""
    try:
        with TestClient(main_mod.app) as client:
            resp = _post_ingest(client, headers={"Authorization": "Bearer foo"})
        assert resp.status_code == 401
    finally:
        main_mod.service = original_service
        main_mod.settings.social_memory_ingest_token = original_token


def test_ingest_http_401_wrong_token() -> None:
    import app.main as main_mod

    original_service = main_mod.service
    original_token = main_mod.settings.social_memory_ingest_token
    main_mod.service = _NoopService()
    main_mod.settings.social_memory_ingest_token = "secret"
    try:
        with TestClient(main_mod.app) as client:
            resp = _post_ingest(client, headers={"Authorization": "Bearer other"})
        assert resp.status_code == 401
    finally:
        main_mod.service = original_service
        main_mod.settings.social_memory_ingest_token = original_token


def test_ingest_http_200_valid_body_publishes_social_turn() -> None:
    import app.main as main_mod

    bus = _FakeBus()
    svc = SocialMemoryService(settings=Settings(ORION_BUS_ENABLED=False), bus=bus)

    async def _noop_start() -> None:
        return None

    original_service = main_mod.service
    original_token = main_mod.settings.social_memory_ingest_token
    original_start = svc.start
    svc.start = _noop_start  # type: ignore[method-assign]
    main_mod.service = svc
    main_mod.settings.social_memory_ingest_token = "secret"
    try:
        with TestClient(main_mod.app) as client:
            resp = _post_ingest(client, headers={"Authorization": "Bearer secret"})
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}
        assert len(bus.published) == 1
        channel, envelope = bus.published[0]
        assert channel == "orion:chat:social:turn"
        assert envelope.kind == "social.turn.v1"
    finally:
        svc.start = original_start
        main_mod.service = original_service
        main_mod.settings.social_memory_ingest_token = original_token


def test_ingest_http_422_invalid_body() -> None:
    import app.main as main_mod

    original_service = main_mod.service
    original_token = main_mod.settings.social_memory_ingest_token
    main_mod.service = _NoopService()
    main_mod.settings.social_memory_ingest_token = "secret"
    try:
        with TestClient(main_mod.app) as client:
            resp = client.post(
                "/ingest-turn",
                headers={"Authorization": "Bearer secret"},
                json={"source": "orion-ai-town"},
            )
        assert resp.status_code == 422
    finally:
        main_mod.service = original_service
        main_mod.settings.social_memory_ingest_token = original_token


def test_social_turn_channel_lists_social_memory_as_producer() -> None:
    doc = yaml.safe_load((REPO_ROOT / "orion/bus/channels.yaml").read_text(encoding="utf-8")) or {}
    entry = next(item for item in doc.get("channels") or [] if item.get("name") == "orion:chat:social:turn")
    assert "orion-social-memory" in entry["producer_services"]
