from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

_REPO = Path(__file__).resolve().parents[3]
_ORCH = _REPO / "services" / "orion-cortex-orch"
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


class _Env:
    kind = "chat.history.turn"


def test_extractor_disabled_noop(monkeypatch) -> None:
    import app.memory_extractor as mod

    class S:
        orion_auto_extractor_enabled = False
        orion_auto_extractor_stage2_enabled = False

    monkeypatch.setattr(mod, "get_settings", lambda: S())
    asyncio.run(mod.handle_memory_history_turn(_Env()))  # type: ignore[arg-type]


def test_stage2_raises(monkeypatch) -> None:
    import app.memory_extractor as mod

    class S:
        orion_auto_extractor_enabled = True
        orion_auto_extractor_stage2_enabled = True

    monkeypatch.setattr(mod, "get_settings", lambda: S())
    with pytest.raises(NotImplementedError):
        asyncio.run(mod.handle_memory_history_turn(_Env()))  # type: ignore[arg-type]


def _turn_env(*, prompt: str) -> BaseEnvelope:
    return BaseEnvelope(
        kind="chat.history",
        source=ServiceRef(name="orion-hub"),
        correlation_id=uuid4(),
        payload={
            "source": "hub_ws",
            "prompt": prompt,
            "response": "ok",
        },
    )


def test_stage1_extracts_and_inserts(monkeypatch) -> None:
    import app.memory_extractor as mod

    monkeypatch.setattr(mod, "_memory_pool", None)
    monkeypatch.setattr(mod, "_memory_pool_failed", False)

    class S:
        orion_auto_extractor_enabled = True
        orion_auto_extractor_stage2_enabled = False

    monkeypatch.setattr(mod, "get_settings", lambda: S())

    async def fake_pool():
        return object()

    insert_mock = AsyncMock()
    exists_mock = AsyncMock(return_value=False)
    monkeypatch.setattr(mod, "_get_memory_pool", fake_pool)
    monkeypatch.setattr(mod.mc_dal, "insert_card", insert_mock)
    monkeypatch.setattr(mod.mc_dal, "card_exists_by_fingerprint", exists_mock)

    asyncio.run(mod.handle_memory_history_turn(_turn_env(prompt="I live in Ogden, UT")))

    insert_mock.assert_awaited_once()
    assert insert_mock.await_args is not None
    pool_arg, card_arg = insert_mock.await_args.args[0], insert_mock.await_args.args[1]
    assert pool_arg is not None
    assert card_arg.provenance == "auto_extractor"
    assert card_arg.status == "pending_review"
    assert "Ogden" in (card_arg.summary or "")
    assert insert_mock.await_args.kwargs.get("actor") == "auto_extractor"
    assert (card_arg.subschema or {}).get("auto_extractor_fingerprint")


def test_stage1_dedupes_via_fingerprint(monkeypatch) -> None:
    import app.memory_extractor as mod

    monkeypatch.setattr(mod, "_memory_pool", None)
    monkeypatch.setattr(mod, "_memory_pool_failed", False)

    class S:
        orion_auto_extractor_enabled = True
        orion_auto_extractor_stage2_enabled = False

    monkeypatch.setattr(mod, "get_settings", lambda: S())

    async def fake_pool():
        return object()

    insert_mock = AsyncMock()
    exists_mock = AsyncMock(return_value=True)
    monkeypatch.setattr(mod, "_get_memory_pool", fake_pool)
    monkeypatch.setattr(mod.mc_dal, "insert_card", insert_mock)
    monkeypatch.setattr(mod.mc_dal, "card_exists_by_fingerprint", exists_mock)

    asyncio.run(mod.handle_memory_history_turn(_turn_env(prompt="I live in Ogden, UT")))

    insert_mock.assert_not_called()


def test_stage1_noop_when_pool_unavailable(monkeypatch) -> None:
    import app.memory_extractor as mod

    monkeypatch.setattr(mod, "_memory_pool", None)
    monkeypatch.setattr(mod, "_memory_pool_failed", False)

    class S:
        orion_auto_extractor_enabled = True
        orion_auto_extractor_stage2_enabled = False

    monkeypatch.setattr(mod, "get_settings", lambda: S())

    async def no_pool():
        return None

    insert_mock = AsyncMock()
    exists_mock = AsyncMock()
    monkeypatch.setattr(mod, "_get_memory_pool", no_pool)
    monkeypatch.setattr(mod.mc_dal, "insert_card", insert_mock)
    monkeypatch.setattr(mod.mc_dal, "card_exists_by_fingerprint", exists_mock)

    asyncio.run(mod.handle_memory_history_turn(_turn_env(prompt="I live in Ogden, UT")))

    insert_mock.assert_not_called()
    exists_mock.assert_not_called()
