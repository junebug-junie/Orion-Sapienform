from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
HUB_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT, HUB_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

os.environ.setdefault("CHANNEL_VOICE_TRANSCRIPT", "orion:voice:transcript")
os.environ.setdefault("CHANNEL_VOICE_LLM", "orion:voice:llm")
os.environ.setdefault("CHANNEL_VOICE_TTS", "orion:voice:tts")

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from scripts.bus_clients.tts_client import TTSClient
from scripts.settings import settings
from orion.schemas.tts import TTSRequestPayload


@pytest.mark.asyncio
async def test_speak_raises_on_system_error() -> None:
    bus = MagicMock()
    bus.codec.decode.return_value = MagicMock(
        ok=True,
        envelope=BaseEnvelope(
            kind="system.error",
            source=ServiceRef(name="whisper-tts", version="0.1.0"),
            correlation_id="00000000-0000-4000-8000-000000000001",
            payload={"error": "tts_synthesis_failed", "details": "XTTS speaker missing"},
        ),
    )
    bus.rpc_request = AsyncMock(return_value={"data": b"x"})
    client = TTSClient(bus)

    with pytest.raises(ValueError, match="TTS error: XTTS speaker missing"):
        await client.speak(TTSRequestPayload(text="hello"))
