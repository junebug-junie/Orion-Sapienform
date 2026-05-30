from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SERVICE_ROOT.parents[1]
for p in (str(REPO_ROOT), str(SERVICE_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.tts import TTSOutput  # noqa: E402


@pytest.mark.asyncio
async def test_typed_reply_includes_metadata() -> None:
    from app.tts_worker import process_tts_request
    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
    from orion.schemas.tts import TTSRequestPayload

    bus = AsyncMock()
    envelope = BaseEnvelope(
        kind="tts.synthesize.request",
        source=ServiceRef(name="hub", version="0.1.0"),
        correlation_id="cid-1",
        reply_to="orion:tts:result:test",
        payload=TTSRequestPayload(
            text="hi",
            voice_id="Ana Florence",
            language="en",
        ).model_dump(),
    )
    fake = TTSOutput(
        audio_b64="YWJj",
        metadata={"backend": "coqui", "model_name": "xtts", "language": "en"},
        duration_sec=1.0,
    )

    with patch("app.tts_worker.get_tts_engine") as get_engine:
        get_engine.return_value.synthesize_to_b64.return_value = fake
        with patch("app.tts_worker.settings") as st:
            st.whisper_tts_synth_timeout_sec = 30.0
            st.service_name = "whisper-tts"
            st.service_version = "0.1.0"
            await process_tts_request(bus, envelope, {})

    bus.publish.assert_awaited_once()
    published = bus.publish.await_args[0][1]
    assert published.payload["audio_b64"] == "YWJj"
    assert published.payload["metadata"]["backend"] == "coqui"
