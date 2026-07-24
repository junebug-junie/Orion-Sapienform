"""Gateway must subscribe to the rpc_health channel pattern and produce real,
distinct-per-service signals end-to-end (Step 3 of docs/superpowers/specs/
2026-07-23-rpc-health-signal-gateway-wiring-design.md's own acceptance checks)."""
from datetime import datetime, timezone
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from app.settings import settings
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef


def test_organ_channels_includes_rpc_health() -> None:
    patterns = settings.ORGAN_CHANNELS
    assert any(
        p in ("orion:rpc_health:*", "orion:rpc_health:snapshot") for p in patterns
    ), f"ORGAN_CHANNELS missing rpc_health pattern: {patterns}"


def _rpc_health_payload(service: str) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "service": service,
        "node": "athena",
        "instance": None,
        "window_start": now,
        "window_end": now,
        "success_count": 9,
        "timeout_count": 1,
        "success_latency_ms_p50": 10.0,
        "success_latency_ms_p95": 25.0,
        "success_latency_ms_max": 30.0,
        "timeout_elapsed_ms_max": 5000.0,
        "channel_counts": {"orion:cortex:exec:request": 10},
        "truncated": False,
    }


@pytest.mark.asyncio
async def test_gateway_processor_produces_rpc_health_signal_end_to_end() -> None:
    """Uses the REAL ADAPTERS list (not a monkeypatched stub) so this actually proves
    the new adapter is registered and reachable through the gateway's real dispatch
    path, not just callable in isolation."""
    from app.normalization_state import NormalizationStateRegistry
    from app.processor import SignalProcessor
    from app.signal_window import SignalWindow

    bus = AsyncMock()
    proc = SignalProcessor(
        bus=bus,
        signal_window=SignalWindow(30.0),
        norm_state=NormalizationStateRegistry(),
        output_channel_prefix="orion:signals",
        passthrough_pattern="orion:signals:*",
        service_ref=ServiceRef(name="orion-signal-gateway", version="0.1.0", node="n"),
    )
    env = BaseEnvelope(
        kind="rpc_health.snapshot.v1",
        source=ServiceRef(name="orion-cortex-exec", version="0.1.1", node="athena"),
        correlation_id=UUID("00000000-0000-4000-8000-0000000000a1"),
        payload=_rpc_health_payload("orion-cortex-exec"),
    )
    await proc.handle_envelope(env)

    assert bus.publish.await_count >= 1
    channels = [c.args[0] for c in bus.publish.await_args_list]
    assert "orion:signals:rpc_health_cortex_exec" in channels


@pytest.mark.asyncio
async def test_gateway_signal_window_does_not_collide_across_producer_services() -> None:
    """The real bug found in review: a shared organ_id made cortex-exec's and
    cortex-orch's snapshots overwrite each other in SignalWindow's current-state view.
    Feed both services' envelopes through the real processor and confirm the window
    retains both, not just whichever published last."""
    from app.normalization_state import NormalizationStateRegistry
    from app.processor import SignalProcessor
    from app.signal_window import SignalWindow

    bus = AsyncMock()
    window = SignalWindow(30.0)
    proc = SignalProcessor(
        bus=bus,
        signal_window=window,
        norm_state=NormalizationStateRegistry(),
        output_channel_prefix="orion:signals",
        passthrough_pattern="orion:signals:*",
        service_ref=ServiceRef(name="orion-signal-gateway", version="0.1.0", node="n"),
    )
    for service, corr in (
        ("orion-cortex-exec", "00000000-0000-4000-8000-0000000000b1"),
        ("orion-cortex-orch", "00000000-0000-4000-8000-0000000000b2"),
    ):
        env = BaseEnvelope(
            kind="rpc_health.snapshot.v1",
            source=ServiceRef(name=service, version="0.1.1", node="athena"),
            correlation_id=UUID(corr),
            payload=_rpc_health_payload(service),
        )
        await proc.handle_envelope(env)

    exec_signal = window.get("rpc_health_cortex_exec")
    orch_signal = window.get("rpc_health_cortex_orch")
    assert exec_signal is not None
    assert orch_signal is not None
    assert exec_signal.organ_id != orch_signal.organ_id
