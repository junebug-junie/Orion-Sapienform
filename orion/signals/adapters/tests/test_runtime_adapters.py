import pytest

from orion.signals.adapters.cortex_gateway import CortexGatewayAdapter
from orion.signals.adapters.cortex_orch import CortexOrchAdapter
from orion.signals.adapters.hub import HubAdapter
from orion.signals.adapters.persistence_writers import SqlWriterAdapter
from orion.signals.normalization import NormalizationContext
from orion.signals.registry import ORGAN_REGISTRY


@pytest.fixture
def norm_ctx() -> NormalizationContext:
    return NormalizationContext()


def test_cortex_gateway_route_decision(norm_ctx: NormalizationContext) -> None:
    adapter = CortexGatewayAdapter()
    payload = {
        "correlation_id": "gw-1",
        "mode": "brain",
        "verb": "chat_general",
        "context": {"messages": []},
        "recall": {"enabled": True},
    }
    signal = adapter.adapt("orion:cortex:gateway:request", payload, ORGAN_REGISTRY, {}, norm_ctx)
    assert signal is not None
    assert signal.organ_id == "cortex_gateway"
    assert signal.signal_kind == "route_decision"
    assert "stub adapter" not in " ".join(signal.notes)


def test_cortex_orch_plan_resolution(norm_ctx: NormalizationContext) -> None:
    gw = CortexGatewayAdapter()
    gw_sig = gw.adapt(
        "orion:cortex:gateway:request",
        {"correlation_id": "gw-1", "mode": "brain", "verb": "chat_general", "context": {}, "recall": {}},
        ORGAN_REGISTRY,
        {},
        norm_ctx,
    )
    adapter = CortexOrchAdapter()
    payload = {
        "correlation_id": "orch-1",
        "mode": "brain",
        "verb": "chat_general",
        "packs": ["core"],
        "context": {"messages": []},
        "recall": {},
    }
    signal = adapter.adapt("cortex.orch.request", payload, ORGAN_REGISTRY, {"cortex_gateway": gw_sig}, norm_ctx)
    assert signal is not None
    assert signal.organ_id == "cortex_orch"
    assert signal.signal_kind == "plan_resolution"
    assert gw_sig is not None and gw_sig.signal_id in signal.causal_parents


def test_hub_chat_turn_no_message_text(norm_ctx: NormalizationContext) -> None:
    adapter = HubAdapter()
    payload = {
        "turn_id": "t-1",
        "session_id": "s-1",
        "prompt": "secret user text",
        "response": "secret assistant text",
        "metadata": {"correlation_id": "corr-hub-1"},
    }
    signal = adapter.adapt("chat.history.turn", payload, ORGAN_REGISTRY, {}, norm_ctx)
    assert signal is not None
    assert signal.organ_id == "hub"
    assert signal.signal_kind == "chat_turn"
    assert "secret" not in (signal.summary or "")


def test_hub_chat_turn_top_level_correlation_without_metadata(norm_ctx: NormalizationContext) -> None:
    adapter = HubAdapter()
    payload = {
        "turn_id": "t-1",
        "session_id": "s-1",
        "correlation_id": "corr-hub-top",
        "prompt": "hello",
        "response": "hi",
    }
    signal = adapter.adapt("chat.history.turn", payload, ORGAN_REGISTRY, {}, norm_ctx)
    assert signal is not None
    assert signal.source_event_id == "corr-hub-top"


def test_sql_writer_persist(norm_ctx: NormalizationContext) -> None:
    adapter = SqlWriterAdapter()
    signal = adapter.adapt(
        "collapse.mirror.stored.v1",
        {"entry_id": "e-1", "latency_ms": 120},
        ORGAN_REGISTRY,
        {},
        norm_ctx,
    )
    assert signal is not None
    assert signal.organ_id == "sql_writer"
    assert signal.signal_kind == "persist"
