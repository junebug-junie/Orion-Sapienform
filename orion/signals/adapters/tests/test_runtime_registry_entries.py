from orion.signals.registry import ORGAN_REGISTRY


def test_cortex_exec_registry_entry() -> None:
    # No active adapter as of 2026-07-28 (CognitionTraceAdapter removed -- see
    # docs/superpowers/specs/2026-07-28-cognition-trace-signal-gateway-consumer-audit.md).
    # Entry itself is kept only because llm_gateway references it as a causal parent.
    entry = ORGAN_REGISTRY["cortex_exec"]
    assert entry.signal_kinds == []
    assert entry.bus_channels == []


def test_llm_gateway_registry_entry() -> None:
    assert ORGAN_REGISTRY["llm_gateway"].organ_id == "llm_gateway"
