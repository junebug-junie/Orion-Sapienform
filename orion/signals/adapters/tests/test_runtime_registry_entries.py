from orion.signals.registry import ORGAN_REGISTRY


def test_cortex_exec_registry_entry() -> None:
    entry = ORGAN_REGISTRY["cortex_exec"]
    assert "cognition_run" in entry.signal_kinds
    assert "orion:cognition:trace" in entry.bus_channels


def test_llm_gateway_registry_entry() -> None:
    assert ORGAN_REGISTRY["llm_gateway"].organ_id == "llm_gateway"
