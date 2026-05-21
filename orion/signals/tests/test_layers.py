from orion.signals.layers import ORGAN_LAYER, organ_layer, layers_export


def test_organ_layer_runtime_organs() -> None:
    assert organ_layer("cortex_exec") == "runtime"
    assert organ_layer("llm_gateway") == "runtime"
    assert organ_layer("cortex_gateway") == "runtime"


def test_organ_layer_cognition_organs() -> None:
    assert organ_layer("chat_stance") == "cognition"
    assert organ_layer("recall") == "cognition"
    assert organ_layer("spark_introspector") == "cognition"


def test_layers_export_includes_all_registry_organs() -> None:
    body = layers_export()
    assert "runtime" in body["options"]
    assert body["organs"]["equilibrium"] == "infra"
    assert len(ORGAN_LAYER) >= 20
