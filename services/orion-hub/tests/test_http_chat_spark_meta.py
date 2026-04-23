from __future__ import annotations

from scripts.api_routes import _http_chat_turn_context_from_result


def test_http_chat_turn_context_merges_gateway_metadata_including_turn_effect() -> None:
    result = {
        "mode": "brain",
        "use_recall": True,
        "routing_debug": {"verb": "chat_quick"},
        "raw": {
            "metadata": {
                "turn_effect": {"turn": {"novelty": 0.42}},
                "turn_effect_status": "present",
                "model": "test-model",
            },
        },
    }
    ctx = _http_chat_turn_context_from_result(result)
    sm = ctx["spark_meta"]
    assert sm["trace_verb"] == "chat_quick"
    assert sm["turn_effect"]["turn"]["novelty"] == 0.42
    assert sm["turn_effect_status"] == "present"
    assert ctx["gateway_meta"]["model"] == "test-model"
