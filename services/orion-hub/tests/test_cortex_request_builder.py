from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "cortex_request_builder.py"
SPEC = importlib.util.spec_from_file_location("hub_cortex_request_builder", MODULE_PATH)
assert SPEC and SPEC.loader
hub_builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hub_builder)


def test_agent_mode_emits_supervised_delivery_ready_request() -> None:
    req, debug, _ = hub_builder.build_chat_request(
        payload={
            "mode": "agent",
            "packs": ["executive_pack"],
        },
        session_id="sid-agent",
        user_id="user-1",
        trace_id="trace-agent",
        default_mode="brain",
        auto_default_enabled=False,
        source_label="hub_http",
        prompt="Write me a deployment guide for this service.",
    )

    assert req.mode == "agent"
    assert req.route_intent == "none"
    assert req.verb is None
    assert req.options["supervised"] is True
    assert debug["selected_ui_route"] == "agent"
    assert debug["supervised"] is True
    assert debug["force_agent_chain"] is False
    assert req.packs == ["executive_pack"]
    assert req.metadata["hub_route"]["selected_ui_route"] == "agent"


def test_auto_mode_emits_auto_route_intent_without_forcing_supervisor() -> None:
    req, debug, _ = hub_builder.build_chat_request(
        payload={
            "mode": "auto",
            "packs": ["executive_pack", "memory_pack"],
        },
        session_id="sid-auto",
        user_id="user-2",
        trace_id="trace-auto",
        default_mode="brain",
        auto_default_enabled=False,
        source_label="hub_ws",
        prompt="Compare Docker Compose and Kubernetes for this deployment.",
    )

    assert req.mode == "auto"
    assert req.route_intent == "auto"
    assert req.options["route_intent"] == "auto"
    assert req.options.get("supervised") is None
    assert debug["selected_ui_route"] == "auto"
    assert debug["supervised"] is False
    assert debug["packs"] == ["executive_pack", "memory_pack"]
    assert req.metadata["hub_route"]["selected_ui_route"] == "auto"
