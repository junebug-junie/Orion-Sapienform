from __future__ import annotations

import importlib
import sys
from pathlib import Path

HUB_ROOT = Path(__file__).resolve().parents[1]
if str(HUB_ROOT) not in sys.path:
    sys.path.insert(0, str(HUB_ROOT))

builder = importlib.import_module("scripts.cortex_request_builder")


def test_build_chat_request_includes_presence_metadata():
    req, debug, _ = builder.build_chat_request(
        payload={
            "mode": "brain",
            "presence_context": {"audience_mode": "kid_present"},
            "surface_context": {"surface": "hub_desktop", "input_modality": "typed"},
            "browser_client_id": "browser-1",
        },
        session_id="sid-1",
        user_id="juniper",
        trace_id="trace-1",
        default_mode="brain",
        auto_default_enabled=False,
        source_label="hub_test",
        prompt="hello",
        messages=[{"role": "user", "content": "hello"}],
    )
    assert req.metadata["presence_context"]["audience_mode"] == "kid_present"
    assert req.metadata["surface_context"]["surface"] == "hub_desktop"
    assert req.metadata["browser_client_id"] == "browser-1"
    assert debug["mode"] == "brain"
