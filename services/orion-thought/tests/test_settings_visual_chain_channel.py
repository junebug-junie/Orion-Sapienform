"""Regression guard (review finding, 2026-08-26): a bad merge silently
reverting channel_vision_host_request's default back to the shared bare
channel would pass every existing visual_chain.py test, since all of them
mock bus.rpc_request and none previously asserted on which channel it was
called with. This pins both the setting's own default AND that
request_caption() actually reads it (not a hardcoded string) -- see
test_visual_chain.py::test_run_visual_chain_once_success for the
call-args assertion on the live code path.
"""
from __future__ import annotations

import importlib


def test_channel_vision_host_request_defaults_to_circe_qwen_lane(monkeypatch):
    monkeypatch.delenv("CHANNEL_VISION_HOST_REQUEST", raising=False)
    import app.settings as s

    importlib.reload(s)
    assert s.settings.channel_vision_host_request == (
        "orion:exec:request:VisionHostService:circe-vl"
    ), (
        "athena's shared BLIP-base instance cannot produce a caption real "
        "enough to clear sanitize_caption for a generated image (live-"
        "confirmed 3/3 ticks, 2026-08-25) -- this must stay pointed at "
        "circe's dedicated Qwen2-VL lane, not silently revert to the "
        "shared bare channel."
    )
