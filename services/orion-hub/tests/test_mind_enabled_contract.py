from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "cortex_request_builder.py"
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SPEC = importlib.util.spec_from_file_location("hub_cortex_request_builder_mind", MODULE_PATH)
assert SPEC and SPEC.loader
hub_builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hub_builder)

APP_JS = Path(__file__).resolve().parents[1] / "static" / "js" / "app.js"
THOUGHT_JS = Path(__file__).resolve().parents[1] / "static" / "js" / "thought-process.js"


def _build(payload: dict) -> tuple:
    return hub_builder.build_chat_request(
        payload=payload,
        session_id="sid-mind",
        user_id="user-1",
        trace_id="trace-mind",
        default_mode="brain",
        auto_default_enabled=False,
        source_label="hub_ws",
        prompt="hello grounded small",
    )


def test_grounded_small_payload_sets_metadata_mind_enabled_true() -> None:
    req, debug, _ = _build(
        {
            "mode": "brain",
            "verbs": [],
            "options": {"llm_route": "quick"},
            "surface_context": {"hub_chat_lane": "grounded_small"},
            "context": {"metadata": {"mind_enabled": True}},
        }
    )
    assert req.metadata.get("mind_enabled") is True
    assert debug.get("mind_requested") is True
    assert debug.get("mind_requested_source") == "payload.context.metadata.mind_enabled"


def test_context_metadata_mind_enabled_true_passes_through() -> None:
    req, debug, _ = _build({"context": {"metadata": {"mind_enabled": True}}})
    assert req.metadata.get("mind_enabled") is True
    assert debug.get("mind_requested") is True


def test_top_level_mind_enabled_true_is_normalized() -> None:
    req, debug, _ = _build({"mind_enabled": True})
    assert req.metadata.get("mind_enabled") is True
    assert debug.get("mind_requested") is True
    assert debug.get("mind_requested_source") == "payload.mind_enabled"


def test_string_true_in_context_metadata_is_normalized() -> None:
    req, debug, _ = _build({"context": {"metadata": {"mind_enabled": "true"}}})
    assert req.metadata.get("mind_enabled") is True
    assert debug.get("mind_requested") is True
    assert debug.get("mind_enabled_metadata_type") == "str"


def test_missing_mind_enabled_reports_not_requested() -> None:
    req, debug, _ = _build({"mode": "brain"})
    assert "mind_enabled" not in req.metadata
    assert debug.get("mind_requested") is False
    assert debug.get("mind_requested_source") is None


def test_mind_requested_from_payload_helper_diagnostics() -> None:
    requested, diag = hub_builder._mind_requested_from_payload(
        {"options": {"mind_enabled": "yes"}, "context": {"metadata": {"mind_enabled": False}}}
    )
    assert requested is True
    assert diag["mind_requested_source"] == "payload.options.mind_enabled"


def test_hub_mind_panel_distinguishes_not_requested() -> None:
    app_js = APP_JS.read_text(encoding="utf-8")
    assert "resolveMindRunsEmptyStatus" in app_js
    assert "Mind was not requested for this turn." in app_js
    assert "Mind was requested but invocation failed." in app_js
    assert "Mind ran but artifact publish failed." in app_js


def test_grounded_small_lane_js_sets_context_metadata_mind_enabled() -> None:
    thought_js = THOUGHT_JS.read_text(encoding="utf-8")
    assert "payload.context.metadata.mind_enabled = true" in thought_js
    assert "LANE_GROUNDED_SMALL" in thought_js
