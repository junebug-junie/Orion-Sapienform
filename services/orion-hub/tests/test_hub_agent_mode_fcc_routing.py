"""2026-09-02: Hub "Agent" Mode routes through FCC (same mechanism as
"Orion" Mode), not orion-context-exec.

Real incident: `orion-context-exec` has zero containers deployed on
athena (confirmed live) -- every Hub Agent-mode turn failed with
"context-exec run unreachable" (services/orion-hub/scripts/
context_exec_client.py). Juniper: "context exec is failed prototype";
she asked for Agent mode to spawn via the FCC/harness-governor
`claude -p` subprocess like Orion mode already does, not a different
backend.

Fix: `client_mode in ("orion", "agent")` now takes the same
run_unified_turn branch in websocket_handler.py, tagged by client_mode
(not a hardcoded "orion" literal) for tracing/cancellation/TTS lane.
2026-10-10: the old context-exec bridge, its HUB_AGENT_CONTEXT_EXEC_ENABLED
gate, and orion-context-exec itself were deleted outright (kill means kill).

This repo has no full WebSocket TestClient harness for
websocket_handler.py (see test_orion_unified_turn_tts.py's own
docstring) -- these are real source control-flow shape assertions, the
same convention that file already established, not a reimplementation
of the dispatch logic.
"""
from __future__ import annotations

from pathlib import Path

HUB_ROOT = Path(__file__).resolve().parents[1]
WS_PATH = HUB_ROOT / "scripts" / "websocket_handler.py"
API_ROUTES_PATH = HUB_ROOT / "scripts" / "api_routes.py"
SETTINGS_PATH = HUB_ROOT / "app" / "settings.py"
ENV_EXAMPLE_PATH = HUB_ROOT / ".env_example"


def _ws_source() -> str:
    return WS_PATH.read_text(encoding="utf-8")


def test_agent_mode_shares_the_orion_fcc_branch():
    """The actual fix: "agent" must be in the same condition as "orion",
    not a separate branch that could silently diverge again."""
    source = _ws_source()
    assert 'if client_mode in ("orion", "agent") and settings.ORION_UNIFIED_TURN_ENABLED:' in source


def test_active_turn_kind_is_tagged_by_client_mode_not_hardcoded_orion():
    """Regression guard for the old hardcoded `active_turn["kind"] = "orion"`
    -- an Agent-mode turn tagged "orion" would misdirect
    turn_cancel.py's kind-based dispatch bookkeeping (harmless today since
    both currently resolve to the same default cancel path, but still a
    real mislabel worth catching)."""
    source = _ws_source()
    assert 'active_turn["kind"] = client_mode' in source
    assert 'active_turn["kind"] = "orion"' not in source


def test_cancel_and_tts_calls_are_also_tagged_by_client_mode():
    source = _ws_source()
    assert "kind=client_mode," in source
    assert "lane=client_mode," in source
    # The only remaining "orion"-literal `kind=`/`lane=` should be the
    # agent-claude branch's own unrelated tagging, not this one.
    assert 'kind="orion",' not in source
    assert 'lane="orion",' not in source


def _api_routes_source() -> str:
    return API_ROUTES_PATH.read_text(encoding="utf-8")


def test_http_fallback_agent_mode_also_shares_the_fcc_branch():
    """Review finding, 2026-09-02: an earlier version of this fix only
    widened websocket_handler.py's condition, silently leaving the HTTP
    /api/chat fallback routing "agent" through the plain cortex_client.chat()
    path instead -- a different, untested behavior change (a degraded
    "context-exec disabled" response, not FCC), not what this PR claims to
    fix. Both transports must share the same widened condition."""
    source = _api_routes_source()
    assert (
        'if str(payload.get("mode") or "").strip().lower() in ("orion", "agent") '
        "and settings.ORION_UNIFIED_TURN_ENABLED:" in source
    )
    # The old exact-match form must be gone, not just supplemented -- an
    # earlier draft could satisfy the assertion above while ALSO leaving a
    # stale `== "orion"` check reachable first.
    assert 'str(payload.get("mode") or "").strip().lower() == "orion"' not in source


def test_success_frames_and_chat_history_tag_the_real_mode_not_a_hardcoded_orion():
    """Live-caught, 2026-09-02: a real HTTP Agent-mode turn against athena's
    running Hub came back with chat_route="unified_turn_harness" (routing
    confirmed correct) but the final frame's own "mode" field said "orion"
    -- turn_orchestrator.py's _success_frames/_publish_unified_turn_chat_history
    hardcoded "orion" regardless of caller, which would have permanently
    mislabeled every persisted Agent-mode chat_history_log row too. Source
    assertions (turn_orchestrator.py has no dedicated test module of its own
    isolated by mode value) rather than a full execute_unified_turn mock,
    matching this repo's existing convention for this exact function
    (test_turn_orchestrator_ws_frames.py's own docstrings)."""
    orch_path = HUB_ROOT.parents[1] / "orion" / "hub" / "turn_orchestrator.py"
    source = orch_path.read_text(encoding="utf-8")
    assert '"mode": "orion",' not in source
    assert '"mode": mode_tag,' in source
    assert 'mode_tag = str(payload.get("mode") or "orion").strip().lower()' in source
    # Both _success_frames call sites inside execute_unified_turn must pass
    # it through -- a fix that only updated the default-frame call site
    # (the finalize_ran path) would leave the finalize_degraded_reason path
    # still silently mislabeling degraded Agent-mode turns.
    assert source.count("mode_tag=mode_tag,") >= 1


def test_context_exec_agent_lane_is_deleted_not_just_gated():
    """2026-10-10 retirement: the Hub must have no path back into
    orion-context-exec -- not a default-off flag, not an unreachable branch.
    The bridge/client modules are gone and neither transport references them."""
    scripts_dir = HUB_ROOT / "scripts"
    for gone in ("context_exec_agent_bridge.py", "context_exec_client.py", "agent_step_relay.py",
                 "proposal_review_client.py", "proposal_review_routes.py"):
        assert not (scripts_dir / gone).exists(), gone
    for source in (_ws_source(), _api_routes_source()):
        assert "context_exec" not in source
        assert "should_use_context_exec_agent_lane" not in source
    settings_source = SETTINGS_PATH.read_text(encoding="utf-8")
    env_source = ENV_EXAMPLE_PATH.read_text(encoding="utf-8")
    for key in ("HUB_AGENT_CONTEXT_EXEC_ENABLED", "HUB_CONTEXT_EXEC_API_URL", "HUB_CONTEXT_EXEC_EVENT_CHANNEL",
                "HUB_PROPOSAL_REVIEW_ENABLED", "HUB_PROPOSAL_REVIEW_API_URL", "CONTEXT_EXEC_INVESTIGATION_V2_ENABLED"):
        assert key not in settings_source, key
        assert key not in env_source, key
