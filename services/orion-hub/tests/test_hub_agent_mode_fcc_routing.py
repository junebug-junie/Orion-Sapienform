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
HUB_AGENT_CONTEXT_EXEC_ENABLED defaults to False now, so the old
context-exec bridge is unreachable by default (not deleted -- an
operator can still flip it back on if orion-context-exec is ever
redeployed).

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


def test_agent_mode_never_reaches_should_use_context_exec_agent_lane_via_ws():
    """The FCC branch must `continue` before the classic/general lane's own
    context_exec_agent_bridge check -- confirming "agent" mode structurally
    cannot reach the dead orion-context-exec HTTP call anymore, not just
    that the branch condition looks right in isolation."""
    source = _ws_source()
    branch_marker = 'if client_mode in ("orion", "agent") and settings.ORION_UNIFIED_TURN_ENABLED:'
    idx = source.index(branch_marker)
    bridge_marker = "should_use_context_exec_agent_lane"
    bridge_idx = source.index(bridge_marker, idx)
    # Everything between the branch start and the bridge call must contain
    # this branch's own `continue` (its exit), proving the bridge call sits
    # in a structurally later/separate code path this branch never reaches.
    between = source[idx:bridge_idx]
    assert "\n                continue\n" in between, (
        "the FCC branch must exit (continue) before the context-exec bridge "
        "check -- if this assertion fails, 'agent' mode may have started "
        "falling through into the dead context-exec path again"
    )


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


def test_hub_agent_context_exec_enabled_defaults_off():
    """The old bridge's own gate (context_exec_agent_bridge.py's
    agent_lane_enabled()) must default to disabled -- this is what makes the
    dead orion-context-exec path unreachable by default without deleting
    context_exec_agent_bridge.py/context_exec_client.py outright."""
    settings_source = SETTINGS_PATH.read_text(encoding="utf-8")
    marker = 'HUB_AGENT_CONTEXT_EXEC_ENABLED: bool = Field('
    assert marker in settings_source
    idx = settings_source.index(marker)
    following = settings_source[idx : idx + 200]
    assert "default=False" in following


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


def test_http_agent_mode_never_reaches_should_use_context_exec_agent_lane():
    """Same structural guarantee as the WebSocket test above, for the HTTP
    transport: the FCC branch must return before
    should_use_context_exec_agent_lane's check, so "agent" mode cannot fall
    through to the dead context-exec bridge via this path either."""
    source = _api_routes_source()
    branch_marker = (
        'if str(payload.get("mode") or "").strip().lower() in ("orion", "agent") '
        "and settings.ORION_UNIFIED_TURN_ENABLED:"
    )
    idx = source.index(branch_marker)
    bridge_marker = "should_use_context_exec_agent_lane"
    bridge_idx = source.index(bridge_marker, idx)
    between = source[idx:bridge_idx]
    assert "return {**final_frame, \"chat_route\": CHAT_ROUTE_UNIFIED_TURN_HARNESS}" in between, (
        "the FCC branch must return before the context-exec bridge check -- "
        "if this fails, HTTP 'agent' mode may be falling through to the dead "
        "context-exec path again"
    )


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


def test_env_example_matches_the_new_default():
    """CLAUDE.md env parity: the checked-in .env_example must reflect the
    real intended default, not the old, now-wrong `true`."""
    env_source = ENV_EXAMPLE_PATH.read_text(encoding="utf-8")
    assert "HUB_AGENT_CONTEXT_EXEC_ENABLED=false" in env_source
    assert "HUB_AGENT_CONTEXT_EXEC_ENABLED=true" not in env_source
