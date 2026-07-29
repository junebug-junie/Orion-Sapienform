"""Hub LLM route selector and context-exec agent lane tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

HUB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for candidate in (str(REPO_ROOT), str(HUB_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

for key, value in {
    "CHANNEL_VOICE_TRANSCRIPT": "orion:voice:transcript",
    "CHANNEL_VOICE_LLM": "orion:voice:llm",
    "CHANNEL_VOICE_TTS": "orion:voice:tts",
    "CHANNEL_COLLAPSE_INTAKE": "orion:collapse:intake",
    "CHANNEL_COLLAPSE_TRIAGE": "orion:collapse:triage",
}.items():
    os.environ.setdefault(key, value)

from scripts.context_exec_agent_bridge import (
    build_context_exec_request,
    format_agent_operator_inline,
    normalize_llm_profile,
    should_use_context_exec_agent_lane,
)
from scripts.cortex_request_builder import build_chat_request
from orion.schemas.context_exec import (
    ContextExecOperatorSummaryV1,
    ContextExecRunV1,
    ContextExecSafetySummaryV1,
)
from orion.schemas.cortex.contracts import CortexChatRequest

INDEX_HTML = HUB_ROOT / "templates" / "index.html"
APP_JS = HUB_ROOT / "static" / "js" / "app.js"
THOUGHT_JS = HUB_ROOT / "static" / "js" / "thought-process.js"


def test_hub_renders_mode_dropdown() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert 'id="hubModeSelect"' in html
    assert '<option value="orion"' in html
    assert "<option value=\"agent\">Agent</option>" in html
    assert 'id="llmRouteSelector"' not in html
    assert 'class="llm-route-btn"' not in html


def test_hub_mode_specs_include_orion_unified_turn() -> None:
    src = APP_JS.read_text(encoding="utf-8")
    assert "orion: { mode: 'orion'" in src
    assert "fccModelLabel: 'MODEL_SONNET'" in src


def test_render_hub_index_html_reads_template_from_disk() -> None:
    import scripts.main as hub_main

    rendered = hub_main.render_hub_index_html(memory_pool_ok=False)
    assert 'id="hubModeSelect"' in rendered
    assert '<option value="orion"' in rendered
    assert "{{HUB_UI_ASSET_VERSION}}" not in rendered


def test_hub_renders_compute_dropdown_not_chip_row() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert 'id="hubComputeSelect"' in html
    assert 'role="radiogroup" aria-label="LLM route"' not in html
    assert 'data-llm-route="chat"' not in html
    assert ">Compute</label>" in html or "for=\"hubComputeSelect\"" in html
    assert ">Route</span>" not in html


def test_app_js_compute_default_is_quick() -> None:
    src = APP_JS.read_text(encoding="utf-8")
    assert "HUB_COMPUTE_DEFAULT = 'quick'" in src
    assert "orion_llm_route') || HUB_COMPUTE_DEFAULT" in src
    assert "llm-route-btn" not in src
    assert "applyLlmRouteButtonSelection" not in src


def test_app_js_down_route_suggests_quick_fallback() -> None:
    src = APP_JS.read_text(encoding="utf-8")
    assert "Use quick instead" in src
    assert "return HUB_COMPUTE_DEFAULT" in src
    assert "Use chat" not in src.split("confirmDownRouteOrProceed")[1].split("async function")[0]


def test_app_js_polls_llm_routes() -> None:
    src = APP_JS.read_text(encoding="utf-8")
    assert "/api/llm-routes" in src
    assert "setInterval" in src
    assert "loadLlmRouteCatalog" in src


def test_regression_no_route_radiogroup_chip_row() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    for route_id in ("chat", "quick", "agent", "metacog"):
        assert f'data-llm-route="{route_id}"' not in html
    assert 'role="radiogroup" aria-label="LLM route"' not in html


def test_normalize_llm_profile_defaults_to_quick() -> None:
    assert normalize_llm_profile(None) == "quick"
    assert normalize_llm_profile("AGENT") == "agent"
    assert normalize_llm_profile("bogus") == "quick"


def test_build_chat_request_includes_llm_route_in_options() -> None:
    req, debug, _ = build_chat_request(
        payload={"mode": "brain", "llm_route": "quick"},
        session_id="s1",
        user_id="u1",
        trace_id="t1",
        default_mode="brain",
        auto_default_enabled=False,
        source_label="test",
        prompt="hello",
        messages=[{"role": "user", "content": "hello"}],
    )
    assert req.options.get("llm_route") == "quick"
    assert debug.get("llm_route") == "quick"


def test_build_chat_request_defaults_llm_route_to_quick() -> None:
    req, debug, _ = build_chat_request(
        payload={"mode": "brain"},
        session_id="s1",
        user_id=None,
        trace_id="t1",
        default_mode="brain",
        auto_default_enabled=False,
        source_label="test",
        prompt="hello",
        messages=[{"role": "user", "content": "hello"}],
    )
    assert req.options.get("llm_route") == "quick"
    assert debug.get("llm_route") == "quick"


def test_should_use_context_exec_agent_lane_when_enabled() -> None:
    req = CortexChatRequest(prompt="x", mode="agent")
    with patch("scripts.context_exec_agent_bridge.agent_lane_enabled", return_value=True):
        assert should_use_context_exec_agent_lane(req) is True
    with patch("scripts.context_exec_agent_bridge.agent_lane_enabled", return_value=False):
        assert should_use_context_exec_agent_lane(req) is False


def test_build_context_exec_request_sets_llm_profile() -> None:
    req = CortexChatRequest(prompt="trace autopsy corr abc", mode="agent", trace_id="corr-1")
    with patch("scripts.context_exec_agent_bridge.agent_repl_enabled", return_value=False):
        body = build_context_exec_request(req=req, prompt=req.prompt, llm_profile="chat")
    assert body.llm_profile == "chat"
    assert body.mode == "trace_autopsy"


@pytest.mark.asyncio
async def test_fetch_routes_normalizes_catalog() -> None:
    from scripts import llm_gateway_client as client

    payload = {
        "default_route": "chat",
        "routes": [
            {
                "id": "chat",
                "served_by": "atlas-worker-1",
                "backend": "llamacpp",
                "status": "up",
                "latency_ms": 12,
                "last_checked_at": "2026-06-14T00:00:00+00:00",
            },
            {
                "id": "agent",
                "served_by": "atlas-worker-agent-1",
                "backend": "llamacpp",
                "status": "down",
                "latency_ms": None,
                "last_checked_at": "2026-06-14T00:00:00+00:00",
            },
        ],
    }

    class _Resp:
        status = 200

        async def json(self):
            return payload

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    class _Session:
        def get(self, url):
            assert url.endswith("/routes")
            return _Resp()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    with patch.object(client.aiohttp, "ClientSession", return_value=_Session()):
        result = await client.fetch_routes()
    assert result["default_route"] == "chat"
    by_id = {r["id"]: r for r in result["routes"]}
    assert by_id["chat"]["status"] == "up"
    assert by_id["agent"]["status"] == "down"
    assert "quick" in by_id and "metacog" in by_id


@pytest.mark.asyncio
async def test_run_hub_agent_via_context_exec_uses_quick_profile() -> None:
    from scripts.context_exec_agent_bridge import run_hub_agent_via_context_exec

    fake_run = ContextExecRunV1(
        run_id="run1",
        status="ok",
        mode="belief_provenance",
        text="Where did the Denver belief come from?",
        final_text="Insufficient evidence for Denver belief.",
        runtime_debug={
            "llm_profile_selected": "quick",
            "route_used": "quick",
            "model_synthesis_used": False,
        },
        operator_summary=ContextExecOperatorSummaryV1(
            title="Belief provenance complete",
            summary="Insufficient evidence for Denver belief.",
            agent_mode="belief_provenance",
            route_used="quick",
            model_synthesis_used=False,
            safety=ContextExecSafetySummaryV1(),
        ),
    )

    with patch(
        "scripts.context_exec_agent_bridge.run_context_exec",
        new=AsyncMock(return_value=fake_run),
    ):
        out = await run_hub_agent_via_context_exec(
            req=CortexChatRequest(
                prompt="Where did the Denver belief come from?",
                mode="agent",
                options={"llm_route": "quick"},
            ),
            prompt="Where did the Denver belief come from?",
            correlation_id="corr-1",
            route_debug={"llm_route": "quick"},
        )
    assert "Agent run complete" in out.get("llm_response", "")
    assert "Route: quick" in out.get("llm_response", "")
    assert out.get("operator_summary", {}).get("route_used") == "quick"
    assert out.get("routing_debug", {}).get("context_exec_lane") is True


@pytest.mark.asyncio
async def test_run_hub_agent_via_context_exec_uses_agent_profile() -> None:
    from scripts.context_exec_agent_bridge import run_hub_agent_via_context_exec

    fake_run = ContextExecRunV1(
        run_id="run-agent",
        status="ok",
        mode="general_investigation",
        text="inspect",
        final_text="done",
        runtime_debug={
            "llm_profile_selected": "agent",
            "route_used": "agent",
            "model_synthesis_used": True,
        },
        operator_summary=ContextExecOperatorSummaryV1(
            title="Investigation complete",
            summary="done",
            agent_mode="general_investigation",
            route_used="agent",
            model_synthesis_used=True,
            safety=ContextExecSafetySummaryV1(),
        ),
    )

    with patch(
        "scripts.context_exec_agent_bridge.run_context_exec",
        new=AsyncMock(return_value=fake_run),
    ) as mock_run:
        await run_hub_agent_via_context_exec(
            req=CortexChatRequest(
                prompt="inspect repo",
                mode="agent",
                options={"llm_route": "agent"},
            ),
            prompt="inspect repo",
            correlation_id="corr-agent",
            route_debug={"llm_route": "agent"},
        )
    body = mock_run.await_args.args[0]
    assert body.llm_profile == "agent"


def test_format_agent_operator_inline_renders_proposal_link() -> None:
    run = ContextExecRunV1(
        run_id="run2",
        status="ok",
        mode="memory_correction_proposal",
        text="correct denver",
        final_text="proposal",
        runtime_debug={"route_used": "agent", "model_synthesis_used": True},
        operator_summary=ContextExecOperatorSummaryV1(
            title="Memory correction proposal drafted",
            summary="Denver claim should be marked uncertain.",
            agent_mode="memory_correction_proposal",
            route_used="agent",
            model_synthesis_used=True,
            proposal_id="prop_denver_1",
            proposal_status="pending_review",
            safety=ContextExecSafetySummaryV1(),
        ),
    )
    inline = format_agent_operator_inline(run)
    assert "Route: agent" in inline
    assert "Synthesis: used" in inline
    assert "prop_denver_1" in inline
    assert "Pending Decisions" in inline
    assert "Mutation: none" in inline


def test_thought_process_syncs_lane_from_mode_dropdown() -> None:
    src = THOUGHT_JS.read_text(encoding="utf-8")
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert "hubModeSelect" in src
    assert "setLane" in src
    assert "LANE_GROUNDED_SMALL" in src
    assert '<option value="orion" selected>Orion</option>' in html


def test_mode_dropdown_only_has_surviving_options() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    mode_select = html.split('id="hubModeSelect"', 1)[1].split("</select>", 1)[0]
    assert 'value="orion"' in mode_select
    assert 'value="quick"' in mode_select
    assert 'value="story"' in mode_select
    assert 'value="agent"' in mode_select
    for killed_value in ("auto", "grounded_small", "brain", "council", "agent_claude_opus", "agent_claude_sonnet", "agent_claude_haiku"):
        assert f'value="{killed_value}"' not in mode_select


def test_recall_profile_dropdown_drops_auto_and_biographical() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    recall_select = html.split('id="recallProfileSelect"', 1)[1].split("</select>", 1)[0]
    assert 'value="assist.light.v1" selected' in recall_select
    assert 'value="biographical.v1"' not in recall_select
    assert 'value="auto"' not in recall_select


def test_recall_profile_auto_follows_mode() -> None:
    app_js = APP_JS.read_text(encoding="utf-8")
    assert "function setRecallProfileAutoState(useAutoDefault)" in app_js
    assert "setRecallProfileAutoState(key === 'orion');" in app_js
    assert "recallProfileSelect.value = 'auto';" in app_js
    assert "recallProfileSelect.disabled = true;" in app_js
    assert "recallProfileSelect.disabled = false;" in app_js
    assert "recallProfileSelect.value = 'assist.light.v1';" in app_js


def test_debug_panel_is_single_card_wrapping_all_rows_in_one_modal() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    app_js = APP_JS.read_text(encoding="utf-8")

    # Main-page surface: one card, one Modal button, no inline row content.
    panel_card = html.split('id="runtimeDebugPanel"', 1)[1].split("</div>\n        </div>", 1)[0]
    assert 'id="debugPanelOpenModal"' in panel_card
    assert 'id="memoryPanelToggle"' not in panel_card

    # All 9 rows live inside the new modal's body, not on the main page directly.
    modal_shell = html.split('id="debugPanelModalRoot"', 1)[1].split('id="autonomyDebugModalRoot"', 1)[0]
    assert 'w-[75vw] h-[75vh]' in modal_shell
    assert 'id="runtimeDebugPanelBody"' in modal_shell
    for row_id in (
        "memoryPanelToggle",
        "agentTraceDebugToggle",
        "autonomyDebugToggle",
        "chatStanceDebugToggle",
        "substrateReviewDebugToggle",
        "selfExperimentsDebugToggle",
        "autonomyReadinessToggle",
        "recallCanaryToggle",
        "cognitiveReviewOpenModal",
    ):
        assert f'id="{row_id}"' in modal_shell

    # Each row still keeps its own original inline expand + its own per-item Modal button.
    assert 'id="memoryDebugOpenModal"' in modal_shell
    assert 'id="autonomyDebugOpenModal"' in modal_shell
    assert "function toggleMemoryPanel()" in app_js
    assert "memoryPanelBody.classList.toggle('hidden', nextHidden);" in app_js
    assert "function openDebugPanelModal()" in app_js
    assert "function closeDebugPanelModal()" in app_js
    assert "debugPanelOpenModal.addEventListener('click'" in app_js


def test_debug_panel_modal_z_index_is_below_all_per_item_modals() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    app_js = APP_JS.read_text(encoding="utf-8")

    # The outer wrapping modal must sit strictly below the shared per-item
    # modal tier (2147483646/647) so a nested modal (e.g. Memory, Agent
    # Trace) opened from within it always paints on top, regardless of
    # DOM-append order or click sequence.
    assert 'id="debugPanelModalRoot"' in html
    debug_root = html.split('id="debugPanelModalRoot"', 1)[1].split(">", 1)[0]
    assert "z-index: 2147483640" in debug_root
    debug_dialog = html.split('id="debugPanelModalDialog"', 1)[1].split(">", 1)[0]
    assert "z-index: 2147483641" in debug_dialog
    assert "debugPanelModalRoot.style.zIndex = '2147483640';" in app_js
    assert "debugPanelModalDialog.style.zIndex = '2147483641';" in app_js

    # Every per-item modal (including Agent Trace, which previously used a
    # much lower z-50 and could never out-stack the wrapping modal) sits at
    # or above the shared 2147483646/647 tier, strictly above the wrapper.
    for modal_id in (
        "memoryDebugModalDialog",
        "autonomyDebugModalDialog",
        "chatStanceDebugModalDialog",
        "substrateReviewModalDialog",
        "selfExperimentsModalDialog",
        "autonomyReadinessModalDialog",
        "recallCanaryModalDialog",
        "cognitiveReviewModalDialog",
    ):
        section = html.split(f'id="{modal_id}"', 1)[1].split(">", 1)[0]
        assert "z-index: 2147483647" in section, f"{modal_id} not above debug panel wrapper"
    agent_trace_section = html.split('id="agentTraceModal"', 1)[1].split(">", 1)[0]
    assert "z-50" not in agent_trace_section
    assert "z-[2147483646]" in agent_trace_section


def test_debug_panel_no_longer_shows_cognition_packs_or_verb_routing() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    modal_shell = html.split('id="debugPanelModalRoot"', 1)[1].split('id="autonomyDebugModalRoot"', 1)[0]
    assert "Cognition Packs" not in modal_shell
    assert "Verb Routing" not in modal_shell
    assert 'id="packContainer"' not in modal_shell
    assert 'id="verbSelectTrigger"' not in modal_shell
    assert 'id="verbDropdown"' not in modal_shell
    # Clear/Copy conversation controls (not verb/pack routing) are kept.
    assert 'id="clearButton"' in modal_shell
    assert 'id="copyButton"' in modal_shell
