from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
TEMPLATE_PATH = REPO_ROOT / "services" / "orion-hub" / "templates" / "index.html"
APP_JS_PATH = REPO_ROOT / "services" / "orion-hub" / "static" / "js" / "app.js"


def test_hub_template_includes_agent_trace_debug_panel_next_to_memory_panel() -> None:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    memory_idx = template.index('id="memoryPanelToggle"')
    agent_idx = template.index('id="agentTraceDebugToggle"')

    assert memory_idx < agent_idx
    assert 'id="agentTraceDebugPanel"' in template
    assert "Agent Trace" in template
    assert 'id="agentTraceDebugOverview"' in template
    assert 'id="agentTraceDebugToolGroups"' in template
    assert 'id="agentTraceDebugTimeline"' in template
    assert 'id="agentTraceDebugRaw"' in template


def test_hub_app_updates_agent_trace_debug_panel_from_fresh_orion_messages() -> None:
    app_js = APP_JS_PATH.read_text(encoding="utf-8")

    assert "function updateAgentTraceDebugPanel(summary, meta = {})" in app_js
    assert "updateAgentTraceDebugPanel(meta.agentTrace, meta);" in app_js
    assert "agentTraceDebugToggle.addEventListener('click', toggleAgentTraceDebugPanel);" in app_js
    assert "clearAgentTraceDebugPanel();" in app_js
