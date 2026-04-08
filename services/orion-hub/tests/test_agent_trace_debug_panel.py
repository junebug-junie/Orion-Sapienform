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
    assert 'id="agentTraceDebugPanel" class="hidden' not in template
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


def test_memory_debug_uses_modal_driven_detail_view_with_autonomy_modal_pattern() -> None:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    app_js = APP_JS_PATH.read_text(encoding="utf-8")

    assert 'id="memoryDebugOpenModal"' in template
    assert 'id="memoryDebugModalRoot" class="hidden fixed inset-0 z-[120]' in template
    assert 'id="memoryDebugModalBackdrop" class="fixed inset-0 z-[120]' in template
    assert 'id="memoryDebugModalDialog" class="fixed inset-x-4 top-8 bottom-8 z-[121]' in template
    assert "function openMemoryDebugModal()" in app_js
    assert "function closeMemoryDebugModal()" in app_js
    assert "function ensureMemoryDebugModalRootOnBody()" in app_js
    assert "memoryDebugOpenModal.addEventListener('click', openMemoryDebugModal);" in app_js


def test_memory_debug_modal_render_path_keeps_full_payload_and_expandable_entries() -> None:
    app_js = APP_JS_PATH.read_text(encoding="utf-8")

    assert "function normalizeMemoryDebugModel(data)" in app_js
    assert "function collectRecallEntries(recallDebug)" in app_js
    assert "function buildMemoryDebugRecallEntryNode(entry, index)" in app_js
    assert "function renderMemoryDebugModal(model)" in app_js
    assert "max-h-72 overflow-y-auto" in app_js
    assert "Recall entries (" in app_js
    assert "memoryDigestPre.textContent = summarizeInlineText(model.memoryDigest);" in app_js
    assert "['Memory digest', safeModel.memoryDigest]," in app_js
    assert "const stableLabel = `Entry ${index + 1}`;" in app_js


def test_memory_debug_long_text_layout_uses_safe_wrap_rules() -> None:
    app_js = APP_JS_PATH.read_text(encoding="utf-8")

    assert "function applyDebugTextLayout(node)" in app_js
    assert "node.style.overflowWrap = 'anywhere';" in app_js
    assert "node.style.wordBreak = 'break-word';" in app_js
    assert "node.style.whiteSpace = 'pre-wrap';" in app_js
    assert "applyDebugTextLayout(pre);" in app_js
    assert "applyDebugTextLayout(rawPre);" in app_js


def test_memory_and_autonomy_modals_coordinate_scroll_lock_and_visibility() -> None:
    app_js = APP_JS_PATH.read_text(encoding="utf-8")

    assert "function syncDebugModalScrollLock()" in app_js
    assert "closeAutonomyDebugModal();" in app_js
    assert "closeMemoryDebugModal();" in app_js
    assert "const shouldLock = isModalVisible(memoryDebugModalRoot) || isModalVisible(autonomyDebugModalRoot);" in app_js
    assert "document.body.classList.toggle('overflow-hidden', shouldLock);" in app_js


def test_agent_trace_shell_stays_visible_and_uses_empty_state_when_payload_absent() -> None:
    app_js = APP_JS_PATH.read_text(encoding="utf-8")

    assert "function clearAgentTraceDebugPanel()" in app_js
    assert "if (agentTraceDebugBody) agentTraceDebugBody.classList.add('hidden');" in app_js
    assert "if (agentTraceDebugMeta) agentTraceDebugMeta.textContent = 'No agent trace on this turn.'" in app_js
    assert "if (agentTraceDebugSummary) agentTraceDebugSummary.textContent = 'No agent trace on this turn.'" in app_js
    assert "if (agentTraceDebugRaw) agentTraceDebugRaw.textContent = 'No agent trace on this turn.'" in app_js
    assert "if (agentTraceDebugPanel) agentTraceDebugPanel.classList.add('hidden');" not in app_js


def test_agent_trace_populated_render_path_clears_stale_sections_before_append() -> None:
    app_js = APP_JS_PATH.read_text(encoding="utf-8")

    assert "agentTraceDebugOverview.innerHTML = '';" in app_js
    assert "agentTraceDebugToolGroups.innerHTML = '';" in app_js
    assert "agentTraceDebugTimeline.innerHTML = '';" in app_js
    assert "agentTraceDebugRaw.innerHTML = '';" in app_js
