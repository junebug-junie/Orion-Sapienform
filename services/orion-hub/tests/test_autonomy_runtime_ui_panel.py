from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
TEMPLATE_PATH = REPO_ROOT / "services" / "orion-hub" / "templates" / "index.html"
APP_JS_PATH = REPO_ROOT / "services" / "orion-hub" / "static" / "js" / "app.js"


def test_template_places_autonomy_runtime_panel_between_agent_trace_and_social_inspection() -> None:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    memory_idx = template.index('id="memoryPanelToggle"')
    agent_idx = template.index('id="agentTraceDebugPanel"')
    autonomy_idx = template.index('id="autonomyDebugPanel"')
    social_idx = template.index("Social Inspection")

    assert memory_idx < agent_idx < autonomy_idx < social_idx
    assert 'id="autonomyDebugToggle"' in template
    assert 'id="autonomyDebugPanel" class="hidden' not in template
    assert 'id="autonomyDebugOverview"' in template
    assert 'id="autonomyDebugState"' in template
    assert 'id="autonomyDebugProposals"' in template
    assert 'id="autonomyDebugAlignment"' in template
    assert 'id="autonomyDebugRaw"' in template
    assert "Raw compact debug" in template
    assert 'id="hubUiBuildLabel"' in template
    assert 'data-ui-version="{{HUB_UI_ASSET_VERSION}}"' in template


def test_template_includes_cache_busted_app_js_asset_url() -> None:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    assert '<script src="/static/js/app.js?v={{HUB_UI_ASSET_VERSION}}" defer></script>' in template


def test_app_js_wires_autonomy_debug_panel_clear_update_and_toggle() -> None:
    app_js = APP_JS_PATH.read_text(encoding="utf-8")

    assert "function clearAutonomyDebugPanel()" in app_js
    assert "function updateAutonomyDebugPanel(summary, debug, meta = {})" in app_js
    assert "function toggleAutonomyDebugPanel()" in app_js
    assert "function openAutonomyDebugModal()" in app_js
    assert "function closeAutonomyDebugModal()" in app_js
    assert "function ensureAutonomyModalRootOnBody()" in app_js
    assert "autonomyDebugToggle.addEventListener('click', toggleAutonomyDebugPanel);" in app_js
    assert "autonomyDebugOpenModal.addEventListener('click', openAutonomyDebugModal);" in app_js
    assert "clearAutonomyDebugPanel();" in app_js
    assert "if (autonomyDebugAlignment) autonomyDebugAlignment.textContent = 'No meaningful autonomy signal for this turn.';" in app_js
    assert "const hubUiVersion = (window.__HUB_UI_VERSION__ || document.body?.dataset?.uiVersion || \"unknown\").trim() || \"unknown\";" in app_js
    assert "console.log(`[HubUI] version=${hubUiVersion}`);" in app_js


def test_app_js_passes_autonomy_payload_through_orion_message_meta_ws_and_http() -> None:
    app_js = APP_JS_PATH.read_text(encoding="utf-8")

    assert "agentTrace: d.agent_trace," in app_js
    assert "workflow: d.workflow," in app_js
    assert "routingDebug: d.routing_debug," in app_js
    assert "autonomySummary: d.autonomy_summary," in app_js
    assert "autonomyDebug: d.autonomy_debug," in app_js
    assert "autonomyStatePreview: d.autonomy_state_preview," in app_js
    assert "autonomyRepositoryStatus: d.autonomy_repository_status," in app_js


def test_append_message_renders_autonomy_between_body_and_agent_trace_for_orion() -> None:
    app_js = APP_JS_PATH.read_text(encoding="utf-8")

    body_idx = app_js.index("div.appendChild(body);")
    autonomy_idx = app_js.index("const autonomyPanel = createAutonomyPanel(")
    trace_idx = app_js.index("const tracePanel = createAgentTracePanel(meta.agentTrace, meta);")
    metacog_idx = app_js.index("const metacogPanel = createMetacogTracePanel(meta.metacogTraces || meta.metacog_traces || []);")

    assert body_idx < autonomy_idx < trace_idx < metacog_idx


def test_autonomy_inline_card_and_debug_panel_use_semantic_and_runtime_payloads() -> None:
    app_js = APP_JS_PATH.read_text(encoding="utf-8")

    assert "function createAutonomyPanel(summary, debug, meta = {})" in app_js
    assert "state_preview: safePreview || {}" in app_js
    assert "repository_status:" in app_js
    assert "normalizeAutonomyModel(summary, debug, meta)" in app_js
    assert "['alignment', model.alignment.alignment_note]" in app_js


def test_autonomy_debug_panel_clears_when_payload_absent() -> None:
    app_js = APP_JS_PATH.read_text(encoding="utf-8")

    assert "const model = normalizeAutonomyModel(summary, debug, meta);" in app_js
    assert "if (!model) {" in app_js
    assert "clearAutonomyDebugPanel();" in app_js
    assert "if (autonomyDebugMeta) autonomyDebugMeta.textContent = 'No autonomy payload on this turn.';" in app_js
    assert "if (autonomyDebugState) autonomyDebugState.textContent = 'No meaningful autonomy signal for this turn.';" in app_js
    assert "if (autonomyDebugRaw) autonomyDebugRaw.textContent = 'No autonomy payload on this turn.';" in app_js
    assert "if (autonomyDebugPanel) autonomyDebugPanel.classList.add('hidden');" not in app_js


def test_inline_autonomy_rendering_uses_high_signal_gate() -> None:
    app_js = APP_JS_PATH.read_text(encoding="utf-8")

    assert "function shouldRenderAutonomyInline(model)" in app_js
    assert "if (!model || !shouldRenderAutonomyInline(model)) return null;" in app_js
    assert "model.driveCompetition" in app_js
    assert "hasDc" in app_js


def test_normalization_uses_state_preview_semantics_when_summary_sparse() -> None:
    app_js = APP_JS_PATH.read_text(encoding="utf-8")

    assert "const safePreview = meta && meta.autonomyStatePreview" in app_js
    assert "(safeSummary && safeSummary.top_drives) || (safePreview && safePreview.top_drives)" in app_js
    assert "(safeSummary && safeSummary.active_tensions) || (safePreview && safePreview.active_tensions)" in app_js
    assert "(safeSummary && safeSummary.proposal_headlines) || (safePreview && safePreview.proposal_headlines)" in app_js
    assert "(safeSummary && safeSummary.dominant_drive) || (safePreview && safePreview.dominant_drive)" in app_js
    assert "safeSummary.drive_competition" in app_js
    assert "competing pressures:" in app_js


def test_clear_flow_clears_autonomy_debug_state() -> None:
    app_js = APP_JS_PATH.read_text(encoding="utf-8")

    assert "if (clearButton && conversationDiv) {" in app_js
    assert "clearButton.addEventListener('click', () => {" in app_js
    assert "clearAutonomyDebugPanel();" in app_js


def test_right_rail_can_render_from_debug_even_without_semantic_inline_signal() -> None:
    app_js = APP_JS_PATH.read_text(encoding="utf-8")

    assert "const hasDebugSignal = !!(safeDebug && typeof safeDebug === 'object' && Object.keys(safeDebug).length);" in app_js
    assert "const hasAnySignal = !!(hasSemanticSignal || hasDebugSignal" in app_js


def test_autonomy_populated_render_path_replaces_prior_turn_content() -> None:
    app_js = APP_JS_PATH.read_text(encoding="utf-8")

    assert "autonomyDebugOverview.innerHTML = '';" in app_js
    assert "autonomyDebugState.innerHTML = '';" in app_js
    assert "autonomyDebugProposals.innerHTML = '';" in app_js
    assert "autonomyDebugAlignment.innerHTML = '';" in app_js
    assert "autonomyDebugRaw.textContent = JSON.stringify(model.raw, null, 2);" in app_js


def test_alignment_rules_are_deterministic_and_bounded() -> None:
    app_js = APP_JS_PATH.read_text(encoding="utf-8")

    assert "function expectedPostureFromDrive(drive)" in app_js
    assert "function deriveVisibleAutonomyCues(replyText)" in app_js
    assert "function computeAutonomyAlignment(model, replyText)" in app_js
    assert "'reply appears aligned'" in app_js
    assert "'reply posture not clearly visible'" in app_js
    assert "'no strong posture expected'" in app_js


def test_template_includes_fixed_high_z_autonomy_runtime_modal() -> None:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    assert 'id="autonomyDebugModalRoot"' in template
    assert 'id="autonomyDebugModalRoot" class="hidden fixed inset-0 z-[120]' in template
    assert 'id="autonomyDebugModalBackdrop" class="fixed inset-0 z-[120]' in template
    assert 'id="autonomyDebugModalDialog" class="fixed inset-x-4 top-8 bottom-8 z-[121]' in template
    assert 'z-index: 2147483646' in template
    assert 'z-index: 2147483647' in template
    assert 'id="autonomyDebugOpenModal"' in template


def test_app_js_mounts_autonomy_modal_root_to_body_and_locks_scroll() -> None:
    app_js = APP_JS_PATH.read_text(encoding="utf-8")

    assert "if (autonomyDebugModalRoot.parentElement !== document.body) {" in app_js
    assert "document.body.appendChild(autonomyDebugModalRoot);" in app_js
    assert "autonomyDebugModalRoot.style.zIndex = '2147483646';" in app_js
    assert "autonomyDebugModalDialog.style.zIndex = '2147483647';" in app_js
    assert "document.body.classList.add('overflow-hidden');" in app_js
    assert "document.body.classList.remove('overflow-hidden');" in app_js


def test_template_includes_autonomy_readiness_panel_card() -> None:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    assert 'id="autonomyReadinessPanel"' in template
    assert 'id="autonomyReadinessToggle"' in template
    assert 'id="autonomyReadinessBody"' in template
    assert 'id="autonomyReadinessMeta"' in template
    assert 'id="autonomyReadinessOverview"' in template
    assert 'id="autonomyReadinessWarnings"' in template
    assert "Autonomy Readiness" in template


def test_app_js_wires_autonomy_readiness_fetch_toggle_and_defensive_rendering() -> None:
    app_js = APP_JS_PATH.read_text(encoding="utf-8")

    assert "function clearAutonomyReadinessPanel()" in app_js
    assert "function toggleAutonomyReadinessPanel()" in app_js
    assert "function updateAutonomyReadinessPanel(snapshot)" in app_js
    assert "function refreshAutonomyReadinessPanel()" in app_js
    assert "substrateReviewFetch('/api/substrate/autonomy-readiness')" in app_js
    assert "autonomyReadinessToggle.addEventListener('click', toggleAutonomyReadinessPanel);" in app_js
    assert "clearAutonomyReadinessPanel();" in app_js
    assert "refreshAutonomyReadinessPanel().catch((err) => {" in app_js
    assert "const warnings = Array.isArray(snapshot && snapshot.warnings) ? snapshot.warnings : [];" in app_js


def test_template_and_js_include_recall_canary_controls_without_unsafe_actions() -> None:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    app_js = APP_JS_PATH.read_text(encoding="utf-8")

    assert 'id="recallCanaryPanel"' in template
    assert 'id="recallCanaryQueryInput"' in template
    assert 'id="recallCanaryRunButton"' in template
    assert 'id="recallCanaryRecordJudgmentButton"' in template
    assert 'id="recallCanaryCreateReviewArtifactButton"' in template
    assert 'id="recallCanaryJudgmentV2Better"' in template
    assert 'id="recallCanaryJudgmentV1Better"' in template
    assert 'id="recallCanaryJudgmentTie"' in template
    assert 'id="recallCanaryJudgmentBothBad"' in template
    assert 'id="recallCanaryJudgmentInconclusive"' in template
    assert "Create Review Artifact (Evidence Only)" in template
    assert "Promote to Production" not in template
    assert "Make V2 Default" not in template
    assert "Apply Recall Patch" not in template
    assert "function runRecallCanaryQuery()" in app_js
    assert "function recordRecallCanaryJudgment()" in app_js
    assert "function createRecallCanaryReviewArtifact()" in app_js
