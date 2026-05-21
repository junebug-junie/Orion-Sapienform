from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
TEMPLATE_PATH = REPO_ROOT / "services" / "orion-hub" / "templates" / "index.html"
APP_JS_PATH = REPO_ROOT / "services" / "orion-hub" / "static" / "js" / "app.js"
THOUGHT_PROCESS_JS_PATH = REPO_ROOT / "services" / "orion-hub" / "static" / "js" / "thought-process.js"
ORCH_MIND_RUNTIME_PATH = REPO_ROOT / "services" / "orion-cortex-orch" / "app" / "mind_runtime.py"
EXEC_SHARED_SPINE_PATH = REPO_ROOT / "services" / "orion-cortex-exec" / "app" / "chat_stance_shared_spine.py"


def test_template_includes_chat_stance_panel_and_outer_modal_button() -> None:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    assert 'id="chatStanceDebugPanel"' in template
    assert 'id="chatStanceDebugToggle"' in template
    assert 'id="chatStanceDebugOpenModal"' in template
    assert template.index('id="chatStanceDebugOpenModal"') < template.index('id="chatStanceDebugBody"')
    assert "Chat Stance" in template


def test_template_includes_chat_stance_modal_shell() -> None:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    assert 'id="chatStanceDebugModalRoot" class="hidden fixed inset-0 z-[120]' in template
    assert 'id="chatStanceDebugModalBackdrop" class="fixed inset-0 z-[120]' in template
    assert 'id="chatStanceDebugModalDialog" class="fixed inset-x-4 top-8 bottom-8 z-[121]' in template
    assert 'id="chatStanceDebugModalBody"' in template


def test_app_js_chat_stance_empty_state_and_grouped_modal_sections() -> None:
    app_js = APP_JS_PATH.read_text(encoding="utf-8")
    assert "function clearChatStanceDebugPanel()" in app_js
    assert "No chat stance debug payload on this turn." in app_js
    assert "function updateChatStanceDebugPanel(payload)" in app_js
    assert "buildChatStanceSection('Overview'" in app_js
    assert "buildChatStanceSection('Source Inputs by Category'" in app_js
    assert "buildChatStanceSection('Synthesized Brief'" in app_js
    assert "buildChatStanceSection('Enforcement / Fallback'" in app_js
    assert "buildChatStanceSection('Final Prompt Contract'" in app_js
    assert "buildChatStanceSection('Raw compact JSON'" in app_js


def test_app_js_wires_chat_stance_modal_and_payload_plumbing() -> None:
    app_js = APP_JS_PATH.read_text(encoding="utf-8")
    assert "chatStanceDebug: d.chat_stance_debug," in app_js
    assert "updateChatStanceDebugPanel(meta.chatStanceDebug || meta.chat_stance_debug);" in app_js
    assert "chatStanceDebugToggle.addEventListener('click', toggleChatStanceDebugPanel);" in app_js
    assert "chatStanceDebugOpenModal.addEventListener('click', (event) => {" in app_js
    assert "openChatStanceDebugModal();" in app_js


def test_thought_process_js_registers_execution_steps_panel() -> None:
    thought_js = THOUGHT_PROCESS_JS_PATH.read_text(encoding="utf-8")
    assert "function fetchCognitionTrace" in thought_js
    assert "function buildExecutionStepsPanel" in thought_js
    assert "function resolveCorrelationId" in thought_js
    assert "function mountExecutionStepsPanel" in thought_js
    assert "Execution Steps" in thought_js
    assert "#signals" in thought_js
    assert "open-organ-signals" in thought_js
    assert "correlation_id:" in thought_js
    assert "root_correlation_id" in thought_js


def test_app_js_wires_organ_signals_correlation_navigation() -> None:
    app_js = APP_JS_PATH.read_text(encoding="utf-8")
    assert "function openOrganSignalsForCorrelation" in app_js
    assert "window.OrionHubOpenOrganSignals = openOrganSignalsForCorrelation" in app_js
    assert 'url.hash = \'#signals\'' in app_js


def test_app_js_wires_execution_steps_panel_hook() -> None:
    app_js = APP_JS_PATH.read_text(encoding="utf-8")
    assert "function appendExecutionStepsPanel" in app_js
    assert "mountExecutionStepsPanel" in app_js
    assert "appendExecutionStepsPanel(div, meta)" in app_js
    assert "appendExecutionStepsPanel(root, meta)" in app_js


def test_thought_process_js_registers_cognitive_projection_inspect_renderer() -> None:
    thought_js = THOUGHT_PROCESS_JS_PATH.read_text(encoding="utf-8")
    assert "global.OrionChatStanceProjectionInspect = api;" in thought_js
    assert "function renderProjectionCard" in thought_js
    assert "function normalizeProjectionBundle" in thought_js
    assert "function renderFromPayload" in thought_js
    assert "function attach" in thought_js
    assert "MutationObserver" in thought_js
    assert "chatStanceDebugRaw" in thought_js
    assert "chatStanceDebugOpenModal" in thought_js


def test_thought_process_js_renders_cognitive_projection_cards_and_fields() -> None:
    thought_js = THOUGHT_PROCESS_JS_PATH.read_text(encoding="utf-8")
    assert "chatStanceCognitiveProjectionInspectCard" in thought_js
    assert "chatStanceCognitiveProjectionInspectModalCard" in thought_js
    assert "cognitive_projection: bundle" in thought_js
    assert "root.raw" in thought_js
    assert "root.raw)?.cognitive_projection" in thought_js
    assert "Cognitive Projection" in thought_js
    assert "Top projection items" in thought_js
    assert "Raw cognitive projection" in thought_js
    assert "shared spine used" in thought_js
    assert "Projection id" in thought_js
    assert "Degraded producers" in thought_js
    assert "Cold anchors" in thought_js
    assert "Lineage" in thought_js


def test_thought_process_js_renders_read_only_cognitive_comparison_surface() -> None:
    thought_js = THOUGHT_PROCESS_JS_PATH.read_text(encoding="utf-8")
    assert "chatStanceCognitiveComparisonInspectCard" in thought_js
    assert "chatStanceCognitiveComparisonInspectModalCard" in thought_js
    assert "Cognitive Comparison" in thought_js
    assert "read-only · no promotion" in thought_js
    assert "Shared projection" in thought_js
    assert "Legacy ChatStanceBrief" in thought_js
    assert "Mind handoff / quality" in thought_js
    assert "Mind shadow synthesis" in thought_js
    assert "Raw comparison sources" in thought_js
    assert "xl:grid-cols-4" in thought_js


def test_thought_process_js_comparison_extracts_legacy_and_mind_fields() -> None:
    thought_js = THOUGHT_PROCESS_JS_PATH.read_text(encoding="utf-8")
    assert "function normalizeLegacyChatStance" in thought_js
    assert "function normalizeMindHandoff" in thought_js
    assert "synthesized_brief" in thought_js
    assert "final_prompt_contract" in thought_js
    assert "mind_handoff" in thought_js
    assert "mind_quality" in thought_js
    assert "mind_run_ok" in thought_js
    assert "mind_contract_only" in thought_js
    assert "visiblyDegraded" in thought_js
    assert "projection_starved" in thought_js
    assert "Mind starved before Exec rebuild" in thought_js
    assert "execRichMindStarved" in thought_js
    assert "mind_skip_stance_synthesis" in thought_js


def test_thought_process_js_comparison_extracts_mind_shadow_synthesis_fields() -> None:
    thought_js = THOUGHT_PROCESS_JS_PATH.read_text(encoding="utf-8")
    assert "function normalizeMindShadowSynthesis" in thought_js
    assert "mind_shadow_synthesis" in thought_js
    assert "mind_shadow_synthesis_present" in thought_js
    assert "mind_authorized_for_stance_skip" in thought_js
    assert "authorized_for_stance_skip" in thought_js
    assert "attention_focus" in thought_js
    assert "curiosity_candidate" in thought_js
    assert "relationship_frame" in thought_js
    assert "projection_refs_used" in thought_js
    assert "hazards" in thought_js
    assert "stance_candidate" in thought_js


def test_thought_process_js_renders_read_only_mind_shadow_evaluation_surface() -> None:
    thought_js = THOUGHT_PROCESS_JS_PATH.read_text(encoding="utf-8")
    assert "global.OrionMindShadowEvaluation = api;" in thought_js
    assert "function evaluateMindShadow" in thought_js
    assert "function renderEvaluationCard" in thought_js
    assert "chatStanceMindShadowEvaluationCard" in thought_js
    assert "chatStanceMindShadowEvaluationModalCard" in thought_js
    assert "Mind shadow evaluation" in thought_js
    assert "operator comparison · read-only" in thought_js
    assert "read-only / no promotion" in thought_js
    assert "Legacy category hits" in thought_js
    assert "Legacy category gaps" in thought_js
    assert "Raw Mind shadow evaluation" in thought_js
    assert "Unexpected authority flag observed; shadow remains display-only in Hub." in thought_js


def test_thought_process_js_registers_grounded_small_lane_contract() -> None:
    thought_js = THOUGHT_PROCESS_JS_PATH.read_text(encoding="utf-8")
    assert "global.OrionHubGroundedSmallLane" in thought_js
    assert "LANE_GROUNDED_SMALL = 'grounded_small'" in thought_js
    assert "Grounded Small" in thought_js
    assert "brainDeepModeBtn" in thought_js
    assert "options.llm_route = 'quick';" in thought_js
    assert "payload.verbs = [];" in thought_js
    assert "delete options.chat_quick_full_stance;" in thought_js
    assert "hub_chat_lane" in thought_js
    assert "payload.context.metadata.mind_enabled = true" in thought_js
    assert "patchWebSocketSend" in thought_js
    assert "patchFetch" in thought_js


def test_mind_projection_handoff_reuses_one_populated_projection() -> None:
    mind_runtime = ORCH_MIND_RUNTIME_PATH.read_text(encoding="utf-8")
    shared_spine = EXEC_SHARED_SPINE_PATH.read_text(encoding="utf-8")

    assert "_build_cold_cognitive_projection_facet" in mind_runtime
    assert "InMemorySubstrateGraphStore" in mind_runtime
    assert "build_cognitive_projection_for_mind_with_diagnostics" in mind_runtime
    assert "metadata[\"cognitive_projection_facet\"] = projection" in mind_runtime
    assert "metadata[\"cognitive_projection\"] = projection" in mind_runtime
    assert "_share_cognitive_projection_with_plan(plan_request, cognitive_projection" in mind_runtime
    assert "resolve_cognitive_projection_for_mind" in mind_runtime

    assert "_inline_projection_from_metadata" in shared_spine
    assert "projection_reused_from_metadata" in shared_spine
    assert "orion_cortex_orch_mind_runtime" in shared_spine
    assert "chat_stance_shared_projection_spine_reused" in shared_spine


def test_thought_process_js_comparison_does_not_promote_mind_or_change_routing() -> None:
    thought_js = THOUGHT_PROCESS_JS_PATH.read_text(encoding="utf-8")
    forbidden_mutations = [
        "mind_skip_stance_synthesis = true",
        "mind_skip_stance_synthesis=true",
        "mind_authorized_for_stance_skip = true",
        "mind_authorized_for_stance_skip=true",
        "authorized_for_stance_skip = true",
        "authorized_for_stance_skip=true",
        "meaningful_synthesis' &&",
        'meaningful_synthesis" &&',
        "fetch(`${API_BASE_URL}/api/chat`",
        "fetch('/api/chat'",
        "selectedVerbs.push",
        "modeVerbOverride",
    ]
    for forbidden in forbidden_mutations:
        assert forbidden not in thought_js
