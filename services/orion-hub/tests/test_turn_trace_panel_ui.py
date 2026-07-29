"""Behavioral tests for the Turn Trace panel (thought-process.js), the UI
surface for idea 1's fused GET /api/chat/turn/{correlation_id}/trace lookup.

Runs the real JS via node (same pattern as test_memory_graph_suggest_coalesce_ui.py)
rather than only asserting the source text contains expected strings -- this
actually calls buildTurnTracePanel() with fixture payloads and inspects the
rendered HTML.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
THOUGHT_PROCESS_JS_PATH = REPO_ROOT / "services" / "orion-hub" / "static" / "js" / "thought-process.js"


def _node_build_turn_trace_panel(payload: Dict[str, Any]) -> str:
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available for turn trace panel behavioral tests")
    script = f"""
const fs = require('fs');
global.window = {{}};
eval(fs.readFileSync({json.dumps(str(THOUGHT_PROCESS_JS_PATH))}, 'utf8'));
const html = window.OrionThoughtProcess.buildTurnTracePanel({json.dumps(payload)});
console.log(html);
"""
    proc = subprocess.run([node, "-e", script], capture_output=True, text=True, check=False, timeout=30)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout or "node buildTurnTracePanel failed")
    return proc.stdout


def test_loading_state_renders_loading_copy() -> None:
    html = _node_build_turn_trace_panel({"correlationId": "corr-1", "loading": True})
    assert "Loading turn trace" in html
    assert "corr-1" in html


def test_error_state_renders_error_copy() -> None:
    html = _node_build_turn_trace_panel({"correlationId": "corr-1", "error": "http_503"})
    assert "Turn trace unavailable" in html
    assert "http_503" in html


def test_renders_stance_decision_disposition_and_reasons() -> None:
    trace = {
        "correlation_id": "corr-1",
        "route_signal_inferred": "unified_turn_harness",
        "sources": {
            "thought_decision": {
                "disposition": "defer",
                "disposition_reasons": ["needs more context", "ambiguous request"],
                "boundary_register": True,
                "imperative": "Ask a clarifying question before proceeding.",
                "tone": "curious",
                "strain_refs": ["strain:continuity"],
                "stance_harness_slice": {
                    "task_mode": "chat",
                    "conversation_frame": "direct",
                    "answer_strategy": "concise",
                },
            }
        },
        "gaps": ["no_cognition_trace"],
    }
    html = _node_build_turn_trace_panel({"correlationId": "corr-1", "trace": trace})
    assert "Ingress / Association / Stance" in html
    assert "defer" in html
    assert "needs more context" in html
    assert "ambiguous request" in html
    assert "boundary_register" in html
    assert "unified_turn_harness" in html
    assert "no_cognition_trace" in html
    assert "Ask a clarifying question before proceeding." in html
    assert "curious" in html
    assert "strain:continuity" in html
    assert "concise" in html


def test_renders_execution_run_step_counts() -> None:
    trace = {
        "correlation_id": "corr-2",
        "route_signal_inferred": "unified_turn_harness",
        "sources": {
            "execution_run": {"started_step_count": 5, "failed_step_count": 1},
        },
        "gaps": [],
    }
    html = _node_build_turn_trace_panel({"correlationId": "corr-2", "trace": trace})
    assert "Harness Work" in html
    assert "5" in html
    assert "1" in html


def test_renders_draft_molecule_and_substrate_appraisal() -> None:
    trace = {
        "correlation_id": "corr-5",
        "route_signal_inferred": "unified_turn_harness",
        "sources": {
            "harness_turn_trace": {
                "run_artifact": {
                    "draft_text": "Here is my draft answer.",
                    "substrate_appraisal": {
                        "surprise_level": 0.15,
                        "open_loop_pressure": 0.05,
                        "alignment_hints": ["hint:consistent_with_prior"],
                        "strain_shift_refs": [],
                        "learning_refs": ["learning:1"],
                    },
                }
            }
        },
        "gaps": [],
    }
    html = _node_build_turn_trace_panel({"correlationId": "corr-5", "trace": trace})
    assert "Draft Molecule + Substrate Appraisal" in html
    assert "Here is my draft answer." in html
    assert "0.15" in html
    assert "hint:consistent_with_prior" in html
    assert "learning:1" in html


def test_renders_reflect_verdict_and_voice_and_outcome_and_closure() -> None:
    trace = {
        "correlation_id": "corr-6",
        "route_signal_inferred": "unified_turn_harness",
        "sources": {
            "harness_turn_trace": {
                "run_artifact": {
                    "reflection": {
                        "alignment_verdict": "misaligned",
                        "alignment_notes": ["tone drifted from draft"],
                        "strain_unresolved": True,
                        "imperative": "Soften the tone before sending.",
                        "tone": "warm",
                        "quick_lane_skipped_llm": False,
                    },
                    "final_text": "Here is the final reply.",
                    "finalize_ran": True,
                    "finalize_changed": True,
                    "compliance_verdict": "completed",
                    "grounding_status": "grounded",
                },
                "outcome_molecule": {
                    "alignment_verdict": "misaligned",
                    "surprise_resolved": False,
                    "surprise_level_at_draft": 0.15,
                    "final_hash": "abcdef0123456789",
                    "finalize_failed": False,
                },
                "closure": {
                    "surprise_unresolved": True,
                    "user_message_excerpt": "what do you think about this plan?",
                },
            }
        },
        "gaps": [],
    }
    html = _node_build_turn_trace_panel({"correlationId": "corr-6", "trace": trace})
    assert "Integrative Reflect + Verdict" in html
    assert "misaligned" in html
    assert "tone drifted from draft" in html
    assert "Soften the tone before sending." in html
    assert "Orion Voice" in html
    assert "Here is the final reply." in html
    assert "finalize_ran" in html
    assert "finalize_changed" in html
    assert "Outcome Molecule" in html
    assert "surprise_unresolved" in html
    assert "Post-Turn Closure" in html
    assert "what do you think about this plan?" in html


def test_renders_no_trace_found_when_all_sources_empty() -> None:
    trace = {
        "correlation_id": "corr-3",
        "route_signal_inferred": "none",
        "sources": {},
        "gaps": ["no_cognition_trace", "no_grammar_trace", "no_execution_run_pressure_signal", "no_thought_decision"],
    }
    html = _node_build_turn_trace_panel({"correlationId": "corr-3", "trace": trace})
    assert "No trace found in any source for this turn." in html


def test_notes_cognition_trace_present_without_duplicating_step_rendering() -> None:
    trace = {
        "correlation_id": "corr-4",
        "route_signal_inferred": "classic_planrunner",
        "sources": {"cognition_trace": {"verb": "chat_general"}},
        "gaps": [],
    }
    html = _node_build_turn_trace_panel({"correlationId": "corr-4", "trace": trace})
    assert "cognition_trace present" in html
    assert "Execution Steps panel above" in html


def test_source_registers_new_api_exports() -> None:
    src = THOUGHT_PROCESS_JS_PATH.read_text(encoding="utf-8")
    assert "function fetchTurnTrace" in src
    assert "function buildTurnTracePanel" in src
    assert "function mountTurnTracePanel" in src
    assert "fetchTurnTrace," in src
    assert "buildTurnTracePanel," in src
    assert "mountTurnTracePanel," in src
    assert "/trace`" in src


def test_app_js_wires_turn_trace_panel_alongside_execution_steps() -> None:
    app_js = (REPO_ROOT / "services" / "orion-hub" / "static" / "js" / "app.js").read_text(encoding="utf-8")
    assert "function appendTurnTracePanel(parent, meta = {})" in app_js
    assert "mountTurnTracePanel" in app_js
    assert "appendExecutionStepsPanel(root, meta);\n    appendTurnTracePanel(root, meta);" in app_js
    assert "appendExecutionStepsPanel(div, meta);\n      appendTurnTracePanel(div, meta);" in app_js
