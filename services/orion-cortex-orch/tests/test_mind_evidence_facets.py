from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_guard = Path(__file__).resolve().parent / "_orch_import_guard.py"
_spec = importlib.util.spec_from_file_location("_orch_guard_boot", _guard)
assert _spec and _spec.loader
_guard_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_guard_mod)

ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


def _orch_prep() -> None:
    _guard_mod.ensure_orion_cortex_orch_app()


def test_build_mind_run_request_attaches_evidence_facets() -> None:
    _orch_prep()
    from app.mind_runtime import build_mind_run_request
    from orion.schemas.cortex.contracts import CortexClientContext, CortexClientRequest, LLMMessage
    from orion.schemas.cortex.schemas import ExecutionPlan, ExecutionStep, PlanExecutionRequest

    pr = PlanExecutionRequest(
        plan=ExecutionPlan(
            verb_name="chat_general",
            steps=[ExecutionStep(verb_name="chat_general", step_name="noop", order=0, services=[])],
        ),
        context={
            "recall_bundle": {"fragments": [{"snippet": "memory", "source": "journal"}]},
            "social_turn_policy": {"mode": "warm"},
            "chat_situation_summary": {"headline": "evening"},
            "orion_identity_summary": ["Orion is a cognitive presence."],
            "metadata": {"mind_enabled": True},
        },
    )
    cr = CortexClientRequest(
        verb="chat_general",
        mode="brain",
        context=CortexClientContext(
            messages=[LLMMessage(role="user", content="hi")],
            user_message="hi",
            metadata={"mind_enabled": True},
        ),
    )
    req = build_mind_run_request(
        cr,
        pr,
        "550e8400-e29b-41d4-a716-446655440000",
        cognitive_projection_facet={
            "schema_version": "cognitive.projection.v1",
            "projection_id": "p1",
            "anchors": {
                "orion": {
                    "items": [{"label": "curiosity", "summary": "notice user tone", "salience": 0.5}]
                }
            },
            "item_count": 1,
        },
    )
    facets = (req.snapshot_inputs or {}).get("facets") or {}
    assert facets.get("recall_bundle", {}).get("fragments")
    assert facets.get("social_compact", {}).get("social_turn_policy")
    assert facets.get("situation_compact", {}).get("chat_situation_summary")
    assert facets.get("identity_background", {}).get("background_identity") is True
    assert facets.get("cognitive_projection")
    # Retired 2026-07-28: the chat_autonomy_state_v2/autonomy_state-sourced
    # autonomy_compact facet was confirmed unreachable repo-wide (no writer
    # anywhere assigns either key) and removed outright, not merely excluded.
    assert "autonomy_compact" not in facets


# --- drive_state_compact facet: REMOVED 2026-07-30 -----------------------------
# This facet (a bounded, fail-open Postgres read of the latest drive_audits row)
# was itself the 2026-07-16 replacement for the dead chat_autonomy_state_v2/
# autonomy_state "autonomy_compact" facet (see the retirement comment on
# test_build_mind_run_request_attaches_evidence_facets above and in
# app/mind_runtime.py's _attach_mind_evidence_facets). It suffered the same
# fate two weeks later: DriveEngine (drive_audits' only producer) was deleted
# 2026-07-30, making drive_audits write-never -- the facet could only ever
# hand Mind a frozen historical row dressed up as live self-state, which is
# worse than no facet at all (CLAUDE.md Section 0A "no empty-shell cognition").
# `fetch_drive_state_facet_for_mind`, `_query_latest_drive_audit_row`, and the
# `drive_state_compact` wiring in `prepare_plan_context_for_mind_projection`/
# `_attach_mind_evidence_facets` were removed outright along with the tests
# that used to cover them (test_fetch_drive_state_facet_*,
# test_build_mind_run_request_attaches_drive_state_facet_and_omits_autonomy_
# compact, test_build_mind_run_request_omits_drive_state_facet_when_absent,
# test_prepare_plan_context_merges_drive_state_and_diag,
# test_query_latest_drive_audit_row_uses_memory_extractors_pool_not_a_duplicate).
