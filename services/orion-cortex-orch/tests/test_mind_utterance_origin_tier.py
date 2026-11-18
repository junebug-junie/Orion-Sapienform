"""cortex-orch hands Hub's "who typed this" mark to orion-mind.

The Hub browser's brain-mode chat with the Mind toggle reaches Mind only through cortex-orch.
Hub's build_chat_request writes metadata.utterance_origin="juniper"; orion-mind lifts a run to the
interactive (never heat-shed) tier only when that mark arrives (engine.carries_live_turn).
Body-economy design Organ 3b (2026-10-23): tiers attach to callers, not lanes.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_guard = Path(__file__).resolve().parent / "_orch_import_guard.py"
_spec = importlib.util.spec_from_file_location("_orch_guard_boot_origin", _guard)
assert _spec and _spec.loader
_guard_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_guard_mod)

ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, APP_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from orion.schemas.cortex.contracts import CortexClientContext, CortexClientRequest, LLMMessage  # noqa: E402
from orion.schemas.cortex.schemas import ExecutionPlan, ExecutionStep, PlanExecutionRequest  # noqa: E402


def _build(metadata: dict):
    _guard_mod.ensure_orion_cortex_orch_app()
    from app.mind_runtime import build_mind_run_request

    client = CortexClientRequest(
        verb="chat_general",
        mode="brain",
        context=CortexClientContext(
            messages=[LLMMessage(role="user", content="hi")],
            session_id="s", trace_id="t", user_message="hi", metadata=metadata,
        ),
    )
    plan = PlanExecutionRequest(
        plan=ExecutionPlan(
            verb_name="chat_general",
            steps=[ExecutionStep(verb_name="chat_general", step_name="noop", order=0, services=[])],
        ),
        context={"metadata": {}},
    )
    return build_mind_run_request(
        client, plan, "550e8400-e29b-41d4-a716-446655440000",
        substrate_telemetry_facet={"status": "absent"},
        cognitive_projection_facet={"schema_version": "cognitive.projection.v1", "projection_id": "x"},
    )


@pytest.mark.parametrize("origin", ["juniper", "orion"])
def test_metadata_origin_reaches_the_mind_request(origin) -> None:
    req = _build({"mind_enabled": True, "utterance_origin": origin})
    assert req.utterance_origin == origin


@pytest.mark.parametrize("raw", [None, "", "JUNIPER", "someone", 1])
def test_missing_or_unknown_origin_is_none(raw) -> None:
    meta = {"mind_enabled": True}
    if raw is not None:
        meta["utterance_origin"] = raw
    assert _build(meta).utterance_origin is None
