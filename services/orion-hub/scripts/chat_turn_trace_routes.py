"""Fused correlation-first chat-turn trace lookup.

Ground truth (see docs/superpowers/plans/2026-07-28-unified-chat-traceability.md
if written, or the brainstorm this implements): a single chat turn's execution
truth is scattered across at least three independently-built, independently-
keyed stores that were never joined behind one lookup:

- ``CognitionTraceCache`` (Runtime Trace Nexus Milestone A) -- covers the
  classic PlanRunner RPC path, keyed by ``correlation_id`` directly.
- Grammar Atlas (``orion.grammar.ledger`` / ``orion.grammar.query``) -- covers
  the unified-turn harness motor's atom/edge trace, keyed by a *derived*
  ``trace_id`` (``cortex_exec_trace_id(node, correlation_id, lane="harness_motor")``).
- ``ExecutionTrajectoryProjectionV1.runs`` (orion-substrate-runtime) -- the
  harness motor's per-run pressure signal (step counts, failure streaks,
  compliance deficit), keyed by the *same* derived trace_id.

This module is a pure read-side join across the three -- no new producer, no
new schema, no new persisted store. ``route_signal_inferred`` is a best-effort
label derived from which sources actually returned data; it is NOT the
authoritative ``chat_route`` tag written into a turn's persisted
``chat_turn`` envelope spark_meta (see ``orion/hub/chat_route.py``) -- wiring
that in as a fourth, authoritative source is a natural follow-up, not done
here, to keep this slice thin.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from fastapi import APIRouter
from sqlalchemy import create_engine, text

from orion.schemas.execution_projection import ExecutionTrajectoryProjectionV1
from orion.substrate.execution_loop.constants import EXECUTION_TRAJECTORY_PROJECTION_ID
from orion.substrate.execution_loop.ids import cortex_exec_trace_id

from . import grammar_atlas_routes
from .settings import settings

logger = logging.getLogger("orion-hub.chat_turn_trace")

router = APIRouter(prefix="/api/chat/turn", tags=["chat-turn-trace"])

_exec_traj_engine = None


def _exec_trajectory_engine():
    global _exec_traj_engine
    if _exec_traj_engine is None:
        uri = os.getenv("POSTGRES_URI", "").strip()
        if not uri:
            return None
        _exec_traj_engine = create_engine(uri, pool_pre_ping=True)
    return _exec_traj_engine


def _load_execution_run(trace_id: str) -> dict[str, Any] | None:
    """Look up this turn's harness-motor run inside the (singleton, capped)
    execution trajectory projection. Degrades to None -- never raises -- on
    any DB/config/parse failure, matching every sibling substrate_*_routes.py
    read helper in this package.
    """
    engine = _exec_trajectory_engine()
    if engine is None:
        return None
    try:
        with engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT projection_json FROM substrate_execution_trajectory_projection
                        WHERE projection_id = :projection_id
                        """
                    ),
                    {"projection_id": EXECUTION_TRAJECTORY_PROJECTION_ID},
                )
                .mappings()
                .first()
            )
    except Exception:
        logger.warning("chat_turn_trace execution_trajectory query failed", exc_info=True)
        return None
    if not row:
        return None
    payload = row["projection_json"]
    if isinstance(payload, str):
        payload = json.loads(payload)
    try:
        projection = ExecutionTrajectoryProjectionV1.model_validate(payload)
    except Exception:
        logger.warning("chat_turn_trace execution_trajectory payload invalid", exc_info=True)
        return None
    run = projection.runs.get(trace_id)
    if run is None:
        return None
    return run.model_dump(mode="json")


async def _load_cognition_trace(correlation_id: str) -> dict[str, Any] | None:
    import scripts.main as hub_main

    cache = getattr(hub_main, "cognition_trace_cache", None)
    if cache is None or not cache.enabled:
        return None
    return await cache.get_redacted(correlation_id)


async def _load_grammar_trace(trace_id: str) -> dict[str, Any] | None:
    try:
        q = grammar_atlas_routes._grammar_query()
    except Exception:
        logger.debug("chat_turn_trace grammar atlas unavailable (import/config)", exc_info=True)
        return None
    try:
        return await grammar_atlas_routes._with_session(lambda sess: q.get_trace(sess, trace_id))
    except Exception:
        logger.debug("chat_turn_trace grammar atlas query failed", exc_info=True)
        return None


async def get_fused_chat_turn_trace(correlation_id: str) -> dict[str, Any]:
    corr = str(correlation_id or "").strip()
    harness_trace_id = cortex_exec_trace_id(settings.NODE_NAME, corr, lane="harness_motor")

    sources: dict[str, Any] = {}
    gaps: list[str] = []

    cognition_trace = await _load_cognition_trace(corr)
    if cognition_trace is not None:
        sources["cognition_trace"] = cognition_trace
    else:
        gaps.append("no_cognition_trace")

    grammar_trace = await _load_grammar_trace(harness_trace_id)
    if grammar_trace is not None:
        sources["grammar_trace"] = grammar_trace
    else:
        gaps.append("no_grammar_trace")

    execution_run = _load_execution_run(harness_trace_id)
    if execution_run is not None:
        sources["execution_run"] = execution_run
    else:
        gaps.append("no_execution_run_pressure_signal")

    has_classic = "cognition_trace" in sources
    has_unified = "grammar_trace" in sources or "execution_run" in sources
    if has_classic and has_unified:
        route_signal = "ambiguous_both_paths_present"
    elif has_classic:
        route_signal = "classic_planrunner"
    elif has_unified:
        route_signal = "unified_turn_harness"
    else:
        route_signal = "none"

    return {
        "correlation_id": corr,
        "harness_trace_id": harness_trace_id,
        "route_signal_inferred": route_signal,
        "sources": sources,
        "complete": bool(sources),
        "gaps": gaps,
    }


@router.get("/{correlation_id}/trace")
async def api_chat_turn_trace(correlation_id: str) -> dict[str, Any]:
    """Fused trace lookup: never 404s on its own -- a turn that produced no
    trace in any of the three stores is a real, reportable fact (``gaps``),
    not an error. 404 is left to callers that need "not found" semantics.
    """
    return await get_fused_chat_turn_trace(correlation_id)
