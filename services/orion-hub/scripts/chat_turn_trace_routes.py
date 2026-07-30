"""Fused correlation-first chat-turn trace lookup.

Ground truth (see docs/superpowers/plans/2026-07-28-unified-chat-traceability.md
if written, or the brainstorm this implements): a single chat turn's execution
truth is scattered across independently-built, independently-keyed stores that
were never joined behind one lookup:

- ``CognitionTraceCache`` (Runtime Trace Nexus Milestone A) -- covers the
  classic PlanRunner RPC path, keyed by ``correlation_id`` directly.
- Grammar Atlas (``orion.grammar.ledger`` / ``orion.grammar.query``) -- covers
  the unified-turn harness motor's atom/edge trace, keyed by a *derived*
  ``trace_id`` (``cortex_exec_trace_id(node, correlation_id, lane="harness_motor")``).
- ``ExecutionTrajectoryProjectionV1.runs`` (orion-substrate-runtime) -- the
  harness motor's per-run pressure signal (step counts, failure streaks,
  compliance deficit), keyed by the *same* derived trace_id.
- ``thought_decision`` (orion-sql-writer) -- the unified turn's stance
  decision (proceed/defer/refuse + reasons), keyed by ``correlation_id``
  directly. sql-writer only started persisting this table as of the same
  patch that added this module (see orion/schemas/thought.py's
  ThoughtDecisionRecordV1) -- ThoughtEventV1 itself was already broadcasting
  live on orion:thought:artifact well before that, just never durably kept.
- ``harness_turn_trace`` (orion-sql-writer) -- the unified turn's finalize
  chain: HarnessRunV1 (5a substrate_appraisal, 5b reflection, draft_text,
  finalize flags), HarnessVerdictMoleculeV1, HarnessTurnOutcomeMoleculeV1,
  HarnessPostTurnClosureV1. All four already PUBLISH unredacted on the bus
  (orion/harness/finalize.py); sql-writer only started persisting them as of
  the same patch that added this source (see
  services/orion-sql-writer/app/harness_turn_trace_persist.py). Keyed by
  correlation_id directly, one row upserted incrementally as each of the
  four molecules arrives -- a partial row mid-turn is normal, not an error.

This module is a pure read-side join -- no new schema beyond
ThoughtDecisionRecordV1 (now unredacted -- imperative/tone/strain_refs/
stance_harness_slice are real Turn Trace operator debug content, not
placeholder) and the harness_turn_trace table. ``route_signal_inferred`` is
a best-effort label derived from which sources actually returned data; it is
NOT the authoritative ``chat_route`` tag written into a turn's persisted
``chat_turn`` envelope spark_meta (see ``orion/hub/chat_route.py``) -- that
stays a candidate future source, since reading it back requires a
chat_history_log lookup this module doesn't otherwise need.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from fastapi import APIRouter
from sqlalchemy import create_engine, text

from orion.schemas.execution_projection import ExecutionTrajectoryProjectionV1
from orion.schemas.thought import ThoughtDecisionRecordV1
from orion.substrate.execution_loop.constants import EXECUTION_TRAJECTORY_PROJECTION_ID
from orion.substrate.execution_loop.ids import cortex_exec_trace_id

from . import grammar_atlas_routes
from .settings import settings

logger = logging.getLogger("orion-hub.chat_turn_trace")

router = APIRouter(prefix="/api/chat/turn", tags=["chat-turn-trace"])

_engine = None


def _postgres_engine():
    """Shared engine for the conjourney Postgres DB -- same POSTGRES_URI
    already used by every sibling substrate_*_routes.py read helper in this
    package, and the same DB orion-substrate-runtime and orion-sql-writer
    write to.
    """
    global _engine
    if _engine is None:
        uri = os.getenv("POSTGRES_URI", "").strip()
        if not uri:
            return None
        _engine = create_engine(uri, pool_pre_ping=True)
    return _engine


def _load_execution_run(trace_id: str) -> dict[str, Any] | None:
    """Look up this turn's harness-motor run inside the (singleton, capped)
    execution trajectory projection. Degrades to None -- never raises -- on
    any DB/config/parse failure, matching every sibling substrate_*_routes.py
    read helper in this package.
    """
    engine = _postgres_engine()
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


def _load_thought_decision(correlation_id: str) -> dict[str, Any] | None:
    """Look up this turn's persisted stance decision. Already redacted at
    write time (ThoughtDecisionSQL only declares the redacted subset of
    columns -- see services/orion-sql-writer/app/models/thought_decision.py),
    so the row is safe to return as-is.
    """
    engine = _postgres_engine()
    if engine is None:
        return None
    try:
        with engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT event_id, correlation_id, session_id, created_at,
                               imperative, tone, strain_refs, stance_harness_slice,
                               disposition, disposition_reasons, boundary_register,
                               repair_pressure_level, trust_rupture_score,
                               llm_profile, producer, model_id
                        FROM thought_decision
                        WHERE correlation_id = :correlation_id
                        ORDER BY created_at DESC
                        LIMIT 1
                        """
                    ),
                    {"correlation_id": correlation_id},
                )
                .mappings()
                .first()
            )
    except Exception:
        logger.warning("chat_turn_trace thought_decision query failed", exc_info=True)
        return None
    if not row:
        return None
    try:
        record = ThoughtDecisionRecordV1.model_validate(dict(row))
    except Exception:
        logger.warning("chat_turn_trace thought_decision row failed schema validation", exc_info=True)
        return None
    return record.model_dump(mode="json")


def _load_harness_turn_trace(correlation_id: str) -> dict[str, Any] | None:
    """Look up the durable finalize-chain capture: run_artifact (HarnessRunV1 --
    draft_text, substrate_appraisal, reflection, finalize flags), verdict_molecule,
    outcome_molecule, closure. All four already publish unredacted on the bus
    (orion/harness/finalize.py); this is a straight read of what
    services/orion-sql-writer/app/harness_turn_trace_persist.py persisted.
    Partial rows are normal -- a turn mid-flight may have run_artifact but not
    yet outcome_molecule/closure.
    """
    engine = _postgres_engine()
    if engine is None:
        return None
    try:
        with engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT correlation_id, created_at, updated_at,
                               run_artifact, verdict_molecule, outcome_molecule, closure
                        FROM harness_turn_trace
                        WHERE correlation_id = :correlation_id
                        """
                    ),
                    {"correlation_id": correlation_id},
                )
                .mappings()
                .first()
            )
    except Exception:
        logger.warning("chat_turn_trace harness_turn_trace query failed", exc_info=True)
        return None
    if not row:
        return None
    result = dict(row)
    for key in ("created_at", "updated_at"):
        if result.get(key) is not None:
            result[key] = str(result[key])
    return result


async def _load_cognition_trace(correlation_id: str) -> dict[str, Any] | None:
    import scripts.main as hub_main

    cache = getattr(hub_main, "cognition_trace_cache", None)
    if cache is None or not cache.enabled:
        return None
    return await cache.get_redacted(correlation_id)


async def _load_grammar_trace(trace_id: str) -> dict[str, Any] | None:
    """Reuses grammar_atlas_routes' query plumbing, but must independently
    respect GRAMMAR_ATLAS_ENABLED -- unlike every real /api/substrate/atlas/*
    route, this helper never calls _require_atlas_available() (that raises
    HTTPException on disable/misconfig, which would fail the whole fused
    lookup rather than degrading to a gap), so the enablement check has to be
    duplicated here rather than inherited.
    """
    try:
        from app.settings import get_settings

        if not get_settings().GRAMMAR_ATLAS_ENABLED:
            return None
    except Exception:
        logger.debug("chat_turn_trace grammar atlas settings unavailable", exc_info=True)
        return None
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

    thought_decision = _load_thought_decision(corr)
    if thought_decision is not None:
        sources["thought_decision"] = thought_decision
    else:
        gaps.append("no_thought_decision")

    harness_turn_trace = _load_harness_turn_trace(corr)
    if harness_turn_trace is not None:
        sources["harness_turn_trace"] = harness_turn_trace
    else:
        gaps.append("no_harness_turn_trace")

    has_classic = "cognition_trace" in sources
    has_unified = (
        "grammar_trace" in sources
        or "execution_run" in sources
        or "thought_decision" in sources
        or "harness_turn_trace" in sources
    )
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
    trace in any of the four stores is a real, reportable fact (``gaps``),
    not an error. 404 is left to callers that need "not found" semantics.
    """
    return await get_fused_chat_turn_trace(correlation_id)
