"""orion-thought → orion-mind advisory enrichment (unified turn coloring).

The unified turn computes stance cold via the ``stance_react`` verb. This module
optionally runs Mind first and selects a strict, mode-agnostic self/attention
subset as an *advisory* prompt prior. ``stance_react`` remains the sole author of
ThoughtEventV1 and reconciles this coloring. Everything fails open.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

import httpx

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.mind.constants import MIND_RUN_ARTIFACT_SCHEMA_ID
from orion.mind.v1 import MindRunPolicyV1, MindRunRequestV1, MindRunResultV1
from orion.schemas.mind.artifact import MindRunArtifactV1
from orion.schemas.thought import StanceReactRequestV1

logger = logging.getLogger("orion-thought.mind_enrichment")


def _envelope_correlation_id(raw: str | None) -> UUID:
    if raw:
        try:
            return UUID(str(raw))
        except ValueError:
            pass
    return uuid4()

# Strict allow-list of coloring keys. Any un-listed ChatStanceBrief / decision
# field is absent by construction (no deny-list, no leakage of future fields).
MIND_COLORING_ALLOWED_KEYS: frozenset[str] = frozenset(
    {
        "attention_frontier",
        "reflective_themes",
        "curiosity_threads",
        "self_relevance",
        "identity_salience",
        "juniper_relevance",
        "mind_quality",
        "mind_run_id",
        "snapshot_hash",
    }
)

_MAX_STR_CHARS = 240
_MAX_USER_TEXT_CHARS = 20_000
_MAX_OPEN_LOOPS = 6


def _clip(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()[:_MAX_STR_CHARS]
    return value


def _clip_str_or_none(value: Any) -> str | None:
    """Coerce a scalar to a stripped, length-bounded str; None otherwise.

    Non-scalar values (dict/list/etc.) are dropped rather than stringified, so a
    nested structure can never inject its keys/values into the coloring — this
    preserves the strict no-nested-leak property even for future schema drift.
    These fields are semantically scalars (a short str / Literal) in ChatStanceBrief.
    """
    if not isinstance(value, (str, int, float, bool)):
        return None
    text = str(value).strip()[:_MAX_STR_CHARS]
    return text or None


def _str_list(value: Any, *, max_items: int) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    out: list[str] = []
    for item in value:
        text = str(item).strip()[:_MAX_STR_CHARS]
        if text:
            out.append(text)
        if len(out) >= max_items:
            break
    return out


def select_mind_coloring(result: MindRunResultV1, *, max_items: int = 3) -> dict[str, Any] | None:
    """Project the mode-agnostic self/attention subset of a Mind run.

    Returns None (skip enrichment) unless the run is ok AND produced
    meaningful_synthesis AND carries at least one substantive signal. Never
    injects an empty shell. Selection is a strict allow-list.
    """
    if not result.ok:
        return None
    brief = result.brief
    if brief.mind_quality != "meaningful_synthesis":
        return None

    frontier = brief.active_frontier
    selected = list(frontier.selected) if frontier is not None else []
    selected = selected[:max_items]
    attention_frontier = [
        {
            "label": _clip(m.label),
            "summary": _clip(m.summary),
            "score": round(float(m.score), 4),
        }
        for m in selected
    ]
    curiosity_threads = [_clip(m.summary) for m in selected if str(m.summary).strip()]

    stance_payload = brief.stance_payload if isinstance(brief.stance_payload, dict) else {}
    reflective_themes = _str_list(stance_payload.get("reflective_themes"), max_items=max_items)
    self_relevance = _clip_str_or_none(stance_payload.get("self_relevance"))
    identity_salience = _clip_str_or_none(stance_payload.get("identity_salience"))
    juniper_relevance = _clip_str_or_none(stance_payload.get("juniper_relevance"))

    # No empty-shell cognition: require at least one substantive signal.
    has_substance = bool(
        attention_frontier or reflective_themes or curiosity_threads
        or self_relevance or juniper_relevance
    )
    if not has_substance:
        return None

    return {
        "attention_frontier": attention_frontier,
        "reflective_themes": reflective_themes,
        "curiosity_threads": curiosity_threads,
        "self_relevance": self_relevance,
        "identity_salience": identity_salience,
        "juniper_relevance": juniper_relevance,
        "mind_quality": brief.mind_quality,
        "mind_run_id": str(result.mind_run_id),
        "snapshot_hash": result.snapshot_hash,
    }


def _situation_compact_from_broadcast(request: StanceReactRequestV1) -> dict[str, Any] | None:
    """Fold real open-loop / selected-description text into the accepted
    situation_compact facet. Returns None when there is no usable text.
    """
    broadcast = request.association.broadcast
    if broadcast is None:
        return None
    loops: list[dict[str, str]] = []
    for loop in (broadcast.frame.open_loops or [])[:_MAX_OPEN_LOOPS]:
        description = (loop.description or "").strip()
        if not description:
            continue
        entry: dict[str, str] = {"description": description[:_MAX_STR_CHARS]}
        why = (loop.why_it_matters or "").strip()
        if why:
            entry["why_it_matters"] = why[:_MAX_STR_CHARS]
        loops.append(entry)
    selected = (broadcast.selected_description or "").strip()
    if not loops and not selected:
        return None
    compact: dict[str, Any] = {"attention_situation": True}
    if selected:
        compact["selected_focus"] = selected[:_MAX_STR_CHARS]
    if loops:
        compact["open_loops"] = loops
    return compact


# --- drive_state_compact facet (bounded, fail-open Postgres fetch) ---
#
# Mirrors orion-cortex-orch's `fetch_drive_state_facet_for_mind` in
# services/orion-cortex-orch/app/mind_runtime.py (same query, same diagnostics
# shape, same fail-open contract) for orion-thought's independent "light Mind"
# path. orion-cortex-orch uses asyncpg via a shared lazy pool; orion-thought
# has no asyncpg dependency, so this rides the existing sync psycopg2/
# SQLAlchemy seam already used by store.py (`_get_engine()`/`_database_url()`)
# and bounds it with asyncio.wait_for(asyncio.to_thread(...)) so a slow/failed
# query can never block the event loop past the configured timeout.
DRIVE_AUDITS_LATEST_QUERY_FOR_THOUGHT = (
    "SELECT dominant_drive, active_drives, drive_pressures, summary, "
    "COALESCE(observed_at, created_at) AS observed_at "
    "FROM drive_audits WHERE subject = 'orion' "
    "ORDER BY COALESCE(observed_at, created_at) DESC LIMIT 1"
)

# asyncio.wait_for/asyncio.to_thread cannot kill a stuck OS thread -- on the
# asyncio-level timeout the executor thread (and whatever Postgres connection
# it checked out from store.py's shared `_get_engine()` pool) keeps running
# until it naturally returns. A server-side statement_timeout, scoped to just
# this query's own transaction via SET LOCAL (reverts automatically, so it
# never leaks into the pooled connection for the *next* checkout by any of
# store.py's other callers), bounds that residual exposure. Same idiom as
# services/orion-cortex-orch/app/memory_inject.py's psycopg2 SET LOCAL
# statement_timeout for its own bounded, fail-open Postgres read.
_DRIVE_STATE_QUERY_STATEMENT_TIMEOUT_MS = 300


def _coerce_jsonb(value: Any) -> Any:
    """psycopg2's default JSONB codec decodes to python objects already; this is a
    guard for driver/codec configurations where the column instead comes back as a
    raw JSON string (mirrors orion-cortex-orch mind_runtime._coerce_jsonb)."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return None
    return value


def _query_latest_drive_audit_row_sync() -> dict[str, Any] | None:
    """Sync single-row fetch via orion-thought's existing SQLAlchemy engine.

    Reuses `store._get_engine()` rather than duplicating engine-creation logic —
    the same lazy module-level engine already used for reverie persistence,
    pointed at the same `conjourney` Postgres instance `orion-sql-writer`
    writes `drive_audits` to (see `store._database_url()`). Runs on a worker
    thread via `asyncio.to_thread` in the caller; must stay synchronous.
    """
    from sqlalchemy import text

    from .store import _get_engine

    engine = _get_engine()
    # engine.begin() (not .connect()) so `SET LOCAL statement_timeout` is
    # transaction-scoped: it bounds only this query and reverts automatically
    # when the transaction ends, instead of persisting on the pooled
    # connection for whichever store.py caller checks it out next.
    with engine.begin() as conn:
        conn.execute(
            text(f"SET LOCAL statement_timeout = '{_DRIVE_STATE_QUERY_STATEMENT_TIMEOUT_MS}ms'")
        )
        row = conn.execute(text(DRIVE_AUDITS_LATEST_QUERY_FOR_THOUGHT)).mappings().first()
    return dict(row) if row is not None else None


async def fetch_drive_state_facet_for_thought(
    correlation_id: str, *, settings: Any
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Bounded, fail-open fetch of the latest DriveEngine `drive_audits` row (subject='orion').

    Returns ``(drive_state_compact_or_none, diagnostics)``. On timeout, connection
    failure, missing table, or no matching/meaningful row, this returns
    ``(None, diagnostics)`` without raising — the Mind request must never stall
    or break on this facet's absence. Same contract as orion-cortex-orch's
    `fetch_drive_state_facet_for_mind`, adapted to orion-thought's sync DB seam.
    """
    timeout_sec = max(0.01, float(getattr(settings, "mind_drive_state_fetch_timeout_sec", 0.4)))
    diagnostics: dict[str, Any] = {
        "correlation_id": correlation_id,
        "timeout_sec": timeout_sec,
        "ok": False,
        "elapsed_ms": 0,
        "timed_out": False,
        "exception_type": None,
        "reason": "start",
    }
    t0 = time.perf_counter()
    try:
        row = await asyncio.wait_for(
            asyncio.to_thread(_query_latest_drive_audit_row_sync), timeout=timeout_sec
        )
    except asyncio.TimeoutError as exc:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        diagnostics.update(
            {
                "elapsed_ms": elapsed_ms,
                "timed_out": True,
                "exception_type": type(exc).__name__,
                "reason": "timeout",
            }
        )
        logger.warning(
            "mind_drive_state_fetch_timeout corr=%s elapsed_ms=%s timeout_sec=%s",
            correlation_id,
            elapsed_ms,
            timeout_sec,
        )
        return None, diagnostics
    except Exception as exc:  # noqa: BLE001 — fail-open by contract
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        diagnostics.update(
            {
                "elapsed_ms": elapsed_ms,
                "exception_type": type(exc).__name__,
                "reason": "exception",
                "degradation_reason": str(exc),
            }
        )
        logger.warning(
            "mind_drive_state_fetch_failed corr=%s elapsed_ms=%s exc_type=%s err=%s",
            correlation_id,
            elapsed_ms,
            type(exc).__name__,
            exc,
        )
        return None, diagnostics

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    diagnostics["elapsed_ms"] = elapsed_ms
    if row is None:
        diagnostics.update({"ok": True, "reason": "no_rows"})
        return None, diagnostics

    active_drives = _coerce_jsonb(row.get("active_drives"))
    dominant_drive = row.get("dominant_drive")
    summary = row.get("summary")
    # Content check, not just row-existence: a quiet tick (nothing crossed
    # activation threshold that tick) writes dominant_drive=None, summary=None,
    # active_drives=[] to drive_audits. Attaching that as a real signal would be
    # the "schema-valid payload with meaningless content" pattern CLAUDE.md
    # Section 0A calls out — fail open here the same way a missing row does.
    if not dominant_drive and not summary and not (isinstance(active_drives, list) and active_drives):
        diagnostics.update({"ok": True, "reason": "no_meaningful_content"})
        return None, diagnostics

    observed_at = row.get("observed_at")
    compact = {
        "dominant_drive": dominant_drive,
        "active_drives": active_drives,
        "drive_pressures": _coerce_jsonb(row.get("drive_pressures")),
        "summary": summary,
        "observed_at": observed_at.isoformat() if hasattr(observed_at, "isoformat") else observed_at,
    }
    diagnostics.update({"ok": True, "reason": "success"})
    logger.info(
        "mind_drive_state_fetch_result corr=%s ok=true elapsed_ms=%s dominant_drive=%s",
        correlation_id,
        elapsed_ms,
        compact["dominant_drive"],
    )
    return compact, diagnostics


def build_light_mind_request(
    request: StanceReactRequestV1,
    *,
    wall_time_ms: int,
    router_profile: str,
    drive_state_compact: dict[str, Any] | None = None,
) -> MindRunRequestV1:
    """Build a bounded Mind request with NO cognitive-projection cold rebuild.

    Evidence in v1 is the current user turn (as a single current_turn item),
    plus recall_bundle (only if already threaded on stance_inputs), a
    situation_compact facet derived from the attention broadcast, and an
    optional drive_state_compact facet (bounded-fetched by the caller via
    `fetch_drive_state_facet_for_thought`). This function stays synchronous/pure;
    the async DB fetch happens in the caller.
    """
    user_text = (request.user_message or "").strip()[:_MAX_USER_TEXT_CHARS]
    snapshot: dict[str, Any] = {"user_text": user_text, "messages_tail": []}

    facets: dict[str, Any] = {}
    stance_inputs = request.stance_inputs if isinstance(request.stance_inputs, dict) else {}
    recall_bundle = stance_inputs.get("recall_bundle")
    if isinstance(recall_bundle, dict) and recall_bundle:
        facets["recall_bundle"] = recall_bundle
    situation = _situation_compact_from_broadcast(request)
    if situation:
        facets["situation_compact"] = situation
    if isinstance(drive_state_compact, dict) and drive_state_compact:
        facets["drive_state_compact"] = drive_state_compact
    if facets:
        snapshot["facets"] = facets

    return MindRunRequestV1(
        correlation_id=request.correlation_id,
        session_id=request.session_id,
        trigger="user_turn",
        snapshot_inputs=snapshot,
        policy=MindRunPolicyV1(
            n_loops_max=1,
            wall_time_ms_max=max(1, int(wall_time_ms)),
            router_profile_id=router_profile or "default",
        ),
    )


async def publish_mind_run_artifact_for_thought(
    bus: Any,
    *,
    source: "ServiceRef",
    request: "StanceReactRequestV1",
    mind_req: "MindRunRequestV1",
    mind_res: MindRunResultV1,
    channel: str,
) -> None:
    """Publish MindRunArtifactV1 for a unified-turn Mind run (mode='orion').

    Log-and-continue: an artifact publish failure must never fail the stance stage.
    """
    try:
        summary = {
            "correlation_id": request.correlation_id,
            "verb": "stance_react",
            "mode": "orion",
            "session_id": request.session_id,
        }
        artifact = MindRunArtifactV1(
            mind_run_id=mind_res.mind_run_id,
            correlation_id=request.correlation_id,
            session_id=request.session_id,
            trigger=mind_req.trigger,
            ok=mind_res.ok,
            error_code=mind_res.error_code,
            snapshot_hash=mind_res.snapshot_hash,
            router_profile_id=mind_req.policy.router_profile_id,
            result_jsonb=mind_res.model_dump(mode="json"),
            request_summary_jsonb=summary,
            created_at_utc=datetime.now(timezone.utc),
        )
        env = BaseEnvelope(
            kind=MIND_RUN_ARTIFACT_SCHEMA_ID,
            source=source,
            correlation_id=_envelope_correlation_id(request.correlation_id),
            payload=artifact.model_dump(mode="json"),
        )
        await bus.publish(channel, env)
        logger.info(
            "mind_run_artifact_publish corr=%s mind_run_id=%s mode=orion ok=%s",
            request.correlation_id,
            artifact.mind_run_id,
            artifact.ok,
        )
    except Exception as exc:  # noqa: BLE001 — observability must never fail the turn
        logger.warning(
            "mind_artifact_publish_failed corr=%s err=%s",
            request.correlation_id,
            exc,
        )


def _mind_transport() -> httpx.BaseTransport | None:
    """Seam for tests to inject an httpx.MockTransport. Returns None in prod
    so AsyncClient uses its default transport.
    """
    return None


async def run_mind_for_thought(
    req: "MindRunRequestV1",
    *,
    settings: Any,
    correlation_id: str,
) -> "MindRunResultV1 | None":
    """POST the Mind request; return the parsed result or None (fail-open)."""
    base = (getattr(settings, "mind_base_url", "") or "").rstrip("/")
    if not base:
        logger.warning("mind_enrichment_failed corr=%s reason=unconfigured_base_url", correlation_id)
        return None
    url = f"{base}/v1/mind/run"
    timeout_sec = float(getattr(settings, "mind_timeout_sec", 15.0))
    timeout = httpx.Timeout(
        connect=min(10.0, timeout_sec),
        read=timeout_sec,
        write=min(30.0, timeout_sec),
        pool=5.0,
    )
    max_body = int(getattr(settings, "mind_max_response_bytes", 2_000_000))
    transport = _mind_transport()
    client_kwargs: dict[str, Any] = {"timeout": timeout}
    if transport is not None:
        client_kwargs["transport"] = transport
    try:
        async with httpx.AsyncClient(**client_kwargs) as client:
            resp = await client.post(url, json=req.model_dump(mode="json"))
            resp.raise_for_status()
            raw = resp.content
            if len(raw) > max_body:
                raise RuntimeError(f"mind_response_too_large:{len(raw)}")
            return MindRunResultV1.model_validate(resp.json())
    except Exception as exc:  # noqa: BLE001 — fail-open by contract
        logger.warning(
            "mind_enrichment_failed corr=%s reason=%s err=%s",
            correlation_id,
            type(exc).__name__,
            exc,
        )
        return None
