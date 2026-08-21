"""Best-effort persistence of the live-chat-scoped attention/curiosity
salience score into the same `attention_salience_trace` table
`orion-thought`'s reverie loop already writes to (`services/orion-thought/
app/store.py::persist_salience_trace`), with `scope="chat"` instead of
`scope="reverie"`.

Why this exists: `chat_stance.py:build_chat_stance_inputs` runs the SAME
`score_loop`/`select_actions`/`build_open_loops` scoring
(`orion.substrate.attention_frame.build_attention_frame`) on every real chat
turn that reverie runs on its own synthetic ticks -- but nothing persisted
it. Confirmed live 2026-08-19: `SELECT count(*) FROM cognition_traces WHERE
metadata::text LIKE '%chat_attention_frame%'` returned 0 rows, ever. That
means `orion/substrate/attention/policy.py`'s `select_actions()` thresholds
(`min_ask`, inline 0.48/0.35 cutoffs -- both explicitly disclosed 2026-07-31
as uncalibrated against the new Borda rank-aggregated score, see that
file's own comment and `orion/sentience_striving_program/README.md`'s
2026-07-31 entry) have never had a single real chat-traffic data point to
recalibrate against -- only reverie's system-substrate-node ticks, which
are a different, much narrower population (100% `substrate:node:*`
descriptions, confirmed live against the 476 rows persisted so far). This
module closes that gap: instrumentation first, recalibration second
(CLAUDE.md's metric-quality-gate step 4, "live-data sanity check", cannot
even start without real rows to look at).

Traces only the selected loop (mirrors reverie's `build_salience_trace`
one-row-per-tick shape), not the full open_loops set -- keeps write volume
proportional to real chat turns, not to open-loop fan-out. Fail-open, never
raises, bounded by an outer asyncio timeout: a slow/unreachable DB must
degrade to "no trace" and never delay or break a real chat turn.

Mirrors this service's own established reader conventions (module-level
cached engine, DSN resolution via the shared felt-state/endogenous-runtime
DSN fallback chain, per-connection `statement_timeout` GUC, `asyncio.
wait_for(asyncio.to_thread(...))`) -- see `metacog_trend_reader.py`'s and
`perception_reader.py`'s docstrings for the same shape on the read side.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

from sqlalchemy import create_engine, text

from orion.core.ids import stable_hash_id
from orion.schemas.attention_frame import AttentionFrameV1

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "yes", "on"}

# Provenance string mirrors orion.substrate.attention.salience.WEIGHTS_VERSION /
# services/orion-thought/app/reverie.py::_weights_version() -- same formula,
# same version tag, just a different `scope` value in the same table.
_WEIGHTS_VERSION = "gwt-coalition-v1"

# Per-connection statement_timeout (ms). This runs on the hot chat-turn path,
# so it stays tight -- same order of magnitude as metacog_trend_reader.py's
# 800ms read budget, not reverie's leisurely best-effort background-tick one.
_QUERY_STATEMENT_TIMEOUT_MS = 800

_ENGINE = None
_ENGINE_URL: str | None = None


def _flag_enabled() -> bool:
    return os.getenv("ENABLE_CHAT_ATTENTION_SALIENCE_TRACE", "true").strip().lower() in _TRUTHY


def _dsn() -> str:
    # Same conjourney instance every other cortex-exec Postgres reader/writer
    # in this service targets. Reuses the felt-state / endogenous-runtime DSN
    # fallback chain rather than adding a third DB-URL env key for the same
    # database (see .env_example's ENABLE_METACOG_TREND_CUE comment: "DB URL
    # reuses ... rather than a third DSN key for the same conjourney instance").
    return (
        os.getenv("SUBSTRATE_FELT_STATE_DATABASE_URL", "").strip()
        or os.getenv("ENDOGENOUS_RUNTIME_SQL_DATABASE_URL", "").strip()
    )


def _timeout_sec() -> float:
    raw = os.getenv("CHAT_ATTENTION_SALIENCE_TRACE_TIMEOUT_SEC", "0.8").strip()
    try:
        return max(0.01, float(raw))
    except ValueError:
        return 0.8


def _get_engine():
    global _ENGINE, _ENGINE_URL
    url = _dsn()
    if not url:
        return None
    if _ENGINE is None or _ENGINE_URL != url:
        _ENGINE = create_engine(
            url,
            pool_pre_ping=True,
            connect_args={"options": f"-c statement_timeout={_QUERY_STATEMENT_TIMEOUT_MS}"},
        )
        _ENGINE_URL = url
    return _ENGINE


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def build_chat_salience_trace_row(frame: AttentionFrameV1) -> dict[str, Any] | None:
    """Project the selected loop's feature vector + score into an
    `attention_salience_trace` row shape. `None` if nothing was selected --
    matches reverie's `build_salience_trace` (no row for an empty field)."""
    selected = frame.selected_action
    if selected is None or not selected.open_loop_id:
        return None
    loop = next((l for l in frame.open_loops if l.id == selected.open_loop_id), None)
    if loop is None:
        return None
    correlation_id = frame.correlation_id or ""
    return {
        # Includes turn_id alongside correlation_id, not just [correlation_id,
        # loop.id] the way reverie's build_salience_trace hashes (2026-08-19
        # review finding): unlike reverie's tick, a chat turn's correlation_id
        # can genuinely be None (orion/substrate/attention_frame.py's own
        # `str(ctx.get("correlation_id") or ctx.get("trace_id") or "") or
        # None` fallback). stable_hash_id() drops empty parts before hashing,
        # so with correlation_id="" the trace_id would collapse to a hash of
        # loop.id alone -- ON CONFLICT (trace_id) DO NOTHING would then
        # silently swallow every subsequent turn that selects the same loop
        # under that condition, not just true retries of the same turn.
        # turn_id is a second, independently-sourced ctx field (falls back to
        # ctx["message_id"]) -- both would have to be simultaneously absent
        # for two distinct real turns to still collide.
        "trace_id": stable_hash_id("saltrace", [correlation_id, frame.turn_id or "", loop.id]),
        "loop_id": loop.id,
        "theme_key": loop.id,
        "description": (loop.description or "")[:200],
        "why_it_matters": (loop.why_it_matters or "")[:500],
        "target_type": str(loop.target_type or "other"),
        "correlation_id": frame.correlation_id,
        "salience": _bounded(loop.salience),
        "weights_version": _WEIGHTS_VERSION,
        "scope": "chat",
        "features": dict(loop.salience_features or {}),
        "created_at": frame.generated_at,
    }


def _persist_sync(row: dict[str, Any]) -> bool:
    engine = _get_engine()
    if engine is None:
        raise RuntimeError("chat_attention_salience_trace_dsn_unset")
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO attention_salience_trace
                    (trace_id, loop_id, theme_key, description, why_it_matters, target_type,
                     correlation_id, salience, weights_version, scope, features, created_at)
                VALUES
                    (:trace_id, :loop_id, :theme_key, :description, :why_it_matters, :target_type,
                     :correlation_id, :salience, :weights_version, :scope, CAST(:features AS jsonb), :created_at)
                ON CONFLICT (trace_id) DO NOTHING
                """
            ),
            {**row, "features": json.dumps(row["features"])},
        )
    return True


async def persist_chat_attention_salience_trace(frame: AttentionFrameV1) -> bool:
    """Bounded fail-open persist. Never raises to the caller -- a DB miss or
    slow write must never delay or break a real chat turn."""
    if not _flag_enabled():
        return False
    row = build_chat_salience_trace_row(frame)
    if row is None:
        return False
    if not _dsn():
        logger.debug(
            "chat_attention_salience_trace_dsn_unset correlation_id=%s", frame.correlation_id
        )
        return False

    timeout_sec = _timeout_sec()
    t0 = time.perf_counter()
    try:
        return await asyncio.wait_for(asyncio.to_thread(_persist_sync, row), timeout=timeout_sec)
    except asyncio.TimeoutError:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        logger.warning(
            "chat_attention_salience_trace_timeout correlation_id=%s elapsed_ms=%s timeout_sec=%s",
            frame.correlation_id,
            elapsed_ms,
            timeout_sec,
        )
        return False
    except Exception as exc:  # noqa: BLE001 -- fail-open by contract
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        logger.warning(
            "chat_attention_salience_trace_persist_failed correlation_id=%s elapsed_ms=%s exc_type=%s err=%s",
            frame.correlation_id,
            elapsed_ms,
            type(exc).__name__,
            exc,
        )
        return False


def reset_chat_attention_salience_trace_engine_for_tests() -> None:
    global _ENGINE, _ENGINE_URL
    _ENGINE = None
    _ENGINE_URL = None
