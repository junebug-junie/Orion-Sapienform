"""Reverie thought persistence (Phase A store).

Best-effort writer for `SpontaneousThoughtV1` into `substrate_reverie_thought`
(migration `manual_migration_substrate_reverie_thought.sql`). Backs the hub
`_reverie_section` panel. Uses a direct sqlalchemy DSN (see `_database_url`) —
never the heavy `orion.substrate` package this thin service does not ship.

Discipline: persistence is best-effort. A DB failure degrades to a logged miss
(returns False) and never breaks the reverie tick. Idempotent on `thought_id`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

logger = logging.getLogger("orion-thought.store")

if TYPE_CHECKING:
    from orion.schemas.reverie import SpontaneousThoughtV1

_engine = None
# Guards the check-then-create-then-assign below. Without it, two threads
# racing through _get_engine() at nearly the same moment (e.g. a real chat
# turn's drive-state fetch landing right as startup's warm_pool() runs) can
# both observe `_engine is None` and each construct a separate Engine/pool;
# whichever assignment lands last silently discards the other's pool without
# disposing it. Found live 2026-07-17 while adding warm_pool() below, which
# is the first caller that makes this race reachable in practice.
_engine_lock = threading.Lock()


def _database_url() -> str:
    # Direct DSN — deliberately NOT via orion.substrate.felt_state_reader, whose
    # package __init__ drags the full graph engine (requests etc.) this thin
    # service does not ship. Writes land where the hub panel reads (conjourney).
    return (
        os.getenv("POSTGRES_URI", "").strip()
        or "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney"
    )


def _get_engine():
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:  # re-check: another thread may have won the race
                from sqlalchemy import create_engine

                _engine = create_engine(_database_url(), pool_pre_ping=True)
    return _engine


def _warm_pool_sync() -> None:
    """Open one throwaway connection so the shared pool isn't cold on first real use.

    Live-verified 2026-07-17: the first query against a freshly-created engine
    pays a full TCP+auth handshake to Postgres (~400ms). That's cheap for
    reverie/salience/etc.'s best-effort writes, but tripped a caller with a
    tight 400ms budget (formerly orion-thought's drive_state_compact facet
    fetch, removed 2026-07-30 -- see mind_enrichment.py) on turn one of every
    fresh container start. Warming here benefits every caller of
    `_get_engine()`, not just that one, so it's unconditional at startup
    rather than gated behind any one feature flag. Never raises — a DB that
    isn't reachable yet at boot just means the first real caller pays the
    cold cost as before (today's status quo for every other consumer of
    this module), not a startup failure.
    """
    try:
        from sqlalchemy import text

        with _get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001 — best-effort warm-up, never fail boot
        logger.warning("pool_warmup_failed err=%s", exc)


async def warm_pool() -> None:
    """Bounded async wrapper for `_warm_pool_sync`, called once at startup.

    Catches both the timeout and any exception the sync side didn't already
    swallow -- the boot sequence must never fail because a best-effort pool
    warm-up couldn't connect, so this is defense-in-depth on top of
    `_warm_pool_sync`'s own internal try/except, not a substitute for it.
    """
    try:
        await asyncio.wait_for(asyncio.to_thread(_warm_pool_sync), timeout=5.0)
    except asyncio.TimeoutError:
        logger.warning("pool_warmup_timeout")
    except Exception as exc:  # noqa: BLE001 — best-effort warm-up, never fail boot
        logger.warning("pool_warmup_wrapper_failed err=%s", exc)


def persist_reverie_thought(thought: "SpontaneousThoughtV1") -> bool:
    """Insert one spontaneous thought. Returns True on write, False on any miss.

    Never raises — a persistence failure must not break the tick.

    `expectation`/`expectation_checkable_by` are written as real columns (not
    just inside `thought_json`) so `load_pending_expectations` below can scan
    them directly. `expectation_verdict`/`expectation_scored_at` are always
    NULL at insert time -- they are only ever written later, in place, by
    `persist_expectation_verdict`.
    """
    try:
        from sqlalchemy import text

        engine = _get_engine()
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO substrate_reverie_thought
                        (thought_id, correlation_id, created_at, salience,
                         interpretation, thought_json, expectation, expectation_checkable_by)
                    VALUES
                        (:thought_id, :correlation_id, :created_at, :salience,
                         :interpretation, CAST(:thought_json AS jsonb),
                         :expectation, :expectation_checkable_by)
                    ON CONFLICT (thought_id) DO NOTHING
                    """
                ),
                {
                    "thought_id": thought.thought_id,
                    "correlation_id": thought.correlation_id,
                    "created_at": thought.created_at,
                    "salience": float(thought.salience),
                    "interpretation": thought.interpretation,
                    "thought_json": json.dumps(thought.model_dump(mode="json")),
                    "expectation": thought.expectation,
                    "expectation_checkable_by": thought.expectation_checkable_by,
                },
            )
        return True
    except Exception as exc:
        logger.warning("reverie thought persist failed id=%s err=%s", thought.thought_id, exc)
        return False


# --- Movement III: expectation scoring (default-off; ORION_REVERIE_EXPECTATION_
# SCORING_ENABLED) -----------------------------------------------------------

# Matches vision_reader.py's bound: a slow database must degrade to "no pending
# expectations" rather than stall the reverie tick's shared event loop. A
# dedicated engine, not the shared `_get_engine()` write pool above --
# vision_reader.py's own docstring explains why this module doesn't reuse a
# single lazily-built engine across readers with different timeout
# requirements: statement_timeout only takes effect on first-use engine
# construction for a given DSN, so sharing one would silently let whichever
# caller constructs it first decide the GUC for both.
_EXPECTATION_QUERY_STATEMENT_TIMEOUT_MS = 1500
_expectation_read_engine = None
_expectation_read_engine_url: str | None = None


def _get_expectation_read_engine():
    global _expectation_read_engine, _expectation_read_engine_url
    url = _database_url()
    if _expectation_read_engine is None or _expectation_read_engine_url != url:
        from sqlalchemy import create_engine

        _expectation_read_engine = create_engine(
            url,
            pool_pre_ping=True,
            connect_args={
                "options": f"-c statement_timeout={_EXPECTATION_QUERY_STATEMENT_TIMEOUT_MS}"
            },
        )
        _expectation_read_engine_url = url
    return _expectation_read_engine


def load_pending_expectations(limit: int = 1) -> list[dict[str, Any]]:
    """Most-overdue-first thoughts with an open, closed-window expectation.

    Bounded (default 1 -- score at most one per tick, same boundedness
    discipline as `MAX_FINALIZE_LOOP_RETRIES` in `orion/harness/finalize.py`).
    Fail-open: [] on any error or non-positive limit, so a lookup failure never
    breaks a reverie tick. All filtering (non-null expectation, unresolved
    verdict, window closed) lives in the SQL WHERE clause before LIMIT/ORDER BY
    -- nothing is re-filtered in Python after the row set comes back.

    Returns [{"thought_id": str, "expectation": str, "expectation_checkable_by":
    datetime}], ordered `expectation_checkable_by ASC` -- the row whose window
    closed longest ago (most overdue) sorts first.
    """
    limit = max(0, int(limit))
    if limit == 0:
        return []
    try:
        from sqlalchemy import text

        engine = _get_expectation_read_engine()
        with engine.connect() as conn:
            rows = (
                conn.execute(
                    text(
                        "SELECT thought_id, expectation, expectation_checkable_by "
                        "FROM substrate_reverie_thought "
                        "WHERE expectation IS NOT NULL "
                        "AND expectation_verdict IS NULL "
                        "AND expectation_checkable_by IS NOT NULL "
                        "AND expectation_checkable_by <= now() "
                        "ORDER BY expectation_checkable_by ASC "
                        "LIMIT :limit"
                    ),
                    {"limit": limit},
                )
                .mappings()
                .all()
            )
        return [dict(r) for r in rows]
    except Exception as exc:
        logger.debug("pending expectation load failed: %s", exc)
        return []


def persist_expectation_verdict(thought_id: str, verdict: str, scored_at: datetime) -> None:
    """Update-in-place: stamp one reverie thought's scoring verdict.

    Fail-open — logs and swallows any failure, never raises into the reverie
    tick. No-op (but never raises) on an empty thought_id.
    """
    if not thought_id:
        return
    try:
        from sqlalchemy import text

        engine = _get_engine()
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE substrate_reverie_thought
                    SET expectation_verdict = :verdict,
                        expectation_scored_at = :scored_at
                    WHERE thought_id = :thought_id
                    """
                ),
                {"thought_id": thought_id, "verdict": verdict, "scored_at": scored_at},
            )
    except Exception as exc:
        logger.warning(
            "expectation verdict persist failed id=%s verdict=%s err=%s",
            thought_id,
            verdict,
            exc,
        )


def persist_salience_trace(trace) -> bool:
    """Persist one salience trace row. Never raises; idempotent on trace_id.

    Requires services/orion-sql-db/manual_migration_attention_salience_trace.sql
    applied (adds why_it_matters/target_type, 2026-08-21) -- if this service is
    redeployed before the migration runs, EVERY insert here raises "column ...
    does not exist" and the except-block below swallows it as a WARNING,
    silently dropping ALL reverie-scope attention_salience_trace writes (not
    just the two new columns) until someone reads the logs and runs the
    migration. Apply the migration BEFORE deploying this file's changes.
    """
    try:
        from sqlalchemy import text

        engine = _get_engine()
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
                {
                    "trace_id": trace.trace_id,
                    "loop_id": trace.loop_id,
                    "theme_key": trace.theme_key,
                    "description": trace.description,
                    "why_it_matters": getattr(trace, "why_it_matters", "") or "",
                    "target_type": getattr(trace, "target_type", "other") or "other",
                    "correlation_id": trace.correlation_id,
                    "salience": float(trace.salience),
                    "weights_version": trace.weights_version,
                    "scope": trace.scope,
                    "features": json.dumps(trace.features),
                    "created_at": trace.created_at,
                },
            )
        return True
    except Exception as exc:
        logger.warning("salience trace persist failed id=%s err=%s", trace.trace_id, exc)
        return False


def persist_reverie_chain(chain) -> bool:
    """Insert one reverie chain readout. Never raises; idempotent on chain_id."""
    try:
        from sqlalchemy import text

        engine = _get_engine()
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO substrate_reverie_chain
                        (chain_id, created_at, theme_key, terminal_reason,
                         ema_salience, committed_proposal_id, chain_json)
                    VALUES
                        (:chain_id, :created_at, :theme_key, :terminal_reason,
                         :ema_salience, :committed_proposal_id, CAST(:chain_json AS jsonb))
                    ON CONFLICT (chain_id) DO NOTHING
                    """
                ),
                {
                    "chain_id": chain.chain_id,
                    "created_at": chain.created_at,
                    "theme_key": chain.theme_key,
                    "terminal_reason": chain.terminal_reason,
                    "ema_salience": float(chain.ema_salience),
                    "committed_proposal_id": chain.committed_proposal_id,
                    "chain_json": json.dumps(chain.model_dump(mode="json")),
                },
            )
        return True
    except Exception as exc:
        logger.warning("reverie chain persist failed id=%s err=%s", chain.chain_id, exc)
        return False


# --- Reverie VISUAL chain (Patch 2 of docs/superpowers/specs/2026-08-20-
# reverie-visual-chain-design.md) -- separate table, same engine, same
# best-effort-never-raises discipline as everything else in this module. See
# visual_chain.py for why the chain row must be inserted before the artifact
# row (reverie_visual_artifact.chain_id is a real FK).


def persist_reverie_visual_chain(chain) -> bool:
    """Insert one visual-chain readout. Never raises; idempotent on chain_id.

    Unlike `persist_reverie_chain` above, `ReverieVisualChainV1` has its OWN
    `chain_json: dict` field (the run's small prompt/artifact/description
    side-channel -- see `orion.schemas.reverie_visual`'s docstring) -- write
    THAT into the `chain_json` column, not the full model dump. The text
    chain's model has no such field of its own, so `model_dump()` is correct
    there; copying that pattern here would self-nest the real data one level
    deeper than every reader expects (review finding, caught before ship).
    """
    try:
        from sqlalchemy import text

        engine = _get_engine()
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO reverie_visual_chain
                        (chain_id, created_at, theme_key, terminal_reason,
                         ema_salience, prior_description, chain_json)
                    VALUES
                        (:chain_id, :created_at, :theme_key, :terminal_reason,
                         :ema_salience, :prior_description, CAST(:chain_json AS jsonb))
                    ON CONFLICT (chain_id) DO NOTHING
                    """
                ),
                {
                    "chain_id": chain.chain_id,
                    "created_at": chain.created_at,
                    "theme_key": chain.theme_key,
                    "terminal_reason": chain.terminal_reason,
                    "ema_salience": float(chain.ema_salience),
                    "prior_description": chain.prior_description,
                    "chain_json": json.dumps(chain.chain_json),
                },
            )
        return True
    except Exception as exc:
        logger.warning("visual chain persist failed id=%s err=%s", chain.chain_id, exc)
        return False


def persist_reverie_visual_artifact(artifact) -> bool:
    """Insert one generated-image pointer row. Never raises; idempotent on sha256.

    Requires the referenced chain_id to already exist (FK) -- callers must
    persist the chain row first.
    """
    try:
        from sqlalchemy import text

        engine = _get_engine()
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO reverie_visual_artifact
                        (sha256, chain_id, step_index, mime, bytes, width, height,
                         path, description, created_at)
                    VALUES
                        (:sha256, :chain_id, :step_index, :mime, :bytes, :width, :height,
                         :path, :description, :created_at)
                    ON CONFLICT (sha256) DO NOTHING
                    """
                ),
                {
                    "sha256": artifact.sha256,
                    "chain_id": artifact.chain_id,
                    "step_index": int(artifact.step_index),
                    "mime": artifact.mime,
                    "bytes": int(artifact.bytes),
                    "width": artifact.width,
                    "height": artifact.height,
                    "path": artifact.path,
                    "description": artifact.description,
                    "created_at": artifact.created_at,
                },
            )
        return True
    except Exception as exc:
        logger.warning("visual artifact persist failed sha256=%s err=%s", artifact.sha256, exc)
        return False


def load_latest_visual_chain_continuity_state() -> tuple[str | None, int]:
    """Most recent visual-chain row's `prior_description` AND
    `continuity_streak`, in ONE round trip (review finding: two separate
    single-column SELECTs against the same latest row wasted a full
    connect+query cycle every tick, and left a theoretical window where the
    two reads could observe different rows if a write ever raced between
    them -- currently prevented only by the single-flight/sequential-worker
    guarantee documented in visual_chain.py's module docstring, not
    something this query should have to rely on to be correct).

    Returns `(prior_description, continuity_streak)`:
      - `prior_description`: the continuity input the next run's prompt is
        built from (design doc §2/§5).
      - `continuity_streak`: how many CONSECUTIVE recent runs used real
        continuity in their prompt -- reads `chain_json.continuity_streak`,
        a small int this service itself writes on every run (design doc
        §15, `visual_chain.py::resolve_visual_chain_continuity`). NOT
        derived from `prior_description`'s own nullness, which reflects
        "did this run get a real caption" and would keep climbing even
        through a run whose PROMPT was forcibly reset. Missing/unparsable
        on an older pre-Patch-4 row degrades to 0 -- the honest "no streak
        recorded yet" answer, same direction a missing counter should fail
        in (under-count, never over-count, so a bad read causes an extra
        continuity run at worst, never gets stuck skipping resets forever).

    Read-only, best-effort: `(None, 0)` on any error or empty table, so a
    lookup failure degrades to "no continuity yet, nothing to cap" (the
    same prompt a first-ever run uses) rather than breaking the tick.
    """
    try:
        from sqlalchemy import text

        engine = _get_engine()
        with engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        "SELECT prior_description, chain_json FROM reverie_visual_chain "
                        "ORDER BY created_at DESC LIMIT 1"
                    )
                )
                .mappings()
                .first()
            )
        if not row:
            return None, 0
        value = row.get("prior_description")
        prior = str(value).strip() or None if value else None
        cj = row.get("chain_json")
        streak = 0
        if isinstance(cj, dict):
            try:
                streak = max(0, int(cj.get("continuity_streak") or 0))
            except (TypeError, ValueError):
                streak = 0
        return prior, streak
    except Exception as exc:
        logger.debug("visual chain continuity state load failed: %s", exc)
        return None, 0


# Cap on the interpretation text handed into a diffusion prompt (§ cap-all-
# collections) -- SpontaneousThoughtV1.interpretation has no length bound of
# its own (it's free LLM narration), but the diffusion model only needs a
# short scene description, not the full text. THIS is the one source of
# truth -- settings.py's reverie_context_char_limit Field imports this
# constant as its default (one-directional import, settings -> store; this
# module still never imports settings, per its own header docstring) rather
# than hardcoding a second, driftable 240.
MAX_REVERIE_CONTEXT_CHARS = 240

# How many recent chain-linked candidates to pull before Python-side
# hollow re-validation (below) picks the first real one. Not 1: the SQL
# WHERE clause only filters on interpretation<>'' and chain-linkage -- the
# hollow re-check happens after fetch (review finding), so a small batch is
# needed in case the single most-recent candidate turns out stale-hollow.
_REVERIE_CONTEXT_CANDIDATE_LIMIT = 10


def load_latest_reverie_interpretation(
    *, char_limit: int | None = None, max_age_sec: float | None = None
) -> str | None:
    """Most recent real (non-hollow, non-empty) text-chain thought's
    interpretation, already linked into a SETTLED chain -- the visual
    chain's context-seed (design doc §8: "which specific recent-activity/
    chat/dream sources feed a step", Patch 3).

    `char_limit`: overrides MAX_REVERIE_CONTEXT_CHARS when given (visual_chain
    passes settings.reverie_context_char_limit, env-configurable like every
    other tunable in that file -- this was a bare module constant until now).

    `max_age_sec`: if given, only candidates within this age are considered
    (added review finding, post-Patch-3): without it, a stalled or disabled
    text-reverie worker (chain.py) leaves the same old thought answering
    every future call, and it keeps being woven into the visual prompt and
    shown in the cockpit as "Orion is currently thinking" long after it
    stopped being current -- the fetch LIMIT above bounds candidate COUNT,
    not age, so this is a real gap the count bound alone doesn't close. This
    only bounds THIS function's callers (currently just visual_chain.py) --
    it does NOT bound the Hub's separate Text sub-view (`reverie_routes.py`
    ::`text_recent`), which has no staleness filter of its own; nor does it
    give any consumer a way to tell "producer genuinely quiet" from
    "producer stalled/dead" -- both look identical here as "no fresh row",
    since there is no liveness/heartbeat signal from chain.py itself to
    check instead (a real gap, not fixed by this patch).

    Both `char_limit`/`max_age_sec` default to None, preserving the exact
    prior unbounded/uncapped behavior for any OTHER caller of this function
    that doesn't pass them (review finding, deliberate choice not an
    oversight: this docstring already names `_project_reverie_glimpse` as
    a consumer of this same underlying table via a different code path
    (`felt_state_reader.py`'s own reader, not this function) -- changing
    this function's own default behavior for hypothetical future callers
    of THIS specific function, rather than leaving it opt-in, is a bigger
    change than this patch's scope; every current, real caller (just
    visual_chain.py) does pass both).

    Deliberately the text chain's own narration, not raw chat: it is already
    the summary layer the coalition-grounding + hollow guard
    (orion/schemas/reverie.py's `SpontaneousThoughtV1.is_hollow`) produce
    before a row is ever written. The privacy claim this rests on --
    "already reaches the same Hub Reverie tab" (`reverie_routes.py`'s
    `text_recent`) -- only holds for a thought that is actually reachable
    there, which requires its ENCLOSING CHAIN to have settled and persisted
    (`chain.py`'s `persist_reverie_chain`, called once at chain end, not per
    thought). `substrate_reverie_thought` rows are written immediately on
    generation (`reverie.py::run_reverie_once`), well before that -- and if
    `ORION_REVERIE_CHAIN_ENABLED=false` while `ORION_REVERIE_ENABLED=true`
    (independent settings.py flags), a thought's chain never settles at all.
    Reading the thought table directly would source content `text_recent`
    might never surface, contradicting the "no new privacy surface" claim
    (review finding). The `EXISTS` clause below closes that gap: a thought
    only qualifies once some settled `substrate_reverie_chain` row's
    `chain_json.thought_ids` already lists it -- the exact set `text_recent`
    can already show, no earlier and no wider.

    Widening the source set to raw chat/dream content is a separate, later
    change that must redo this same privacy check, not something this
    function does.

    Hollow re-validation happens in Python, not SQL (review finding): a raw
    `thought_json->>'hollow'` cast trusts a flag stamped at write time, which
    `services/orion-cortex-exec/app/chat_stance.py::_project_reverie_glimpse`
    -- the other real consumer of this same table -- explicitly does NOT do,
    re-deriving via `SpontaneousThoughtV1.is_hollow()` because a stored flag
    can go stale if the hollow-guard logic changes after the row was
    written. Same discipline here: gate on BOTH the stored flag and a fresh
    `is_hollow()` re-check.

    Read-only, best-effort: None on any error, empty table, a table with no
    chain-linked/non-hollow rows, or a row that no longer validates as
    `SpontaneousThoughtV1` -- degrades to the fixed seed prompt exactly like
    an absent prior_description does (visual_chain.build_visual_prompt).
    """
    try:
        from sqlalchemy import text

        from orion.cognition.compactor.truncate import truncate_at_word_boundary
        from orion.schemas.reverie import SpontaneousThoughtV1

        where_sql = (
            "t.interpretation <> '' "
            "AND EXISTS ("
            "  SELECT 1 FROM substrate_reverie_chain c "
            "  WHERE c.chain_json -> 'thought_ids' ? t.thought_id"
            ")"
        )
        params: dict[str, Any] = {"limit": _REVERIE_CONTEXT_CANDIDATE_LIMIT}
        if max_age_sec is not None:
            # `now() - make_interval(secs => :x)` freshness idiom (review
            # finding: at least 6 other repo call sites already hand-roll
            # this same shape -- vision_reader.py, orion-hub's
            # curiosity_hint.py/chat_history_rehydrate.py/
            # tension_outreach_trigger.py, orion-sql-writer's
            # vision_object_permanence.py, orion-field-digester's store.py).
            # Not extracted into a shared helper here: doing that well means
            # touching multiple services' modules, a bigger change than this
            # small additive patch's scope -- a real repo-wide cleanup
            # candidate, not something to half-do as a side effect of one
            # new call site.
            where_sql += " AND t.created_at > now() - make_interval(secs => :max_age_sec)"
            params["max_age_sec"] = float(max_age_sec)

        engine = _get_engine()
        with engine.connect() as conn:
            rows = (
                conn.execute(
                    text(
                        "SELECT thought_json FROM substrate_reverie_thought t "
                        f"WHERE {where_sql} "
                        "ORDER BY t.created_at DESC LIMIT :limit"
                    ),
                    params,
                )
                .mappings()
                .all()
            )
        for row in rows:
            payload = row.get("thought_json")
            if not isinstance(payload, dict):
                continue
            try:
                thought = SpontaneousThoughtV1.model_validate(payload)
            except Exception:
                continue
            if thought.hollow or thought.is_hollow():
                continue
            value = thought.interpretation.strip()
            if value:
                # Word-boundary truncation (review finding), not a raw slice
                # -- same helper chat_history_compactor/github_compactor use
                # for the identical "cap free-form narration for a
                # downstream reader" problem, so a 240-char cut degrades to
                # a coherent prefix instead of a mid-word fragment baked
                # into the diffusion prompt and rendered verbatim in the Hub
                # Reverie tab.
                limit_chars = MAX_REVERIE_CONTEXT_CHARS if char_limit is None else char_limit
                trimmed, _truncated = truncate_at_word_boundary(value, limit_chars)
                return trimmed
        return None
    except Exception as exc:
        logger.debug("reverie context-seed load failed: %s", exc)
        return None


def persist_compaction_request(request) -> bool:
    """Enqueue one compaction request (Phase E). Never raises; idempotent."""
    try:
        from sqlalchemy import text

        engine = _get_engine()
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO dream_compaction_request_queue
                        (request_id, theme, op_hint, reason, origin_chain_id,
                         created_at, request_json)
                    VALUES
                        (:request_id, :theme, :op_hint, :reason, :origin_chain_id,
                         :created_at, CAST(:request_json AS jsonb))
                    ON CONFLICT (request_id) DO NOTHING
                    """
                ),
                {
                    "request_id": request.request_id,
                    "theme": request.theme,
                    "op_hint": request.op_hint,
                    "reason": request.reason,
                    "origin_chain_id": request.origin_chain_id,
                    "created_at": request.created_at,
                    "request_json": json.dumps(request.model_dump(mode="json")),
                },
            )
        return True
    except Exception as exc:
        logger.warning("compaction request persist failed id=%s err=%s", request.request_id, exc)
        return False


def load_recent_chain_theme_events(limit: int) -> list[tuple[str, object]]:
    """Recent (theme_key, created_at) chain rows for the resonance detector.

    Read-only, best-effort — returns [] on any miss so the tripwire degrades to
    "no evidence" rather than raising. Skips null/unknown themes."""
    limit = max(0, int(limit))
    if limit == 0:
        return []
    try:
        from sqlalchemy import text

        engine = _get_engine()
        with engine.connect() as conn:
            rows = (
                conn.execute(
                    text(
                        "SELECT theme_key, created_at FROM substrate_reverie_chain "
                        "WHERE theme_key IS NOT NULL AND theme_key <> 'unknown' "
                        "ORDER BY created_at DESC LIMIT :limit"
                    ),
                    {"limit": limit},
                )
                .mappings()
                .all()
            )
        return [(str(r["theme_key"]), r["created_at"]) for r in rows if r.get("created_at")]
    except Exception as exc:
        logger.debug("resonance chain-event load failed: %s", exc)
        return []


def persist_resonance_alert(alert) -> bool:
    """Persist one resonance alert. Never raises; idempotent on alert_id."""
    try:
        from sqlalchemy import text

        engine = _get_engine()
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO substrate_reverie_resonance_alert
                        (alert_id, theme_key, violation_count, refractory_sec,
                         min_gap_sec, occurrences, created_at, alert_json)
                    VALUES
                        (:alert_id, :theme_key, :violation_count, :refractory_sec,
                         :min_gap_sec, :occurrences, :created_at, CAST(:alert_json AS jsonb))
                    ON CONFLICT (alert_id) DO NOTHING
                    """
                ),
                {
                    "alert_id": alert.alert_id,
                    "theme_key": alert.theme_key,
                    "violation_count": int(alert.violation_count),
                    "refractory_sec": float(alert.refractory_sec),
                    "min_gap_sec": float(alert.min_gap_sec),
                    "occurrences": int(alert.occurrences),
                    "created_at": alert.created_at,
                    "alert_json": json.dumps(alert.model_dump(mode="json")),
                },
            )
        return True
    except Exception as exc:
        logger.warning("resonance alert persist failed id=%s err=%s", alert.alert_id, exc)
        return False


def load_recent_resonance_alerts(theme_key: str, limit: int = 2) -> list[dict]:
    """Most recent persisted resonance alerts for one theme, newest first.

    Read-only, best-effort — returns [] on any miss. Used by the health monitor
    to compare violation_count across the last 2 samples (is this theme's
    resonance getting worse, or is a new alert just re-reporting the same
    historical burst still inside the detector's lookback window?).
    """
    limit = max(0, int(limit))
    if limit == 0 or not theme_key:
        return []
    try:
        from sqlalchemy import text

        engine = _get_engine()
        with engine.connect() as conn:
            rows = (
                conn.execute(
                    text(
                        "SELECT theme_key, violation_count, refractory_sec, min_gap_sec, "
                        "occurrences, created_at FROM substrate_reverie_resonance_alert "
                        "WHERE theme_key = :theme_key "
                        "ORDER BY created_at DESC LIMIT :limit"
                    ),
                    {"theme_key": theme_key, "limit": limit},
                )
                .mappings()
                .all()
            )
        return [dict(r) for r in rows]
    except Exception as exc:
        logger.debug("resonance alert history load failed theme=%s err=%s", theme_key, exc)
        return []


def reverie_refractory_is_suppressed(theme_key: str, now) -> bool:
    """True if the theme is currently suppressed. Best-effort (False on error)."""
    try:
        from sqlalchemy import text

        engine = _get_engine()
        with engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        "SELECT suppressed_until FROM substrate_reverie_refractory "
                        "WHERE theme_key = :k"
                    ),
                    {"k": theme_key},
                )
                .mappings()
                .first()
            )
        if not row:
            return False
        until = row.get("suppressed_until")
        return until is not None and until > now
    except Exception:
        return False


def load_recent_loop_outcomes(loop_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Most recent `attention_loop_outcome` verdict per loop_id (orion-hub table,
    written by a human's Resolve/Dismiss action; read directly, no orion-hub import).

    Read-only, best-effort — returns {} on any miss or empty input, so a lookup
    failure never breaks a reverie tick. Keyed by the *bare* loop id, matching
    `attention_loops_store.suppress_loop`'s write format (see `chain.theme_key_for`
    for the sibling refractory-key fix this mirrors).

    Returns {loop_id: {"verdict": str, "note": str, "age_days": int}} — age is
    computed here (deterministic code), not left for the prompt/LLM to infer from
    a raw timestamp. `age_days` is omitted (never `None`) when `created_at` can't
    be read, so the prompt never has to render a null age.
    """
    ids = [str(i) for i in (loop_ids or []) if i]
    if not ids:
        return {}
    try:
        from sqlalchemy import bindparam, text

        engine = _get_engine()
        stmt = text(
            """
            SELECT DISTINCT ON (loop_id) loop_id, verdict, note, created_at
            FROM attention_loop_outcome
            WHERE loop_id IN :ids
            ORDER BY loop_id, created_at DESC
            """
        ).bindparams(bindparam("ids", expanding=True))
        with engine.connect() as conn:
            rows = conn.execute(stmt, {"ids": ids}).mappings().all()

        now = datetime.now(timezone.utc)
        out: dict[str, dict[str, Any]] = {}
        for r in rows:
            entry: dict[str, Any] = {
                "verdict": str(r.get("verdict") or ""),
                "note": str(r.get("note") or ""),
            }
            created = r.get("created_at")
            if isinstance(created, datetime):
                c = created if created.tzinfo else created.replace(tzinfo=timezone.utc)
                entry["age_days"] = max(0, int((now - c).total_seconds() // 86400))
            out[str(r["loop_id"])] = entry
        return out
    except Exception as exc:
        logger.debug("loop outcome load failed ids=%s err=%s", ids, exc)
        return {}


def reverie_refractory_suppress(theme_key: str, until) -> bool:
    """Upsert a refractory suppression window. Never raises."""
    try:
        from sqlalchemy import text

        engine = _get_engine()
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO substrate_reverie_refractory (theme_key, suppressed_until)
                    VALUES (:k, :until)
                    ON CONFLICT (theme_key)
                    DO UPDATE SET suppressed_until = EXCLUDED.suppressed_until,
                                  updated_at = now()
                    """
                ),
                {"k": theme_key, "until": until},
            )
        return True
    except Exception as exc:
        logger.warning("reverie refractory suppress failed theme=%s err=%s", theme_key, exc)
        return False
