"""Act ledger: persist act_opened / act_closed grammar events and grade closed acts.

Rung 2 of docs/superpowers/specs/2026-10-14-orion-thinking-architecture-design.md
("doing on one track"). Every act writes an efference copy (ActV1): what it
expects, opened BEFORE it runs; what happened, closed by act_id AFTER.

This module is the writer side, called by orion-sql-writer:

- ``apply_act_events``: persist opens/closes from the grammar stream into
  ``grammar_acts``. On open, the expectation's magnitude (``predicted``) is
  stamped from the current EffectPosterior, never typed by a producer.
- ``close_act``: one close. The first close wins (a late producer close after
  the horizon resolver already closed the act is ignored). A close is graded
  through the EXISTING L10 machinery -- one ``substrate_action_outcomes`` row
  plus the ``substrate_action_effect_posterior`` upsert, with the same two
  double-count guards orion-feedback-runtime uses -- and no new learner.
- ``resolve_due_acts``: the horizon resolver. ``juniper_replied`` acts are
  closed from the start time of Juniper's next turn (1 as soon as she starts a
  hub chat turn after the act -- even one that later errors -- 0 once the
  horizon passes without one; see _JUNIPER_REPLY_SQL). Producer-reported
  signals (``artifact_produced``) that pass their horizon unclosed are closed
  at 0 as ``expired``: the signal is "produced BY the horizon", so a missing
  close past it is a real 0, not a guess.

Two notions of "how wrong" live on a closed act, on purpose: ``error`` is the
outcome minus the ``predicted`` stamped at OPEN (what the act expected when it
started), while ``surprise_nats`` and the posterior update use the belief current
at CLOSE (the conjugate update must start from the latest belief). For long
horizons (outreach, 10 h) the two beliefs can differ.

All SQL is plain ``text()`` so the module runs on any SQLAlchemy session or
connection; the caller owns the transaction.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import text

from orion.autonomy.prediction import (
    EffectPosterior,
    act_dispatch_kind,
    cold_act_posterior,
    score_act_outcome,
)
from orion.regulation.juniper_turns import JUNIPER_INGRESS_PREDICATE, JUNIPER_TURN_PREDICATE
from orion.schemas.grammar import ActExpectV1, ActOutcomeV1, ActV1, GrammarEventV1

logger = logging.getLogger("orion.grammar.act_ledger")

ACT_EVENT_KINDS = frozenset({"act_opened", "act_closed"})

# How long past its horizon a producer-reported act may still be closed by its
# producer before the resolver records it as not produced. Covers bus + writer
# lag only (grammar batches flush in seconds); it is not a second horizon.
EXPIRY_GRACE_SEC = 120.0


# --- rung 3: bind an act to the workspace winner it answers, and close that loop ----------
#
# Thinking architecture rung 3 ("bind the winner to action"). An act whose actor answers a
# world-first workspace source is stamped, at open, with the winning loop it was opened under
# (``workspace_ref`` = the broadcast's selected open_loop_id), using the ONE binding rule the
# proposal layer already uses (orion.autonomy.self_shed.bind_workspace_winner: an action was
# selected, the winner is <= 90 s old and held >= 2 ticks, and the attended ref is one this actor
# answers). When the act closes, Orion writes its own non-final ``acted`` verdict on that loop in
# ``attention_loop_outcome`` (actor='orion') with the act's outcome in features_at_close. Never
# ``resolved``/``dismissed``: those are terminal and would let acting silence Orion's own
# attention (orion/substrate/attention/verdicts.py TERMINAL_VERDICTS is unchanged).
#
# Which actors answer which world source. One row, because it is the only pairing joined in the
# data rather than asserted: Juniper's chat activity (``world:chat``, and its substrate node) wins
# the workspace when she is busier than her own week, and Orion's reply is the act that answers it
# (live 2026-10-21: 11 of 20 hub.chat_reply acts opened while that winner was bindable).
ACT_ANSWERS_WORKSPACE_REFS: dict[str, tuple[str, ...]] = {
    "hub.chat_reply": ("world:chat", "node:substrate.chat"),
}

_WORKSPACE_BINDING_ENABLED = True


def configure_act_workspace_binding(enabled: bool) -> None:
    """SQL_WRITER_ACT_WORKSPACE_BINDING_ENABLED. Off: no stamping, no Orion loop verdicts."""
    global _WORKSPACE_BINDING_ENABLED
    _WORKSPACE_BINDING_ENABLED = bool(enabled)


def act_workspace_binding_enabled() -> bool:
    return _WORKSPACE_BINDING_ENABLED


_BROADCAST_AT_SQL = text(
    """
    SELECT log_id, generated_at,
           projection_json->>'selected_action_type' AS selected_action_type,
           projection_json->>'selected_open_loop_id' AS selected_open_loop_id,
           projection_json->'attended_node_ids' AS attended_node_ids,
           projection_json->>'dwell_ticks' AS dwell_ticks
      FROM substrate_attention_broadcast_log
     WHERE generated_at <= :at AND generated_at > :since
     ORDER BY generated_at DESC
     LIMIT 1
    """
)


def bind_act_to_workspace(conn: Any, *, actor: str, opened_at: datetime) -> str | None:
    """The open_loop_id of the winner this act answers, or None. Reads the broadcast row current
    at ``opened_at`` (not "now": the writer may lag the act by seconds), so it is replayable."""
    refs = ACT_ANSWERS_WORKSPACE_REFS.get(actor)
    if not refs or not _WORKSPACE_BINDING_ENABLED:
        return None
    from orion.autonomy.self_shed import WINNER_MAX_AGE_SEC, bind_workspace_winner

    at = _aware(opened_at)
    row = conn.execute(_BROADCAST_AT_SQL, {"at": at, "since": at - timedelta(seconds=WINNER_MAX_AGE_SEC)}).mappings().first()
    if row is None:
        return None
    attended = row["attended_node_ids"]
    if isinstance(attended, str):
        attended = json.loads(attended)
    projection = {
        "selected_action_type": row["selected_action_type"],
        "selected_open_loop_id": row["selected_open_loop_id"],
        "generated_at": _aware(row["generated_at"]),
        "dwell_ticks": int(row["dwell_ticks"] or 0),
        "attended_node_ids": list(attended or []),
    }
    winner, _why = bind_workspace_winner(projection, broadcast_log_id=str(row["log_id"]), binds_to_nodes=refs, now=at)
    return None if winner is None else winner.open_loop_id


_LOOP_SALIENCE_SQL = text(
    """
    SELECT (ol->>'salience')::float
      FROM substrate_attention_broadcast_log l,
           jsonb_array_elements(l.projection_json->'frame'->'open_loops') ol
     WHERE l.generated_at >= :since AND l.generated_at <= :until AND ol->>'id' = :loop
     ORDER BY l.generated_at DESC
     LIMIT 1
    """
)


def write_orion_loop_outcome(conn: Any, *, loop_id: str, act: ActV1, value: float, error: float | None,
                             nats: float | None, source: str, opened_at: datetime | None,
                             closed_at: datetime) -> str:
    """Orion's non-final ``acted`` verdict on the loop its act answered. Idempotent per act."""
    from orion.core.ids import stable_hash_id

    outcome_id = stable_hash_id("loopoutcome", [loop_id, "acted", "orion", act.act_id])
    from orion.autonomy.self_shed import WINNER_MAX_AGE_SEC

    # From the bound tick (up to 90 s before the open) to the close: the newest salience the loop had.
    start = _aware(opened_at) if opened_at is not None else _aware(closed_at) - timedelta(hours=24)
    since = start - timedelta(seconds=WINNER_MAX_AGE_SEC)
    salience = conn.execute(_LOOP_SALIENCE_SQL, {"since": since, "until": _aware(closed_at), "loop": loop_id}).scalar()
    features = {
        "act_id": act.act_id, "actor": act.actor, "verb": act.verb,
        "signal_id": act.expect.signal_id if act.expect else None,
        "outcome_value": value, "outcome_source": source, "error": error, "surprise_nats": nats,
        "opened_at": _aware(opened_at).isoformat() if opened_at is not None else None,
    }
    conn.execute(
        text(
            """
            INSERT INTO attention_loop_outcome
                (outcome_id, loop_id, theme_key, verdict, actor, note, salience_at_close,
                 weights_version, features_at_close, created_at)
            VALUES (:outcome_id, :loop_id, :loop_id, 'acted', 'orion', :note, :salience,
                    'gwt-coalition-v1', CAST(:features AS jsonb), :created_at)
            ON CONFLICT (outcome_id) DO NOTHING
            """
        ),
        {"outcome_id": outcome_id, "loop_id": loop_id,
         "note": f"{act.actor} {act.verb} closed {value:.0f} ({source})"[:500],
         "salience": max(0.0, min(1.0, float(salience or 0.0))),
         "features": json.dumps(features, default=str), "created_at": _aware(closed_at)},
    )
    return outcome_id


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(ts: datetime) -> datetime:
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)


def _load_posterior_for_update(conn: Any, kind: str, actor: str, signal_id: str) -> EffectPosterior:
    # Seed the cold row first so FOR UPDATE always has a row to lock: without it two
    # concurrent first closes both read "no row", both compute n=1, and the
    # `posterior_n <` guard silently drops one observation.
    cold = cold_act_posterior()
    conn.execute(
        text(
            """
            INSERT INTO substrate_action_effect_posterior (
                dispatch_kind, target_id, signal_id, baseline_bin,
                posterior_mean, posterior_variance, posterior_n, updated_at
            ) VALUES (:k, :t, :s, 0, :pm, :pv, 0, now())
            ON CONFLICT (dispatch_kind, target_id, signal_id, baseline_bin) DO NOTHING
            """
        ),
        {"k": kind, "t": actor, "s": signal_id, "pm": cold.mean, "pv": cold.variance},
    )
    row = conn.execute(
        text(
            """
            SELECT posterior_mean, posterior_variance, posterior_n
              FROM substrate_action_effect_posterior
             WHERE dispatch_kind = :k AND target_id = :t AND signal_id = :s AND baseline_bin = 0
             FOR UPDATE
            """
        ),
        {"k": kind, "t": actor, "s": signal_id},
    ).first()
    if row is None:
        return cold_act_posterior()
    return EffectPosterior(mean=float(row[0]), variance=float(row[1]), n=int(row[2]))


def _load_posterior(conn: Any, kind: str, actor: str, signal_id: str) -> EffectPosterior:
    row = conn.execute(
        text(
            """
            SELECT posterior_mean, posterior_variance, posterior_n
              FROM substrate_action_effect_posterior
             WHERE dispatch_kind = :k AND target_id = :t AND signal_id = :s AND baseline_bin = 0
            """
        ),
        {"k": kind, "t": actor, "s": signal_id},
    ).first()
    if row is None:
        return cold_act_posterior()
    return EffectPosterior(mean=float(row[0]), variance=float(row[1]), n=int(row[2]))


def _open_params(event: GrammarEventV1, act: ActV1, *, stamp: EffectPosterior | None) -> dict[str, Any]:
    opened_at = _aware(event.observed_at or event.emitted_at)
    exp = act.expect
    horizon_at = opened_at + timedelta(seconds=exp.horizon_sec) if exp is not None else None
    return {
        "act_id": act.act_id,
        "parent_act_id": act.parent_act_id,
        "trace_id": event.trace_id,
        "level": act.level,
        "actor": act.actor,
        "verb": act.verb,
        "object_ref": act.object_ref,
        "test_before": act.test_before,
        "signal_id": exp.signal_id if exp else None,
        "direction": exp.direction if exp else None,
        "horizon_sec": exp.horizon_sec if exp else None,
        "expect_basis": exp.basis if exp else None,
        "predicted": stamp.mean if (exp and stamp) else None,
        "predicted_n": stamp.n if (exp and stamp) else None,
        "cold_start": (stamp.n == 0) if (exp and stamp) else None,
        "goal_ref": act.goal_ref,
        "workspace_ref": act.workspace_ref,
        "session_id": event.session_id,
        "correlation_id": event.correlation_id,
        "opened_at": opened_at,
        "horizon_at": horizon_at,
        "open_event_id": event.event_id,
        "cost_json": json.dumps(act.cost.model_dump(mode="json")) if act.cost else None,
        "now": _utc_now(),
    }


_INSERT_OPEN_SQL = text(
    """
    INSERT INTO grammar_acts (
        act_id, parent_act_id, trace_id, level, actor, verb, object_ref, test_before,
        signal_id, direction, horizon_sec, expect_basis, predicted, predicted_n, cold_start,
        goal_ref, workspace_ref, session_id, correlation_id, opened_at, horizon_at,
        open_event_id, cost_json, created_at, updated_at
    ) VALUES (
        :act_id, :parent_act_id, :trace_id, :level, :actor, :verb, :object_ref, :test_before,
        :signal_id, :direction, :horizon_sec, :expect_basis, :predicted, :predicted_n, :cold_start,
        :goal_ref, :workspace_ref, :session_id, :correlation_id, :opened_at, :horizon_at,
        :open_event_id, CAST(:cost_json AS jsonb), :now, :now
    )
    ON CONFLICT (act_id) DO UPDATE SET
        -- Only a close-before-open row (opened_at NULL) takes the open's fields.
        -- predicted is NOT overwritten: the close already graded against the
        -- belief current at close time and recorded that as predicted.
        parent_act_id = EXCLUDED.parent_act_id,
        test_before = EXCLUDED.test_before,
        session_id = COALESCE(grammar_acts.session_id, EXCLUDED.session_id),
        workspace_ref = COALESCE(grammar_acts.workspace_ref, EXCLUDED.workspace_ref),
        correlation_id = COALESCE(grammar_acts.correlation_id, EXCLUDED.correlation_id),
        opened_at = EXCLUDED.opened_at,
        horizon_at = EXCLUDED.horizon_at,
        open_event_id = EXCLUDED.open_event_id,
        updated_at = EXCLUDED.updated_at
     WHERE grammar_acts.opened_at IS NULL
    """
)


def _isolated(conn: Any, what: str, act_id: str, fn: Any) -> Any:
    """Run a rung-3 side step in its own SAVEPOINT. The binding is a trace of the act: a failure
    (no broadcast table, a bad row) must never abort the act's own open or close, which share
    the caller's transaction. Returns None on failure."""
    nested = conn.begin_nested() if hasattr(conn, "begin_nested") else None
    try:
        out = fn()
        if nested is not None:
            nested.commit()
        return out
    except Exception:
        if nested is not None:
            nested.rollback()
        logger.warning("act_workspace_%s_failed act_id=%s", what, act_id, exc_info=True)
        return None


def _stamp_workspace(conn: Any, params: dict[str, Any], act: ActV1, event: GrammarEventV1) -> None:
    """Writer-stamped workspace_ref (rung 3), only when the producer left it unset."""
    if params.get("workspace_ref") or act.actor not in ACT_ANSWERS_WORKSPACE_REFS or not _WORKSPACE_BINDING_ENABLED:
        return
    loop_id = _isolated(conn, "bind", act.act_id, lambda: bind_act_to_workspace(
        conn, actor=act.actor, opened_at=event.observed_at or event.emitted_at))
    if loop_id:
        params["workspace_ref"] = loop_id


def open_act(conn: Any, event: GrammarEventV1) -> None:
    act = event.act
    assert act is not None
    stamp = None
    if act.expect is not None:
        stamp = _load_posterior(conn, act_dispatch_kind(act.verb), act.actor, act.expect.signal_id)
    params = _open_params(event, act, stamp=stamp)
    _stamp_workspace(conn, params, act, event)
    conn.execute(_INSERT_OPEN_SQL, params)


@dataclass(frozen=True)
class CloseResult:
    act_id: str
    closed: bool
    reason: str
    error: float | None = None
    surprise_nats: float | None = None
    outcome_row_id: int | None = None


def close_act(
    conn: Any,
    event: GrammarEventV1 | None,
    *,
    act: ActV1,
    outcome: ActOutcomeV1,
    close_event_id: str,
) -> CloseResult:
    """Close one act by act_id and grade it into L10. First close wins."""
    row = conn.execute(
        text(
            "SELECT closed_at, predicted, opened_at, horizon_at, workspace_ref FROM grammar_acts "
            "WHERE act_id = :a FOR UPDATE"
        ),
        {"a": act.act_id},
    ).first()
    if row is not None and row[0] is not None:
        return CloseResult(act.act_id, False, "already_closed")
    if row is None:
        # Close arrived before (or without) its open: create the row from the
        # close, which carries the same identity and expect.
        if event is None:
            return CloseResult(act.act_id, False, "no_open_row")
        stamp = None
        if act.expect is not None:
            stamp = _load_posterior(conn, act_dispatch_kind(act.verb), act.actor, act.expect.signal_id)
        params = _open_params(event, act, stamp=stamp)
        params["opened_at"] = None
        params["horizon_at"] = None
        params["open_event_id"] = None
        # No open time is known; the close is the closest moment the act is placed at.
        _stamp_workspace(conn, params, act, event)
        conn.execute(_INSERT_OPEN_SQL, params)
        predicted_at_open = stamp.mean if stamp else None
        workspace_ref = params.get("workspace_ref")
        opened_at_row = None
    else:
        predicted_at_open = None if row[1] is None else float(row[1])
        workspace_ref = row[4]
        opened_at_row = row[2]

    value = float(outcome.value)
    observed_at = _aware(outcome.observed_at)
    note = outcome.note or ""
    horizon_at = None if row is None or row[3] is None else _aware(row[3])
    if horizon_at is not None and observed_at > horizon_at and value == 1.0:
        # The expectation is "by the horizon". A success reported after it is a miss,
        # graded the same whether it lands inside the resolver's grace or after it.
        value = 0.0
        note = f"late:{(observed_at - horizon_at).total_seconds():.0f}s {note}"
    error = None
    nats = None
    outcome_row_id = None
    exp = act.expect
    if exp is not None:
        kind = act_dispatch_kind(act.verb)
        prior = _load_posterior_for_update(conn, kind, act.actor, exp.signal_id)
        predicted = predicted_at_open if predicted_at_open is not None else prior.mean
        posterior, nats, _ = score_act_outcome(prior, value)
        error = value - predicted
        if exp.direction == "increase":
            upheld = value == 1.0
        else:
            upheld = value == 0.0
        latency_ms = None
        if act.cost is not None and act.cost.seconds is not None:
            latency_ms = act.cost.seconds * 1000.0
        inserted = conn.execute(
            text(
                """
                INSERT INTO substrate_action_outcomes (
                    dispatch_id, dispatch_frame_id, feedback_frame_id,
                    dispatch_kind, target_id, signal_id, direction,
                    arm, baseline_bin, frame_dispatch_count,
                    observed_at, baseline, observed_after, observed_delta,
                    predicted_delta, prediction_error, surprise_nats,
                    posterior_mean, posterior_variance, posterior_n,
                    co_predictors, latency_ms, claim_upheld
                ) VALUES (
                    :dispatch_id, :dispatch_frame_id, :feedback_frame_id,
                    :dispatch_kind, :target_id, :signal_id, :direction,
                    'dispatched', 0, 1,
                    :observed_at, 0.0, :value, :value,
                    :predicted, :error, :nats,
                    :pm, :pv, :pn,
                    0, :latency_ms, :upheld
                )
                ON CONFLICT (dispatch_id, signal_id, dispatch_frame_id) DO NOTHING
                RETURNING id
                """
            ),
            {
                "dispatch_id": act.act_id,
                "dispatch_frame_id": f"act:{act.act_id}",
                "feedback_frame_id": close_event_id,
                "dispatch_kind": kind,
                "target_id": act.actor,
                "signal_id": exp.signal_id,
                "direction": exp.direction,
                "observed_at": observed_at,
                "value": value,
                "predicted": predicted,
                "error": error,
                "nats": nats,
                "pm": posterior.mean,
                "pv": posterior.variance,
                "pn": posterior.n,
                "latency_ms": latency_ms,
                "upheld": upheld,
            },
        ).first()
        if inserted is not None:
            outcome_row_id = int(inserted[0])
            conn.execute(
                text(
                    """
                    INSERT INTO substrate_action_effect_posterior (
                        dispatch_kind, target_id, signal_id, baseline_bin,
                        posterior_mean, posterior_variance, posterior_n, updated_at
                    ) VALUES (:k, :t, :s, 0, :pm, :pv, :pn, now())
                    ON CONFLICT (dispatch_kind, target_id, signal_id, baseline_bin)
                    DO UPDATE SET
                        posterior_mean = EXCLUDED.posterior_mean,
                        posterior_variance = EXCLUDED.posterior_variance,
                        posterior_n = EXCLUDED.posterior_n,
                        updated_at = now()
                     WHERE substrate_action_effect_posterior.posterior_n < EXCLUDED.posterior_n
                    """
                ),
                {"k": kind, "t": act.actor, "s": exp.signal_id,
                 "pm": posterior.mean, "pv": posterior.variance, "pn": posterior.n},
            )
        if predicted_at_open is None:
            conn.execute(
                text("UPDATE grammar_acts SET predicted = :p, predicted_n = :n, cold_start = :c WHERE act_id = :a"),
                {"p": predicted, "n": prior.n, "c": prior.n == 0, "a": act.act_id},
            )

    conn.execute(
        text(
            """
            UPDATE grammar_acts SET
                outcome_value = :value, outcome_source = :source, outcome_note = :note,
                error = :error, surprise_nats = :nats, outcome_row_id = :row_id,
                closed_at = :closed_at, close_event_id = :close_event_id,
                cost_json = COALESCE(CAST(:cost_json AS jsonb), cost_json),
                updated_at = :now
             WHERE act_id = :act_id AND closed_at IS NULL
            """
        ),
        {
            "value": value,
            "source": outcome.source,
            "note": note[:500],
            "error": error,
            "nats": nats,
            "row_id": outcome_row_id,
            "closed_at": observed_at,
            "close_event_id": close_event_id,
            "cost_json": json.dumps(act.cost.model_dump(mode="json")) if act.cost else None,
            "now": _utc_now(),
            "act_id": act.act_id,
        },
    )
    if workspace_ref and _WORKSPACE_BINDING_ENABLED and act.actor in ACT_ANSWERS_WORKSPACE_REFS:
        # Same transaction as the close, in its own savepoint: the act and Orion's verdict on its
        # loop land together, and a failed verdict never loses the close.
        _isolated(conn, "loop_outcome", act.act_id, lambda: write_orion_loop_outcome(
            conn, loop_id=str(workspace_ref), act=act, value=value, error=error, nats=nats,
            source=outcome.source, opened_at=opened_at_row, closed_at=observed_at))
    return CloseResult(act.act_id, True, "closed", error=error, surprise_nats=nats, outcome_row_id=outcome_row_id)


def apply_act_events(conn: Any, events: list[GrammarEventV1]) -> int:
    """Persist act events in stream order. Returns how many were applied.

    A bad act never poisons the batch: each is applied inside a SAVEPOINT
    when the connection supports one, and a failure is logged with its act_id.
    """
    applied = 0
    for event in events:
        if event.event_kind not in ACT_EVENT_KINDS or event.act is None:
            continue
        nested = conn.begin_nested() if hasattr(conn, "begin_nested") else None
        try:
            if event.event_kind == "act_opened":
                open_act(conn, event)
            else:
                assert event.act.outcome is not None
                close_act(conn, event, act=event.act, outcome=event.act.outcome, close_event_id=event.event_id)
            if nested is not None:
                nested.commit()
            applied += 1
        except Exception:
            if nested is not None:
                nested.rollback()
            logger.exception("act_event_failed act_id=%s kind=%s", event.act.act_id, event.event_kind)
    return applied


# --- horizon resolver ------------------------------------------------------

# "Did Juniper answer by the horizon" is read from when her next turn STARTED, not
# from chat_history_log.created_at: that row is written only after Orion finishes
# answering (live 2026-10-16: 1 to 41 minutes after she sent the turn) and never for
# a turn that errored, so a turn she sent before the horizon read 0. Two arms:
#
# 1. A logged Juniper turn (canonical JUNIPER_TURN_PREDICATE: non-empty prompt, not
#    unsolicited outreach), started at its hub ingress hop when one was recorded,
#    else at its row's created_at (non-unified hub paths record no ingress). That
#    fallback keeps the old limits for those turns only: one sent before the act
#    but logged after it still counts, and a slow one can still miss the horizon.
# 2. A Juniper ingress hop with no logged row needed (JUNIPER_INGRESS_PREDICATE):
#    the turn she sent that errored, or whose reply is still being written.
#
# Both clocks are naive timestamps on the server's own clock, compared in the
# server's TimeZone. The act's own turn is excluded by correlation_id, otherwise a
# chat reply would always "see" its own turn. The ingress window scan is
# MATERIALIZED on purpose: it reads only (stage, seq, ts), and raw is tested by a
# primary-key probe on the in-window hops only, so the wide raw jsonb of every
# hop in the table is never detoasted (live: 7.6 s with the raw test inline,
# 0.18 s like this, cockpit_turn_sighting at 40k rows / 1.7 GB).
_JUNIPER_REPLY_SQL = text(
    f"""
    WITH bounds AS (
        SELECT CAST(:opened_at AS timestamptz) AT TIME ZONE current_setting('TimeZone') AS lo,
               CAST(:horizon_at AS timestamptz) AT TIME ZONE current_setting('TimeZone') AS hi
    ),
    window_ingress AS MATERIALIZED (
        SELECT s.correlation_id, s.ts
          FROM cockpit_turn_sighting s, bounds b
         WHERE s.stage = 'ingress' AND s.seq = 0 AND s.ts > b.lo AND s.ts <= b.hi
    ),
    starts AS (
        SELECT coalesce(
                   (SELECT i.ts FROM cockpit_turn_sighting i
                     WHERE i.correlation_id = h.correlation_id AND i.stage = 'ingress' AND i.seq = 0),
                   h.created_at) AS started_at,
               h.correlation_id
          FROM (SELECT correlation_id, created_at FROM chat_history_log
                 WHERE {JUNIPER_TURN_PREDICATE}
                   AND created_at > (SELECT lo FROM bounds)) h
        UNION ALL
        SELECT w.ts AS started_at, w.correlation_id
          FROM window_ingress w
         WHERE EXISTS (
                   SELECT 1 FROM cockpit_turn_sighting s
                    WHERE s.correlation_id = w.correlation_id AND {JUNIPER_INGRESS_PREDICATE})
    )
    SELECT min(started_at) AT TIME ZONE current_setting('TimeZone')
      FROM starts, bounds b
     WHERE started_at > b.lo AND started_at <= b.hi
       AND coalesce(correlation_id, '') <> coalesce(:correlation_id, '')
    """
)

_DUE_SQL = text(
    """
    SELECT act_id, level, actor, verb, signal_id, direction, horizon_sec, expect_basis,
           opened_at, horizon_at, correlation_id
      FROM grammar_acts
     WHERE closed_at IS NULL AND opened_at IS NOT NULL AND signal_id IS NOT NULL
       AND (signal_id = 'juniper_replied' OR horizon_at < :expired_before)
     ORDER BY opened_at
     LIMIT :limit
    """
)


@dataclass(frozen=True)
class ResolveSummary:
    examined: int
    closed_replied: int
    closed_not_replied: int
    closed_expired: int
    recovered_producer_closes: int = 0


def _act_from_row(row: Any) -> ActV1:
    return ActV1(
        act_id=row["act_id"],
        level=row["level"],
        actor=row["actor"],
        verb=row["verb"],
        expect=ActExpectV1(
            signal_id=row["signal_id"],
            direction=row["direction"] or "increase",
            horizon_sec=float(row["horizon_sec"]),
            basis=row["expect_basis"] or "unknown",
        ),
    )


_PERSISTED_CLOSE_SQL = text(
    """
    SELECT event_json FROM grammar_events
     WHERE trace_id = :trace_id AND event_kind = 'act_closed'
     ORDER BY created_at, event_id
     LIMIT 1
    """
)


def _persisted_close(conn: Any, act_id: str) -> GrammarEventV1 | None:
    """A producer close that reached grammar_events but never reached grammar_acts.

    That happens when the close's act-ledger savepoint lost a lock fight (deadlock,
    lock timeout) or the close was shed and replayed late: the event row is
    committed, and the fresh-only rule means nothing re-applies it. Checked before
    any expiry so a real report is never overwritten by a 0.
    """
    raw = conn.execute(_PERSISTED_CLOSE_SQL, {"trace_id": f"act:{act_id}"}).scalar()
    if raw is None:
        return None
    try:
        payload = raw if isinstance(raw, dict) else json.loads(raw)
        event = GrammarEventV1.model_validate(payload)
    except Exception:  # noqa: BLE001
        logger.warning("act_persisted_close_unparseable act_id=%s", act_id)
        return None
    return event if event.act is not None and event.act.outcome is not None else None


def _one_close(conn: Any, commit: Any, event: GrammarEventV1 | None, act: ActV1,
               outcome: ActOutcomeV1, close_event_id: str) -> bool:
    """One close in its own savepoint, then commit (if given) so no locks outlive it."""
    nested = conn.begin_nested() if hasattr(conn, "begin_nested") else None
    try:
        close_act(conn, event, act=act, outcome=outcome, close_event_id=close_event_id)
        if nested is not None:
            nested.commit()
        if commit is not None:
            commit()
        return True
    except Exception:
        if nested is not None:
            nested.rollback()
        logger.exception("act_resolve_failed act_id=%s", act.act_id)
        return False


def resolve_due_acts(
    conn: Any,
    *,
    now: datetime | None = None,
    limit: int = 500,
    commit: Any = None,
) -> ResolveSummary:
    """Close acts whose outcome is now known from the world or from the clock.

    ``commit`` (the session's commit) is called after every act, so the resolver
    never holds act or posterior row locks across a whole tick: a long transaction
    here would deadlock against grammar lanes closing acts on the same posterior.
    """
    now = _aware(now or _utc_now())
    rows = conn.execute(
        _DUE_SQL, {"expired_before": now - timedelta(seconds=EXPIRY_GRACE_SEC), "limit": limit}
    ).mappings().all()
    replied = not_replied = expired = recovered = 0
    for row in rows:
        act = _act_from_row(row)
        opened_at = _aware(row["opened_at"])
        horizon_at = _aware(row["horizon_at"])
        if row["signal_id"] == "juniper_replied":
            first = conn.execute(
                _JUNIPER_REPLY_SQL,
                {"opened_at": opened_at, "horizon_at": min(horizon_at, now),
                 "correlation_id": row["correlation_id"]},
            ).scalar()
            if first is not None:
                outcome = ActOutcomeV1(value=1.0, observed_at=_aware(first), source="resolver",
                                       note="juniper chat turn started before horizon")
                replied += 1
            elif now >= horizon_at:
                outcome = ActOutcomeV1(value=0.0, observed_at=horizon_at, source="resolver",
                                       note="no juniper chat turn started before horizon")
                not_replied += 1
            else:
                continue
        else:
            persisted = _persisted_close(conn, act.act_id)
            if persisted is not None:
                if _one_close(conn, commit, persisted, persisted.act, persisted.act.outcome,
                              persisted.event_id):
                    recovered += 1
                continue
            outcome = ActOutcomeV1(value=0.0, observed_at=horizon_at, source="expired",
                                   note=f"no producer close within horizon + {EXPIRY_GRACE_SEC:.0f}s grace")
            expired += 1
        _one_close(conn, commit, None, act, outcome, f"resolver:{act.act_id}")
    return ResolveSummary(len(rows), replied, not_replied, expired, recovered)


# --- debug surface ---------------------------------------------------------

_SUMMARY_SQL = text(
    """
    SELECT verb, actor, level, signal_id,
           count(*)                                              AS opened,
           count(*) FILTER (WHERE closed_at IS NOT NULL)         AS closed,
           count(*) FILTER (WHERE closed_at IS NULL)             AS open,
           -- An outcome actually observed by the horizon: a producer report, or the
           -- resolver seeing Juniper reply. Resolver/expiry zeros are stamped AT the
           -- horizon by construction and are counted separately, not as coverage.
           count(*) FILTER (WHERE closed_at IS NOT NULL AND horizon_at IS NOT NULL
                            AND closed_at <= horizon_at
                            AND (outcome_source = 'producer'
                                 OR (outcome_source = 'resolver' AND outcome_value = 1))
                           )                                     AS closed_in_horizon,
           count(*) FILTER (WHERE outcome_source = 'resolver' AND outcome_value = 0)
                                                                 AS resolved_absent,
           count(*) FILTER (WHERE expect_basis IS NOT NULL AND expect_basis <> '') AS with_expect,
           avg(outcome_value)                                    AS outcome_rate,
           avg(predicted) FILTER (WHERE closed_at IS NOT NULL)   AS mean_predicted,
           avg(abs(error))                                       AS mean_abs_error,
           avg(surprise_nats)                                    AS mean_surprise_nats,
           count(*) FILTER (WHERE outcome_source = 'expired')    AS expired,
           max(coalesce(closed_at, opened_at))                   AS latest
      FROM grammar_acts
     WHERE coalesce(opened_at, closed_at) >= :since
     GROUP BY verb, actor, level, signal_id
     ORDER BY opened DESC
    """
)


def ledger_summary(conn: Any, *, days: float = 7.0, now: datetime | None = None) -> dict[str, Any]:
    """What GET /act/ledger returns: per (verb, actor) coverage, closure and error."""
    now = _aware(now or _utc_now())
    rows = conn.execute(_SUMMARY_SQL, {"since": now - timedelta(days=days)}).mappings().all()

    def _f(v):
        return None if v is None else round(float(v), 4)

    return {
        "window_days": days,
        "generated_at": now.isoformat(),
        "by_act": [
            {
                "verb": r["verb"], "actor": r["actor"], "level": r["level"], "signal_id": r["signal_id"],
                "opened": int(r["opened"]), "closed": int(r["closed"]), "open": int(r["open"]),
                "closed_in_horizon": int(r["closed_in_horizon"]), "with_expect": int(r["with_expect"]),
                "expired": int(r["expired"]), "resolved_absent": int(r["resolved_absent"]),
                "outcome_rate": _f(r["outcome_rate"]), "mean_predicted": _f(r["mean_predicted"]),
                "mean_abs_error": _f(r["mean_abs_error"]), "mean_surprise_nats": _f(r["mean_surprise_nats"]),
                "latest": r["latest"].isoformat() if r["latest"] else None,
            }
            for r in rows
        ],
    }
