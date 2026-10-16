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
  closed from ``chat_history_log`` (1 as soon as Juniper sends a hub chat turn
  after the act, 0 once the horizon passes without one). Producer-reported
  signals (``artifact_produced``) that pass their horizon unclosed are closed
  at 0 as ``expired``: the signal is "produced BY the horizon", so a missing
  close past it is a real 0, not a guess.

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
from orion.regulation.juniper_turns import JUNIPER_TURN_PREDICATE
from orion.schemas.grammar import ActExpectV1, ActOutcomeV1, ActV1, GrammarEventV1

logger = logging.getLogger("orion.grammar.act_ledger")

ACT_EVENT_KINDS = frozenset({"act_opened", "act_closed"})

# How long past its horizon a producer-reported act may still be closed by its
# producer before the resolver records it as not produced. Covers bus + writer
# lag only (grammar batches flush in seconds); it is not a second horizon.
EXPIRY_GRACE_SEC = 120.0


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(ts: datetime) -> datetime:
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)


def _load_posterior_for_update(conn: Any, kind: str, actor: str, signal_id: str) -> EffectPosterior:
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
        correlation_id = COALESCE(grammar_acts.correlation_id, EXCLUDED.correlation_id),
        opened_at = EXCLUDED.opened_at,
        horizon_at = EXCLUDED.horizon_at,
        open_event_id = EXCLUDED.open_event_id,
        updated_at = EXCLUDED.updated_at
     WHERE grammar_acts.opened_at IS NULL
    """
)


def open_act(conn: Any, event: GrammarEventV1) -> None:
    act = event.act
    assert act is not None
    stamp = None
    if act.expect is not None:
        stamp = _load_posterior(conn, act_dispatch_kind(act.verb), act.actor, act.expect.signal_id)
    conn.execute(_INSERT_OPEN_SQL, _open_params(event, act, stamp=stamp))


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
            "SELECT closed_at, predicted, opened_at FROM grammar_acts WHERE act_id = :a FOR UPDATE"
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
        conn.execute(_INSERT_OPEN_SQL, params)
        predicted_at_open = stamp.mean if stamp else None
    else:
        predicted_at_open = None if row[1] is None else float(row[1])

    value = float(outcome.value)
    observed_at = _aware(outcome.observed_at)
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
            "note": (outcome.note or "")[:500],
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

# A chat turn from Juniper, by the canonical predicate
# (orion.regulation.juniper_turns.JUNIPER_TURN_PREDICATE: non-empty prompt and
# not an unsolicited outreach row). chat_history_log.created_at is a naive
# timestamp written by the server's now(), so it is compared in the server's
# own TimeZone setting. The act's own turn row is excluded by correlation_id,
# otherwise a chat reply would always "see" its own turn.
_JUNIPER_REPLY_SQL = text(
    f"""
    SELECT min(created_at AT TIME ZONE current_setting('TimeZone'))
      FROM chat_history_log
     WHERE {JUNIPER_TURN_PREDICATE}
       AND created_at > (CAST(:opened_at AS timestamptz) AT TIME ZONE current_setting('TimeZone'))
       AND created_at <= (CAST(:horizon_at AS timestamptz) AT TIME ZONE current_setting('TimeZone'))
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


def resolve_due_acts(conn: Any, *, now: datetime | None = None, limit: int = 500) -> ResolveSummary:
    """Close acts whose outcome is now known from the world or from the clock."""
    now = _aware(now or _utc_now())
    rows = conn.execute(
        _DUE_SQL, {"expired_before": now - timedelta(seconds=EXPIRY_GRACE_SEC), "limit": limit}
    ).mappings().all()
    replied = not_replied = expired = 0
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
                                       note="juniper chat turn before horizon (chat_history_log)")
                replied += 1
            elif now >= horizon_at:
                outcome = ActOutcomeV1(value=0.0, observed_at=horizon_at, source="resolver",
                                       note="no juniper chat turn before horizon (chat_history_log)")
                not_replied += 1
            else:
                continue
        else:
            outcome = ActOutcomeV1(value=0.0, observed_at=horizon_at, source="expired",
                                   note=f"no producer close within horizon + {EXPIRY_GRACE_SEC:.0f}s grace")
            expired += 1
        nested = conn.begin_nested() if hasattr(conn, "begin_nested") else None
        try:
            close_act(conn, None, act=act, outcome=outcome, close_event_id=f"resolver:{act.act_id}")
            if nested is not None:
                nested.commit()
        except Exception:
            if nested is not None:
                nested.rollback()
            logger.exception("act_resolve_failed act_id=%s", act.act_id)
    return ResolveSummary(len(rows), replied, not_replied, expired)


# --- debug surface ---------------------------------------------------------

_SUMMARY_SQL = text(
    """
    SELECT verb, actor, level, signal_id,
           count(*)                                              AS opened,
           count(*) FILTER (WHERE closed_at IS NOT NULL)         AS closed,
           count(*) FILTER (WHERE closed_at IS NULL)             AS open,
           count(*) FILTER (WHERE closed_at IS NOT NULL AND horizon_at IS NOT NULL
                            AND closed_at <= horizon_at)         AS closed_in_horizon,
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
                "expired": int(r["expired"]),
                "outcome_rate": _f(r["outcome_rate"]), "mean_predicted": _f(r["mean_predicted"]),
                "mean_abs_error": _f(r["mean_abs_error"]), "mean_surprise_nats": _f(r["mean_surprise_nats"]),
                "latest": r["latest"].isoformat() if r["latest"] else None,
            }
            for r in rows
        ],
    }
