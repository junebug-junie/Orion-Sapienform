"""Settle-time scoring pass for world-action episodes, and the store it reads and writes.

Moved unchanged from orion-feedback-runtime (``app/worker.py::_score_world_episodes`` and
``app/store.py``) when the feedback rung retired on 2026-11-04
(docs/superpowers/specs/2026-11-04-retire-feedback-and-consolidation-rungs.md). Hosted now by
orion-sql-writer (``app/world_settlement_loop.py``), beside the act ledger's horizon resolver.

What it does (attend-to-act loop, docs/superpowers/specs/2026-09-29-attend-to-act-loop-design.md):
an execution-dispatch world action (today only ``shed_background_gpu``) writes an episode row
(treated or randomized-holdback control) before it acts. Once the episode's settle window has closed,
this pass scores it on the cabinet sensor (``orion.autonomy.world_settlement.score_world_episode``,
pure) and writes, in one transaction:

* a ``substrate_action_outcomes`` row (+ the treated cell's ``substrate_action_effect_posterior``
  advance), or one observation into the control cell (``substrate_signal_control_cells``);
* Orion's non-final ``acted`` verdict on the attention loop (``attention_loop_outcome``), never
  ``resolved``/``dismissed``;
* the episode's score (``substrate_world_action_episodes``).

Record-only: it acts on nothing, and only finds episodes once execution dispatch made one.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import text

from orion.autonomy.cabinet_heat import CABINET_HEAT_SIGNAL, cabinet_heat_pressure, minute_mean
from orion.autonomy.contrast import ControlCell, ControlCellKey, TreatedCellKey, baseline_bin
from orion.autonomy.prediction import EffectPosterior
from orion.autonomy.world_settlement import CONTROL_ARM, score_world_episode
from orion.schemas.action_prediction import ActionOutcomeRecordV1

logger = logging.getLogger("orion.autonomy.world_settlement_store")


class WorldSettlementStore:
    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def load_world_episodes_due(self, *, now: datetime, limit: int = 20) -> list[dict]:
        """Unscored episodes whose window has closed. [] while the ledger does not exist yet
        (execution dispatch creates it on its first world decision)."""
        from orion.autonomy import world_episodes

        with self._engine.connect() as conn:
            if conn.execute(text("SELECT to_regclass(:t)"), {"t": world_episodes.TABLE}).scalar() is None:
                return []
            rows = world_episodes.due_for_scoring(conn, now=now, limit=limit)
        for row in rows:
            for key in ("eligibility", "expected_effect", "settlement", "outcome"):
                if isinstance(row.get(key), str):
                    row[key] = json.loads(row[key])
        return rows

    def load_effect_posteriors(self) -> dict[TreatedCellKey, EffectPosterior]:
        with self._engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT dispatch_kind, target_id, signal_id, baseline_bin, posterior_mean, posterior_variance, "
                "posterior_n FROM substrate_action_effect_posterior")).mappings().all()
        return {(r["dispatch_kind"], r["target_id"], r["signal_id"], int(r["baseline_bin"])):
                EffectPosterior(mean=float(r["posterior_mean"]), variance=float(r["posterior_variance"]),
                                n=int(r["posterior_n"])) for r in rows}

    def load_control_posteriors(self) -> dict[ControlCellKey, ControlCell]:
        with self._engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT signal_id, arm, baseline_bin, posterior_mean, posterior_variance, posterior_n, moved_n, "
                "move_rate FROM substrate_signal_control_cells")).mappings().all()
        return {(r["signal_id"], r["arm"], int(r["baseline_bin"])): ControlCell(
            posterior=EffectPosterior(mean=float(r["posterior_mean"]), variance=float(r["posterior_variance"]),
                                      n=int(r["posterior_n"])),
            moved_n=int(r["moved_n"]), move_rate=float(r["move_rate"])) for r in rows}

    def load_cabinet_points(self, *, since: datetime, until: datetime) -> list:
        from orion.autonomy.cabinet_heat import load_cabinet_points

        with self._engine.connect() as conn:
            return load_cabinet_points(conn, since=since, until=until)

    def load_hardware_incidents_opened(self, *, since: datetime, until: datetime) -> list[dict]:
        """hardware_watch_incident rows that OPENED in the window (overlap tags). Raises when the
        table cannot be read: a scorer that cannot see incidents must not score as if none opened."""
        with self._engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT incident_id, rule, subject, opened_at FROM hardware_watch_incident "
                "WHERE opened_at >= :since AND opened_at <= :until"), {"since": since, "until": until}).mappings().all()
        return [dict(r) for r in rows]

    def load_loop_salience(self, *, open_loop_id: str, since: datetime) -> float | None:
        """The loop's salience on the newest broadcast tick that still carries it (the 'next tick')."""
        with self._engine.connect() as conn:
            value = conn.execute(text(
                "SELECT (ol->>'salience')::float FROM substrate_attention_broadcast_log l, "
                "jsonb_array_elements(l.projection_json->'frame'->'open_loops') ol "
                "WHERE l.generated_at >= :since AND ol->>'id' = :loop ORDER BY l.generated_at DESC LIMIT 1"),
                {"since": since, "loop": open_loop_id}).scalar()
        return None if value is None else float(value)

    def save_world_score(self, *, episode_id: str, score, scored_at: datetime) -> bool:
        """One transaction: ledger row (+ posterior for a treated, scorable row) or control cell, the
        episode's score, and Orion's non-final `acted` loop verdict. Idempotent on the episode."""
        from orion.autonomy import world_episodes
        from orion.core.ids import stable_hash_id

        lo = score.loop_outcome
        if lo is not None and (lo.get("verdict") != "acted" or lo.get("actor") != "orion"):
            raise ValueError("the world settle path may only write Orion's non-final 'acted' verdict")
        outcome = dict(score.outcome)
        loop_outcome_id = None
        with self._engine.begin() as conn:
            if score.record is not None:
                _write_action_outcome(conn, score.record)
                outcome["outcome_row_id"] = conn.execute(text(
                    "SELECT id FROM substrate_action_outcomes WHERE dispatch_id = :d AND signal_id = :s "
                    "AND dispatch_frame_id = :f"), {"d": score.record.dispatch_id, "s": score.record.signal_id,
                                                      "f": score.record.dispatch_frame_id}).scalar()
            if score.control_cell is not None:
                key, cell = score.control_cell
                _write_control_cell(conn, key, cell, dispatch_frame_id=f"world_settle:{episode_id}")
            if lo is not None:
                loop_outcome_id = stable_hash_id("loopoutcome", [lo["loop_id"], "acted", "orion", episode_id])
                conn.execute(text(
                    """
                    INSERT INTO attention_loop_outcome
                        (outcome_id, loop_id, theme_key, verdict, actor, note, salience_at_close,
                         weights_version, features_at_close, created_at)
                    VALUES (:outcome_id, :loop_id, :loop_id, 'acted', 'orion', :note, :salience,
                            'gwt-coalition-v1', CAST(:features AS jsonb), :created_at)
                    ON CONFLICT (outcome_id) DO NOTHING
                    """), {"outcome_id": loop_outcome_id, "loop_id": lo["loop_id"], "note": lo.get("note", "")[:500],
                             "salience": max(0.0, min(1.0, float(lo.get("salience_at_close") or 0.0))),
                             "features": json.dumps(lo.get("features_at_close") or {}, default=str),
                             "created_at": scored_at})
            return world_episodes.record_score(conn, episode_id=episode_id, outcome=outcome,
                                               loop_outcome_id=loop_outcome_id, scored_at=scored_at)


def _write_action_outcome(conn, record: ActionOutcomeRecordV1) -> None:
    """Append one scored outcome and advance its posterior. Two double-count guards: the ledger
    insert is ON CONFLICT DO NOTHING and RETURNS only a genuinely new row; the posterior advances only
    for that new row and only forward in ``posterior_n``. A non-dispatched arm keeps its row but never
    moves the action's own belief (it would record the weather as the action's effect)."""
    inserted = conn.execute(text(
        """
        INSERT INTO substrate_action_outcomes (
            dispatch_id, dispatch_frame_id, feedback_frame_id, dispatch_kind, target_id, signal_id, direction,
            arm, baseline_bin, frame_dispatch_count, observed_at, baseline, observed_after, observed_delta,
            predicted_delta, prediction_error, surprise_nats, posterior_mean, posterior_variance, posterior_n,
            co_predictors, latency_ms, claim_upheld
        ) VALUES (
            :dispatch_id, :dispatch_frame_id, :feedback_frame_id, :dispatch_kind, :target_id, :signal_id, :direction,
            :arm, :baseline_bin, :frame_dispatch_count, :observed_at, :baseline, :observed_after, :observed_delta,
            :predicted_delta, :prediction_error, :surprise_nats, :posterior_mean, :posterior_variance, :posterior_n,
            :co_predictors, :latency_ms, :claim_upheld
        )
        ON CONFLICT (dispatch_id, signal_id, dispatch_frame_id) DO NOTHING
        RETURNING id
        """), {k: getattr(record, k) for k in (
            "dispatch_id", "dispatch_frame_id", "feedback_frame_id", "dispatch_kind", "target_id", "signal_id",
            "direction", "arm", "baseline_bin", "frame_dispatch_count", "observed_at", "baseline", "observed_after",
            "observed_delta", "predicted_delta", "prediction_error", "surprise_nats", "posterior_mean",
            "posterior_variance", "posterior_n", "co_predictors", "latency_ms", "claim_upheld")}).first()
    if inserted is None or record.arm != "dispatched":
        return
    conn.execute(text(
        """
        INSERT INTO substrate_action_effect_posterior (
            dispatch_kind, target_id, signal_id, baseline_bin, posterior_mean, posterior_variance, posterior_n, updated_at
        ) VALUES (:dispatch_kind, :target_id, :signal_id, :baseline_bin, :posterior_mean, :posterior_variance,
                  :posterior_n, now())
        ON CONFLICT (dispatch_kind, target_id, signal_id, baseline_bin) DO UPDATE SET
            posterior_mean = EXCLUDED.posterior_mean, posterior_variance = EXCLUDED.posterior_variance,
            posterior_n = EXCLUDED.posterior_n, updated_at = now()
         WHERE substrate_action_effect_posterior.posterior_n < EXCLUDED.posterior_n
        """), {k: getattr(record, k) for k in (
            "dispatch_kind", "target_id", "signal_id", "baseline_bin", "posterior_mean", "posterior_variance",
            "posterior_n")})


def _write_control_cell(conn, key: ControlCellKey, cell: ControlCell, *, dispatch_frame_id: str) -> None:
    """Advance the untreated arm once per episode: ``posterior_n`` only moves forward, and
    ``last_dispatch_frame_id`` refuses the same episode folded twice."""
    signal_id, arm, bin_index = key
    conn.execute(text(
        """
        INSERT INTO substrate_signal_control_cells (
            signal_id, arm, baseline_bin, posterior_mean, posterior_variance, posterior_n,
            moved_n, move_rate, last_dispatch_frame_id, updated_at
        ) VALUES (:signal_id, :arm, :baseline_bin, :posterior_mean, :posterior_variance, :posterior_n,
                  :moved_n, :move_rate, :dispatch_frame_id, now())
        ON CONFLICT (signal_id, arm, baseline_bin) DO UPDATE SET
            posterior_mean = EXCLUDED.posterior_mean, posterior_variance = EXCLUDED.posterior_variance,
            posterior_n = EXCLUDED.posterior_n, moved_n = EXCLUDED.moved_n, move_rate = EXCLUDED.move_rate,
            last_dispatch_frame_id = EXCLUDED.last_dispatch_frame_id, updated_at = now()
         WHERE substrate_signal_control_cells.posterior_n < EXCLUDED.posterior_n
           AND substrate_signal_control_cells.last_dispatch_frame_id IS DISTINCT FROM EXCLUDED.last_dispatch_frame_id
        """), {"signal_id": signal_id, "arm": arm, "baseline_bin": bin_index,
               "posterior_mean": cell.posterior.mean, "posterior_variance": cell.posterior.variance,
               "posterior_n": cell.posterior.n, "moved_n": cell.moved_n, "move_rate": cell.move_rate,
               "dispatch_frame_id": dispatch_frame_id})


def score_due_world_episodes(store: Any, *, now: Optional[datetime] = None) -> int:
    """Score every due episode (both arms). Returns the number scored. An episode whose inputs
    cannot be read is skipped (and retried next pass), never scored as if nothing happened."""
    now = now or datetime.now(timezone.utc)
    episodes = store.load_world_episodes_due(now=now)
    if not episodes:
        return 0
    posteriors = store.load_effect_posteriors()
    controls = store.load_control_posteriors()
    scored = 0
    for ep in episodes:
        t0 = ep["decided_at"]
        ttl = float((ep.get("settlement") or {}).get("ttl_sec") or 900.0)
        end = t0 + timedelta(seconds=ttl + 300.0)
        try:
            points = store.load_cabinet_points(since=t0 - timedelta(minutes=2), until=end + timedelta(minutes=2))
            incidents = store.load_hardware_incidents_opened(since=t0, until=end)
        except Exception:
            logger.warning("world_settlement_inputs_unavailable episode_id=%s", ep.get("episode_id"), exc_info=True)
            continue
        bin_index = baseline_bin(float(cabinet_heat_pressure(minute_mean(points, t0)) or 0.0)) if points else 0
        prior = posteriors.get((ep.get("dispatch_kind"), ep.get("target_id"), CABINET_HEAT_SIGNAL, bin_index))
        control = controls.get((CABINET_HEAT_SIGNAL, CONTROL_ARM, bin_index))
        score = score_world_episode(episode=ep, points=points, incidents=incidents, prior=prior,
                                    control_prior=control, now=now)
        if score is None:
            continue
        if score.loop_outcome is not None:
            try:
                sal = store.load_loop_salience(open_loop_id=score.loop_outcome["loop_id"], since=end)
            except Exception:
                sal = None
            score.loop_outcome["salience_at_close"] = sal or 0.0
            score.loop_outcome["features_at_close"]["next_tick_salience"] = sal
        if store.save_world_score(episode_id=str(ep["episode_id"]), score=score, scored_at=now):
            scored += 1
            logger.info("world_episode_scored episode_id=%s arm=%s terminal=%s delta=%s excluded=%s overlap=%s "
                        "posterior_updated=%s loop_outcome=%s before_c=%s after_c=%s",
                        ep["episode_id"], ep.get("arm"), score.outcome.get("terminal"),
                        score.outcome.get("observed_delta"), score.outcome.get("excluded_reason"),
                        score.outcome.get("overlap"), score.outcome.get("posterior_updated"),
                        bool(score.loop_outcome), score.outcome.get("before_temp_c"), score.outcome.get("after_temp_c"))
    return scored


__all__ = ["WorldSettlementStore", "score_due_world_episodes"]
