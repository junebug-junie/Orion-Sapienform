from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from orion.metacog.anomaly_episode import (
    AnomalyEpisodeDecision,
    AnomalyEpisodeStateV1,
    TurnActivityTracker,
    episode_upstream,
)
from orion.metacog.anomaly_episode import step as episode_step
from orion.schemas.telemetry.metacog_trigger import MetacogTriggerV1

logger = logging.getLogger("orion.equilibrium.telemetry_anomaly_metacog_gate")


def build_telemetry_anomaly_metacog_trigger(
    *,
    correlation_id: str,
    score: dict[str, Any] | None,
    zen_state: str,
    pressure: float,
    recall_enabled: bool,
    threshold_multiplier: float,
) -> MetacogTriggerV1 | None:
    """Turn a live field_channel_corpus.v1 anomaly score (published on
    orion:field_channel:anomaly_score by orion-field-digester's periodic
    anomaly-scoring loop, app/anomaly_scorer.py) into a "telemetry_anomaly"
    metacog trigger.

    Applies this service's OWN threshold_multiplier against the score's raw
    recon_loss/recon_error_p95 rather than trusting the producer's embedded
    `anomalous` flag -- same pattern as build_repair_pressure_metacog_trigger
    applying its own level/confidence floors, so trigger sensitivity is an
    equilibrium-service operator setting, not something that requires
    redeploying orion-field-digester to tune.
    """
    if not isinstance(score, dict):
        return None

    recon_loss = score.get("recon_loss")
    recon_error_p95 = score.get("recon_error_p95")
    if not isinstance(recon_loss, (int, float)) or not isinstance(recon_error_p95, (int, float)):
        return None

    threshold = float(recon_error_p95) * threshold_multiplier
    if float(recon_loss) <= threshold:
        return None

    top_channels = score.get("top_channels")
    top_channels = top_channels if isinstance(top_channels, list) else []
    direction = score.get("deviation_direction")
    direction = direction if isinstance(direction, str) else "mixed"
    reason = f"telemetry_anomaly:{direction}:recon_loss={float(recon_loss):.5f}:threshold={threshold:.5f}"
    if top_channels:
        # top_channels entries are already "channel=mse" strings (see
        # orion.mood_arc.fit_encoder.top_channel_attribution) -- folding the
        # top one into reason means the *summary* says which channel moved,
        # not just the upstream dict a consumer has to know to dig into.
        reason = f"{reason}:top={top_channels[0]}"

    return MetacogTriggerV1(
        trigger_kind="telemetry_anomaly",
        reason=reason[:500],
        zen_state=zen_state,
        pressure=pressure,
        recall_enabled=recall_enabled,
        signal_refs=[correlation_id] if correlation_id else [],
        upstream={
            "recon_loss": recon_loss,
            "recon_error_p95": recon_error_p95,
            "threshold": threshold,
            "threshold_multiplier": threshold_multiplier,
            "window_start": score.get("window_start"),
            "window_end": score.get("window_end"),
            "encoder_id": score.get("encoder_id"),
            "encoder_version": score.get("encoder_version"),
            "top_channels": top_channels,
            "mean_signed_deviation": score.get("mean_signed_deviation"),
            "deviation_direction": direction,
        },
    )


# ---------------------------------------------------------------------------
# Episode gate (2026-10-13). One fire per anomaly episode instead of one per
# flagged 60s window, with turn-coupled windows left to the chat_turn trigger.
# Pure logic lives in orion/metacog/anomaly_episode.py; this is the adapter
# from a raw anomaly-score payload to that state machine.
# ---------------------------------------------------------------------------

# Producer identity of a harness (chat) turn on orion:grammar:event -- the
# service whose execution_run deltas feed the task_execution channels
# (services/orion-field-digester/app/ingest/state_deltas.py, ":harness_motor").
HARNESS_TURN_SOURCE_SERVICE = "orion-harness-governor"

_glossary_warned = False


def lead_channel_category(top_channels: list[Any]) -> str | None:
    """Glossary category (config/field/field_channel_glossary.v1.yaml) of the
    lead attributed channel. None when there is no lead or the glossary is not
    readable -- None never matches a turn-coupled category, so a missing
    glossary fails open (fires) rather than suppressing everything."""
    global _glossary_warned
    if not top_channels:
        return None
    lead = str(top_channels[0]).split("=", 1)[0].strip()
    if not lead:
        return None
    try:
        from orion.field.channel_glossary import resolve_channel_entry

        entry = resolve_channel_entry(lead)
    except Exception:
        if not _glossary_warned:
            _glossary_warned = True
            logger.warning("telemetry_anomaly_glossary_unavailable lead=%s", lead, exc_info=True)
        return None
    return entry.category if entry is not None else None


def _epoch(value: Any) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def evaluate_telemetry_anomaly_episode(
    state: AnomalyEpisodeStateV1,
    *,
    correlation_id: str,
    score: dict[str, Any] | None,
    turns: TurnActivityTracker | None,
    zen_state: str,
    pressure: float,
    recall_enabled: bool,
    threshold_multiplier: float,
    calm_gap_sec: float,
    sustained_windows: int,
    now: float,
) -> tuple[AnomalyEpisodeStateV1, AnomalyEpisodeDecision | None, MetacogTriggerV1 | None]:
    """Fold one score into the episode state.

    Returns (new_state, decision, trigger). trigger is set only when the
    episode gate decides to fire; the caller must call
    orion.metacog.anomaly_episode.mark_published() on the returned state only
    after the publish really succeeded. decision is None for a malformed score
    (state unchanged)."""
    if not isinstance(score, dict):
        return state, None, None
    recon_loss = score.get("recon_loss")
    recon_error_p95 = score.get("recon_error_p95")
    if not isinstance(recon_loss, (int, float)) or not isinstance(recon_error_p95, (int, float)):
        return state, None, None
    threshold = float(recon_error_p95) * threshold_multiplier
    if threshold <= 0:
        return state, None, None
    ratio = float(recon_loss) / threshold
    window_end = _epoch(score.get("window_end")) or now
    window_start = _epoch(score.get("window_start")) or window_end
    top_channels = score.get("top_channels") if isinstance(score.get("top_channels"), list) else []
    flagged = float(recon_loss) > threshold
    category = lead_channel_category(top_channels) if flagged else None
    turn_active = bool(turns is not None and flagged and turns.active_between(window_start, window_end))

    decision = episode_step(
        state,
        window_end=window_end,
        window_end_iso=str(score.get("window_end") or ""),
        flagged=flagged,
        ratio=ratio,
        lead_category=category,
        turn_active=turn_active,
        calm_gap_sec=calm_gap_sec,
        turn_suppress=turns is not None,
        sustained_windows=sustained_windows,
    )
    if not decision.fire:
        return decision.state, decision, None
    trigger = build_telemetry_anomaly_metacog_trigger(
        correlation_id=correlation_id,
        score=score,
        zen_state=zen_state,
        pressure=pressure,
        recall_enabled=recall_enabled,
        threshold_multiplier=threshold_multiplier,
    )
    if trigger is None:  # cannot happen when flagged; keep the episode unfired
        return decision.state, decision, None
    trigger.upstream.update(episode_upstream(decision))
    trigger.upstream["lead_channel_category"] = category
    trigger.upstream["turn_active"] = turn_active
    return decision.state, decision, trigger
