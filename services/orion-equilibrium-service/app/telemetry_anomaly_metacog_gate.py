from __future__ import annotations

import logging
from typing import Any

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

    reason = f"telemetry_anomaly:recon_loss={float(recon_loss):.5f}:threshold={threshold:.5f}"

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
        },
    )
