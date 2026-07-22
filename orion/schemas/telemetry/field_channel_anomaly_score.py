"""field_channel_corpus.v1 anomaly-score envelope -- published by
orion-field-digester's periodic anomaly-scoring loop (app/anomaly_scorer.py),
consumed by orion-equilibrium-service's telemetry_anomaly_metacog_gate to
fire the telemetry_anomaly metacog trigger.

Mirrors orion.schemas.repair_pressure_appraisal.RepairPressureAppraisalV1's
role for the relational trigger: the producer publishes the raw measurement
(recon_loss vs. its own train-time reference), the consumer applies its own
configurable multiplier to decide whether to trigger -- so the trigger
threshold can be tuned as an equilibrium-service operator setting without
redeploying orion-field-digester.
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class FieldChannelAnomalyScoreV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    correlation_id: str
    encoder_id: str
    encoder_version: str
    recon_loss: float
    recon_error_p95: float
    threshold_multiplier: float
    threshold: float
    anomalous: bool
    window_start: datetime
    window_end: datetime
    window_size: int
    # Top-N channels by per-window reconstruction error, highest first, as
    # "channel=mse" strings -- same shape as the now-deleted
    # orion.self_state.builder.evidence_for_dimension()'s dominant_evidence
    # output (module removed 2026-07-22, SelfStateV1 burn). Added 2026-07-21
    # so a firing says *which* channels moved, not
    # just that something did. Empty list is a valid, honest state (e.g. the
    # producer failed to compute attribution) -- never fabricated.
    top_channels: list[str] = Field(default_factory=list)
    # Signed mean of (x - xhat) over the window, plus its coarse label --
    # see orion.mood_arc.fit_encoder.mean_signed_deviation()/
    # deviation_direction(). recon_loss alone can't distinguish an elevated
    # (load spike) window from a depressed (quiet lull) one; both read as
    # equally "anomalous" under squared error. Added 2026-07-21. 0.0/"mixed"
    # is the honest default when not computed, not a fabricated reading.
    mean_signed_deviation: float = 0.0
    deviation_direction: str = "mixed"
    # False means top_channels/mean_signed_deviation/deviation_direction are
    # all fallback defaults because the attribution computation itself
    # raised, NOT that the window was genuinely near-zero/mixed -- the two
    # were indistinguishable before this field (review finding, 2026-07-21).
    attribution_ok: bool = True
