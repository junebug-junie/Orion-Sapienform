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

from pydantic import BaseModel, ConfigDict


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
