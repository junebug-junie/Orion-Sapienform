from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    project: str = Field("orion-athena", alias="PROJECT")
    service_name: str = Field("orion-proposal-runtime", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")
    node_name: str = Field("athena", alias="NODE_NAME")

    # Bus-native SystemHealthV1 heartbeat (orion:system:health). Proposal-runtime had no bus
    # connection at all before this -- these fields exist solely to carry an independent
    # heartbeat chassis. See
    # docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md.
    orion_bus_url: str = Field("redis://100.92.216.81:6379/0", alias="ORION_BUS_URL")
    orion_bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")
    heartbeat_interval_sec: float = Field(10.0, alias="HEARTBEAT_INTERVAL_SEC")

    postgres_uri: str = Field(..., alias="POSTGRES_URI")
    proposal_policy_path: str = Field(
        "config/proposals/proposal_policy.v1.yaml",
        alias="PROPOSAL_POLICY_PATH",
    )
    proposal_poll_interval_sec: float = Field(2.0, alias="PROPOSAL_POLL_INTERVAL_SEC")
    enable_proposal_runtime: bool = Field(True, alias="ENABLE_PROPOSAL_RUNTIME")
    enable_transport_proposals: bool = Field(False, alias="ENABLE_TRANSPORT_PROPOSALS")
    transport_proposal_mode: str = Field("read_only", alias="TRANSPORT_PROPOSAL_MODE")
    # Phase B: incorporate spontaneous-thought (reverie) proposals into Layer 7.
    # Default-off; candidates are operator_review-gated (never auto-dispatch).
    reverie_propose_enabled: bool = Field(False, alias="ORION_REVERIE_PROPOSE_ENABLED")
    # Phase B dispatch gate ("sends email"). Default-off. Recorded as an inspectable
    # posture on the candidate; it NEVER lowers the operator_review gate — a human
    # still gates any reverie action. Arming it only signals dispatch intent to L9.
    reverie_autoaction_enabled: bool = Field(False, alias="ORION_REVERIE_AUTOACTION_ENABLED")
    # 2026-11-05 (ladder retirement PR 5): the workspace-winner world candidates are gone. Their only
    # template, shed_background_gpu, is the cooling world act, now decided by the want market in
    # orion-durable-runs (app/cooling_act.py); ORION_WORKSPACE_WINNER_PROPOSALS_ENABLED,
    # ORION_HARDWARE_WATCH_HEALTH_URL and ORION_SHED_RISE_THRESHOLD_C moved there or were removed.
    # Temporal Self rev 4, R3 reader 4: while Orion's arousal (Redis orion:regulation:latest,
    # written by orion-durable-runs) reads `strained`, a due visual-reverie baseline image is
    # deferred by the policy's retry_sec (reason arousal_strained in the frame warnings).
    # Missing / stale (> max age) / unknown / Redis down / flag off = today's behaviour.
    visual_baseline_arousal_enabled: bool = Field(True, alias="ORION_VISUAL_BASELINE_AROUSAL_ENABLED")
    regulation_reader_max_age_sec: float = Field(
        360.0, gt=0.0, alias="ORION_REGULATION_READER_MAX_AGE_SEC"
    )
    log_level: str = Field("INFO", alias="LOG_LEVEL")


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
