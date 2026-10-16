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
    # Attend-to-act loop (docs/superpowers/specs/2026-09-29-attend-to-act-loop-design.md): build the
    # workspace-winner-bound world candidates (today: shed_background_gpu). Record-only on its own:
    # execution dispatch blocks every such candidate `world_actions_disabled` until Juniper sets
    # ORION_WORLD_ACTIONS_ENABLED there. OFF in code, ON in .env_example.
    workspace_winner_proposals_enabled: bool = Field(
        False, alias="ORION_WORKSPACE_WINNER_PROPOSALS_ENABLED"
    )
    hardware_watch_health_url: str = Field(
        "http://hardware-watch:8131/health", alias="ORION_HARDWARE_WATCH_HEALTH_URL"
    )
    world_action_rise_threshold_c: float = Field(
        0.5, ge=0.0, alias="ORION_SHED_RISE_THRESHOLD_C"
    )
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
