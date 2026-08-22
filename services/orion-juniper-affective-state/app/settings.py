from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    SERVICE_NAME: str = "juniper-affective-state"
    SERVICE_VERSION: str = "0.1.0"
    # Deliberately "circe", NOT athena. video_path/audio_path in a request
    # are resolved on the WORKER's filesystem (orion-affectgpt-worker also
    # runs on circe) -- circe and athena share no filesystem (confirmed live,
    # reference_circe_gpu_inventory_and_lane_map: /mnt/telemetry is
    # athena-local ext4, no NFS/exports; /mnt/scripts is a separate clone per
    # host, not synced). Colocating this service with the worker sidesteps
    # that gap entirely rather than half-solving cross-host byte transfer for
    # a capture pipeline that doesn't exist yet. If a live capture source
    # ever lands on a different host, THAT is when real upload/streaming
    # needs building -- not guessed at here.
    NODE_NAME: str = "circe"
    LOG_LEVEL: str = "INFO"

    ORION_BUS_ENABLED: bool = True
    ORION_BUS_ENFORCE_CATALOG: bool = False
    ORION_BUS_URL: str = "redis://localhost:6379/0"
    HEARTBEAT_INTERVAL_SEC: float = 10.0

    # Where the request goes.
    CHANNEL_AFFECTGPT_INTAKE: str = "orion:exec:request:AffectGptWorkerService"
    # Prefix for the per-request reply channel this service listens on
    # (actual channel is f"{prefix}:{corr_id}"). Must match the worker's own
    # CHANNEL_AFFECTGPT_REPLY_PREFIX -- both default to the same value, but
    # this was a hardcoded literal here until review (2026-08-22) caught
    # that it made the worker's matching setting a non-functional config
    # knob (the worker always honors envelope.reply_to over its own prefix
    # setting, so nothing broke live, but the two could silently desync).
    CHANNEL_AFFECTGPT_REPLY_PREFIX: str = "orion:affectgpt:reply"
    # Where this service's own domain event goes after wrapping the worker's
    # reply -- see orion/schemas/affectgpt.py for why this is deliberately
    # NOT orion:substrate:juniper_affective_state (the existing, narrower,
    # text-only signal).
    CHANNEL_AFFECTGPT_ASSESSMENT: str = "orion:affectgpt:assessment"

    AFFECTGPT_RPC_TIMEOUT_S: float = 120.0
