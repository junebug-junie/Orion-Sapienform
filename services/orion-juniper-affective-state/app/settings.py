from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    SERVICE_NAME: str = "juniper-affective-state"
    SERVICE_VERSION: str = "0.1.0"
    # Deliberately "circe", NOT athena. video_path/audio_path fed to the
    # worker are resolved on the WORKER's filesystem (orion-affectgpt-worker
    # also runs on circe) -- circe and athena share no filesystem (confirmed
    # live, reference_circe_gpu_inventory_and_lane_map: /mnt/telemetry is
    # athena-local ext4, no NFS/exports; /mnt/scripts is a separate clone per
    # host, not synced). Colocating this service with the worker sidesteps
    # that gap entirely. capture_and_assess() below is the real cross-host
    # bridge for a live capture source (carbon, 2026-08-22): it fetches
    # percept-store blobs to a local temp dir HERE, then hands the worker
    # ordinary local paths same as before -- the worker's own contract never
    # had to change.
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

    # Bus-reachable clip capture on carbon (or any node running
    # orion-vision-retina with RETINA_CLIP_ENABLED=true) -- see
    # orion/bus/channels.yaml's orion:exec:request:RetinaClipCaptureService.
    # Values must match that service's own CHANNEL_RETINA_CLIP_INTAKE /
    # CHANNEL_RETINA_CLIP_REPLY_PREFIX, same "shared constant, not just a
    # coincidentally-equal default" caveat as CHANNEL_AFFECTGPT_REPLY_PREFIX
    # above.
    CHANNEL_RETINA_CLIP_INTAKE: str = "orion:exec:request:RetinaClipCaptureService"
    CHANNEL_RETINA_CLIP_REPLY_PREFIX: str = "orion:retina:clip:reply"
    # Generous: retina's own RETINA_CLIP_TIMEOUT_SEC ceiling is 30s on top of
    # an ~8s capture, plus RPC/bus overhead. A caller-facing timeout tighter
    # than retina's own would mask retina's real error with a generic "did
    # not reply in time" here.
    RETINA_CLIP_RPC_TIMEOUT_S: float = 60.0

    # Where capture_and_assess() fetches the video/audio blobs retina
    # uploaded. Same base-URL convention as orion-vision-retina's
    # RETINA_PERCEPT_STORE_URL (includes the /percepts suffix already,
    # e.g. http://100.92.216.81:8021/percepts) -- GET {base}/{sha256}.
    PERCEPT_STORE_BASE_URL: str = ""
    PERCEPT_STORE_TIMEOUT_SEC: float = 15.0
    # Sent as X-Orion-Percept-Token if set -- same shared-secret convention
    # retina's own upload path already uses. Added so PERCEPT_STORE_TOKEN
    # can actually be turned on for orion-percept-store (it defaults
    # disabled, "acceptable only on a closed tailnet") without silently
    # breaking every capture_and_assess() fetch with a 401 (review finding,
    # 2026-08-22: this setting didn't exist at all before, so there was no
    # way to close that gap from this side even if an operator wanted to).
    PERCEPT_STORE_TOKEN: str = ""

    # WHERE the fetched clip is written before being handed to the worker.
    # Must be the SAME shared volume orion-affectgpt-worker mounts read-only
    # at the identical container path (see both services' docker-compose.yml)
    # -- video_path/audio_path in AffectGptAssessRequestPayload are resolved
    # on the WORKER's filesystem, so a plain tempfile.TemporaryDirectory()
    # (which defaults to /tmp, private to THIS container) would write
    # somewhere the worker container can never see. Confirmed live pattern:
    # orion-affectgpt-worker/docker-compose.yml mounts
    # /mnt/scripts/orion-affectgpt-scratch:/mnt/scripts/orion-affectgpt-scratch:ro
    # at the same path this service mounts read-write.
    AFFECTGPT_SCRATCH_DIR: str = "/mnt/scripts/orion-affectgpt-scratch"
