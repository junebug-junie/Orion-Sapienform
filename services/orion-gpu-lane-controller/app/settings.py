from __future__ import annotations


from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # GPU pool (docs/superpowers/specs/2026-09-29-gpu-pool-stage5-world-diffusion-generic-actuation.md).
    # A card moves only on GpuActuateV1 from the pool (bus), fenced by the pool's generation
    # (persisted at GPU_POOL_FENCE_STATE_PATH) and launch_digest, and runs the role's `launch` block
    # from this checkout's config/gpu_pool.yaml. Stage 5.6 deleted the gpu2 bridge and its GPU2_* keys,
    # the GPU1 affect/agent flip and GPU_LANE_CONTROLLER_TOKEN.
    # This host's actuator name in config/gpu_pool.yaml `actuators:`; requests for another name are ignored.
    GPU_POOL_ACTUATOR_NAME: str = "circe"
    # Last accepted pool generation + recent action results. Must be on a volume that survives a
    # container recreate, or a restart would re-admit an old generation. (Renamed from
    # GPU2_POOL_FENCE_STATE_PATH in 5.6; same default file, so the recorded fence carries over.)
    GPU_POOL_FENCE_STATE_PATH: str = "/state/gpu2_pool_fence.json"

    SERVICE_NAME: str = "gpu-lane-controller"
    SERVICE_VERSION: str = "0.1.0"
    NODE_NAME: str = "circe"
    LOG_LEVEL: str = "INFO"

    # Bus: heartbeat, plus (stage 4.2) intake on orion:gpu_pool:actuate:request and results on
    # orion:gpu_pool:actuate:result. The only control surface (stage 5.6 deleted the HTTP ones).
    ORION_BUS_ENABLED: bool = True
    ORION_BUS_ENFORCE_CATALOG: bool = False
    ORION_BUS_URL: str = "redis://localhost:6379/0"
    HEARTBEAT_INTERVAL_SEC: float = 10.0

    # Repo root as seen INSIDE this container. The checked-out repo is
    # bind-mounted read-only here (see docker-compose.yml) so `docker
    # compose` -- invoked against the host's docker socket, also mounted --
    # can see the same services/*/docker-compose.yml and env files a human
    # operator on circe would. This is what actually crosses the athena
    # (cortex-exec/Hub) / circe host boundary: cortex-exec's own
    # skills.docker.compose_service_bringup.v1 can only ever reach its own
    # host's repo checkout, never circe's.
    GPU_LANE_REPO_ROOT: str = "/repo"

    # Per `docker compose stop`/`up` call. Generous like cortex-exec's own
    # SKILLS_DOCKER_COMPOSE_BRINGUP_* defaults -- a stop can wait on a slow shutdown.
    GPU_LANE_COMMAND_TIMEOUT_SEC: float = 900.0
    # Max wait for a `launch.drain` role's in-flight work to finish before its stop (launch.drain has
    # no timeout key). Renamed from GPU2_DRAIN_TIMEOUT_SEC in 5.6; same meaning and default.
    GPU_LANE_DRAIN_TIMEOUT_SEC: float = 300.0


settings = Settings()
