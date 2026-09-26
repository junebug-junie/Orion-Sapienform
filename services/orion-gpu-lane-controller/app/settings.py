from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    GPU2_ENABLED: bool = False
    GPU2_DIFFUSION_URL: str = "http://100.112.254.99:8014"
    GPU2_AGENT_URL: str = "http://100.112.254.99:8016"
    GPU2_AUTHORITY_URL: str = "http://100.92.216.81:8124"
    GPU2_DRAIN_TIMEOUT_SEC: float = 300.0
    GPU2_MODEL_READY_TIMEOUT_SEC: float = 600.0

    # GPU pool stage 4.2 (docs/superpowers/specs/2026-09-25-gpu-pool-stage4-durable-runs-and-actuation.md).
    # Who may move gpu2. `durable` (default) = today: durable-runs' HTTP activate route, fenced by
    # GPU2_AUTHORITY_URL/elastic/status; pool actuation requests are refused `authority_durable`.
    # `pool` = only GpuActuateV1 on the bus, fenced by the pool's generation (persisted at
    # GPU2_POOL_FENCE_STATE_PATH) and launch_digest; the HTTP activate route is refused. Transitional:
    # the `durable` branch is deleted in stage 4.6.
    GPU2_AUTHORITY: Literal["durable", "pool"] = "durable"
    # This host's actuator name in config/gpu_pool.yaml `actuators:`; requests for another name are ignored.
    GPU_POOL_ACTUATOR_NAME: str = "circe"
    # Last accepted pool generation + recent action results. Must be on a volume that survives a
    # container recreate, or a restart would re-admit an old generation.
    GPU2_POOL_FENCE_STATE_PATH: str = "/state/gpu2_pool_fence.json"

    SERVICE_NAME: str = "gpu-lane-controller"
    SERVICE_VERSION: str = "0.1.0"
    NODE_NAME: str = "circe"
    LOG_LEVEL: str = "INFO"

    # Bus: heartbeat, plus (stage 4.2) intake on orion:gpu_pool:actuate:request and results on
    # orion:gpu_pool:actuate:result. GPU1's flip stays HTTP-only.
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

    # Shared-secret auth, required on the one write endpoint (POST
    # /v1/gpu-lane/flip) only -- not on GET /health or GET
    # /v1/gpu-lane/status. This is a control-plane surface (stops/starts
    # GPU-heavy containers) reachable over tailscale from athena; the
    # existing cross-host precedent (biometrics_node_client.py) is
    # unauthenticated, but that's read-only telemetry, a materially
    # different risk than this. Empty string means "reject every flip
    # request" (fail closed), not "auth disabled" -- see lane_control.py.
    GPU_LANE_CONTROLLER_TOKEN: str = ""

    # Fixed compose targets. Deliberately NOT a caller-supplied service name
    # (unlike cortex-exec's compose_service_bringup.v1) -- this controller
    # only ever flips between these two, so the command surface stays
    # narrow by construction rather than by an allowlist check.
    AFFECT_COMPOSE_RELPATH: str = "services/orion-affectgpt-worker/docker-compose.yml"
    AFFECT_ENV_RELPATH: str = "services/orion-affectgpt-worker/.env"
    AFFECT_COMPOSE_SERVICE: str = "affectgpt-worker"

    # docker-compose.atlas-workers.yml defines FOUR worker lanes (chat/
    # metacog/fast/agent) in one file -- every invocation against it below
    # MUST name atlas-agent explicitly. A bare `stop`/`up` with no service
    # argument would touch chat/metacog/fast too.
    AGENT_COMPOSE_RELPATH: str = "services/orion-llamacpp-host/docker-compose.atlas-workers.yml"
    # NOTE: the README's own quickstart says `.env.atlas` -- that is NOT
    # what's actually deployed. Confirmed live on circe 2026-09-02: the real
    # file every ATLAS_*/agent-lane container reads is plain `.env` in this
    # same directory (ls'd directly; no `.env.atlas` exists there at all).
    # Runtime truth over documented convention -- see CLAUDE.md's "Runtime
    # truth beats config truth."
    AGENT_ENV_RELPATH: str = "services/orion-llamacpp-host/.env"
    AGENT_COMPOSE_SERVICE: str = "atlas-agent"
    AGENT_COMPOSE_PROFILE: str = "agent-split"

    # GPU1 is the entire point of this lane -- fixed here rather than
    # trusted from whatever ATLAS_AGENT_CUDA_VISIBLE_DEVICES happens to
    # already be set to in the env file, since a stale/different value there
    # would silently start the agent worker on the wrong card. Confirmed
    # live 2026-09-02: it's currently "2" there (Tesla PG500-216) -- a card
    # shared with diffusion-host, which idles near 0 but spikes to ~25GB
    # while actively generating, real OOM-contention risk for anything
    # permanently resident alongside it. GPU1 (post affect-gpt eviction) has
    # no such neighbor. Passed as an explicit process-env override at
    # invocation time (compose variable substitution prefers real process
    # env over --env-file), not relied on from the env file.
    AGENT_GPU1_CUDA_VISIBLE_DEVICES: str = "1"

    # Generous like cortex-exec's own SKILLS_DOCKER_COMPOSE_BRINGUP_*
    # defaults (900s/60s) -- a 27B GGUF cold load is real time, and a first
    # `docker compose build` for either image can also be slow.
    GPU_LANE_COMMAND_TIMEOUT_SEC: float = 900.0
    GPU_LANE_HEALTH_POLL_SEC: float = 180.0


settings = Settings()
