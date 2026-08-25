from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    SERVICE_NAME: str = "vision-council"
    SERVICE_VERSION: str = "0.1.0"
    NODE_NAME: str = "athena"
    LOG_LEVEL: str = "INFO"

    # Bus
    ORION_BUS_URL: str = "redis://localhost:6379/0"
    ORION_BUS_ENFORCE_CATALOG: bool = False
    # Bus-native SystemHealthV1 heartbeat cadence (orion:system:health). See
    # docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md.
    HEARTBEAT_INTERVAL_SEC: float = 10.0

    # Channels
    CHANNEL_COUNCIL_INTAKE: str = "orion:vision:windows"
    CHANNEL_COUNCIL_PUB: str = "orion:vision:events"

    # Cortex Exec
    CHANNEL_COUNCIL_REQUEST: str = "orion:exec:request:VisionCouncilService"

    CHANNEL_LLM_REQUEST: str = "orion:exec:request:LLMGatewayService"
    CHANNEL_LLM_REPLY_PREFIX: str = "orion:council:reply"

    # Config
    COUNCIL_MODEL: str = "llama-3-8b-instruct-q4_k_m"
    COUNCIL_LLM_ROUTE: str = "metacog"
    COUNCIL_LLM_MAX_TOKENS: int = 1024
    COUNCIL_LLM_TIMEOUT_SEC: float = 90.0
    COUNCIL_STRUCTURED_OUTPUT_METHOD: str = "json_object_schema"

    # Foveal probe (docs/superpowers/specs/2026-08-12-perception-frontier-design.md's
    # "Foveal tier" -- rare, event-driven, richer-than-BLIP interpretation of the
    # current frame, distinct from the always-on peripheral pipeline). Manually
    # triggered via POST /debug/foveal-probe today; not yet wired to any
    # automatic trigger (surprise-driven foveation is P2, blocked on
    # want_embeddings -- see that design doc's pragmatic ladder).
    #
    # Empty by default -- unset means "no foveal host configured", and the
    # probe refuses cleanly rather than silently calling a channel nobody is
    # listening on. Point this at a dedicated, ISOLATED vision-host intake
    # channel (e.g. circe's orion:exec:request:VisionHostService:circe-vl),
    # never at the shared orion:exec:request:VisionHostService channel the
    # frame-router's continuous pipeline uses -- two consumers on that shared
    # channel race on every task and the faster (usually wrong) reply wins;
    # see PR #1859 / this session's own incident record for why this is a
    # hard requirement, not a style preference.
    CHANNEL_FOVEAL_HOST_REQUEST: str = ""
    CHANNEL_FOVEAL_HOST_REPLY_PREFIX: str = "orion:vision:reply:foveal"
    FOVEAL_HOST_TIMEOUT_SEC: float = 45.0

    # Where captured frames already live on this node's local disk (read-only
    # mount -- same convention orion-vision-host itself uses via
    # VISION_FRAMES_DIR). The probe reads the newest .jpg here, uploads it to
    # the percept store, and hands the resulting sha256 to the foveal host --
    # never a local path, since the foveal host is on a different machine
    # with no shared filesystem.
    FOVEAL_FRAMES_DIR: str = "/mnt/telemetry/vision/frames"

    FOVEAL_PERCEPT_STORE_URL: str = ""
    FOVEAL_PERCEPT_STORE_TOKEN: str = ""
    FOVEAL_PERCEPT_UPLOAD_TIMEOUT_SEC: float = 10.0

    # Host-pipe transition gate: interpret only on hard_labels / person-presence changes (evidence_transition.py).
    COUNCIL_TRANSITION_GATE_ENABLED: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "COUNCIL_TRANSITION_GATE_ENABLED",
            "COUNCIL_EVIDENCE_SKIP_ENABLED",
        ),
    )
    # Force refresh at least this often even when labels are stable (0 = never force).
    COUNCIL_TRANSITION_REFRESH_SEC: float = Field(
        default=0.0,
        validation_alias=AliasChoices(
            "COUNCIL_TRANSITION_REFRESH_SEC",
            "COUNCIL_EVIDENCE_SKIP_MAX_SEC",
        ),
    )
