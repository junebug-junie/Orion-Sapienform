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

    # Host-pipe transition gate: interpret only on hard_labels / person-presence changes (evidence_transition.py).
    COUNCIL_TRANSITION_GATE_ENABLED: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "COUNCIL_TRANSITION_GATE_ENABLED",
            "COUNCIL_EVIDENCE_SKIP_ENABLED",
        ),
    )
    # Force refresh at least this often even when labels are stable (0 = never force).
    #
    # Live incident, 2026-08-23/25: this defaulted to 0 (never force), and a
    # home-office scene's coarse hard_labels (chair/clothing/desk/door/person/
    # table) genuinely never change turn to turn -- so evidence_transition.py's
    # gate correctly, deterministically decided "stable_scene" on every single
    # window for 44+ continuous hours, and `orion:vision:events`/`vision_events`
    # went completely silent with zero errors anywhere in the pipeline (vision-
    # host kept doing GPU inference the whole time; council kept ticking the
    # whole time; the gate was working exactly as coded, just with no ceiling).
    # 600s (10 min) bounds the worst case comfortably inside the 900s staleness
    # cutoff `orion/situational/context.py::_build_perception_context` already
    # uses to decide "stale" vs "live", without re-running the metacog LLM call
    # anywhere near as often as genuine scene changes do (~11/hour observed,
    # i.e. every ~5.5 min on average per orion-substrate-runtime/app/settings.py).
    #
    # Known, accepted gap (not fully closed by this fix): reverie's own percept
    # freshness gate (`services/orion-thought/app/settings.py`'s
    # `reverie_perception_max_age_sec`, 180s) is tighter than this 600s ceiling.
    # On a genuinely static scene, reverie will still see "no fresh percept"
    # for roughly 70% of every refresh cycle (420 of 600s) -- a real improvement
    # over the ~100%-blind 44h outage this fix closes, but not "always fresh."
    # Deliberately not lowered to 180s to match: that would mean forcing an LLM
    # reconfirmation call every 3 minutes even when nothing is happening, which
    # is *more* frequent than the natural ~5.5min cadence of genuine changes --
    # defeating the point of the gate during exactly the quiet periods it
    # exists for. If reverie's blind fraction on stable scenes needs to shrink
    # further, that's a follow-up tuned on reverie's own gate, not this one.
    COUNCIL_TRANSITION_REFRESH_SEC: float = Field(
        default=600.0,
        validation_alias=AliasChoices(
            "COUNCIL_TRANSITION_REFRESH_SEC",
            "COUNCIL_EVIDENCE_SKIP_MAX_SEC",
        ),
    )
