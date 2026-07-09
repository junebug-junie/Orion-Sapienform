from __future__ import annotations

import logging
from pathlib import Path

from dotenv import load_dotenv
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

logger = logging.getLogger("orion-thought.settings")

# orion-mind runs 3 sequential LLM phases (semantic → appraisal → stance), each
# capped by MIND_LLM_TIMEOUT_SEC (default 25s on the orion-mind service). A wall
# budget below ~3× that ceiling guarantees synthesis is cut off mid-pipeline and
# the Mind degrades to contract_only (the empty-shell cognition failure mode).
# See fix/mind-enrichment-wall-budget.
MIND_LLM_TIMEOUT_SEC_ASSUMED: float = 25.0
MIND_ENRICHMENT_PHASE_COUNT: int = 3
MIND_ENRICHMENT_MIN_VIABLE_WALL_MS: int = int(
    MIND_LLM_TIMEOUT_SEC_ASSUMED * MIND_ENRICHMENT_PHASE_COUNT * 1000
)


class ThoughtSettings(BaseSettings):
    service_name: str = Field("orion-thought", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")
    node_name: str = Field("athena", alias="NODE_NAME")
    port: int = Field(7155, alias="THOUGHT_PORT")

    orion_bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")
    orion_bus_enforce_catalog: bool = Field(False, alias="ORION_BUS_ENFORCE_CATALOG")
    orion_bus_url: str = Field("redis://100.x.x.x:6379/0", alias="ORION_BUS_URL")

    channel_thought_request: str = Field(
        "orion:thought:request",
        alias="CHANNEL_THOUGHT_REQUEST",
    )
    channel_thought_artifact: str = Field(
        "orion:thought:artifact",
        alias="CHANNEL_THOUGHT_ARTIFACT",
    )
    channel_thought_result_prefix: str = Field(
        "orion:thought:result:",
        alias="CHANNEL_THOUGHT_RESULT_PREFIX",
    )
    channel_cortex_exec_request: str = Field(
        "orion:cortex:exec:request",
        validation_alias=AliasChoices("CHANNEL_CORTEX_EXEC_REQUEST", "CORTEX_EXEC_REQUEST_CHANNEL"),
        alias="CHANNEL_CORTEX_EXEC_REQUEST",
    )
    channel_cortex_exec_result_prefix: str = Field(
        "orion:exec:result",
        validation_alias=AliasChoices("CHANNEL_CORTEX_EXEC_RESULT_PREFIX", "CORTEX_EXEC_RESULT_PREFIX"),
        alias="CHANNEL_CORTEX_EXEC_RESULT_PREFIX",
    )
    stance_react_timeout_sec: float = Field(120.0, alias="STANCE_REACT_TIMEOUT_SEC")

    # --- Reverie: spontaneous-thought mode (Phase A, default-off) ---
    reverie_enabled: bool = Field(False, alias="ORION_REVERIE_ENABLED")
    reverie_interval_sec: float = Field(90.0, alias="ORION_REVERIE_INTERVAL_SEC")
    reverie_min_salience: float = Field(0.0, alias="ORION_REVERIE_MIN_SALIENCE")
    channel_reverie_thought: str = Field(
        "orion:reverie:thought",
        alias="CHANNEL_REVERIE_THOUGHT",
    )

    # --- Reverie semantic lift (v1, default-off) ---
    reverie_semantic_lift_enabled: bool = Field(
        False, alias="ORION_REVERIE_SEMANTIC_LIFT_ENABLED"
    )
    reverie_referent_max_age_hours: float = Field(
        24.0, alias="ORION_REVERIE_REFERENT_MAX_AGE_HOURS"
    )
    channel_reverie_cortex_exec_request: str = Field(
        "orion:cortex:exec:request:background",
        alias="CHANNEL_REVERIE_CORTEX_EXEC_REQUEST",
    )

    # --- Reverie chain (Phase C, default-off) ---
    reverie_chain_enabled: bool = Field(False, alias="ORION_REVERIE_CHAIN_ENABLED")
    reverie_chain_max_steps: int = Field(4, alias="ORION_REVERIE_CHAIN_MAX_STEPS")
    reverie_refractory_sec: float = Field(900.0, alias="ORION_REVERIE_REFRACTORY_SEC")
    reverie_drift_temp: float = Field(0.7, alias="ORION_REVERIE_DRIFT_TEMP")
    channel_reverie_chain: str = Field(
        "orion:reverie:chain",
        alias="CHANNEL_REVERIE_CHAIN",
    )

    # --- Reverie grounding (Phase D, default-off, read-only) ---
    reverie_ground_consolidation: bool = Field(
        False, alias="ORION_REVERIE_GROUND_CONSOLIDATION"
    )

    # --- Compaction request (Phase E, default-off, queue only) ---
    reverie_compaction_request_enabled: bool = Field(
        False, alias="ORION_REVERIE_COMPACTION_REQUEST_ENABLED"
    )
    channel_dream_compaction_request: str = Field(
        "orion:dream:compaction-request",
        alias="CHANNEL_DREAM_COMPACTION_REQUEST",
    )

    # --- Resonance tripwire (Phase H, default-off, observation only) ---
    reverie_resonance_alert_enabled: bool = Field(
        False, alias="ORION_REVERIE_RESONANCE_ALERT_ENABLED"
    )
    channel_reverie_resonance_alert: str = Field(
        "orion:reverie:resonance-alert",
        alias="CHANNEL_REVERIE_RESONANCE_ALERT",
    )
    # How many recent chain rows to scan for a runaway theme.
    reverie_resonance_window: int = Field(200, alias="ORION_REVERIE_RESONANCE_WINDOW")

    # --- Computed salience v2 (shadow-first, default-off) ---
    attention_salience_v2_enabled: bool = Field(False, alias="ORION_ATTENTION_SALIENCE_V2_ENABLED")
    attention_habituation_enabled: bool = Field(False, alias="ORION_ATTENTION_HABITUATION_ENABLED")
    channel_attention_salience_trace: str = Field(
        "orion:attention:salience:trace",
        alias="CHANNEL_ATTENTION_SALIENCE_TRACE",
    )

    # --- Mind stance enrichment (unified turn; default-off) ---
    # Runs orion-mind before stance_react and injects an advisory self/attention
    # coloring. Silent no-op unless orion-mind has MIND_LLM_SYNTHESIS_ENABLED=true
    # (a separate service — not visible to this service's env-parity check).
    #
    # Wall/timeout budget: orion-mind runs THREE sequential LLM phases (semantic →
    # appraisal → stance), each capped by MIND_LLM_TIMEOUT_SEC (default 25s on the
    # orion-mind service). A wall below ~3× that ceiling cuts synthesis off
    # mid-pipeline and forces contract_only degradation, so the wall default must
    # stay >= MIND_ENRICHMENT_MIN_VIABLE_WALL_MS and the HTTP read timeout must
    # exceed the wall (so Mind's own fail-open result is returned, not aborted).
    mind_enrichment_enabled: bool = Field(False, alias="ORION_THOUGHT_MIND_ENRICHMENT_ENABLED")
    mind_base_url: str = Field("http://orion-mind:6611", alias="ORION_MIND_BASE_URL")
    mind_timeout_sec: float = Field(100.0, alias="ORION_THOUGHT_MIND_TIMEOUT_SEC")
    mind_wall_ms: int = Field(90_000, alias="ORION_THOUGHT_MIND_WALL_MS")
    mind_router_profile: str = Field("default", alias="ORION_THOUGHT_MIND_ROUTER_PROFILE")
    mind_max_response_bytes: int = Field(2_000_000, alias="ORION_THOUGHT_MIND_MAX_RESPONSE_BYTES")
    mind_artifact_publish_enabled: bool = Field(
        False, alias="ORION_THOUGHT_MIND_ARTIFACT_PUBLISH_ENABLED"
    )
    mind_coloring_max_items: int = Field(3, alias="ORION_THOUGHT_MIND_COLORING_MAX_ITEMS")
    channel_mind_artifact: str = Field("orion:mind:artifact", alias="CHANNEL_MIND_ARTIFACT")

    # --- Reasoning activity projection (always-on consumer; harmless when idle) ---
    # Consume per-call ReasoningCallV1 telemetry and materialize a rolling-window
    # ReasoningActivityV1 for φ. The buffer is capped (max_calls) so memory is
    # bounded regardless of producer rate.
    channel_reasoning_call: str = Field(
        "orion:cognition:reasoning_call", alias="CHANNEL_REASONING_CALL"
    )
    reasoning_activity_window_sec: float = Field(
        120.0, alias="REASONING_ACTIVITY_WINDOW_SEC"
    )
    reasoning_activity_max_calls: int = Field(
        2000, alias="REASONING_ACTIVITY_MAX_CALLS"
    )


def mind_enrichment_config_warnings(s: "ThoughtSettings") -> list[str]:
    """Deterministic boot-time coherence checks for the Mind enrichment budget.

    Only meaningful when enrichment is enabled. Returns human-readable warnings
    for budget settings that would silently force contract_only degradation.
    """
    warnings: list[str] = []
    if not s.mind_enrichment_enabled:
        return warnings
    if s.mind_wall_ms < MIND_ENRICHMENT_MIN_VIABLE_WALL_MS:
        warnings.append(
            f"wall_too_small ORION_THOUGHT_MIND_WALL_MS={s.mind_wall_ms} < "
            f"min_viable={MIND_ENRICHMENT_MIN_VIABLE_WALL_MS}: 3-phase LLM synthesis "
            "will be cut off mid-pipeline and Mind will degrade to contract_only"
        )
    if s.mind_timeout_sec * 1000.0 <= s.mind_wall_ms:
        warnings.append(
            f"http_timeout_not_above_wall ORION_THOUGHT_MIND_TIMEOUT_SEC={s.mind_timeout_sec}s "
            f"(<= ORION_THOUGHT_MIND_WALL_MS={s.mind_wall_ms}ms): the HTTP client may abort "
            "before Mind returns its own fail-open result, losing diagnostics/artifact"
        )
    return warnings


settings = ThoughtSettings()
logger.info(
    "Loaded orion-thought settings service=%s v=%s port=%s",
    settings.service_name,
    settings.service_version,
    settings.port,
)
for _mind_cfg_warning in mind_enrichment_config_warnings(settings):
    logger.warning("mind_enrichment_config %s", _mind_cfg_warning)
