from __future__ import annotations

import logging
from pathlib import Path

from dotenv import load_dotenv
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

logger = logging.getLogger("orion-harness-governor.settings")


class HarnessGovernorSettings(BaseSettings):
    service_name: str = Field("orion-harness-governor", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")
    node_name: str = Field("athena", alias="NODE_NAME")
    port: int = Field(7156, alias="HARNESS_GOVERNOR_PORT")

    orion_bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")
    orion_bus_enforce_catalog: bool = Field(False, alias="ORION_BUS_ENFORCE_CATALOG")
    orion_bus_url: str = Field("redis://100.x.x.x:6379/0", alias="ORION_BUS_URL")
    orion_harness_governor_enabled: bool = Field(True, alias="ORION_HARNESS_GOVERNOR_ENABLED")

    channel_harness_run_request: str = Field(
        "orion:harness:run:request",
        alias="CHANNEL_HARNESS_RUN_REQUEST",
    )
    channel_harness_run_artifact: str = Field(
        "orion:harness:run:artifact",
        alias="CHANNEL_HARNESS_RUN_ARTIFACT",
    )
    channel_harness_result_prefix: str = Field(
        "orion:harness:run:result:",
        alias="CHANNEL_HARNESS_RESULT_PREFIX",
    )
    channel_grammar_event: str = Field(
        "orion:grammar:event",
        alias="CHANNEL_GRAMMAR_EVENT",
    )
    channel_cortex_exec_request: str = Field(
        "orion:cortex:exec:request:background",
        validation_alias=AliasChoices("CHANNEL_CORTEX_EXEC_REQUEST", "CORTEX_EXEC_REQUEST_CHANNEL"),
        alias="CHANNEL_CORTEX_EXEC_REQUEST",
    )
    channel_cortex_exec_result_prefix: str = Field(
        "orion:exec:result",
        validation_alias=AliasChoices("CHANNEL_CORTEX_EXEC_RESULT_PREFIX", "CORTEX_EXEC_RESULT_PREFIX"),
        alias="CHANNEL_CORTEX_EXEC_RESULT_PREFIX",
    )
    channel_finalize_appraisal_request: str = Field(
        "orion:substrate:finalize_appraisal:request",
        alias="CHANNEL_FINALIZE_APPRAISAL_REQUEST",
    )
    channel_finalize_appraisal_result_prefix: str = Field(
        "orion:substrate:finalize_appraisal:result:",
        alias="CHANNEL_FINALIZE_APPRAISAL_RESULT_PREFIX",
    )
    channel_post_turn_closure: str = Field(
        "orion:substrate:post_turn_closure",
        alias="CHANNEL_POST_TURN_CLOSURE",
    )
    channel_system_error: str = Field(
        "orion:system:error",
        validation_alias=AliasChoices("CHANNEL_SYSTEM_ERROR", "ORION_ERROR_CHANNEL"),
        alias="CHANNEL_SYSTEM_ERROR",
    )
    channel_harness_run_step: str = Field(
        "orion:harness:run:step",
        alias="CHANNEL_HARNESS_RUN_STEP",
    )
    channel_harness_run_cancel: str = Field(
        "orion:harness:run:cancel",
        alias="CHANNEL_HARNESS_RUN_CANCEL",
    )

    fcc_timeout_sec: float = Field(900.0, alias="HARNESS_FCC_TIMEOUT_SEC")
    finalize_reflect_timeout_sec: float = Field(180.0, alias="FINALIZE_REFLECT_TIMEOUT_SEC")
    voice_finalize_timeout_sec: float = Field(300.0, alias="VOICE_FINALIZE_TIMEOUT_SEC")
    substrate_finalize_timeout_sec: float = Field(5.0, alias="SUBSTRATE_FINALIZE_TIMEOUT_SEC")
    finalize_quick_gate_epsilon: float = Field(0.08, alias="FINALIZE_QUICK_GATE_EPSILON")
    harness_fcc_stream_idle_timeout_sec: float = Field(
        180.0, alias="HARNESS_FCC_STREAM_IDLE_TIMEOUT_SEC"
    )
    harness_fcc_include_partial_messages: bool = Field(
        True, alias="HARNESS_FCC_INCLUDE_PARTIAL_MESSAGES"
    )
    harness_fcc_partial_stream_timeout_sec: float = Field(
        90.0, alias="HARNESS_FCC_PARTIAL_STREAM_TIMEOUT_SEC"
    )
    harness_fcc_partial_progress_interval_sec: float = Field(
        15.0, alias="HARNESS_FCC_PARTIAL_PROGRESS_INTERVAL_SEC"
    )

    harness_fcc_mcp_enabled: bool = Field(False, alias="HARNESS_FCC_MCP_ENABLED")
    harness_aitown_enabled: bool = Field(False, alias="HARNESS_AITOWN_ENABLED")
    # Semantic self-indexing MCPs — read at spawn time from the environment in
    # orion/harness/fcc_motor; mirrored here so operators see effective flags.
    harness_fcc_gitnexus_enabled: bool = Field(False, alias="HARNESS_FCC_GITNEXUS_ENABLED")
    harness_fcc_context_mode_enabled: bool = Field(False, alias="HARNESS_FCC_CONTEXT_MODE_ENABLED")
    harness_fcc_context_mode_dir: str = Field(
        "/var/lib/orion/context-mode", alias="HARNESS_FCC_CONTEXT_MODE_DIR"
    )
    harness_fcc_context_mode_hooks_enabled: bool = Field(
        True, alias="HARNESS_FCC_CONTEXT_MODE_HOOKS_ENABLED"
    )
    harness_fcc_force_no_thinking_model: bool = Field(
        True, alias="HARNESS_FCC_FORCE_NO_THINKING_MODEL"
    )

    # (D) embodiment: publish a deliberate approach intent on the turn correlation_id
    # after a finalized relational turn. Default-off, fail-open (never breaks a turn).
    embodiment_d_finalize_enabled: bool = Field(False, alias="EMBODIMENT_D_FINALIZE_ENABLED")


settings = HarnessGovernorSettings()
logger.info(
    "Loaded orion-harness-governor settings service=%s v=%s port=%s",
    settings.service_name,
    settings.service_version,
    settings.port,
)
