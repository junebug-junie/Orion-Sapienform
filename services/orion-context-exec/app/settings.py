from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

logger = logging.getLogger("orion-context-exec.settings")


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


class ContextExecSettings(BaseSettings):
    service_name: str = Field("context-exec", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")
    node_name: str = Field("athena", alias="NODE_NAME")
    port: int = Field(8096, alias="CONTEXT_EXEC_PORT")

    orion_bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")
    orion_bus_enforce_catalog: bool = Field(False, alias="ORION_BUS_ENFORCE_CATALOG")
    orion_bus_url: str = Field("redis://100.92.216.81:6379/0", alias="ORION_BUS_URL")

    channel_context_exec_intake: str = Field(
        "orion:exec:request:ContextExecService",
        validation_alias=AliasChoices("CHANNEL_CONTEXT_EXEC_INTAKE", "CONTEXT_EXEC_REQUEST_CHANNEL"),
        alias="CHANNEL_CONTEXT_EXEC_INTAKE",
    )
    channel_context_exec_reply_prefix: str = Field(
        "orion:exec:result:ContextExecService",
        alias="CHANNEL_CONTEXT_EXEC_REPLY_PREFIX",
    )
    channel_context_exec_event: str = Field(
        "orion:context_exec:event",
        alias="CHANNEL_CONTEXT_EXEC_EVENT",
    )
    channel_llm_intake: str = Field(
        "orion:exec:request:LLMGatewayService",
        alias="CHANNEL_LLM_INTAKE",
    )
    channel_recall_intake: str = Field(
        "orion:exec:request:RecallService",
        alias="CHANNEL_RECALL_INTAKE",
    )
    channel_recall_reply_prefix: str = Field(
        "orion:recall:reply",
        alias="CHANNEL_RECALL_REPLY_PREFIX",
    )

    context_exec_enabled: bool = Field(True, alias="CONTEXT_EXEC_ENABLED")
    context_exec_sandbox_mode: str = Field("docker", alias="CONTEXT_EXEC_SANDBOX_MODE")
    context_exec_max_depth: int = Field(1, alias="CONTEXT_EXEC_MAX_DEPTH")
    context_exec_max_subcalls: int = Field(6, alias="CONTEXT_EXEC_MAX_SUBCALLS")
    context_exec_max_seconds: float = Field(45.0, alias="CONTEXT_EXEC_MAX_SECONDS")
    context_exec_write_enabled: bool = Field(False, alias="CONTEXT_EXEC_WRITE_ENABLED")
    context_exec_network_enabled: bool = Field(False, alias="CONTEXT_EXEC_NETWORK_ENABLED")
    context_exec_repl_output_chars: int = Field(8192, alias="CONTEXT_EXEC_REPL_OUTPUT_CHARS")
    context_exec_compat_agent_chain_enabled: bool = Field(
        False,
        alias="CONTEXT_EXEC_COMPAT_AGENT_CHAIN_ENABLED",
    )
    context_exec_fake_organs_enabled: bool = Field(
        False,
        alias="CONTEXT_EXEC_FAKE_ORGANS_ENABLED",
    )
    context_exec_agent_chain_intake_alias: str = Field(
        "orion:exec:request:AgentChainService",
        alias="CONTEXT_EXEC_AGENT_CHAIN_INTAKE_ALIAS",
    )

    context_exec_real_trace_enabled: bool = Field(True, alias="CONTEXT_EXEC_REAL_TRACE_ENABLED")
    context_exec_real_recall_enabled: bool = Field(True, alias="CONTEXT_EXEC_REAL_RECALL_ENABLED")
    context_exec_real_repo_enabled: bool = Field(True, alias="CONTEXT_EXEC_REAL_REPO_ENABLED")
    context_exec_repo_root: str = Field("/app", alias="CONTEXT_EXEC_REPO_ROOT")
    context_exec_repo_max_file_chars: int = Field(12000, alias="CONTEXT_EXEC_REPO_MAX_FILE_CHARS")
    context_exec_trace_limit: int = Field(40, alias="CONTEXT_EXEC_TRACE_LIMIT")
    context_exec_recall_limit: int = Field(12, alias="CONTEXT_EXEC_RECALL_LIMIT")
    context_exec_recall_timeout_sec: float = Field(25.0, alias="CONTEXT_EXEC_RECALL_TIMEOUT_SEC")

    orion_repo_root: str = Field("/app", alias="ORION_REPO_ROOT")
    rlm_engine: str = Field("fake", alias="CONTEXT_EXEC_RLM_ENGINE")
    context_exec_rlm_fallback_enabled: bool = Field(True, alias="CONTEXT_EXEC_RLM_FALLBACK_ENABLED")

    context_exec_proposal_ledger_enabled: bool = Field(
        False,
        alias="CONTEXT_EXEC_PROPOSAL_LEDGER_ENABLED",
    )
    context_exec_proposal_ledger_store_path: str = Field(
        "",
        alias="CONTEXT_EXEC_PROPOSAL_LEDGER_STORE_PATH",
    )
    context_exec_proposal_ledger_auto_triage: bool = Field(
        False,
        alias="CONTEXT_EXEC_PROPOSAL_LEDGER_AUTO_TRIAGE",
    )

    proposal_review_api_enabled: bool = Field(True, alias="PROPOSAL_REVIEW_API_ENABLED")
    proposal_ledger_store_path: str = Field("", alias="PROPOSAL_LEDGER_STORE_PATH")


settings = ContextExecSettings()
logger.info(
    "Loaded context-exec settings service=%s v=%s port=%s sandbox=%s write=%s",
    settings.service_name,
    settings.service_version,
    settings.port,
    settings.context_exec_sandbox_mode,
    settings.context_exec_write_enabled,
)
