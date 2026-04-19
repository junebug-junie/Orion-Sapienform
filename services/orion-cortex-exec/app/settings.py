from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Identity
    service_name: str = Field("cortex-exec", alias="SERVICE_NAME")
    service_version: str = Field("0.2.0", alias="SERVICE_VERSION")
    node_name: str = Field("athena", alias="NODE_NAME")

    # Bus
    orion_bus_url: str = Field("redis://100.92.216.81:6379/0", alias="ORION_BUS_URL")
    orion_bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")
    orion_bus_enforce_catalog: bool = Field(False, alias="ORION_BUS_ENFORCE_CATALOG")
    heartbeat_interval_sec: float = Field(10.0, alias="HEARTBEAT_INTERVAL_SEC")

    # Intake channel (hub or orch -> exec)
    channel_exec_request: str = Field("orion:cortex:exec:request", alias="CHANNEL_EXEC_REQUEST")

    # Downstream routing (exec -> step services)
    exec_request_prefix: str = Field("orion:exec:request", alias="EXEC_REQUEST_PREFIX")
    exec_result_prefix: str = Field("orion:exec:result", alias="EXEC_RESULT_PREFIX")

    # CHANGED: 8000 -> 60000 (60s). LLMs need time.
    step_timeout_ms: int = Field(60000, alias="STEP_TIMEOUT_MS")

    # Chat lane generation budgets (completion tokens)
    llm_chat_max_tokens_default: int = Field(512, alias="LLM_CHAT_MAX_TOKENS_DEFAULT")
    llm_chat_quick_max_tokens: int = Field(512, alias="LLM_CHAT_QUICK_MAX_TOKENS")
    llm_chat_general_max_tokens: int = Field(512, alias="LLM_CHAT_GENERAL_MAX_TOKENS")

    # CHANGED: "orion-llm:intake" -> "orion:exec:request:LLMGatewayService"
    channel_llm_intake: str = Field("orion:exec:request:LLMGatewayService", alias="CHANNEL_LLM_INTAKE")
    channel_recall_intake: str = Field("orion:exec:request:RecallService", alias="CHANNEL_RECALL_INTAKE")
    # Bus RPC wait for RecallService reply (collapse mirror & other recall steps). Independent of STEP_TIMEOUT_MS.
    recall_rpc_timeout_sec: float = Field(90.0, alias="RECALL_RPC_TIMEOUT_SEC")
    channel_agent_chain_intake: str = Field("orion:exec:request:AgentChainService", alias="CHANNEL_AGENT_CHAIN_INTAKE")
    channel_planner_intake: str = Field("orion:exec:request:PlannerReactService", alias="CHANNEL_PLANNER_INTAKE")
    channel_council_intake: str = Field("orion:agent-council:intake", alias="CHANNEL_COUNCIL_INTAKE")
    channel_council_reply_prefix: str = Field("orion:council:reply", alias="CHANNEL_COUNCIL_REPLY_PREFIX")
    channel_cognition_trace_pub: str = Field("orion:cognition:trace", alias="CHANNEL_COGNITION_TRACE_PUB")
    channel_metacog_trace_pub: str = Field("orion:metacog:trace", alias="CHANNEL_METACOG_TRACE_PUB")
    channel_dream_log: str = Field("orion:dream:log", alias="CHANNEL_DREAM_LOG")
    channel_collapse_sql_write: str = Field("orion:collapse:sql-write", alias="CHANNEL_COLLAPSE_SQL_WRITE")
    channel_collapse_intake: str = Field("orion:collapse:intake", alias="CHANNEL_COLLAPSE_INTAKE")
    channel_pad_rpc_request: str = Field("orion:pad:rpc:request", alias="CHANNEL_PAD_RPC_REQUEST")
    channel_state_request: str = Field("orion:state:request", alias="CHANNEL_STATE_REQUEST")
    channel_pad_rpc_reply_prefix: str = Field("orion:pad:rpc:reply", alias="CHANNEL_PAD_RPC_REPLY_PREFIX")
    channel_state_reply_prefix: str = Field("orion:state:reply", alias="CHANNEL_STATE_REPLY_PREFIX")
    channel_core_events: str = Field("orion:core:events", alias="CHANNEL_CORE_EVENTS")
    llm_chat_quick_max_tokens: int = Field(384, alias="LLM_CHAT_QUICK_MAX_TOKENS")
    llm_chat_general_max_tokens: int = Field(768, alias="LLM_CHAT_GENERAL_MAX_TOKENS")
    llm_chat_fallback_max_tokens: int = Field(512, alias="LLM_CHAT_FALLBACK_MAX_TOKENS")
    # dream_cycle / dream_synthesis only (does not affect chat_quick / chat_general budgets)
    llm_dream_max_tokens: int = Field(32768, alias="LLM_DREAM_MAX_TOKENS")
    atlas_metacog_profile_name: str | None = Field(None, alias="ATLAS_METACOG_PROFILE_NAME")

    diagnostic_mode: bool = Field(False, alias="DIAGNOSTIC_MODE")
    diagnostic_recall_timeout_sec: float = Field(5.0, alias="DIAGNOSTIC_RECALL_TIMEOUT_SEC")
    diagnostic_agent_timeout_sec: float = Field(15.0, alias="DIAGNOSTIC_AGENT_TIMEOUT_SEC")
    orion_verb_backdoor_enabled: bool = Field(False, alias="ORION_VERB_BACKDOOR_ENABLED")
    notify_url: str = Field("http://orion-notify:7140", alias="NOTIFY_URL")
    notify_api_token: str | None = Field(None, alias="NOTIFY_API_TOKEN")
    orion_tz: str = Field("America/Denver", alias="ORION_TZ")
    skills_command_timeout_sec: float = Field(8.0, alias="SKILLS_COMMAND_TIMEOUT_SEC")
    skills_mesh_ops_timeout_sec: float = Field(12.0, alias="SKILLS_MESH_OPS_TIMEOUT_SEC")
    docker_sock_path: str = Field("/var/run/docker.sock", alias="DOCKER_SOCK_PATH")
    tailscale_path: str = Field("tailscale", alias="ORION_ACTIONS_TAILSCALE_PATH")
    # Optional absolute path to nvidia-smi (host bind-mount or image-installed). When unset, skill resolves PATH.
    nvidia_smi_path: str | None = Field(None, alias="ORION_ACTIONS_NVIDIA_SMI_PATH")
    smartctl_path: str = Field("smartctl", alias="ORION_ACTIONS_SMARTCTL_PATH")
    nvme_path: str = Field("nvme", alias="ORION_ACTIONS_NVME_PATH")
    github_api_url: str = Field("https://api.github.com", alias="ORION_ACTIONS_GITHUB_API_URL")
    github_token: str | None = Field(None, alias="GITHUB_TOKEN")
    github_owner: str | None = Field(None, alias="ORION_ACTIONS_GITHUB_OWNER")
    github_repo: str | None = Field(None, alias="ORION_ACTIONS_GITHUB_REPO")
    mesh_default_lookback_days: int = Field(7, alias="ORION_ACTIONS_MESH_DEFAULT_LOOKBACK_DAYS")
    docker_prune_default_until: str = Field("72h", alias="ORION_ACTIONS_DOCKER_PRUNE_DEFAULT_UNTIL")
    docker_protected_labels: str = Field("orion.keep=true,keep=true,protected=true", alias="ORION_ACTIONS_DOCKER_PROTECTED_LABELS")
    skills_allow_mutating_runtime_housekeeping: bool = Field(False, alias="SKILLS_ALLOW_MUTATING_RUNTIME_HOUSEKEEPING")
    biometrics_service_url: str = Field("http://orion-athena-biometrics:8100", alias="BIOMETRICS_SERVICE_URL")
    biometrics_http_timeout_sec: float = Field(5.0, alias="BIOMETRICS_HTTP_TIMEOUT_SEC")
    endogenous_runtime_enabled: bool = Field(False, alias="ENDOGENOUS_RUNTIME_ENABLED")
    endogenous_runtime_surface_chat_reflective_enabled: bool = Field(
        False,
        alias="ENDOGENOUS_RUNTIME_SURFACE_CHAT_REFLECTIVE_ENABLED",
    )
    endogenous_runtime_surface_operator_enabled: bool = Field(
        False,
        alias="ENDOGENOUS_RUNTIME_SURFACE_OPERATOR_ENABLED",
    )
    endogenous_runtime_allowed_workflow_types: str = Field(
        "contradiction_review,concept_refinement,reflective_journal",
        alias="ENDOGENOUS_RUNTIME_ALLOWED_WORKFLOW_TYPES",
    )
    endogenous_runtime_allow_mentor_branch: bool = Field(
        False,
        alias="ENDOGENOUS_RUNTIME_ALLOW_MENTOR_BRANCH",
    )
    endogenous_runtime_sample_rate: float = Field(1.0, alias="ENDOGENOUS_RUNTIME_SAMPLE_RATE")
    endogenous_runtime_max_actions: int = Field(5, alias="ENDOGENOUS_RUNTIME_MAX_ACTIONS")
    endogenous_runtime_store_backend: str = Field("memory", alias="ENDOGENOUS_RUNTIME_STORE_BACKEND")
    endogenous_runtime_store_path: str = Field(
        "/tmp/orion_endogenous_runtime_records.jsonl",
        alias="ENDOGENOUS_RUNTIME_STORE_PATH",
    )
    endogenous_runtime_store_max_records: int = Field(2000, alias="ENDOGENOUS_RUNTIME_STORE_MAX_RECORDS")
    endogenous_runtime_sql_read_enabled: bool = Field(True, alias="ENDOGENOUS_RUNTIME_SQL_READ_ENABLED")
    endogenous_runtime_sql_database_url: str = Field(
        "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney",
        alias="ENDOGENOUS_RUNTIME_SQL_DATABASE_URL",
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
