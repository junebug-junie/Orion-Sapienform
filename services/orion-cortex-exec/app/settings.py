from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

# Align os.environ with service .env for modules that read autonomy/GraphDB via os.getenv
# (chat_stance, graph_gate). override=False keeps compose/K8s-injected values.
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)


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
    exec_lane: str = Field("legacy", alias="EXEC_LANE")

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
    # Hub quick lane: cap RecallService bus wait so a slow mirror cannot stall Quick for minutes.
    chat_quick_recall_rpc_timeout_sec: float = Field(30.0, alias="CHAT_QUICK_RECALL_TIMEOUT_SEC")
    # Default recall profile for chat_quick when the client did not set profile_explicit (vector-light; see orion/recall/profiles).
    chat_quick_recall_profile: str = Field("assist.light.v1", alias="CHAT_QUICK_RECALL_PROFILE")
    chat_kids_story_recall_profile: str = Field("chat.story.kids.v1", alias="CHAT_KIDS_STORY_RECALL_PROFILE")
    channel_agent_chain_intake: str = Field("orion:exec:request:AgentChainService", alias="CHANNEL_AGENT_CHAIN_INTAKE")
    channel_context_exec_intake: str = Field(
        "orion:exec:request:ContextExecService",
        alias="CHANNEL_CONTEXT_EXEC_INTAKE",
    )
    channel_context_exec_reply_prefix: str = Field(
        "orion:exec:result:ContextExecService",
        alias="CHANNEL_CONTEXT_EXEC_REPLY_PREFIX",
    )
    context_exec_enabled: bool = Field(False, alias="CONTEXT_EXEC_ENABLED")
    context_exec_timeout_sec: float = Field(60.0, alias="CONTEXT_EXEC_TIMEOUT_SEC")
    context_exec_depth2_default: bool = Field(False, alias="CONTEXT_EXEC_DEPTH2_DEFAULT")
    context_exec_legacy_fallback: bool = Field(True, alias="CONTEXT_EXEC_LEGACY_FALLBACK")
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
    llm_memory_graph_suggest_max_tokens: int = Field(
        4096,
        alias="LLM_MEMORY_GRAPH_SUGGEST_MAX_TOKENS",
        description="Completion budget for memory_graph_suggest JSON drafts (must exceed minimal JSON).",
    )
    # dream_cycle / dream_synthesis only (does not affect chat_quick / chat_general budgets)
    llm_dream_max_tokens: int = Field(32768, alias="LLM_DREAM_MAX_TOKENS")
    atlas_metacog_profile_name: str | None = Field(None, alias="ATLAS_METACOG_PROFILE_NAME")
    cortex_metacog_return_logprobs: bool = Field(False, alias="CORTEX_METACOG_RETURN_LOGPROBS")
    cortex_metacog_logprob_probe_mode: str = Field(
        default="",
        alias="CORTEX_METACOG_LOGPROB_PROBE_MODE",
        description="Pass-2 uncertainty probe mode. Only native_completion is supported (llama.cpp /completion). Other values skip pass 2.",
    )
    cortex_metacog_uncertainty_probe_enabled: bool = Field(
        True,
        alias="CORTEX_METACOG_UNCERTAINTY_PROBE_ENABLED",
        description="When CORTEX_METACOG_RETURN_LOGPROBS: run pass-2 native probe after successful draft parse.",
    )
    daily_metacog_prompt_max_chars: int = Field(
        8192,
        alias="CORTEX_DAILY_METACOG_PROMPT_MAX_CHARS",
        description="Fail daily_metacog_v1 LLM step before call when rendered prompt exceeds this char budget.",
    )
    cortex_metacog_draft_prompt_max_chars: int = Field(
        16384,
        alias="CORTEX_METACOG_DRAFT_PROMPT_MAX_CHARS",
        description="Lane A log_orion_metacognition: skip MetacogDraftService LLM when rendered prompt exceeds this char budget.",
    )
    cortex_metacog_enrich_prompt_max_chars: int = Field(
        20480,
        alias="CORTEX_METACOG_ENRICH_PROMPT_MAX_CHARS",
        description="Lane A log_orion_metacognition: skip MetacogEnrichService LLM when rendered prompt exceeds this char budget.",
    )
    cortex_metacog_draft_worker_ctx_char_budget: int = Field(
        8000,
        alias="CORTEX_METACOG_DRAFT_WORKER_CTX_CHAR_BUDGET",
        description="MetacogDraftService: trim metacog_biometrics_cue/spark_state_json and re-render when prompt exceeds worker ctx char budget.",
    )
    cortex_metacog_enrich_worker_ctx_char_budget: int = Field(
        12000,
        alias="CORTEX_METACOG_ENRICH_WORKER_CTX_CHAR_BUDGET",
        description="MetacogEnrichService: trim metacog_biometrics_cue then spark_state_json when prompt exceeds worker ctx char budget.",
    )

    publish_cortex_exec_grammar: bool = Field(False, alias="PUBLISH_CORTEX_EXEC_GRAMMAR")
    grammar_event_channel: str = Field("orion:grammar:event", alias="GRAMMAR_EVENT_CHANNEL")

    diagnostic_mode: bool = Field(False, alias="DIAGNOSTIC_MODE")
    diagnostic_recall_timeout_sec: float = Field(5.0, alias="DIAGNOSTIC_RECALL_TIMEOUT_SEC")
    diagnostic_agent_timeout_sec: float = Field(15.0, alias="DIAGNOSTIC_AGENT_TIMEOUT_SEC")
    orion_verb_backdoor_enabled: bool = Field(False, alias="ORION_VERB_BACKDOOR_ENABLED")
    notify_url: str = Field("http://orion-notify:7140", alias="NOTIFY_URL")
    notify_api_token: str | None = Field(None, alias="NOTIFY_API_TOKEN")
    orion_tz: str = Field("America/Denver", alias="ORION_TZ")
    orion_situation_enabled: bool = Field(True, alias="ORION_SITUATION_ENABLED")
    orion_situation_ttl_seconds: int = Field(300, alias="ORION_SITUATION_TTL_SECONDS")
    orion_situation_prompt_max_chars: int = Field(1200, alias="ORION_SITUATION_PROMPT_MAX_CHARS")
    orion_situation_timezone: str = Field("America/Denver", alias="ORION_SITUATION_TIMEZONE")
    orion_situation_location_label: str = Field("Unknown", alias="ORION_SITUATION_LOCATION_LABEL")
    orion_situation_locality: str | None = Field(None, alias="ORION_SITUATION_LOCALITY")
    orion_situation_region: str | None = Field(None, alias="ORION_SITUATION_REGION")
    orion_situation_country: str | None = Field(None, alias="ORION_SITUATION_COUNTRY")
    orion_situation_location_precision: str = Field("city", alias="ORION_SITUATION_LOCATION_PRECISION")
    orion_situation_weather_enabled: bool = Field(True, alias="ORION_SITUATION_WEATHER_ENABLED")
    orion_situation_weather_provider: str = Field("stub", alias="ORION_SITUATION_WEATHER_PROVIDER")
    orion_situation_weather_lat: float | None = Field(None, alias="ORION_SITUATION_WEATHER_LAT")
    orion_situation_weather_lon: float | None = Field(None, alias="ORION_SITUATION_WEATHER_LON")
    orion_situation_weather_ttl_seconds: int = Field(600, alias="ORION_SITUATION_WEATHER_TTL_SECONDS")
    orion_situation_umbrella_precip_prob_threshold: int = Field(40, alias="ORION_SITUATION_UMBRELLA_PRECIP_PROB_THRESHOLD")
    orion_situation_jacket_temp_f_threshold: int = Field(55, alias="ORION_SITUATION_JACKET_TEMP_F_THRESHOLD")
    orion_situation_high_wind_mph_threshold: int = Field(25, alias="ORION_SITUATION_HIGH_WIND_MPH_THRESHOLD")
    orion_situation_hot_car_temp_f_threshold: int = Field(80, alias="ORION_SITUATION_HOT_CAR_TEMP_F_THRESHOLD")
    orion_situation_agenda_enabled: bool = Field(False, alias="ORION_SITUATION_AGENDA_ENABLED")
    orion_situation_lab_context_enabled: bool = Field(True, alias="ORION_SITUATION_LAB_CONTEXT_ENABLED")
    orion_situation_lab_provider: str = Field("stub", alias="ORION_SITUATION_LAB_PROVIDER")
    orion_presence_session_ttl_seconds: int = Field(14400, alias="ORION_PRESENCE_SESSION_TTL_SECONDS")
    orion_presence_default_requestor: str = Field("Juniper", alias="ORION_PRESENCE_DEFAULT_REQUESTOR")
    orion_presence_persist_allowed: bool = Field(False, alias="ORION_PRESENCE_PERSIST_ALLOWED")
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
    skills_allow_mesh_service_scripts: bool = Field(False, alias="SKILLS_ALLOW_MESH_SERVICE_SCRIPTS")
    skills_mesh_service_script_timeout_sec: float = Field(900.0, alias="SKILLS_MESH_SERVICE_SCRIPT_TIMEOUT_SEC")
    biometrics_service_url: str = Field("http://orion-athena-biometrics:8100", alias="BIOMETRICS_SERVICE_URL")
    # PageIndex query provenance includes enriched journal trigger/stance/facet metadata.
    journal_pageindex_service_url: str = Field("http://orion-pageindex:8360", alias="JOURNAL_PAGEINDEX_SERVICE_URL")
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
    # Global RDF / substrate semantic graph (os.getenv in orion.substrate.graphdb_store; mirrored for Hub settings parity)
    graph_backend: str = Field("fuseki", alias="GRAPH_BACKEND")
    rdf_store_backend: str = Field("fuseki", alias="RDF_STORE_BACKEND")
    rdf_store_base_url: str = Field("", alias="RDF_STORE_BASE_URL")
    rdf_store_dataset: str = Field("orion", alias="RDF_STORE_DATASET")
    rdf_store_query_url: str = Field("", alias="RDF_STORE_QUERY_URL")
    rdf_store_graph_store_url: str = Field("", alias="RDF_STORE_GRAPH_STORE_URL")
    rdf_store_update_url: str = Field("", alias="RDF_STORE_UPDATE_URL")
    rdf_store_user: str = Field("", alias="RDF_STORE_USER")
    rdf_store_pass: str = Field("", alias="RDF_STORE_PASS")
    fuseki_user: str = Field("", alias="FUSEKI_USER")
    fuseki_pass: str = Field("", alias="FUSEKI_PASS")
    gdb_client_enabled: bool = Field(False, alias="GDB_CLIENT_ENABLED")
    substrate_store_backend: str = Field("sparql", alias="SUBSTRATE_STORE_BACKEND")
    substrate_graph_query_url: str = Field("", alias="SUBSTRATE_GRAPH_QUERY_URL")
    substrate_graph_update_url: str = Field("", alias="SUBSTRATE_GRAPH_UPDATE_URL")
    substrate_graph_uri: str = Field("", alias="SUBSTRATE_GRAPH_URI")
    substrate_graph_timeout_sec: float = Field(5.0, alias="SUBSTRATE_GRAPH_TIMEOUT_SEC")
    substrate_graph_user: str = Field("", alias="SUBSTRATE_GRAPH_USER")
    substrate_graph_pass: str = Field("", alias="SUBSTRATE_GRAPH_PASS")
    substrate_graphdb_endpoint: str = Field("", alias="SUBSTRATE_GRAPHDB_ENDPOINT")
    substrate_graphdb_graph_uri: str = Field("", alias="SUBSTRATE_GRAPHDB_GRAPH_URI")
    substrate_graphdb_timeout_sec: float = Field(5.0, alias="SUBSTRATE_GRAPHDB_TIMEOUT_SEC")
    substrate_graphdb_user: str = Field("", alias="SUBSTRATE_GRAPHDB_USER")
    substrate_graphdb_pass: str = Field("", alias="SUBSTRATE_GRAPHDB_PASS")
    # Autonomy GraphDB reads (chat stance / unified-beliefs adapter): see docs/architecture/rdf_store_v1_cutover.md
    autonomy_graph_backend: str = Field("auto", alias="AUTONOMY_GRAPH_BACKEND")
    autonomy_quick_graph_timeout_sec: float = Field(3.0, alias="AUTONOMY_QUICK_GRAPH_TIMEOUT_SEC")
    autonomy_quick_graph_subjects: str = Field("orion", alias="AUTONOMY_QUICK_GRAPH_SUBJECTS")
    autonomy_quick_graph_subqueries: str = Field("identity", alias="AUTONOMY_QUICK_GRAPH_SUBQUERIES")
    repair_pressure_speech_wiring_enabled: bool = Field(
        True,
        alias="ENABLE_REPAIR_PRESSURE_SPEECH_WIRING",
    )
    world_pulse_stance_enabled: bool = Field(False, alias="WORLD_PULSE_STANCE_ENABLED")
    world_pulse_stance_max_topics: int = Field(5, alias="WORLD_PULSE_STANCE_MAX_TOPICS")
    world_pulse_stance_min_confidence: float = Field(0.65, alias="WORLD_PULSE_STANCE_MIN_CONFIDENCE")
    world_pulse_stance_max_age_hours: int = Field(36, alias="WORLD_PULSE_STANCE_MAX_AGE_HOURS")
    world_pulse_politics_stance_default: str = Field(
        "only_when_requested",
        alias="WORLD_PULSE_POLITICS_STANCE_DEFAULT",
    )
    health_http_port: int = Field(8070, alias="HEALTH_HTTP_PORT")

    @field_validator("orion_situation_weather_lat", "orion_situation_weather_lon", mode="before")
    @classmethod
    def _blank_env_float_to_none(cls, value: object) -> object:
        if value is None or value == "":
            return None
        return value

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
