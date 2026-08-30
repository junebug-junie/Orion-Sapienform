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

    # RPC-health snapshot publish (Step 3 of docs/superpowers/specs/2026-07-23-rpc-health-
    # signal-gateway-wiring-design.md). Off by default until live-verified per that spec's
    # acceptance checks; draining is free (in-memory) but this gates the new periodic publish.
    rpc_health_publish_enabled: bool = Field(False, alias="RPC_HEALTH_PUBLISH_ENABLED")
    rpc_health_publish_interval_sec: float = Field(30.0, alias="RPC_HEALTH_PUBLISH_INTERVAL_SEC")

    # Intake channel (hub or orch -> exec)
    channel_exec_request: str = Field("orion:cortex:exec:request", alias="CHANNEL_EXEC_REQUEST")
    exec_lane: str = Field("legacy", alias="EXEC_LANE")
    # ROADMAP A3. When true, a step the lane resolver already classifies as low-priority
    # (Orion's own initiative: introspect_spark, dream_cycle, dream_synthesis,
    # log_orion_metacognition, reverie_narrate) that would otherwise route to `quick` goes to
    # `quick_background` instead -- same upstream and model, but behind the gateway's
    # background admission gate, so Orion waits rather than Juniper.
    #
    # This is the roadmap's kill gate: set false to restore the previous behaviour exactly.
    # Nothing else needs redeploying and there is no migration.
    exec_autonomous_background_routing: bool = Field(
        True, alias="EXEC_AUTONOMOUS_BACKGROUND_ROUTING"
    )
    # Allow parallel plan execution on this lane (harness finalize must not queue behind chat_general).
    exec_concurrent_handlers: bool = Field(True, alias="EXEC_CONCURRENT_HANDLERS")

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
    chat_pcr_enabled: bool = Field(True, alias="CHAT_PCR_ENABLED")
    chat_pcr_post_stance_recall: bool = Field(True, alias="CHAT_PCR_POST_STANCE_RECALL")
    chat_pcr_skip_on_low_info: bool = Field(True, alias="CHAT_PCR_SKIP_ON_LOW_INFO")
    chat_pcr_quick_phase3: bool = Field(False, alias="CHAT_PCR_QUICK_PHASE3")
    chat_pcr_skip_max_novelty: float = Field(0.25, alias="CHAT_PCR_SKIP_MAX_NOVELTY")
    chat_pcr_skip_shift_novelty_floor: float = Field(0.35, alias="CHAT_PCR_SKIP_SHIFT_NOVELTY_FLOOR")
    memory_cognition_brain_belief_default: bool = Field(
        True, alias="MEMORY_COGNITION_BRAIN_BELIEF_DEFAULT"
    )
    orion_unified_grounding_enabled: bool = Field(True, alias="ORION_UNIFIED_GROUNDING_ENABLED")
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
    channel_council_intake: str = Field("orion:agent-council:intake", alias="CHANNEL_COUNCIL_INTAKE")
    channel_council_reply_prefix: str = Field("orion:council:reply", alias="CHANNEL_COUNCIL_REPLY_PREFIX")
    channel_cognition_trace_pub: str = Field("orion:cognition:trace", alias="CHANNEL_COGNITION_TRACE_PUB")
    channel_metacog_trace_pub: str = Field("orion:metacog:trace", alias="CHANNEL_METACOG_TRACE_PUB")
    channel_dream_log: str = Field("orion:dream:log", alias="CHANNEL_DREAM_LOG")
    channel_collapse_sql_write: str = Field("orion:collapse:sql-write", alias="CHANNEL_COLLAPSE_SQL_WRITE")
    channel_metacog_sql_write: str = Field("orion:metacog:sql-write", alias="CHANNEL_METACOG_SQL_WRITE")
    channel_collapse_intake: str = Field("orion:collapse:intake", alias="CHANNEL_COLLAPSE_INTAKE")
    channel_state_request: str = Field("orion:state:request", alias="CHANNEL_STATE_REQUEST")
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
    cortex_chat_return_logprobs: bool = Field(
        False,
        alias="CORTEX_CHAT_RETURN_LOGPROBS",
        description=(
            "Request per-token logprob/top1-margin telemetry on real chat-turn (route=chat) "
            "replies. Rides the existing OpenAI-compat gateway call (llm_backend.py's "
            "_execute_openai_chat) -- no endpoint switch, no response_format set, no separate "
            "probe request. Feeds chat_history_log.llm_* columns via "
            "_forward_llm_uncertainty_metadata's existing spark_meta merge. Distinct from "
            "CORTEX_METACOG_RETURN_LOGPROBS, which gates MetacogDraftService's own separate "
            "native-completion probe and never touches the user-facing reply."
        ),
    )
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
    cortex_metacog_draft_worker_ctx_char_budget: int = Field(
        8000,
        alias="CORTEX_METACOG_DRAFT_WORKER_CTX_CHAR_BUDGET",
        description="MetacogDraftService: trim metacog_biometrics_cue/spark_state_json and re-render when prompt exceeds worker ctx char budget.",
    )

    publish_cortex_exec_grammar: bool = Field(True, alias="PUBLISH_CORTEX_EXEC_GRAMMAR")
    grammar_event_channel: str = Field("orion:grammar:event", alias="GRAMMAR_EVENT_CHANNEL")

    # Per-call reasoning telemetry (metadata only; default OFF).
    publish_reasoning_telemetry: bool = Field(False, alias="PUBLISH_REASONING_TELEMETRY")
    channel_reasoning_call: str = Field(
        "orion:cognition:reasoning_call", alias="CHANNEL_REASONING_CALL"
    )

    diagnostic_mode: bool = Field(False, alias="DIAGNOSTIC_MODE")
    diagnostic_recall_timeout_sec: float = Field(5.0, alias="DIAGNOSTIC_RECALL_TIMEOUT_SEC")
    diagnostic_agent_timeout_sec: float = Field(15.0, alias="DIAGNOSTIC_AGENT_TIMEOUT_SEC")
    orion_verb_backdoor_enabled: bool = Field(False, alias="ORION_VERB_BACKDOOR_ENABLED")
    notify_url: str = Field("http://orion-notify:7140", alias="NOTIFY_URL")
    notify_api_token: str | None = Field(None, alias="NOTIFY_API_TOKEN")
    orion_tz: str = Field("America/Denver", alias="ORION_TZ")
    orion_situation_enabled: bool = Field(True, alias="ORION_SITUATION_ENABLED")
    orion_situation_ttl_seconds: int = Field(300, alias="ORION_SITUATION_TTL_SECONDS")
    # 2026-08-30, Juniper's explicit request: raised from 1200 to 7200 (6x)
    # now that the harness runs a large-context model. The
    # truncation/caution-survival logic in
    # orion.situational.context._build_prompt_fragment is unchanged, only
    # the ceiling moved.
    orion_situation_prompt_max_chars: int = Field(7200, alias="ORION_SITUATION_PROMPT_MAX_CHARS")
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
    # P4: the camera percept in the situation brief. Default OFF -- this puts
    # camera-derived content about a private home into the prompt, so it is
    # opt-in. See PerceptionContextV1's docstring for the exposed-field
    # contract (scene summary + age only; no frames, boxes, or identities).
    orion_situation_perception_enabled: bool = Field(
        False, alias="ORION_SITUATION_PERCEPTION_ENABLED"
    )
    # Older than this and the percept is withheld entirely, rendering as
    # "haven't seen anything recently" rather than as a current observation.
    # 900s tolerates a few missed windows at the measured ~5min event cadence.
    orion_situation_perception_max_age_seconds: int = Field(
        900, alias="ORION_SITUATION_PERCEPTION_MAX_AGE_SECONDS"
    )
    orion_situation_perception_stream_id: str = Field(
        "cam0", alias="ORION_SITUATION_PERCEPTION_STREAM_ID"
    )
    # 2026-08-26: how long Orion stays quiet about a fresh identity mismatch
    # after asking about it once for this camera stream -- see
    # orion/situational/identity_ask_cooldown.py's module docstring. 1200s
    # (20min) default.
    orion_situation_identity_ask_cooldown_seconds: int = Field(
        1200, alias="ORION_SITUATION_IDENTITY_ASK_COOLDOWN_SECONDS"
    )
    # 2026-08-29. Comma-separated; empty falls back to the single stream_id
    # above, so an unset key is exactly the previous behavior. Measured that
    # day: this replica read cam0 (the room camera, absent 70 minutes) while
    # carbon -- Juniper's laptop webcam -- had someone present, so the prompt
    # was narrating an empty room at a person sitting at her desk.
    orion_situation_perception_stream_ids: str = Field(
        "carbon,cam0", alias="ORION_SITUATION_PERCEPTION_STREAM_IDS"
    )
    # 2026-08-29: the cooldown for the OTHER ask reason -- "I have no fresh
    # confirmed read of who this is at all" (lid closed, camera off, empty
    # frame). 21600s (6h), far longer than the mismatch cooldown above,
    # because this one describes a condition that can hold all evening
    # rather than a transient mis-recognition.
    orion_situation_identity_ask_unconfirmed_cooldown_seconds: int = Field(
        21600, alias="ORION_SITUATION_IDENTITY_ASK_UNCONFIRMED_COOLDOWN_SECONDS"
    )
    # 2026-08-29: how fresh a presence row must be to count as a live read.
    # orion-vision-window rewrites it about every 5s while its camera is
    # alive, so >120s means that camera stopped reporting -- which is the
    # signal, not an edge case.
    orion_situation_identity_ask_max_presence_age_seconds: int = Field(
        120, alias="ORION_SITUATION_IDENTITY_ASK_MAX_PRESENCE_AGE_SECONDS"
    )
    # 2026-08-25: Juniper's facial+vocal affect read (orion-affectgpt-worker
    # via orion-juniper-affective-state), mirrored into a single Redis key
    # by orion/situational/juniper_affect_state.py. Default ON, unlike
    # perception: the capture is already an explicit Juniper action (Hub's
    # "Check now"/ambient toggle), so this is surfacing an already-consented
    # read, not new surveillance -- see AffectContextV1's docstring
    # (orion/schemas/situation.py) for the privacy contract (excerpt only,
    # never the verbatim transcript).
    orion_situation_affect_enabled: bool = Field(True, alias="ORION_SITUATION_AFFECT_ENABLED")
    # 300s: matches Hub's ambient-capture cadence (~5min). Tighter than
    # perception's 900s on purpose -- a stale mood read is more likely to
    # mislead a reply than a stale room description.
    orion_situation_affect_max_age_seconds: int = Field(
        300, alias="ORION_SITUATION_AFFECT_MAX_AGE_SECONDS"
    )
    # 2026-08-30: Orion's own open world-priors, folded into the situation
    # brief. Default ON -- no private-home content. NOTE: cortex-exec has no
    # established connection to Orion's `orion_worldview` FalkorDB graph
    # today (only orion-hub does, via HUB_CURIOSITY_GRAPH_*) -- there is
    # deliberately no `orion_situation_curiosity_graph_host` field here, so
    # this flag stays enabled but the provider degrades to "unconfigured"
    # for this process (see settings_from_runtime's own default of "" for
    # that field). Adding the graph connection to cortex-exec too is a
    # follow-up, not a gap in this patch's parity.
    orion_situation_curiosity_enabled: bool = Field(
        True, alias="ORION_SITUATION_CURIOSITY_ENABLED"
    )
    orion_situation_curiosity_ttl_seconds: int = Field(
        180, alias="ORION_SITUATION_CURIOSITY_TTL_SECONDS"
    )
    # 2026-08-30: Orion's most recent dream/reverie interpretations. Default
    # ON -- reuses the POSTGRES_URI this process already reads for
    # orion/situational/perception_reader.py, via
    # orion/situational/reverie_reader.py's identical DSN resolution.
    orion_situation_reverie_enabled: bool = Field(
        True, alias="ORION_SITUATION_REVERIE_ENABLED"
    )
    orion_situation_reverie_ttl_seconds: int = Field(
        180, alias="ORION_SITUATION_REVERIE_TTL_SECONDS"
    )
    orion_situation_lab_provider: str = Field("stub", alias="ORION_SITUATION_LAB_PROVIDER")
    # 2026-08-14: "does Orion know what model it's running on" (Juniper).
    # Default ON -- unlike perception, this carries no private-home content,
    # just a route name and a model id already visible in orion-llm-gateway's
    # own logs. Probes orion-llm-gateway's GET /routes (already cached there
    # 15s) for the `orion_situation_runtime_route`'s live model id; never
    # infers, degrades to unavailable on any failure. See RuntimeContextV1
    # in orion/schemas/situation.py.
    orion_situation_runtime_enabled: bool = Field(True, alias="ORION_SITUATION_RUNTIME_ENABLED")
    orion_situation_runtime_route: str = Field("chat", alias="ORION_SITUATION_RUNTIME_ROUTE")
    cortex_exec_llm_gateway_url: str = Field(
        "http://llm-gateway:8210", alias="CORTEX_EXEC_LLM_GATEWAY_URL"
    )
    # ROADMAP A5: read orion-llm-gateway's GET /admission and put "was my background thinking
    # made to wait, and for how long" into the metacog cue. Default ON, but the cue key is
    # simply absent whenever the gateway cannot be read -- unknown never renders as calm.
    # Set false to remove both the fetch and the key.
    cortex_exec_admission_cue_enabled: bool = Field(
        True, alias="CORTEX_EXEC_ADMISSION_CUE_ENABLED"
    )
    # 6h. Long enough that a lane which fills a few times a day is visible in a single pass,
    # short enough that yesterday's contention does not read as current.
    cortex_exec_admission_cue_window_s: float = Field(
        21600.0, alias="CORTEX_EXEC_ADMISSION_CUE_WINDOW_S"
    )
    cortex_exec_admission_cue_ttl_sec: float = Field(
        60.0, alias="CORTEX_EXEC_ADMISSION_CUE_TTL_SEC"
    )
    cortex_exec_admission_cue_timeout_sec: float = Field(
        2.0, alias="CORTEX_EXEC_ADMISSION_CUE_TIMEOUT_SEC"
    )
    orion_situation_runtime_probe_timeout_sec: float = Field(
        2.0, alias="ORION_SITUATION_RUNTIME_PROBE_TIMEOUT_SEC"
    )
    # Shorter than weather_ttl_seconds (600s): a model swap on the chat route
    # is an operator action Orion should reflect fairly promptly, and the
    # underlying orion-llm-gateway /routes read is already cached there 15s,
    # so this cache is a second, cheap layer on top, not the only one.
    orion_situation_runtime_ttl_seconds: int = Field(
        120, alias="ORION_SITUATION_RUNTIME_TTL_SECONDS"
    )
    orion_presence_session_ttl_seconds: int = Field(14400, alias="ORION_PRESENCE_SESSION_TTL_SECONDS")
    orion_presence_default_requestor: str = Field("Juniper", alias="ORION_PRESENCE_DEFAULT_REQUESTOR")
    orion_presence_persist_allowed: bool = Field(False, alias="ORION_PRESENCE_PERSIST_ALLOWED")
    skills_command_timeout_sec: float = Field(8.0, alias="SKILLS_COMMAND_TIMEOUT_SEC")
    skills_mesh_ops_timeout_sec: float = Field(12.0, alias="SKILLS_MESH_OPS_TIMEOUT_SEC")
    # Container-side path of the host Docker data-root bind mount (see this
    # service's docker-compose.yml). Where skills.runtime.builder_prune.v1
    # measures real filesystem usage.
    builder_prune_mount_path: str = Field("/hostfs/docker", alias="BUILDER_PRUNE_MOUNT_PATH")
    # 2026-08-12: the prune's own subprocess budget, 10 minutes.
    #
    # This verb previously used skills_mesh_ops_timeout_sec, which is 12
    # SECONDS live -- so a real `docker builder prune` over 142 GB / 15,506
    # cache entries would have been killed roughly 12 seconds in, partway
    # through deleting, on every single attempt. Every other skill sharing
    # that setting (ping, smartctl, nvidia-smi, mesh HTTP probes) genuinely
    # is a seconds-scale command and must keep the short budget, so this
    # needs its own key rather than a raise of the shared one.
    #
    # Must stay BELOW the verb's own timeout_ms (660s) and the maintenance
    # route's rpc_timeout_sec (720s), so a real overrun is attributed to the
    # docker command rather than surfacing as an opaque RPC timeout.
    builder_prune_timeout_sec: float = Field(600.0, alias="BUILDER_PRUNE_TIMEOUT_SEC")
    # Same footing as its sibling above. 2026-08-13 review finding: this was
    # hardcoded in verb_adapters.py, so changing it needed a code edit and a
    # redeploy -- while BUILDER_PRUNE_TIMEOUT_SEC exists precisely because the
    # shared 12s skills-mesh default was catastrophically wrong for a prune.
    image_prune_timeout_sec: float = Field(600.0, alias="IMAGE_PRUNE_TIMEOUT_SEC")
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
    skills_allow_docker_compose_bringup: bool = Field(False, alias="SKILLS_ALLOW_DOCKER_COMPOSE_BRINGUP")
    skills_docker_compose_bringup_timeout_sec: float = Field(900.0, alias="SKILLS_DOCKER_COMPOSE_BRINGUP_TIMEOUT_SEC")
    skills_docker_compose_bringup_health_poll_sec: float = Field(60.0, alias="SKILLS_DOCKER_COMPOSE_BRINGUP_HEALTH_POLL_SEC")
    biometrics_service_url: str = Field("http://orion-athena-biometrics:8100", alias="BIOMETRICS_SERVICE_URL")
    # PageIndex query provenance includes enriched journal trigger/stance/facet metadata.
    journal_pageindex_service_url: str = Field("http://orion-pageindex:8360", alias="JOURNAL_PAGEINDEX_SERVICE_URL")
    biometrics_http_timeout_sec: float = Field(5.0, alias="BIOMETRICS_HTTP_TIMEOUT_SEC")
    # skills.perception.look_at_camera.v1 -- reads orion-vision-window's already-
    # projected current-window snapshot (GET /api/vision-window/current). Does
    # NOT trigger a fresh capture; that would need a direct vision-host RPC, out
    # of scope for this first cut. Internal container port (8000), not the
    # host-mapped VISION_WINDOW_HTTP_PORT (8019 by convention).
    vision_window_service_url: str = Field(
        "http://orion-athena-vision-window:8000", alias="VISION_WINDOW_SERVICE_URL"
    )
    vision_window_http_timeout_sec: float = Field(5.0, alias="VISION_WINDOW_HTTP_TIMEOUT_SEC")
    # skills.perception.ask_camera.v1 -- the "direct vision-host RPC" the
    # comment above named as out of scope for look_at_camera's first cut.
    # Posts task_type=vqa straight to vision-host's own HTTP endpoint,
    # bypassing orion-vision-window/orion-vision-council entirely (real
    # on-demand VQA, not a read of the passive pipeline's last projection).
    # Internal container port (6600, per orion-vision-host's own Dockerfile
    # EXPOSE), not any host-mapped port.
    vision_host_service_url: str = Field(
        "http://orion-athena-vision-host:6600", alias="VISION_HOST_SERVICE_URL"
    )
    # Generous relative to vision_window's 5s default -- a cold vlm_vqa
    # profile lazy-loads its own model on first real request (see
    # services/orion-vision-host/app/runner.py::_run_vlm_vqa), which is
    # meaningfully slower than reading an already-warm projection.
    vision_host_http_timeout_sec: float = Field(30.0, alias="VISION_HOST_HTTP_TIMEOUT_SEC")
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
    repair_pressure_weights_v2_path: str = Field(
        "config/substrate/repair_pressure_weights.v2.yaml",
        alias="REPAIR_PRESSURE_WEIGHTS_V2_PATH",
    )
    repair_pressure_probe_route: str = Field("quick", alias="REPAIR_PRESSURE_PROBE_ROUTE")
    # Same-turn LLM novelty/salience judgment for the chat-scoped attention/
    # curiosity pipeline (app/current_turn_llm_signals.py), replacing the
    # deleted LegacyRegexSignalDetector's "any capitalized word" regex.
    # Quick-lane classification call, not a generation call -- see that
    # module's docstring for the full rationale.
    current_turn_signal_probe_route: str = Field("quick", alias="CURRENT_TURN_SIGNAL_PROBE_ROUTE")
    current_turn_signal_probe_timeout_sec: float = Field(3.0, alias="CURRENT_TURN_SIGNAL_PROBE_TIMEOUT_SEC")
    current_turn_signal_probe_max_tokens: int = Field(80, alias="CURRENT_TURN_SIGNAL_PROBE_MAX_TOKENS")
    enable_pre_turn_appraisal_handler: bool = Field(
        True,
        alias="ENABLE_PRE_TURN_APPRAISAL_HANDLER",
    )
    channel_pre_turn_appraisal_request: str = Field(
        "orion:cortex:pre_turn_appraisal:request",
        alias="CHANNEL_PRE_TURN_APPRAISAL_REQUEST",
    )
    channel_pre_turn_appraisal_result_prefix: str = Field(
        "orion:cortex:pre_turn_appraisal:result",
        alias="CHANNEL_PRE_TURN_APPRAISAL_RESULT_PREFIX",
    )
    world_pulse_stance_enabled: bool = Field(False, alias="WORLD_PULSE_STANCE_ENABLED")
    world_pulse_stance_max_topics: int = Field(5, alias="WORLD_PULSE_STANCE_MAX_TOPICS")
    world_pulse_stance_min_confidence: float = Field(0.65, alias="WORLD_PULSE_STANCE_MIN_CONFIDENCE")
    world_pulse_stance_max_age_hours: int = Field(36, alias="WORLD_PULSE_STANCE_MAX_AGE_HOURS")
    world_pulse_politics_stance_default: str = Field(
        "only_when_requested",
        alias="WORLD_PULSE_POLITICS_STANCE_DEFAULT",
    )
    # Fallback fetch when the caller's own request metadata carries no capsule:
    # pulls the latest already-built capsule from orion-world-pulse over its
    # documented HTTP contract (GET /api/world-pulse/latest -> .capsule).
    # Bounded timeout + short cache, fails open (None) on any error.
    world_pulse_base_url: str = Field(
        "http://orion-world-pulse:8628", alias="WORLD_PULSE_BASE_URL"
    )
    world_pulse_capsule_fetch_timeout_seconds: float = Field(
        2.0, alias="WORLD_PULSE_CAPSULE_FETCH_TIMEOUT_SECONDS"
    )
    world_pulse_capsule_cache_ttl_seconds: int = Field(
        300, alias="WORLD_PULSE_CAPSULE_CACHE_TTL_SECONDS"
    )
    health_http_port: int = Field(8070, alias="HEALTH_HTTP_PORT")

    # --- Embodiment (D) background emit + perception read-model (default-off, fail-open) ---
    embodiment_perception_cortex_enabled: bool = Field(
        False, alias="EMBODIMENT_PERCEPTION_CORTEX_ENABLED"
    )
    embodiment_d_background_enabled: bool = Field(False, alias="EMBODIMENT_D_BACKGROUND_ENABLED")
    embodiment_channel_perception: str = Field(
        "orion:embodiment:perception", alias="EMBODIMENT_CHANNEL_PERCEPTION"
    )
    embodiment_channel_intent: str = Field(
        "orion:embodiment:intent", alias="EMBODIMENT_CHANNEL_INTENT"
    )
    embodiment_background_interval_sec: float = Field(
        30.0, alias="EMBODIMENT_BACKGROUND_INTERVAL_SEC"
    )
    embodiment_orion_player_id: str = Field("", alias="AITOWN_ORION_PLAYER_ID")

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
