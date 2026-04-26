from __future__ import annotations

import json
import logging
import os
import ipaddress
import threading
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from uuid import uuid4
from typing import Annotated, Optional, Any, List, Dict, Tuple, Literal
from urllib.parse import urlparse
import aiohttp

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field
import requests

from .settings import settings
from .session import ensure_session
from .chat_history import (
    build_chat_history_envelope,
    build_chat_response_feedback_envelope,
    build_chat_turn_envelope,
    publish_chat_history,
    publish_chat_response_feedback,
    publish_chat_turn,
    publish_social_room_turn,
    select_reasoning_trace_for_history,
)
from .library import scan_cognition_library
from .trace_payloads import extract_agent_trace_payload
from .autonomy_payloads import extract_autonomy_payload
from .workflow_payloads import extract_workflow_payload
from .cortex_chat_display import hub_effective_chat_text
from .cortex_request_builder import build_chat_request, build_continuity_messages, validate_single_verb_override
from .mutation_cognition_context import build_mutation_cognition_context
from .autonomy_constitution import (
    COGNITIVE_LIVE_APPLY_ENABLED,
    PRODUCTION_RECALL_MODE,
    RECALL_LIVE_APPLY_ENABLED,
    constitution_summary,
    load_autonomy_constitution,
    validate_autonomy_constitution,
)
from .social_room import is_social_room_payload, social_room_client_meta
from .service_logs import collect_service_inventory
from orion.cognition.verb_activation import build_verb_list
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.collapse_mirror import CollapseMirrorEntry
from orion.schemas.cortex.contracts import CortexChatRequest, CortexChatResult
from orion.schemas.metacognitive_trace import MetacognitiveTraceV1
from orion.schemas.workflow_execution import WorkflowScheduleManageRequestV1, WorkflowScheduleManageResponseV1
from orion.schemas.chat_response_feedback import ChatResponseFeedbackV1, build_feedback_category_options
from orion.schemas.notify import (
    ChatAttentionAck,
    ChatMessageReceipt,
    NotificationPreferencesUpdate,
    PreferenceResolutionRequest,
    RecipientProfileUpdate,
)
from orion.schemas.situation import (
    ConversationPhaseContextV1,
    PlaceContextV1,
    PresenceContextV1,
    SituationBriefV1,
    TimeContextV1,
)

from orion.core.schemas.substrate_review_queue import GraphReviewCyclePolicyV1
from orion.core.schemas.substrate_review_runtime import GraphReviewRuntimeRequestV1
from orion.core.schemas.substrate_review_telemetry import GraphReviewTelemetryQueryV1, GraphReviewTelemetryRecordV1
from orion.core.schemas.frontier_expansion import FrontierTargetZoneV1
from orion.core.schemas.substrate_policy_adoption import (
    SubstratePolicyAdoptionRequestV1,
    SubstratePolicyOverridesV1,
    SubstratePolicyRolloutScopeV1,
)
from orion.core.schemas.substrate_policy_comparison import SubstratePolicyComparisonRequestV1
from orion.core.schemas.substrate_review_runtime import GraphReviewRuntimeSurfaceV1
from orion.core.schemas.substrate_mutation import (
    CognitiveProposalDraftV1,
    CognitiveProposalReviewV1,
    CognitiveStanceNoteV1,
    RecallCanaryJudgmentRecordV1,
    RecallCanaryReviewArtifactV1,
    RecallCanaryRunV1,
    MutationPressureEvidenceV1,
    MutationPressureV1,
    RecallProductionCandidateReviewV1,
    RecallShadowEvalRunV1,
    RecallStrategyProfileV1,
)
from orion.substrate import build_substrate_policy_store_from_env, build_substrate_store_from_env
from orion.substrate.consolidation import GraphConsolidationEvaluator
from orion.substrate.policy_comparison import SubstratePolicyComparisonService
from orion.substrate.review_queue import GraphReviewQueue
from orion.substrate.review_bootstrap import GraphReviewBootstrapper
from orion.substrate.review_runtime import GraphReviewRuntimeExecutor
from orion.substrate.review_schedule import GraphReviewScheduler
from orion.substrate.review_telemetry import GraphReviewCalibrationAnalyzer, GraphReviewTelemetryRecorder
from orion.substrate.mutation_apply import PatchApplier
from orion.substrate.mutation_decision import DecisionEngine
from orion.substrate.mutation_detectors import MutationDetectors
from orion.substrate.mutation_monitor import PostAdoptionMonitor
from orion.substrate.mutation_pressure import PressureAccumulator
from orion.substrate.mutation_proposals import ProposalFactory
from orion.substrate.mutation_queue import SubstrateMutationStore
from orion.substrate.mutation_scoring import ClassSpecificScorer
from orion.substrate.mutation_trials import ReplayCorpusRegistry, SubstrateTrialRunner
from orion.substrate.mutation_worker import AdaptationCycleBudget, SubstrateAdaptationWorker
from orion.substrate.mutation_control_surface import inspect_chat_reflective_lane_threshold
from orion.substrate.recall_strategy_readiness import (
    compute_recall_strategy_readiness,
    default_eval_corpus_total_cases,
    readiness_from_telemetry_records,
)

logger = logging.getLogger("orion-hub.api")
PROCESS_STARTED_AT_UTC = datetime.now(timezone.utc)

router = APIRouter()


def _hub_uses_host_network_mode() -> bool:
    mode = str(os.getenv("HUB_DOCKER_NETWORK_MODE", "")).strip().lower()
    return mode == "host"


def _postgres_url_host(postgres_url: str) -> str:
    try:
        return str(urlparse(postgres_url).hostname or "").strip()
    except Exception:
        return ""


def _looks_like_docker_service_hostname(postgres_url: str) -> bool:
    host = _postgres_url_host(postgres_url)
    if not host:
        return False
    if host in {"localhost", "host.docker.internal"}:
        return False
    try:
        ipaddress.ip_address(host)
        return False
    except ValueError:
        pass
    return "." not in host


def _resolve_control_plane_postgres_url() -> Optional[str]:
    control_plane_url = str(os.getenv("SUBSTRATE_CONTROL_PLANE_POSTGRES_URL", "")).strip()
    policy_url = str(os.getenv("SUBSTRATE_POLICY_POSTGRES_URL", "")).strip()
    database_url = str(os.getenv("DATABASE_URL", "")).strip()
    resolved = control_plane_url or policy_url or database_url or None
    if resolved and _hub_uses_host_network_mode() and _looks_like_docker_service_hostname(resolved):
        logger.warning(
            "Hub is configured for host networking (HUB_DOCKER_NETWORK_MODE=host), "
            "but the resolved Postgres hostname '%s' looks like a Docker service name. "
            "Use a host-reachable address (for example 127.0.0.1, LAN IP, or Tailscale IP).",
            _postgres_url_host(resolved),
        )
    return resolved


SUBSTRATE_REVIEW_QUEUE_STORE = GraphReviewQueue(
    max_items=200,
    sql_db_path=str(os.getenv("SUBSTRATE_REVIEW_QUEUE_SQL_DB_PATH", "")).strip() or None,
    postgres_url=_resolve_control_plane_postgres_url(),
)
SUBSTRATE_REVIEW_TELEMETRY_STORE = GraphReviewTelemetryRecorder(
    max_records=2000,
    sql_db_path=str(os.getenv("SUBSTRATE_REVIEW_TELEMETRY_SQL_DB_PATH", "")).strip() or None,
    postgres_url=_resolve_control_plane_postgres_url(),
)
SUBSTRATE_SEMANTIC_STORE = build_substrate_store_from_env()
SUBSTRATE_POLICY_STORE = build_substrate_policy_store_from_env()
SUBSTRATE_POLICY_COMPARISON = SubstratePolicyComparisonService(
    policy_store=SUBSTRATE_POLICY_STORE,
    telemetry_recorder=SUBSTRATE_REVIEW_TELEMETRY_STORE,
)
SUBSTRATE_REVIEW_RUNTIME_EXECUTOR = GraphReviewRuntimeExecutor(
    queue=SUBSTRATE_REVIEW_QUEUE_STORE,
    consolidation_evaluator=GraphConsolidationEvaluator(store=SUBSTRATE_SEMANTIC_STORE),
    scheduler=GraphReviewScheduler(
        queue=SUBSTRATE_REVIEW_QUEUE_STORE,
        policy_profiles=SUBSTRATE_POLICY_STORE,
    ),
    telemetry_recorder=SUBSTRATE_REVIEW_TELEMETRY_STORE,
    policy_profiles=SUBSTRATE_POLICY_STORE,
)
SUBSTRATE_REVIEW_BOOTSTRAPPER = GraphReviewBootstrapper(
    scheduler=SUBSTRATE_REVIEW_RUNTIME_EXECUTOR.scheduler,
    semantic_store=SUBSTRATE_SEMANTIC_STORE,
)
SUBSTRATE_MUTATION_STORE = SubstrateMutationStore(
    sql_db_path=str(os.getenv("SUBSTRATE_MUTATION_SQL_DB_PATH", "")).strip() or None,
    postgres_url=_resolve_control_plane_postgres_url(),
)
SUBSTRATE_MUTATION_SURFACES: Dict[str, Dict[str, Any]] = {}
SUBSTRATE_AUTONOMY_LOCAL_CYCLE_LOCK = threading.Lock()


def _thought_debug_enabled() -> bool:
    return str(os.getenv("DEBUG_THOUGHT_PROCESS", "false")).strip().lower() in {"1", "true", "yes", "on"}


def _debug_len(value: Any) -> int:
    return len(str(value or ""))


def _debug_snippet(value: Any, max_len: int = 200) -> str:
    text = str(value or "").strip()
    if len(text) <= max_len:
        return text
    return f"{text[:max_len]}…"


def _coerce_metacog_trace(trace: Any) -> Optional[MetacognitiveTraceV1]:
    if isinstance(trace, MetacognitiveTraceV1):
        return trace
    if isinstance(trace, dict):
        try:
            return MetacognitiveTraceV1.model_validate(trace)
        except Exception:
            return None
    return None

class AttentionAckRequest(BaseModel):
    ack_type: str = Field("seen")
    note: Optional[str] = None


class ChatMessageReceiptRequest(BaseModel):
    session_id: str
    receipt_type: str = Field("opened")


class PreferencesResolveProxyRequest(BaseModel):
    recipient_group: str
    event_kind: str
    severity: str


class WorkflowScheduleUpdateRequest(BaseModel):
    run_at_utc: Optional[str] = None
    cadence: Optional[str] = None
    day_of_week: Optional[int] = None
    hour_local: Optional[int] = None
    minute_local: Optional[int] = None
    timezone: Optional[str] = None
    notify_on: Optional[str] = None
    expected_revision: Optional[int] = None


class SubstrateReviewExecuteRequest(BaseModel):
    explicit_queue_item_id: Optional[str] = None


class SubstrateReviewSmokeCheckRequest(BaseModel):
    limit: int = Field(default=10, ge=1, le=100)


class SubstrateReviewBootstrapRequest(BaseModel):
    limit: int = Field(default=12, ge=1, le=32)


class SubstrateMutationExecuteRequest(BaseModel):
    dry_run: bool = True
    max_signals: int = Field(default=32, ge=1, le=256)
    max_proposals: int = Field(default=8, ge=1, le=32)
    max_trials: int = Field(default=8, ge=1, le=32)
    apply_enabled: bool = False
    telemetry: list[Dict[str, Any]] = Field(default_factory=list)
    class_metrics: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    post_adoption_delta_by_target_surface: Dict[str, float] = Field(default_factory=dict)


class RecallEvalSuiteRecordRequest(BaseModel):
    """Operator-triggered ingest of recall_eval suite rows into mutation review telemetry (proposal-only path)."""

    model_config = ConfigDict(extra="forbid")
    rows: list[dict[str, Any]] = Field(default_factory=list, max_length=64)
    suite_run_id: str | None = None


class RecallProposalStageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    override: bool = False
    operator_rationale: str = ""
    created_by: str = "operator"


class RecallShadowActivateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operator_rationale: str = ""


class RecallShadowEvaluateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    corpus_limit: int | None = Field(default=None, ge=1, le=512)
    case_ids: list[str] = Field(default_factory=list, max_length=128)
    dry_run: bool = True
    record_pressure_events: bool = True
    operator_rationale: str = ""
    eval_rows: list[dict[str, Any]] = Field(default_factory=list, max_length=512)


class RecallProductionCandidateReviewCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    override: bool = False
    operator_rationale: str = ""
    created_by: str = "operator"
    operator_checklist: dict[str, Any] = Field(default_factory=dict)


class RecallCanaryQueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_text: str = Field(min_length=1, max_length=4096)
    profile_id: str | None = None
    session_id: str | None = None
    node_id: str | None = None


class RecallCanaryJudgmentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    judgment: Literal["v2_better", "v1_better", "tie", "both_bad", "inconclusive"]
    failure_modes: list[
        Literal[
            "missing_exact_anchor",
            "irrelevant_semantic_neighbor",
            "stale_memory",
            "unsupported_memory_claim",
            "insufficient_context",
            "wrong_project",
            "wrong_timeframe",
            "empty_result",
            "overbroad_result",
        ]
    ] = Field(default_factory=list, max_length=32)
    operator_note: str = ""
    should_emit_pressure: bool = True
    should_mark_review_candidate: bool = False


class RecallCanaryReviewArtifactRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    review_type: str = "production_candidate_evidence"
    operator_note: str = ""
    include_comparison_summary: bool = True
    include_operator_judgment: bool = True


class CognitiveProposalReviewRequest(BaseModel):
    state: Literal["pending_review", "accepted_as_draft", "rejected", "superseded", "archived"]
    reviewer: str = "operator"
    rationale: str = ""
    notes: list[str] = Field(default_factory=list, max_length=32)


class CognitiveProposalReviewActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["accept_as_draft", "reject", "archive", "supersede"]
    reviewer: str = "operator"
    rationale: str = ""
    review_labels: list[str] = Field(default_factory=list, max_length=32)
    supersedes_proposal_id: str | None = None
    should_emit_pressure: bool = False
    create_stance_note: bool = False
    stance_note: str = ""
    stance_summary: str = ""
    stance_visibility: Literal["metacog_only", "stance_and_metacog"] = "metacog_only"
    stance_ttl_turns: int = Field(default=20, ge=1, le=200)


class CognitiveDraftArchiveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reviewer: str = "operator"
    rationale: str = ""


class CognitiveCreateStanceNoteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reviewer: str = "operator"
    rationale: str = ""
    note: str = ""
    summary: str = ""
    visibility: Literal["metacog_only", "stance_and_metacog"] = "metacog_only"
    ttl_turns: int = Field(default=20, ge=1, le=200)


class SubstratePolicyAdoptHubRequest(BaseModel):
    """Hub-facing adopt payload; defaults match Substrate Inspector operator_review cycles.

    Empty ``target_zones`` applies the profile to every review zone (see
    ``SubstratePolicyProfileStore._matches_scope``); a single zone caused
    autonomy_graph/world_ontology runs to stay on baseline while Postgres
    still showed an active profile for concept_graph only.
    """

    model_config = ConfigDict(extra="forbid")

    activate_now: bool = True
    operator_id: Optional[str] = None
    rationale: str = "substrate-inspector"
    invocation_surfaces: list[GraphReviewRuntimeSurfaceV1] = Field(default_factory=lambda: ["operator_review"])
    target_zones: list[FrontierTargetZoneV1] = Field(default_factory=list)
    operator_only: bool = False
    policy_overrides: SubstratePolicyOverridesV1 = Field(default_factory=SubstratePolicyOverridesV1)


async def _execute_workflow_schedule_management(*, session_id: str, user_id: Optional[str], request: WorkflowScheduleManageRequestV1) -> Dict[str, Any]:
    from .main import cortex_client

    if cortex_client is None:
        raise RuntimeError("Cortex client unavailable")

    corr_id = str(uuid4())
    chat_req = CortexChatRequest(
        prompt="workflow schedule management",
        mode="brain",
        route_intent="none",
        session_id=session_id,
        user_id=user_id,
        trace_id=corr_id,
        metadata={"workflow_schedule_management": request.model_dump(mode="json")},
    )
    resp: CortexChatResult = await cortex_client.chat(chat_req, correlation_id=corr_id)
    metadata = resp.cortex_result.metadata if isinstance(resp.cortex_result.metadata, dict) else {}
    payload = metadata.get("workflow_schedule_management") if isinstance(metadata.get("workflow_schedule_management"), dict) else {}
    if payload:
        try:
            return WorkflowScheduleManageResponseV1.model_validate(payload).model_dump(mode="json")
        except Exception:
            return payload
    return {
        "ok": bool(resp.cortex_result.ok),
        "operation": request.operation,
        "request_id": request.request_id,
        "message": resp.final_text or "No schedule response payload.",
        "schedules": [],
        "history": [],
        "ambiguous": False,
        "error_code": "invalid_management_payload",
        "error_details": {},
    }


async def _proxy_request(request: Request, base_url: str, path: str) -> Response:
    url = f"{base_url.rstrip('/')}/{path}"
    headers = {key: value for key, value in request.headers.items() if key.lower() not in {"host", "content-length"}}
    body = await request.body()
    timeout = aiohttp.ClientTimeout(total=settings.TIMEOUT_SEC)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.request(
            request.method,
            url,
            params=dict(request.query_params),
            data=body if body else None,
            headers=headers,
        ) as response:
            payload = await response.read()
            content_type = response.headers.get("content-type", "application/json")
            return Response(content=payload, status_code=response.status, media_type=content_type)


async def _fetch_landing_pad(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    base_url = settings.LANDING_PAD_URL.rstrip("/")
    url = f"{base_url}{path}"
    timeout = aiohttp.ClientTimeout(total=settings.LANDING_PAD_TIMEOUT_SEC)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url, params=params) as response:
            response.raise_for_status()
            return await response.json()


async def _fetch_social_memory(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    base_url = settings.SOCIAL_MEMORY_BASE_URL.rstrip("/")
    url = f"{base_url}{path}"
    timeout = aiohttp.ClientTimeout(total=settings.TIMEOUT_SEC)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url, params=params) as response:
            response.raise_for_status()
            return await response.json()


def _normalize_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _log_hub_route_decision(
    *,
    corr_id: str,
    session_id: str,
    route_debug: Dict[str, Any],
    user_prompt: str,
) -> None:
    emitted_mode = route_debug.get("mode")
    emitted_verb = route_debug.get("verb")
    effective_verb = emitted_verb
    if not effective_verb and emitted_mode not in {"agent", "council"}:
        effective_verb = "chat_general"
    summary = {
        "corr_id": corr_id,
        "session_id": session_id,
        "selected_ui_route": route_debug.get("selected_ui_route"),
        "emitted_mode": emitted_mode,
        "emitted_verb": emitted_verb,
        "effective_verb": effective_verb,
        "emitted_options": route_debug.get("options") or {},
        "packs": route_debug.get("packs") or [],
        "force_agent_chain": bool(route_debug.get("force_agent_chain")),
        "supervised": bool(route_debug.get("supervised")),
        "diagnostic": bool(route_debug.get("diagnostic")),
        "last_user_head": (user_prompt or "")[:120],
    }
    logger.info("hub_route_egress %s", json.dumps(summary, sort_keys=True, default=str))


def _rec_tape_req(
    *,
    corr_id: str,
    session_id: str,
    mode: str,
    use_recall: bool,
    recall_profile: Optional[str],
    user_head: str,
    no_write: bool,
) -> None:
    if not settings.HUB_DEBUG_RECALL:
        return
    logger.info(
        "REC_TAPE REQ corr_id=%s sid=%s mode=%s recall=%s profile=%s user_head=%r no_write=%s",
        corr_id,
        session_id,
        mode,
        use_recall,
        recall_profile,
        user_head,
        no_write,
    )


def _rec_tape_rsp(
    *,
    corr_id: str,
    memory_used: bool,
    recall_count: int,
    backend_counts: Dict[str, Any] | None,
    memory_digest: Optional[str],
) -> None:
    if not settings.HUB_DEBUG_RECALL:
        return
    digest_chars = len(memory_digest or "")
    logger.info(
        "REC_TAPE RSP corr_id=%s memory_used=%s digest_chars=%s recall_count=%s backend_counts=%s",
        corr_id,
        memory_used,
        digest_chars,
        recall_count,
        backend_counts or {},
    )

# ======================================================================
# 🏠 ROOT + STATIC HTML
# ======================================================================
@router.get("/")
async def root():
    """Serves the main Hub UI (index.html)."""
    from .main import html_content
    return HTMLResponse(
        content=html_content,
        status_code=200,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@router.get("/health")
def health():
    """Simple health check endpoint."""
    return {"status": "ok", "service": settings.SERVICE_NAME}


def _runtime_identity() -> dict:
    return {
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "node": settings.NODE_NAME,
        "git_sha": os.getenv("GIT_SHA") or os.getenv("SOURCE_COMMIT") or "unknown",
        "build_timestamp": os.getenv("BUILD_TIMESTAMP") or "unknown",
        "environment": os.getenv("ORION_ENV") or os.getenv("ENVIRONMENT") or "unknown",
        "process_started_at": PROCESS_STARTED_AT_UTC.isoformat(),
        "now_utc": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/api/debug/build")
def api_debug_build():
    return {
        "hub": _runtime_identity(),
        "downstream": {
            "cortex_gateway_request_channel": settings.CORTEX_GATEWAY_REQUEST_CHANNEL,
            "notify_base_url": settings.NOTIFY_BASE_URL,
            "landing_pad_url": settings.LANDING_PAD_URL,
        },
    }



@router.get("/api/service-logs/services")
def api_service_logs_services() -> Dict[str, Any]:
    return collect_service_inventory()

@router.get("/api/notifications")
async def api_notifications(limit: int = 50):
    from .main import notification_cache
    if not notification_cache:
        return []
    return await notification_cache.get_latest(limit)


def _default_presence_context() -> dict:
    return {
        "kind": "presence.context.v1",
        "requestor": {
            "display_name": settings.ORION_PRESENCE_DEFAULT_REQUESTOR,
            "relationship_to_orion": "primary_operator",
            "source": "default",
            "confidence": "medium",
        },
        "companions": [],
        "audience_mode": "solo",
        "source": "default",
        "persist_to_memory": False,
        "privacy_mode": "session_only",
    }


def _situation_time_context(now_local: datetime) -> TimeContextV1:
    hour = now_local.hour
    if hour < 5:
        label = "pre_dawn"
        phase = "pre_dawn"
    elif hour < 8:
        label = "early_morning"
        phase = "morning"
    elif hour < 10:
        label = "mid_morning"
        phase = "morning"
    elif hour < 12:
        label = "late_morning"
        phase = "morning"
    elif hour < 14:
        label = "midday"
        phase = "midday"
    elif hour < 16:
        label = "early_afternoon"
        phase = "afternoon"
    elif hour < 18:
        label = "late_afternoon"
        phase = "afternoon"
    elif hour < 21:
        label = "evening"
        phase = "dusk"
    elif hour < 23:
        label = "late_evening"
        phase = "night"
    else:
        label = "night"
        phase = "night"
    return TimeContextV1(
        timezone=settings.ORION_SITUATION_TIMEZONE,
        local_datetime=now_local.isoformat(),
        local_date=now_local.strftime("%Y-%m-%d"),
        local_time=now_local.strftime("%H:%M"),
        weekday=now_local.strftime("%A"),
        is_weekend=now_local.weekday() >= 5,
        season_local="unknown",
        time_of_day_label=label,  # type: ignore[arg-type]
        day_phase=phase,  # type: ignore[arg-type]
    )


@router.get("/api/presence")
def api_presence(x_orion_session_id: Optional[str] = Header(None)):
    from .main import presence_context_store

    session_key = str(x_orion_session_id or "anonymous")
    payload = presence_context_store.get(session_key) if presence_context_store else None
    return payload or _default_presence_context()


@router.post("/api/presence")
def api_presence_set(payload: dict, x_orion_session_id: Optional[str] = Header(None)):
    from .main import presence_context_store

    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="presence payload must be an object")
    session_key = str(x_orion_session_id or payload.get("browser_client_id") or "anonymous")
    merged = _default_presence_context()
    merged.update(payload or {})
    companions = merged.get("companions")
    if not isinstance(companions, list):
        merged["companions"] = []
    merged["persist_to_memory"] = bool(merged.get("persist_to_memory", False) and settings.ORION_PRESENCE_PERSIST_ALLOWED)
    if presence_context_store:
        return presence_context_store.set(session_key, merged)
    return merged


@router.delete("/api/presence")
def api_presence_clear(x_orion_session_id: Optional[str] = Header(None)):
    from .main import presence_context_store

    session_key = str(x_orion_session_id or "anonymous")
    if presence_context_store:
        presence_context_store.clear(session_key)
    return _default_presence_context()


@router.get("/api/presence/connections")
def api_presence_connections():
    from .main import presence_state

    if not presence_state:
        return {"active": False, "last_seen": None, "active_connections": 0}
    return presence_state.snapshot()


@router.get("/api/situation/status")
def api_situation_status():
    return {
        "enabled": bool(settings.ORION_SITUATION_ENABLED),
        "providers": {
            "time": "enabled",
            "location": "configured",
            "weather": settings.ORION_SITUATION_WEATHER_PROVIDER,
            "agenda": "stub",
            "lab": "stub",
            "presence": "hub_manual",
        },
        "ttl_seconds": int(settings.ORION_SITUATION_TTL_SECONDS),
        "timezone": settings.ORION_SITUATION_TIMEZONE,
    }


@router.get("/api/situation/brief")
def api_situation_brief(x_orion_session_id: Optional[str] = Header(None)):
    from .main import presence_context_store

    session_key = str(x_orion_session_id or "anonymous")
    presence = presence_context_store.get(session_key) if presence_context_store else None
    now_utc = datetime.now(timezone.utc)
    now_local = now_utc.astimezone(ZoneInfo(settings.ORION_SITUATION_TIMEZONE))
    time_ctx = _situation_time_context(now_local)
    brief = SituationBriefV1(
        generated_at=now_utc,
        ttl_seconds=int(settings.ORION_SITUATION_TTL_SECONDS),
        source_summary={"presence": "hub_manual", "weather": "stub"},
        presence=PresenceContextV1.model_validate(presence or _default_presence_context()),
        time=time_ctx,
        conversation_phase=ConversationPhaseContextV1(),
        place=PlaceContextV1(timezone=settings.ORION_SITUATION_TIMEZONE),
    )
    return brief.model_dump(mode="json")


@router.get("/api/notify/recipients")
def api_notify_recipients():
    if not settings.NOTIFY_BASE_URL:
        return []
    headers = {}
    if settings.NOTIFY_API_TOKEN:
        headers["X-Orion-Notify-Token"] = settings.NOTIFY_API_TOKEN
    try:
        resp = requests.get(f"{settings.NOTIFY_BASE_URL}/recipients", headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("Failed to fetch recipient profiles: %s", exc)
        return []


@router.get("/api/notify/recipients/{recipient_group}")
def api_notify_recipient(recipient_group: str):
    if not settings.NOTIFY_BASE_URL:
        raise HTTPException(status_code=400, detail="Notify base URL not configured")
    headers = {}
    if settings.NOTIFY_API_TOKEN:
        headers["X-Orion-Notify-Token"] = settings.NOTIFY_API_TOKEN
    try:
        resp = requests.get(
            f"{settings.NOTIFY_BASE_URL}/recipients/{recipient_group}",
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("Failed to fetch recipient profile %s: %s", recipient_group, exc)
        raise HTTPException(status_code=502, detail="Failed to fetch recipient profile") from exc


@router.api_route("/api/topic-foundry/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def proxy_topic_foundry(path: str, request: Request) -> Response:
    if not settings.TOPIC_FOUNDRY_BASE_URL:
        raise HTTPException(status_code=400, detail="Topic Foundry base URL not configured")
    try:
        return await _proxy_request(request, settings.TOPIC_FOUNDRY_BASE_URL, path)
    except aiohttp.ClientError as exc:
        logger.warning("Topic Foundry proxy error: %s", exc)
        raise HTTPException(status_code=502, detail="Topic Foundry proxy request failed") from exc


@router.api_route("/api/world-pulse/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def proxy_world_pulse(path: str, request: Request) -> Response:
    if not settings.WORLD_PULSE_BASE_URL:
        raise HTTPException(status_code=400, detail="World Pulse base URL not configured")
    try:
        return await _proxy_request(request, settings.WORLD_PULSE_BASE_URL, path)
    except aiohttp.ClientError as exc:
        logger.warning("World Pulse proxy error: %s", exc)
        raise HTTPException(status_code=502, detail="World Pulse proxy request failed") from exc


@router.get("/api/social-memory/inspection")
async def api_social_memory_inspection(
    platform: str = Query(...),
    room_id: str = Query(...),
    participant_id: str | None = Query(None),
):
    if not settings.SOCIAL_MEMORY_BASE_URL:
        raise HTTPException(status_code=400, detail="Social memory base URL not configured")
    try:
        return await _fetch_social_memory(
            "/inspection",
            {"platform": platform, "room_id": room_id, "participant_id": participant_id},
        )
    except aiohttp.ClientResponseError as exc:
        detail = exc.message or "Social memory inspection request failed"
        raise HTTPException(status_code=exc.status or 502, detail=detail) from exc
    except aiohttp.ClientError as exc:
        logger.warning("Social memory inspection proxy error: %s", exc)
        raise HTTPException(status_code=502, detail="Social memory inspection request failed") from exc


@router.put("/api/notify/recipients/{recipient_group}")
def api_notify_recipient_update(recipient_group: str, payload: RecipientProfileUpdate):
    if not settings.NOTIFY_BASE_URL:
        raise HTTPException(status_code=400, detail="Notify base URL not configured")
    headers = {"Content-Type": "application/json"}
    if settings.NOTIFY_API_TOKEN:
        headers["X-Orion-Notify-Token"] = settings.NOTIFY_API_TOKEN
    try:
        resp = requests.put(
            f"{settings.NOTIFY_BASE_URL}/recipients/{recipient_group}",
            json=payload.model_dump(mode="json"),
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("Failed to update recipient profile %s: %s", recipient_group, exc)
        raise HTTPException(status_code=502, detail="Failed to update recipient profile") from exc


@router.get("/api/notify/recipients/{recipient_group}/preferences")
def api_notify_preferences(recipient_group: str):
    if not settings.NOTIFY_BASE_URL:
        return []
    headers = {}
    if settings.NOTIFY_API_TOKEN:
        headers["X-Orion-Notify-Token"] = settings.NOTIFY_API_TOKEN
    try:
        resp = requests.get(
            f"{settings.NOTIFY_BASE_URL}/recipients/{recipient_group}/preferences",
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("Failed to fetch preferences for %s: %s", recipient_group, exc)
        return []


@router.put("/api/notify/recipients/{recipient_group}/preferences")
def api_notify_preferences_update(recipient_group: str, payload: NotificationPreferencesUpdate):
    if not settings.NOTIFY_BASE_URL:
        raise HTTPException(status_code=400, detail="Notify base URL not configured")
    headers = {"Content-Type": "application/json"}
    if settings.NOTIFY_API_TOKEN:
        headers["X-Orion-Notify-Token"] = settings.NOTIFY_API_TOKEN
    try:
        resp = requests.put(
            f"{settings.NOTIFY_BASE_URL}/recipients/{recipient_group}/preferences",
            json=payload.model_dump(mode="json"),
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("Failed to update preferences for %s: %s", recipient_group, exc)
        raise HTTPException(status_code=502, detail="Failed to update preferences") from exc


@router.post("/api/notify/preferences/resolve")
def api_notify_preferences_resolve(payload: PreferencesResolveProxyRequest):
    if not settings.NOTIFY_BASE_URL:
        raise HTTPException(status_code=400, detail="Notify base URL not configured")
    headers = {"Content-Type": "application/json"}
    if settings.NOTIFY_API_TOKEN:
        headers["X-Orion-Notify-Token"] = settings.NOTIFY_API_TOKEN
    req = PreferenceResolutionRequest(
        recipient_group=payload.recipient_group,
        event_kind=payload.event_kind,
        severity=payload.severity,
    )
    try:
        resp = requests.post(
            f"{settings.NOTIFY_BASE_URL}/preferences/resolve",
            json=req.model_dump(mode="json"),
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("Failed to resolve preferences: %s", exc)
        raise HTTPException(status_code=502, detail="Failed to resolve preferences") from exc


@router.get("/api/attention")
def api_attention(limit: int = 50, status: Optional[str] = None):
    if not settings.NOTIFY_BASE_URL:
        return []
    headers = {}
    if settings.NOTIFY_API_TOKEN:
        headers["X-Orion-Notify-Token"] = settings.NOTIFY_API_TOKEN
    try:
        resp = requests.get(
            f"{settings.NOTIFY_BASE_URL}/attention",
            params={"limit": limit, "status": status},
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("Failed to fetch attention list: %s", exc)
        return []


@router.post("/api/attention/{attention_id}/ack")
def api_attention_ack(attention_id: str, payload: AttentionAckRequest):
    if not settings.NOTIFY_BASE_URL:
        raise HTTPException(status_code=400, detail="Notify base URL not configured")
    headers = {"Content-Type": "application/json"}
    if settings.NOTIFY_API_TOKEN:
        headers["X-Orion-Notify-Token"] = settings.NOTIFY_API_TOKEN
    ack = ChatAttentionAck(attention_id=attention_id, ack_type=payload.ack_type, note=payload.note)
    try:
        resp = requests.post(
            f"{settings.NOTIFY_BASE_URL}/attention/{attention_id}/ack",
            json=ack.model_dump(mode="json"),
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("Failed to acknowledge attention %s: %s", attention_id, exc)
        raise HTTPException(status_code=502, detail="Failed to acknowledge attention") from exc


@router.get("/api/chat/messages")
def api_chat_messages(limit: int = 50, status: Optional[str] = None, session_id: Optional[str] = None):
    if not settings.NOTIFY_BASE_URL:
        return []
    headers = {}
    if settings.NOTIFY_API_TOKEN:
        headers["X-Orion-Notify-Token"] = settings.NOTIFY_API_TOKEN
    try:
        resp = requests.get(
            f"{settings.NOTIFY_BASE_URL}/chat/messages",
            params={"limit": limit, "status": status, "session_id": session_id},
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("Failed to fetch chat messages: %s", exc)
        return []


@router.post("/api/chat/message/{message_id}/receipt")
def api_chat_message_receipt(message_id: str, payload: ChatMessageReceiptRequest):
    if not settings.NOTIFY_BASE_URL:
        raise HTTPException(status_code=400, detail="Notify base URL not configured")
    headers = {"Content-Type": "application/json"}
    if settings.NOTIFY_API_TOKEN:
        headers["X-Orion-Notify-Token"] = settings.NOTIFY_API_TOKEN
    receipt = ChatMessageReceipt(
        message_id=message_id,
        session_id=payload.session_id,
        receipt_type=payload.receipt_type,
    )
    try:
        resp = requests.post(
            f"{settings.NOTIFY_BASE_URL}/chat/message/{message_id}/receipt",
            json=receipt.model_dump(mode="json"),
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("Failed to send chat message receipt %s: %s", message_id, exc)
        raise HTTPException(status_code=502, detail="Failed to acknowledge chat message") from exc

@router.post("/api/chat/response-feedback")
async def api_chat_response_feedback(payload: Dict[str, Any]):
    from .main import bus

    if not bus or not getattr(bus, "enabled", False):
        raise HTTPException(status_code=503, detail="Bus unavailable")

    try:
        payload_data = dict(payload or {})
        payload_data.setdefault("source", "hub_ui")
        payload_data.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        feedback_payload = ChatResponseFeedbackV1.model_validate(payload_data)
    except Exception as exc:
        logger.warning(
            "chat_response_feedback_rejected reason=%s feedback_id=%s corr=%s sid=%s",
            exc,
            (payload or {}).get("feedback_id"),
            (payload or {}).get("target_correlation_id"),
            (payload or {}).get("session_id"),
        )
        raise HTTPException(status_code=422, detail=f"Invalid feedback payload: {exc}") from exc

    env = build_chat_response_feedback_envelope(
        feedback_payload=feedback_payload,
        correlation_id=feedback_payload.target_correlation_id,
    )
    await publish_chat_response_feedback(bus, env)
    feedback_events = _producer_pressure_events_from_feedback(feedback_payload)
    _record_pressure_events_as_telemetry(
        events=feedback_events,
        correlation_id=feedback_payload.target_correlation_id,
        source_event_id=feedback_payload.feedback_id,
    )
    return {"ok": True, "feedback_id": feedback_payload.feedback_id}


@router.get("/api/chat/response-feedback/options")
def api_chat_response_feedback_options() -> Dict[str, Any]:
    return {
        "feedback_values": ["up", "down"],
        "categories": build_feedback_category_options(),
        "max_free_text_chars": 2000,
    }


# ======================================================================
# 🧠 SESSION MANAGEMENT
# ======================================================================
@router.get("/api/session")
async def api_session(x_orion_session_id: Optional[str] = Header(None)):
    """
    Called by Hub UI on load.
    Always returns a warm-started session_id.
    """
    from .main import bus
    if not bus:
        raise RuntimeError("OrionBus not initialized.")

    session_id = await ensure_session(x_orion_session_id, bus)
    return {"session_id": session_id}

# ======================================================================
# 📚 COGNITION LIBRARY (Verbs & Packs)
# ======================================================================
@router.get("/api/cognition/library")
def get_cognition_library():
    """
    Returns the scanned list of Packs and Verbs available in the system.
    Used by the UI to populate dropdowns and filters.
    """
    return scan_cognition_library()



@router.get("/api/verbs")
def api_verbs(include_inactive: int = Query(default=0, ge=0, le=1)):
    include = bool(include_inactive)
    return {"verbs": build_verb_list(node_name=settings.NODE_NAME, include_inactive=include)}

# ======================================================================
# 💬 SHARED CHAT CORE (HTTP + WS)
# ======================================================================

_PRESSURE_FEEDBACK_CATEGORIES = {
    "fabricated_recall_memory",
    "missed_relevant_context",
    "lost_conversation_continuity",
    "incomplete_truncated",
    "wrong_tool_wrong_routing_wrong_mode",
}


def _build_pressure_event(
    *,
    category: str,
    source_service: str,
    source_event_id: str,
    correlation_id: str | None,
    confidence: float,
    evidence_refs: list[str],
    metadata: Dict[str, Any] | None = None,
) -> MutationPressureEvidenceV1:
    refs = [item for item in evidence_refs if str(item or "").strip()]
    if not refs:
        refs = [f"source_event:{source_event_id}"]
    return MutationPressureEvidenceV1(
        source_service=source_service,
        source_event_id=source_event_id,
        correlation_id=correlation_id,
        pressure_category=category,  # type: ignore[arg-type]
        confidence=max(0.0, min(float(confidence), 1.0)),
        evidence_refs=refs[:32],
        metadata=dict(metadata or {}),
    )


def _producer_pressure_events_from_chat_result(
    *,
    correlation_id: str | None,
    route_debug: Dict[str, Any] | None,
    autonomy_payload: Dict[str, Any] | None,
    recall_debug: Dict[str, Any] | None = None,
) -> list[MutationPressureEvidenceV1]:
    payload = autonomy_payload if isinstance(autonomy_payload, dict) else {}
    route = route_debug if isinstance(route_debug, dict) else {}
    diagnostics = payload.get("runtime_response_diagnostics") if isinstance(payload.get("runtime_response_diagnostics"), dict) else {}
    chat_stance_debug = payload.get("chat_stance_debug") if isinstance(payload.get("chat_stance_debug"), dict) else {}
    social_bridge = (
        (chat_stance_debug.get("source_inputs") or {}).get("social_bridge")
        if isinstance(chat_stance_debug.get("source_inputs"), dict)
        else {}
    )
    social_hazards = list((social_bridge or {}).get("hazards") or [])
    events: list[MutationPressureEvidenceV1] = []
    corr = str(correlation_id or "")
    source_event_id = f"chat_result:{corr or uuid4()}"
    if diagnostics.get("truncation_detected") or str(diagnostics.get("provider_finish_reason") or "").lower() == "length":
        events.append(
            _build_pressure_event(
                category="response_truncation_or_length_finish",
                source_service="orion-cortex-exec",
                source_event_id=source_event_id,
                correlation_id=correlation_id,
                confidence=0.72,
                evidence_refs=[
                    f"corr:{corr}",
                    f"finish_reason:{diagnostics.get('provider_finish_reason')}",
                    f"truncation_detected:{bool(diagnostics.get('truncation_detected'))}",
                ],
                metadata={"route_mode": route.get("mode"), "trace_verb": route.get("verb")},
            )
        )
    if str(diagnostics.get("status") or "").lower() in {"fail", "partial"}:
        events.append(
            _build_pressure_event(
                category="runtime_degradation_or_timeout",
                source_service="orion-cortex-exec",
                source_event_id=source_event_id,
                correlation_id=correlation_id,
                confidence=0.66,
                evidence_refs=[f"corr:{corr}", f"status:{diagnostics.get('status')}"],
                metadata={"diagnostics": diagnostics},
            )
        )
    if "self_message_loop" in social_hazards or "not_addressed" in social_hazards:
        events.append(
            _build_pressure_event(
                category="social_addressedness_gap",
                source_service="orion-cortex-exec",
                source_event_id=source_event_id,
                correlation_id=correlation_id,
                confidence=0.61,
                evidence_refs=[f"corr:{corr}", *[f"social_hazard:{item}" for item in social_hazards[:4]]],
                metadata={"social_hazards": social_hazards[:8]},
            )
        )
    recall_payload = recall_debug if isinstance(recall_debug, dict) else {}
    nested_rd = (
        (recall_payload.get("decision") or {}).get("recall_debug")
        if isinstance(recall_payload.get("decision"), dict)
        else None
    )
    recall_pressure_events = list(recall_payload.get("pressure_events") or [])
    if not recall_pressure_events and isinstance(nested_rd, dict):
        recall_pressure_events = list(nested_rd.get("pressure_events") or [])
    compare_summary = recall_payload.get("compare_summary")
    if not isinstance(compare_summary, dict) or not compare_summary:
        compare_summary = (
            (recall_payload.get("debug") or {}).get("compare_summary")
            if isinstance(recall_payload.get("debug"), dict)
            else {}
        )
    if (not isinstance(compare_summary, dict) or not compare_summary) and isinstance(nested_rd, dict):
        compare_summary = nested_rd.get("compare_summary") or {}
    anchor_plan = recall_payload.get("anchor_plan_summary")
    if not isinstance(anchor_plan, dict) or not anchor_plan:
        anchor_plan = (
            (recall_payload.get("debug") or {}).get("anchor_plan_summary")
            if isinstance(recall_payload.get("debug"), dict)
            else {}
        )
    if (not isinstance(anchor_plan, dict) or not anchor_plan) and isinstance(nested_rd, dict):
        anchor_plan = nested_rd.get("anchor_plan_summary") or {}
    selected_cards = recall_payload.get("selected_evidence_cards")
    if not isinstance(selected_cards, list) or not selected_cards:
        selected_cards = (
            (recall_payload.get("debug") or {}).get("selected_evidence_cards")
            if isinstance(recall_payload.get("debug"), dict)
            else []
        )
    if (not isinstance(selected_cards, list) or not selected_cards) and isinstance(nested_rd, dict):
        selected_cards = list(nested_rd.get("selected_evidence_cards") or [])
    for row in recall_pressure_events:
        if not isinstance(row, dict):
            continue
        try:
            event = MutationPressureEvidenceV1.model_validate(row)
        except Exception:
            continue
        metadata = dict(event.metadata or {})
        if isinstance(compare_summary, dict) and compare_summary:
            metadata.setdefault("v1_v2_compare", compare_summary)
        if isinstance(anchor_plan, dict) and anchor_plan:
            metadata.setdefault("anchor_plan", anchor_plan)
        if isinstance(selected_cards, list) and selected_cards:
            metadata.setdefault("selected_evidence_cards", selected_cards[:8])
        events.append(event.model_copy(update={"metadata": metadata}))
    return events[:8]


def _producer_pressure_events_from_feedback(feedback: ChatResponseFeedbackV1) -> list[MutationPressureEvidenceV1]:
    categories = {str(item or "").strip() for item in feedback.categories}
    corr = feedback.target_correlation_id
    source_event_id = feedback.feedback_id
    events: list[MutationPressureEvidenceV1] = []
    if feedback.feedback_value == "down" and (categories & _PRESSURE_FEEDBACK_CATEGORIES):
        events.append(
            _build_pressure_event(
                category="recall_miss_or_dissatisfaction",
                source_service="orion-hub",
                source_event_id=source_event_id,
                correlation_id=corr,
                confidence=0.84,
                evidence_refs=[f"feedback:{feedback.feedback_id}", *[f"feedback_category:{item}" for item in sorted(categories)]],
                metadata={"feedback_value": feedback.feedback_value},
            )
        )
    if feedback.feedback_value == "down" and "wrong_tool_wrong_routing_wrong_mode" in categories:
        events.append(
            _build_pressure_event(
                category="routing_false_downgrade",
                source_service="orion-hub",
                source_event_id=source_event_id,
                correlation_id=corr,
                confidence=0.7,
                evidence_refs=[f"feedback:{feedback.feedback_id}", "feedback_category:wrong_tool_wrong_routing_wrong_mode"],
                metadata={"feedback_value": feedback.feedback_value},
            )
        )
        events.append(
            _build_pressure_event(
                category="routing_false_escalation",
                source_service="orion-hub",
                source_event_id=source_event_id,
                correlation_id=corr,
                confidence=0.7,
                evidence_refs=[f"feedback:{feedback.feedback_id}", "feedback_category:wrong_tool_wrong_routing_wrong_mode"],
                metadata={"feedback_value": feedback.feedback_value},
            )
        )
    return events[:8]


def _record_pressure_events_as_telemetry(
    *,
    events: list[MutationPressureEvidenceV1],
    correlation_id: str | None,
    source_event_id: str,
    invocation_surface: GraphReviewRuntimeSurfaceV1 = "chat_reflective_lane",
    ingest_notes: list[str] | None = None,
) -> None:
    if not events:
        return
    notes = list(ingest_notes or ["producer_pressure_event_ingest"])
    try:
        telemetry = GraphReviewTelemetryRecordV1(
            correlation_id=correlation_id,
            invocation_surface=invocation_surface,
            anchor_scope="orion",
            subject_ref="entity:orion",
            target_zone="autonomy_graph",
            selection_reason=f"producer_pressure_events:{source_event_id}",
            selected_priority=50,
            execution_outcome="executed",
            runtime_duration_ms=0,
            notes=notes[:64],
            pressure_events=events,
        )
        SUBSTRATE_REVIEW_TELEMETRY_STORE.record(telemetry)
    except Exception as exc:
        logger.warning("pressure_event_telemetry_record_failed source_event_id=%s error=%s", source_event_id, exc)


def record_chat_turn_pressure_telemetry(
    *,
    correlation_id: str | None,
    route_debug: Dict[str, Any] | None,
    autonomy_payload: Dict[str, Any] | None,
    recall_debug: Dict[str, Any] | None,
    source_event_id: str,
) -> None:
    """Shared HTTP + WebSocket path for producer pressure extraction and telemetry recording."""
    events = _producer_pressure_events_from_chat_result(
        correlation_id=correlation_id,
        route_debug=route_debug,
        autonomy_payload=autonomy_payload,
        recall_debug=recall_debug,
    )
    _record_pressure_events_as_telemetry(
        events=events,
        correlation_id=correlation_id,
        source_event_id=source_event_id,
    )

def _http_chat_turn_context_from_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    WebSocket parity for HTTP /api/chat: merge gateway metadata (turn_effect, model, …)
    into spark_meta and align reasoning extraction with websocket_handler.
    """
    raw = result.get("raw") or {}
    gateway_meta = raw.get("metadata") if isinstance(raw, dict) else {}
    if not isinstance(gateway_meta, dict):
        gateway_meta = {}
    routing = result.get("routing_debug") or {}
    trace_verb = str(
        gateway_meta.get("trace_verb")
        or routing.get("verb")
        or raw.get("verb")
        or ""
    ).strip() or None
    explicit_reasoning_trace = (
        raw.get("reasoning_trace") if isinstance(raw.get("reasoning_trace"), dict) else None
    )
    reasoning_content = (
        gateway_meta.get("reasoning_content")
        or raw.get("reasoning_content")
        or (
            ((raw.get("raw") or {}).get("reasoning_content"))
            if isinstance(raw.get("raw"), dict)
            else None
        )
    )
    inline_think_content = (
        gateway_meta.get("inline_think_content")
        or raw.get("inline_think_content")
    )
    raw_thinking_source = gateway_meta.get("thinking_source") or raw.get("thinking_source")
    thinking_source = "none"
    if isinstance(raw_thinking_source, str) and raw_thinking_source.strip():
        thinking_source = raw_thinking_source.strip()
    elif isinstance(reasoning_content, str) and reasoning_content.strip():
        thinking_source = "provider_reasoning"
    elif isinstance(inline_think_content, str) and inline_think_content.strip():
        thinking_source = "inline_think_full_block"
    if reasoning_content and not (
        isinstance(explicit_reasoning_trace, dict)
        and str(explicit_reasoning_trace.get("content") or "").strip()
    ):
        explicit_reasoning_trace = {
            "trace_role": "reasoning",
            "trace_stage": "post_answer",
            "content": str(reasoning_content).strip(),
            "metadata": {"source": "hub_reasoning_content_fallback"},
        }
    spark_meta = {
        "mode": result.get("mode"),
        "trace_verb": trace_verb,
        "use_recall": result.get("use_recall"),
        "reasoning_content": reasoning_content,
        "inline_think_content": inline_think_content,
        "thinking_source": thinking_source,
        "thought_capture_step": "llm_chat_general"
        if str(trace_verb or "").strip() == "chat_general"
        else None,
        **gateway_meta,
    }
    return {
        "spark_meta": spark_meta,
        "explicit_reasoning_trace": explicit_reasoning_trace,
        "reasoning_content": reasoning_content,
        "inline_think_content": inline_think_content,
        "thinking_source": thinking_source,
        "gateway_meta": gateway_meta,
    }


async def handle_chat_request(
    cortex_client,
    payload: dict,
    session_id: str,
    no_write: bool,
) -> Dict[str, Any]:
    """
    Core chat handler used by both HTTP /api/chat and (optionally) WebSocket.
    Delegate strict typed requests to orion-cortex-gateway via Bus.
    """
    user_messages = payload.get("messages", [])

    # Respect client toggle; default to True if missing
    raw_recall = payload.get("use_recall", None)
    use_recall = _normalize_bool(raw_recall, default=True)

    if not isinstance(user_messages, list) or len(user_messages) == 0:
        return {"error": "Invalid payload: missing messages[]"}

    user_prompt = user_messages[-1].get("content", "") or ""

    logger.info(
        "HTTP Chat Request payload session_id=%s messages_len=%s last_user_len=%s last_user_head=%r",
        session_id,
        len(user_messages),
        len(user_prompt or ""),
        (user_prompt or "")[:120],
    )

    inactive = validate_single_verb_override(payload, node_name=settings.NODE_NAME)
    if inactive:
        return inactive

    corr_id = str(uuid4())
    context_turns = int(payload.get("context_turns") or getattr(settings, "HUB_CONTEXT_TURNS", 10))
    continuity_messages = build_continuity_messages(
        history=user_messages,
        latest_user_prompt=user_prompt,
        turns=context_turns,
    )
    routed_payload = dict(payload)
    routed_payload["no_write"] = bool(no_write)
    if "presence_context" not in routed_payload:
        try:
            from .main import presence_context_store

            stored_presence = presence_context_store.get(session_id) if presence_context_store else None
            if stored_presence:
                routed_payload["presence_context"] = stored_presence
        except Exception:
            pass
    routed_payload["mutation_cognition_context"] = build_mutation_cognition_context(store=SUBSTRATE_MUTATION_STORE)
    req, route_debug, _ = build_chat_request(
        payload=routed_payload,
        session_id=session_id,
        user_id=payload.get("user_id"),
        trace_id=None,
        default_mode="brain",
        auto_default_enabled=bool(settings.HUB_AUTO_DEFAULT_ENABLED),
        source_label="hub_http",
        prompt=user_prompt,
        messages=continuity_messages,
    )
    workflow_request = req.metadata.get("workflow_request") if isinstance(req.metadata, dict) else None
    execution_policy = workflow_request.get("execution_policy") if isinstance(workflow_request, dict) else None
    logger.info(
        "workflow_resolution_result %s",
        json.dumps(
            {
                "correlation_id": corr_id,
                "matched_workflow_id": (workflow_request or {}).get("workflow_id") if isinstance(workflow_request, dict) else None,
                "fallback_route": route_debug.get("fallback_route"),
                "reason": route_debug.get("workflow_resolution_reason"),
            },
            sort_keys=True,
            default=str,
        ),
    )
    logger.info(
        "hub_workflow_request corr=%s sid=%s workflow_id=%s invocation_mode=%s schedule_kind=%s",
        corr_id,
        session_id,
        (workflow_request or {}).get("workflow_id") if isinstance(workflow_request, dict) else None,
        (execution_policy or {}).get("invocation_mode") if isinstance(execution_policy, dict) else None,
        ((execution_policy or {}).get("schedule") or {}).get("kind") if isinstance(execution_policy, dict) else None,
    )

    recall_payload = req.recall or {"enabled": use_recall}
    mode = req.mode

    logger.info(f"Chat Request recall config: {recall_payload} session_id={session_id}")

    diagnostic = bool(payload.get("diagnostic") or (isinstance(payload.get("options"), dict) and payload.get("options", {}).get("diagnostic")))
    if diagnostic:
        logger.info("HTTP outbound CortexChatRequest corr=%s payload=%s", corr_id, req.model_dump(mode="json"))
    logger.info(
        "hub_egress corr=%s sid=%s mode=%s verb=%s route_intent=%s allowed_verbs=%s packs=%s",
        corr_id,
        session_id,
        req.mode,
        req.verb,
        (req.options or {}).get("route_intent") or "none",
        len(((req.options or {}).get("allowed_verbs") or [])),
        req.packs or [],
    )
    logger.info(
        "hub_context_messages corr=%s sid=%s mode=%s count=%s roles=%s",
        corr_id,
        session_id,
        req.mode,
        len(req.messages or []),
        [m.role if hasattr(m, "role") else m.get("role") for m in (req.messages or [])][:12],
    )
    _log_hub_route_decision(
        corr_id=corr_id,
        session_id=session_id,
        route_debug=route_debug,
        user_prompt=user_prompt,
    )

    _rec_tape_req(
        corr_id=corr_id,
        session_id=session_id,
        mode=mode,
        use_recall=use_recall,
        recall_profile=recall_payload.get("profile"),
        user_head=(user_prompt or "")[:80],
        no_write=no_write,
    )
    try:
        # Call Bus RPC - Hub/Client generates correlation_id internally for RPC
        resp: CortexChatResult = await cortex_client.chat(req, correlation_id=corr_id)

        # Map raw result for UI debug (HTTP + WS clients use this to coalesce display text)
        raw_result = resp.cortex_result.model_dump(mode="json")

        # Extract Text — prefer longest answer if top-level vs nested diverge (parity with curl `raw.final_text`)
        text = hub_effective_chat_text(resp)

        # Use the correlation_id from the response (gateway) if available
        # or it might be passed back from the client logic if modified to do so.
        # Here we rely on CortexChatResult having it.
        correlation_id = resp.cortex_result.correlation_id or corr_id

        memory_digest = None
        recall_debug = None
        agent_trace = None
        workflow = None
        autonomy_payload = {}
        metacog_traces = []
        if resp.cortex_result and isinstance(resp.cortex_result.recall_debug, dict):
            recall_debug = resp.cortex_result.recall_debug
            memory_digest = recall_debug.get("memory_digest")
        agent_trace = extract_agent_trace_payload(resp.cortex_result)
        raw_traces = getattr(resp.cortex_result, "metacog_traces", None)
        if isinstance(raw_traces, list):
            metacog_traces = [t for t in raw_traces if isinstance(t, dict)]
        logger.info(
            "hub_metacog_received corr=%s source=http traces=%s",
            correlation_id,
            len(metacog_traces),
        )
        workflow = extract_workflow_payload(resp.cortex_result)
        autonomy_payload = extract_autonomy_payload(resp.cortex_result)
        if isinstance(workflow, dict):
            logger.info(
                "hub_workflow_response corr=%s workflow_id=%s status=%s scheduled_count=%s persisted_count=%s rendered_path=%s",
                correlation_id,
                workflow.get("workflow_id"),
                workflow.get("status"),
                len(workflow.get("scheduled") or []),
                len(workflow.get("persisted") or []),
                "scheduled_confirmation" if len(workflow.get("scheduled") or []) else "immediate_or_unscheduled",
            )

        recall_count = 0
        backend_counts = None
        if isinstance(recall_debug, dict):
            recall_count = int(recall_debug.get("count") or 0)
            backend_counts = recall_debug.get("backend_counts")
            if backend_counts is None and isinstance(recall_debug.get("debug"), dict):
                backend_counts = recall_debug["debug"].get("backend_counts")
        memory_used = bool(getattr(resp.cortex_result, "memory_used", False))
        if not memory_used:
            memory_used = bool(recall_count)
        logger.info(
            "hub_ingress_result corr=%s sid=%s mode=%s status=%s final_len=%s memory_used=%s recall_count=%s",
            correlation_id,
            session_id,
            mode,
            getattr(resp.cortex_result, "status", None),
            len(text or ""),
            memory_used,
            recall_count,
        )
        _rec_tape_rsp(
            corr_id=str(correlation_id),
            memory_used=memory_used,
            recall_count=recall_count,
            backend_counts=backend_counts,
            memory_digest=memory_digest,
        )
        turn_ctx = _http_chat_turn_context_from_result(
            {
                "mode": mode,
                "use_recall": use_recall,
                "routing_debug": route_debug,
                "raw": raw_result,
            }
        )
        record_chat_turn_pressure_telemetry(
            correlation_id=str(correlation_id),
            route_debug=route_debug,
            autonomy_payload=autonomy_payload,
            recall_debug=recall_debug,
            source_event_id=f"chat_result:{correlation_id}",
        )

        return {
            "session_id": session_id,
            "mode": mode,
            "use_recall": use_recall,
            "text": text,
            "tokens": len(text.split()), # simple approx
            "raw": raw_result,
            "recall_debug": recall_debug,
            "agent_trace": agent_trace,
            "workflow": workflow,
            "memory_used": memory_used,
            "memory_digest": memory_digest,
            "no_write": no_write,
            "spark_meta": dict(autonomy_payload) if isinstance(autonomy_payload, dict) else {},
            "correlation_id": correlation_id,
            "routing_debug": route_debug,
            "metacog_traces": metacog_traces,
            "reasoning_content": turn_ctx.get("reasoning_content"),
            "inline_think_content": turn_ctx.get("inline_think_content"),
            "thinking_source": turn_ctx.get("thinking_source"),
            "reasoning_trace": turn_ctx.get("explicit_reasoning_trace"),
            **autonomy_payload,
        }

    except Exception as e:
        logger.error(f"Chat RPC failed: {e}", exc_info=True)
        return {"error": str(e)}


# ======================================================================
# 💬 CHAT ENDPOINT (HTTP wrapper around core)
# ======================================================================
@router.post("/api/chat")
async def api_chat(
    payload: dict,
    x_orion_session_id: Optional[str] = Header(None),
    x_orion_no_write: Optional[str] = Header(None),
):
    """
    Main LLM chat endpoint.
    Delegates to Cortex Gateway via Bus RPC.
    """
    from .main import bus, cortex_client
    if not bus or not cortex_client:
        raise RuntimeError("OrionBus/Client not initialized.")

    # Ensure warm-started session
    session_id = await ensure_session(x_orion_session_id, bus)
    no_write = _normalize_bool(payload.get("no_write"), default=False) or _normalize_bool(
        x_orion_no_write, default=False
    )

    # Core chat handling
    result = await handle_chat_request(cortex_client, payload, session_id, no_write)

    # ─────────────────────────────────────────────
    # 📡 Publish HTTP chat → chat history log
    # ─────────────────────────────────────────────
    text = result.get("text")
    correlation_id = result.get("correlation_id")

    if text and getattr(bus, "enabled", False) and not no_write:
        try:
            user_messages = payload.get("messages", [])
            latest_user_prompt = ""
            if isinstance(user_messages, list) and user_messages:
                latest_user_prompt = user_messages[-1].get("content", "") or ""

            use_recall = bool(result.get("use_recall", True))

            # If we didn't get a correlation_id from gateway, fallback to new UUID
            # (but ideally we got it).
            final_corr_id = correlation_id or str(uuid4())

            turn_ctx = _http_chat_turn_context_from_result(result)
            metacog_traces = result.get("metacog_traces") or []
            selected_reasoning_trace, _ = select_reasoning_trace_for_history(
                correlation_id=final_corr_id,
                reasoning_trace=turn_ctx.get("explicit_reasoning_trace"),
                metacog_traces=metacog_traces if isinstance(metacog_traces, list) else None,
                reasoning_content=turn_ctx.get("reasoning_content"),
                session_id=session_id,
                message_id=f"{final_corr_id}:assistant",
                model=(
                    (turn_ctx.get("gateway_meta") or {}).get("model")
                    if isinstance(turn_ctx.get("gateway_meta"), dict)
                    else None
                )
                or (settings.GATEWAY_MODEL if hasattr(settings, "GATEWAY_MODEL") else None),
            )

            envelopes = []
            social_meta = {}
            if is_social_room_payload(payload):
                social_meta = social_room_client_meta(
                    payload=payload,
                    route_debug=result.get("routing_debug") or {},
                    trace_verb=(result.get("routing_debug") or {}).get("verb"),
                    memory_digest=result.get("memory_digest"),
                )
            if latest_user_prompt:
                envelopes.append(
                    build_chat_history_envelope(
                        content=latest_user_prompt,
                        role="user",
                        session_id=session_id,
                        correlation_id=final_corr_id,
                        speaker=payload.get("user_id") or "user",
                        tags=[result.get("mode", "brain")],
                        client_meta=social_meta or None,
                    )
                )
            envelopes.append(
                build_chat_history_envelope(
                    content=text,
                    role="assistant",
                    session_id=session_id,
                    correlation_id=final_corr_id,
                    speaker=settings.SERVICE_NAME,
                    tags=[result.get("mode", "brain")],
                    client_meta=social_meta or None,
                    reasoning_trace=selected_reasoning_trace,
                )
            )
            await publish_chat_history(bus, envelopes)
            if social_meta:
                await publish_social_room_turn(
                    bus,
                    prompt=latest_user_prompt,
                    response=text,
                    session_id=session_id,
                    correlation_id=final_corr_id,
                    user_id=payload.get("user_id"),
                    source_label="hub_http",
                    recall_profile=((result.get("routing_debug") or {}).get("recall_profile")),
                    trace_verb=(result.get("routing_debug") or {}).get("verb"),
                    client_meta=social_meta,
                    memory_digest=result.get("memory_digest"),
                )
            if isinstance(metacog_traces, list):
                for trace in metacog_traces:
                    coerced_trace = _coerce_metacog_trace(trace)
                    if coerced_trace is None:
                        continue
                    trace_debug = trace if isinstance(trace, dict) else coerced_trace.model_dump(mode="json")
                    if _thought_debug_enabled():
                        logger.info(
                            "THOUGHT_DEBUG_METACOG_PUB stage=hub_http_prepare corr=%s trace_role=%s trace_stage=%s model=%s content_len=%s content_snippet=%r",
                            final_corr_id,
                            trace_debug.get("trace_role") or trace_debug.get("role"),
                            trace_debug.get("trace_stage") or trace_debug.get("stage"),
                            trace_debug.get("model"),
                            _debug_len(trace_debug.get("content")),
                            _debug_snippet(trace_debug.get("content")),
                        )
                    trace_env = BaseEnvelope(
                        kind="metacognitive.trace.v1",
                        source=ServiceRef(
                            name=settings.SERVICE_NAME,
                            node=settings.NODE_NAME,
                            version=settings.SERVICE_VERSION,
                        ),
                        correlation_id=final_corr_id,
                            payload=coerced_trace,
                    )
                    await bus.publish("orion:metacog:trace", trace_env)
                if _thought_debug_enabled() and not any(isinstance(t, dict) for t in metacog_traces):
                    logger.info("THOUGHT_DEBUG_METACOG_PUB stage=hub_http_skipped corr=%s reason=no_valid_trace_dicts", final_corr_id)
                logger.info(
                    "hub_metacog_published corr=%s source=http channel=%s traces=%s",
                    final_corr_id,
                    "orion:metacog:trace",
                    len([t for t in metacog_traces if isinstance(t, dict)]),
                )

            spark_meta = turn_ctx.get("spark_meta") if isinstance(turn_ctx.get("spark_meta"), dict) else {}
            env_turn = build_chat_turn_envelope(
                prompt=latest_user_prompt,
                response=text,
                session_id=session_id,
                correlation_id=final_corr_id,
                user_id=payload.get("user_id"),
                source_label="hub_http",
                spark_meta=spark_meta,
                turn_id=final_corr_id,
                memory_status="accepted",
                memory_tier="ephemeral",
                client_meta=social_meta or None,
                reasoning_content=turn_ctx.get("reasoning_content"),
                inline_think_content=turn_ctx.get("inline_think_content"),
                thinking_source=turn_ctx.get("thinking_source"),
                reasoning_trace=selected_reasoning_trace,
            )
            await publish_chat_turn(bus, env_turn)
            logger.info("Published chat.history turn row -> %s", settings.chat_history_turn_channel)

            # Legacy log for downstream SQL-writer compatibility
            chat_log_payload = {
                "correlation_id": final_corr_id,
                "source": settings.SERVICE_NAME,
                "prompt": latest_user_prompt,
                "response": text,
                "session_id": session_id,
                "mode": result.get("mode", "brain"),
                "recall": use_recall,
                "user_id": None,
                "spark_meta": spark_meta,
                "reasoning_trace": selected_reasoning_trace,
            }
            if _thought_debug_enabled():
                reasoning_trace = chat_log_payload.get("reasoning_trace")
                reasoning_content = chat_log_payload.get("reasoning_content")
                thought_candidate = (
                    (reasoning_trace.get("content") if isinstance(reasoning_trace, dict) else None)
                    or reasoning_content
                )
                logger.info(
                    "THOUGHT_DEBUG_HUB stage=legacy_chat_history_publish corr=%s channel=%s target=chat.history.turn_payload reasoning_trace_exists=%s reasoning_content_exists=%s thought_candidate_len=%s thought_candidate_snippet=%r",
                    final_corr_id,
                    settings.chat_history_channel,
                    isinstance(reasoning_trace, dict),
                    bool(str(reasoning_content or "").strip()),
                    _debug_len(thought_candidate),
                    _debug_snippet(thought_candidate),
                )

            await bus.publish(
                settings.chat_history_channel,
                chat_log_payload,
            )
        except Exception as e:
            logger.warning(
                "Failed to publish HTTP chat to chat history log: %s",
                e,
                exc_info=True,
            )

    return result


@router.get("/api/workflow/schedules")
async def api_workflow_schedule_list(
    x_orion_session_id: Optional[str] = Header(None),
    user_id: Optional[str] = Query(default=None),
    include_history: bool = Query(default=False),
):
    from .main import bus
    if not bus:
        raise HTTPException(status_code=503, detail="bus unavailable")
    session_id = await ensure_session(x_orion_session_id, bus)
    req = WorkflowScheduleManageRequestV1(
        operation="list",
        request_id=str(uuid4()),
        include_history=include_history,
        session_id=session_id,
        origin_user_id=user_id,
    )
    return await _execute_workflow_schedule_management(session_id=session_id, user_id=user_id, request=req)


@router.get("/api/workflow/schedules/{schedule_id}/history")
async def api_workflow_schedule_history(
    schedule_id: str,
    x_orion_session_id: Optional[str] = Header(None),
    user_id: Optional[str] = Query(default=None),
):
    from .main import bus
    if not bus:
        raise HTTPException(status_code=503, detail="bus unavailable")
    session_id = await ensure_session(x_orion_session_id, bus)
    req = WorkflowScheduleManageRequestV1(
        operation="history",
        request_id=str(uuid4()),
        schedule_id=schedule_id,
        include_history=True,
        session_id=session_id,
        origin_user_id=user_id,
    )
    return await _execute_workflow_schedule_management(session_id=session_id, user_id=user_id, request=req)


@router.post("/api/workflow/schedules/{schedule_id}/{operation}")
async def api_workflow_schedule_action(
    schedule_id: str,
    operation: str,
    payload: Optional[WorkflowScheduleUpdateRequest] = None,
    x_orion_session_id: Optional[str] = Header(None),
    user_id: Optional[str] = Query(default=None),
):
    normalized_operation = str(operation or "").strip().lower()
    if normalized_operation not in {"cancel", "pause", "resume", "update"}:
        raise HTTPException(status_code=400, detail="unsupported_operation")

    from .main import bus
    if not bus:
        raise HTTPException(status_code=503, detail="bus unavailable")
    session_id = await ensure_session(x_orion_session_id, bus)
    patch = payload.model_dump(exclude_none=True) if payload is not None else None
    req = WorkflowScheduleManageRequestV1(
        operation=normalized_operation,
        request_id=str(uuid4()),
        schedule_id=schedule_id,
        patch=patch,
        session_id=session_id,
        origin_user_id=user_id,
    )
    return await _execute_workflow_schedule_management(session_id=session_id, user_id=user_id, request=req)

# ======================================================================
# 📿 COLLAPSE MIRROR ENDPOINTS
# ======================================================================
@router.get("/schema/collapse")
def get_collapse_schema():
    """Exposes the CollapseMirrorEntry schema for UI templating."""
    logger.info("Fetching CollapseMirrorEntry schema")
    return JSONResponse(CollapseMirrorEntry.schema())


@router.post("/submit-collapse")
async def submit_collapse(data: dict):
    """
    Receives Collapse Mirror data and publishes it to the bus.
    """
    from .main import bus
    logger.info(f"🔥 /submit-collapse called with: {data}")

    if not bus or not bus.enabled:
        logger.error("Submission failed: OrionBus is disabled or not connected.")
        return JSONResponse(
            status_code=503,
            content={"success": False, "error": "OrionBus disabled or unavailable"},
        )

    try:
        # Normalize legacy ServiceRef objects coming from the UI.
        # Some clients still send {"service": "hub", "node": "..."}
        # but our canonical model uses {"name": "...", "node": "..."}.
        src = data.get("source")
        if isinstance(src, dict) and "name" not in src and "service" in src:
            src = dict(src)
            src["name"] = src.pop("service")
            data["source"] = src

        entry = CollapseMirrorEntry(**data).with_defaults()

        from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

        # Note: we do NOT explicitly set correlation_id here.
        # BaseEnvelope will generate a random one, but our worker heuristic (empty causality chain)
        # will treat it as ad-hoc and not persist it to DB.
        env = BaseEnvelope(
            kind="collapse.submit"
          , source=ServiceRef(name="hub", node=settings.NODE_NAME)
          , payload=entry.model_dump(mode="json")
        )

        await bus.publish(settings.CHANNEL_COLLAPSE_INTAKE, env)
        logger.info(
            "📡 Published Collapse Mirror → %s",
            settings.CHANNEL_COLLAPSE_INTAKE,
        )

        return {"success": True}

    except Exception as e:
        logger.error(f"❌ Hub publish error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )

@router.get("/api/debug/spark-last")
async def api_debug_spark_last(
    x_orion_session_id: Optional[str] = Header(None),
):
    # Legacy debug endpoint - likely broken but kept safe
    return JSONResponse(
        {
            "session_id": x_orion_session_id,
            "spark_meta": None,
            "note": "Spark debug deprecated in dumb hub.",
        }
    )


def _source_meta(*, kind: str, degraded: bool, limit: int, error: Optional[str] = None) -> Dict[str, Any]:
    return {
        "kind": kind,
        "degraded": degraded,
        "error": error,
        "query": {"limit": limit},
        "last_refreshed": datetime.now(timezone.utc).isoformat(),
    }


def _serialize_slice(result) -> Dict[str, Any]:
    return {
        "nodes": [node.model_dump(mode="json") for node in result.slice.nodes],
        "edges": [edge.model_dump(mode="json") for edge in result.slice.edges],
        "truncated": bool(result.truncated),
        "limits": dict(result.limits or {}),
        "query_kind": result.query_kind,
        "generated_at": result.generated_at,
        "details": dict(result.details or {}),
    }


def _graphdb_overview_payload(*, limit: int = 10) -> Dict[str, Any]:
    hotspot = SUBSTRATE_SEMANTIC_STORE.query_hotspot_region(limit_nodes=limit, limit_edges=max(20, limit * 2))
    contradiction = SUBSTRATE_SEMANTIC_STORE.query_contradiction_region(limit_nodes=limit, limit_edges=max(20, limit * 2))
    concept = SUBSTRATE_SEMANTIC_STORE.query_concept_region(limit_nodes=limit, limit_edges=max(20, limit * 2))

    source_kind = hotspot.source_kind
    degraded = bool(hotspot.degraded or contradiction.degraded or concept.degraded)
    error = hotspot.error or contradiction.error or concept.error

    return {
        "source": {
            **_source_meta(kind=source_kind, degraded=degraded, limit=limit, error=error),
            "query_kind": "overview",
            "semantic_backend": type(SUBSTRATE_SEMANTIC_STORE).__name__,
            "truncated": bool(hotspot.truncated or contradiction.truncated or concept.truncated),
        },
        "data": {
            "coherence": None,
            "identity_conflict": None,
            "goal_pressure": None,
            "concept_drift": None,
            "contradiction_count": len(contradiction.slice.nodes),
            "top_hotspots": [node.model_dump(mode="json") for node in hotspot.slice.nodes[:limit]],
            "top_tensions": [node.model_dump(mode="json") for node in hotspot.slice.nodes if node.node_kind == "tension"][:limit],
            "top_stabilizers": [node.model_dump(mode="json") for node in concept.slice.nodes[:limit]],
            "metacog_brief": None,
            "regions": {
                "hotspot": _serialize_slice(hotspot),
                "contradiction": _serialize_slice(contradiction),
                "concept": _serialize_slice(concept),
            },
        },
    }


def _graphdb_hotspots_payload(*, limit: int = 20) -> Dict[str, Any]:
    hotspot = SUBSTRATE_SEMANTIC_STORE.query_hotspot_region(limit_nodes=limit, limit_edges=max(40, limit * 2))
    contradiction = SUBSTRATE_SEMANTIC_STORE.query_contradiction_region(limit_nodes=limit, limit_edges=max(40, limit * 2))
    concept = SUBSTRATE_SEMANTIC_STORE.query_concept_region(limit_nodes=limit, limit_edges=max(40, limit * 2))

    degraded = bool(hotspot.degraded or contradiction.degraded or concept.degraded)
    error = hotspot.error or contradiction.error or concept.error

    return {
        "source": {
            **_source_meta(kind=hotspot.source_kind, degraded=degraded, limit=limit, error=error),
            "query_kind": "hotspots",
            "semantic_backend": type(SUBSTRATE_SEMANTIC_STORE).__name__,
            "truncated": bool(hotspot.truncated or contradiction.truncated or concept.truncated),
        },
        "data": {
            "active_regions": _serialize_slice(hotspot),
            "contradiction_hotspots": _serialize_slice(contradiction),
            "pressure_hotspots": _serialize_slice(hotspot),
            "drift_hotspots": _serialize_slice(concept),
        },
    }


def _sql_review_queue_payload(*, limit: int = 50) -> Dict[str, Any]:
    snapshot = SUBSTRATE_REVIEW_QUEUE_STORE.snapshot(limit=limit)
    return {
        "source": _source_meta(
            kind=SUBSTRATE_REVIEW_QUEUE_STORE.source_kind(),
            degraded=SUBSTRATE_REVIEW_QUEUE_STORE.degraded(),
            limit=limit,
            error=SUBSTRATE_REVIEW_QUEUE_STORE.last_error(),
        ),
        "data": snapshot.model_dump(mode="json"),
    }


def _sql_review_executions_payload(*, limit: int = 50) -> Dict[str, Any]:
    records = SUBSTRATE_REVIEW_TELEMETRY_STORE.query(GraphReviewTelemetryQueryV1(limit=limit))
    return {
        "source": _source_meta(
            kind=SUBSTRATE_REVIEW_TELEMETRY_STORE.source_kind(),
            degraded=SUBSTRATE_REVIEW_TELEMETRY_STORE.degraded(),
            limit=limit,
            error=SUBSTRATE_REVIEW_TELEMETRY_STORE.last_error(),
        ),
        "data": [record.model_dump(mode="json") for record in records],
    }


def _sql_telemetry_summary_payload(*, limit: int = 200) -> Dict[str, Any]:
    summary = SUBSTRATE_REVIEW_TELEMETRY_STORE.summary(GraphReviewTelemetryQueryV1(limit=limit))
    recommendations = GraphReviewCalibrationAnalyzer().recommend(summary=summary)
    return {
        "source": _source_meta(
            kind=SUBSTRATE_REVIEW_TELEMETRY_STORE.source_kind(),
            degraded=SUBSTRATE_REVIEW_TELEMETRY_STORE.degraded(),
            limit=limit,
            error=SUBSTRATE_REVIEW_TELEMETRY_STORE.last_error(),
        ),
        "data": {
            "summary": summary.model_dump(mode="json"),
            "recommendations": [rec.model_dump(mode="json") for rec in recommendations],
        },
    }


def _sql_calibration_payload(*, limit: int = 20) -> Dict[str, Any]:
    summary = SUBSTRATE_REVIEW_TELEMETRY_STORE.summary(GraphReviewTelemetryQueryV1(limit=200))
    recommendations = GraphReviewCalibrationAnalyzer().recommend(summary=summary)
    baseline_policy = GraphReviewCyclePolicyV1().model_dump(mode="json")
    inspection = SUBSTRATE_POLICY_STORE.inspect(audit_limit=limit)
    actives_cal = list(inspection.active_profiles)
    if actives_cal:
        actives_cal.sort(key=lambda p: p.activated_at or p.created_at, reverse=True)
        active_profile_dump = actives_cal[0].model_dump(mode="json")
    else:
        active_profile_dump = baseline_policy
    source_kind = "postgres" if (
        SUBSTRATE_POLICY_STORE.source_kind() == "postgres"
        and SUBSTRATE_REVIEW_TELEMETRY_STORE.source_kind() == "postgres"
    ) else SUBSTRATE_POLICY_STORE.source_kind()
    return {
        "source": _source_meta(
            kind=source_kind,
            degraded=bool(SUBSTRATE_POLICY_STORE.degraded() or SUBSTRATE_REVIEW_TELEMETRY_STORE.degraded()),
            limit=limit,
            error=SUBSTRATE_POLICY_STORE.last_error() or SUBSTRATE_REVIEW_TELEMETRY_STORE.last_error(),
        ),
        "data": {
            "active_profile": active_profile_dump,
            "staged_profiles": [item.model_dump(mode="json") for item in inspection.staged_profiles],
            "recent_audit_events": [item.model_dump(mode="json") for item in inspection.recent_audit_events],
            "rolled_back_profiles": [item.model_dump(mode="json") for item in inspection.rolled_back_profiles],
            "advisory_recommendations": [rec.model_dump(mode="json") for rec in recommendations],
        },
    }


def _sql_policy_comparison_payload(
    *,
    pair_mode: str = "baseline_vs_active",
    baseline_profile_id: str | None = None,
    candidate_profile_id: str | None = None,
    window_seconds: int = 86400,
    sample_limit: int = 500,
) -> Dict[str, Any]:
    try:
        if not isinstance(pair_mode, str):
            pair_mode = "baseline_vs_active"
        if not isinstance(baseline_profile_id, str):
            baseline_profile_id = None
        if not isinstance(candidate_profile_id, str):
            candidate_profile_id = None
        if not isinstance(window_seconds, int):
            window_seconds = 86400
        if not isinstance(sample_limit, int):
            sample_limit = 500
        request = SubstratePolicyComparisonRequestV1(
            pair_mode=pair_mode,  # type: ignore[arg-type]
            baseline_profile_id=baseline_profile_id,
            candidate_profile_id=candidate_profile_id,
            window_seconds=window_seconds,
            sample_limit=sample_limit,
        )
        data = SUBSTRATE_POLICY_COMPARISON.compare(request=request)
        source_kind = "postgres" if (
            SUBSTRATE_POLICY_STORE.source_kind() == "postgres"
            and SUBSTRATE_REVIEW_TELEMETRY_STORE.source_kind() == "postgres"
        ) else (
            "fallback" if (SUBSTRATE_POLICY_STORE.degraded() or SUBSTRATE_REVIEW_TELEMETRY_STORE.degraded()) else (
                "sqlite" if (
                    SUBSTRATE_POLICY_STORE.source_kind() == "sqlite"
                    or SUBSTRATE_REVIEW_TELEMETRY_STORE.source_kind() == "sqlite"
                ) else "sql"
            )
        )
        return {
            "source": _source_meta(
                kind=source_kind,
                degraded=bool(SUBSTRATE_POLICY_STORE.degraded() or SUBSTRATE_REVIEW_TELEMETRY_STORE.degraded()),
                limit=sample_limit,
                error=SUBSTRATE_POLICY_STORE.last_error() or SUBSTRATE_REVIEW_TELEMETRY_STORE.last_error(),
            ),
            "data": data,
        }
    except Exception as exc:
        return {
            "source": _source_meta(kind="sql", degraded=True, limit=sample_limit, error=str(exc)),
            "data": {
                "pair": {
                    "mode": pair_mode,
                    "baseline_profile_id": baseline_profile_id,
                    "candidate_profile_id": candidate_profile_id,
                },
                "report": {
                    "verdict": "insufficient_data",
                    "confidence": 0.2,
                    "notes": [f"comparison_error:{exc}"],
                },
                "advisory": {"mutating": False, "message": "comparison failed safely"},
            },
        }


def _substrate_source_posture() -> Dict[str, Any]:
    control_kind = SUBSTRATE_REVIEW_QUEUE_STORE.source_kind()
    telemetry_kind = SUBSTRATE_REVIEW_TELEMETRY_STORE.source_kind()
    semantic_kind_raw = getattr(SUBSTRATE_SEMANTIC_STORE, "source_kind", "graphdb")
    semantic_kind = semantic_kind_raw() if callable(semantic_kind_raw) else semantic_kind_raw
    policy_kind = SUBSTRATE_POLICY_STORE.source_kind()
    control_degraded = bool(SUBSTRATE_REVIEW_QUEUE_STORE.degraded() or SUBSTRATE_REVIEW_TELEMETRY_STORE.degraded())
    semantic_degraded_raw = getattr(SUBSTRATE_SEMANTIC_STORE, "degraded", False)
    semantic_degraded = bool(semantic_degraded_raw() if callable(semantic_degraded_raw) else semantic_degraded_raw)
    policy_degraded = bool(SUBSTRATE_POLICY_STORE.degraded())
    control_posture = "degraded" if control_degraded else ("postgres" if control_kind == "postgres" else "fallback")
    return {
        "control_plane": {
            "kind": control_kind,
            "telemetry_kind": telemetry_kind,
            "degraded": control_degraded,
            "posture": control_posture,
            "error": SUBSTRATE_REVIEW_QUEUE_STORE.last_error() or SUBSTRATE_REVIEW_TELEMETRY_STORE.last_error(),
        },
        "semantic": {
            "kind": semantic_kind,
            "degraded": semantic_degraded,
            "posture": "degraded" if semantic_degraded else semantic_kind,
            "error": getattr(SUBSTRATE_SEMANTIC_STORE, "last_error", lambda: None)(),
        },
        "policy": {
            "kind": policy_kind,
            "degraded": policy_degraded,
            "posture": "degraded" if policy_degraded else ("postgres" if policy_kind == "postgres" else "fallback"),
            "error": SUBSTRATE_POLICY_STORE.last_error(),
        },
    }


def _parse_substrate_status_instant(value: object) -> datetime | None:
    """Parse ISO-8601 strings or datetimes for substrate status clock comparisons."""
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    if isinstance(value, str):
        text = value.strip().replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    return None


def _last_execution_predates_active_profile(
    last_execution: Dict[str, Any] | None,
    active_profile: Dict[str, Any],
) -> bool | None:
    """True when the newest telemetry row is older than the active policy's activation time."""
    if not last_execution or not active_profile.get("activated_at"):
        return None
    selected = _parse_substrate_status_instant(last_execution.get("selected_at"))
    activated = _parse_substrate_status_instant(active_profile.get("activated_at"))
    if selected is None or activated is None:
        return None
    return selected < activated


def _substrate_runtime_status_payload(*, queue_limit: int = 20, telemetry_limit: int = 20) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    queue_snapshot = SUBSTRATE_REVIEW_QUEUE_STORE.snapshot(limit=queue_limit)
    due_items = SUBSTRATE_REVIEW_QUEUE_STORE.list_eligible(now=now, limit=queue_limit)
    recent_exec = SUBSTRATE_REVIEW_TELEMETRY_STORE.query(GraphReviewTelemetryQueryV1(limit=1))
    telemetry_summary = SUBSTRATE_REVIEW_TELEMETRY_STORE.summary(GraphReviewTelemetryQueryV1(limit=telemetry_limit))
    policy_inspection = SUBSTRATE_POLICY_STORE.inspect(audit_limit=5)
    actives = list(policy_inspection.active_profiles)
    if actives:
        actives.sort(key=lambda p: p.activated_at or p.created_at, reverse=True)
        active_profile = actives[0].model_dump(mode="json")
    else:
        active_profile = {}
    last_execution = recent_exec[0].model_dump(mode="json") if recent_exec else None
    last_predates = _last_execution_predates_active_profile(last_execution, active_profile)
    next_due = min((item.next_review_at for item in due_items), default=None)

    return {
        "generated_at": now.isoformat(),
        "summary": {
            "queue_count": len(queue_snapshot.queue_items),
            "due_count": len(due_items),
            "next_due_at": next_due.isoformat() if next_due else None,
            "next_item": (
                {
                    "queue_item_id": due_items[0].queue_item_id,
                    "target_zone": due_items[0].target_zone,
                    "subject_ref": due_items[0].subject_ref,
                    "remaining_cycles": due_items[0].cycle_budget.remaining_cycles,
                    "next_review_at": due_items[0].next_review_at.isoformat(),
                }
                if due_items
                else None
            ),
            "last_execution": last_execution,
            "last_execution_predates_active_profile": last_predates,
            "telemetry_count": (
                telemetry_summary.total_executions
                + telemetry_summary.total_noops
                + telemetry_summary.total_suppressed
                + telemetry_summary.total_terminated
                + telemetry_summary.total_failed
            ),
            "policy_active_profile_id": active_profile.get("profile_id"),
            "policy_active_profile": active_profile,
        },
        "source": _substrate_source_posture(),
    }


def _mutation_manual_apply_policy_allows() -> bool:
    return str(os.getenv("SUBSTRATE_MUTATION_MANUAL_APPLY_ENABLED", "false")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _require_mutation_operator_guard(token: str | None) -> None:
    expected = str(os.getenv("SUBSTRATE_MUTATION_OPERATOR_TOKEN", "")).strip()
    if not expected:
        raise HTTPException(status_code=503, detail="mutation_operator_token_not_configured")
    if not token or token.strip() != expected:
        raise HTTPException(status_code=403, detail="operator_guard_rejected")


class _ManualCyclePatchApplier(PatchApplier):
    def __init__(self, *, surfaces: dict[str, dict[str, Any]], allow_apply: bool, blocked_reason: str | None = None) -> None:
        super().__init__(surfaces=surfaces)
        self.allow_apply = bool(allow_apply)
        self.blocked_reason = blocked_reason
        self.attempted = 0
        self.blocked = 0
        self.completed = 0

    def apply(self, *, proposal, decision):  # type: ignore[override]
        if decision.action == "auto_promote":
            self.attempted += 1
        if not self.allow_apply:
            if decision.action == "auto_promote":
                self.blocked += 1
            return None
        adoption = super().apply(proposal=proposal, decision=decision)
        if decision.action == "auto_promote":
            if adoption is None:
                self.blocked += 1
            else:
                self.completed += 1
        return adoption


class _ScheduledCyclePatchApplier(PatchApplier):
    def __init__(
        self,
        *,
        surfaces: dict[str, dict[str, Any]],
        allow_apply: bool,
        allowed_classes: set[str] | None = None,
    ) -> None:
        super().__init__(surfaces=surfaces)
        self.allow_apply = bool(allow_apply)
        self.allowed_classes = set(allowed_classes or set())
        self.attempted = 0
        self.blocked = 0
        self.completed = 0

    def apply(self, *, proposal, decision):  # type: ignore[override]
        if decision.action == "auto_promote":
            self.attempted += 1
        if self.allowed_classes and proposal.mutation_class not in self.allowed_classes:
            if decision.action == "auto_promote":
                self.blocked += 1
            return None
        if not self.allow_apply:
            if decision.action == "auto_promote":
                self.blocked += 1
            return None
        adoption = super().apply(proposal=proposal, decision=decision)
        if decision.action == "auto_promote":
            if adoption is None:
                self.blocked += 1
            else:
                self.completed += 1
        return adoption


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = str(os.getenv(name, "true" if default else "false")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, *, default: int, minimum: int = 1, maximum: int = 256) -> int:
    raw = str(os.getenv(name, str(default))).strip()
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


def _env_float(name: str, *, default: float, minimum: float, maximum: float) -> float:
    raw = str(os.getenv(name, str(default))).strip()
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


def _emit_substrate_autonomy_scheduler_log(*, payload: Dict[str, Any]) -> None:
    logger.info("substrate_mutation_scheduler %s", json.dumps(payload, sort_keys=True, default=str))


def substrate_autonomy_runtime_supported() -> tuple[bool, str]:
    if not SUBSTRATE_MUTATION_STORE.postgres_url:
        return False, "postgres_url_unset"
    if SUBSTRATE_MUTATION_STORE.source_kind() != "postgres":
        return False, f"unsupported_store_kind:{SUBSTRATE_MUTATION_STORE.source_kind()}"
    if SUBSTRATE_MUTATION_STORE.degraded():
        return False, "mutation_store_degraded"
    return True, "supported"


def execute_substrate_mutation_scheduled_cycle(
    *,
    now: datetime | None = None,
    telemetry_override: list[Dict[str, Any]] | None = None,
    class_metrics_override: Dict[str, Dict[str, float]] | None = None,
    post_adoption_delta_by_target_surface_override: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    tick_now = now or datetime.now(timezone.utc)
    tick_id = f"mutation-scheduler-{uuid4()}"
    interval_sec = float(str(os.getenv("SUBSTRATE_AUTONOMY_INTERVAL_SEC", "30")).strip() or "30")
    if not _env_flag("SUBSTRATE_AUTONOMY_ENABLED", default=False):
        payload = {
            "event": "mutation_scheduler_tick",
            "tick_id": tick_id,
            "status": "disabled",
            "interval_sec": interval_sec,
            "at": tick_now.isoformat(),
        }
        _emit_substrate_autonomy_scheduler_log(payload=payload)
        return payload
    supported, reason = substrate_autonomy_runtime_supported()
    if not supported:
        payload = {
            "event": "mutation_scheduler_tick",
            "tick_id": tick_id,
            "status": "unsafe_mode_noop",
            "reason": reason,
            "interval_sec": interval_sec,
            "at": tick_now.isoformat(),
            "store_kind": SUBSTRATE_MUTATION_STORE.source_kind(),
            "store_degraded": SUBSTRATE_MUTATION_STORE.degraded(),
        }
        _emit_substrate_autonomy_scheduler_log(payload=payload)
        return payload

    lock_acquired = SUBSTRATE_AUTONOMY_LOCAL_CYCLE_LOCK.acquire(blocking=False)
    if not lock_acquired:
        payload = {
            "event": "mutation_scheduler_tick",
            "tick_id": tick_id,
            "status": "blocked",
            "reason": "local_cycle_lock_not_acquired",
            "interval_sec": interval_sec,
            "at": tick_now.isoformat(),
        }
        _emit_substrate_autonomy_scheduler_log(payload=payload)
        return payload

    traces: list[dict[str, Any]] = []
    try:
        proposals_enabled = _env_flag("SUBSTRATE_AUTONOMY_PROPOSALS_ENABLED", default=True)
        routing_proposals_enabled = _env_flag("SUBSTRATE_AUTONOMY_ROUTING_PROPOSALS_ENABLED", default=True)
        cognitive_proposals_enabled = _env_flag("SUBSTRATE_AUTONOMY_COGNITIVE_PROPOSALS_ENABLED", default=False)
        apply_enabled_global = _env_flag("SUBSTRATE_AUTONOMY_APPLY_ENABLED", default=False)
        routing_apply_enabled = _env_flag("SUBSTRATE_AUTONOMY_ROUTING_APPLY_ENABLED", default=False)
        apply_enabled = bool(apply_enabled_global and routing_apply_enabled)
        monitor_enabled = _env_flag("SUBSTRATE_AUTONOMY_MONITOR_ENABLED", default=True)
        routing_rollback_threshold = _env_float(
            "SUBSTRATE_AUTONOMY_ROUTING_ROLLBACK_DELTA_THRESHOLD",
            default=-0.05,
            minimum=-1.0,
            maximum=0.0,
        )

        applier = _ScheduledCyclePatchApplier(
            surfaces=SUBSTRATE_MUTATION_SURFACES,
            allow_apply=apply_enabled,
            allowed_classes={"routing_threshold_patch"},
        )
        corpus = ReplayCorpusRegistry(
            corpus_by_class={
                "routing_threshold_patch": "replay-routing-v1",
                "recall_weighting_patch": "replay-recall-v1",
                "graph_consolidation_param_patch": "replay-consolidation-v1",
                "approved_prompt_profile_variant_promotion": "replay-prompt-profile-v1",
            },
            baseline_metric_ref_by_class={
                "routing_threshold_patch": "baseline-routing-v1",
                "recall_weighting_patch": "baseline-recall-v1",
                "graph_consolidation_param_patch": "baseline-consolidation-v1",
                "approved_prompt_profile_variant_promotion": "baseline-prompt-profile-v1",
            },
        )
        worker = SubstrateAdaptationWorker(
            store=SUBSTRATE_MUTATION_STORE,
            detectors=MutationDetectors(allow_cognitive_lane=cognitive_proposals_enabled),
            pressure=PressureAccumulator(),
            proposals=ProposalFactory(),
            trial_runner=SubstrateTrialRunner(scorer=ClassSpecificScorer(), corpus_registry=corpus),
            decision_engine=DecisionEngine(),
            applier=applier,
            monitor=PostAdoptionMonitor(regression_threshold=routing_rollback_threshold),
            budget=AdaptationCycleBudget(
                max_signals=0
                if not routing_proposals_enabled
                else _env_int("SUBSTRATE_AUTONOMY_MAX_SIGNALS", default=32),
                max_proposals=0
                if (not proposals_enabled or not routing_proposals_enabled)
                else _env_int("SUBSTRATE_AUTONOMY_MAX_PROPOSALS", default=8, maximum=64),
                max_trials=_env_int("SUBSTRATE_AUTONOMY_MAX_TRIALS", default=8, maximum=64),
                max_adoptions=_env_int("SUBSTRATE_AUTONOMY_MAX_ADOPTIONS", default=1, maximum=8),
            ),
            kill_switch_env="SUBSTRATE_AUTONOMY_ENABLED",
            trace_logger=traces.append,
        )

        if telemetry_override is not None:
            telemetry = [GraphReviewTelemetryRecordV1.model_validate(item or {}) for item in telemetry_override]
        else:
            telemetry = SUBSTRATE_REVIEW_TELEMETRY_STORE.query(
                GraphReviewTelemetryQueryV1(
                    limit=worker.budget.max_signals,
                    invocation_surface="operator_review",
                )
            )
        # Narrow ramp scope: scheduler-driven mutation autonomy is routing-class only.
        allowed_zones = {"autonomy_graph"}
        if cognitive_proposals_enabled:
            allowed_zones.add("self_relationship_graph")
        telemetry = [item for item in telemetry if item.target_zone in allowed_zones]

        _emit_substrate_autonomy_scheduler_log(
            payload={
                "event": "mutation_scheduler_tick",
                "tick_id": tick_id,
                "status": "running",
                "interval_sec": interval_sec,
                "at": tick_now.isoformat(),
                "proposals_enabled": proposals_enabled,
                "routing_proposals_enabled": routing_proposals_enabled,
                "cognitive_proposals_enabled": cognitive_proposals_enabled,
                "apply_enabled": apply_enabled,
                "apply_enabled_global": apply_enabled_global,
                "routing_apply_enabled": routing_apply_enabled,
                "monitor_enabled": monitor_enabled,
                "routing_rollback_delta_threshold": routing_rollback_threshold,
            }
        )
        result = worker.run_cycle(
            telemetry=telemetry,
            measured_metrics_by_proposal={},
            measured_metrics_by_class=class_metrics_override or None,
            post_adoption_delta_by_target_surface=(
                post_adoption_delta_by_target_surface_override if monitor_enabled else None
            ),
            replay_telemetry=telemetry,
            now=tick_now,
        )
        scheduler_summary = {
            "event": "mutation_scheduler_cycle_finished",
            "tick_id": tick_id,
            "status": "completed",
            "notes": list(result.get("notes") or []),
            "signals_processed": int(result.get("signals", 0)),
            "proposals_created": int(result.get("proposals", 0)),
            "trials_executed": int(result.get("trials", 0)),
            "adoptions_completed": int(result.get("adoptions", 0)),
            "decisions_made": len([event for event in traces if event.get("event") == "mutation_decision_recorded"]),
            "applies_attempted": applier.attempted,
            "applies_blocked": applier.blocked,
            "applies_executed": applier.completed,
            "routing_proposals_enabled": routing_proposals_enabled,
            "cognitive_proposals_enabled": cognitive_proposals_enabled,
            "routing_apply_enabled": routing_apply_enabled,
            "apply_enabled_global": apply_enabled_global,
            "routing_only_scope": not cognitive_proposals_enabled,
            "db_lock_acquired": any(
                event.get("event") == "mutation_lock_acquire" and bool(event.get("lock_acquired"))
                for event in traces
            ),
            "db_lock_blocked": any(
                event.get("event") == "mutation_lock_acquire" and not bool(event.get("lock_acquired"))
                for event in traces
            ),
        }
        _emit_substrate_autonomy_scheduler_log(payload=scheduler_summary)
        return {
            "tick_id": tick_id,
            "status": "completed",
            "result": result,
            "summary": scheduler_summary,
            "trace": traces,
        }
    finally:
        SUBSTRATE_AUTONOMY_LOCAL_CYCLE_LOCK.release()


def _mutation_lineage_id_from_event(event: Dict[str, Any]) -> str:
    for key in ("lineage_id", "signal_id", "proposal_id", "pressure_id", "queue_item_id", "trial_id", "decision_id", "rollback_id"):
        value = event.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return "unknown"


def _emit_mutation_lifecycle_logs(*, route_invocation_id: str, events: list[Dict[str, Any]]) -> None:
    bounded = events[:200]
    for event in bounded:
        payload = {
            "event_kind": "substrate_mutation_lifecycle_v1",
            "route_invocation_id": route_invocation_id,
            "event": event.get("event"),
            "cycle_id": event.get("cycle_id"),
            "lineage_id": _mutation_lineage_id_from_event(event),
            "signal_id": event.get("signal_id"),
            "pressure_key": event.get("pressure_key"),
            "proposal_id": event.get("proposal_id"),
            "queue_item_id": event.get("queue_item_id"),
            "trial_id": event.get("trial_id"),
            "decision": event.get("decision"),
            "queue_status_before": event.get("queue_status_before"),
            "queue_status_after": event.get("queue_status_after"),
            "surface_key": event.get("surface_key"),
            "blocked_reason": event.get("blocked_reason"),
            "applied": event.get("applied"),
            "notes": event.get("notes"),
        }
        logger.info("substrate_mutation_lifecycle %s", json.dumps(payload, sort_keys=True, default=str))


def _execute_substrate_mutation_cycle(*, request: SubstrateMutationExecuteRequest) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    route_invocation_id = f"mutation-manual-{uuid4()}"
    before_signal_count = len(SUBSTRATE_MUTATION_STORE._signals)
    before_pressure_keys = set(SUBSTRATE_MUTATION_STORE._pressures.keys())
    before_queue_status_by_id = {item.queue_item_id: item.status for item in SUBSTRATE_MUTATION_STORE._queue.values()}

    apply_policy_allowed = _mutation_manual_apply_policy_allows()
    apply_allowed = bool((not request.dry_run) and request.apply_enabled and apply_policy_allowed)
    apply_blockers: list[str] = []
    if request.apply_enabled and request.dry_run:
        apply_blockers.append("dry_run_forces_apply_disabled")
    if request.apply_enabled and not apply_policy_allowed:
        apply_blockers.append("manual_apply_policy_disallows_override")

    traces: list[dict[str, Any]] = []
    applier = _ManualCyclePatchApplier(
        surfaces=SUBSTRATE_MUTATION_SURFACES,
        allow_apply=apply_allowed,
        blocked_reason=";".join(apply_blockers) if apply_blockers else None,
    )
    corpus = ReplayCorpusRegistry(
        corpus_by_class={
            "routing_threshold_patch": "replay-routing-v1",
            "recall_weighting_patch": "replay-recall-v1",
            "graph_consolidation_param_patch": "replay-consolidation-v1",
            "approved_prompt_profile_variant_promotion": "replay-prompt-profile-v1",
        },
        baseline_metric_ref_by_class={
            "routing_threshold_patch": "baseline-routing-v1",
            "recall_weighting_patch": "baseline-recall-v1",
            "graph_consolidation_param_patch": "baseline-consolidation-v1",
            "approved_prompt_profile_variant_promotion": "baseline-prompt-profile-v1",
        },
    )
    worker = SubstrateAdaptationWorker(
        store=SUBSTRATE_MUTATION_STORE,
        detectors=MutationDetectors(
            allow_cognitive_lane=_env_flag("SUBSTRATE_AUTONOMY_COGNITIVE_PROPOSALS_ENABLED", default=False)
        ),
        pressure=PressureAccumulator(),
        proposals=ProposalFactory(),
        trial_runner=SubstrateTrialRunner(scorer=ClassSpecificScorer(), corpus_registry=corpus),
        decision_engine=DecisionEngine(),
        applier=applier,
        monitor=PostAdoptionMonitor(),
        budget=AdaptationCycleBudget(
            max_signals=request.max_signals,
            max_proposals=request.max_proposals,
            max_trials=request.max_trials,
            max_adoptions=1,
        ),
        trace_logger=traces.append,
    )

    telemetry: list[GraphReviewTelemetryRecordV1]
    if request.telemetry:
        telemetry = []
        for item in request.telemetry[: request.max_signals]:
            payload = dict(item or {})
            payload.setdefault("invocation_surface", "operator_review")
            payload.setdefault("execution_outcome", "failed")
            payload.setdefault("selection_reason", "manual_mutation_route")
            payload.setdefault("runtime_duration_ms", 10)
            payload.setdefault("anchor_scope", "orion")
            payload.setdefault("subject_ref", "entity:orion")
            payload.setdefault("target_zone", "autonomy_graph")
            telemetry.append(GraphReviewTelemetryRecordV1.model_validate(payload))
    else:
        telemetry = SUBSTRATE_REVIEW_TELEMETRY_STORE.query(
            GraphReviewTelemetryQueryV1(limit=request.max_signals, invocation_surface="operator_review")
        )
    result_first = worker.run_cycle(
        telemetry=telemetry,
        measured_metrics_by_proposal={},
        measured_metrics_by_class=request.class_metrics or None,
        post_adoption_delta_by_target_surface=request.post_adoption_delta_by_target_surface or None,
        replay_telemetry=telemetry,
        now=now,
    )
    # Queue items are created with due_at ~= current time; run one immediate settle pass to deterministically
    # execute trial/decision/apply paths within a single manual invocation.
    result_second = worker.run_cycle(
        telemetry=[],
        measured_metrics_by_proposal={},
        measured_metrics_by_class=request.class_metrics or None,
        post_adoption_delta_by_target_surface=request.post_adoption_delta_by_target_surface or None,
        replay_telemetry=telemetry,
        now=now + timedelta(milliseconds=1),
    )
    result = {
        "signals": int(result_first.get("signals", 0)) + int(result_second.get("signals", 0)),
        "proposals": int(result_first.get("proposals", 0)) + int(result_second.get("proposals", 0)),
        "trials": int(result_first.get("trials", 0)) + int(result_second.get("trials", 0)),
        "adoptions": int(result_first.get("adoptions", 0)) + int(result_second.get("adoptions", 0)),
        "notes": list(result_first.get("notes") or []) + list(result_second.get("notes") or []),
    }
    _emit_mutation_lifecycle_logs(route_invocation_id=route_invocation_id, events=traces)

    pressure_updates = [event for event in traces if event.get("event") == "mutation_pressure_recorded"]
    proposal_events = [event for event in traces if event.get("event") == "mutation_proposal_enqueued"]
    trial_events = [event for event in traces if event.get("event") == "mutation_trial_recorded"]
    decision_events = [event for event in traces if event.get("event") == "mutation_decision_recorded"]
    apply_block_events = [event for event in traces if event.get("event") == "mutation_apply_blocked"]
    rollback_events = [event for event in traces if event.get("event") == "mutation_rollback_recorded"]
    queue_item_ids = sorted({str(event.get("queue_item_id")) for event in traces if event.get("queue_item_id")})
    queue_status_changes: list[Dict[str, str]] = []
    for item in SUBSTRATE_MUTATION_STORE._queue.values():
        before_status = before_queue_status_by_id.get(item.queue_item_id)
        if before_status is not None and before_status != item.status:
            queue_status_changes.append(
                {
                    "queue_item_id": item.queue_item_id,
                    "before": before_status,
                    "after": item.status,
                }
            )
    blockers = list(result.get("notes") or [])
    blockers.extend(apply_blockers)
    blockers.extend(str(event.get("blocked_reason")) for event in apply_block_events if event.get("blocked_reason"))

    return {
        "generated_at": now.isoformat(),
        "request": {
            "route_invocation_id": route_invocation_id,
            "dry_run": request.dry_run,
            "max_signals": request.max_signals,
            "max_proposals": request.max_proposals,
            "max_trials": request.max_trials,
            "apply_enabled_requested": request.apply_enabled,
            "apply_enabled_effective": apply_allowed,
            "apply_policy_allowed": apply_policy_allowed,
            "single_cycle": True,
            "invocation_surface": "operator_review",
        },
        "summary": {
            "signals_produced": int(result.get("signals", 0)),
            "pressures_updated": len(pressure_updates),
            "proposals_created": len(proposal_events),
            "queue_items_touched": queue_item_ids,
            "queue_status_changes": queue_status_changes,
            "trials_run": len(trial_events),
            "decisions_made": len(decision_events),
            "applies_attempted": applier.attempted,
            "applies_blocked": applier.blocked,
            "applies_completed": applier.completed,
            "monitoring_windows_opened": int(result.get("adoptions", 0)),
            "rollbacks_recorded": len(rollback_events),
            "kill_switch_or_policy_blockers": blockers,
            "pressure_keys_before": sorted(before_pressure_keys),
            "pressure_keys_after": sorted(SUBSTRATE_MUTATION_STORE._pressures.keys()),
            "signal_count_before": before_signal_count,
            "signal_count_after": len(SUBSTRATE_MUTATION_STORE._signals),
        },
        "source": {
            "mutation_store_kind": SUBSTRATE_MUTATION_STORE.source_kind(),
            "mutation_store_degraded": SUBSTRATE_MUTATION_STORE.degraded(),
            "mutation_store_error": SUBSTRATE_MUTATION_STORE.last_error(),
            "review_telemetry_kind": SUBSTRATE_REVIEW_TELEMETRY_STORE.source_kind(),
        },
        "trace": {
            "events": traces,
            "notes": list(result.get("notes") or []),
        },
    }


def _mutation_lineage_payload(*, proposal_id: str | None = None, limit: int = 20) -> Dict[str, Any]:
    if proposal_id:
        lifecycle = SUBSTRATE_MUTATION_STORE.lifecycle_for_proposal(proposal_id)
        lifecycles = [lifecycle] if lifecycle is not None else []
    else:
        lifecycles = SUBSTRATE_MUTATION_STORE.recent_lifecycles(limit=limit)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": _source_meta(
            kind=SUBSTRATE_MUTATION_STORE.source_kind(),
            degraded=SUBSTRATE_MUTATION_STORE.degraded(),
            limit=limit,
            error=SUBSTRATE_MUTATION_STORE.last_error(),
        ),
        "data": {
            "lifecycles": lifecycles,
            "active_surfaces": SUBSTRATE_MUTATION_STORE.active_surfaces_snapshot(),
            "recent_blocked_applies": SUBSTRATE_MUTATION_STORE.recent_blocked_applies(limit=limit),
            "recent_rollbacks": SUBSTRATE_MUTATION_STORE.recent_rollbacks(limit=limit),
        },
    }


def _routing_replay_inspection_payload(*, limit: int = 50) -> Dict[str, Any]:
    telemetry = SUBSTRATE_REVIEW_TELEMETRY_STORE.query(
        GraphReviewTelemetryQueryV1(
            limit=max(1, min(limit, 200)),
            invocation_surface="operator_review",
        )
    )
    routing_proposals = [
        proposal
        for proposal in SUBSTRATE_MUTATION_STORE._proposals.values()
        if proposal.mutation_class == "routing_threshold_patch"
    ]
    routing_proposals.sort(key=lambda item: item.created_at, reverse=True)
    proposal = routing_proposals[0] if routing_proposals else ProposalFactory().from_pressure(
        MutationPressureV1(
            anchor_scope="orion",
            subject_ref="entity:orion",
            target_surface="routing",
            pressure_kind="runtime_failure",
            pressure_score=5.0,
            evidence_refs=["telemetry:replay_inspection_seed"],
            source_signal_ids=["signal:replay_inspection_seed"],
        )
    )
    if proposal is None:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "data": {"error": "routing_proposal_unavailable"},
        }
    corpus = ReplayCorpusRegistry(
        corpus_by_class={"routing_threshold_patch": "replay-routing-v1"},
        baseline_metric_ref_by_class={"routing_threshold_patch": "baseline-routing-v1"},
    )
    trial_runner = SubstrateTrialRunner(
        scorer=ClassSpecificScorer(),
        corpus_registry=corpus,
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data": trial_runner.inspect_routing_replay(
            proposal=proposal,
            replay_records=telemetry,
        ),
    }


def _routing_live_ramp_posture_payload() -> Dict[str, Any]:
    proposals_enabled_global = _env_flag("SUBSTRATE_AUTONOMY_PROPOSALS_ENABLED", default=True)
    routing_proposals_enabled = _env_flag("SUBSTRATE_AUTONOMY_ROUTING_PROPOSALS_ENABLED", default=True)
    apply_enabled_global = _env_flag("SUBSTRATE_AUTONOMY_APPLY_ENABLED", default=False)
    routing_apply_enabled = _env_flag("SUBSTRATE_AUTONOMY_ROUTING_APPLY_ENABLED", default=False)
    monitor_enabled = _env_flag("SUBSTRATE_AUTONOMY_MONITOR_ENABLED", default=True)
    rollback_threshold = _env_float(
        "SUBSTRATE_AUTONOMY_ROUTING_ROLLBACK_DELTA_THRESHOLD",
        default=-0.05,
        minimum=-1.0,
        maximum=0.0,
    )
    decisions = sorted(
        SUBSTRATE_MUTATION_STORE._decisions.values(),
        key=lambda item: item.created_at,
        reverse=True,
    )
    routing_decision = next(
        (
            item
            for item in decisions
            if (
                (proposal := SUBSTRATE_MUTATION_STORE.get_proposal(item.proposal_id)) is not None
                and proposal.mutation_class == "routing_threshold_patch"
            )
        ),
        None,
    )
    adoptions = sorted(
        SUBSTRATE_MUTATION_STORE._adoptions.values(),
        key=lambda item: item.created_at,
        reverse=True,
    )
    routing_adoption = next(
        (
            item
            for item in adoptions
            if (
                (proposal := SUBSTRATE_MUTATION_STORE.get_proposal(item.proposal_id)) is not None
                and proposal.mutation_class == "routing_threshold_patch"
            )
        ),
        None,
    )
    rollbacks = sorted(
        SUBSTRATE_MUTATION_STORE._rollbacks.values(),
        key=lambda item: item.created_at,
        reverse=True,
    )
    routing_rollback = next(
        (
            item
            for item in rollbacks
            if (
                (proposal := SUBSTRATE_MUTATION_STORE.get_proposal(item.proposal_id)) is not None
                and proposal.mutation_class == "routing_threshold_patch"
            )
        ),
        None,
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": _source_meta(
            kind=SUBSTRATE_MUTATION_STORE.source_kind(),
            degraded=SUBSTRATE_MUTATION_STORE.degraded(),
            limit=20,
            error=SUBSTRATE_MUTATION_STORE.last_error(),
        ),
        "data": {
            "ramp_scope": "routing_threshold_patch_only",
            "proposals_enabled": bool(proposals_enabled_global and routing_proposals_enabled),
            "proposals_enabled_global": proposals_enabled_global,
            "routing_proposals_enabled": routing_proposals_enabled,
            "apply_enabled": bool(apply_enabled_global and routing_apply_enabled),
            "apply_enabled_global": apply_enabled_global,
            "routing_apply_enabled": routing_apply_enabled,
            "monitor_enabled": monitor_enabled,
            "routing_rollback_delta_threshold": rollback_threshold,
            "last_decision": routing_decision.model_dump(mode="json") if routing_decision is not None else None,
            "last_adoption": routing_adoption.model_dump(mode="json") if routing_adoption is not None else None,
            "last_rollback": routing_rollback.model_dump(mode="json") if routing_rollback is not None else None,
            "live_surface": inspect_chat_reflective_lane_threshold(),
        },
    }


def _recall_canary_rollups(*, limit: int = 50) -> dict[str, Any]:
    runs = SUBSTRATE_MUTATION_STORE.list_recall_canary_runs(limit=limit)
    judgments = SUBSTRATE_MUTATION_STORE.list_recall_canary_judgments(limit=limit)
    artifacts = SUBSTRATE_MUTATION_STORE.list_recall_canary_review_artifacts(limit=limit)
    judgment_counts: dict[str, int] = {
        "v2_better": 0,
        "v1_better": 0,
        "tie": 0,
        "both_bad": 0,
        "inconclusive": 0,
    }
    failure_mode_counts: dict[str, int] = {}
    for row in judgments:
        key = str(row.get("judgment") or "")
        if key in judgment_counts:
            judgment_counts[key] = judgment_counts.get(key, 0) + 1
        for mode in list(row.get("failure_modes") or []):
            m = str(mode or "")
            if not m:
                continue
            failure_mode_counts[m] = failure_mode_counts.get(m, 0) + 1
    last_review_artifact_at = artifacts[0]["created_at"] if artifacts else None
    return {
        "run_count": len(runs),
        "judgment_counts": judgment_counts,
        "failure_mode_counts": failure_mode_counts,
        "recent_runs": runs[:10],
        "recent_judgments": judgments[:10],
        "review_artifact_count": len(artifacts),
        "last_review_artifact_at": last_review_artifact_at,
    }


def _recall_canary_profile_catalog(*, limit: int = 50) -> tuple[list[dict[str, Any]], str | None]:
    rows = SUBSTRATE_MUTATION_STORE.list_recall_strategy_profiles(limit=limit)
    available_profiles: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        profile_id = str(row.get("profile_id") or "").strip()
        if not profile_id or profile_id in seen:
            continue
        seen.add(profile_id)
        raw_status = str(row.get("status") or "").strip() or "staged"
        available_profiles.append(
            {
                "profile_id": profile_id,
                "label": str(row.get("source_proposal_id") or profile_id),
                "status": "shadow_canary_review_only",
                "description": "Recall strategy profile available for shadow/canary review only.",
                "production_default": False,
                "live_apply_enabled": False,
                "profile_state": raw_status,
            }
        )
    default_canary_profile_id: str | None = None
    active = SUBSTRATE_MUTATION_STORE.active_recall_shadow_profile()
    if active is not None:
        active_id = str(active.profile_id or "").strip()
        if active_id and active_id in seen:
            default_canary_profile_id = active_id
    if not default_canary_profile_id and available_profiles:
        default_canary_profile_id = str(available_profiles[0].get("profile_id") or "")
    return available_profiles, (default_canary_profile_id or None)


def _failure_mode_to_pressure_category(mode: str | None) -> str:
    m = str(mode or "").strip().lower()
    if m == "missing_exact_anchor":
        return "missing_exact_anchor"
    if m == "irrelevant_semantic_neighbor":
        return "irrelevant_semantic_neighbor"
    if m in {"stale_memory", "stale_memory_selected"}:
        return "stale_memory_selected"
    if m == "unsupported_memory_claim":
        return "unsupported_memory_claim"
    return "recall_miss_or_dissatisfaction"


def _recommended_canary_action(*, active_shadow: dict[str, Any] | None, rollups: dict[str, Any], recall_readiness: dict[str, Any]) -> str:
    if not active_shadow:
        return "activate_shadow_profile_first"
    run_count = int(rollups.get("run_count") or 0)
    if run_count <= 0:
        return "run_more_canaries"
    counts = rollups.get("judgment_counts") or {}
    v2_better = int(counts.get("v2_better", 0))
    both_bad = int(counts.get("both_bad", 0))
    inconclusive = int(counts.get("inconclusive", 0))
    if both_bad + inconclusive >= max(2, run_count // 2):
        return "inspect_failures"
    if v2_better >= max(2, run_count // 2):
        if run_count < 5:
            return "run_more_canaries"
        recommendation = str((recall_readiness or {}).get("recommendation") or "")
        if recommendation in {"ready_for_shadow_expansion", "ready_for_operator_promotion"}:
            return "create_review_artifact"
    return "not_ready"


def _autonomy_readiness_payload() -> Dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()
    warnings: list[str] = []
    constitution = load_autonomy_constitution()
    constitution_surface_rows = [row.model_dump(mode="json") for row in constitution.surfaces]
    logger.info(
        "event=autonomy_readiness_policy_matrix_loaded surface_count=%s warning_count=%s",
        len(constitution_surface_rows),
        len(constitution.warnings),
    )
    payload: dict[str, Any] = {
        "schema_version": "autonomy_readiness.v1",
        "generated_at": generated_at,
        "overall": {
            "autonomy_level": "level_2_5_bounded_self_mutation",
            "summary": "bounded self-mutation with routing-only narrow live apply and recall/cognitive proposal-shadow controls",
            "safe_next_action": "build_recall_v2_manual_canary",
            "highest_risk": "misconfigured autonomy apply gate outside routing narrow class",
            "warnings": [],
        },
        "scheduler": {"enabled": False, "proposal_enabled": False, "apply_enabled": False, "source": "env/runtime", "gates": {}},
        "policy_matrix": {"surfaces": constitution_surface_rows},
        "surfaces": {"live": [], "shadow": [], "proposal_only": [], "blocked": []},
        "routing": {
            "current_controls": {},
            "recent_applies": [],
            "recent_rollbacks": [],
            "recent_blocked_applies": [],
            "pressure_summary": {},
        },
        "recall": {
            "production_mode": PRODUCTION_RECALL_MODE,
            "live_apply_enabled": RECALL_LIVE_APPLY_ENABLED,
            "active_shadow_profile": None,
            "staged_profiles": [],
            "readiness": None,
            "recent_eval_runs": [],
            "production_candidate_reviews": [],
            "manual_canary": {
                "run_count": 0,
                "judgment_counts": {},
                "failure_mode_counts": {},
                "recent_judgments": [],
                "review_artifact_count": 0,
                "last_review_artifact_at": None,
                "recommended_canary_action": "not_ready",
            },
            "warnings": [],
        },
        "cognitive": {
            "live_apply_enabled": COGNITIVE_LIVE_APPLY_ENABLED,
            "proposal_classes": [
                "cognitive_contradiction_reconciliation",
                "cognitive_identity_continuity_adjustment",
                "cognitive_stance_continuity_adjustment",
                "cognitive_social_continuity_repair",
            ],
            "counts_by_state": {},
            "recent_proposals": [],
            "warnings": [],
        },
        "pressure": {
            "top_pressure_keys": [],
            "recent_evidence": [],
            "high_confidence_unresolved": [],
            "warnings": [],
        },
        "recent_activity": {
            "applies": [],
            "rollbacks": [],
            "blocked_applies": [],
            "skipped": [],
            "staged": [],
            "reviews": [],
        },
        "safe_next_actions": [
            {"rank": 1, "action": "build_recall_v2_manual_canary", "reason": "recall is shadow/eval/review capable while production apply remains blocked"},
            {"rank": 2, "action": "run_cognitive_operator_review", "reason": "cognitive proposals are bounded to draft/review and need explicit operator workflow"},
        ],
        "warnings": warnings,
    }
    if constitution.warnings:
        warnings.extend([f"constitution:{item}" for item in constitution.warnings[:16]])
    try:
        g_autonomy = _env_flag("SUBSTRATE_AUTONOMY_ENABLED", default=False)
        g_prop = _env_flag("SUBSTRATE_AUTONOMY_PROPOSALS_ENABLED", default=True)
        g_apply = _env_flag("SUBSTRATE_AUTONOMY_APPLY_ENABLED", default=False)
        g_route_prop = _env_flag("SUBSTRATE_AUTONOMY_ROUTING_PROPOSALS_ENABLED", default=True)
        g_route_apply = _env_flag("SUBSTRATE_AUTONOMY_ROUTING_APPLY_ENABLED", default=False)
        payload["scheduler"] = {
            "enabled": bool(g_autonomy),
            "proposal_enabled": bool(g_autonomy and g_prop),
            "apply_enabled": bool(g_autonomy and g_apply),
            "source": "env/runtime",
            "gates": {
                "SUBSTRATE_AUTONOMY_ENABLED": {"value": g_autonomy, "effective": g_autonomy},
                "SUBSTRATE_AUTONOMY_PROPOSALS_ENABLED": {"value": g_prop, "effective": bool(g_autonomy and g_prop)},
                "SUBSTRATE_AUTONOMY_APPLY_ENABLED": {"value": g_apply, "effective": bool(g_autonomy and g_apply)},
                "SUBSTRATE_AUTONOMY_ROUTING_PROPOSALS_ENABLED": {"value": g_route_prop, "effective": bool(g_autonomy and g_prop and g_route_prop)},
                "SUBSTRATE_AUTONOMY_ROUTING_APPLY_ENABLED": {"value": g_route_apply, "effective": bool(g_autonomy and g_apply and g_route_apply)},
            },
        }
    except Exception as exc:
        warnings.append(f"scheduler_unavailable:{exc}")

    try:
        posture = _routing_live_ramp_posture_payload()
        payload["routing"]["current_controls"] = posture.get("data", {}).get("live_surface") or inspect_chat_reflective_lane_threshold()
        payload["routing"]["recent_applies"] = [row.model_dump(mode="json") for row in sorted(SUBSTRATE_MUTATION_STORE._adoptions.values(), key=lambda item: item.created_at, reverse=True)[:10]]
        payload["routing"]["recent_rollbacks"] = SUBSTRATE_MUTATION_STORE.recent_rollbacks(limit=10)
        payload["routing"]["recent_blocked_applies"] = SUBSTRATE_MUTATION_STORE.recent_blocked_applies(limit=10)
    except Exception as exc:
        warnings.append(f"routing_posture_unavailable:{exc}")

    try:
        live_surfaces = []
        for row in SUBSTRATE_MUTATION_STORE.active_surfaces_snapshot():
            if str(row.get("target_surface")) == "routing":
                live_surfaces.append(
                    {
                        "surface": "routing_threshold_patch",
                        "control_key": "routing.chat_reflective_lane_threshold",
                        "status": "live",
                        "apply_mode": "gated_auto",
                        "rollback_supported": True,
                    }
                )
        active_shadow = SUBSTRATE_MUTATION_STORE.active_recall_shadow_profile()
        staged_profiles = SUBSTRATE_MUTATION_STORE.list_recall_strategy_profiles(limit=50)
        staged_only = [item for item in staged_profiles if str(item.get("status")) == "staged"]
        recall_readiness = readiness_from_telemetry_records(SUBSTRATE_REVIEW_TELEMETRY_STORE.query(GraphReviewTelemetryQueryV1(limit=200)))
        payload["surfaces"]["live"] = live_surfaces
        payload["surfaces"]["shadow"] = (
            [{"surface": "recall_strategy_profile", "status": "shadow_active_or_available", "production_default": "recall_v1"}]
            if active_shadow is not None or staged_profiles
            else []
        )
        payload["surfaces"]["proposal_only"] = [{"surface": "cognitive_self_model", "status": "proposal_draft_only"}]
        payload["surfaces"]["blocked"] = [
            {"surface": "recall_weighting_patch", "reason": "live recall apply explicitly blocked"},
            {"surface": "identity_kernel_rewrite", "reason": "forbidden by current safety posture"},
        ]
        payload["recall"]["active_shadow_profile"] = active_shadow.model_dump(mode="json") if active_shadow else None
        payload["recall"]["staged_profiles"] = staged_only[:20]
        payload["recall"]["readiness"] = recall_readiness.model_dump(mode="json")
        payload["recall"]["recent_eval_runs"] = SUBSTRATE_MUTATION_STORE.list_recall_shadow_eval_runs(limit=20, profile_id=(active_shadow.profile_id if active_shadow else None))
        payload["recall"]["production_candidate_reviews"] = SUBSTRATE_MUTATION_STORE.list_recall_production_candidate_reviews(limit=20, profile_id=(active_shadow.profile_id if active_shadow else None))
        rollups = _recall_canary_rollups(limit=50)
        payload["recall"]["manual_canary"] = {
            **rollups,
            "recommended_canary_action": _recommended_canary_action(
                active_shadow=(active_shadow.model_dump(mode="json") if active_shadow else None),
                rollups=rollups,
                recall_readiness=recall_readiness.model_dump(mode="json"),
            ),
        }
    except Exception as exc:
        warnings.append(f"recall_posture_unavailable:{exc}")
        payload["recall"]["warnings"] = list(payload["recall"]["warnings"]) + [str(exc)]

    try:
        cognitive_proposals = api_substrate_mutation_runtime_cognitive_proposals(limit=50).get("data", {}).get("recent_cognitive_proposals", [])
        counts: dict[str, int] = {}
        counts_by_class: dict[str, int] = {}
        for item in cognitive_proposals:
            state = str(item.get("rollout_state") or "proposed")
            counts[state] = counts.get(state, 0) + 1
            klass = str(item.get("mutation_class") or "unknown")
            counts_by_class[klass] = counts_by_class.get(klass, 0) + 1
        drafts = SUBSTRATE_MUTATION_STORE.list_cognitive_proposal_drafts(limit=50)
        stance_notes = SUBSTRATE_MUTATION_STORE.list_cognitive_stance_notes(limit=50)
        payload["cognitive"]["counts_by_state"] = counts
        payload["cognitive"]["counts_by_class"] = counts_by_class
        payload["cognitive"]["recent_proposals"] = cognitive_proposals[:20]
        payload["cognitive"]["recent_drafts"] = drafts[:20]
        payload["cognitive"]["recent_reviews"] = SUBSTRATE_MUTATION_STORE.recent_cognitive_reviews(limit=20)
        payload["cognitive"]["active_stance_notes"] = [row for row in stance_notes if str(row.get("status") or "") == "active"][:20]
        payload["cognitive"]["review_posture"] = _cognitive_review_posture_summary(cognitive_proposals, drafts, stance_notes)
        payload["cognitive"]["safety"] = {
            "identity_kernel_rewrite_performed": False,
            "production_self_model_rewrite_performed": False,
            "policy_override_performed": False,
            "prompt_rewrite_performed": False,
            "live_apply_performed": False,
            "execute_once_performed": False,
        }
    except Exception as exc:
        warnings.append(f"cognitive_posture_unavailable:{exc}")
        payload["cognitive"]["warnings"] = list(payload["cognitive"]["warnings"]) + [str(exc)]

    try:
        pressures = SUBSTRATE_MUTATION_STORE.recent_recall_pressures(limit=30)
        recent_events = api_substrate_mutation_runtime_recall_pressure_events(limit=30).get("data", {}).get("recent_recall_pressure_events", [])
        top_keys: dict[str, int] = {}
        high_conf = []
        for ev in recent_events:
            key = str(ev.get("pressure_category") or "unknown")
            top_keys[key] = top_keys.get(key, 0) + 1
            conf = float(ev.get("confidence") or 0.0)
            if conf >= 0.8:
                high_conf.append({"pressure_event_id": ev.get("pressure_event_id"), "pressure_category": key, "confidence": conf})
        ranked = sorted(top_keys.items(), key=lambda x: x[1], reverse=True)[:10]
        payload["pressure"]["top_pressure_keys"] = [{"key": k, "count": c} for k, c in ranked]
        payload["pressure"]["recent_evidence"] = recent_events[:20]
        payload["pressure"]["high_confidence_unresolved"] = high_conf[:20]
        payload["routing"]["pressure_summary"] = {"recall_pressure_count": len(pressures), "recall_event_count": len(recent_events)}
    except Exception as exc:
        warnings.append(f"pressure_unavailable:{exc}")
        payload["pressure"]["warnings"] = list(payload["pressure"]["warnings"]) + [str(exc)]

    payload["recent_activity"]["applies"] = payload["routing"]["recent_applies"][:10]
    payload["recent_activity"]["rollbacks"] = payload["routing"]["recent_rollbacks"][:10]
    payload["recent_activity"]["blocked_applies"] = payload["routing"]["recent_blocked_applies"][:10]
    payload["recent_activity"]["staged"] = payload["recall"]["staged_profiles"][:10]
    payload["recent_activity"]["reviews"] = payload["recall"]["production_candidate_reviews"][:10]
    recall_ready = payload["recall"].get("readiness") or {}
    if str(recall_ready.get("recommendation") or "") not in {"ready_for_shadow_expansion", "ready_for_operator_promotion"}:
        payload["overall"]["safe_next_action"] = "expand_recall_shadow_corpus"
    payload["overall"]["warnings"] = list(warnings)
    payload["warnings"] = warnings
    logger.info(
        "event=autonomy_readiness_snapshot schema_version=%s generated_at=%s warning_count=%s live_surface_count=%s shadow_surface_count=%s proposal_only_count=%s blocked_surface_count=%s safe_next_action=%s",
        payload.get("schema_version"),
        payload.get("generated_at"),
        len(warnings),
        len(payload["surfaces"]["live"]),
        len(payload["surfaces"]["shadow"]),
        len(payload["surfaces"]["proposal_only"]),
        len(payload["surfaces"]["blocked"]),
        payload["overall"].get("safe_next_action"),
    )
    return payload


def _execute_substrate_review_cycle(*, allow_followup: bool, explicit_queue_item_id: str | None = None) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    SUBSTRATE_REVIEW_QUEUE_STORE.refresh_from_storage()
    before_due = len(SUBSTRATE_REVIEW_QUEUE_STORE.list_eligible(now=now, limit=200))
    before_count = len(SUBSTRATE_REVIEW_QUEUE_STORE.snapshot(limit=200).queue_items)
    request = GraphReviewRuntimeRequestV1(
        invocation_surface="operator_review",
        explicit_queue_item_id=explicit_queue_item_id,
        execute_frontier_followup_allowed=allow_followup,
        operator_override_strict_zone=False,
        max_items_to_consider=20,
    )
    result = SUBSTRATE_REVIEW_RUNTIME_EXECUTOR.execute_once(request=request, now=now)
    after_now = datetime.now(timezone.utc)
    after_due = len(SUBSTRATE_REVIEW_QUEUE_STORE.list_eligible(now=after_now, limit=200))
    after_count = len(SUBSTRATE_REVIEW_QUEUE_STORE.snapshot(limit=200).queue_items)
    return {
        "request": {
            "invocation_surface": "operator_review",
            "allow_frontier_followup": allow_followup,
            "explicit_queue_item_id": explicit_queue_item_id,
            "single_cycle": True,
        },
        "result": result.model_dump(mode="json"),
        "queue_before": {"count": before_count, "due_count": before_due},
        "queue_after": {"count": after_count, "due_count": after_due},
        "source": _substrate_source_posture(),
    }


def _bootstrap_substrate_review_frontier(*, limit: int = 12) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    SUBSTRATE_REVIEW_QUEUE_STORE.refresh_from_storage()
    execution = SUBSTRATE_REVIEW_BOOTSTRAPPER.bootstrap(now=now, query_limit=limit)
    return {
        "generated_at": now.isoformat(),
        "items_before": execution.items_before,
        "items_enqueued": execution.items_enqueued,
        "items_after": execution.items_after,
        "due_after": execution.due_after,
        "source_posture": _substrate_source_posture(),
        "audit_summary": {
            "scheduled_decision_count": execution.scheduled_decision_count,
            "semantic_source": execution.semantic_source,
            "semantic_degraded": execution.semantic_degraded,
        },
        "notes": execution.notes,
    }


def _queue_count(payload: Dict[str, Any]) -> int:
    data = payload.get("data")
    if isinstance(data, dict):
        items = data.get("queue_items")
        if isinstance(items, list):
            return len(items)
    return 0


def _classify_substrate_debug_diagnosis(
    *,
    source_posture: Dict[str, Any],
    bootstrap_payload: Dict[str, Any],
    baseline_queue_payload: Dict[str, Any],
    post_bootstrap_queue_payload: Dict[str, Any],
    execute_payload: Dict[str, Any],
    final_queue_payload: Dict[str, Any],
) -> Dict[str, Any]:
    categories: list[str] = []
    highlights: list[str] = []
    severity = "ok"

    control_plane = source_posture.get("control_plane") if isinstance(source_posture, dict) else {}
    semantic = source_posture.get("semantic") if isinstance(source_posture, dict) else {}

    control_degraded = bool(control_plane.get("degraded")) if isinstance(control_plane, dict) else False
    semantic_degraded = bool(semantic.get("degraded")) if isinstance(semantic, dict) else False

    if control_degraded:
        categories.append("control plane degraded")
        highlights.append("Control-plane storage/reporting is degraded.")
        severity = "degraded"

    if semantic_degraded:
        categories.append("semantic substrate degraded")
        highlights.append("Semantic substrate queries are degraded.")
        severity = "degraded"

    bootstrap_items_enqueued = int(bootstrap_payload.get("items_enqueued") or 0)
    bootstrap_notes = bootstrap_payload.get("notes")
    if isinstance(bootstrap_notes, list) and any("seed_skipped" in str(note) for note in bootstrap_notes):
        categories.append("likely weak seed heuristics for current graph shape")
        highlights.append("Bootstrap skipped one or more seed regions.")
        severity = "warning" if severity != "degraded" else severity

    if "items_enqueued" not in bootstrap_payload and "error" in bootstrap_payload:
        categories.append("bootstrap route/path unavailable")
        highlights.append("Bootstrap path could not be invoked.")
        severity = "degraded"
    elif bootstrap_items_enqueued <= 0:
        categories.append("bootstrap produced zero items")
        highlights.append("Bootstrap completed without queue growth.")
        severity = "warning" if severity != "degraded" else severity

    baseline_queue_count = _queue_count(baseline_queue_payload)
    post_bootstrap_queue_count = _queue_count(post_bootstrap_queue_payload)
    final_queue_count = _queue_count(final_queue_payload)

    if bootstrap_items_enqueued > 0 and post_bootstrap_queue_count <= baseline_queue_count:
        categories.append("bootstrap claimed to enqueue but queue remained empty")
        highlights.append("Bootstrap reported enqueues, but queue did not increase.")
        severity = "degraded"

    execute_result = execute_payload.get("result") if isinstance(execute_payload, dict) else {}
    execute_outcome = execute_result.get("outcome") if isinstance(execute_result, dict) else None
    execute_reason = ""
    execute_audit = execute_result.get("audit_summary") if isinstance(execute_result, dict) else {}
    if isinstance(execute_audit, dict):
        execute_reason = str(execute_audit.get("selection_reason") or "").strip().lower()
    execute_notes = execute_result.get("notes") if isinstance(execute_result, dict) else []
    execute_notes_lower = " ".join(str(note).lower() for note in execute_notes) if isinstance(execute_notes, list) else ""

    if post_bootstrap_queue_count > 0 and execute_outcome == "noop" and (
        "no eligible queue items" in execute_reason or "not due yet" in execute_reason
    ):
        categories.append("queue nonempty but execute-once still noop due to no eligible items")
        highlights.append("Queue has items, but none were eligible for this cycle.")
        severity = "warning" if severity == "ok" else severity

    if bootstrap_items_enqueued > 0 and execute_outcome == "executed":
        categories.append("bootstrap produced items and execute-once succeeded")
        highlights.append("Bootstrap seeded the queue and one cycle executed successfully.")

    if "seed_skipped" in execute_notes_lower and "likely weak seed heuristics for current graph shape" not in categories:
        categories.append("likely weak seed heuristics for current graph shape")
        highlights.append("Execution notes indicate weak substrate seed coverage.")
        severity = "warning" if severity == "ok" else severity

    summary = " | ".join(highlights) if highlights else "Debug pass completed."
    return {
        "severity": severity,
        "categories": categories,
        "summary": summary,
        "facts": {
            "baseline_queue_count": baseline_queue_count,
            "post_bootstrap_queue_count": post_bootstrap_queue_count,
            "final_queue_count": final_queue_count,
            "bootstrap_items_enqueued": bootstrap_items_enqueued,
            "execute_outcome": execute_outcome,
        },
    }


def _run_substrate_review_debug_pass(*, bootstrap_limit: int = 12) -> Dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()
    baseline_status = _substrate_runtime_status_payload(queue_limit=50, telemetry_limit=20)
    baseline_queue = _sql_review_queue_payload(limit=50)
    baseline_overview = _graphdb_overview_payload(limit=10)
    baseline_hotspots = _graphdb_hotspots_payload(limit=20)
    baseline_executions = _sql_review_executions_payload(limit=20)

    bootstrap = _bootstrap_substrate_review_frontier(limit=bootstrap_limit)
    post_bootstrap_queue = _sql_review_queue_payload(limit=50)
    post_bootstrap_status = _substrate_runtime_status_payload(queue_limit=50, telemetry_limit=20)

    execute_once = _execute_substrate_review_cycle(allow_followup=False)
    final_queue = _sql_review_queue_payload(limit=50)
    final_status = _substrate_runtime_status_payload(queue_limit=50, telemetry_limit=20)
    final_executions = _sql_review_executions_payload(limit=20)
    source_posture = _substrate_source_posture()
    diagnosis = _classify_substrate_debug_diagnosis(
        source_posture=source_posture,
        bootstrap_payload=bootstrap,
        baseline_queue_payload=baseline_queue,
        post_bootstrap_queue_payload=post_bootstrap_queue,
        execute_payload=execute_once,
        final_queue_payload=final_queue,
    )

    return {
        "generated_at": generated_at,
        "baseline": {
            "runtime_status": baseline_status,
            "review_queue": baseline_queue,
            "overview": baseline_overview,
            "hotspots": baseline_hotspots,
            "recent_executions": baseline_executions,
        },
        "bootstrap": bootstrap,
        "post_bootstrap": {
            "review_queue": post_bootstrap_queue,
            "runtime_status": post_bootstrap_status,
        },
        "execute_once": execute_once,
        "final": {
            "review_queue": final_queue,
            "runtime_status": final_status,
            "recent_executions": final_executions,
        },
        "diagnosis": diagnosis,
        "source_posture": source_posture,
    }


@router.get("/substrate")
async def substrate_page() -> HTMLResponse:
    from .main import TEMPLATES_DIR, build_hub_ui_asset_version

    template = (TEMPLATES_DIR / "substrate.html").read_text(encoding="utf-8")
    rendered = template.replace("{{HUB_UI_ASSET_VERSION}}", build_hub_ui_asset_version())
    return HTMLResponse(
        content=rendered,
        status_code=200,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@router.get("/api/substrate/overview")
def api_substrate_overview(limit: int = Query(default=10, ge=1, le=100)) -> Dict[str, Any]:
    return _graphdb_overview_payload(limit=limit)


@router.get("/api/substrate/hotspots")
def api_substrate_hotspots(limit: int = Query(default=20, ge=1, le=200)) -> Dict[str, Any]:
    return _graphdb_hotspots_payload(limit=limit)


@router.get("/api/substrate/review-queue")
def api_substrate_review_queue(limit: int = Query(default=50, ge=1, le=200)) -> Dict[str, Any]:
    return _sql_review_queue_payload(limit=limit)


@router.get("/api/substrate/review-executions")
def api_substrate_review_executions(limit: int = Query(default=50, ge=1, le=200)) -> Dict[str, Any]:
    return _sql_review_executions_payload(limit=limit)


@router.get("/api/substrate/telemetry-summary")
def api_substrate_telemetry_summary(limit: int = Query(default=200, ge=1, le=500)) -> Dict[str, Any]:
    return _sql_telemetry_summary_payload(limit=limit)


@router.get("/api/substrate/calibration")
def api_substrate_calibration(limit: int = Query(default=20, ge=1, le=100)) -> Dict[str, Any]:
    return _sql_calibration_payload(limit=limit)


@router.get("/api/substrate/policy-comparison")
def api_substrate_policy_comparison(
    pair_mode: str = Query(default="baseline_vs_active"),
    baseline_profile_id: str | None = Query(default=None),
    candidate_profile_id: str | None = Query(default=None),
    window_seconds: int = Query(default=86400, ge=60, le=60 * 60 * 24 * 30),
    sample_limit: int = Query(default=500, ge=1, le=500),
) -> Dict[str, Any]:
    return _sql_policy_comparison_payload(
        pair_mode=pair_mode,
        baseline_profile_id=baseline_profile_id,
        candidate_profile_id=candidate_profile_id,
        window_seconds=window_seconds,
        sample_limit=sample_limit,
    )


@router.post("/api/substrate/policy/adopt")
def api_substrate_policy_adopt(request: SubstratePolicyAdoptHubRequest) -> Dict[str, Any]:
    """Stage or activate a substrate review policy profile (operator-controlled)."""
    adoption = SubstratePolicyAdoptionRequestV1(
        rollout_scope=SubstratePolicyRolloutScopeV1(
            invocation_surfaces=list(request.invocation_surfaces),
            target_zones=list(request.target_zones),
            operator_only=request.operator_only,
        ),
        policy_overrides=request.policy_overrides,
        activate_now=request.activate_now,
        operator_id=request.operator_id,
        rationale=request.rationale,
    )
    result = SUBSTRATE_POLICY_STORE.adopt(adoption)
    return {
        "source": _source_meta(
            kind=SUBSTRATE_POLICY_STORE.source_kind(),
            degraded=SUBSTRATE_POLICY_STORE.degraded(),
            limit=1,
            error=SUBSTRATE_POLICY_STORE.last_error(),
        ),
        "data": result.model_dump(mode="json"),
    }


@router.get("/api/substrate/review-runtime/status")
def api_substrate_review_runtime_status(
    queue_limit: int = Query(default=20, ge=1, le=200),
    telemetry_limit: int = Query(default=20, ge=1, le=500),
) -> Dict[str, Any]:
    return _substrate_runtime_status_payload(queue_limit=queue_limit, telemetry_limit=telemetry_limit)


@router.post("/api/substrate/review-runtime/execute-once")
def api_substrate_review_runtime_execute_once(request: SubstrateReviewExecuteRequest | None = None) -> Dict[str, Any]:
    req = request or SubstrateReviewExecuteRequest()
    return _execute_substrate_review_cycle(
        allow_followup=False,
        explicit_queue_item_id=req.explicit_queue_item_id,
    )


@router.post("/api/substrate/review-runtime/execute-once-followup")
def api_substrate_review_runtime_execute_once_followup(request: SubstrateReviewExecuteRequest | None = None) -> Dict[str, Any]:
    req = request or SubstrateReviewExecuteRequest()
    return _execute_substrate_review_cycle(
        allow_followup=True,
        explicit_queue_item_id=req.explicit_queue_item_id,
    )


@router.post("/api/substrate/review-runtime/bootstrap")
def api_substrate_review_runtime_bootstrap(request: SubstrateReviewBootstrapRequest | None = None) -> Dict[str, Any]:
    req = request or SubstrateReviewBootstrapRequest()
    return _bootstrap_substrate_review_frontier(limit=req.limit)


@router.post("/api/substrate/review-runtime/debug-run")
def api_substrate_review_runtime_debug_run(request: SubstrateReviewBootstrapRequest | None = None) -> Dict[str, Any]:
    req = request or SubstrateReviewBootstrapRequest()
    return _run_substrate_review_debug_pass(bootstrap_limit=req.limit)


@router.post("/api/substrate/review-runtime/smoke-check")
def api_substrate_review_runtime_smoke_check(request: SubstrateReviewSmokeCheckRequest | None = None) -> Dict[str, Any]:
    req = request or SubstrateReviewSmokeCheckRequest()
    now = datetime.now(timezone.utc)
    queue_snapshot = SUBSTRATE_REVIEW_QUEUE_STORE.snapshot(limit=req.limit)
    due_items = SUBSTRATE_REVIEW_QUEUE_STORE.list_eligible(now=now, limit=req.limit)
    semantic_probe = SUBSTRATE_SEMANTIC_STORE.query_hotspot_region(limit_nodes=min(5, req.limit), limit_edges=max(10, req.limit * 2))
    source = _substrate_source_posture()
    return {
        "generated_at": now.isoformat(),
        "source": source,
        "checks": {
            "queue_available": len(queue_snapshot.queue_items) >= 0,
            "runtime_eligible": len(due_items) > 0,
            "semantic_available": not bool(semantic_probe.degraded),
            "control_plane_available": not bool(source["control_plane"]["degraded"]),
        },
        "summary": {
            "queue_count": len(queue_snapshot.queue_items),
            "due_count": len(due_items),
            "semantic_query_kind": semantic_probe.query_kind,
            "semantic_truncated": bool(semantic_probe.truncated),
            "semantic_error": semantic_probe.error,
        },
    }


@router.post("/api/substrate/mutation-runtime/execute-once")
def api_substrate_mutation_runtime_execute_once(
    request: SubstrateMutationExecuteRequest | None = None,
    x_orion_operator_token: str | None = Header(default=None),
) -> Dict[str, Any]:
    _require_mutation_operator_guard(x_orion_operator_token)
    req = request or SubstrateMutationExecuteRequest()
    return _execute_substrate_mutation_cycle(request=req)


@router.get("/api/substrate/mutation-runtime/lineage")
def api_substrate_mutation_runtime_lineage(
    proposal_id: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=200),
) -> Dict[str, Any]:
    return _mutation_lineage_payload(proposal_id=proposal_id, limit=limit)


@router.get("/api/substrate/mutation-runtime/active-surfaces")
def api_substrate_mutation_runtime_active_surfaces() -> Dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": _source_meta(
            kind=SUBSTRATE_MUTATION_STORE.source_kind(),
            degraded=SUBSTRATE_MUTATION_STORE.degraded(),
            limit=200,
            error=SUBSTRATE_MUTATION_STORE.last_error(),
        ),
        "data": {
            "active_surfaces": SUBSTRATE_MUTATION_STORE.active_surfaces_snapshot(),
        },
    }


@router.get("/api/substrate/mutation-runtime/blocked-applies")
def api_substrate_mutation_runtime_blocked_applies(limit: int = Query(default=20, ge=1, le=200)) -> Dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": _source_meta(
            kind=SUBSTRATE_MUTATION_STORE.source_kind(),
            degraded=SUBSTRATE_MUTATION_STORE.degraded(),
            limit=limit,
            error=SUBSTRATE_MUTATION_STORE.last_error(),
        ),
        "data": {
            "recent_blocked_applies": SUBSTRATE_MUTATION_STORE.recent_blocked_applies(limit=limit),
        },
    }


@router.get("/api/substrate/mutation-runtime/rollbacks")
def api_substrate_mutation_runtime_rollbacks(limit: int = Query(default=20, ge=1, le=200)) -> Dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": _source_meta(
            kind=SUBSTRATE_MUTATION_STORE.source_kind(),
            degraded=SUBSTRATE_MUTATION_STORE.degraded(),
            limit=limit,
            error=SUBSTRATE_MUTATION_STORE.last_error(),
        ),
        "data": {
            "recent_rollbacks": SUBSTRATE_MUTATION_STORE.recent_rollbacks(limit=limit),
        },
    }


@router.get("/api/substrate/mutation-runtime/routing-pressure-sources")
def api_substrate_mutation_runtime_routing_pressure_sources(
    limit: int = Query(default=50, ge=1, le=200),
) -> Dict[str, Any]:
    rows = SUBSTRATE_MUTATION_STORE.recent_signals(limit=limit, target_surface="routing")
    sources = [
        {
            "signal_id": row.get("signal_id"),
            "detected_at": row.get("detected_at"),
            "source_ref": row.get("source_ref"),
            "source_kind": (row.get("metadata") or {}).get("source_kind"),
            "derived_signal_kind": row.get("event_kind"),
            "confidence": row.get("strength"),
            "evidence_refs": list(row.get("evidence_refs") or []),
        }
        for row in rows
    ]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": _source_meta(
            kind=SUBSTRATE_MUTATION_STORE.source_kind(),
            degraded=SUBSTRATE_MUTATION_STORE.degraded(),
            limit=limit,
            error=SUBSTRATE_MUTATION_STORE.last_error(),
        ),
        "data": {
            "routing_pressure_sources": sources,
        },
    }


@router.get("/api/substrate/mutation-runtime/producer-pressure-events")
def api_substrate_mutation_runtime_producer_pressure_events(
    limit: int = Query(default=50, ge=1, le=200),
) -> Dict[str, Any]:
    telemetry_rows = SUBSTRATE_REVIEW_TELEMETRY_STORE.query(GraphReviewTelemetryQueryV1(limit=max(limit, 200)))
    routing_signals = SUBSTRATE_MUTATION_STORE.recent_signals(limit=500, target_surface="routing")
    signal_links: dict[str, list[str]] = {}
    for signal in routing_signals:
        signal_id = str(signal.get("signal_id") or "")
        for ref in list(signal.get("evidence_refs") or []):
            if str(ref).startswith("pressure_event:"):
                event_id = str(ref).split("pressure_event:", 1)[1].strip()
                if event_id:
                    signal_links.setdefault(event_id, []).append(signal_id)
    events: list[dict[str, Any]] = []
    for row in telemetry_rows:
        if not row.pressure_events:
            continue
        for event in row.pressure_events:
            events.append(
                {
                    "pressure_event_id": event.pressure_event_id,
                    "source_service": event.source_service,
                    "source_event_id": event.source_event_id,
                    "correlation_id": event.correlation_id,
                    "pressure_category": event.pressure_category,
                    "confidence": event.confidence,
                    "evidence_refs": list(event.evidence_refs),
                    "observed_at": event.observed_at.isoformat(),
                    "linked_signal_ids": sorted(signal_links.get(event.pressure_event_id, [])),
                }
            )
    events.sort(key=lambda item: str(item.get("observed_at") or ""), reverse=True)
    events = events[:limit]
    grouped: dict[str, int] = {}
    for event in events:
        key = f"{event['source_service']}::{event['pressure_category']}"
        grouped[key] = grouped.get(key, 0) + 1
    grouped_rows = [
        {"source_service": key.split("::", 1)[0], "pressure_category": key.split("::", 1)[1], "count": count}
        for key, count in sorted(grouped.items())
    ]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "telemetry": _source_meta(
                kind=SUBSTRATE_REVIEW_TELEMETRY_STORE.source_kind(),
                degraded=SUBSTRATE_REVIEW_TELEMETRY_STORE.degraded(),
                limit=limit,
                error=SUBSTRATE_REVIEW_TELEMETRY_STORE.last_error(),
            ),
            "mutation_signals": _source_meta(
                kind=SUBSTRATE_MUTATION_STORE.source_kind(),
                degraded=SUBSTRATE_MUTATION_STORE.degraded(),
                limit=500,
                error=SUBSTRATE_MUTATION_STORE.last_error(),
            ),
        },
        "data": {
            "recent_events": events,
            "grouped_by_source_and_category": grouped_rows,
        },
    }


_COGNITIVE_SURFACES = {
    "cognitive_contradiction_reconciliation",
    "cognitive_identity_continuity_adjustment",
    "cognitive_stance_continuity_adjustment",
    "cognitive_social_continuity_repair",
}

_RECALL_SURFACES = {
    "recall_strategy_profile",
    "recall_anchor_policy",
    "recall_page_index_profile",
    "recall_graph_expansion_policy",
}

_RECALL_PROPOSAL_CLASSES = {
    "recall_strategy_profile_candidate",
    "recall_anchor_policy_candidate",
    "recall_page_index_profile_candidate",
    "recall_graph_expansion_policy_candidate",
}

_RECALL_READY_RECOMMENDATIONS = {"review_candidate", "ready_for_shadow_expansion", "ready_for_operator_promotion"}


def _strategy_kind_for_proposal(proposal: dict[str, Any]) -> str:
    mutation_class = str(proposal.get("mutation_class") or "")
    mapping = {
        "recall_strategy_profile_candidate": "strategy_profile",
        "recall_anchor_policy_candidate": "anchor_policy",
        "recall_page_index_profile_candidate": "page_index_policy",
        "recall_graph_expansion_policy_candidate": "graph_expansion_policy",
    }
    return mapping.get(mutation_class, "strategy_profile")


def _recall_profile_from_proposal(
    *,
    proposal: dict[str, Any],
    pressure: dict[str, Any] | None,
    created_by: str,
) -> RecallStrategyProfileV1:
    patch = dict((proposal.get("patch") if isinstance(proposal.get("patch"), dict) else {}).get("patch") or {})
    readiness = dict(patch.get("recall_strategy_readiness") or {})
    source_pressure_ids: list[str] = []
    if pressure and pressure.get("pressure_id"):
        source_pressure_ids.append(str(pressure["pressure_id"]))
    source_evidence_refs = [str(ref) for ref in list(proposal.get("evidence_refs") or [])][:128]
    anchor_snapshot = dict(patch.get("anchor_plan_summary") or {})
    page_snapshot = dict(patch.get("page_index_policy_snapshot") or {})
    graph_snapshot = dict(patch.get("graph_expansion_policy_snapshot") or {})
    if not page_snapshot:
        page_snapshot = {"selected_evidence_cards_mode": "from_proposal", "selected_count": len(list(patch.get("selected_evidence_cards") or []))}
    if not graph_snapshot:
        graph_snapshot = {"failure_category": patch.get("failure_category")}
    recall_config = {
        "profile": "recall.v2.shadow",
        "strategy_kind": _strategy_kind_for_proposal(proposal),
        "shadow_only_status": patch.get("shadow_only_status"),
        "not_applied_status": patch.get("not_applied_status"),
    }
    return RecallStrategyProfileV1(
        source_proposal_id=str(proposal.get("proposal_id") or ""),
        source_pressure_ids=source_pressure_ids,
        source_evidence_refs=source_evidence_refs,
        readiness_snapshot=readiness,
        strategy_kind=_strategy_kind_for_proposal(proposal),
        recall_v2_config_snapshot=recall_config,
        anchor_policy_snapshot=anchor_snapshot,
        page_index_policy_snapshot=page_snapshot,
        graph_expansion_policy_snapshot=graph_snapshot,
        eval_evidence_refs=[],
        created_by=created_by,
        status="staged",
    )


def _proposal_readiness_gate_payload(proposal: dict[str, Any]) -> tuple[dict[str, Any], str]:
    patch = dict((proposal.get("patch") if isinstance(proposal.get("patch"), dict) else {}).get("patch") or {})
    readiness = dict(patch.get("recall_strategy_readiness") or {})
    recommendation = str(readiness.get("recommendation") or "not_ready")
    return readiness, recommendation


def _readiness_delta_summary(*, before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    metrics = ("corpus_coverage", "precision_proxy", "irrelevant_cousin_rate", "explainability_completeness", "latency_delta_ms_mean")
    delta: dict[str, Any] = {}
    for key in metrics:
        b = before.get(key)
        a = after.get(key)
        if isinstance(b, (int, float)) and isinstance(a, (int, float)):
            delta[f"{key}_delta"] = round(float(a) - float(b), 4)
    delta["recommendation_before"] = before.get("recommendation")
    delta["recommendation_after"] = after.get("recommendation")
    return delta


@router.get("/api/substrate/mutation-runtime/recall-pressure-store")
def api_substrate_mutation_runtime_recall_pressure_store(
    limit: int = Query(default=20, ge=1, le=100),
) -> Dict[str, Any]:
    """Recent recall-surface mutation pressures including bounded recall_evidence_history."""
    rows = SUBSTRATE_MUTATION_STORE.recent_recall_pressures(limit=limit)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": _source_meta(
            kind=SUBSTRATE_MUTATION_STORE.source_kind(),
            degraded=SUBSTRATE_MUTATION_STORE.degraded(),
            limit=limit,
            error=SUBSTRATE_MUTATION_STORE.last_error(),
        ),
        "data": {"recent_recall_pressures": rows},
    }


@router.post("/api/substrate/mutation-runtime/recall-eval-suite/record")
def api_substrate_mutation_runtime_recall_eval_suite_record(
    body: RecallEvalSuiteRecordRequest,
    x_orion_operator_token: str | None = Header(default=None),
) -> Dict[str, Any]:
    """
    Record recall_eval-shaped rows as first-class mutation pressure telemetry.
    Requires operator token and HUB_RECALL_EVAL_RECORDING_ENABLED=true (disabled by default).
    """
    if not bool(getattr(settings, "HUB_RECALL_EVAL_RECORDING_ENABLED", False)):
        raise HTTPException(status_code=403, detail="recall_eval_recording_disabled")
    _require_mutation_operator_guard(x_orion_operator_token)
    from orion.substrate.recall_eval_bridge import pressure_evidence_from_eval_suite_rows

    run_id = str(body.suite_run_id or "").strip() or f"recall-eval-{uuid4()}"
    events = pressure_evidence_from_eval_suite_rows(body.rows, suite_run_id=run_id)
    if not events:
        raise HTTPException(status_code=422, detail="recall_eval_rows_empty_or_invalid")
    recorded = 0
    for offset in range(0, len(events), 16):
        chunk = events[offset : offset + 16]
        _record_pressure_events_as_telemetry(
            events=chunk,
            correlation_id=run_id,
            source_event_id=f"recall_eval_suite:{run_id}:chunk{offset // 16}",
            invocation_surface="operator_review",
            ingest_notes=["recall_eval_suite_manual_ingest", "operator_triggered"],
        )
        recorded += len(chunk)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data": {"recorded": recorded, "suite_run_id": run_id},
    }


@router.get("/api/substrate/mutation-runtime/recall-pressure-events")
def api_substrate_mutation_runtime_recall_pressure_events(
    limit: int = Query(default=50, ge=1, le=200),
) -> Dict[str, Any]:
    telemetry_rows = SUBSTRATE_REVIEW_TELEMETRY_STORE.query(GraphReviewTelemetryQueryV1(limit=max(limit, 200)))
    recall_signals = SUBSTRATE_MUTATION_STORE.recent_signals(limit=500)
    signal_links: dict[str, list[str]] = {}
    for signal in recall_signals:
        if str(signal.get("target_surface") or "") not in _RECALL_SURFACES:
            continue
        signal_id = str(signal.get("signal_id") or "")
        for ref in list(signal.get("evidence_refs") or []):
            if str(ref).startswith("pressure_event:"):
                event_id = str(ref).split("pressure_event:", 1)[1].strip()
                if event_id:
                    signal_links.setdefault(event_id, []).append(signal_id)
    events: list[dict[str, Any]] = []
    recall_categories = {
        "recall_miss_or_dissatisfaction",
        "unsupported_memory_claim",
        "irrelevant_semantic_neighbor",
        "missing_exact_anchor",
        "stale_memory_selected",
    }
    for row in telemetry_rows:
        if not row.pressure_events:
            continue
        for event in row.pressure_events:
            if str(event.pressure_category) not in recall_categories:
                continue
            meta = dict(event.metadata or {})
            events.append(
                {
                    "pressure_event_id": event.pressure_event_id,
                    "source_service": event.source_service,
                    "source_event_id": event.source_event_id,
                    "correlation_id": event.correlation_id,
                    "pressure_category": event.pressure_category,
                    "confidence": event.confidence,
                    "evidence_refs": list(event.evidence_refs),
                    "metadata": meta,
                    "observed_at": event.observed_at.isoformat(),
                    "linked_signal_ids": sorted(signal_links.get(event.pressure_event_id, [])),
                }
            )
    events.sort(key=lambda item: str(item.get("observed_at") or ""), reverse=True)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": _source_meta(
            kind=SUBSTRATE_REVIEW_TELEMETRY_STORE.source_kind(),
            degraded=SUBSTRATE_REVIEW_TELEMETRY_STORE.degraded(),
            limit=limit,
            error=SUBSTRATE_REVIEW_TELEMETRY_STORE.last_error(),
        ),
        "data": {"recent_recall_pressure_events": events[:limit]},
    }


@router.get("/api/substrate/mutation-runtime/recall-strategy-proposals")
def api_substrate_mutation_runtime_recall_strategy_proposals(
    limit: int = Query(default=20, ge=1, le=200),
) -> Dict[str, Any]:
    proposals = sorted(SUBSTRATE_MUTATION_STORE._proposals.values(), key=lambda item: item.created_at, reverse=True)
    filtered: list[dict[str, Any]] = []
    for proposal in proposals:
        payload = proposal.model_dump(mode="json")
        mutation_class = str(getattr(proposal, "mutation_class", payload.get("mutation_class")) or "")
        target_surface = str(getattr(proposal, "target_surface", payload.get("target_surface")) or "")
        if mutation_class in _RECALL_PROPOSAL_CLASSES or target_surface in _RECALL_SURFACES:
            filtered.append(payload)
        if len(filtered) >= limit:
            break
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": _source_meta(
            kind=SUBSTRATE_MUTATION_STORE.source_kind(),
            degraded=SUBSTRATE_MUTATION_STORE.degraded(),
            limit=limit,
            error=SUBSTRATE_MUTATION_STORE.last_error(),
        ),
        "data": {"recent_recall_strategy_proposals": filtered},
    }


@router.get("/api/substrate/mutation-runtime/recall-strategy-proposals/{proposal_id}/lineage")
def api_substrate_mutation_runtime_recall_strategy_proposal_lineage(proposal_id: str) -> Dict[str, Any]:
    lifecycle = SUBSTRATE_MUTATION_STORE.lifecycle_for_proposal(proposal_id)
    if lifecycle is None:
        raise HTTPException(status_code=404, detail="recall_strategy_proposal_not_found")
    proposal = lifecycle.get("proposal") if isinstance(lifecycle, dict) else None
    if not isinstance(proposal, dict):
        raise HTTPException(status_code=404, detail="recall_strategy_proposal_not_found")
    if (
        str(proposal.get("mutation_class") or "") not in _RECALL_PROPOSAL_CLASSES
        and str(proposal.get("target_surface") or "") not in _RECALL_SURFACES
    ):
        raise HTTPException(status_code=404, detail="recall_strategy_proposal_not_found")
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": _source_meta(
            kind=SUBSTRATE_MUTATION_STORE.source_kind(),
            degraded=SUBSTRATE_MUTATION_STORE.degraded(),
            limit=1,
            error=SUBSTRATE_MUTATION_STORE.last_error(),
        ),
        "data": {"lineage": lifecycle},
    }


@router.post("/api/substrate/mutation-runtime/recall-strategy-proposals/{proposal_id}/promote-to-staged-profile")
def api_substrate_mutation_runtime_recall_strategy_proposal_promote_to_staged_profile(
    proposal_id: str,
    request: RecallProposalStageRequest,
    x_orion_operator_token: str | None = Header(default=None),
) -> Dict[str, Any]:
    _require_mutation_operator_guard(x_orion_operator_token)
    lifecycle = SUBSTRATE_MUTATION_STORE.lifecycle_for_proposal(proposal_id)
    if lifecycle is None:
        raise HTTPException(status_code=404, detail="recall_strategy_proposal_not_found")
    proposal = lifecycle.get("proposal") if isinstance(lifecycle, dict) else None
    if not isinstance(proposal, dict) or str(proposal.get("mutation_class") or "") not in _RECALL_PROPOSAL_CLASSES:
        raise HTTPException(status_code=404, detail="recall_strategy_proposal_not_found")
    queue_state = str(((lifecycle.get("queue_item") if isinstance(lifecycle.get("queue_item"), dict) else {}) or {}).get("status") or "")
    if queue_state != "pending_review":
        raise HTTPException(status_code=409, detail="recall_strategy_proposal_not_operator_gated")
    readiness, recommendation = _proposal_readiness_gate_payload(proposal)
    if recommendation not in _RECALL_READY_RECOMMENDATIONS:
        if not request.override:
            raise HTTPException(status_code=409, detail="recall_strategy_readiness_not_sufficient")
    gates = list(readiness.get("gates_blocked") or [])
    if gates and not request.override:
        raise HTTPException(status_code=409, detail="recall_strategy_readiness_gates_blocked")
    if request.override and not str(request.operator_rationale or "").strip():
        raise HTTPException(status_code=422, detail="override_operator_rationale_required")
    pressure = lifecycle.get("pressure") if isinstance(lifecycle.get("pressure"), dict) else None
    profile = _recall_profile_from_proposal(
        proposal=proposal,
        pressure=pressure if isinstance(pressure, dict) else None,
        created_by=str(request.created_by or "operator"),
    )
    staged = SUBSTRATE_MUTATION_STORE.stage_recall_profile(profile=profile)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": _source_meta(
            kind=SUBSTRATE_MUTATION_STORE.source_kind(),
            degraded=SUBSTRATE_MUTATION_STORE.degraded(),
            limit=1,
            error=SUBSTRATE_MUTATION_STORE.last_error(),
        ),
        "data": {
            "profile": staged.model_dump(mode="json"),
            "production_recall_mode": "v1",
            "override_applied": bool(request.override),
            "operator_rationale": str(request.operator_rationale or ""),
        },
    }


@router.get("/api/substrate/mutation-runtime/recall-strategy-profiles")
def api_substrate_mutation_runtime_recall_strategy_profiles(
    limit: int = Query(default=20, ge=1, le=200),
) -> Dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": _source_meta(
            kind=SUBSTRATE_MUTATION_STORE.source_kind(),
            degraded=SUBSTRATE_MUTATION_STORE.degraded(),
            limit=limit,
            error=SUBSTRATE_MUTATION_STORE.last_error(),
        ),
        "data": {"profiles": SUBSTRATE_MUTATION_STORE.list_recall_strategy_profiles(limit=limit)},
    }


@router.get("/api/substrate/mutation-runtime/recall-strategy-profiles/{profile_id}")
def api_substrate_mutation_runtime_recall_strategy_profile_detail(profile_id: str) -> Dict[str, Any]:
    profile = SUBSTRATE_MUTATION_STORE.get_recall_strategy_profile(profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="recall_strategy_profile_not_found")
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": _source_meta(
            kind=SUBSTRATE_MUTATION_STORE.source_kind(),
            degraded=SUBSTRATE_MUTATION_STORE.degraded(),
            limit=1,
            error=SUBSTRATE_MUTATION_STORE.last_error(),
        ),
        "data": {"profile": profile.model_dump(mode="json")},
    }


@router.get("/api/substrate/mutation-runtime/recall-strategy-profiles/{profile_id}/lineage")
def api_substrate_mutation_runtime_recall_strategy_profile_lineage(profile_id: str) -> Dict[str, Any]:
    payload = SUBSTRATE_MUTATION_STORE.recall_strategy_profile_lineage(profile_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="recall_strategy_profile_not_found")
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": _source_meta(
            kind=SUBSTRATE_MUTATION_STORE.source_kind(),
            degraded=SUBSTRATE_MUTATION_STORE.degraded(),
            limit=1,
            error=SUBSTRATE_MUTATION_STORE.last_error(),
        ),
        "data": payload,
    }


@router.post("/api/substrate/mutation-runtime/recall-strategy-profiles/{profile_id}/create-production-candidate-review")
def api_substrate_mutation_runtime_recall_strategy_profile_create_production_candidate_review(
    profile_id: str,
    request: RecallProductionCandidateReviewCreateRequest,
    x_orion_operator_token: str | None = Header(default=None),
) -> Dict[str, Any]:
    _require_mutation_operator_guard(x_orion_operator_token)
    profile = SUBSTRATE_MUTATION_STORE.get_recall_strategy_profile(profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="recall_strategy_profile_not_found")
    runs = SUBSTRATE_MUTATION_STORE.list_recall_shadow_eval_runs(limit=200, profile_id=profile_id)
    non_dry = [row for row in runs if not bool(row.get("dry_run")) and str(row.get("status") or "") == "completed"]
    if profile.status not in {"shadow_active", "staged"}:
        raise HTTPException(status_code=409, detail="recall_strategy_profile_not_reviewable")
    if not non_dry and not request.override:
        raise HTTPException(status_code=409, detail="recall_candidate_requires_non_dry_eval_history")
    readiness = dict(profile.readiness_snapshot or {})
    recommendation = str(readiness.get("recommendation") or "")
    if recommendation not in {"ready_for_shadow_expansion", "ready_for_operator_promotion"} and not request.override:
        raise HTTPException(status_code=409, detail="recall_candidate_readiness_not_sufficient")
    if request.override and not str(request.operator_rationale or "").strip():
        raise HTTPException(status_code=422, detail="override_operator_rationale_required")
    source_run_ids = [str(row.get("run_id")) for row in non_dry[:64] if row.get("run_id")]
    latest_non_dry = non_dry[0] if non_dry else None
    delta = dict(latest_non_dry.get("readiness_delta_summary") or {}) if isinstance(latest_non_dry, dict) else {}
    improvements: list[str] = []
    regressions: list[str] = []
    for k, v in delta.items():
        if not str(k).endswith("_delta") or not isinstance(v, (int, float)):
            continue
        if float(v) > 0:
            improvements.append(f"{k}={v}")
        elif float(v) < 0:
            regressions.append(f"{k}={v}")
    review = RecallProductionCandidateReviewV1(
        profile_id=profile_id,
        source_eval_run_ids=source_run_ids,
        readiness_snapshot=readiness,
        risk_summary=list(readiness.get("gates_blocked") or [])[:64],
        observed_improvements=improvements[:64],
        observed_regressions=regressions[:64],
        operator_checklist=dict(request.operator_checklist or {}),
        recommendation=(
            "ready_for_manual_canary"
            if recommendation == "ready_for_operator_promotion"
            else ("expand_shadow_corpus" if recommendation == "ready_for_shadow_expansion" else "keep_shadowing")
        ),
        status="draft",
        created_by=str(request.created_by or "operator"),
    )
    saved = SUBSTRATE_MUTATION_STORE.record_recall_production_candidate_review(review)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": _source_meta(
            kind=SUBSTRATE_MUTATION_STORE.source_kind(),
            degraded=SUBSTRATE_MUTATION_STORE.degraded(),
            limit=1,
            error=SUBSTRATE_MUTATION_STORE.last_error(),
        ),
        "data": {
            "review": saved.model_dump(mode="json"),
            "production_recall_mode": "v1",
            "recall_live_apply_enabled": False,
            "override_applied": bool(request.override),
            "operator_rationale": str(request.operator_rationale or ""),
        },
    }


@router.get("/api/substrate/mutation-runtime/recall-production-candidate-reviews")
def api_substrate_mutation_runtime_recall_production_candidate_reviews(
    limit: int = Query(default=20, ge=1, le=200),
    profile_id: str | None = None,
) -> Dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": _source_meta(
            kind=SUBSTRATE_MUTATION_STORE.source_kind(),
            degraded=SUBSTRATE_MUTATION_STORE.degraded(),
            limit=limit,
            error=SUBSTRATE_MUTATION_STORE.last_error(),
        ),
        "data": {"reviews": SUBSTRATE_MUTATION_STORE.list_recall_production_candidate_reviews(limit=limit, profile_id=profile_id)},
    }


@router.get("/api/substrate/mutation-runtime/recall-production-candidate-reviews/{review_id}")
def api_substrate_mutation_runtime_recall_production_candidate_review_detail(review_id: str) -> Dict[str, Any]:
    review = SUBSTRATE_MUTATION_STORE.get_recall_production_candidate_review(review_id)
    if review is None:
        raise HTTPException(status_code=404, detail="recall_production_candidate_review_not_found")
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": _source_meta(
            kind=SUBSTRATE_MUTATION_STORE.source_kind(),
            degraded=SUBSTRATE_MUTATION_STORE.degraded(),
            limit=1,
            error=SUBSTRATE_MUTATION_STORE.last_error(),
        ),
        "data": {"review": review.model_dump(mode="json")},
    }


@router.post("/api/substrate/mutation-runtime/recall-strategy-profiles/{profile_id}/activate-shadow")
def api_substrate_mutation_runtime_recall_strategy_profile_activate_shadow(
    profile_id: str,
    request: RecallShadowActivateRequest,
    x_orion_operator_token: str | None = Header(default=None),
) -> Dict[str, Any]:
    _require_mutation_operator_guard(x_orion_operator_token)
    profile = SUBSTRATE_MUTATION_STORE.get_recall_strategy_profile(profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="recall_strategy_profile_not_found")
    if profile.status != "staged":
        raise HTTPException(status_code=409, detail="recall_strategy_profile_not_staged")
    activated = SUBSTRATE_MUTATION_STORE.activate_recall_shadow_profile(profile_id)
    if activated is None:
        raise HTTPException(status_code=404, detail="recall_strategy_profile_not_found")
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": _source_meta(
            kind=SUBSTRATE_MUTATION_STORE.source_kind(),
            degraded=SUBSTRATE_MUTATION_STORE.degraded(),
            limit=1,
            error=SUBSTRATE_MUTATION_STORE.last_error(),
        ),
        "data": {
            "profile": activated.model_dump(mode="json"),
            "operator_rationale": str(request.operator_rationale or ""),
            "production_recall_mode": "v1",
        },
    }


@router.get("/api/substrate/mutation-runtime/recall-shadow-profile-posture")
def api_substrate_mutation_runtime_recall_shadow_profile_posture() -> Dict[str, Any]:
    active = SUBSTRATE_MUTATION_STORE.active_recall_shadow_profile()
    eval_summary = api_substrate_mutation_runtime_recall_v1_v2_latest_eval().get("data", {})
    readiness = active.readiness_snapshot if active is not None else {}
    recent_runs = SUBSTRATE_MUTATION_STORE.list_recall_shadow_eval_runs(
        limit=5,
        profile_id=(active.profile_id if active is not None else None),
    )
    recent_reviews = SUBSTRATE_MUTATION_STORE.list_recall_production_candidate_reviews(
        limit=5,
        profile_id=(active.profile_id if active is not None else None),
    )
    lineage = (
        SUBSTRATE_MUTATION_STORE.recall_strategy_profile_lineage(active.profile_id)
        if active is not None
        else None
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": _source_meta(
            kind=SUBSTRATE_MUTATION_STORE.source_kind(),
            degraded=SUBSTRATE_MUTATION_STORE.degraded(),
            limit=1,
            error=SUBSTRATE_MUTATION_STORE.last_error(),
        ),
        "data": {
            "active_shadow_profile_id": active.profile_id if active else None,
            "readiness_snapshot": readiness,
            "current_recall_v2_shadow_config": (active.recall_v2_config_snapshot if active is not None else {}),
            "last_eval_summary": eval_summary.get("latest_recall_v1_v2_eval_summary"),
            "recent_shadow_eval_runs": recent_runs,
            "recent_production_candidate_reviews": recent_reviews,
            "last_proposal_source_lineage": lineage,
            "production_recall_mode": "v1",
            "production_recall_still_v1": True,
        },
    }


@router.get("/api/substrate/recall-canary/status")
def api_substrate_recall_canary_status(
    limit: int = Query(default=20, ge=1, le=200),
) -> Dict[str, Any]:
    rollups = _recall_canary_rollups(limit=limit)
    available_profiles, default_canary_profile_id = _recall_canary_profile_catalog(limit=50)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": _source_meta(
            kind=SUBSTRATE_MUTATION_STORE.source_kind(),
            degraded=SUBSTRATE_MUTATION_STORE.degraded(),
            limit=limit,
            error=SUBSTRATE_MUTATION_STORE.last_error(),
        ),
        "data": {
            "schema_version": "recall_canary_status.v1",
            **rollups,
            "available_profiles": available_profiles,
            "default_canary_profile_id": default_canary_profile_id,
            "production_recall_mode": PRODUCTION_RECALL_MODE,
            "recall_live_apply_enabled": RECALL_LIVE_APPLY_ENABLED,
        },
    }


@router.post("/api/substrate/recall-canary/query")
def api_substrate_recall_canary_query(
    request: RecallCanaryQueryRequest,
    x_orion_operator_token: str | None = Header(default=None),
) -> Dict[str, Any]:
    _require_mutation_operator_guard(x_orion_operator_token)
    base = settings.recall_service_url
    warnings: list[str] = []
    available_profiles, default_canary_profile_id = _recall_canary_profile_catalog(limit=200)
    profile_by_id: dict[str, dict[str, Any]] = {
        str(row.get("profile_id")): dict(row) for row in available_profiles if row.get("profile_id")
    }
    selected_profile_id = str(request.profile_id or "").strip() or str(default_canary_profile_id or "").strip()
    if not selected_profile_id:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "recall_canary_profile_required",
                "message": "No recall canary profiles are available.",
                "allowed_profile_ids": sorted(profile_by_id.keys()),
            },
        )
    selected_profile = profile_by_id.get(selected_profile_id)
    if selected_profile is None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_recall_canary_profile_id",
                "message": f"Unknown profile_id: {selected_profile_id}",
                "allowed_profile_ids": sorted(profile_by_id.keys()),
            },
        )
    compare_payload: dict[str, Any] = {}
    try:
        resp = requests.post(
            f"{base}/debug/recall/compare",
            json={
                "query_text": request.query_text,
                "profile": selected_profile_id,
                "session_id": request.session_id,
                "node_id": request.node_id,
            },
            timeout=float(settings.HUB_RECALL_SHADOW_EVAL_TIMEOUT_SEC),
        )
        compare_payload = resp.json() if resp.status_code < 500 else {}
        if resp.status_code >= 400:
            warnings.append(f"recall_compare_status:{resp.status_code}")
    except Exception as exc:
        warnings.append(f"recall_compare_unavailable:{exc.__class__.__name__}")
    compare = dict(compare_payload.get("compare") or {})
    v1 = dict(compare_payload.get("v1") or {})
    v2 = dict(compare_payload.get("v2") or {})
    run = RecallCanaryRunV1(
        profile_id=selected_profile_id,
        query_text=request.query_text,
        query_profile=selected_profile_id,
        profile_metadata=selected_profile,
        comparison_summary=compare,
        v1_summary={
            "selected_count": (((v1.get("bundle") or {}).get("items") and len((v1.get("bundle") or {}).get("items"))) or compare.get("v1_selected_count")),
            "latency_ms": compare.get("v1_latency_ms"),
        },
        v2_summary={
            "selected_count": (((v2.get("bundle") or {}).get("items") and len((v2.get("bundle") or {}).get("items"))) or compare.get("v2_selected_count")),
            "latency_ms": compare.get("v2_latency_ms"),
        },
    )
    saved = SUBSTRATE_MUTATION_STORE.record_recall_canary_run(run)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": _source_meta(
            kind=SUBSTRATE_MUTATION_STORE.source_kind(),
            degraded=SUBSTRATE_MUTATION_STORE.degraded(),
            limit=1,
            error=SUBSTRATE_MUTATION_STORE.last_error(),
        ),
        "data": {
            "schema_version": "recall_canary_query_result.v1",
            "canary_run_id": saved.canary_run_id,
            "canary_run": saved.model_dump(mode="json"),
            "selected_profile": selected_profile,
            "comparison": compare,
            "v1": v1,
            "v2": v2,
            "warnings": warnings,
            "production_recall_mode": PRODUCTION_RECALL_MODE,
            "recall_live_apply_enabled": RECALL_LIVE_APPLY_ENABLED,
            "safety": {
                "production_default_unchanged": True,
                "promotion_performed": False,
                "apply_performed": False,
            },
        },
    }


@router.post("/api/substrate/recall-canary/runs/{canary_run_id}/judgment")
def api_substrate_recall_canary_run_judgment(
    canary_run_id: str,
    request: RecallCanaryJudgmentRequest,
    x_orion_operator_token: str | None = Header(default=None),
) -> Dict[str, Any]:
    _require_mutation_operator_guard(x_orion_operator_token)
    row = SUBSTRATE_MUTATION_STORE.get_recall_canary_run(canary_run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="recall_canary_run_not_found")
    pressure_refs: list[str] = []
    if request.should_emit_pressure:
        pressure = MutationPressureEvidenceV1(
            source_service="orion-hub",
            source_event_id=f"recall_canary_judgment:{canary_run_id}",
            correlation_id=f"recall_canary_judgment:{canary_run_id}",
            pressure_category=_failure_mode_to_pressure_category(request.failure_modes[0] if request.failure_modes else None),  # type: ignore[arg-type]
            confidence=0.75,
            evidence_refs=[f"recall_canary_run:{canary_run_id}", f"judgment:{request.judgment}"],
            metadata={
                "recall_canary_judgment": request.judgment,
                "failure_modes": list(request.failure_modes),
                "operator_note": str(request.operator_note or ""),
            },
        )
        _record_pressure_events_as_telemetry(
            events=[pressure],
            correlation_id=f"recall_canary_judgment:{canary_run_id}",
            source_event_id=f"recall_canary_judgment:{canary_run_id}",
            invocation_surface="operator_review",
            ingest_notes=["recall_canary_judgment", f"run:{canary_run_id}"],
        )
        pressure_refs.append(f"pressure_event:{pressure.pressure_event_id}")
    review_candidate_marked = False
    if request.should_mark_review_candidate and row.profile_id:
        profile = SUBSTRATE_MUTATION_STORE.get_recall_strategy_profile(row.profile_id)
        if profile is not None:
            snapshot = dict(profile.readiness_snapshot or {})
            snapshot["manual_canary_review_candidate"] = True
            snapshot["manual_canary_last_judgment"] = request.judgment
            SUBSTRATE_MUTATION_STORE.update_recall_strategy_profile(
                profile_id=profile.profile_id,
                readiness_snapshot=snapshot,
            )
            review_candidate_marked = True
    record = RecallCanaryJudgmentRecordV1(
        canary_run_id=canary_run_id,
        profile_id=row.profile_id,
        query_text=row.query_text,
        judgment=request.judgment,
        failure_modes=list(request.failure_modes),
        operator_note=str(request.operator_note or ""),
        should_emit_pressure=bool(request.should_emit_pressure),
        should_mark_review_candidate=bool(request.should_mark_review_candidate),
        pressure_event_refs=pressure_refs,
        review_candidate_marked=review_candidate_marked,
    )
    SUBSTRATE_MUTATION_STORE.record_recall_canary_judgment(record)
    return {
        "schema_version": "recall_canary_judgment_result.v1",
        "canary_run_id": canary_run_id,
        "judgment_recorded": True,
        "pressure_emitted": bool(pressure_refs),
        "review_candidate_marked": review_candidate_marked,
        "warnings": [],
    }


@router.post("/api/substrate/recall-canary/runs/{canary_run_id}/create-review-artifact")
def api_substrate_recall_canary_create_review_artifact(
    canary_run_id: str,
    request: RecallCanaryReviewArtifactRequest,
    x_orion_operator_token: str | None = Header(default=None),
) -> Dict[str, Any]:
    _require_mutation_operator_guard(x_orion_operator_token)
    run = SUBSTRATE_MUTATION_STORE.get_recall_canary_run(canary_run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="recall_canary_run_not_found")
    warnings: list[str] = []
    active = SUBSTRATE_MUTATION_STORE.active_recall_shadow_profile()
    if active is None:
        warnings.append("no_active_shadow_profile")
        return {
            "schema_version": "recall_canary_review_artifact_result.v1",
            "canary_run_id": canary_run_id,
            "review_artifact_created": False,
            "review_artifact_id": None,
            "safety": {
                "production_default_unchanged": True,
                "promotion_performed": False,
                "apply_performed": False,
            },
            "warnings": warnings,
        }
    latest_judgment = SUBSTRATE_MUTATION_STORE.latest_recall_canary_judgment_for_run(canary_run_id)
    run_profile_id = str(run.profile_id or "").strip() or active.profile_id
    review = RecallProductionCandidateReviewV1(
        profile_id=run_profile_id,
        source_eval_run_ids=[],
        readiness_snapshot=dict(active.readiness_snapshot or {}),
        risk_summary=(list((latest_judgment.failure_modes if latest_judgment else []))[:16] if latest_judgment else []),
        observed_improvements=([f"judgment:{latest_judgment.judgment}"] if latest_judgment else []),
        observed_regressions=[],
        operator_checklist={
            "review_type": request.review_type,
            "canary_run_id": canary_run_id,
            "include_comparison_summary": bool(request.include_comparison_summary),
            "include_operator_judgment": bool(request.include_operator_judgment),
            "operator_note": str(request.operator_note or ""),
        },
        recommendation="keep_shadowing",
        status="draft",
        created_by="operator",
    )
    saved_review = SUBSTRATE_MUTATION_STORE.record_recall_production_candidate_review(review)
    artifact = RecallCanaryReviewArtifactV1(
        canary_run_id=canary_run_id,
        profile_id=run_profile_id,
        linked_review_id=saved_review.review_id,
        review_type=request.review_type,
        include_comparison_summary=bool(request.include_comparison_summary),
        include_operator_judgment=bool(request.include_operator_judgment),
        operator_note=str(request.operator_note or ""),
        summary={
            "selected_profile": dict(run.profile_metadata or {}),
            "comparison": (run.comparison_summary if request.include_comparison_summary else {}),
            "judgment": (latest_judgment.model_dump(mode="json") if (request.include_operator_judgment and latest_judgment) else {}),
        },
    )
    saved = SUBSTRATE_MUTATION_STORE.record_recall_canary_review_artifact(artifact)
    return {
        "schema_version": "recall_canary_review_artifact_result.v1",
        "canary_run_id": canary_run_id,
        "review_artifact_created": True,
        "review_artifact_id": saved.review_artifact_id,
        "safety": {
            "production_default_unchanged": True,
            "promotion_performed": False,
            "apply_performed": False,
        },
        "warnings": warnings,
    }


@router.post("/api/substrate/mutation-runtime/recall-shadow-profile/evaluate")
def api_substrate_mutation_runtime_recall_shadow_profile_evaluate(
    request: RecallShadowEvaluateRequest,
    x_orion_operator_token: str | None = Header(default=None),
) -> Dict[str, Any]:
    _require_mutation_operator_guard(x_orion_operator_token)
    active = SUBSTRATE_MUTATION_STORE.active_recall_shadow_profile()
    if active is None:
        raise HTTPException(status_code=409, detail="no_active_shadow_profile")

    max_rows = max(1, min(int(settings.HUB_RECALL_SHADOW_EVAL_MAX_ROWS_PER_RUN), 512))
    requested_limit = int(request.corpus_limit or settings.HUB_RECALL_SHADOW_EVAL_DEFAULT_CORPUS_LIMIT)
    corpus_limit = max(1, min(requested_limit, max_rows))
    eval_rows: list[dict[str, Any]] = []
    failure_reason: str | None = None
    run_status = "dry_run" if request.dry_run else "completed"
    if request.eval_rows:
        eval_rows = [dict(row) for row in request.eval_rows if isinstance(row, dict)][:max_rows]
    else:
        base = settings.recall_service_url
        try:
            resp = requests.get(f"{base}/debug/recall/eval-suite", timeout=float(settings.HUB_RECALL_SHADOW_EVAL_TIMEOUT_SEC))
            payload = resp.json() if resp.status_code < 500 else {}
            eval_rows = [dict(row) for row in list(payload.get("rows") or []) if isinstance(row, dict)][:max_rows]
            if resp.status_code >= 500:
                failure_reason = f"recall_service_status_{resp.status_code}"
        except Exception as exc:
            eval_rows = []
            failure_reason = f"recall_service_unavailable:{exc.__class__.__name__}"
    if request.case_ids:
        keep = {str(cid) for cid in request.case_ids}
        eval_rows = [row for row in eval_rows if str(row.get("case_id")) in keep]
    eval_rows = eval_rows[:corpus_limit]
    readiness_before = dict(active.readiness_snapshot or {})
    if not eval_rows:
        run = RecallShadowEvalRunV1(
            profile_id=active.profile_id,
            dry_run=bool(request.dry_run),
            recorded_pressure_events=0,
            corpus_limit=corpus_limit,
            case_ids=[str(cid) for cid in request.case_ids][:256],
            eval_row_count=0,
            readiness_before=readiness_before,
            readiness_after=readiness_before,
            readiness_delta_summary={},
            pressure_event_refs=[],
            operator_rationale=str(request.operator_rationale or ""),
            status="failed" if not request.dry_run else "dry_run",
            failure_reason=failure_reason or "recall_shadow_eval_rows_empty",
        )
        saved_run = SUBSTRATE_MUTATION_STORE.record_recall_shadow_eval_run(run)
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": _source_meta(
                kind=SUBSTRATE_REVIEW_TELEMETRY_STORE.source_kind(),
                degraded=SUBSTRATE_REVIEW_TELEMETRY_STORE.degraded(),
                limit=corpus_limit,
                error=SUBSTRATE_REVIEW_TELEMETRY_STORE.last_error(),
            ),
            "data": {
                "profile_id": active.profile_id,
                "dry_run": bool(request.dry_run),
                "record_pressure_events": bool(request.record_pressure_events),
                "recorded_pressure_events": 0,
                "readiness_snapshot": readiness_before,
                "profile": active.model_dump(mode="json"),
                "operator_rationale": str(request.operator_rationale or ""),
                "production_recall_mode": "v1",
                "eval_run": saved_run.model_dump(mode="json"),
            },
        }

    from orion.substrate.recall_eval_bridge import pressure_evidence_from_eval_suite_rows

    suite_run_id = f"shadow-profile-eval:{active.profile_id}:{uuid4()}"
    events = pressure_evidence_from_eval_suite_rows(eval_rows, suite_run_id=suite_run_id)
    compare_rows: list[dict[str, Any]] = []
    failure_categories: list[str] = []
    recorded = 0
    eval_refs = list(active.eval_evidence_refs)
    pressure_event_refs: list[str] = []
    for ev in events:
        meta = dict(ev.metadata or {})
        cmp = meta.get("v1_v2_compare")
        if isinstance(cmp, dict):
            compare_rows.append(dict(cmp))
            failure_categories.append(str(ev.pressure_category))
            case_id = str(cmp.get("case_id") or "")
            if case_id:
                eval_refs.append(f"recall_eval_case:{case_id}")
        if request.record_pressure_events and not request.dry_run:
            _record_pressure_events_as_telemetry(
                events=[ev],
                correlation_id=str(ev.correlation_id or suite_run_id),
                source_event_id=str(ev.source_event_id or suite_run_id),
                invocation_surface="operator_review",
                ingest_notes=["recall_shadow_profile_evaluate", f"profile:{active.profile_id}"],
            )
            recorded += 1
            pressure_event_refs.append(f"pressure_event:{ev.pressure_event_id}")
    readiness = compute_recall_strategy_readiness(
        compare_rows=compare_rows,
        failure_categories=failure_categories,
        corpus_total_cases=default_eval_corpus_total_cases(),
    )
    updated = active
    if not request.dry_run:
        saved = SUBSTRATE_MUTATION_STORE.update_recall_strategy_profile(
            profile_id=active.profile_id,
            readiness_snapshot=readiness.model_dump(mode="json"),
            eval_evidence_refs=list(dict.fromkeys(eval_refs))[-128:],
        )
        if saved is not None:
            updated = saved
    run = RecallShadowEvalRunV1(
        profile_id=active.profile_id,
        dry_run=bool(request.dry_run),
        recorded_pressure_events=recorded,
        corpus_limit=corpus_limit,
        case_ids=[str(cid) for cid in request.case_ids][:256],
        eval_row_count=len(eval_rows),
        readiness_before=readiness_before,
        readiness_after=readiness.model_dump(mode="json"),
        readiness_delta_summary=_readiness_delta_summary(before=readiness_before, after=readiness.model_dump(mode="json")),
        pressure_event_refs=pressure_event_refs,
        operator_rationale=str(request.operator_rationale or ""),
        status=run_status,  # type: ignore[arg-type]
        failure_reason=failure_reason,
    )
    saved_run = SUBSTRATE_MUTATION_STORE.record_recall_shadow_eval_run(run)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": _source_meta(
            kind=SUBSTRATE_REVIEW_TELEMETRY_STORE.source_kind(),
            degraded=SUBSTRATE_REVIEW_TELEMETRY_STORE.degraded(),
            limit=corpus_limit,
            error=SUBSTRATE_REVIEW_TELEMETRY_STORE.last_error(),
        ),
        "data": {
            "profile_id": active.profile_id,
            "dry_run": bool(request.dry_run),
            "record_pressure_events": bool(request.record_pressure_events),
            "recorded_pressure_events": recorded,
            "readiness_snapshot": readiness.model_dump(mode="json"),
            "profile": updated.model_dump(mode="json"),
            "operator_rationale": str(request.operator_rationale or ""),
            "production_recall_mode": "v1",
            "eval_run": saved_run.model_dump(mode="json"),
        },
    }


@router.get("/api/substrate/mutation-runtime/recall-shadow-eval-runs")
def api_substrate_mutation_runtime_recall_shadow_eval_runs(
    limit: int = Query(default=20, ge=1, le=200),
    profile_id: str | None = None,
) -> Dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": _source_meta(
            kind=SUBSTRATE_MUTATION_STORE.source_kind(),
            degraded=SUBSTRATE_MUTATION_STORE.degraded(),
            limit=limit,
            error=SUBSTRATE_MUTATION_STORE.last_error(),
        ),
        "data": {
            "runs": SUBSTRATE_MUTATION_STORE.list_recall_shadow_eval_runs(limit=limit, profile_id=profile_id),
        },
    }


@router.get("/api/substrate/mutation-runtime/recall-shadow-eval-runs/{run_id}")
def api_substrate_mutation_runtime_recall_shadow_eval_run_detail(run_id: str) -> Dict[str, Any]:
    run = SUBSTRATE_MUTATION_STORE.get_recall_shadow_eval_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="recall_shadow_eval_run_not_found")
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": _source_meta(
            kind=SUBSTRATE_MUTATION_STORE.source_kind(),
            degraded=SUBSTRATE_MUTATION_STORE.degraded(),
            limit=1,
            error=SUBSTRATE_MUTATION_STORE.last_error(),
        ),
        "data": {"run": run.model_dump(mode="json")},
    }


@router.get("/api/substrate/mutation-runtime/recall-v1-v2-latest-eval")
def api_substrate_mutation_runtime_recall_v1_v2_latest_eval(
    eval_history_limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> Dict[str, Any]:
    telemetry_rows = SUBSTRATE_REVIEW_TELEMETRY_STORE.query(GraphReviewTelemetryQueryV1(limit=200))
    latest: dict[str, Any] | None = None
    latest_ts = ""
    latest_eval: dict[str, Any] | None = None
    latest_eval_ts = ""
    eval_history: list[dict[str, Any]] = []
    for row in telemetry_rows:
        for event in list(row.pressure_events or []):
            meta = event.metadata if isinstance(event.metadata, dict) else {}
            compare = meta.get("v1_v2_compare") if isinstance(meta.get("v1_v2_compare"), dict) else None
            if not isinstance(compare, dict):
                continue
            observed = event.observed_at.isoformat()
            if observed > latest_ts:
                latest_ts = observed
                latest = {
                    "observed_at": observed,
                    "pressure_event_id": event.pressure_event_id,
                    "pressure_category": event.pressure_category,
                    "v1_v2_compare": compare,
                    "anchor_plan": meta.get("anchor_plan"),
                    "selected_evidence_cards": meta.get("selected_evidence_cards"),
                    "recall_evidence_kind": meta.get("recall_evidence_kind"),
                }
            if str(compare.get("source") or "") == "recall_eval_suite":
                row_payload = {
                    "observed_at": observed,
                    "pressure_event_id": event.pressure_event_id,
                    "pressure_category": event.pressure_category,
                    "recall_eval_case": meta.get("recall_eval_case"),
                    "v1_v2_compare": compare,
                    "invocation_surface": row.invocation_surface,
                    "telemetry_selection_reason": row.selection_reason,
                }
                eval_history.append(row_payload)
                if observed > latest_eval_ts:
                    latest_eval_ts = observed
                    latest_eval = {
                        "observed_at": observed,
                        "pressure_event_id": event.pressure_event_id,
                        "pressure_category": event.pressure_category,
                        "recall_eval_case": meta.get("recall_eval_case"),
                        "v1_v2_compare": compare,
                    }
    eval_history.sort(key=lambda item: str(item.get("observed_at") or ""), reverse=True)
    hist_limit = max(1, min(int(eval_history_limit), 100))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": _source_meta(
            kind=SUBSTRATE_REVIEW_TELEMETRY_STORE.source_kind(),
            degraded=SUBSTRATE_REVIEW_TELEMETRY_STORE.degraded(),
            limit=200,
            error=SUBSTRATE_REVIEW_TELEMETRY_STORE.last_error(),
        ),
        "data": {
            "latest_recall_v1_v2_eval_summary": latest,
            "latest_recall_eval_suite_row": latest_eval,
            "recent_recall_eval_suite_rows": eval_history[:hist_limit],
        },
    }


@router.get("/api/substrate/mutation-runtime/recall-strategy-readiness")
def api_substrate_mutation_runtime_recall_strategy_readiness(
    telemetry_limit: Annotated[int, Query(ge=10, le=500)] = 200,
) -> Dict[str, Any]:
    """
    Read-only advisory readiness for Recall V2 / recall-strategy candidates from recent review telemetry.
    Does not change routing, apply surfaces, or promotion state.
    """
    rows = SUBSTRATE_REVIEW_TELEMETRY_STORE.query(GraphReviewTelemetryQueryV1(limit=telemetry_limit))
    readiness = readiness_from_telemetry_records(rows)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": _source_meta(
            kind=SUBSTRATE_REVIEW_TELEMETRY_STORE.source_kind(),
            degraded=SUBSTRATE_REVIEW_TELEMETRY_STORE.degraded(),
            limit=telemetry_limit,
            error=SUBSTRATE_REVIEW_TELEMETRY_STORE.last_error(),
        ),
        "data": {"readiness": readiness.model_dump(mode="json")},
    }


def _is_cognitive_proposal_payload(payload: Dict[str, Any]) -> bool:
    return str(payload.get("lane") or "") == "cognitive" or str(payload.get("target_surface") or "") in _COGNITIVE_SURFACES


def _cognitive_review_state_for_decision(decision: str) -> str:
    mapping = {
        "accept_as_draft": "accepted_as_draft",
        "reject": "rejected",
        "archive": "archived",
        "supersede": "superseded",
    }
    return mapping.get(decision, "pending_review")


def _cognitive_safety_block() -> dict[str, bool]:
    return {
        "live_apply_enabled": False,
        "identity_kernel_rewrite_enabled": False,
        "production_self_model_rewrite_enabled": False,
        "policy_override_enabled": False,
        "freeform_prompt_self_rewrite_enabled": False,
        "identity_kernel_rewrite_forbidden": True,
        "production_self_model_rewrite_forbidden": True,
        "policy_override_forbidden": True,
        "prompt_rewrite_forbidden": True,
        "cognitive_live_apply_forbidden": True,
        "mutation_execute_once_forbidden": True,
    }


def _cognitive_review_posture_summary(proposals: list[dict[str, Any]], drafts: list[dict[str, Any]], notes: list[dict[str, Any]]) -> dict[str, Any]:
    pending_reviews = sum(1 for row in proposals if str(row.get("rollout_state") or "") in {"pending_review", "approved"})
    active_drafts = sum(1 for row in drafts if str(row.get("state") or "") == "active_draft")
    active_notes = sum(1 for row in notes if str(row.get("status") or "") == "active")
    recommended_action = "review_pending_proposals" if pending_reviews else ("curate_active_drafts" if active_drafts else "monitor")
    return {
        "pending_review_count": pending_reviews,
        "active_draft_count": active_drafts,
        "active_stance_note_count": active_notes,
        "recommended_action": recommended_action,
    }


@router.get("/api/substrate/mutation-runtime/cognitive-pressure")
def api_substrate_mutation_runtime_cognitive_pressure(
    limit: int = Query(default=50, ge=1, le=200),
) -> Dict[str, Any]:
    rows = SUBSTRATE_MUTATION_STORE.recent_signals(limit=500)
    filtered = [
        {
            "signal_id": row.get("signal_id"),
            "event_kind": row.get("event_kind"),
            "target_surface": row.get("target_surface"),
            "target_zone": row.get("target_zone"),
            "strength": row.get("strength"),
            "source_ref": row.get("source_ref"),
            "evidence_refs": list(row.get("evidence_refs") or []),
            "metadata": dict(row.get("metadata") or {}),
            "detected_at": row.get("detected_at"),
        }
        for row in rows
        if str(row.get("target_surface") or "") in _COGNITIVE_SURFACES
        or str(row.get("event_kind") or "") in {
            "contradiction_pressure",
            "identity_continuity_pressure",
            "stance_drift_pressure",
            "social_continuity_pressure",
        }
    ][:limit]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": _source_meta(
            kind=SUBSTRATE_MUTATION_STORE.source_kind(),
            degraded=SUBSTRATE_MUTATION_STORE.degraded(),
            limit=limit,
            error=SUBSTRATE_MUTATION_STORE.last_error(),
        ),
        "data": {"recent_cognitive_pressure": filtered},
    }


@router.get("/api/substrate/mutation-runtime/cognitive-proposals")
def api_substrate_mutation_runtime_cognitive_proposals(
    limit: int = Query(default=20, ge=1, le=200),
) -> Dict[str, Any]:
    proposals = sorted(SUBSTRATE_MUTATION_STORE._proposals.values(), key=lambda item: item.created_at, reverse=True)
    filtered = [
        proposal.model_dump(mode="json")
        for proposal in proposals
        if proposal.lane == "cognitive" or proposal.target_surface in _COGNITIVE_SURFACES
    ][:limit]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": _source_meta(
            kind=SUBSTRATE_MUTATION_STORE.source_kind(),
            degraded=SUBSTRATE_MUTATION_STORE.degraded(),
            limit=limit,
            error=SUBSTRATE_MUTATION_STORE.last_error(),
        ),
        "data": {"recent_cognitive_proposals": filtered},
    }


@router.get("/api/substrate/cognitive-proposals/status")
def api_substrate_cognitive_proposals_status(
    limit: int = Query(default=20, ge=1, le=200),
) -> Dict[str, Any]:
    proposals = api_substrate_mutation_runtime_cognitive_proposals(limit=limit).get("data", {}).get("recent_cognitive_proposals", [])
    drafts = SUBSTRATE_MUTATION_STORE.list_cognitive_proposal_drafts(limit=limit)
    stance_notes = SUBSTRATE_MUTATION_STORE.list_cognitive_stance_notes(limit=limit)
    counts_by_state: dict[str, int] = {}
    counts_by_class: dict[str, int] = {}
    for row in proposals:
        state = str(row.get("rollout_state") or "unknown")
        counts_by_state[state] = counts_by_state.get(state, 0) + 1
        klass = str(row.get("mutation_class") or "unknown")
        counts_by_class[klass] = counts_by_class.get(klass, 0) + 1
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": _source_meta(
            kind=SUBSTRATE_MUTATION_STORE.source_kind(),
            degraded=SUBSTRATE_MUTATION_STORE.degraded(),
            limit=limit,
            error=SUBSTRATE_MUTATION_STORE.last_error(),
        ),
        "data": {
            "schema_version": "cognitive_proposal_status.v1",
            "live_apply_enabled": False,
            "safety": _cognitive_safety_block(),
            "counts_by_state": counts_by_state,
            "counts_by_class": counts_by_class,
            "recent_proposals": proposals[: min(limit, 12)],
            "recent_drafts": drafts[: min(limit, 12)],
            "recent_reviews": SUBSTRATE_MUTATION_STORE.recent_cognitive_reviews(limit=min(limit, 12)),
            "active_stance_notes": [row for row in stance_notes if str(row.get("status") or "") == "active"][: min(limit, 12)],
            "review_posture": _cognitive_review_posture_summary(proposals, drafts, stance_notes),
            "warnings": [],
        },
    }


@router.get("/api/substrate/cognitive-proposals")
def api_substrate_cognitive_proposals(
    limit: int = Query(default=20, ge=1, le=200),
) -> Dict[str, Any]:
    return api_substrate_mutation_runtime_cognitive_proposals(limit=limit)


@router.get("/api/substrate/mutation-runtime/cognitive-proposals/{proposal_id}/lineage")
def api_substrate_mutation_runtime_cognitive_proposal_lineage(proposal_id: str) -> Dict[str, Any]:
    lifecycle = SUBSTRATE_MUTATION_STORE.lifecycle_for_proposal(proposal_id)
    if lifecycle is None:
        raise HTTPException(status_code=404, detail="cognitive_proposal_not_found")
    proposal = lifecycle.get("proposal") if isinstance(lifecycle, dict) else None
    if not isinstance(proposal, dict) or not _is_cognitive_proposal_payload(proposal):
        raise HTTPException(status_code=404, detail="cognitive_proposal_not_found")
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": _source_meta(
            kind=SUBSTRATE_MUTATION_STORE.source_kind(),
            degraded=SUBSTRATE_MUTATION_STORE.degraded(),
            limit=1,
            error=SUBSTRATE_MUTATION_STORE.last_error(),
        ),
        "data": {"lineage": lifecycle},
    }


@router.get("/api/substrate/mutation-runtime/cognitive-proposals/{proposal_id}")
def api_substrate_mutation_runtime_cognitive_proposal_detail(proposal_id: str) -> Dict[str, Any]:
    proposal = SUBSTRATE_MUTATION_STORE.get_proposal(proposal_id)
    if proposal is None:
        raise HTTPException(status_code=404, detail="cognitive_proposal_not_found")
    payload = proposal.model_dump(mode="json")
    if not _is_cognitive_proposal_payload(payload):
        raise HTTPException(status_code=404, detail="cognitive_proposal_not_found")
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": _source_meta(
            kind=SUBSTRATE_MUTATION_STORE.source_kind(),
            degraded=SUBSTRATE_MUTATION_STORE.degraded(),
            limit=1,
            error=SUBSTRATE_MUTATION_STORE.last_error(),
        ),
        "data": {"proposal": payload},
    }


@router.get("/api/substrate/cognitive-proposals/{proposal_id}")
def api_substrate_cognitive_proposal_detail(proposal_id: str) -> Dict[str, Any]:
    return api_substrate_mutation_runtime_cognitive_proposal_detail(proposal_id)


@router.get("/api/substrate/mutation-runtime/cognitive-proposals/{proposal_id}/evidence")
def api_substrate_mutation_runtime_cognitive_proposal_evidence(proposal_id: str) -> Dict[str, Any]:
    lifecycle = SUBSTRATE_MUTATION_STORE.lifecycle_for_proposal(proposal_id)
    if lifecycle is None:
        raise HTTPException(status_code=404, detail="cognitive_proposal_not_found")
    proposal = lifecycle.get("proposal") if isinstance(lifecycle, dict) else None
    if not isinstance(proposal, dict) or not _is_cognitive_proposal_payload(proposal):
        raise HTTPException(status_code=404, detail="cognitive_proposal_not_found")
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": _source_meta(
            kind=SUBSTRATE_MUTATION_STORE.source_kind(),
            degraded=SUBSTRATE_MUTATION_STORE.degraded(),
            limit=1,
            error=SUBSTRATE_MUTATION_STORE.last_error(),
        ),
        "data": {
            "proposal_id": proposal_id,
            "evidence_refs": list(proposal.get("evidence_refs") or []),
            "signals": list((lifecycle.get("signals") if isinstance(lifecycle, dict) else []) or []),
            "pressure": (lifecycle.get("pressure") if isinstance(lifecycle, dict) else None),
        },
    }


@router.post("/api/substrate/mutation-runtime/cognitive-proposals/{proposal_id}/review")
def api_substrate_mutation_runtime_cognitive_proposal_review(
    proposal_id: str,
    request: CognitiveProposalReviewRequest,
) -> Dict[str, Any]:
    proposal = SUBSTRATE_MUTATION_STORE.get_proposal(proposal_id)
    if proposal is None:
        raise HTTPException(status_code=404, detail="cognitive_proposal_not_found")
    payload = proposal.model_dump(mode="json")
    if not _is_cognitive_proposal_payload(payload):
        raise HTTPException(status_code=404, detail="cognitive_proposal_not_found")
    review = CognitiveProposalReviewV1(
        proposal_id=proposal_id,
        state=request.state,
        reviewer=request.reviewer,
        rationale=request.rationale,
        notes=list(request.notes),
    )
    draft = SUBSTRATE_MUTATION_STORE.record_cognitive_review(review)
    return {
        "ok": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data": {
            "review": review.model_dump(mode="json"),
            "draft_recommendation": draft.model_dump(mode="json") if draft is not None else None,
        },
    }


@router.post("/api/substrate/cognitive-proposals/{proposal_id}/review")
def api_substrate_cognitive_proposal_review(
    proposal_id: str,
    request: CognitiveProposalReviewActionRequest,
    x_orion_operator_token: str | None = Header(default=None),
) -> Dict[str, Any]:
    _require_mutation_operator_guard(x_orion_operator_token)
    proposal = SUBSTRATE_MUTATION_STORE.get_proposal(proposal_id)
    if proposal is None:
        raise HTTPException(status_code=404, detail="cognitive_proposal_not_found")
    payload = proposal.model_dump(mode="json")
    if not _is_cognitive_proposal_payload(payload):
        raise HTTPException(status_code=404, detail="cognitive_proposal_not_found")
    review_state = _cognitive_review_state_for_decision(request.decision)
    review = CognitiveProposalReviewV1(
        proposal_id=proposal_id,
        state=review_state,  # type: ignore[arg-type]
        reviewer=request.reviewer,
        rationale=request.rationale,
        notes=list(request.review_labels or []),
    )
    created_draft = SUBSTRATE_MUTATION_STORE.record_cognitive_review(review)
    created_note: CognitiveStanceNoteV1 | None = None
    if request.create_stance_note:
        active_draft = created_draft
        if active_draft is None:
            existing = SUBSTRATE_MUTATION_STORE.list_cognitive_proposal_drafts(limit=50)
            row = next((item for item in existing if str(item.get("proposal_id") or "") == proposal_id and str(item.get("state") or "") == "active_draft"), None)
            if row is not None:
                active_draft = CognitiveProposalDraftV1.model_validate(row)
        if active_draft is not None:
            created_note = SUBSTRATE_MUTATION_STORE.record_cognitive_stance_note(
                CognitiveStanceNoteV1(
                    source_proposal_id=proposal_id,
                    source_draft_id=active_draft.draft_id,
                    proposal_class=proposal.mutation_class,
                    visibility=request.stance_visibility,
                    ttl_turns=request.stance_ttl_turns,
                    summary=request.stance_summary or "operator accepted cognitive draft context",
                    note=request.stance_note or request.rationale,
                    evidence_refs=list(proposal.evidence_refs),
                    review_ref=review.review_id,
                    safety_scope={
                        "authoritative": False,
                        "identity_or_policy_rewrite": False,
                        "live_apply": False,
                    },
                    lineage={"proposal_id": proposal_id, "review_id": review.review_id, "draft_id": active_draft.draft_id},
                )
            )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data": {
            "schema_version": "cognitive_proposal_review_result.v1",
            "proposal_id": proposal_id,
            "review_recorded": True,
            "decision": request.decision,
            "review": review.model_dump(mode="json"),
            "draft": created_draft.model_dump(mode="json") if created_draft is not None else None,
            "stance_note": created_note.model_dump(mode="json") if created_note is not None else None,
            "pressure_emitted": False if not request.should_emit_pressure else False,
            "safety": {
                "production_default_unchanged": True,
                "promotion_performed": False,
                "apply_performed": False,
                "execute_once_invoked": False,
            },
            "warnings": [],
        },
    }


@router.get("/api/substrate/mutation-runtime/cognitive-drafts")
def api_substrate_mutation_runtime_cognitive_drafts(
    limit: int = Query(default=20, ge=1, le=200),
) -> Dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": _source_meta(
            kind=SUBSTRATE_MUTATION_STORE.source_kind(),
            degraded=SUBSTRATE_MUTATION_STORE.degraded(),
            limit=limit,
            error=SUBSTRATE_MUTATION_STORE.last_error(),
        ),
        "data": {
            "draft_recommendations": SUBSTRATE_MUTATION_STORE.recent_cognitive_drafts(limit=limit),
            "recent_reviews": SUBSTRATE_MUTATION_STORE.recent_cognitive_reviews(limit=limit),
        },
    }


@router.get("/api/substrate/cognitive-drafts")
def api_substrate_cognitive_drafts(
    limit: int = Query(default=20, ge=1, le=200),
    state: str | None = None,
) -> Dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": _source_meta(
            kind=SUBSTRATE_MUTATION_STORE.source_kind(),
            degraded=SUBSTRATE_MUTATION_STORE.degraded(),
            limit=limit,
            error=SUBSTRATE_MUTATION_STORE.last_error(),
        ),
        "data": {
            "drafts": SUBSTRATE_MUTATION_STORE.list_cognitive_proposal_drafts(limit=limit, state=state),
            "safety": _cognitive_safety_block(),
        },
    }


@router.get("/api/substrate/cognitive-drafts/{draft_id}")
def api_substrate_cognitive_draft_detail(draft_id: str) -> Dict[str, Any]:
    draft = SUBSTRATE_MUTATION_STORE.get_cognitive_proposal_draft(draft_id)
    if draft is None:
        raise HTTPException(status_code=404, detail="cognitive_draft_not_found")
    return {"generated_at": datetime.now(timezone.utc).isoformat(), "data": {"draft": draft.model_dump(mode="json")}}


@router.post("/api/substrate/cognitive-drafts/{draft_id}/archive")
def api_substrate_cognitive_draft_archive(
    draft_id: str,
    _: CognitiveDraftArchiveRequest,
    x_orion_operator_token: str | None = Header(default=None),
) -> Dict[str, Any]:
    _require_mutation_operator_guard(x_orion_operator_token)
    draft = SUBSTRATE_MUTATION_STORE.archive_cognitive_proposal_draft(draft_id)
    if draft is None:
        raise HTTPException(status_code=404, detail="cognitive_draft_not_found")
    return {"generated_at": datetime.now(timezone.utc).isoformat(), "data": {"draft": draft.model_dump(mode="json"), "archived": True}}


@router.post("/api/substrate/cognitive-drafts/{draft_id}/create-stance-note")
def api_substrate_cognitive_draft_create_stance_note(
    draft_id: str,
    request: CognitiveCreateStanceNoteRequest,
    x_orion_operator_token: str | None = Header(default=None),
) -> Dict[str, Any]:
    _require_mutation_operator_guard(x_orion_operator_token)
    draft = SUBSTRATE_MUTATION_STORE.get_cognitive_proposal_draft(draft_id)
    if draft is None:
        raise HTTPException(status_code=404, detail="cognitive_draft_not_found")
    proposal = SUBSTRATE_MUTATION_STORE.get_proposal(draft.proposal_id)
    if proposal is None:
        raise HTTPException(status_code=404, detail="cognitive_proposal_not_found")
    note = SUBSTRATE_MUTATION_STORE.record_cognitive_stance_note(
        CognitiveStanceNoteV1(
            source_proposal_id=draft.proposal_id,
            source_draft_id=draft_id,
            proposal_class=proposal.mutation_class,
            visibility=request.visibility,
            ttl_turns=request.ttl_turns,
            summary=request.summary or draft.summary,
            note=request.note or request.rationale,
            evidence_refs=list(draft.evidence_refs),
            safety_scope={"authoritative": False, "live_apply": False},
            lineage={"draft_id": draft_id, "proposal_id": draft.proposal_id},
        )
    )
    return {"generated_at": datetime.now(timezone.utc).isoformat(), "data": {"stance_note": note.model_dump(mode="json"), "created": True}}


@router.get("/api/substrate/cognitive-stance-notes")
def api_substrate_cognitive_stance_notes(
    limit: int = Query(default=20, ge=1, le=200),
    status: str | None = None,
) -> Dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data": {
            "stance_notes": SUBSTRATE_MUTATION_STORE.list_cognitive_stance_notes(limit=limit, status=status),
            "safety": {"authoritative": False, "identity_or_policy_rewrite": False, "live_apply": False},
        },
    }


@router.post("/api/substrate/cognitive-stance-notes/{stance_note_id}/archive")
def api_substrate_cognitive_stance_note_archive(
    stance_note_id: str,
    __: CognitiveDraftArchiveRequest,
    x_orion_operator_token: str | None = Header(default=None),
) -> Dict[str, Any]:
    _require_mutation_operator_guard(x_orion_operator_token)
    note = SUBSTRATE_MUTATION_STORE.archive_cognitive_stance_note(stance_note_id)
    if note is None:
        raise HTTPException(status_code=404, detail="cognitive_stance_note_not_found")
    return {"generated_at": datetime.now(timezone.utc).isoformat(), "data": {"stance_note": note.model_dump(mode="json"), "archived": True}}


@router.get("/api/substrate/mutation-runtime/live-routing-surface")
def api_substrate_mutation_runtime_live_routing_surface() -> Dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data": inspect_chat_reflective_lane_threshold(),
    }


@router.get("/api/substrate/mutation-runtime/routing-replay-inspect")
def api_substrate_mutation_runtime_routing_replay_inspect(
    limit: int = Query(default=50, ge=1, le=200),
) -> Dict[str, Any]:
    return _routing_replay_inspection_payload(limit=limit)


@router.get("/api/substrate/mutation-runtime/cognition-context")
def api_substrate_mutation_runtime_cognition_context() -> Dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data": build_mutation_cognition_context(store=SUBSTRATE_MUTATION_STORE),
    }


@router.get("/api/substrate/mutation-runtime/routing-live-ramp-posture")
def api_substrate_mutation_runtime_routing_live_ramp_posture() -> Dict[str, Any]:
    return _routing_live_ramp_posture_payload()


@router.get("/api/substrate/autonomy-constitution")
def api_substrate_autonomy_constitution() -> Dict[str, Any]:
    constitution = load_autonomy_constitution()
    summary = constitution_summary(constitution)
    payload = {
        "schema_version": constitution.schema_version,
        "loaded_at": constitution.loaded_at,
        "source": constitution.source,
        "surfaces": [row.model_dump(mode="json") for row in constitution.surfaces],
        "safety_invariants": list(constitution.safety_invariants),
        "summary": summary,
        "warnings": list(constitution.warnings),
    }
    logger.info(
        "event=autonomy_constitution_endpoint_generated surface_count=%s live_apply_surface_count=%s blocked_surface_count=%s protected_surface_count=%s warning_count=%s",
        len(payload["surfaces"]),
        len(summary.get("live_apply_surfaces") or []),
        len(summary.get("blocked_surfaces") or []),
        len(summary.get("protected_surfaces") or []),
        len(payload["warnings"]),
    )
    return payload


@router.get("/api/substrate/autonomy-readiness")
def api_substrate_autonomy_readiness() -> Dict[str, Any]:
    """Unified read-only autonomy readiness posture snapshot."""
    return _autonomy_readiness_payload()
