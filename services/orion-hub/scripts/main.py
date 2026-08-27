# scripts/main.py

import functools
import html
import json
import logging
import os
import subprocess
import asyncio
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from scripts.settings import settings
from scripts.api_routes import router as api_router
from orion.core.storage.memory_cards import apply_memory_cards_schema
from orion.memory.crystallization.repository import apply_memory_crystallizations_schema
from scripts.memory_routes import router as memory_router
from scripts.crystallization_routes import router as crystallization_router
from scripts.mind_routes import router as mind_router
from scripts.memory_graph_routes import router as memory_graph_router
from scripts.memory_consolidation_draft_routes import router as memory_consolidation_draft_router
from scripts.proposal_review_routes import router as proposal_review_router
from scripts.concept_atlas_routes import router as concept_atlas_router
from scripts.self_brain_routes import router as self_brain_router
from scripts.chat_attachments import router as chat_attachments_router
import scripts.api_routes as api_routes_runtime
import scripts.concept_atlas_routes as concept_atlas_routes_runtime
import scripts.vision_affect_ambient as vision_affect_ambient_runtime
import scripts.vision_frame_cache as vision_frame_cache_runtime
from scripts.websocket_handler import websocket_endpoint
from scripts.service_logs_ws import service_logs_websocket_endpoint
from scripts.biometrics_cache import BiometricsCache
from scripts.notification_cache import NotificationCache
from scripts.bus_synaptic_trigger_notifier import BusSynapticTriggerNotifier
from orion.core.bus.bus_schemas import ServiceRef
from scripts.curiosity_investigation import CuriosityInvestigation
from scripts.endogenous_outreach import EndogenousOutreach
import scripts.tension_outreach_trigger as tension_outreach_trigger
from scripts.room_claude_relay import RoomClaudeRelay
from scripts.agent_step_relay import AgentStepRelay
from scripts.harness_step_relay import HarnessStepRelay
from scripts.signals_inspect_cache import SignalsInspectCache
from scripts.cognition_trace_cache import CognitionTraceCache
from scripts.embodiment_outcome_cache import EmbodimentOutcomeCache

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_service_chassis import ChassisConfig, HeartbeatOnly
from scripts.bus_clients.cortex_client import CortexGatewayClient
from scripts.bus_clients.tts_client import TTSClient


# ───────────────────────────────────────────────────────────────
# 🪵 Logging Setup
# ───────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("orion-hub")
SERVICE_ROOT = Path(__file__).resolve().parents[1]
TEMPLATES_DIR = SERVICE_ROOT / "templates"
STATIC_DIR = SERVICE_ROOT / "static"


def _discover_git_sha() -> str:
    """Best-effort git SHA discovery for cache-bust tokens when env vars are absent."""
    repo_root = SERVICE_ROOT.parent.parent
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return (result.stdout or "").strip()
    except Exception:
        return ""


def _ui_asset_mtime_token() -> str:
    """Best-effort mtime token so uncommitted UI edits bust browser caches.

    Globs rather than naming files. The previous hardcoded four-file list
    silently excluded every other module the template loads -- confirmed
    2026-08-14 when an edit to memory-crystallization-ui.js produced no new ?v=,
    so a browser with the old file cached would keep running it. A list like
    that goes stale on the exact patch that needed it, and the failure is
    invisible: the server is serving new code, the browser just never asks for
    it.

    rglob, not glob("js/*.js"): a first pass at this fix still missed
    static/js/vendor/ and every template except index.html, which is the same
    bug one directory down. Templates are globbed too because the standalone
    pages (causal_geometry, concept_atlas, substrate, substrate_atlas) are
    served with the same token. sorted() keeps the token stable for a given set
    of mtimes.
    """
    candidates = [
        *sorted(STATIC_DIR.rglob("*.js")),
        *sorted(STATIC_DIR.rglob("*.css")),
        *sorted(TEMPLATES_DIR.glob("*.html")),
    ]
    mtimes: list[int] = []
    for path in candidates:
        try:
            mtimes.append(int(path.stat().st_mtime))
        except Exception:
            continue
    return str(max(mtimes)) if mtimes else ""


def build_hub_ui_asset_version() -> str:
    """Build an explicit cache-busting token for Hub static assets."""
    explicit = os.getenv("HUB_UI_BUILD")
    build_id = os.getenv("BUILD_ID")
    git_sha = os.getenv("GIT_SHA") or os.getenv("SOURCE_COMMIT")
    discovered_git_sha = _discover_git_sha()
    build_ts = os.getenv("BUILD_TIMESTAMP")
    service_version = settings.SERVICE_VERSION
    mtime_token = _ui_asset_mtime_token()

    # CI/build ids identify a build, but volume-mounted static/ can change without
    # a new image. Append the max mtime of key UI files so each Hub restart (and
    # any template reload) can surface a new ?v= for script tags.
    for candidate in (explicit, build_id, git_sha, build_ts):
        value = str(candidate or "").strip()
        if value:
            if mtime_token:
                return f"{value}-{mtime_token}"
            return value
    # For local/dev paths, include mtime token so restarts pick up UI edits.
    if discovered_git_sha and mtime_token:
        return f"{discovered_git_sha}-{mtime_token}"
    if discovered_git_sha:
        return discovered_git_sha
    if service_version and mtime_token:
        return f"{service_version}-{mtime_token}"
    if service_version:
        return service_version
    return "dev"


def _memory_store_banner(*, pool_ok: bool, dsn_configured: bool) -> tuple[str, str, str]:
    if pool_ok:
        return (
            "true",
            "border-emerald-800/60 bg-emerald-950/30 text-emerald-100",
            "Memory store connected. Operator curation and /api/memory/* are available.",
        )
    if not dsn_configured:
        return (
            "false",
            "border-amber-800/60 bg-amber-950/40 text-amber-100",
            (
                "Memory store unavailable: RECALL_PG_DSN is not set. "
                "Set it to your conjourney Postgres URL (same DB recall uses), then restart Hub."
            ),
        )
    return (
        "false",
        "border-amber-800/60 bg-amber-950/40 text-amber-100",
        (
            "Memory store unavailable: RECALL_PG_DSN is set but Postgres did not open a pool. "
            "Check Hub logs, credentials, and database reachability."
        ),
    )


def render_hub_index_html(*, memory_pool_ok: bool | None = None) -> str:
    """Render Hub index.html from disk so volume-mounted template edits apply without stale cache."""
    try:
        rendered = (TEMPLATES_DIR / "index.html").read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.error("CRITICAL: 'templates/index.html' not found.")
        return "<html><body><h1>UI template missing</h1></body></html>"

    ui_asset_version = build_hub_ui_asset_version()
    rendered = rendered.replace("{{NOTIFY_TOAST_SECONDS}}", str(settings.NOTIFY_TOAST_SECONDS))
    rendered = rendered.replace("{{HUB_UI_ASSET_VERSION}}", ui_asset_version)

    from scripts.memory_graph_suggest_timeout import hub_client_fetch_timeout_ms

    mg_escalation = bool(getattr(settings, "MEMORY_GRAPH_SUGGEST_ENABLE_ESCALATION", True))
    hub_cfg = {
        "apiBaseOverride": settings.HUB_API_BASE_OVERRIDE or "",
        "wsBaseOverride": settings.HUB_WS_BASE_OVERRIDE or "",
        "autoDefaultEnabled": bool(settings.HUB_AUTO_DEFAULT_ENABLED),
        "agentClaudeEnabled": bool(getattr(settings, "HUB_AGENT_CLAUDE_ENABLED", False)),
        "aitownEnabled": bool(getattr(settings, "HUB_AITOWN_ENABLED", False)),
        "worldPulseFixtureRunEnabled": bool(settings.WORLD_PULSE_UI_FIXTURE_RUN_ENABLED),
        "proposalReviewEnabled": bool(getattr(settings, "HUB_PROPOSAL_REVIEW_ENABLED", False)),
        "memoryGraphSuggestFetchTimeoutMs": hub_client_fetch_timeout_ms(
            settings, escalation_enabled=mg_escalation
        ),
    }
    rendered = rendered.replace("{{HUB_CFG}}", json.dumps(hub_cfg))

    from scripts.aitown_ui import render_aitown_tab_blocks

    aitown_nav, aitown_panel = render_aitown_tab_blocks(settings)
    rendered = rendered.replace("{{HUB_AITOWN_TAB_NAV}}", aitown_nav)
    rendered = rendered.replace("{{HUB_AITOWN_PANEL}}", aitown_panel)

    proposal_review_panel = ""
    proposal_review_script = ""
    if bool(getattr(settings, "HUB_PROPOSAL_REVIEW_ENABLED", False)):
        proposal_review_panel = """
      <div class="w-full bg-gray-900 rounded-2xl shadow-lg p-5 space-y-3" id="proposalReviewPanel">
        <div class="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 class="text-lg font-semibold text-white">Pending Decisions</h3>
            <div class="text-[11px] text-gray-400">Agent proposals from context-exec that need human approve/reject before anything can run.</div>
          </div>
          <div class="flex items-center gap-2">
            <select id="proposalReviewFilter" class="hidden bg-gray-800 text-gray-200 rounded border border-gray-700 px-2 py-1 text-xs">
              <option value="pending_review" selected>Pending review</option>
              <option value="blocked">Blocked</option>
              <option value="stored">Stored</option>
              <option value="approved">Approved history</option>
              <option value="rejected">Rejected history</option>
            </select>
            <button id="proposalReviewRefreshButton" type="button" class="text-xs bg-gray-800 hover:bg-gray-700 text-gray-200 rounded-lg px-3 py-1 border border-gray-700">Refresh</button>
          </div>
        </div>
        <div id="proposalReviewStatus" class="text-xs text-gray-500">Loading…</div>
        <div id="proposalReviewList" class="space-y-2 max-h-56 overflow-y-auto text-xs"></div>
        <div id="proposalReviewDetail" class="hidden rounded-xl border border-gray-700 bg-gray-900/50 p-3 text-[11px] space-y-2 text-gray-300"></div>
      </div>"""
        proposal_review_script = (
            f'<script src="/static/js/proposal-review-ui.js?v={ui_asset_version}" defer></script>'
        )
    rendered = rendered.replace("{{HUB_PROPOSAL_REVIEW_PANEL}}", proposal_review_panel)
    rendered = rendered.replace("{{HUB_PROPOSAL_REVIEW_SCRIPT}}", proposal_review_script)

    from scripts.api_routes import resolve_hub_autonomy_subject_display

    rendered = rendered.replace(
        "{{HUB_AUTONOMY_SUBJECT_DISPLAY}}",
        resolve_hub_autonomy_subject_display(),
    )

    if "{{HUB_MEMORY_STORE_READY}}" in rendered:
        if memory_pool_ok is None:
            memory_pool_ok = False
        dsn_configured = bool(str(getattr(settings, "RECALL_PG_DSN", "") or "").strip())
        ready, banner_class, banner_text = _memory_store_banner(
            pool_ok=memory_pool_ok,
            dsn_configured=dsn_configured,
        )
        rendered = rendered.replace("{{HUB_MEMORY_STORE_READY}}", ready)
        rendered = rendered.replace("{{HUB_MEMORY_STORE_BANNER_CLASS}}", banner_class)
        rendered = rendered.replace("{{HUB_MEMORY_STORE_BANNER_TEXT}}", html.escape(banner_text))

    return rendered


# ───────────────────────────────────────────────────────────────
# 🌐 FastAPI App & Shared Service Handles
# ───────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.SERVICE_NAME,
    version=settings.SERVICE_VERSION,
)

# These are populated on startup and imported by other modules:
bus: Optional[OrionBusAsync] = None
rpc_bus: Optional[OrionBusAsync] = None
cortex_client: Optional[CortexGatewayClient] = None
tts_client: Optional[TTSClient] = None
html_content: str = "<html><body><h1>Error loading UI</h1></body></html>"
biometrics_cache: Optional[BiometricsCache] = None
notification_cache: Optional[NotificationCache] = None
bus_synaptic_trigger_notifier: Optional[BusSynapticTriggerNotifier] = None

endogenous_outreach: Optional[EndogenousOutreach] = None
curiosity_investigation: Optional[CuriosityInvestigation] = None
room_claude_relay: Optional[RoomClaudeRelay] = None
agent_step_relay: Optional[AgentStepRelay] = None
harness_step_relay: Optional[HarnessStepRelay] = None
signals_inspect_cache: Optional[SignalsInspectCache] = None
cognition_trace_cache: Optional[CognitionTraceCache] = None
embodiment_outcome_cache: Optional[EmbodimentOutcomeCache] = None
presence_state: Optional["PresenceState"] = None
presence_context_store: Optional["PresenceContextStore"] = None
substrate_autonomy_task: Optional[asyncio.Task] = None
substrate_decay_task: Optional[asyncio.Task] = None
substrate_topic_foundry_scheduler_task: Optional[asyncio.Task] = None
affect_ambient_loop_task: Optional[asyncio.Task] = None
heartbeat_chassis: Optional[HeartbeatOnly] = None


def build_heartbeat_chassis() -> HeartbeatOnly:
    """Own, independent bus connection publishing SystemHealthV1 to orion:system:health
    every HEARTBEAT_INTERVAL_SEC. Deliberately separate from `bus` above (Hub's main
    RPC/cache bus) so this pilot-5 rollout (see
    docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md) cannot
    interfere with Hub's existing bus lifecycle."""
    return HeartbeatOnly(
        ChassisConfig(
            service_name=settings.SERVICE_NAME,
            service_version=settings.SERVICE_VERSION,
            node_name=settings.NODE_NAME,
            bus_url=settings.ORION_BUS_URL,
            bus_enabled=settings.ORION_BUS_ENABLED,
            heartbeat_interval_sec=settings.HEARTBEAT_INTERVAL_SEC,
        )
    )


class PresenceState:
    def __init__(self) -> None:
        self.active_connections = 0
        self.last_seen: Optional[datetime] = None

    def connected(self) -> None:
        self.active_connections += 1
        self.last_seen = datetime.now(timezone.utc)

    def disconnected(self) -> None:
        self.active_connections = max(0, self.active_connections - 1)
        self.last_seen = datetime.now(timezone.utc)

    def heartbeat(self) -> None:
        self.last_seen = datetime.now(timezone.utc)

    def snapshot(self) -> dict:
        return {
            "active": self.active_connections > 0,
            "active_connections": self.active_connections,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
        }


class PresenceContextStore:
    def __init__(self, *, ttl_seconds: int = 14400) -> None:
        self.ttl_seconds = max(60, int(ttl_seconds))
        self._store: dict[str, tuple[datetime, dict]] = {}

    def get(self, session_key: str) -> dict | None:
        item = self._store.get(session_key)
        if not item:
            return None
        expires_at, payload = item
        if datetime.now(timezone.utc) > expires_at:
            self._store.pop(session_key, None)
            return None
        return payload

    def set(self, session_key: str, payload: dict) -> dict:
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=self.ttl_seconds)
        normalized = dict(payload or {})
        normalized.setdefault("submitted_at", datetime.now(timezone.utc).isoformat())
        normalized["expires_at"] = expires_at.isoformat()
        self._store[session_key] = (expires_at, normalized)
        return normalized

    def clear(self, session_key: str) -> None:
        self._store.pop(session_key, None)


class HubStaticFiles(StaticFiles):
    """Static responses are revalidated to avoid stale Hub JS when operators refresh."""

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


# ───────────────────────────────────────────────────────────────
# 🚀 Startup Initialization
# ───────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """
    Initializes all shared services at application startup.
    OrionBus + Clients + UI template.
    """
    global bus, rpc_bus, cortex_client, tts_client, html_content, biometrics_cache, notification_cache, bus_synaptic_trigger_notifier, endogenous_outreach, curiosity_investigation, room_claude_relay, agent_step_relay, harness_step_relay, signals_inspect_cache, cognition_trace_cache, embodiment_outcome_cache, presence_state, presence_context_store, substrate_autonomy_task, substrate_decay_task, substrate_topic_foundry_scheduler_task, affect_ambient_loop_task, heartbeat_chassis

    # ------------------------------------------------------------
    # Bus-native SystemHealthV1 heartbeat (pilot-5 rollout, see
    # docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md).
    # Own bus connection, independent of Hub's main `bus`/`rpc_bus` below.
    # ------------------------------------------------------------
    try:
        heartbeat_chassis = build_heartbeat_chassis()
        await heartbeat_chassis.start_background()
        logger.info(
            "system_health_heartbeat_started service=%s interval_sec=%s",
            settings.SERVICE_NAME,
            settings.HEARTBEAT_INTERVAL_SEC,
        )
    except Exception as exc:
        logger.warning("system_health_heartbeat_start_failed error=%s", exc)
        heartbeat_chassis = None

    # ------------------------------------------------------------
    # Orion Bus Initialization
    # ------------------------------------------------------------
    if settings.ORION_BUS_ENABLED:
        try:
            logger.info(f"Connecting OrionBus → {settings.ORION_BUS_URL}")
            # Use the new OrionBus API (client instead of redis)
            bus = OrionBusAsync(
                settings.ORION_BUS_URL,
                enabled=settings.ORION_BUS_ENABLED,
            )
            await bus.connect()
            logger.info("OrionBusAsync connection established successfully.")

            # 2026-08-25: lets orion.situational.context.py's
            # _build_affect_context read Juniper's latest affect capture
            # (see orion/situational/juniper_affect_state.py) for the
            # situation brief every "orion" mode chat turn builds via
            # orion.hub.turn_orchestrator.run_unified_turn -- same bind
            # pattern services/orion-cortex-exec/app/main.py already uses
            # for session_turn_phase.py's own module-level bus handle.
            from orion.situational.juniper_affect_state import bind_juniper_affect_state_bus

            bind_juniper_affect_state_bus(bus)

            # Outbound RPC uses a forked bus + worker so long-lived Hub subscribers
            # (trace/biometrics caches) cannot steal gateway/TTS/embedding replies.
            from orion.core.bus.rpc_fork import fork_rpc_client

            rpc_bus = await fork_rpc_client(bus)
            cortex_client = CortexGatewayClient(rpc_bus)
            tts_client = TTSClient(rpc_bus)
            logger.info("Bus Clients initialized (Hub RPC on forked bus).")

            # Biometrics cache (singleton)
            biometrics_cache = BiometricsCache(
                enabled=settings.BIOMETRICS_ENABLED,
                stale_after_sec=settings.BIOMETRICS_STALE_AFTER_SEC,
                no_signal_after_sec=settings.BIOMETRICS_NO_SIGNAL_AFTER_SEC,
                role_weights_json=settings.BIOMETRICS_ROLE_WEIGHTS_JSON,
            )
            await biometrics_cache.start(bus)

            # Latest-frame-per-stream cache for the Vision panel's "Carbon
            # (live)" option -- see scripts/vision_frame_cache.py module
            # docstring. Module-level singleton (same pattern as
            # vision_affect_ambient.state), not instantiated-in-main.py like
            # biometrics_cache above, so api_routes.py can read it directly.
            vision_frame_cache_runtime.cache = vision_frame_cache_runtime.VisionFrameCache(
                enabled=settings.VISION_FRAME_CACHE_ENABLED,
                stream_ids={
                    s.strip()
                    for s in settings.VISION_FRAME_CACHE_STREAM_IDS.split(",")
                    if s.strip()
                },
                channel=settings.VISION_FRAME_CHANNEL,
            )
            await vision_frame_cache_runtime.cache.start(bus)

            notification_cache = NotificationCache(
                max_items=settings.NOTIFY_IN_APP_MAX,
                channel=settings.NOTIFY_IN_APP_CHANNEL,
            )
            if settings.NOTIFY_IN_APP_ENABLED:
                await notification_cache.start(bus)

            bus_synaptic_trigger_notifier = BusSynapticTriggerNotifier(
                enabled=True,
                metacog_trigger_channel="orion:equilibrium:metacog:trigger",
                notify_channel=settings.NOTIFY_IN_APP_CHANNEL,
            )
            await bus_synaptic_trigger_notifier.start(bus)

            # Orion speaks first, on a real trigger (2026-08-16 -- see
            # scripts/tension_outreach_trigger.py). Off unless explicitly
            # enabled -- see scripts/endogenous_outreach.py for the safety gates.
            # min_run_length is operator-tunable (settings.py's own comment on
            # HUB_ENDOGENOUS_OUTREACH_MIN_RUN_LENGTH explains why); everything
            # else about the trigger's internals stays fixed.
            endogenous_outreach = EndogenousOutreach(
                enabled=settings.HUB_ENDOGENOUS_OUTREACH_ENABLED,
                tick_interval_sec=settings.HUB_ENDOGENOUS_OUTREACH_TICK_SEC,
                min_cooldown_sec=settings.HUB_ENDOGENOUS_OUTREACH_MIN_COOLDOWN_SEC,
                daily_cap=settings.HUB_ENDOGENOUS_OUTREACH_DAILY_CAP,
                quiet_start_hour=settings.HUB_ENDOGENOUS_OUTREACH_QUIET_START_HOUR,
                quiet_end_hour=settings.HUB_ENDOGENOUS_OUTREACH_QUIET_END_HOUR,
                timeout_sec=settings.HUB_ENDOGENOUS_OUTREACH_TIMEOUT_SEC,
                notify_channel=settings.NOTIFY_IN_APP_CHANNEL,
                fallback_session_id=settings.HUB_ENDOGENOUS_OUTREACH_FALLBACK_SESSION_ID,
                timezone_name=settings.HUB_ENDOGENOUS_OUTREACH_TZ,
                trigger_evaluator=functools.partial(
                    tension_outreach_trigger.current_run,
                    min_run_length=settings.HUB_ENDOGENOUS_OUTREACH_MIN_RUN_LENGTH,
                ),
            )
            # 2026-08-19: generation now goes through the real
            # orion.hub.turn_orchestrator.execute_unified_turn pipeline (see
            # endogenous_outreach.py's own docstring), the same
            # harness-governed path `websocket_handler.py` uses for a real
            # client_mode == "orion" turn -- not the bare CortexGatewayClient
            # call this used before. harness_rpc_bus mirrors that call
            # site's own `harness_rpc_bus=rpc_bus or bus` convention.
            await endogenous_outreach.start(bus, harness_rpc_bus=rpc_bus)

            # Orion notices what Juniper has been talking about, and goes and
            # finds out why (2026-08-26). Same lifecycle and the same real
            # unified-turn pipeline as outreach above -- see
            # scripts/curiosity_investigation.py for why this is a Hub loop and
            # not a service (the harness RPC worker and several module bus
            # binds live in THIS event loop; a standalone process times out).
            curiosity_investigation = CuriosityInvestigation(
                enabled=settings.HUB_CURIOSITY_INVESTIGATION_ENABLED,
                tick_interval_sec=settings.HUB_CURIOSITY_INVESTIGATION_TICK_SEC,
                min_cooldown_sec=settings.HUB_CURIOSITY_INVESTIGATION_MIN_COOLDOWN_SEC,
                daily_cap=settings.HUB_CURIOSITY_INVESTIGATION_DAILY_CAP,
                timeout_sec=settings.HUB_CURIOSITY_INVESTIGATION_TIMEOUT_SEC,
                session_id=settings.HUB_CURIOSITY_INVESTIGATION_SESSION_ID,
                crystallization_sample=settings.HUB_CURIOSITY_INVESTIGATION_CONCEPT_SAMPLE,
                relation_sample=settings.HUB_CURIOSITY_INVESTIGATION_RELATION_SAMPLE,
                timezone_name=settings.HUB_ENDOGENOUS_OUTREACH_TZ,
                # The same asyncpg pool crystallization_routes.py reads. Passed
                # as a callable rather than a value because the pool is created
                # later in startup than this construction.
                pool_provider=lambda: getattr(app.state, "memory_pg_pool", None),
                source_ref=ServiceRef(
                    name=settings.SERVICE_NAME,
                    version=settings.SERVICE_VERSION,
                    node=settings.NODE_NAME,
                ),
                # Orion's own graph. Hub reads it back read-only and asserts
                # the FalkorDB ACL that lets Orion write it -- see
                # orion/curiosity/acl.py for why that assert is required rather
                # than belt-and-braces (aclfile is unset AND immutable here, so
                # the grant does not survive a FalkorDB restart).
                graph_host=settings.HUB_CURIOSITY_GRAPH_HOST,
                graph_port=settings.HUB_CURIOSITY_GRAPH_PORT,
                graph_own=settings.HUB_CURIOSITY_GRAPH_OWN,
                graph_atlas=settings.HUB_CURIOSITY_GRAPH_ATLAS,
                graph_user=settings.HUB_CURIOSITY_GRAPH_ORION_USER,
                graph_password=settings.HUB_CURIOSITY_GRAPH_ORION_PASSWORD,
                hub_url=settings.HUB_CURIOSITY_SANDBOX_HUB_URL,
                prior_sample=settings.HUB_CURIOSITY_PRIOR_SAMPLE,
                stale_prior_tests=settings.HUB_CURIOSITY_STALE_PRIOR_TESTS,
                max_hops=settings.HUB_CURIOSITY_MAX_HOPS,
                pg_readonly_role=settings.HUB_CURIOSITY_PG_READONLY_ROLE,
                # A finding Orion judges worth saying goes through a SECOND
                # turn (its own stance gate) and then through outreach's OWN
                # gates -- quiet hours, daily cap, cooldown are SHARED with
                # tension-triggered outreach, because from Juniper's end they
                # are the same interruption. Read through a callable because
                # `endogenous_outreach` is a module global reassigned above.
                outreach_enabled=settings.HUB_CURIOSITY_OUTREACH_ENABLED,
                outreach_provider=lambda: endogenous_outreach,
                # Liveness only. Without this the loop cannot reach Hub's own
                # soft-ceiling extension and a long turn is killed mid-run --
                # see curiosity_investigation._generate.
                step_relay_provider=lambda: harness_step_relay,
            )
            await curiosity_investigation.start(bus, harness_rpc_bus=rpc_bus)

            # Claude as a third room participant. Hub only publishes the
            # invite and relays the reply -- orion-room-companion owns the
            # credential and the subprocess.
            room_claude_relay = RoomClaudeRelay(
                request_channel=settings.CHANNEL_ROOM_CLAUDE_REQUEST,
                utterance_channel=settings.CHANNEL_ROOM_CLAUDE_UTTERANCE,
                participant_name=settings.HUB_ROOM_CLAUDE_PARTICIPANT_NAME,
                auto_respond=settings.HUB_ROOM_CLAUDE_AUTO_RESPOND,
                auto_min_gap_sec=settings.HUB_ROOM_CLAUDE_AUTO_MIN_GAP_SEC,
                service_name=settings.SERVICE_NAME,
                service_version=settings.SERVICE_VERSION,
                node_name=settings.NODE_NAME,
                enabled=settings.HUB_ROOM_CLAUDE_ENABLED,
            )
            await room_claude_relay.start(bus)

            agent_step_relay = AgentStepRelay(channel=settings.HUB_CONTEXT_EXEC_EVENT_CHANNEL)
            await agent_step_relay.start(bus)

            harness_step_relay = HarnessStepRelay(
                channel=settings.CHANNEL_HARNESS_RUN_STEP,
                last_seen_ttl_sec=settings.HUB_HARNESS_STEP_RELAY_LIVENESS_TTL_SEC,
                last_seen_max_entries=settings.HUB_HARNESS_STEP_RELAY_LIVENESS_MAX_ENTRIES,
            )
            await harness_step_relay.start(bus)

            sic: Optional[SignalsInspectCache] = None
            try:
                sic = SignalsInspectCache(
                    enabled=settings.SIGNALS_INSPECT_ENABLED,
                    subscribe_pattern=settings.SIGNALS_INSPECT_SUBSCRIBE_PATTERN,
                    window_sec=settings.SIGNALS_INSPECT_WINDOW_SEC,
                    trace_enabled=settings.SIGNALS_TRACE_CACHE_ENABLED,
                    trace_max_traces=settings.TRACE_CACHE_MAX_TRACES,
                    trace_ttl_sec=settings.TRACE_CACHE_TTL_SEC,
                    trace_max_signals_per_trace=settings.TRACE_CACHE_MAX_SIGNALS_PER_TRACE,
                )
                await sic.start(bus)
                signals_inspect_cache = sic
            except Exception as exc:
                logger.warning("signals_inspect_cache_start_failed error=%s", exc)
                if sic is not None:
                    try:
                        await sic.stop()
                    except Exception:
                        pass
                signals_inspect_cache = None

            ctc: Optional[CognitionTraceCache] = None
            try:
                ctc = CognitionTraceCache(
                    enabled=settings.COGNITION_TRACE_CACHE_ENABLED,
                    subscribe_channel=settings.COGNITION_TRACE_SUBSCRIBE_CHANNEL,
                    max_entries=settings.COGNITION_TRACE_CACHE_MAX,
                    ttl_sec=settings.COGNITION_TRACE_CACHE_TTL_SEC,
                    api_debug=settings.COGNITION_TRACE_API_DEBUG,
                )
                await ctc.start(bus)
                cognition_trace_cache = ctc
            except Exception as exc:
                logger.warning("cognition_trace_cache_start_failed error=%s", exc)
                if ctc is not None:
                    try:
                        await ctc.stop()
                    except Exception:
                        pass
                cognition_trace_cache = None

            eoc: Optional[EmbodimentOutcomeCache] = None
            try:
                eoc = EmbodimentOutcomeCache(
                    enabled=settings.EMBODIMENT_OUTCOME_TRACE_ENABLED,
                    channel=settings.EMBODIMENT_OUTCOME_CHANNEL,
                    max_entries=settings.EMBODIMENT_OUTCOME_CACHE_MAX,
                )
                await eoc.start(bus)
                embodiment_outcome_cache = eoc
            except Exception as exc:
                logger.warning("embodiment_outcome_cache_start_failed error=%s", exc)
                if eoc is not None:
                    try:
                        await eoc.stop()
                    except Exception:
                        pass
                embodiment_outcome_cache = None

        except Exception as e:
            logger.error(f"Failed to initialize OrionBus: {e}")
            bus = None
            cortex_client = None
            tts_client = None
            signals_inspect_cache = None
            cognition_trace_cache = None
    else:
        logger.warning("OrionBus is DISABLED — Hub will not publish/subscribe.")

    presence_state = PresenceState()
    presence_context_store = PresenceContextStore(ttl_seconds=int(getattr(settings, "ORION_PRESENCE_SESSION_TTL_SECONDS", 14400)))

    if settings.SUBSTRATE_CONCEPT_SEED_ENABLED:
        # Offloaded to a thread: when SUBSTRATE_STORE_BACKEND=sparql (this
        # service's own .env_example default), the underlying upsert_node/
        # upsert_edge calls are synchronous, blocking HTTP requests to Fuseki
        # (see orion/substrate/graphdb_store.py). Running them directly here
        # would block the event loop for the duration of Hub's boot on every
        # restart if Fuseki is slow/unreachable. seed_golden_concepts_at_startup()
        # still never raises either way -- this is purely to keep a slow/degraded
        # graph backend from stalling startup.
        seeded_count = await asyncio.to_thread(api_routes_runtime.seed_golden_concepts_at_startup)
        logger.info("substrate_concept_seed_loaded count=%s", seeded_count)
    else:
        logger.info("substrate_concept_seed_disabled reason=env_disabled")

    if settings.SUBSTRATE_AUTONOMY_ENABLED:
        supported, reason = api_routes_runtime.substrate_autonomy_runtime_supported()
        if not supported:
            logger.warning(
                "substrate_autonomy_scheduler_noop reason=%s store_kind=%s store_degraded=%s",
                reason,
                api_routes_runtime.SUBSTRATE_MUTATION_STORE.source_kind(),
                api_routes_runtime.SUBSTRATE_MUTATION_STORE.degraded(),
            )
        else:
            interval_sec = max(1.0, float(settings.SUBSTRATE_AUTONOMY_INTERVAL_SEC))

            async def _run_substrate_autonomy_scheduler() -> None:
                while True:
                    try:
                        api_routes_runtime.execute_substrate_mutation_scheduled_cycle()
                    except Exception as exc:  # advisory runtime loop; never crash service startup
                        logger.warning("substrate_autonomy_scheduler_error error=%s", exc)
                    await asyncio.sleep(interval_sec)

            substrate_autonomy_task = asyncio.create_task(
                _run_substrate_autonomy_scheduler(),
                name="hub-substrate-autonomy-scheduler",
            )
            logger.info("substrate_autonomy_scheduler_enabled interval_sec=%s", interval_sec)
    else:
        logger.info("substrate_autonomy_scheduler_disabled reason=env_disabled")

    if settings.SUBSTRATE_DECAY_SCHEDULER_ENABLED:
        decay_interval_sec = max(1.0, float(settings.SUBSTRATE_DECAY_SCHEDULER_INTERVAL_SEC))

        async def _run_substrate_decay_scheduler() -> None:
            # Tracks true wall-clock time between ticks (not just decay_interval_sec)
            # so a slow to_thread call or scheduling jitter doesn't desync the decay
            # window from what actually elapsed -- see decay_concept_activations()'s
            # docstring for why passing an explicit, per-tick elapsed_seconds (rather
            # than falling back to its node.temporal.observed_at-based one-shot mode)
            # is required for a function called repeatedly on a loop.
            last_tick_monotonic = time.monotonic()
            while True:
                await asyncio.sleep(decay_interval_sec)
                now_monotonic = time.monotonic()
                tick_elapsed_sec = now_monotonic - last_tick_monotonic
                last_tick_monotonic = now_monotonic
                try:
                    summary = await asyncio.to_thread(
                        api_routes_runtime.decay_concept_activations,
                        elapsed_seconds=tick_elapsed_sec,
                    )
                    logger.info(
                        "substrate_decay_scheduler_tick decayed=%s skipped=%s errors=%s total_concepts=%s elapsed_sec=%.1f",
                        summary.get("decayed"),
                        summary.get("skipped"),
                        summary.get("errors"),
                        summary.get("total_concepts"),
                        tick_elapsed_sec,
                    )
                except Exception as exc:  # advisory runtime loop; never crash service startup
                    logger.warning("substrate_decay_scheduler_error error=%s", exc)

        substrate_decay_task = asyncio.create_task(
            _run_substrate_decay_scheduler(),
            name="hub-substrate-decay-scheduler",
        )
        logger.info("substrate_decay_scheduler_enabled interval_sec=%s", decay_interval_sec)
    else:
        logger.info("substrate_decay_scheduler_disabled reason=env_disabled")

    if settings.SUBSTRATE_TOPIC_FOUNDRY_SCHEDULER_ENABLED:
        topic_foundry_interval_sec = max(
            1.0, float(settings.SUBSTRATE_TOPIC_FOUNDRY_SCHEDULER_INTERVAL_SEC)
        )

        async def _run_substrate_topic_foundry_scheduler() -> None:
            # Three independent steps per tick, not a blocking call chain:
            # (1) trigger a training run for the current rolling window
            # (async on topic-foundry's own side -- this returns as soon as
            # the run is queued, or immediately if spec_hash dedup finds an
            # identical run already exists), (2) trigger enrichment for
            # whatever the latest COMPLETED run currently is (added
            # 2026-07-28, own SUBSTRATE_TOPIC_FOUNDRY_ENRICH_ENABLE gate --
            # also async on topic-foundry's side), (3) ingest whatever the
            # latest COMPLETED run currently is. Steps 2/3 may act on a run
            # from a PREVIOUS tick (training/enrichment both take real time),
            # or find nothing new -- both are expected, not errors. See
            # trigger_topic_foundry_training_run()'s and
            # trigger_topic_foundry_enrichment()'s docstrings.
            while True:
                await asyncio.sleep(topic_foundry_interval_sec)
                try:
                    trigger_summary = await asyncio.to_thread(
                        concept_atlas_routes_runtime.trigger_topic_foundry_training_run
                    )
                    logger.info(
                        "substrate_topic_foundry_scheduler_trigger_tick triggered=%s run_id=%s status=%s reason=%s",
                        trigger_summary.get("triggered"),
                        trigger_summary.get("run_id"),
                        trigger_summary.get("status"),
                        trigger_summary.get("reason"),
                    )
                except Exception as exc:  # advisory runtime loop; never crash service startup
                    logger.warning("substrate_topic_foundry_scheduler_trigger_error error=%s", exc)

                try:
                    enrich_summary = await asyncio.to_thread(
                        concept_atlas_routes_runtime.trigger_topic_foundry_enrichment
                    )
                    logger.info(
                        "substrate_topic_foundry_scheduler_enrich_tick triggered=%s run_id=%s status=%s reason=%s "
                        "enriched_count=%s failed_count=%s",
                        enrich_summary.get("triggered"),
                        enrich_summary.get("run_id"),
                        enrich_summary.get("status"),
                        enrich_summary.get("reason"),
                        enrich_summary.get("enriched_count"),
                        enrich_summary.get("failed_count"),
                    )
                except Exception as exc:  # advisory runtime loop; never crash service startup
                    logger.warning("substrate_topic_foundry_scheduler_enrich_error error=%s", exc)

                try:
                    ingest_summary = await asyncio.to_thread(
                        concept_atlas_routes_runtime.concept_atlas_ingest_topic_foundry
                    )
                    logger.info(
                        "substrate_topic_foundry_scheduler_ingest_tick available=%s run_id=%s concepts_written=%s entities_written=%s edges_written=%s typed_edges_written=%s",
                        ingest_summary.get("available"),
                        ingest_summary.get("run_id"),
                        ingest_summary.get("concepts_written"),
                        ingest_summary.get("entities_written"),
                        ingest_summary.get("edges_written"),
                        ingest_summary.get("typed_edges_written"),
                    )
                except Exception as exc:  # advisory runtime loop; never crash service startup
                    logger.warning("substrate_topic_foundry_scheduler_ingest_error error=%s", exc)

                # AI Town's own concept graph (docs/superpowers/specs/2026-08-18-
                # aitown-concept-graph-split-and-atlas-readability-design.md,
                # "AI Town's own concept graph") -- same three-step shape as
                # Orion's above, riding on the same tick interval (independent
                # CADENCE was raised as a real open question by that spec's own
                # "Missing questions" and deliberately deferred -- no real
                # AI-Town cluster-quality data yet to tune a second interval
                # against). Independent ENABLE, unlike cadence, is a real,
                # already-justified need (an operator kill switch that doesn't
                # also disable Orion's own production pipeline) -- review
                # finding 2026-08-20, not deferred. Writes into a different
                # FalkorDB graph (FALKORDB_AITOWN_SUBSTRATE_GRAPH) and a
                # different topic-foundry dataset/model
                # (source_table=aitown_chat_history_log) -- interpretability-
                # only, never feeds Orion's own cognition.
                if settings.SUBSTRATE_TOPIC_FOUNDRY_AITOWN_SCHEDULER_ENABLED:
                    try:
                        aitown_trigger_summary = await asyncio.to_thread(
                            concept_atlas_routes_runtime.trigger_topic_foundry_aitown_training_run
                        )
                        logger.info(
                            "substrate_topic_foundry_aitown_scheduler_trigger_tick triggered=%s run_id=%s status=%s reason=%s",
                            aitown_trigger_summary.get("triggered"),
                            aitown_trigger_summary.get("run_id"),
                            aitown_trigger_summary.get("status"),
                            aitown_trigger_summary.get("reason"),
                        )
                    except Exception as exc:  # advisory runtime loop; never crash service startup
                        logger.warning("substrate_topic_foundry_aitown_scheduler_trigger_error error=%s", exc)

                    try:
                        aitown_enrich_summary = await asyncio.to_thread(
                            concept_atlas_routes_runtime.trigger_topic_foundry_aitown_enrichment
                        )
                        logger.info(
                            "substrate_topic_foundry_aitown_scheduler_enrich_tick triggered=%s run_id=%s status=%s reason=%s "
                            "enriched_count=%s failed_count=%s",
                            aitown_enrich_summary.get("triggered"),
                            aitown_enrich_summary.get("run_id"),
                            aitown_enrich_summary.get("status"),
                            aitown_enrich_summary.get("reason"),
                            aitown_enrich_summary.get("enriched_count"),
                            aitown_enrich_summary.get("failed_count"),
                        )
                    except Exception as exc:  # advisory runtime loop; never crash service startup
                        logger.warning("substrate_topic_foundry_aitown_scheduler_enrich_error error=%s", exc)

                    try:
                        aitown_ingest_summary = await asyncio.to_thread(
                            concept_atlas_routes_runtime.concept_atlas_ingest_topic_foundry_aitown
                        )
                        logger.info(
                            "substrate_topic_foundry_aitown_scheduler_ingest_tick available=%s run_id=%s concepts_written=%s entities_written=%s edges_written=%s typed_edges_written=%s",
                            aitown_ingest_summary.get("available"),
                            aitown_ingest_summary.get("run_id"),
                            aitown_ingest_summary.get("concepts_written"),
                            aitown_ingest_summary.get("entities_written"),
                            aitown_ingest_summary.get("edges_written"),
                            aitown_ingest_summary.get("typed_edges_written"),
                        )
                    except Exception as exc:  # advisory runtime loop; never crash service startup
                        logger.warning("substrate_topic_foundry_aitown_scheduler_ingest_error error=%s", exc)

        substrate_topic_foundry_scheduler_task = asyncio.create_task(
            _run_substrate_topic_foundry_scheduler(),
            name="hub-substrate-topic-foundry-scheduler",
        )
        logger.info(
            "substrate_topic_foundry_scheduler_enabled interval_sec=%s window_days=%s",
            topic_foundry_interval_sec,
            settings.SUBSTRATE_TOPIC_FOUNDRY_WINDOW_DAYS,
        )
    else:
        logger.info("substrate_topic_foundry_scheduler_disabled reason=env_disabled")

    # Ambient (recurring) AffectGPT capture toggle -- see
    # scripts/vision_affect_ambient.py module docstring for the full design
    # (2026-08-22 correction: Hub owns this loop, not the orchestrator).
    # AFFECT_AMBIENT_ENABLED gates whether the loop TASK is CREATED at
    # startup, same spirit as SUBSTRATE_TOPIC_FOUNDRY_SCHEDULER_ENABLED
    # above -- it is a boot-time switch, NOT a live kill switch (review
    # finding, 2026-08-22: an earlier comment here called it one; flipping
    # this env var to false and NOT restarting Hub does nothing to a loop
    # already running -- only the runtime toggle, state.enabled via
    # POST /api/vision/affect-ambient, stops live capture immediately).
    # Also requires JUNIPER_AFFECTIVE_STATE_BASE_URL to be configured -- no
    # point running a loop with nowhere to call.
    if settings.AFFECT_AMBIENT_ENABLED and settings.JUNIPER_AFFECTIVE_STATE_BASE_URL:
        affect_ambient_loop_task = asyncio.create_task(
            vision_affect_ambient_runtime.affect_ambient_loop(
                base_url=settings.JUNIPER_AFFECTIVE_STATE_BASE_URL,
                interval_sec=max(1.0, float(settings.AFFECT_AMBIENT_INTERVAL_SEC)),
                timeout_sec=float(settings.JUNIPER_AFFECTIVE_STATE_TIMEOUT_SEC),
                poll_sec=max(1.0, float(settings.AFFECT_AMBIENT_POLL_SEC)),
            ),
            name="hub-affect-ambient-loop",
        )
        logger.info(
            "affect_ambient_loop_task_started interval_sec=%s poll_sec=%s enabled_at_boot=False",
            settings.AFFECT_AMBIENT_INTERVAL_SEC,
            settings.AFFECT_AMBIENT_POLL_SEC,
        )
    else:
        logger.info(
            "affect_ambient_loop_disabled reason=%s",
            "env_disabled" if not settings.AFFECT_AMBIENT_ENABLED else "no_base_url_configured",
        )

    # ------------------------------------------------------------
    # Validate UI HTML Template (served fresh from disk on each GET /)
    # ------------------------------------------------------------
    try:
        html_content = render_hub_index_html(memory_pool_ok=False)
        logger.info(
            "UI template validated (ui_asset_version=%s).",
            build_hub_ui_asset_version(),
        )
    except Exception as exc:
        logger.error("CRITICAL: failed to render index.html: %s", exc)
        html_content = "<html><body><h1>UI template missing</h1></body></html>"

    dsn = str(getattr(settings, "RECALL_PG_DSN", "") or "").strip()
    if dsn:
        try:
            import asyncpg  # type: ignore
        except Exception:
            asyncpg = None
        if asyncpg is not None:
            try:
                app.state.memory_pg_pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=6)
                logger.info("memory_pg_pool_ready dsn_configured=true")
                try:
                    apply_memory_cards_schema(dsn)
                    logger.info("memory_cards_schema_applied ok=true")
                    try:
                        apply_memory_crystallizations_schema(dsn)
                        logger.info("memory_crystallizations_schema_applied ok=true")
                    except Exception as crys_exc:
                        logger.error("memory_crystallizations_schema_apply_failed error=%s", crys_exc, exc_info=True)
                except Exception as schema_exc:
                    logger.error("memory_cards_schema_apply_failed error=%s", schema_exc, exc_info=True)
            except Exception as exc:
                logger.error("memory_pg_pool_failed error=%s", exc)
                app.state.memory_pg_pool = None
        else:
            app.state.memory_pg_pool = None
            logger.warning("memory_pg_pool_skipped reason=asyncpg_import_failed")
    else:
        app.state.memory_pg_pool = None
        logger.info("memory_pg_pool_skipped reason=RECALL_PG_DSN_unset")

    pool_ok = getattr(app.state, "memory_pg_pool", None) is not None
    dsn_configured = bool(dsn)
    html_content = render_hub_index_html(memory_pool_ok=pool_ok)
    if pool_ok:
        logger.info("memory_store_banner=connected")
    elif not dsn_configured:
        logger.info("memory_store_banner=dsn_unset")
    else:
        logger.info("memory_store_banner=pool_unavailable")

    logger.info("Startup complete — Hub is ready.")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    global bus, rpc_bus, biometrics_cache, notification_cache, bus_synaptic_trigger_notifier, endogenous_outreach, curiosity_investigation, room_claude_relay, agent_step_relay, harness_step_relay, signals_inspect_cache, cognition_trace_cache, embodiment_outcome_cache, substrate_autonomy_task, substrate_decay_task, substrate_topic_foundry_scheduler_task, affect_ambient_loop_task, heartbeat_chassis
    if heartbeat_chassis is not None:
        try:
            await heartbeat_chassis.stop()
        except Exception as exc:
            logger.warning("system_health_heartbeat_stop_error error=%s", exc)
        heartbeat_chassis = None
    pool = getattr(app.state, "memory_pg_pool", None)
    if pool is not None:
        try:
            await pool.close()
            logger.info("memory_pg_pool_closed")
        except Exception as exc:
            logger.warning("memory_pg_pool_close_error error=%s", exc)
        app.state.memory_pg_pool = None
    if substrate_autonomy_task is not None:
        substrate_autonomy_task.cancel()
        try:
            await substrate_autonomy_task
        except asyncio.CancelledError:
            pass
        substrate_autonomy_task = None
    if substrate_decay_task is not None:
        substrate_decay_task.cancel()
        try:
            await substrate_decay_task
        except asyncio.CancelledError:
            pass
        substrate_decay_task = None
    if substrate_topic_foundry_scheduler_task is not None:
        substrate_topic_foundry_scheduler_task.cancel()
        try:
            await substrate_topic_foundry_scheduler_task
        except asyncio.CancelledError:
            pass
        substrate_topic_foundry_scheduler_task = None
    if affect_ambient_loop_task is not None:
        affect_ambient_loop_task.cancel()
        try:
            await affect_ambient_loop_task
        except asyncio.CancelledError:
            pass
        affect_ambient_loop_task = None
    if biometrics_cache is not None:
        await biometrics_cache.stop()
    if vision_frame_cache_runtime.cache is not None:
        await vision_frame_cache_runtime.cache.stop()
    if notification_cache is not None:
        await notification_cache.stop()
    if bus_synaptic_trigger_notifier is not None:
        try:
            await bus_synaptic_trigger_notifier.stop()
        except Exception:
            pass
    if curiosity_investigation is not None:
        try:
            await curiosity_investigation.stop()
        except Exception:  # noqa: BLE001
            logger.warning("curiosity_investigation_stop_failed", exc_info=True)
        curiosity_investigation = None
    if endogenous_outreach is not None:
        try:
            await endogenous_outreach.stop()
        except Exception:
            pass
        endogenous_outreach = None
    if room_claude_relay is not None:
        try:
            await room_claude_relay.stop()
        except Exception:
            pass
        room_claude_relay = None
    if agent_step_relay is not None:
        try:
            await agent_step_relay.stop()
        except Exception:
            pass
    if harness_step_relay is not None:
        try:
            await harness_step_relay.stop()
        except Exception:
            pass
    if signals_inspect_cache is not None:
        await signals_inspect_cache.stop()
    if cognition_trace_cache is not None:
        await cognition_trace_cache.stop()
    if embodiment_outcome_cache is not None:
        try:
            await embodiment_outcome_cache.stop()
        except Exception:
            pass
    if rpc_bus is not None:
        try:
            await rpc_bus.close()
            logger.info("Hub RPC OrionBusAsync fork closed.")
        except Exception as e:
            logger.warning("Error while closing Hub RPC bus fork: %s", e)
        rpc_bus = None
    if bus is not None:
        try:
            await bus.close()
            logger.info("OrionBusAsync closed.")
        except Exception as e:
            logger.warning("Error while closing OrionBusAsync: %s", e)


# ───────────────────────────────────────────────────────────────
# 🔗 API Routes + WebSockets + Static Files
# ───────────────────────────────────────────────────────────────

app.include_router(api_router)
app.include_router(memory_router)
app.include_router(crystallization_router)
app.include_router(mind_router)
app.include_router(memory_graph_router)
app.include_router(memory_consolidation_draft_router)
app.include_router(proposal_review_router)
app.include_router(concept_atlas_router)
app.include_router(self_brain_router)
app.include_router(chat_attachments_router)

# Real-time WS endpoint (also /hub/ws for path-prefixed reverse proxies where the browser path includes /hub)
app.add_websocket_route("/ws", websocket_endpoint)
app.add_websocket_route("/hub/ws", websocket_endpoint)
app.add_websocket_route("/ws/service-logs", service_logs_websocket_endpoint)

# Static files for JS/CSS
app.mount("/static", HubStaticFiles(directory=str(STATIC_DIR)), name="static")

logger.info("Routes, WebSocket endpoint, and static mounts ready.")
