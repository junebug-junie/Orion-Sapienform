# scripts/main.py

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
import scripts.api_routes as api_routes_runtime
from scripts.websocket_handler import websocket_endpoint
from scripts.service_logs_ws import service_logs_websocket_endpoint
from scripts.biometrics_cache import BiometricsCache
from scripts.notification_cache import NotificationCache
from scripts.agent_step_relay import AgentStepRelay
from scripts.harness_step_relay import HarnessStepRelay
from scripts.signals_inspect_cache import SignalsInspectCache
from scripts.cognition_trace_cache import CognitionTraceCache
from scripts.embodiment_outcome_cache import EmbodimentOutcomeCache

from orion.core.bus.async_service import OrionBusAsync
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
    """Best-effort mtime token so uncommitted UI edits bust browser caches."""
    candidates = [
        STATIC_DIR / "js" / "app.js",
        STATIC_DIR / "js" / "memory-graph-draft-ui.js",
        STATIC_DIR / "js" / "organ-signals-graph-ui.js",
        STATIC_DIR / "js" / "workflow-ui.js",
        TEMPLATES_DIR / "index.html",
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

    if bool(getattr(settings, "HUB_AGENT_CLAUDE_ENABLED", False)):
        agent_claude_mode_options = (
            '<option value="agent_claude_opus">Agent Claude - Opus</option>'
            '<option value="agent_claude_sonnet">Agent Claude - Sonnet</option>'
            '<option value="agent_claude_haiku">Agent Claude - Haiku</option>'
        )
    else:
        agent_claude_mode_options = ""
    rendered = rendered.replace("{{HUB_AGENT_CLAUDE_MODE_OPTIONS}}", agent_claude_mode_options)

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
agent_step_relay: Optional[AgentStepRelay] = None
harness_step_relay: Optional[HarnessStepRelay] = None
signals_inspect_cache: Optional[SignalsInspectCache] = None
cognition_trace_cache: Optional[CognitionTraceCache] = None
embodiment_outcome_cache: Optional[EmbodimentOutcomeCache] = None
presence_state: Optional["PresenceState"] = None
presence_context_store: Optional["PresenceContextStore"] = None
substrate_autonomy_task: Optional[asyncio.Task] = None
substrate_decay_task: Optional[asyncio.Task] = None


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
    global bus, rpc_bus, cortex_client, tts_client, html_content, biometrics_cache, notification_cache, agent_step_relay, harness_step_relay, signals_inspect_cache, cognition_trace_cache, embodiment_outcome_cache, presence_state, presence_context_store, substrate_autonomy_task, substrate_decay_task

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

            notification_cache = NotificationCache(
                max_items=settings.NOTIFY_IN_APP_MAX,
                channel=settings.NOTIFY_IN_APP_CHANNEL,
            )
            if settings.NOTIFY_IN_APP_ENABLED:
                await notification_cache.start(bus)

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
    global bus, rpc_bus, biometrics_cache, notification_cache, agent_step_relay, harness_step_relay, signals_inspect_cache, cognition_trace_cache, embodiment_outcome_cache, substrate_autonomy_task, substrate_decay_task
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
    if biometrics_cache is not None:
        await biometrics_cache.stop()
    if notification_cache is not None:
        await notification_cache.stop()
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

# Real-time WS endpoint (also /hub/ws for path-prefixed reverse proxies where the browser path includes /hub)
app.add_websocket_route("/ws", websocket_endpoint)
app.add_websocket_route("/hub/ws", websocket_endpoint)
app.add_websocket_route("/ws/service-logs", service_logs_websocket_endpoint)

# Static files for JS/CSS
app.mount("/static", HubStaticFiles(directory=str(STATIC_DIR)), name="static")

logger.info("Routes, WebSocket endpoint, and static mounts ready.")
