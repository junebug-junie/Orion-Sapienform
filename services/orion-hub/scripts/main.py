# scripts/main.py

import html
import json
import logging
import os
import subprocess
import asyncio
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from scripts.settings import settings
from scripts.api_routes import router as api_router
from orion.core.storage.memory_cards import apply_memory_cards_schema
from scripts.memory_routes import router as memory_router
from scripts.mind_routes import router as mind_router
from scripts.memory_graph_routes import router as memory_graph_router
import scripts.api_routes as api_routes_runtime
from scripts.websocket_handler import websocket_endpoint
from scripts.service_logs_ws import service_logs_websocket_endpoint
from scripts.biometrics_cache import BiometricsCache
from scripts.notification_cache import NotificationCache
from scripts.signals_inspect_cache import SignalsInspectCache

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


# ───────────────────────────────────────────────────────────────
# 🌐 FastAPI App & Shared Service Handles
# ───────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.SERVICE_NAME,
    version=settings.SERVICE_VERSION,
)

# These are populated on startup and imported by other modules:
bus: Optional[OrionBusAsync] = None
cortex_client: Optional[CortexGatewayClient] = None
tts_client: Optional[TTSClient] = None
html_content: str = "<html><body><h1>Error loading UI</h1></body></html>"
biometrics_cache: Optional[BiometricsCache] = None
notification_cache: Optional[NotificationCache] = None
signals_inspect_cache: Optional[SignalsInspectCache] = None
presence_state: Optional["PresenceState"] = None
presence_context_store: Optional["PresenceContextStore"] = None
substrate_autonomy_task: Optional[asyncio.Task] = None


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
    global bus, cortex_client, tts_client, html_content, biometrics_cache, notification_cache, signals_inspect_cache, presence_state, presence_context_store, substrate_autonomy_task

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

            # Initialize RPC Clients
            cortex_client = CortexGatewayClient(bus)
            tts_client = TTSClient(bus)
            logger.info("Bus Clients initialized.")

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

        except Exception as e:
            logger.error(f"Failed to initialize OrionBus: {e}")
            bus = None
            cortex_client = None
            tts_client = None
            signals_inspect_cache = None
    else:
        logger.warning("OrionBus is DISABLED — Hub will not publish/subscribe.")

    presence_state = PresenceState()
    presence_context_store = PresenceContextStore(ttl_seconds=int(getattr(settings, "ORION_PRESENCE_SESSION_TTL_SECONDS", 14400)))

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


    # ------------------------------------------------------------
    # Load UI HTML Template
    # ------------------------------------------------------------
    try:
        with open(TEMPLATES_DIR / "index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        ui_asset_version = build_hub_ui_asset_version()
        html_content = html_content.replace(
            "{{NOTIFY_TOAST_SECONDS}}",
            str(settings.NOTIFY_TOAST_SECONDS),
        )
        html_content = html_content.replace(
            "{{HUB_UI_ASSET_VERSION}}",
            ui_asset_version,
        )
        hub_cfg = {
            "apiBaseOverride": settings.HUB_API_BASE_OVERRIDE or "",
            "wsBaseOverride": settings.HUB_WS_BASE_OVERRIDE or "",
            "autoDefaultEnabled": bool(settings.HUB_AUTO_DEFAULT_ENABLED),
            "worldPulseFixtureRunEnabled": bool(settings.WORLD_PULSE_UI_FIXTURE_RUN_ENABLED),
        }
        html_content = html_content.replace(
            "{{HUB_CFG}}",
            json.dumps(hub_cfg),
        )
        logger.info("UI template loaded successfully (ui_asset_version=%s).", ui_asset_version)
    except FileNotFoundError:
        logger.error("CRITICAL: 'templates/index.html' not found.")
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
    if pool_ok:
        banner_class = "border-emerald-800/60 bg-emerald-950/30 text-emerald-100"
        banner_text = "Memory store connected. Operator curation and /api/memory/* are available."
    else:
        banner_class = "border-amber-800/60 bg-amber-950/40 text-amber-100"
        if not dsn_configured:
            banner_text = (
                "Memory store unavailable: RECALL_PG_DSN is not set. "
                "Set it to your conjourney Postgres URL (same DB recall uses), then restart Hub."
            )
        else:
            banner_text = (
                "Memory store unavailable: RECALL_PG_DSN is set but Postgres did not open a pool. "
                "Check Hub logs, credentials, and database reachability."
            )
    if "{{HUB_MEMORY_STORE_READY}}" in html_content:
        html_content = html_content.replace("{{HUB_MEMORY_STORE_READY}}", "true" if pool_ok else "false")
        html_content = html_content.replace("{{HUB_MEMORY_STORE_BANNER_CLASS}}", banner_class)
        html_content = html_content.replace("{{HUB_MEMORY_STORE_BANNER_TEXT}}", html.escape(banner_text))

    logger.info("Startup complete — Hub is ready.")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    global bus, biometrics_cache, notification_cache, signals_inspect_cache, substrate_autonomy_task
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
    if biometrics_cache is not None:
        await biometrics_cache.stop()
    if notification_cache is not None:
        await notification_cache.stop()
    if signals_inspect_cache is not None:
        await signals_inspect_cache.stop()
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
app.include_router(mind_router)
app.include_router(memory_graph_router)

# Real-time WS endpoint
app.add_websocket_route("/ws", websocket_endpoint)
app.add_websocket_route("/ws/service-logs", service_logs_websocket_endpoint)

# Static files for JS/CSS
app.mount("/static", HubStaticFiles(directory=str(STATIC_DIR)), name="static")

logger.info("Routes, WebSocket endpoint, and static mounts ready.")
