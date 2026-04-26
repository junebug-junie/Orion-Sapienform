# scripts/main.py

import json
import logging
import os
import subprocess
import asyncio
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from scripts.settings import settings
from scripts.api_routes import router as api_router
import scripts.api_routes as api_routes_runtime
from scripts.websocket_handler import websocket_endpoint
from scripts.service_logs_ws import service_logs_websocket_endpoint
from scripts.biometrics_cache import BiometricsCache
from scripts.notification_cache import NotificationCache

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

    # Honor explicit CI/build ids exactly.
    for candidate in (explicit, build_id, git_sha, build_ts):
        value = str(candidate or "").strip()
        if value:
            return value
    # For local/dev paths, include mtime token so restarts pick up UI edits.
    mtime_token = _ui_asset_mtime_token()
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
presence_state: Optional["PresenceState"] = None
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
    global bus, cortex_client, tts_client, html_content, biometrics_cache, notification_cache, presence_state, substrate_autonomy_task

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

        except Exception as e:
            logger.error(f"Failed to initialize OrionBus: {e}")
            bus = None
            cortex_client = None
            tts_client = None
    else:
        logger.warning("OrionBus is DISABLED — Hub will not publish/subscribe.")

    presence_state = PresenceState()

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
        }
        html_content = html_content.replace(
            "{{HUB_CFG}}",
            json.dumps(hub_cfg),
        )
        logger.info("UI template loaded successfully (ui_asset_version=%s).", ui_asset_version)
    except FileNotFoundError:
        logger.error("CRITICAL: 'templates/index.html' not found.")
        html_content = "<html><body><h1>UI template missing</h1></body></html>"

    logger.info("Startup complete — Hub is ready.")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    global bus, biometrics_cache, notification_cache, substrate_autonomy_task
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

# Real-time WS endpoint
app.add_websocket_route("/ws", websocket_endpoint)
app.add_websocket_route("/ws/service-logs", service_logs_websocket_endpoint)

# Static files for JS/CSS
app.mount("/static", HubStaticFiles(directory=str(STATIC_DIR)), name="static")

logger.info("Routes, WebSocket endpoint, and static mounts ready.")
