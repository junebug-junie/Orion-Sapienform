# scripts/main.py

import json
import logging
import os
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from scripts.settings import settings
from scripts.api_routes import router as api_router
from scripts.websocket_handler import websocket_endpoint
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


def build_hub_ui_asset_version() -> str:
    """Build an explicit cache-busting token for Hub static assets."""
    explicit = os.getenv("HUB_UI_BUILD")
    build_id = os.getenv("BUILD_ID")
    git_sha = os.getenv("GIT_SHA") or os.getenv("SOURCE_COMMIT")
    discovered_git_sha = _discover_git_sha()
    build_ts = os.getenv("BUILD_TIMESTAMP")
    service_version = settings.SERVICE_VERSION

    for candidate in (explicit, build_id, git_sha, discovered_git_sha, build_ts, service_version):
        value = str(candidate or "").strip()
        if value:
            return value
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
    global bus, cortex_client, tts_client, html_content, biometrics_cache, notification_cache, presence_state

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
    global bus, biometrics_cache, notification_cache
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

# Static files for JS/CSS
app.mount("/static", HubStaticFiles(directory=str(STATIC_DIR)), name="static")

logger.info("Routes, WebSocket endpoint, and static mounts ready.")
