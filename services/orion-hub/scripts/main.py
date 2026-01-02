# scripts/main.py

import logging
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from scripts.settings import settings
from scripts.api_routes import router as api_router
from scripts.websocket_handler import websocket_endpoint

from orion.core.bus.async_service import OrionBusAsync


# ───────────────────────────────────────────────────────────────
# 🪵 Logging Setup
# ───────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("orion-hub")


# ───────────────────────────────────────────────────────────────
# 🌐 FastAPI App & Shared Service Handles
# ───────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.SERVICE_NAME,
    version=settings.SERVICE_VERSION,
)

# These are populated on startup and imported by other modules:
bus: OrionBusAsync | None = None
html_content: str = "<html><body><h1>Error loading UI</h1></body></html>"


# ───────────────────────────────────────────────────────────────
# 🚀 Startup Initialization
# ───────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """
    Initializes all shared services at application startup.
    OrionBus + UI template.
    """
    global bus, html_content

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

        except Exception as e:
            logger.error(f"Failed to initialize OrionBus: {e}")
            bus = None
    else:
        logger.warning("OrionBus is DISABLED — Hub will not publish/subscribe.")


    # ------------------------------------------------------------
    # Load UI HTML Template
    # ------------------------------------------------------------
    try:
        with open("templates/index.html", "r") as f:
            html_content = f.read()
        logger.info("UI template loaded successfully.")
    except FileNotFoundError:
        logger.error("CRITICAL: 'templates/index.html' not found.")
        html_content = "<html><body><h1>UI template missing</h1></body></html>"

    logger.info("Startup complete — Hub is ready.")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    global bus
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
app.mount("/static", StaticFiles(directory="static"), name="static")

logger.info("Routes, WebSocket endpoint, and static mounts ready.")
