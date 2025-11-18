# scripts/main.py

import logging
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from scripts.settings import settings
from scripts.api_routes import router as api_router
from scripts.websocket_handler import websocket_endpoint
from scripts.asr import ASR

from orion.core.bus.service import OrionBus


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
asr: ASR | None = None
bus: OrionBus | None = None
html_content: str = "<html><body><h1>Error loading UI</h1></body></html>"


# ───────────────────────────────────────────────────────────────
# 🚀 Startup Initialization
# ───────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """
    Initializes all shared services at application startup.
    ASR + OrionBus + UI template.
    """
    global asr, bus, html_content

    # ------------------------------------------------------------
    # ASR Initialization
    # ------------------------------------------------------------
    logger.info(
        f"Loading Whisper model '{settings.WHISPER_MODEL_SIZE}' "
        f"on {settings.WHISPER_DEVICE}/{settings.WHISPER_COMPUTE_TYPE}"
    )

    # ASR(size, device, compute_type)
    asr = ASR(
        settings.WHISPER_MODEL_SIZE,
        settings.WHISPER_DEVICE,
        settings.WHISPER_COMPUTE_TYPE,
    )

    # ------------------------------------------------------------
    # Orion Bus Initialization
    # ------------------------------------------------------------
    if settings.ORION_BUS_ENABLED:
        try:
            logger.info(f"Connecting OrionBus → {settings.ORION_BUS_URL}")
            bus = OrionBus(url=settings.ORION_BUS_URL)
            # OrionBus itself will log its connection status.
            logger.info("OrionBus initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize OrionBus: {e}", exc_info=True)
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


# ───────────────────────────────────────────────────────────────
# 🔗 API Routes + WebSockets + Static Files
# ───────────────────────────────────────────────────────────────

app.include_router(api_router)

# Real-time WS endpoint
app.add_websocket_route("/ws", websocket_endpoint)

# Static files for JS/CSS
app.mount("/static", StaticFiles(directory="static"), name="static")

logger.info("Routes, WebSocket endpoint, and static mounts ready.")
