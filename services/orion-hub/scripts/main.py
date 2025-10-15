import logging
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# Refactored to import handlers and settings
from scripts.settings import settings
from scripts.api_routes import router as api_router
from scripts.websocket_handler import websocket_endpoint
from scripts.asr import ASR
from scripts.tts import TTS
from orion.core.bus.service import OrionBus

# ───────────────────────────────────────────────────────────────
# 🪵 Logging
# ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("voice-app")

# ───────────────────────────────────────────────────────────────
# 🚀 FastAPI & Global Objects
# ───────────────────────────────────────────────────────────────
app = FastAPI(title=settings.SERVICE_NAME, version=settings.SERVICE_VERSION)

# These global objects are shared across the different modules
asr: ASR | None = None
tts: TTS | None = None
bus: OrionBus | None = None
html_content: str = "<html><body><h1>Error: templates/index.html not found</h1></body></html>"

@app.on_event("startup")
async def startup_event():
    """
    Initializes all shared services when the application starts.
    """
    global asr, tts, bus, html_content
    logger.info(f"Loading Whisper model '{settings.WHISPER_MODEL_SIZE}' on {settings.WHISPER_DEVICE}/{settings.WHISPER_COMPUTE_TYPE}")
    asr = ASR(settings.WHISPER_MODEL_SIZE, settings.WHISPER_DEVICE, settings.WHISPER_COMPUTE_TYPE)
    tts = TTS()

    if settings.ORION_BUS_ENABLED:
        logger.info(f"Initializing OrionBus connection to {settings.ORION_BUS_URL}")
        bus = OrionBus(url=settings.ORION_BUS_URL)
    else:
        logger.warning("OrionBus is disabled. No messages will be published.")
    
    # Load HTML content at startup
    try:
        with open("templates/index.html", "r") as f:
            html_content = f.read()
    except FileNotFoundError:
        logger.error("CRITICAL: Could not read 'templates/index.html'.")

    logger.info("Startup complete.")

# ───────────────────────────────────────────────────────────────
# 🔗 Mount Routers and Static Files
# ───────────────────────────────────────────────────────────────

# Include all the HTTP routes from the api_routes.py file
app.include_router(api_router)

# Add the WebSocket endpoint
app.add_websocket_route("/ws", websocket_endpoint)

# Mount the static directory for CSS, JS, etc.
app.mount("/static", StaticFiles(directory="static"), name="static")

logger.info("Application routes and WebSocket endpoint have been mounted.")

