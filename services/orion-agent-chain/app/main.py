# services/orion-agent-chain/app/main.py

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .settings import settings
from .api import router as api_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger("agent-chain.main")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.service_name,
        version=settings.service_version,
    )

    # CORS — allow all for internal/hub usage
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount the Agent Chain API
    app.include_router(api_router, prefix="/chain", tags=["agent-chain"])

    return app


app = create_app()
