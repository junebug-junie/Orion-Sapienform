"""orion-diffusion-host — Patch 1 skeleton.

Only a `/health` endpoint exists. No model is loaded, no bus consumer runs,
no GPU is touched. See README.md "Current skeleton status" for exactly what
does and does not exist in this patch.
"""

from __future__ import annotations

from fastapi import FastAPI

from .settings import Settings

settings = Settings()

app = FastAPI(title="Orion Diffusion Host", version=settings.SERVICE_VERSION)


@app.get("/health")
async def health() -> dict:
    return {
        "ok": True,
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "node": settings.NODE_NAME,
        "model_loaded": False,
        "note": "skeleton only -- no diffusion model wired yet",
    }
