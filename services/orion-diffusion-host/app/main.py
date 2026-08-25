"""orion-diffusion-host — real model wired (follow-up to Patch 1).

Patch 1 (docs/superpowers/specs/2026-08-20-reverie-visual-chain-design.md)
shipped a FastAPI process with only `/health`, no model, no GPU touch. This
patch loads one diffusion model at startup on Circe's GPU 2 (README.md
"Node/port/GPU assignment") and exposes `POST /generate`, returning raw
image bytes -- no bus consumer, no chain orchestration, no context-seeding
yet (still Patch 2 / Patch 3). A caller stores the returned bytes via
`orion.reverie.visual_storage.store_visual_artifact`, which expects exactly
this: raw bytes, mime sniffed from magic bytes, not declared by this
service.

Concurrency: this service owns a single, dedicated GPU (README.md) and is
called at a slow, capacity-gated cadence (design doc §4) -- there is no
multi-profile / multi-GPU scheduler here the way orion-vision-host has one
(app/scheduler.py there exists because that service juggles several models
across several cards under real concurrent load; this one does not). A
single `asyncio.Lock` serializes generation calls onto the one GPU. No
request-level cancel-on-timeout: canceling the *awaiting* coroutine would
not stop the blocking diffusers call already running in the executor
thread, so a naive timeout would let a second request start generating
while the first is still occupying the GPU underneath it -- worse than no
timeout at all. A slow/stuck generation is a real, visible symptom (the
caller's own request hangs) rather than a silently-timed-out one.
"""

from __future__ import annotations

import asyncio
import io
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from loguru import logger
from pydantic import BaseModel, Field

from .settings import Settings

settings = Settings()

_pipe = None
_load_error: Optional[str] = None
_generation_lock = asyncio.Lock()


def _load_pipeline():
    """Blocking model load -- runs in a worker thread (see lifespan below)
    so /health stays responsive for however long a cold weight download
    takes. Any exception here is caught by the caller and recorded as
    `_load_error`, not raised into the app's startup -- a bad
    DIFFUSION_MODEL_ID or a download failure must show up as
    model_loaded=false on /health, not crash-loop the container the way
    orion-vision-host's liveness watcher once did on a bad env value
    (README.md cross-reference, same failure shape)."""
    import torch
    from diffusers import AutoPipelineForText2Image

    dtype = torch.float16 if settings.DIFFUSION_DTYPE == "fp16" else torch.float32
    pipe = AutoPipelineForText2Image.from_pretrained(
        settings.DIFFUSION_MODEL_ID,
        torch_dtype=dtype,
        cache_dir=settings.MODEL_CACHE_DIR,
    )
    return pipe.to(settings.DIFFUSION_DEVICE)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipe, _load_error
    loop = asyncio.get_running_loop()
    logger.info(
        "loading diffusion model {} on {}",
        settings.DIFFUSION_MODEL_ID,
        settings.DIFFUSION_DEVICE,
    )
    try:
        _pipe = await loop.run_in_executor(None, _load_pipeline)
        logger.info("diffusion model loaded: {}", settings.DIFFUSION_MODEL_ID)
    except Exception as exc:  # noqa: BLE001 -- see _load_pipeline docstring
        _load_error = str(exc)
        logger.error("diffusion model load failed: {}", exc)
    yield


app = FastAPI(
    title="Orion Diffusion Host",
    version=settings.SERVICE_VERSION,
    lifespan=lifespan,
)


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    negative_prompt: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    seed: Optional[int] = None


@app.get("/health")
async def health() -> dict:
    return {
        "ok": True,
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "node": settings.NODE_NAME,
        "model_loaded": _pipe is not None,
        "model_id": settings.DIFFUSION_MODEL_ID if _pipe is not None else None,
        "load_error": _load_error,
        "note": "model wired -- see /ready for load status, POST /generate to produce an image",
    }


@app.get("/ready")
async def ready():
    body = {
        "ready": _pipe is not None,
        "model_loaded": _pipe is not None,
        "load_error": _load_error,
    }
    return JSONResponse(body, status_code=200 if body["ready"] else 503)


def _run_generation(req: GenerateRequest) -> bytes:
    """Blocking diffusers call -- runs in a worker thread, holding
    `_generation_lock` for the whole call so nothing else on this
    single-GPU service touches the pipeline concurrently. torch is imported
    lazily, only when a seed is given -- unlike `_load_pipeline`, this
    function runs on every request, including in tests that inject a fake
    pipe and never install real torch."""
    generator = None
    if req.seed is not None:
        import torch

        generator = torch.Generator(device=settings.DIFFUSION_DEVICE).manual_seed(req.seed)

    result = _pipe(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        width=req.width or settings.DIFFUSION_DEFAULT_WIDTH,
        height=req.height or settings.DIFFUSION_DEFAULT_HEIGHT,
        num_inference_steps=(
            req.num_inference_steps
            if req.num_inference_steps is not None
            else settings.DIFFUSION_NUM_INFERENCE_STEPS
        ),
        guidance_scale=(
            req.guidance_scale
            if req.guidance_scale is not None
            else settings.DIFFUSION_GUIDANCE_SCALE
        ),
        generator=generator,
    )
    image = result.images[0]
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


@app.post("/generate")
async def generate(req: GenerateRequest) -> Response:
    if len(req.prompt) > settings.DIFFUSION_MAX_PROMPT_CHARS:
        raise HTTPException(
            status_code=422,
            detail=f"prompt exceeds DIFFUSION_MAX_PROMPT_CHARS={settings.DIFFUSION_MAX_PROMPT_CHARS}",
        )
    if _pipe is None:
        raise HTTPException(
            status_code=503,
            detail=f"model not loaded{': ' + _load_error if _load_error else ' (still loading)'}",
        )

    async with _generation_lock:
        loop = asyncio.get_running_loop()
        start = time.monotonic()
        try:
            png_bytes = await loop.run_in_executor(None, _run_generation, req)
        except Exception as exc:  # noqa: BLE001 -- report to caller, do not crash the service
            logger.error("generation failed: {}", exc)
            raise HTTPException(status_code=500, detail=f"generation failed: {exc}")
        elapsed = time.monotonic() - start
        logger.info("generated image in {:.2f}s ({} bytes)", elapsed, len(png_bytes))

    return Response(content=png_bytes, media_type="image/png")
