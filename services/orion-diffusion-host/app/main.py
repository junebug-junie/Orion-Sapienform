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

Startup: the model load runs as a **background task**, not awaited inside
`lifespan()` before `yield`. Review finding (HIGH), confirmed live: awaiting
a blocking call before `yield` in an `@asynccontextmanager` lifespan holds
uvicorn's `lifespan.startup.complete` until that call resolves, so *every*
HTTP route -- including `/health` -- connection-refuses or hangs for the
full duration of a cold multi-GB weight download. That is exactly the
crash-loop-on-slow-startup failure shape `_load_pipeline`'s own docstring
says it exists to avoid; offloading to a thread does not fix it, because
the thing being awaited is still the one gating startup completion.

Concurrency: this service owns a single, dedicated GPU (README.md) and is
called at a slow, capacity-gated cadence (design doc §4) -- there is no
multi-profile / multi-GPU scheduler here the way orion-vision-host has one
(app/scheduler.py there exists because that service juggles several models
across several cards under real concurrent load; this one does not). Model
load and every generation call run on `_gpu_executor`, a dedicated
single-worker `ThreadPoolExecutor` -- not the default shared thread pool --
so exclusive GPU access is a structural property of the executor, not just
an `asyncio.Lock` convention a future edit could bypass (review finding:
the lock alone doesn't stop a future `--workers N` or a new call site that
forgets to acquire it; the single-worker executor does, because only one
thread in it can ever be running at a time regardless of who submits to
it). `_generation_lock` still exists on top of that, purely to give a
busy caller an immediate 429 (see `/generate`) instead of silently
queuing on the executor with no signal it's waiting.

No request-level cancel-on-timeout: canceling the *awaiting* coroutine
would not stop the blocking diffusers call already running in the executor
thread, so a naive timeout would let a second request start generating
while the first is still occupying the GPU underneath it -- worse than no
timeout at all. Fast-reject (429) when busy, not a timeout, is the
backpressure mechanism.
"""

from __future__ import annotations

import asyncio
import io
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from loguru import logger
from pydantic import BaseModel, Field

from orion.core.bus.bus_service_chassis import ChassisConfig, HeartbeatOnly

from .settings import Settings

settings = Settings()

_pipe = None
_load_error: Optional[str] = None
_generation_lock = asyncio.Lock()
_heartbeat_chassis: Optional[HeartbeatOnly] = None

# Dedicated, single-worker executor -- see module docstring "Concurrency".
# Both model load and every /generate call run here, never on the default
# shared thread pool.
_gpu_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="diffusion-gpu")

# Bounded retry on load failure -- a real observed cause on this exact
# service is a prior GPU-2 occupant's VRAM not yet released by the driver
# at the instant this container starts (README.md's two documented Circe
# collisions). Not infinite: a genuinely bad DIFFUSION_MODEL_ID or missing
# weights should surface as a permanent load_error, not retry forever.
_LOAD_RETRY_ATTEMPTS = 3
_LOAD_RETRY_BACKOFF_SEC = (5.0, 15.0, 30.0)


def _load_pipeline():
    """Blocking model load -- runs on `_gpu_executor` (see lifespan below).
    Any exception here is caught by the caller and recorded as
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


async def _load_model_background() -> None:
    """Runs as a fire-and-forget task from `lifespan()` -- never awaited
    before `yield`, so it cannot block startup (module docstring). Retries
    a bounded number of times with backoff before giving up and recording a
    permanent `_load_error`."""
    global _pipe, _load_error
    loop = asyncio.get_running_loop()
    for attempt in range(1, _LOAD_RETRY_ATTEMPTS + 1):
        logger.info(
            "loading diffusion model {} on {} (attempt {}/{})",
            settings.DIFFUSION_MODEL_ID,
            settings.DIFFUSION_DEVICE,
            attempt,
            _LOAD_RETRY_ATTEMPTS,
        )
        try:
            _pipe = await loop.run_in_executor(_gpu_executor, _load_pipeline)
            _load_error = None
            logger.info("diffusion model loaded: {}", settings.DIFFUSION_MODEL_ID)
            return
        except Exception as exc:  # noqa: BLE001 -- see _load_pipeline docstring
            _load_error = str(exc)
            logger.error(
                "diffusion model load failed (attempt {}/{}): {}",
                attempt,
                _LOAD_RETRY_ATTEMPTS,
                exc,
            )
            if attempt < _LOAD_RETRY_ATTEMPTS:
                await asyncio.sleep(_LOAD_RETRY_BACKOFF_SEC[attempt - 1])
    logger.error(
        "diffusion model load permanently failed after {} attempts -- "
        "/ready will stay 503 until this container is restarted",
        _LOAD_RETRY_ATTEMPTS,
    )


def build_heartbeat_chassis() -> HeartbeatOnly:
    """Bus-native SystemHealthV1 liveness, independent of model-load state --
    see docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-
    design.md. Off by default here (ORION_BUS_ENABLED=false, matching
    Patch 1's decision to defer all bus wiring), but wired now using the
    same existing chassis every other GPU-host service in this repo uses,
    so flipping the env flag on is all an operator needs to do -- no code
    gap left for a future patch to rediscover."""
    return HeartbeatOnly(
        ChassisConfig(
            service_name=settings.SERVICE_NAME,
            service_version=settings.SERVICE_VERSION,
            node_name=settings.NODE_NAME,
            bus_url=settings.ORION_BUS_URL,
            bus_enabled=settings.ORION_BUS_ENABLED,
            heartbeat_interval_sec=settings.HEARTBEAT_INTERVAL_SEC,
        )
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _heartbeat_chassis
    try:
        _heartbeat_chassis = build_heartbeat_chassis()
        await _heartbeat_chassis.start_background()
    except Exception as exc:  # noqa: BLE001 -- heartbeat is additive, never fatal to startup
        logger.warning("heartbeat start failed: {}", exc)
        _heartbeat_chassis = None

    # Fire-and-forget: startup completes immediately, /health and /ready
    # are live right away. See module docstring "Startup".
    asyncio.create_task(_load_model_background())

    yield

    if _heartbeat_chassis is not None:
        await _heartbeat_chassis.stop()
    _gpu_executor.shutdown(wait=False)


app = FastAPI(
    title="Orion Diffusion Host",
    version=settings.SERVICE_VERSION,
    lifespan=lifespan,
)


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    negative_prompt: Optional[str] = None
    width: Optional[int] = Field(default=None, gt=0, le=1536)
    height: Optional[int] = Field(default=None, gt=0, le=1536)
    num_inference_steps: Optional[int] = Field(default=None, gt=0, le=50)
    guidance_scale: Optional[float] = Field(default=None, ge=0.0, le=20.0)
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


def _log_prompt_token_budget(prompt: str) -> None:
    """Real-tokenizer visibility into whether `prompt` fits the loaded
    model's actual attention window.

    diffusers' own `encode_prompt()` truncates any prompt exceeding the
    text encoder's `model_max_length` (77 for every CLIP/OpenCLIP encoder
    SDXL-family models use) completely silently -- no exception, no
    response header, nothing in the 200 this endpoint returns. Confirmed
    live 2026-08-28: a caller (`orion-thought`'s visual chain, three
    patches deep) had been sending prompts up to 191 real tokens with no
    visibility that ~60% of the content -- including its most recently
    added context-seeds -- was silently discarded before the model ever
    saw it. That incident was only found by forensically re-tokenizing an
    already-stored prompt after the fact; this makes the same fact visible
    the moment it happens, in this service's own logs (CLAUDE.md §0A
    "runtime truth beats config truth": a log line with correlation
    evidence, not a config value or a clean 200 response).

    Uses the REAL tokenizer(s) already loaded as part of `_pipe` --
    `transformers` is already a hard dependency of this service
    (`requirements.txt`), so this is zero new cost, and it is the actual
    encoder the running model uses, not a same-family approximation.
    SDXL-family pipelines carry a second encoder (`tokenizer_2`,
    OpenCLIP-bigG) -- checked too, via the same `getattr` guard, since not
    every `AutoPipelineForText2Image` target has one.

    Log-only, best-effort: this must never change what gets generated or
    block a request -- any failure here (missing tokenizer attribute,
    tokenizer call error) is swallowed at DEBUG, not raised.
    """
    for attr in ("tokenizer", "tokenizer_2"):
        tok = getattr(_pipe, attr, None)
        if tok is None:
            continue
        try:
            max_len = getattr(tok, "model_max_length", None)
            n_tokens = len(tok(prompt)["input_ids"])
            if max_len and n_tokens > max_len:
                logger.warning(
                    "prompt exceeds {} budget: {} tokens > model_max_length={} "
                    "(prompt is {} chars) -- diffusers silently truncates; only the "
                    "first {} tokens are actually seen by the model",
                    attr,
                    n_tokens,
                    max_len,
                    len(prompt),
                    max_len,
                )
        except Exception as exc:  # noqa: BLE001 -- visibility only, never block generation
            logger.debug("prompt token-budget check on {} failed (non-fatal): {}", attr, exc)


def _run_generation(req: GenerateRequest) -> bytes:
    """Blocking diffusers call -- runs on `_gpu_executor` (module docstring
    "Concurrency"). torch is imported lazily, only when a seed is given --
    unlike `_load_pipeline`, this function runs on every request, including
    in tests that inject a fake pipe and never install real torch."""
    _log_prompt_token_budget(req.prompt)
    generator = None
    if req.seed is not None:
        import torch

        generator = torch.Generator(device=settings.DIFFUSION_DEVICE).manual_seed(req.seed)

    effective_guidance = (
        req.guidance_scale if req.guidance_scale is not None else settings.DIFFUSION_GUIDANCE_SCALE
    )
    if req.negative_prompt and effective_guidance <= 1.0:
        # diffusers only applies classifier-free guidance (and therefore
        # negative-prompt conditioning) above guidance_scale=1.0. The
        # documented sdxl-turbo operating point is 0.0 (settings.py), so a
        # caller-supplied negative_prompt silently does nothing there --
        # surfaced as a log line since the API response has no natural
        # place to put a warning on an otherwise-successful 200.
        logger.warning(
            "negative_prompt given but guidance_scale={} <= 1.0 -- diffusers will not "
            "apply it (classifier-free guidance requires guidance_scale > 1.0)",
            effective_guidance,
        )

    result = _pipe(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        width=req.width if req.width is not None else settings.DIFFUSION_DEFAULT_WIDTH,
        height=req.height if req.height is not None else settings.DIFFUSION_DEFAULT_HEIGHT,
        num_inference_steps=(
            req.num_inference_steps
            if req.num_inference_steps is not None
            else settings.DIFFUSION_NUM_INFERENCE_STEPS
        ),
        guidance_scale=effective_guidance,
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
    if _generation_lock.locked():
        # Fast-reject, not queue-and-wait -- module docstring "Concurrency".
        raise HTTPException(status_code=429, detail="another generation is already in flight")

    async with _generation_lock:
        loop = asyncio.get_running_loop()
        start = time.monotonic()
        try:
            png_bytes = await loop.run_in_executor(_gpu_executor, _run_generation, req)
        except Exception as exc:  # noqa: BLE001 -- report to caller, do not crash the service
            # Full exception (message + traceback) logged server-side only.
            # The client-facing detail is deliberately generic -- review
            # finding: the raw exception text can embed local filesystem
            # paths (MODEL_CACHE_DIR, HF_HOME) or CUDA driver internals, and
            # this is a plain-HTTP endpoint with no auth.
            logger.error("generation failed: {}\n{}", exc, traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=f"generation failed ({type(exc).__name__}) -- see service logs",
            )
        elapsed = time.monotonic() - start
        logger.info("generated image in {:.2f}s ({} bytes)", elapsed, len(png_bytes))

    return Response(content=png_bytes, media_type="image/png")
