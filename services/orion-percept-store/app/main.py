"""orion-percept-store: short-lived, content-addressed camera frames.

Exists so a frame can cross from wherever it was captured to wherever a model
can look at it, without becoming an archive.

**Why this is not Hub.** Hub already serves a content-addressed attachment
store at `/api/chat/attachments/{sha256}`, and the gateway already knows how to
fetch from it. Reusing it would have been fewer moving parts and it is the
wrong call: that store holds user chat uploads, keeps them indefinitely, and
lives in the process that holds the docker socket. Camera frames of a private
home should not inherit any of those three. A separate service is the privacy
boundary made structural rather than promised.

**Retention is the point, not housekeeping.** Default 1 hour. A percept exists
to be interpreted once; the interpretation is the durable artifact. The sweep
runs in-process on a timer so "we will add retention later" cannot happen --
this repo has enough unbounded tables already.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from loguru import logger

from app.settings import get_settings
from app.storage import SHA256_RE, PerceptStore, sniff_mime

settings = get_settings()
store = PerceptStore(settings.PERCEPT_STORE_DIR)


def _check_token(supplied: Optional[str]) -> None:
    """Shared-secret gate. Empty config disables it -- see settings."""
    expected = str(settings.PERCEPT_STORE_TOKEN or "")
    if not expected:
        return
    if supplied != expected:
        raise HTTPException(status_code=401, detail="bad or missing X-Orion-Percept-Token")


async def _sweep_loop() -> None:
    while True:
        try:
            removed = store.sweep(max_age_seconds=settings.PERCEPT_RETENTION_SECONDS)
            if removed:
                logger.info(
                    "[PERCEPT] swept {} blob(s) older than {}s",
                    removed,
                    settings.PERCEPT_RETENTION_SECONDS,
                )
        except Exception as exc:
            # A sweep failure must not take the store down -- serving reads
            # matters more than pruning on schedule.
            logger.warning("[PERCEPT] sweep failed: {}", exc)
        await asyncio.sleep(max(30, int(settings.PERCEPT_SWEEP_INTERVAL_SECONDS)))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Sweep once at boot: a crash-restart loop must not let expired frames
    # survive indefinitely by never reaching the first timer tick.
    try:
        removed = store.sweep(max_age_seconds=settings.PERCEPT_RETENTION_SECONDS)
        logger.info("[PERCEPT] boot sweep removed {} expired blob(s)", removed)
    except Exception as exc:
        logger.warning("[PERCEPT] boot sweep failed: {}", exc)
    task = asyncio.create_task(_sweep_loop())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass


app = FastAPI(title="orion-percept-store", version=settings.SERVICE_VERSION, lifespan=lifespan)


@app.get("/healthz")
def healthz():
    return {"status": "ok", "service": settings.SERVICE_NAME, "version": settings.SERVICE_VERSION}


@app.get("/readyz")
def readyz():
    return {"status": "ready", "detail": "ok"}


@app.get("/stats")
def stats():
    """Blob count/size/oldest. No hashes -- this endpoint must not enumerate."""
    return {
        **store.stats(),
        "retention_seconds": settings.PERCEPT_RETENTION_SECONDS,
    }


@app.post("/percepts")
async def put_percept(
    request: Request,
    x_orion_percept_token: Optional[str] = Header(default=None, alias="X-Orion-Percept-Token"),
):
    """Store raw image bytes, return the sha256 to fetch them by.

    Takes a raw body rather than multipart: the only caller is a capture agent
    posting one JPEG, and multipart would add a dependency and a parsing
    surface for no benefit.
    """
    _check_token(x_orion_percept_token)
    data = await request.body()
    if not data:
        raise HTTPException(status_code=400, detail="empty body")
    if len(data) > settings.PERCEPT_MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"percept exceeds PERCEPT_MAX_BYTES ({settings.PERCEPT_MAX_BYTES})",
        )
    # Sniffed, never trusted from the header: the value ends up in a model
    # prompt downstream.
    mime = sniff_mime(data)
    allowed = {m.strip() for m in settings.PERCEPT_ALLOWED_MIMES.split(",") if m.strip()}
    if mime is None or mime not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"unsupported percept type (sniffed {mime!r}); allowed: {sorted(allowed)}",
        )
    sha256 = store.put(data, mime=mime)
    logger.info("[PERCEPT] stored sha={} mime={} bytes={}", sha256[:12], mime, len(data))
    return JSONResponse({"sha256": sha256, "mime": mime, "bytes": len(data)})


@app.get("/percepts/{sha256}")
def get_percept(
    sha256: str,
    x_orion_percept_token: Optional[str] = Header(default=None, alias="X-Orion-Percept-Token"),
):
    _check_token(x_orion_percept_token)
    if not SHA256_RE.match(sha256):
        # 400 not 404: a malformed address is a caller bug, and saying so does
        # not leak whether any particular blob exists.
        raise HTTPException(status_code=400, detail="sha256 must be 64 hex chars")
    try:
        data, mime = store.get(sha256)
    except KeyError:
        raise HTTPException(status_code=404, detail="percept not found or expired")
    except ValueError as exc:
        logger.error("[PERCEPT] integrity failure for {}: {}", sha256[:12], exc)
        raise HTTPException(status_code=500, detail="stored content failed its own hash check")
    return Response(content=data, media_type=mime)
