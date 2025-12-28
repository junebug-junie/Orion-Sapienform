from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

from . import routes
from .bus_runtime import start_services
from .settings import settings

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    stop_event = asyncio.Event()
    await start_services(stop_event)
    try:
        yield
    finally:
        stop_event.set()


app = FastAPI(
    title=settings.SERVICE_NAME,
    version=settings.SERVICE_VERSION,
    lifespan=lifespan,
)

app.include_router(routes.router, prefix="/api")


@app.get("/health")
def health():
    return {"ok": True, "service": settings.SERVICE_NAME, "version": settings.SERVICE_VERSION}


@app.get("/")
def read_root():
    return {"message": f"{settings.SERVICE_NAME} is alive"}
