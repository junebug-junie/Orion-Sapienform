from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from ..settings import Settings
from ..store.redis_store import PadStore
from .api import build_router


def _join_path(base: str, suffix: str) -> str:
    if base == "/":
        return suffix
    return f"{base}{suffix}"


def mount_web(app: FastAPI, *, store: PadStore, settings: Settings) -> None:
    router = build_router(store=store, settings=settings)
    base_path = settings.public_base_path

    app.include_router(router)
    if base_path != "/":
        app.include_router(router, prefix=base_path)
        app.add_api_route(base_path, lambda: RedirectResponse(url=f"{base_path}/ui"), methods=["GET"])
        app.add_api_route(f"{base_path}/", lambda: RedirectResponse(url=f"{base_path}/ui"), methods=["GET"])

    app.mount("/static", StaticFiles(directory="app/static"), name="static")
    if base_path != "/":
        app.mount(_join_path(base_path, "/static"), StaticFiles(directory="app/static"), name="static_base")
