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


def _normalize_prefix(prefix: str) -> str:
    if not prefix:
        return ""
    p = prefix.strip()
    if not p.startswith("/"):
        p = "/" + p
    if p != "/" and p.endswith("/"):
        p = p.rstrip("/")
    return p


def _static_mount_name(prefix: str) -> str:
    if not prefix or prefix == "/":
        return "static"
    safe = prefix.strip("/").replace("/", "_")
    return f"static_{safe}"


def mount_web(app: FastAPI, *, store: PadStore, settings: Settings) -> None:
    router = build_router(store=store, settings=settings)
    base_path = _normalize_prefix(settings.public_base_path)
    fallback_path = "/landing-pad"

    # Always serve at root (covers proxies that strip /landing-pad before forwarding)
    app.include_router(router)

    # Serve under configured base path (covers proxies that preserve the prefix)
    prefixes = set()
    if base_path and base_path != "/":
        prefixes.add(base_path)
    prefixes.add(fallback_path)

    for prefix in sorted(prefixes):
        if not prefix or prefix == "/":
            continue
        app.include_router(router, prefix=prefix)

        # Make /<prefix> and /<prefix>/ land on the UI consistently
        app.add_api_route(prefix, lambda p=prefix: RedirectResponse(url=f"{p}/ui"), methods=["GET"])
        app.add_api_route(f"{prefix}/", lambda p=prefix: RedirectResponse(url=f"{p}/ui"), methods=["GET"])

    # Static mounted at root and at every prefix (mirrors spark-introspector approach)
    app.mount("/static", StaticFiles(directory="app/static"), name=_static_mount_name(""))
    for prefix in sorted(prefixes):
        if not prefix or prefix == "/":
            continue
        app.mount(
            _join_path(prefix, "/static"),
            StaticFiles(directory="app/static"),
            name=_static_mount_name(prefix),
        )
