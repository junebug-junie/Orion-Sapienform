"""orion-hardware-watch: AC failure and CPU/GPU heat -> alert, urgent investigation, pool shedding.

Bus first: incidents go out on orion:hardware:watch:incident, investigations on
orion:curiosity:urgent:request. HTTP is /health, the incident list, the operator resolve, and (only
with HARDWARE_WATCH_TEST_HOOK_ENABLED) a simulated AC incident for the live smoke.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import ServiceRef
from orion.core.bus.bus_service_chassis import ChassisConfig, HeartbeatOnly
from orion.notify.client import NotifyClient

from app.settings import get_settings
from app.store import PostgresStore
from app.watcher import Watcher

_settings = get_settings()
logging.basicConfig(level=getattr(logging, _settings.log_level.upper(), logging.INFO))
logger = logging.getLogger("orion-hardware-watch.main")

watcher: Watcher | None = None
_bus: OrionBusAsync | None = None
_store: PostgresStore | None = None
_tasks: list[asyncio.Task] = []
_chassis: list[Any] = []
_stop = asyncio.Event()


def _source() -> ServiceRef:
    return ServiceRef(name=_settings.service_name, version=_settings.service_version, node=_settings.node_name)


async def _tick_forever() -> None:
    while not _stop.is_set():
        try:
            await watcher.tick()
        except Exception:  # noqa: BLE001 -- one bad tick must not stop the watcher
            logger.exception("hardware_watch_tick_failed")
        try:
            await asyncio.wait_for(_stop.wait(), timeout=_settings.tick_sec)
        except asyncio.TimeoutError:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    global watcher, _bus, _store
    heartbeat = HeartbeatOnly(ChassisConfig(
        service_name=_settings.service_name, service_version=_settings.service_version,
        node_name=_settings.node_name, bus_url=_settings.orion_bus_url, bus_enabled=_settings.orion_bus_enabled,
        heartbeat_interval_sec=_settings.heartbeat_interval_sec))
    await heartbeat.start_background()
    _chassis.append(heartbeat)
    _store = PostgresStore(_settings.postgres_uri, cooling_role=_settings.ac_role)
    await asyncio.to_thread(_store.check_schema)
    _bus = OrionBusAsync(url=_settings.orion_bus_url, enabled=_settings.orion_bus_enabled)
    await _bus.connect()
    notify = NotifyClient(_settings.notify_base_url, api_token=_settings.notify_api_token or None)
    watcher = Watcher(settings=_settings, store=_store, publish=_bus.publish, notify=notify.send, source=_source())
    logger.info("hardware_watch_ready enabled=%s controller=%s shed=%s urgent=%s test_hook=%s tick=%ss",
                _settings.enabled, _settings.heat_controller, _settings.shed_enabled, _settings.urgent_enabled,
                _settings.test_hook_enabled, _settings.tick_sec)
    _stop.clear()
    _tasks.append(asyncio.create_task(_tick_forever()))
    try:
        yield
    finally:
        _stop.set()
        for t in _tasks:
            t.cancel()
        await asyncio.gather(*_tasks, return_exceptions=True)
        for c in _chassis:
            try:
                await c.stop()
            except Exception:  # noqa: BLE001
                pass
        if _bus is not None:
            await _bus.close()
        if _store is not None:
            _store.close()


app = FastAPI(title="orion-hardware-watch", lifespan=lifespan)


def _jsonable(row: dict | None) -> dict | None:
    if row is None:
        return None
    return {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in row.items()}


@app.get("/health")
async def health() -> dict[str, Any]:
    last = watcher.last if watcher else None
    open_rows = await asyncio.to_thread(_store.open_incidents) if _store else []
    return {
        "ok": watcher is not None and (last is None or last.ok),
        "service": _settings.service_name,
        "enabled": _settings.enabled, "shed_enabled": _settings.shed_enabled,
        "urgent_enabled": _settings.urgent_enabled, "test_hook_enabled": _settings.test_hook_enabled,
        "heat_controller": _settings.heat_controller,
        # v2 (D2): the reflex's own current claim. Orion's learned shed (orion/autonomy/self_shed.py,
        # D8) is refused only while this is active -- not by any open incident.
        "reflex_shed": watcher.reflex_snapshot() if watcher else None,
        "last_tick_at": last.at.isoformat() if last and last.at else None,
        "last_tick_ok": last.ok if last else None,
        "errors": last.errors if last else {},
        "verdicts": last.verdicts if last else {},
        "open_incidents": [
            {k: _jsonable(r)[k] for k in ("incident_id", "rule", "subject", "open_reason", "opened_at",
                                          "shed_requested", "shed_reason", "alert_sent_at", "urgent_requested_at")}
            for r in open_rows],
    }


@app.get("/incidents")
async def incidents(limit: int = 50) -> dict[str, Any]:
    if _store is None:
        raise HTTPException(503, "starting")
    rows = await asyncio.to_thread(_store.list_incidents, max(1, min(limit, 500)))
    return {"incidents": [_jsonable(r) for r in rows]}


class ResolveBody(BaseModel):
    by: str = Field("juniper", max_length=64)


@app.post("/incidents/{incident_id}/resolve")
async def resolve(incident_id: str, body: ResolveBody | None = None) -> dict[str, Any]:
    """Juniper closes an incident by hand (false alarm, fixed). The pool's shed clears with it; the
    same rule/subject does not re-open for HARDWARE_WATCH_OPERATOR_SNOOZE_SEC."""
    if watcher is None:
        raise HTTPException(503, "starting")
    row = await watcher.resolve_by_operator(incident_id, (body.by if body else "juniper"))
    if row is None:
        raise HTTPException(404, "unknown incident")
    return {"incident": _jsonable(row)}


@app.post("/incidents/simulate")
async def simulate() -> dict[str, Any]:
    """Live-smoke hook: open a SIMULATED cooling incident (alert, urgent run, shed as the real one
    would). Closed only by POST /incidents/{id}/resolve. Refused unless the test hook is on."""
    if watcher is None:
        raise HTTPException(503, "starting")
    if not _settings.test_hook_enabled:
        raise HTTPException(403, "HARDWARE_WATCH_TEST_HOOK_ENABLED is false")
    row = await watcher.simulate_cooling()
    if row is None:
        raise HTTPException(409, "a cooling incident is already open")
    return {"incident": _jsonable(row)}
