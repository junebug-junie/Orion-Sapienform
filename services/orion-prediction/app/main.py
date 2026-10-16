"""orion-prediction: forecasting, evaluation, and cause-and-effect estimates over Orion's telemetry.

Bus: forecasts on orion:prediction:forecast. Monitoring alerts are in /health and /series/{id}/live.
HTTP: health, series + scoreboard, latest forecast, live metrics, ad-hoc backtest, scenario
(conditional), interventions + effect estimates (observational), switchback experiments
(experimental), and GET /effects for the best available answer.
Spec: docs/superpowers/specs/2026-10-15-orion-prediction-service-design.md
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
import uuid
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import AwareDatetime, BaseModel, Field

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import ServiceRef
from orion.core.bus.bus_service_chassis import ChassisConfig, HeartbeatOnly

from app.causal import CausalError
from app.effects import Effects
from app.engine import Engine
from app.series import load_registry
from app.settings import get_settings
from app.store import PostgresStore

_settings = get_settings()
logging.basicConfig(level=getattr(logging, _settings.log_level.upper(), logging.INFO))
logger = logging.getLogger("orion-prediction.main")

engine: Engine | None = None
effects: Effects | None = None
_store: PostgresStore | None = None
_bus: OrionBusAsync | None = None
_tasks: list[asyncio.Task] = []
_chassis: list[Any] = []
_stop = asyncio.Event()
_last_tick: dict[str, Any] = {"at": None, "ok": None, "error": None}


async def _sync(fn, *args):
    return await asyncio.to_thread(fn, *args)


async def _tick_forever() -> None:
    while not _stop.is_set():
        try:
            await engine.tick(_sync)
            _last_tick.update(at=engine.clock(), ok=True, error=None)
        except Exception as exc:  # noqa: BLE001 -- one bad tick must not stop the loop
            _last_tick.update(at=engine.clock(), ok=False, error=f"{type(exc).__name__}: {exc}"[:300])
            logger.exception("prediction_tick_failed")
        try:
            await asyncio.wait_for(_stop.wait(), timeout=_settings.tick_sec)
        except asyncio.TimeoutError:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, effects, _store, _bus
    heartbeat = HeartbeatOnly(ChassisConfig(
        service_name=_settings.service_name, service_version=_settings.service_version,
        node_name=_settings.node_name, bus_url=_settings.orion_bus_url, bus_enabled=_settings.orion_bus_enabled,
        heartbeat_interval_sec=_settings.heartbeat_interval_sec))
    await heartbeat.start_background()
    _chassis.append(heartbeat)
    registry = load_registry(_settings.series_path)
    _store = PostgresStore(_settings.postgres_uri)
    await asyncio.to_thread(_store.check_schema)
    _bus = OrionBusAsync(url=_settings.orion_bus_url, enabled=_settings.orion_bus_enabled)
    await _bus.connect()
    engine = Engine(registry, _store, backtest_windows=_settings.backtest_windows,
                    backtest_step_min=_settings.backtest_step_min,
                    select_interval_sec=_settings.select_interval_sec, settle_sec=_settings.settle_sec,
                    publish_enabled=_settings.publish_enabled, publish=_bus.publish,
                    source=ServiceRef(name=_settings.service_name, version=_settings.service_version,
                                      node=_settings.node_name))
    effects = Effects(engine)
    logger.info("prediction_ready enabled=%s publish=%s targets=%s", _settings.enabled, _settings.publish_enabled,
                [s.series_id for s in registry.targets()])
    _stop.clear()
    if _settings.enabled:
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


app = FastAPI(title="orion-prediction", lifespan=lifespan)


def _ready() -> Engine:
    if engine is None:
        raise HTTPException(503, "starting")
    return engine


def _jsonable(v: Any) -> Any:
    if isinstance(v, datetime):
        return v.isoformat()
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


async def _causal(fn, *args):
    try:
        return _jsonable(await asyncio.to_thread(fn, *args))
    except CausalError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/health")
async def health() -> dict[str, Any]:
    e = engine
    series = {}
    if e is not None:
        for sid, st in e.state.items():
            sv = st.served
            series[sid] = {
                "selected_at": sv.selected_at.isoformat() if sv else None,
                "next_select_at": st.next_select_at.isoformat() if st.next_select_at else None,
                "assignment": sv.selection.assignment if sv else None,
                "served_baseline": sv.selection.served_baseline if sv else None,
                "refused_seasonalities": list(sv.refused) if sv else None,
                "last_select_seconds": st.last_select_seconds,
                "last_input_ts": st.last_input_ts.isoformat() if st.last_input_ts else None,
                "alerts": st.alerts,
            }
    return {"ok": e is not None and _last_tick["ok"] is not False, "service": _settings.service_name,
            "enabled": _settings.enabled, "publish_enabled": _settings.publish_enabled,
            "last_tick_at": _jsonable(_last_tick["at"]), "last_tick_error": _last_tick["error"], "series": series}


@app.get("/series")
async def series_list() -> dict[str, Any]:
    e = _ready()
    return {"series": [{"series_id": s.series_id, "role": s.role, "unit": s.unit, "grain_min": s.grain_min,
                        "horizons_min": list(s.horizons_min), "covariates": list(s.covariates),
                        "seasonalities": [x.__dict__ for x in s.seasonalities],
                        "thresholds": [x.__dict__ for x in s.thresholds], "consumers": list(s.consumers)}
                       for s in e.registry.series.values()]}


@app.get("/series/{series_id}/models")
async def scoreboard(series_id: str) -> dict[str, Any]:
    """Latest selection: every model's backtest metrics per horizon, and which model serves which horizon."""
    _ready()
    rows = await asyncio.to_thread(_store.latest_selection, series_id)
    if not rows:
        raise HTTPException(404, "no selection yet")
    keep = ("run_id", "kind", "model", "fitted_at", "n_obs", "n_windows", "metrics", "assignment", "served_baseline",
            "usable_seasonalities", "refused_seasonalities", "backtest_seconds", "errors")
    return {"series_id": series_id, "runs": [_jsonable({k: r[k] for k in keep}) for r in rows]}


@app.get("/series/{series_id}/forecast")
async def latest_forecast(series_id: str) -> dict[str, Any]:
    _ready()
    rows = await asyncio.to_thread(_store.latest_forecast, series_id)
    if not rows:
        raise HTTPException(404, "no forecast yet")
    keep = ("origin_ts", "issued_at", "target_ts", "horizon_min", "point", "quantiles", "score_status", "actual")
    return {"series_id": series_id, "steps": [_jsonable({k: r[k] for k in keep}) for r in rows]}


@app.get("/series/{series_id}/live")
async def live(series_id: str) -> dict[str, Any]:
    e = _ready()
    spec = e.registry.get(series_id)
    if spec is None or spec.role != "target":
        raise HTTPException(404, "unknown target series")
    out = await asyncio.to_thread(e.live_metrics, spec, e.clock())
    return {"series_id": series_id, "live": out, "alerts": e.state[series_id].alerts}


class ScenarioBody(BaseModel):
    series_id: str
    covariate: str
    path: list[float] = Field(..., min_length=1, max_length=2016)
    days: float = Field(7.0, ge=2, le=21)


@app.post("/scenario")
async def post_scenario(body: ScenarioBody) -> dict[str, Any]:
    _ready()
    return await _causal(effects.scenario, body.series_id, body.covariate, body.path, body.days)


class InterventionBody(BaseModel):
    action: str = Field(..., pattern=r"^[a-z0-9_.:-]{1,120}$")
    started_at: AwareDatetime
    ended_at: AwareDatetime
    source: str = Field(..., min_length=1, max_length=80)
    notes: str = Field("", max_length=2000)
    arm: str | None = Field(None, pattern=r"^(on|off)$")
    experiment_id: uuid.UUID | None = None


@app.post("/interventions")
async def post_intervention(body: InterventionBody) -> dict[str, Any]:
    _ready()
    if (body.arm is None) != (body.experiment_id is None):
        raise HTTPException(422, "arm and experiment_id go together")
    return await _causal(effects.log_intervention, body.action, body.started_at, body.ended_at, body.source,
                         body.notes, body.arm, str(body.experiment_id) if body.experiment_id else None)


@app.get("/interventions")
async def get_interventions(action: str | None = None, limit: int = 100) -> dict[str, Any]:
    _ready()
    rows = await asyncio.to_thread(_store.interventions, action, None, max(1, min(limit, 500)))
    return {"interventions": _jsonable(rows)}


class EstimateBody(BaseModel):
    intervention_id: uuid.UUID
    series_id: str
    pre_days: float = Field(3.0, ge=1, le=7)
    lag_min: int = Field(30, ge=0, le=720)


@app.post("/effects/estimate")
async def post_estimate(body: EstimateBody) -> dict[str, Any]:
    _ready()
    return await _causal(effects.estimate_intervention, str(body.intervention_id), body.series_id, body.pre_days,
                         body.lag_min)


@app.get("/experiments/guard")
async def experiment_guard(series_id: str) -> dict[str, Any]:
    e = _ready()
    spec = e.registry.get(series_id)
    if spec is None or spec.role != "target":
        raise HTTPException(404, "unknown target series")
    return await _causal(effects.guard, spec)


class DesignBody(BaseModel):
    action: str = Field(..., pattern=r"^[a-z0-9_.:-]{1,120}$")
    series_id: str
    start_at: AwareDatetime
    block_min: int = Field(20, ge=5, le=240)
    n_blocks: int = Field(8, ge=8, le=96)
    burn_in_min: int = Field(5, ge=0, le=120)
    seed: int | None = None


@app.post("/experiments/design")
async def post_design(body: DesignBody) -> dict[str, Any]:
    _ready()
    return await _causal(effects.design_experiment, body.action, body.series_id, body.start_at, body.block_min,
                         body.n_blocks, body.burn_in_min, body.seed)


@app.post("/experiments/{experiment_id}/analyze")
async def post_analyze(experiment_id: uuid.UUID) -> dict[str, Any]:
    _ready()
    return await _causal(effects.analyze_experiment, str(experiment_id))


@app.get("/effects/{action}")
async def get_effect(action: str, series_id: str) -> dict[str, Any]:
    """The Orion-facing answer: best available estimate of how much `action` moves `series_id`."""
    _ready()
    return await _causal(effects.best_effect, action, series_id)
