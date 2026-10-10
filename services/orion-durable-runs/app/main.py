from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.bus_service_chassis import ChassisConfig, HeartbeatOnly, Hunter
from orion.core.bus.rpc_health_publish import RpcHealthPublisher
from orion.schemas.chat_history import CHAT_HISTORY_TURN_KIND
from orion.schemas.durable_run import (
    DURABLE_RUN_RECEIPT_KIND,
    DURABLE_RUN_REQUEST_KIND,
    DURABLE_RUN_STATE_KIND,
    DurableRunReceiptV1,
    DurableRunRequestV1,
    DurableRunStateV1,
)
from orion.schemas.situation_state import SITUATION_STATE_CHANNEL, SITUATION_STATE_KIND, SITUATION_STATE_REDIS_KEY
from orion.schemas.vision_sighting import IDENTITY_SIGHTING_KIND
from orion.schemas.gpu_pool import GPU_POOL_EVENT_CHANNEL, GPU_POOL_EVENT_KIND
from orion.schemas.resource_admission import RESOURCE_EVENT_CHANNEL, RESOURCE_EVENT_KIND, ResourceEventV1
from orion.schemas.memory_episode import MEMORY_EPISODE_CLOSED_KIND

from app.settings import get_settings

_settings = get_settings()
logging.basicConfig(level=getattr(logging, _settings.log_level.upper(), logging.INFO))
logger = logging.getLogger("orion-durable-runs.main")

runner: Any = None
rpc_bus: OrionBusAsync | None = None
hunter: Hunter | None = None
heartbeat: HeartbeatOnly | None = None
_stop = asyncio.Event()
_sweep_task: asyncio.Task[None] | None = None
_checkpointer_cm: Any = None
admission: Any = None
_admission_task: asyncio.Task | None = None
_reconcile_task: asyncio.Task | None = None
situation: Any = None
temporal_self: Any = None
rpc_health_publisher: RpcHealthPublisher | None = None


def _rpc_bus_getter() -> OrionBusAsync | None:
    return rpc_bus


def build_rpc_health_publisher() -> RpcHealthPublisher:
    """Publishes the long-lived rpc_bus's RPC-health window: the runner's rpc_request
    calls (harness turn, verb dispatch) and the GPU pool lease RPCs (``gpu_pool_lease``).
    Stage 4.5 removed every outbound HTTP hop (gateway /routes, lane /slots, elastic
    controller, cabinet, thought). Single container -> instance="main"."""
    s = _settings
    return RpcHealthPublisher(
        enabled=s.rpc_health_publish_enabled and s.orion_bus_enabled,
        bus_getter=_rpc_bus_getter,
        service=s.service_name,
        node=s.node_name,
        instance="main",
        source=ServiceRef(name=s.service_name, version=s.service_version, node=s.node_name),
        interval_sec=s.rpc_health_publish_interval_sec,
        include_channel_latency=s.rpc_health_channel_latency_enabled,
    )


def _chassis_cfg() -> ChassisConfig:
    s = _settings
    return ChassisConfig(
        service_name=s.service_name,
        service_version=s.service_version,
        node_name=s.node_name,
        bus_url=s.orion_bus_url,
        bus_enabled=s.orion_bus_enabled,
        heartbeat_interval_sec=s.heartbeat_interval_sec,
    )


async def _handle_request(env: BaseEnvelope) -> None:
    if env.kind == IDENTITY_SIGHTING_KIND:
        if situation is not None and isinstance(env.payload, dict):
            from app.situation_driver import event_from_sighting

            situation.offer(event_from_sighting(env.payload))
        return
    if env.kind in (CHAT_HISTORY_TURN_KIND, DURABLE_RUN_STATE_KIND):
        # Situation graph inputs (shadow), and a Juniper turn wakes the regulate step.
        if env.kind == CHAT_HISTORY_TURN_KIND and temporal_self is not None and isinstance(env.payload, dict):
            temporal_self.offer_chat_turn(env.payload)
        if situation is not None and isinstance(env.payload, dict):
            from app.situation_driver import event_from_chat_turn, event_from_run_state

            situation.offer(event_from_chat_turn(env.payload) if env.kind == CHAT_HISTORY_TURN_KIND
                            else event_from_run_state(env.payload))
        return
    if env.kind == GPU_POOL_EVENT_KIND:
        # The pool's lease lifecycle: a waiting run wakes on "granted" for its own holder.
        if admission is not None and isinstance(env.payload, dict):
            await admission.on_pool_event(env.payload)
        return
    if env.kind == RESOURCE_EVENT_KIND:
        if admission is not None:
            await admission.wakeup(ResourceEventV1.model_validate(env.payload))
        return
    if env.kind == MEMORY_EPISODE_CLOSED_KIND:
        await _submit_episode_distill(env)
        return
    if env.kind != DURABLE_RUN_REQUEST_KIND:
        logger.warning("durable_run_request_unexpected_kind kind=%s", env.kind)
        return
    try:
        request = DurableRunRequestV1.model_validate(env.payload or {})
    except Exception as exc:  # noqa: BLE001
        logger.warning("durable_run_request_invalid corr=%s err=%s", env.correlation_id, exc)
        return
    if runner is None:
        logger.warning("durable_run_request_dropped reason=runner_not_started run=%s", request.run_id)
        return
    logger.info("durable_run_request run=%s workflow=%s corr=%s", request.run_id, request.workflow, request.correlation_id)
    if request.admission is not None:
        if admission is None:
            logger.error("durable_admission_disabled run=%s", request.run_id)
            return
        receipt = DurableRunReceiptV1(**await admission.submit(request))
        if env.reply_to and rpc_bus is not None:
            await rpc_bus.publish(env.reply_to, BaseEnvelope(kind=DURABLE_RUN_RECEIPT_KIND,
                source=ServiceRef(name=_settings.service_name, version=_settings.service_version, node=_settings.node_name),
                correlation_id=env.correlation_id, payload=receipt.model_dump(mode="json")))
        return
    await runner.start_run(request)


async def _submit_episode_distill(env: BaseEnvelope) -> None:
    """orion:memory:episode:closed -> one admitted memory.episode_distill run (SHADOW).

    The run id is deterministic per episode, so a re-delivered close event is refused by the
    admission store as a duplicate. Skipped (command-only) episodes get no run.
    """
    if not _settings.memory_episode_writer_enabled:
        return
    if admission is None:
        logger.warning("memory_episode_distill_dropped reason=admission_disabled corr=%s", env.correlation_id)
        return
    from app.episode_distill_graph import request_from_closed_event

    try:
        request = request_from_closed_event(env.payload or {}, settings=_settings)
    except Exception as exc:  # noqa: BLE001
        logger.warning("memory_episode_closed_invalid corr=%s err=%s", env.correlation_id, exc)
        return
    if request is None:
        logger.info("memory_episode_distill_skipped episode=%s", (env.payload or {}).get("episode_id"))
        return
    try:
        receipt = await admission.submit(request)
    except ValueError as exc:   # SubmissionConflict: this episode was already submitted
        logger.info("memory_episode_distill_duplicate run=%s err=%s", request.run_id, exc)
        return
    logger.info("memory_episode_distill_submitted run=%s status=%s", request.run_id, receipt.get("status"))


async def _open_checkpointer():
    """LangGraph's Postgres saver on a CONNECTION POOL, tables created on
    first boot.

    Live incident 2026-09-06 20:52Z: `from_conn_string` gives the saver ONE
    psycopg async connection. A node that awaits for an hour (the harness
    turn RPC) and a concurrent `aget_state` (a second run's kickoff, the
    failure handler) contend for that single connection and the runner
    deadlocked -- no state rows, second request never seeded, health still
    listing the first run as active. Concurrency here is the norm, not the
    exception, so the saver gets a pool (LangGraph's own recommendation).
    """
    global _checkpointer_cm
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    _checkpointer_cm = AsyncConnectionPool(
        conninfo=_settings.postgres_uri,
        min_size=1,
        max_size=8,
        open=False,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    await _checkpointer_cm.open()
    saver = AsyncPostgresSaver(_checkpointer_cm)
    await saver.setup()
    return saver


def _build_situation(saver: Any):
    """The situation.update writer: reads episode_memory on the checkpointer's pool, projects to
    Redis + bus on the long-lived rpc_bus, traces each step on the durable state channel."""
    from datetime import datetime, timedelta, timezone
    from uuid import UUID, uuid4

    from app import situation_store
    from app.situation_driver import SituationDriver
    from app.situation_graph import SituationDeps
    from orion.core.bus.bus_schemas import BaseEnvelope as _Env

    s = _settings
    ttl = timedelta(hours=s.situation_default_ttl_hours)
    source = ServiceRef(name=s.service_name, version=s.service_version, node=s.node_name)

    async def project(model) -> None:
        if rpc_bus is None:
            return
        body = model.model_dump_json()
        try:
            await rpc_bus.redis.setex(SITUATION_STATE_REDIS_KEY, s.situation_redis_ttl_sec, body)
        except Exception:  # noqa: BLE001
            logger.warning("situation_redis_write_failed", exc_info=True)
        await rpc_bus.publish(SITUATION_STATE_CHANNEL, _Env(kind=SITUATION_STATE_KIND, source=source,
                                                            correlation_id=uuid4(), payload=model.model_dump(mode="json")))

    def _corr(raw: str):
        # sql-writer stamps the row's correlation_id from the ENVELOPE, so it must carry the
        # triggering turn's id (live 2026-10-07: rows showed a random id instead of the turn's).
        try:
            return UUID(str(raw))
        except (TypeError, ValueError):
            return uuid4()

    async def publish_state(row: DurableRunStateV1) -> None:
        if rpc_bus is None:
            return
        try:
            await rpc_bus.publish(s.state_channel, _Env(kind=DURABLE_RUN_STATE_KIND, source=source,
                                                        correlation_id=_corr(row.correlation_id),
                                                        payload=row.model_dump(mode="json")))
        except Exception:  # noqa: BLE001
            logger.warning("situation_state_publish_failed", exc_info=True)

    deps = SituationDeps(
        load_facts=lambda now: situation_store.load_facts(_checkpointer_cm, now, ttl),
        prime=lambda cues, exclude, limit: situation_store.prime(_checkpointer_cm, cues, exclude, limit),
        project=project,
        now=lambda: datetime.now(timezone.utc),
        default_ttl=ttl,
        prime_timeout_sec=s.situation_prime_timeout_sec,
        sighting_hold=timedelta(hours=s.situation_sighting_hold_hours),
    )
    return SituationDriver(checkpointer=saver, deps=deps, publish_state=publish_state,
                           tick_sec=s.situation_tick_sec, retention_days=s.situation_retention_days)


def _build_temporal_self(saver: Any):
    """The temporal_self.update writer (regulate node): reads chat/cabinet/GPU history on the
    checkpointer's pool, the rest drive from Redis, projects RegulationStateV1 to Redis."""
    from datetime import datetime, timezone

    from app import regulation_store
    from app.temporal_self_driver import TemporalSelfDriver
    from app.temporal_self_graph import TemporalSelfDeps
    from orion.regulation.rest_drive import rest_drive_view
    from orion.schemas.drive_reading import REST_DRIVE_REDIS_KEY, parse_drive_reading
    from orion.schemas.regulation import REGULATION_STATE_REDIS_KEY

    s = _settings

    async def read_inputs(now, prev, last_turn_at):
        return await regulation_store.read_arousal_inputs(
            _checkpointer_cm, now, prev, last_turn_event_at=last_turn_at,
            gpu_queue_floor=s.regulation_strained_gpu_queue_min,
            gpu_sustain_sec=s.regulation_strained_gpu_queue_sec)

    async def read_drives(now):
        """The rest drive verbatim from its owner's Redis key (orion-dream, 1800 s TTL). Trace only:
        arousal never reads it. Absent/stale/unparseable is a warning, never a fabricated reading."""
        if rpc_bus is None:
            return [], ["rest_drive:no_bus"]
        try:
            raw = await asyncio.wait_for(rpc_bus.redis.get(REST_DRIVE_REDIS_KEY), timeout=1.0)
        except Exception:  # noqa: BLE001
            return [], ["rest_drive:read_failed"]
        reading = parse_drive_reading(raw)
        if reading is None:
            return [], [f"rest_drive:{'absent' if raw is None else 'unparseable'}"]
        view = rest_drive_view(reading, now=now, max_age_sec=1800.0)
        return [reading], ([f"rest_drive:{view.reason}"] if view.verdict == "unknown" else [])

    async def project(model) -> None:
        if rpc_bus is None:
            raise RuntimeError("no bus")
        await rpc_bus.redis.setex(REGULATION_STATE_REDIS_KEY, s.regulation_redis_ttl_sec, model.model_dump_json())

    async def record_transition(prev, new, day_id) -> bool:
        return await regulation_store.record_transition(
            _checkpointer_cm, regulation_store.transition_row(prev, new, day_id))

    deps = TemporalSelfDeps(
        read_inputs=read_inputs, read_drives=read_drives, project=project,
        record_transition=record_transition, now=lambda: datetime.now(timezone.utc),
        arousal_enabled=s.regulation_arousal_enabled, engaged_minutes=s.dream_idle_minutes,
        gpu_queue_floor=s.regulation_strained_gpu_queue_min,
        gpu_sustain_sec=s.regulation_strained_gpu_queue_sec,
        clear_sec=s.regulation_strained_clear_sec,
        max_prev_gap_sec=3.0 * s.temporal_self_tick_sec,
    )
    return TemporalSelfDriver(checkpointer=saver, deps=deps,
                              timezone_name=s.orion_situation_timezone, tick_sec=s.temporal_self_tick_sec,
                              retention_days=s.temporal_self_retention_days)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global runner, rpc_bus, hunter, heartbeat, _sweep_task, admission, _admission_task, _reconcile_task
    global rpc_health_publisher, situation, temporal_self
    from app.runner import DurableRunner

    try:
        heartbeat = HeartbeatOnly(_chassis_cfg())
        await heartbeat.start_background()
    except Exception as exc:  # noqa: BLE001
        logger.warning("system_health_heartbeat_start_failed error=%s", exc)
        heartbeat = None

    if _settings.enabled:
        saver = await _open_checkpointer()
        if _settings.orion_bus_enabled:
            rpc_bus = OrionBusAsync(url=_settings.orion_bus_url)
            await rpc_bus.connect()
            rpc_health_publisher = build_rpc_health_publisher()
            rpc_health_publisher.start()
        runner = DurableRunner(_settings, bus=rpc_bus, checkpointer=saver)
        if _settings.admission_enabled:
            from app.admission_runtime import AdmissionRuntime
            admission = AdmissionRuntime(_settings, runner, _checkpointer_cm)
            # Admission migration is operator-managed, unlike saver migrations.
            # Fail startup if it has not been applied; never accept into memory.
            await admission.store.list_pending(limit=1)
            await admission.store.pending_outbox(limit=1)
        if _settings.resume_on_boot:
            try:
                counts = await runner.resume_unfinished()
                logger.info("durable_runs_resume_on_boot %s", counts)
            except Exception:  # noqa: BLE001
                logger.exception("durable_runs_resume_on_boot_failed")
        _stop.clear()
        _sweep_task = asyncio.create_task(runner.sweep_forever(_stop))
        if _settings.situation_graph_enabled and _settings.orion_bus_enabled:
            situation = _build_situation(saver)
            await situation.start(_stop)
        if _settings.temporal_self_enabled and _settings.orion_bus_enabled:
            temporal_self = _build_temporal_self(saver)
            await temporal_self.start(_stop)
        if admission is not None:
            _admission_task = asyncio.create_task(admission.run(_stop))
            if _settings.memory_episode_writer_enabled:
                from app.episode_distill_reconcile import run_reconcile_loop

                _reconcile_task = asyncio.create_task(
                    run_reconcile_loop(admission.pool, admission.submit, _settings, _stop))
        if _settings.orion_bus_enabled:
            patterns = [_settings.request_channel, RESOURCE_EVENT_CHANNEL]
            if admission is not None:
                patterns.append(GPU_POOL_EVENT_CHANNEL)  # wakes waiting runs on their hold's grant
                if _settings.memory_episode_writer_enabled:
                    patterns.append(_settings.memory_episode_closed_channel)
            if situation is not None:
                patterns += [_settings.chat_history_turn_channel, _settings.state_channel,
                             _settings.identity_sighting_channel]
            if temporal_self is not None and _settings.chat_history_turn_channel not in patterns:
                patterns.append(_settings.chat_history_turn_channel)
            hunter = Hunter(_chassis_cfg(), handler=_handle_request, patterns=patterns)
            await hunter.start_background()
            logger.info("durable_runs_listening channel=%s", _settings.request_channel)
    else:
        logger.info("durable_runs disabled (DURABLE_RUNS_ENABLED=false)")

    try:
        yield
    finally:
        _stop.set()
        if _sweep_task is not None:
            _sweep_task.cancel()
        if _reconcile_task is not None:
            _reconcile_task.cancel()
        if situation is not None:
            await situation.close()
        if temporal_self is not None:
            await temporal_self.close()
        if _admission_task is not None:
            _admission_task.cancel()
            await asyncio.gather(_admission_task, return_exceptions=True)
        if admission is not None:
            await admission.close()
        for closer in (hunter, heartbeat):
            if closer is not None:
                try:
                    await closer.stop()
                except Exception:  # noqa: BLE001
                    pass
        if rpc_health_publisher is not None:
            await rpc_health_publisher.stop()
            rpc_health_publisher = None
        if rpc_bus is not None:
            try:
                await rpc_bus.close()
            except Exception:  # noqa: BLE001
                pass
        if _checkpointer_cm is not None:
            try:
                await _checkpointer_cm.close()
            except Exception:  # noqa: BLE001
                pass


app = FastAPI(title="orion-durable-runs", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "service": _settings.service_name,
        "enabled": _settings.enabled,
        "active_runs": runner.active_run_ids if runner is not None else [],
        "admission_enabled": admission is not None,
        "admitted_active_runs": sorted(admission.active) if admission is not None else [],
        "situation": situation.health() if situation is not None else None,
        "temporal_self": temporal_self.health() if temporal_self is not None else None,
    }


@app.get("/regulation/state")
async def regulation_state():
    """The latest RegulationStateV1, verbatim (the same body as Redis orion:regulation:latest)."""
    if temporal_self is None:
        raise HTTPException(503, "temporal_self thread is disabled")
    if temporal_self.latest is None:
        raise HTTPException(404, "no regulation step has completed yet")
    return temporal_self.latest.model_dump(mode="json")


@app.get("/runs/unfinished")
async def unfinished() -> dict[str, Any]:
    if runner is None:
        return {"threads": []}
    threads = await runner.unfinished_threads()
    return {
        "threads": [
            {"thread_id": t, "next_node": n, "checkpoint_ts": ts.isoformat() if ts else None, "workflow": w}
            for t, n, ts, w in threads
        ]
    }


def _admission():
    if admission is None:
        raise HTTPException(503, "durable resource admission is disabled")
    return admission


@app.post("/runs", status_code=202)
async def submit(request: DurableRunRequestV1):
    try:
        return await _admission().submit(request)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get("/runs/{run_id}")
async def run_status(run_id: str):
    try:
        return await _admission().status(run_id)
    except KeyError as exc:
        raise HTTPException(404, "run not found") from exc


@app.post("/runs/{run_id}/{action}")
async def run_control(run_id: str, action: str):
    if action not in {"pause", "resume", "cancel"}:
        raise HTTPException(400, "action must be pause, resume or cancel")
    try:
        return await _admission().control(run_id, action)
    except KeyError as exc:
        raise HTTPException(404, "run not found") from exc


@app.post("/runs/{run_id}/release-outreach-lease")
async def release_outreach_lease(run_id: str):
    """Hub Door-A finished composing: release the run's GPU pool hold kept past finish.

    Only acts when the run is already ``terminal=completed`` (the Door-A hold). A live in-flight
    investigation cannot be released through this path. Idempotent when the hold is already gone.
    """
    try:
        return await _admission().release_outreach(run_id)
    except KeyError as exc:
        raise HTTPException(404, "run not found") from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc

