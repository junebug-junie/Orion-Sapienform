import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Dict, Optional

from fastapi import FastAPI

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_service_chassis import ChassisConfig, Clock, Hunter
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.telemetry.biometrics import (
    BiometricsPayload,
    BiometricsSampleV1,
    BiometricsSummaryV1,
    BiometricsInductionV1,
    BiometricsClusterV1,
)
from orion.schemas.telemetry.spark_signal import SparkSignalV1
from app.metrics import collect_biometrics
from orion.telemetry.biometrics_pipeline import BiometricsPipeline, PipelineConfig
from app.settings import settings

logging.basicConfig(
    level=settings.LOG_LEVEL.upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(settings.SERVICE_NAME)


def chassis_cfg() -> ChassisConfig:
    return ChassisConfig(
        service_name=settings.SERVICE_NAME,
        service_version=settings.SERVICE_VERSION,
        node_name=settings.NODE_NAME,
        bus_url=settings.ORION_BUS_URL,
        bus_enabled=settings.ORION_BUS_ENABLED,
        heartbeat_interval_sec=settings.HEARTBEAT_INTERVAL_SEC,
        health_channel=settings.ORION_HEALTH_CHANNEL,
        error_channel=settings.ERROR_CHANNEL,
        shutdown_timeout_sec=settings.SHUTDOWN_GRACE_SEC,
    )


def _source() -> ServiceRef:
    return ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION, node=settings.NODE_NAME)


_pipeline = BiometricsPipeline(
    PipelineConfig(
        thermal_min_c=settings.THERMAL_MIN_C,
        thermal_max_c=settings.THERMAL_MAX_C,
        disk_bw_mbps=settings.DISK_BW_MBPS,
        net_bw_mbps=settings.NET_BW_MBPS,
        power_band_alpha=settings.POWER_BAND_ALPHA,
    ),
    window_sec=settings.TELEMETRY_INTERVAL,
)


async def publish_metrics(bus: OrionBusAsync) -> None:
    if not settings.ORION_BUS_ENABLED:
        return

    try:
        sample_data = collect_biometrics()
        sample = BiometricsSampleV1.model_validate(sample_data)
        summary, induction = _pipeline.update(sample.model_dump(mode="python"))

        raw_payload = BiometricsPayload(
            timestamp=sample.timestamp.isoformat(),
            gpu=sample.gpu,
            cpu=sample.cpu,
            node=sample.node,
            service_name=sample.service_name,
            service_version=sample.service_version,
        )

        await _publish(bus, settings.TELEMETRY_PUBLISH_CHANNEL, "biometrics.telemetry", raw_payload)
        await _publish(bus, settings.BIOMETRICS_SAMPLE_CHANNEL, "biometrics.sample.v1", sample)
        await _publish(bus, settings.BIOMETRICS_SUMMARY_CHANNEL, "biometrics.summary.v1", summary)
        await _publish(bus, settings.BIOMETRICS_INDUCTION_CHANNEL, "biometrics.induction.v1", induction)
        logger.debug("Published biometrics sample/summary/induction")
    except Exception as exc:
        logger.error(f"Failed to publish biometrics: {exc}")


async def _publish(bus: OrionBusAsync, channel: str, kind: str, payload: object) -> None:
    env = BaseEnvelope(
        kind=kind,
        source=_source(),
        payload=(payload.model_dump(mode="json") if hasattr(payload, "model_dump") else payload),
    )
    await bus.publish(channel, env)


class BiometricsWorker(Clock):
    def __init__(self, cfg: ChassisConfig, *, interval_sec: float):
        super().__init__(cfg, interval_sec=interval_sec, tick=self.do_tick)

    async def do_tick(self) -> None:
        await publish_metrics(self.bus)


class BiometricsHub:
    def __init__(self) -> None:
        self._latest_summary: Dict[str, BiometricsSummaryV1] = {}
        self._latest_induction: Dict[str, BiometricsInductionV1] = {}

    async def handle_biometrics(self, env: BaseEnvelope) -> None:
        payload_obj = env.payload
        if hasattr(payload_obj, "model_dump"):
            payload_obj = payload_obj.model_dump(mode="json")
        if not isinstance(payload_obj, dict):
            return
        if env.kind == "biometrics.summary.v1":
            summary = BiometricsSummaryV1.model_validate(payload_obj)
            if summary.node:
                self._latest_summary[summary.node] = summary
        elif env.kind == "biometrics.induction.v1":
            induction = BiometricsInductionV1.model_validate(payload_obj)
            if induction.node:
                self._latest_induction[induction.node] = induction

    async def publish_cluster(self, bus: OrionBusAsync) -> None:
        if not self._latest_summary:
            return
        weights = settings.role_weights
        weighted_pressures = {}
        weighted_headroom = {}
        weighted_composites = {}
        sources = []
        weight_total = 0.0

        for node, summary in self._latest_summary.items():
            role = "atlas" if "atlas" in node else "athena" if "athena" in node else "other"
            weight = float(weights.get(role, 1.0))
            weight_total += weight
            sources.append(node)
            for k, v in summary.pressures.items():
                weighted_pressures[k] = weighted_pressures.get(k, 0.0) + weight * float(v)
            for k, v in summary.headroom.items():
                weighted_headroom[k] = weighted_headroom.get(k, 0.0) + weight * float(v)
            for k, v in summary.composites.items():
                weighted_composites[k] = weighted_composites.get(k, 0.0) + weight * float(v)

        if weight_total <= 0:
            weight_total = 1.0

        pressures = {k: min(1.0, v / weight_total) for k, v in weighted_pressures.items()}
        headroom = {k: min(1.0, v / weight_total) for k, v in weighted_headroom.items()}
        composites = {k: min(1.0, v / weight_total) for k, v in weighted_composites.items()}

        constraint = "NONE"
        if pressures:
            key, value = max(pressures.items(), key=lambda item: item[1])
            if value >= 0.7:
                constraint = key.upper()

        cluster = BiometricsClusterV1(
            timestamp=datetime.now(timezone.utc),
            sources=sources,
            role_weights=weights,
            pressures=pressures,
            headroom=headroom,
            composites=composites,
            constraint=constraint,
        )
        await _publish(bus, settings.BIOMETRICS_CLUSTER_CHANNEL, "biometrics.cluster.v1", cluster)

        strain = composites.get("strain", 0.0)
        signal = SparkSignalV1(
            signal_type="resource",
            intensity=float(strain),
            as_of_ts=datetime.now(timezone.utc),
            ttl_ms=settings.SPARK_SIGNAL_TTL_MS,
            source_service=settings.SERVICE_NAME,
            source_node=settings.NODE_NAME,
        )
        signal_env = BaseEnvelope(kind="spark.signal.v1", source=_source(), payload=signal.model_dump(mode="json"))
        await bus.publish(settings.SPARK_SIGNAL_CHANNEL, signal_env)


class BiometricsHubWorker(Clock):
    def __init__(self, cfg: ChassisConfig, hub: BiometricsHub, *, interval_sec: float):
        super().__init__(cfg, interval_sec=interval_sec, tick=self.do_tick)
        self.hub = hub

    async def do_tick(self) -> None:
        await self.hub.publish_cluster(self.bus)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"🚀 Starting {settings.SERVICE_NAME} v{settings.SERVICE_VERSION} (node={settings.NODE_NAME})")

    workers = []
    hunter_task: Optional[asyncio.Task] = None
    hunter_stop = asyncio.Event()

    if settings.BIOMETRICS_MODE in {"agent", "both"}:
        metrics_worker = BiometricsWorker(chassis_cfg(), interval_sec=settings.TELEMETRY_INTERVAL)
        await metrics_worker.start_background()
        workers.append(metrics_worker)

    if settings.BIOMETRICS_MODE in {"hub", "both"}:
        hub = BiometricsHub()
        hunter = Hunter(
            chassis_cfg(),
            patterns=[settings.BIOMETRICS_SUMMARY_CHANNEL, settings.BIOMETRICS_INDUCTION_CHANNEL],
            handler=hub.handle_biometrics,
        )
        hunter_task = asyncio.create_task(hunter.start_background(hunter_stop))
        hub_worker = BiometricsHubWorker(chassis_cfg(), hub, interval_sec=settings.CLUSTER_PUBLISH_INTERVAL)
        await hub_worker.start_background()
        workers.append(hub_worker)

    try:
        yield
    finally:
        hunter_stop.set()
        for worker in workers:
            await worker.stop()
        if hunter_task:
            try:
                await asyncio.wait_for(hunter_task, timeout=2.0)
            except Exception:
                hunter_task.cancel()


app = FastAPI(title=settings.SERVICE_NAME, version=settings.SERVICE_VERSION, lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "telemetry_publish_channel": settings.TELEMETRY_PUBLISH_CHANNEL,
        "sample_channel": settings.BIOMETRICS_SAMPLE_CHANNEL,
        "summary_channel": settings.BIOMETRICS_SUMMARY_CHANNEL,
        "induction_channel": settings.BIOMETRICS_INDUCTION_CHANNEL,
        "cluster_channel": settings.BIOMETRICS_CLUSTER_CHANNEL,
        "bus_url": settings.ORION_BUS_URL,
        "node": settings.NODE_NAME,
        "mode": settings.BIOMETRICS_MODE,
    }
