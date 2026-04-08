import asyncio
import logging
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Deque, Dict, Optional

from fastapi import FastAPI, Query

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


_RAW_RECENT: Deque[Dict[str, Any]] = deque(maxlen=120)
_SUMMARY_BY_NODE: Dict[str, Dict[str, Any]] = {}
_INDUCTION_BY_NODE: Dict[str, Dict[str, Any]] = {}
_CLUSTER: Optional[Dict[str, Any]] = None


def _iso(ts: Optional[datetime]) -> Optional[str]:
    if ts is None:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.isoformat()


def _freshness_seconds(ts: Optional[datetime]) -> Optional[float]:
    if ts is None:
        return None
    return max(0.0, (datetime.now(timezone.utc) - ts.astimezone(timezone.utc)).total_seconds())


def _status_for_freshness(freshness_s: Optional[float]) -> tuple[str, str]:
    if freshness_s is None:
        return "NO_SIGNAL", "no_recent_samples"
    if freshness_s > 300:
        return "NO_SIGNAL", "stale_over_no_signal_threshold"
    if freshness_s > 90:
        return "STALE", "stale_over_threshold"
    return "OK", "fresh"


def _cluster_trend(induction_by_node: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    if not induction_by_node:
        return {}
    totals: Dict[str, Dict[str, float]] = {}
    count = 0
    for item in induction_by_node.values():
        metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
        if not metrics:
            continue
        count += 1
        for name, data in metrics.items():
            if not isinstance(data, dict):
                continue
            bucket = totals.setdefault(name, {"trend": 0.0, "volatility": 0.0, "spike": 0.0})
            for key in ("trend", "volatility", "spike"):
                try:
                    bucket[key] += float(data.get(key) or 0.0)
                except Exception:
                    pass
    if count <= 0:
        return {}
    return {name: {key: min(1.0, value / count) for key, value in data.items()} for name, data in totals.items()}


def _constraint_from_snapshot(summary_by_node: Dict[str, Dict[str, Any]], cluster: Optional[Dict[str, Any]]) -> str:
    if cluster and cluster.get("constraint"):
        return str(cluster.get("constraint"))
    best_name = "NONE"
    best_value = 0.0
    for item in summary_by_node.values():
        pressures = item.get("pressures") if isinstance(item.get("pressures"), dict) else {}
        for name, value in pressures.items():
            try:
                val = float(value)
            except Exception:
                continue
            if val > best_value:
                best_name = str(name).upper()
                best_value = val
    return best_name if best_value >= 0.7 else "NONE"


def _build_snapshot_payload() -> Dict[str, Any]:
    nodes: Dict[str, Any] = {}
    latest_ts: Optional[datetime] = None
    all_nodes = sorted(set(_SUMMARY_BY_NODE.keys()) | set(_INDUCTION_BY_NODE.keys()))
    for node in all_nodes:
        summary = _SUMMARY_BY_NODE.get(node)
        induction = _INDUCTION_BY_NODE.get(node)
        summary_ts = summary.get("timestamp") if isinstance(summary, dict) else None
        induction_ts = induction.get("timestamp") if isinstance(induction, dict) else None
        candidates = [ts for ts in [summary_ts, induction_ts] if isinstance(ts, datetime)]
        node_ts = max(candidates) if candidates else None
        freshness_s = _freshness_seconds(node_ts)
        status, reason = _status_for_freshness(freshness_s)
        if node_ts and (latest_ts is None or node_ts > latest_ts):
            latest_ts = node_ts
        nodes[node] = {
            "summary": summary.get("payload") if isinstance(summary, dict) else None,
            "induction": induction.get("payload") if isinstance(induction, dict) else None,
            "as_of": _iso(node_ts),
            "freshness_s": freshness_s,
            "status": status,
            "reason": reason,
        }

    cluster_payload = _CLUSTER.get("payload") if isinstance(_CLUSTER, dict) else None
    cluster_ts = _CLUSTER.get("timestamp") if isinstance(_CLUSTER, dict) else None
    as_of_ts = cluster_ts or latest_ts
    freshness_s = _freshness_seconds(as_of_ts)
    status, reason = _status_for_freshness(freshness_s)
    return {
        "status": status,
        "reason": reason,
        "as_of": _iso(as_of_ts),
        "freshness_s": freshness_s,
        "constraint": _constraint_from_snapshot({k: v.get("payload") or {} for k, v in _SUMMARY_BY_NODE.items()}, cluster_payload),
        "cluster": {
            "composite": (cluster_payload or {}).get("composites") if isinstance(cluster_payload, dict) else {},
            "trend": _cluster_trend({k: v.get("payload") or {} for k, v in _INDUCTION_BY_NODE.items()}),
        },
        "nodes": nodes,
    }


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

        _RAW_RECENT.append({
            "timestamp": sample.timestamp,
            "node": sample.node,
            "raw": raw_payload.model_dump(mode="json"),
            "sample": sample.model_dump(mode="json"),
        })
        _SUMMARY_BY_NODE[sample.node] = {"payload": summary.model_dump(mode="json"), "timestamp": summary.timestamp}
        _INDUCTION_BY_NODE[sample.node] = {"payload": induction.model_dump(mode="json"), "timestamp": induction.timestamp}

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
                _SUMMARY_BY_NODE[summary.node] = {"payload": summary.model_dump(mode="json"), "timestamp": summary.timestamp}
        elif env.kind == "biometrics.induction.v1":
            induction = BiometricsInductionV1.model_validate(payload_obj)
            if induction.node:
                self._latest_induction[induction.node] = induction
                _INDUCTION_BY_NODE[induction.node] = {"payload": induction.model_dump(mode="json"), "timestamp": induction.timestamp}

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
        global _CLUSTER
        _CLUSTER = {"payload": cluster.model_dump(mode="json"), "timestamp": cluster.timestamp}
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


@app.get("/snapshot")
def snapshot() -> Dict[str, Any]:
    return _build_snapshot_payload()


@app.get("/raw/recent")
def raw_recent(limit: int = Query(10, ge=1, le=100), node: Optional[str] = Query(None)) -> Dict[str, Any]:
    items = []
    for item in reversed(list(_RAW_RECENT)):
        if node and str(item.get("node") or "") != str(node):
            continue
        items.append({
            "timestamp": _iso(item.get("timestamp")),
            "node": item.get("node"),
            "raw": item.get("raw"),
            "sample": item.get("sample"),
        })
        if len(items) >= limit:
            break
    return {"items": items, "count": len(items)}


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
