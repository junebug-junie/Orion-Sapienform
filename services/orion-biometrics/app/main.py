import asyncio
import logging
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Deque, Dict, Optional

from fastapi import FastAPI, Query

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_service_chassis import ChassisConfig, Clock, Hunter

from app.power_intent import sample_gpu_watts, settle
from orion.schemas.power import PowerIntentV1
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.telemetry.biometrics import (
    BiometricsPayload,
    BiometricsSampleV1,
    BiometricsSummaryV1,
    BiometricsInductionV1,
    BiometricsClusterV1,
)
from orion.schemas.telemetry.spark_signal import SparkSignalV1
from app.ambient_audio_snapshot import load_ambient_audio_snapshot
from app.ambient_spike_detector import AmbientSpikeDetector, AmbientSpikeDetectorConfig
from app.cabinet_snapshot import load_cabinet_sensors_snapshot
from app.metrics import collect_biometrics, collect_disk_capacity
from app.ilo import IloPoller
from app.pdu import PduPoller, parse_outlets, parse_proxy_outlets
from orion.telemetry.biometrics_pipeline import (
    BiometricsPipeline,
    PipelineConfig,
    aggregate_fleet_measurements,
)
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


_ilo_poller = IloPoller(
    settings.ILO_HOST,
    settings.ILO_USERNAME,
    settings.ILO_PASSWORD,
    interval_sec=settings.ILO_POLL_INTERVAL_SEC,
    timeout_sec=settings.ILO_REQUEST_TIMEOUT_SEC,
)

# Per-outlet power from the rack PDU. This is what gives circe a chassis figure at all: it
# has no reachable BMC, so `_ilo_poller` is permanently `not_configured` there and every fleet
# total has carried `measurements_missing: {"chassis_watts": ["circe"]}`. Configured per node
# with that node's own outlets, exactly like ILO_HOST -- a node not on a metered PDU (athena)
# leaves PDU_OUTLETS empty and the poller stays disabled.
_pdu_poller = PduPoller(
    settings.PDU_HOST,
    parse_outlets(settings.PDU_OUTLETS),
    community=settings.PDU_SNMP_COMMUNITY,
    port=settings.PDU_SNMP_PORT,
    interval_sec=settings.PDU_POLL_INTERVAL_SEC,
    timeout_sec=settings.PDU_REQUEST_TIMEOUT_SEC,
)


# Outlets the HUB polls on behalf of nodes that cannot reach the PDU themselves.
# circe's NIC is dead: it reaches the bus over Tailscale but has no LAN path to the PDU, so its
# own poller fails every 65 s. athena can reach it. See parse_proxy_outlets for the full story.
_pdu_proxy_pollers: Dict[str, PduPoller] = {
    node: PduPoller(
        settings.PDU_HOST,
        outlets,
        community=settings.PDU_SNMP_COMMUNITY,
        port=settings.PDU_SNMP_PORT,
        interval_sec=settings.PDU_POLL_INTERVAL_SEC,
        timeout_sec=settings.PDU_REQUEST_TIMEOUT_SEC,
    )
    for node, outlets in parse_proxy_outlets(settings.PDU_PROXY_OUTLETS).items()
}


def _proxy_measurements() -> Dict[str, Dict[str, float]]:
    """Per-node measurements this hub read on another node's behalf.

    Returns only nodes with a live reading. A failed or not-yet-polled proxy contributes
    nothing, so the node stays in `measurements_missing` exactly as before -- a proxy that is
    itself broken must not look like a measurement.
    """
    out: Dict[str, Dict[str, float]] = {}
    for node, poller in _pdu_proxy_pollers.items():
        detail = poller.details()
        watts = detail.get("pdu_watts")
        if isinstance(watts, (int, float)) and not isinstance(watts, bool) and watts >= 0.0:
            out[node] = {"chassis_watts": float(watts), "pdu_watts": float(watts)}
    return out


def _heartbeat_details() -> Dict[str, Any]:
    """Extra data folded into every SystemHealthV1 heartbeat's `details` dict.

    Real host disk-capacity telemetry plus (when this node has iLO/BMC
    credentials configured) real thermal/fan/power telemetry, piggybacked onto
    the existing bus-native heartbeat rather than a new channel/service (see
    docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md).
    Node-level, not per-service -- once this same code is redeployed on atlas/circe
    (each already an independent orion-biometrics deployment publishing its own
    per-node heartbeat, confirmed live via `orion:system:health`), each node reports
    its own mounts/BMC with no new SSH/credential surface. `_ilo_poller.details()`
    reads a background-cached snapshot (see IloPoller docstring) rather than making
    a live network call here -- this function runs inside the heartbeat's own
    bounded per-tick budget, too tight for a real BMC round-trip.
    """
    return {
        **collect_disk_capacity(settings.DISK_CAPACITY_MOUNTS),
        **_ilo_poller.details(),
    }


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

from app.node_catalog import NodeCatalog
from app.grammar_emit import build_biometrics_node_grammar_events
_NODE_CATALOG = NodeCatalog.load(settings.NODE_CATALOG_PATH)
_AMBIENT_SPIKE_DETECTOR = AmbientSpikeDetector(
    AmbientSpikeDetectorConfig(
        activity_threshold=settings.AMBIENT_AUDIO_SPIKE_ACTIVITY_THRESHOLD,
        consecutive_ticks=settings.AMBIENT_AUDIO_SPIKE_CONSECUTIVE_TICKS,
        cooldown_sec=settings.AMBIENT_AUDIO_SPIKE_COOLDOWN_SEC,
    )
)


async def publish_metrics(bus: OrionBusAsync) -> None:
    if not settings.ORION_BUS_ENABLED:
        return

    try:
        sample_data = collect_biometrics()
        sensors = load_cabinet_sensors_snapshot(
            settings.CABINET_SENSORS_PATH,
            stale_after_sec=settings.CABINET_SENSOR_STALE_AFTER_SEC,
        )
        if sensors is not None:
            sample_data["sensors"] = sensors
        ambient_audio = load_ambient_audio_snapshot(
            settings.AMBIENT_AUDIO_PATH,
            stale_after_sec=settings.AMBIENT_AUDIO_STALE_AFTER_SEC,
        )
        if ambient_audio is not None:
            sample_data["ambient_audio"] = ambient_audio
        sample = BiometricsSampleV1.model_validate(sample_data)
        # BiometricsSampleV1 has extra="ignore" -- these two keys would be silently
        # dropped by model_validate() above, so they're merged into the pipeline's
        # own input dict after validation, not before. They never reach the bus
        # (raw_payload/sample below are built from `sample`, not this dict), so this
        # doesn't change what's published on BIOMETRICS_SAMPLE_CHANNEL.
        pipeline_input = sample.model_dump(mode="python")
        pipeline_input["ilo"] = _ilo_poller.details()
        pipeline_input["pdu"] = _pdu_poller.details()
        pipeline_input["disk_capacity"] = collect_disk_capacity(settings.DISK_CAPACITY_MOUNTS)
        summary, induction = _pipeline.update(pipeline_input)

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
        if settings.AMBIENT_AUDIO_SPIKE_ENABLED:
            try:
                spike = _AMBIENT_SPIKE_DETECTOR.observe(
                    node=sample.node or summary.node or settings.NODE_NAME,
                    timestamp=summary.timestamp,
                    summary=summary,
                    source_service=settings.SERVICE_NAME,
                    source_node=settings.NODE_NAME,
                )
                if spike is not None:
                    await _publish(
                        bus,
                        settings.CABINET_AMBIENT_SPIKE_CHANNEL,
                        "cabinet.ambient.spike.v1",
                        spike,
                    )
            except Exception as exc:
                logger.warning(
                    "Failed to publish cabinet ambient spike: %s",
                    exc,
                    exc_info=True,
                )
        if settings.PUBLISH_BIOMETRICS_GRAMMAR:
            try:
                node_profile = _NODE_CATALOG.resolve(
                    sample.node or summary.node or induction.node
                )
                grammar_events = build_biometrics_node_grammar_events(
                    sample=sample,
                    summary=summary,
                    induction=induction,
                    node_profile=node_profile,
                    source_channel=settings.BIOMETRICS_INDUCTION_CHANNEL,
                    code_version=settings.SERVICE_VERSION,
                )
                for event in grammar_events:
                    await _publish(
                        bus,
                        settings.GRAMMAR_EVENT_CHANNEL,
                        "grammar.event.v1",
                        event,
                    )
            except Exception as exc:
                logger.warning(
                    "Failed to publish biometrics grammar events: %s",
                    exc,
                    exc_info=True,
                )
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
        super().__init__(cfg, interval_sec=interval_sec, tick=self.do_tick, heartbeat_details=_heartbeat_details)

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
            # atlas is decommissioned (2026-08-21, see config/biometrics/node_catalog.yaml).
            # circe took over its role as the fleet's dominant, always-up GPU workhorse (6
            # GPUs after absorbing atlas's 2x V100 16GB + athena's old P100) -- inherits its
            # 0.7 weight below instead of falling into the generic "other":0.5 bucket.
            role = "circe" if "circe" in node else "athena" if "athena" in node else "other"
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

        # Raw physical units aggregate separately from the pressures above: watts are summed
        # in watts, not weighted-averaged and clamped to 1.0. `measurements_missing` travels
        # with the totals so a partial fleet sum cannot be read as a complete one.
        # Merge in anything this hub measured on another node's behalf, BEFORE aggregating, so
        # a proxied node stops being counted as missing.
        #
        # PROVENANCE IS NOT OPTIONAL HERE. `chassis_watts` has been a self-report -- the node
        # that owns the number measures it. A proxy reading breaks that, and a future reader
        # finding circe with chassis_watts would otherwise conclude its BMC came back. So the
        # cluster says who measured what, and a proxied value NEVER silently overwrites a
        # node's own reading: self-report wins, proxy only fills a gap.
        per_node = {node: summary.measurements for node, summary in self._latest_summary.items()}
        proxied: Dict[str, list] = {}
        for node, values in _proxy_measurements().items():
            own = per_node.get(node) or {}
            filled = {k: v for k, v in values.items() if own.get(k) is None}
            if not filled:
                continue
            per_node[node] = {**own, **filled}
            proxied[node] = sorted(filled)

        fleet_measurements, fleet_missing = aggregate_fleet_measurements(per_node)

        # Which known nodes are absent entirely. `measurements_missing` only names nodes that
        # DID report and lacked a key -- a node that stops publishing never reaches `sources`
        # and would otherwise vanish without trace, leaving a partial fleet total looking whole.
        contributed = {
            _NODE_CATALOG.resolve(node).node_id for node in self._latest_summary
        }
        nodes_absent = sorted(
            node_id for node_id in _NODE_CATALOG.profiles if node_id not in contributed
        )

        # Fleet binding constraint: max over nodes, NOT the role-weighted mean above. One
        # saturated machine must not be averaged into looking half-busy, and the node has to
        # travel with the number or it cannot be acted on.
        peak_value: float | None = None
        peak_channel: str | None = None
        peak_node: str | None = None
        for node, summary in self._latest_summary.items():
            value = summary.peak_pressure
            if value is None:
                continue
            if peak_value is None or value > peak_value:
                peak_value, peak_channel, peak_node = value, summary.peak_pressure_channel, node

        cluster = BiometricsClusterV1(
            timestamp=datetime.now(timezone.utc),
            sources=sources,
            nodes_absent=nodes_absent or None,
            peak_pressure=peak_value,
            peak_pressure_channel=peak_channel,
            peak_pressure_node=peak_node,
            role_weights=weights,
            pressures=pressures,
            headroom=headroom,
            composites=composites,
            constraint=constraint,
            measurements=fleet_measurements or None,
            measurements_missing=fleet_missing or None,
            measurements_proxied=proxied or None,
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
        super().__init__(cfg, interval_sec=interval_sec, tick=self.do_tick, heartbeat_details=_heartbeat_details)
        self.hub = hub

    async def do_tick(self) -> None:
        await self.hub.publish_cluster(self.bus)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"🚀 Starting {settings.SERVICE_NAME} v{settings.SERVICE_VERSION} (node={settings.NODE_NAME})")

    workers = []
    hunter_task: Optional[asyncio.Task] = None
    hunter_stop = asyncio.Event()

    await _ilo_poller.start_background()
    await _pdu_poller.start_background()
    for _proxy in _pdu_proxy_pollers.values():
        await _proxy.start_background()

    if settings.BIOMETRICS_MODE in {"agent", "both"}:
        metrics_worker = BiometricsWorker(chassis_cfg(), interval_sec=settings.TELEMETRY_INTERVAL)
        await metrics_worker.start_background()
        workers.append(metrics_worker)

    # Power intent settlement. Runs in EVERY mode, not just hub: the settler must be on
    # the node that physically owns the GPU, and only agent-mode deployments run there.
    # An intent for another node is ignored, so all nodes can subscribe safely.
    if settings.POWER_INTENT_SETTLE_ENABLED:

        async def _handle_power_intent(env: BaseEnvelope) -> None:
            payload = env.payload
            if hasattr(payload, "model_dump"):
                payload = payload.model_dump(mode="json")
            if not isinstance(payload, dict):
                return
            try:
                intent = PowerIntentV1.model_validate(payload)
            except Exception:
                logger.exception("power_intent_invalid_payload")
                return
            if intent.node != settings.NODE_NAME:
                return  # another node's meter, not ours to read

            async def _run() -> None:
                try:
                    settled = await settle(
                        intent,
                        sampler=sample_gpu_watts,
                        sample_interval_sec=settings.POWER_INTENT_SAMPLE_INTERVAL_SEC,
                    )
                except Exception:
                    logger.exception("power_intent_settle_failed intent=%s", intent.intent_id)
                    return
                await _publish(
                    bus,
                    settings.POWER_INTENT_SETTLED_CHANNEL,
                    "power.intent.settled.v1",
                    settled,
                )

            # Fire-and-forget: the window must not block the intent consumer, or a
            # second workload declaring mid-window would queue behind the first.
            asyncio.create_task(_run())

        power_hunter = Hunter(
            chassis_cfg(),
            patterns=[settings.POWER_INTENT_CHANNEL],
            handler=_handle_power_intent,
        )
        asyncio.create_task(power_hunter.start_background(hunter_stop))

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
        await _ilo_poller.stop()
        await _pdu_poller.stop()
        for _proxy in _pdu_proxy_pollers.values():
            await _proxy.stop()


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


def _power_telemetry_health() -> Dict[str, Any]:
    """Is this node's power telemetry wired, and if not, why not.

    Both pollers are OPTIONAL and both go quiet by design -- a node with no BMC, or no PDU
    outlets, reports nothing rather than a fake zero. That is the right behaviour for the DATA
    and it is exactly what makes the INSTRUMENT undiagnosable: "configured and working",
    "configured but unreachable", and "running an image that predates the feature" all look
    identical from outside the container.

    That cost was paid twice for real. circe's BMC has been unreachable long enough that nobody
    knows when it broke. And the day the PDU poller shipped, answering "does atlas have it
    configured, or is it failing, or is it on an older image?" took a round trip of
    `docker exec` on a remote host, because nothing exposed the answer.

    `absent` remains the honest reading of a measurement. It is not an acceptable answer to
    "is the instrument installed and healthy" -- that is a different question and it needs its
    own surface. One `curl /health` from anywhere now answers it.
    """
    ilo_detail = _ilo_poller.details()
    pdu_detail = _pdu_poller.details()

    def _state(enabled: bool, detail: Dict[str, Any], value_key: str) -> Dict[str, Any]:
        if not enabled:
            # Not a fault. Most nodes legitimately have no BMC or no metered outlets.
            return {"configured": False, "healthy": None, "reason": "not_configured"}
        error = detail.get(f"{value_key}_error") or detail.get("error")
        if error:
            return {"configured": True, "healthy": False, "reason": str(error)}
        has_value = any(k.endswith("_watts") and detail.get(k) is not None for k in detail)
        return {
            "configured": True,
            # None, not False, before the first poll completes: "has not reported yet" is not
            # the same as "is broken", and a health check that conflates them cries wolf on
            # every restart.
            "healthy": True if has_value else None,
            "reason": None if has_value else "no_reading_yet",
        }

    return {
        "ilo": _state(_ilo_poller.enabled, ilo_detail, "ilo"),
        "pdu": {
            **_state(_pdu_poller.enabled, pdu_detail, "pdu"),
            # Echoed so a misconfigured outlet map is visible without shelling into the host --
            # the map is physical cabling that nothing can verify automatically.
            "outlets": parse_outlets(settings.PDU_OUTLETS),
            "host": settings.PDU_HOST or None,
        },
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "status": "ok",
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "telemetry_publish_channel": settings.TELEMETRY_PUBLISH_CHANNEL,
        "sample_channel": settings.BIOMETRICS_SAMPLE_CHANNEL,
        "summary_channel": settings.BIOMETRICS_SUMMARY_CHANNEL,
        "induction_channel": settings.BIOMETRICS_INDUCTION_CHANNEL,
        "cluster_channel": settings.BIOMETRICS_CLUSTER_CHANNEL,
        "grammar_event_channel": settings.GRAMMAR_EVENT_CHANNEL,
        "publish_biometrics_grammar": settings.PUBLISH_BIOMETRICS_GRAMMAR,
        "bus_url": settings.ORION_BUS_URL,
        "node": settings.NODE_NAME,
        "mode": settings.BIOMETRICS_MODE,
        "power_telemetry": _power_telemetry_health(),
    }
