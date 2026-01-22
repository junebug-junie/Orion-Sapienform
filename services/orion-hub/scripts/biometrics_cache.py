from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from orion.core.bus.async_service import OrionBusAsync
from orion.schemas.telemetry.biometrics import (
    BiometricsClusterV1,
    BiometricsInductionV1,
    BiometricsSummaryV1,
)

logger = logging.getLogger("orion-hub.biometrics")

CONSTRAINTS = {
    "thermal": "THERMAL",
    "power": "POWER",
    "gpu_mem": "GPU_MEM",
    "gpu_util": "GPU_UTIL",
    "mem": "MEM",
    "disk": "DISK",
    "net": "NET",
    "containers": "CONTAINERS",
}

CONSTRAINT_PRIORITY = {
    "NONE": 0,
    "NET": 1,
    "DISK": 2,
    "MEM": 3,
    "GPU_UTIL": 4,
    "GPU_MEM": 5,
    "POWER": 6,
    "THERMAL": 7,
    "CONTAINERS": 8,
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _clamp01(value: float | None) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, val))


def _isoformat(ts: Optional[datetime]) -> Optional[str]:
    if not ts:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.isoformat()


class BiometricsCache:
    def __init__(
        self,
        *,
        enabled: bool,
        stale_after_sec: float,
        no_signal_after_sec: float,
        role_weights_json: str,
    ) -> None:
        self.enabled = enabled
        self.stale_after_sec = float(stale_after_sec)
        self.no_signal_after_sec = float(no_signal_after_sec)
        self.role_weights = self._parse_role_weights(role_weights_json)
        self._summary_by_node: Dict[str, Dict[str, Any]] = {}
        self._induction_by_node: Dict[str, Dict[str, Any]] = {}
        self._cluster: Optional[Dict[str, Any]] = None
        self._cluster_ts: Optional[datetime] = None
        self._lock = asyncio.Lock()
        self._task: Optional[asyncio.Task] = None
        self._bus: Optional[OrionBusAsync] = None

    async def start(self, bus: OrionBusAsync) -> None:
        if not self.enabled:
            logger.info("Biometrics cache disabled via settings.")
            return
        if not bus or not bus.enabled:
            logger.warning("Biometrics cache not started (bus unavailable).")
            return
        if self._task and not self._task.done():
            return
        self._bus = bus
        self._task = asyncio.create_task(self._run(), name="hub-biometrics-cache")

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    async def _run(self) -> None:
        if not self._bus:
            return
        channels = [
            "orion:biometrics:summary",
            "orion:biometrics:induction",
            "orion:biometrics:cluster",
        ]
        logger.info("Subscribing to biometrics channels: %s", channels)
        try:
            async with self._bus.subscribe(*channels) as pubsub:
                async for msg in self._bus.iter_messages(pubsub):
                    await self._handle_message(msg)
        except asyncio.CancelledError:
            logger.info("Biometrics cache task cancelled.")
        except Exception as exc:
            logger.error("Biometrics cache loop failed: %s", exc, exc_info=True)

    async def _handle_message(self, msg: Dict[str, Any]) -> None:
        if not self._bus:
            return
        decoded = self._bus.codec.decode(msg.get("data"))
        if not decoded.ok:
            return
        env = decoded.envelope
        payload = env.payload
        timestamp = env.created_at or _utcnow()

        if env.kind == "biometrics.summary.v1":
            try:
                summary = BiometricsSummaryV1.model_validate(payload)
            except Exception:
                return
            node = (summary.node or env.source.node or "unknown").strip() or "unknown"
            async with self._lock:
                self._summary_by_node[node] = {
                    "data": summary.model_dump(mode="json"),
                    "ts": timestamp,
                }
            return

        if env.kind == "biometrics.induction.v1":
            try:
                induction = BiometricsInductionV1.model_validate(payload)
            except Exception:
                return
            node = (induction.node or env.source.node or "unknown").strip() or "unknown"
            async with self._lock:
                self._induction_by_node[node] = {
                    "data": induction.model_dump(mode="json"),
                    "ts": timestamp,
                }
            return

        if env.kind == "biometrics.cluster.v1":
            try:
                cluster = BiometricsClusterV1.model_validate(payload)
            except Exception:
                return
            async with self._lock:
                self._cluster = cluster.model_dump(mode="json")
                self._cluster_ts = timestamp

    async def get_snapshot(self) -> Dict[str, Any]:
        if not self.enabled:
            return self._default_snapshot(reason="disabled")
        async with self._lock:
            summary_by_node = dict(self._summary_by_node)
            induction_by_node = dict(self._induction_by_node)
            cluster = self._cluster.copy() if self._cluster else None
            cluster_ts = self._cluster_ts
        return self._build_snapshot(summary_by_node, induction_by_node, cluster, cluster_ts)

    def _build_snapshot(
        self,
        summary_by_node: Dict[str, Dict[str, Any]],
        induction_by_node: Dict[str, Dict[str, Any]],
        cluster: Optional[Dict[str, Any]],
        cluster_ts: Optional[datetime],
    ) -> Dict[str, Any]:
        now = _utcnow()
        nodes: Dict[str, Any] = {}
        all_nodes = set(summary_by_node.keys()) | set(induction_by_node.keys())
        latest_ts: Optional[datetime] = None

        for node in sorted(all_nodes):
            summary_entry = summary_by_node.get(node)
            induction_entry = induction_by_node.get(node)
            summary_ts = summary_entry.get("ts") if summary_entry else None
            induction_ts = induction_entry.get("ts") if induction_entry else None
            node_ts = max([ts for ts in [summary_ts, induction_ts] if ts], default=None)
            freshness_s = (now - node_ts).total_seconds() if node_ts else None
            status, reason = self._status_for_freshness(
                freshness_s,
                has_signal=node_ts is not None,
                summary=summary_entry.get("data") if summary_entry else None,
            )
            if node_ts and (latest_ts is None or node_ts > latest_ts):
                latest_ts = node_ts
            nodes[node] = {
                "summary": summary_entry.get("data") if summary_entry else None,
                "induction": induction_entry.get("data") if induction_entry else None,
                "as_of": _isoformat(node_ts),
                "freshness_s": freshness_s,
                "status": status,
                "reason": reason,
            }

        overall_ts = cluster_ts or latest_ts
        freshness_s = (now - overall_ts).total_seconds() if overall_ts else None
        status, reason = self._status_for_freshness(
            freshness_s,
            has_signal=overall_ts is not None,
            summary=None,
            nodes=nodes,
        )

        composite = self._cluster_composite(cluster, summary_by_node)
        trend = self._cluster_trend(induction_by_node)
        constraint = self._cluster_constraint(cluster, summary_by_node)

        return {
            "status": status,
            "reason": reason,
            "as_of": _isoformat(overall_ts),
            "freshness_s": freshness_s,
            "constraint": constraint,
            "cluster": {
                "composite": composite,
                "trend": trend,
            },
            "nodes": nodes,
        }

    def _status_for_freshness(
        self,
        freshness_s: Optional[float],
        *,
        has_signal: bool,
        summary: Optional[Dict[str, Any]],
        nodes: Optional[Dict[str, Any]] = None,
    ) -> tuple[str, str]:
        if not has_signal or freshness_s is None:
            return "NO_SIGNAL", "no_recent_samples"
        if freshness_s > self.no_signal_after_sec:
            return "NO_SIGNAL", "stale_over_no_signal_threshold"
        if freshness_s > self.stale_after_sec:
            return "STALE", "stale_over_threshold"
        if summary:
            telemetry_error_rate = _clamp01(summary.get("telemetry_error_rate"))
            if telemetry_error_rate >= 0.5:
                return "DEGRADED", "telemetry_error_rate_high"
        if nodes:
            degraded_nodes = [n for n in nodes.values() if n.get("status") == "DEGRADED"]
            if degraded_nodes:
                return "DEGRADED", "partial_coverage"
        return "OK", "fresh"

    def _cluster_composite(
        self,
        cluster: Optional[Dict[str, Any]],
        summary_by_node: Dict[str, Dict[str, Any]],
    ) -> Dict[str, float]:
        if cluster and isinstance(cluster.get("composites"), dict):
            composites = cluster.get("composites") or {}
            return {
                "strain": _clamp01(composites.get("strain")),
                "homeostasis": _clamp01(composites.get("homeostasis")),
                "stability": _clamp01(composites.get("stability", 1.0)),
            }

        total_weight = 0.0
        strain_sum = 0.0
        stability_sum = 0.0
        homeostasis_vals = []
        for node, entry in summary_by_node.items():
            data = entry.get("data") or {}
            composites = data.get("composites") or {}
            weight = self.role_weights.get(node, 1.0)
            strain = composites.get("strain")
            stability = composites.get("stability")
            homeostasis = composites.get("homeostasis")
            has_value = False
            if strain is not None:
                strain_sum += weight * _clamp01(strain)
                has_value = True
            if stability is not None:
                stability_sum += weight * _clamp01(stability)
                has_value = True
            if homeostasis is not None:
                homeostasis_vals.append(_clamp01(homeostasis))
                has_value = True
            if has_value:
                total_weight += weight
        if total_weight <= 0:
            return {"strain": 0.0, "homeostasis": 0.0, "stability": 1.0}
        strain_avg = strain_sum / total_weight
        stability_avg = stability_sum / total_weight if stability_sum else 1.0
        homeostasis = max(homeostasis_vals) if homeostasis_vals else _clamp01(1.0 - strain_avg)
        return {
            "strain": _clamp01(strain_avg),
            "homeostasis": _clamp01(homeostasis),
            "stability": _clamp01(stability_avg),
        }

    def _cluster_trend(self, induction_by_node: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        metrics = {
            "strain": {},
            "homeostasis": {},
            "stability": {},
        }
        totals = {name: 0.0 for name in metrics}
        sums: Dict[str, Dict[str, float]] = {
            name: {"trend": 0.0, "volatility": 0.0, "spike_rate": 0.0} for name in metrics
        }
        for node, entry in induction_by_node.items():
            data = entry.get("data") or {}
            metric_map = data.get("metrics") or {}
            weight = self.role_weights.get(node, 1.0)
            for name in metrics:
                metric = metric_map.get(name)
                if not isinstance(metric, dict):
                    continue
                totals[name] += weight
                sums[name]["trend"] += weight * _clamp01(metric.get("trend"))
                sums[name]["volatility"] += weight * _clamp01(metric.get("volatility"))
                sums[name]["spike_rate"] += weight * _clamp01(metric.get("spike_rate"))
        for name in metrics:
            if totals[name] <= 0:
                metrics[name] = {"trend": 0.5, "volatility": 0.0, "spike_rate": 0.0}
            else:
                metrics[name] = {
                    "trend": _clamp01(sums[name]["trend"] / totals[name]),
                    "volatility": _clamp01(sums[name]["volatility"] / totals[name]),
                    "spike_rate": _clamp01(sums[name]["spike_rate"] / totals[name]),
                }
        return metrics

    def _cluster_constraint(
        self,
        cluster: Optional[Dict[str, Any]],
        summary_by_node: Dict[str, Dict[str, Any]],
    ) -> str:
        if cluster:
            constraint = cluster.get("constraint")
            if constraint:
                return str(constraint)

        worst_constraint = "NONE"
        worst_score = CONSTRAINT_PRIORITY.get("NONE", 0)
        for entry in summary_by_node.values():
            data = entry.get("data") or {}
            composites = data.get("composites") or {}
            homeostasis = _clamp01(composites.get("homeostasis"))
            summary_constraint = data.get("constraint")
            if homeostasis >= 0.75 and (not summary_constraint or summary_constraint == "NONE"):
                summary_constraint = "THERMAL"
            if not summary_constraint:
                pressures = data.get("pressures") or {}
                if pressures:
                    key, value = max(pressures.items(), key=lambda item: item[1])
                    if value >= 0.7:
                        summary_constraint = CONSTRAINTS.get(key, "NONE")
            if not summary_constraint:
                summary_constraint = "NONE"
            score = CONSTRAINT_PRIORITY.get(str(summary_constraint), 0)
            if score > worst_score:
                worst_score = score
                worst_constraint = str(summary_constraint)
        return worst_constraint

    def _parse_role_weights(self, raw: str) -> Dict[str, float]:
        try:
            parsed = json.loads(raw or "{}")
        except json.JSONDecodeError:
            logger.warning("Invalid BIOMETRICS_ROLE_WEIGHTS_JSON; using defaults.")
            return {}
        if not isinstance(parsed, dict):
            return {}
        weights: Dict[str, float] = {}
        for key, value in parsed.items():
            try:
                weights[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        return weights

    def _default_snapshot(self, *, reason: str) -> Dict[str, Any]:
        return {
            "status": "NO_SIGNAL",
            "reason": reason,
            "as_of": None,
            "freshness_s": None,
            "constraint": "NONE",
            "cluster": {
                "composite": {"strain": 0.0, "homeostasis": 0.0, "stability": 1.0},
                "trend": {
                    "strain": {"trend": 0.5, "volatility": 0.0, "spike_rate": 0.0},
                    "homeostasis": {"trend": 0.5, "volatility": 0.0, "spike_rate": 0.0},
                    "stability": {"trend": 0.5, "volatility": 0.0, "spike_rate": 0.0},
                },
            },
            "nodes": {},
        }
