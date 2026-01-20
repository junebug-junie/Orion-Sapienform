from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import orjson
from redis.asyncio import Redis

from orion.schemas.telemetry.spark import SparkStateSnapshotV1
from orion.schemas.telemetry.biometrics import BiometricsSummaryV1, BiometricsInductionV1, BiometricsClusterV1
from orion.schemas.state.contracts import StateLatestReply, StateGetLatestRequest, BiometricsContext


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _age_ms(as_of: datetime) -> int:
    return int((_utcnow() - as_of).total_seconds() * 1000)


def _status(snapshot: Optional[SparkStateSnapshotV1]) -> Tuple[str, Optional[int]]:
    if snapshot is None:
        return "missing", None
    try:
        age = _age_ms(snapshot.snapshot_ts)
        return ("fresh" if age <= int(snapshot.valid_for_ms) else "stale"), age
    except Exception:
        return "stale", None


def _status_for_timestamp(ts: Optional[datetime], *, stale_after_sec: float) -> Tuple[str, Optional[int]]:
    if ts is None:
        return "missing", None
    age = _age_ms(ts)
    return ("fresh" if age <= int(stale_after_sec * 1000) else "stale"), age


@dataclass
class CacheKeys:
    prefix: str

    @property
    def global_key(self) -> str:
        return f"{self.prefix}:latest:global"

    def node_key(self, node: str) -> str:
        return f"{self.prefix}:latest:node:{node}"


class StateStore:
    """In-memory + Redis cache for SparkStateSnapshotV1."""

    def __init__(self, *, redis: Redis, key_prefix: str, primary_node: str):
        self.redis = redis
        self.keys = CacheKeys(prefix=key_prefix)
        self.primary_node = primary_node

        self._lock = asyncio.Lock()
        self._latest_by_node: Dict[str, SparkStateSnapshotV1] = {}
        self._latest_global: Optional[SparkStateSnapshotV1] = None

        # dedupe/out-of-order protection
        self._max_seq_by_boot: Dict[str, int] = {}

        # Biometrics summaries/induction
        self._biometrics_summary_by_node: Dict[str, BiometricsSummaryV1] = {}
        self._biometrics_induction_by_node: Dict[str, BiometricsInductionV1] = {}
        self._biometrics_cluster: Optional[BiometricsClusterV1] = None

    async def hydrate_from_snapshots(self, snapshots: list[SparkStateSnapshotV1], *, note: str) -> int:
        accepted = 0
        for snap in snapshots:
            if await self.ingest_snapshot(snap, note=note, write_cache=True):
                accepted += 1
        return accepted

    async def ingest_snapshot(
        self,
        snap: SparkStateSnapshotV1,
        *,
        note: str = "",
        write_cache: bool = True,
    ) -> bool:
        """Ingest snapshot; returns True if accepted (newer than what we have)."""
        node = (snap.source_node or self.primary_node or "unknown").strip() or "unknown"
        boot = str(snap.producer_boot_id)
        seq = int(snap.seq)

        async with self._lock:
            max_seq = self._max_seq_by_boot.get(boot, -1)
            if seq <= max_seq:
                # duplicate or out-of-order
                return False

            self._max_seq_by_boot[boot] = seq
            self._latest_by_node[node] = snap

            # global rollup: authoritative node wins if configured, else most recent
            if node == self.primary_node:
                self._latest_global = snap
            else:
                if self._latest_global is None:
                    self._latest_global = snap
                else:
                    try:
                        if snap.snapshot_ts >= self._latest_global.snapshot_ts:
                            self._latest_global = snap
                    except Exception:
                        self._latest_global = snap

        if write_cache:
            await self._write_cache(node=node, snap=snap, note=note)

        return True

    async def _write_cache(self, *, node: str, snap: SparkStateSnapshotV1, note: str) -> None:
        payload = {
            "snapshot": snap.model_dump(mode="json"),
            "ingested_at": _utcnow().isoformat(),
            "note": note,
        }
        data = orjson.dumps(payload)
        try:
            await self.redis.set(self.keys.node_key(node), data)
            # global key follows current computed global snapshot
            async with self._lock:
                g = self._latest_global
            if g is not None:
                g_payload = {
                    "snapshot": g.model_dump(mode="json"),
                    "ingested_at": _utcnow().isoformat(),
                    "note": note,
                }
                await self.redis.set(self.keys.global_key, orjson.dumps(g_payload))
        except Exception:
            # cache is best-effort; do not explode
            return

    async def load_global_from_cache(self) -> bool:
        """Best-effort load of global snapshot from Redis."""
        try:
            raw = await self.redis.get(self.keys.global_key)
            if not raw:
                return False
            obj = orjson.loads(raw)
            snap = SparkStateSnapshotV1.model_validate(obj.get("snapshot") or {})
            async with self._lock:
                self._latest_global = snap
            return True
        except Exception:
            return False

    async def ingest_biometrics_summary(self, summary: BiometricsSummaryV1) -> None:
        node = (summary.node or self.primary_node or "unknown").strip() or "unknown"
        async with self._lock:
            self._biometrics_summary_by_node[node] = summary

    async def ingest_biometrics_induction(self, induction: BiometricsInductionV1) -> None:
        node = (induction.node or self.primary_node or "unknown").strip() or "unknown"
        async with self._lock:
            self._biometrics_induction_by_node[node] = induction

    async def ingest_biometrics_cluster(self, cluster: BiometricsClusterV1) -> None:
        async with self._lock:
            self._biometrics_cluster = cluster

    def _pick_latest(self, items: Dict[str, Any]) -> Optional[Any]:
        if not items:
            return None
        primary = items.get(self.primary_node)
        if primary is not None:
            return primary
        return max(items.values(), key=lambda v: getattr(v, "timestamp", _utcnow()))

    async def get_latest(self, req: StateGetLatestRequest, *, biometrics_stale_after_sec: float) -> StateLatestReply:
        if req.scope == "node":
            node = (req.node or "").strip()
            if not node:
                return StateLatestReply(ok=True, status="missing", note="node_required_for_scope_node")
            async with self._lock:
                snap = self._latest_by_node.get(node)
                summary = self._biometrics_summary_by_node.get(node)
                induction = self._biometrics_induction_by_node.get(node)
                cluster = self._biometrics_cluster
            st, age = _status(snap)
            bio_status, bio_age = _status_for_timestamp(
                summary.timestamp if summary else None,
                stale_after_sec=biometrics_stale_after_sec,
            )
            return StateLatestReply(
                ok=True,
                status=st,
                as_of_ts=(snap.snapshot_ts if snap else None),
                age_ms=age,
                snapshot=snap,
                source="cache",
                biometrics=BiometricsContext(
                    status=bio_status,
                    age_ms=bio_age,
                    note="node",
                    summary=summary,
                    induction=induction,
                    cluster=cluster,
                ),
            )

        # global
        async with self._lock:
            snap = self._latest_global
            summary = self._pick_latest(self._biometrics_summary_by_node)
            induction = self._pick_latest(self._biometrics_induction_by_node)
            cluster = self._biometrics_cluster
        st, age = _status(snap)
        bio_status, bio_age = _status_for_timestamp(
            summary.timestamp if summary else None,
            stale_after_sec=biometrics_stale_after_sec,
        )
        return StateLatestReply(
            ok=True,
            status=st,
            as_of_ts=(snap.snapshot_ts if snap else None),
            age_ms=age,
            snapshot=snap,
            source="cache",
            biometrics=BiometricsContext(
                status=bio_status,
                age_ms=bio_age,
                note="global",
                summary=summary,
                induction=induction,
                cluster=cluster,
            ),
        )

    async def debug_state(self) -> Dict[str, Any]:
        async with self._lock:
            nodes = {k: v.snapshot_ts.isoformat() for k, v in self._latest_by_node.items()}
            g = self._latest_global.snapshot_ts.isoformat() if self._latest_global else None
        return {"nodes": nodes, "global_ts": g, "primary_node": self.primary_node}
