from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import asyncpg
from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_service_chassis import BaseChassis, ChassisConfig
from orion.core.bus.bus_schemas import BaseEnvelope
from orion.core.bus.codec import OrionCodec
from orion.schemas.telemetry.spark import SparkStateSnapshotV1
from orion.schemas.telemetry.system_health import EquilibriumSnapshotV1

from .settings import settings


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class StateJournaler(BaseChassis):
    def __init__(self) -> None:
        super().__init__(
            ChassisConfig(
                service_name=settings.service_name,
                service_version=settings.service_version,
                node_name=settings.node_name,
                bus_url=settings.orion_bus_url,
                bus_enabled=settings.orion_bus_enabled,
                heartbeat_interval_sec=settings.heartbeat_interval_sec,
            )
        )
        self.bus.codec = OrionCodec()
        self.codec = self.bus.codec
        self.spark_events: List[Dict[str, Any]] = []
        self.distress_events: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()

    async def _ensure_table(self) -> None:
        sql = """
        CREATE TABLE IF NOT EXISTS {table} (
            bucket_ts TIMESTAMPTZ NOT NULL,
            window_sec INT NOT NULL,
            node TEXT NOT NULL DEFAULT '',
            avg_valence DOUBLE PRECISION,
            avg_arousal DOUBLE PRECISION,
            avg_coherence DOUBLE PRECISION,
            avg_novelty DOUBLE PRECISION,
            pct_missing DOUBLE PRECISION,
            pct_stale DOUBLE PRECISION,
            avg_distress DOUBLE PRECISION,
            PRIMARY KEY(bucket_ts, window_sec, node)
        );
        """.format(table=settings.rollup_table)
        conn = await asyncpg.connect(dsn=settings.postgres_uri)
        try:
            await conn.execute(sql)
        finally:
            await conn.close()

    async def _persist_rollup(self, bucket_ts: datetime, window_sec: int, node: str, roll: Dict[str, float]) -> None:
        sql = f"""
        INSERT INTO {settings.rollup_table}
        (bucket_ts, window_sec, node, avg_valence, avg_arousal, avg_coherence, avg_novelty, pct_missing, pct_stale, avg_distress)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        ON CONFLICT (bucket_ts, window_sec, node)
        DO UPDATE SET
            avg_valence=EXCLUDED.avg_valence,
            avg_arousal=EXCLUDED.avg_arousal,
            avg_coherence=EXCLUDED.avg_coherence,
            avg_novelty=EXCLUDED.avg_novelty,
            pct_missing=EXCLUDED.pct_missing,
            pct_stale=EXCLUDED.pct_stale,
            avg_distress=EXCLUDED.avg_distress;
        """
        conn = await asyncpg.connect(dsn=settings.postgres_uri)
        try:
            await conn.execute(
                sql,
                bucket_ts,
                int(window_sec),
                node,
                roll.get("avg_valence"),
                roll.get("avg_arousal"),
                roll.get("avg_coherence"),
                roll.get("avg_novelty"),
                roll.get("pct_missing"),
                roll.get("pct_stale"),
                roll.get("avg_distress"),
            )
        finally:
            await conn.close()

    def _add_spark_snapshot(self, payload: Dict[str, Any]) -> None:
        try:
            snap = SparkStateSnapshotV1.model_validate(payload)
        except Exception:
            return
        ts = snap.snapshot_ts
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        self.spark_events.append(
            {
                "ts": ts,
                "phi": snap.phi,
                "valence": snap.valence,
                "arousal": snap.arousal,
                "coherence": snap.phi.get("coherence", snap.arousal),
                "novelty": snap.phi.get("novelty", 0.0),
                "node": snap.source_node or "",
            }
        )

    def _add_equilibrium_snapshot(self, payload: Dict[str, Any]) -> None:
        try:
            snap = EquilibriumSnapshotV1.model_validate(payload)
        except Exception:
            return
        ts = snap.generated_at
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        self.distress_events.append({"ts": ts, "distress": snap.distress_score})

    def _prune(self, now: datetime) -> None:
        cutoff = now - timedelta(hours=settings.retention_hours)
        self.spark_events = [e for e in self.spark_events if e["ts"] >= cutoff]
        self.distress_events = [e for e in self.distress_events if e["ts"] >= cutoff]

    def _bucket_start(self, now: datetime, window: int) -> datetime:
        epoch = int(now.timestamp())
        bucket_epoch = (epoch // window) * window
        return datetime.fromtimestamp(bucket_epoch, tz=timezone.utc)

    def _compute_rollup(self, now: datetime, window: int) -> Dict[str, float]:
        window_start = now - timedelta(seconds=window)
        events = [e for e in self.spark_events if e["ts"] >= window_start]
        if events:
            valence = [float(e.get("valence", e["phi"].get("valence", 0.0))) for e in events]
            arousal = [float(e.get("arousal", e["phi"].get("energy", 0.0))) for e in events]
            coherence = [float(e.get("coherence", e["phi"].get("coherence", 0.0))) for e in events]
            novelty = [float(e.get("novelty", e["phi"].get("novelty", 0.0))) for e in events]
            avg_valence = sum(valence) / len(valence)
            avg_arousal = sum(arousal) / len(arousal)
            avg_coherence = sum(coherence) / len(coherence)
            avg_novelty = sum(novelty) / len(novelty)
            pct_missing = 0.0
        else:
            avg_valence = avg_arousal = avg_coherence = avg_novelty = 0.0
            pct_missing = 1.0

        distress_events = [d for d in self.distress_events if d["ts"] >= window_start]
        if distress_events:
            avg_distress = sum(float(d["distress"]) for d in distress_events) / len(distress_events)
        else:
            avg_distress = 0.0

        return {
            "avg_valence": avg_valence,
            "avg_arousal": avg_arousal,
            "avg_coherence": avg_coherence,
            "avg_novelty": avg_novelty,
            "pct_missing": pct_missing,
            "pct_stale": 0.0,
            "avg_distress": avg_distress,
        }

    async def _rollup_loop(self) -> None:
        await self._ensure_table()
        while not self._stop.is_set():
            now = _utcnow()
            async with self._lock:
                self._prune(now)
                for window in settings.windows_sec:
                    roll = self._compute_rollup(now, window)
                    bucket_ts = self._bucket_start(now, window)
                    await self._persist_rollup(bucket_ts, window, "", roll)
            await asyncio.sleep(float(settings.rollup_interval_sec))

    async def _run(self) -> None:
        await self._ensure_table()
        roll_task = asyncio.create_task(self._rollup_loop())
        async with self.bus.subscribe(
            settings.channel_spark_state_snapshot,
            settings.channel_equilibrium_snapshot,
        ) as pubsub:
            async for msg in self.bus.iter_messages(pubsub):
                if self._stop.is_set():
                    break
                decoded = self.codec.decode(msg.get("data"))
                if not decoded.ok:
                    continue
                env = decoded.envelope
                payload = env.payload if isinstance(env.payload, dict) else {}
                async with self._lock:
                    if env.kind == "spark.state.snapshot.v1":
                        self._add_spark_snapshot(payload)
                    elif env.kind == "equilibrium.snapshot.v1":
                        self._add_equilibrium_snapshot(payload)

        roll_task.cancel()
        try:
            await roll_task
        except Exception:
            pass
