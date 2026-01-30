from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Dict

LEGACY_KINDS = {"spark.introspection.log", "spark.introspection"}
CANONICAL_KINDS = {"spark.telemetry", "spark.state.snapshot.v1"}


@dataclass
class SparkContractMetrics:
    emit_interval_sec: float = 60.0
    counts_total: int = 0
    counts_by_kind: Dict[str, int] = field(default_factory=dict)
    legacy_count: int = 0
    canonical_count: int = 0
    other_spark_count: int = 0
    first_seen_ts: float = field(default_factory=lambda: time.time())
    last_emit_ts: float = field(default_factory=lambda: 0.0)

    def observe(self, kind: str) -> None:
        self.counts_total += 1
        self.counts_by_kind[kind] = self.counts_by_kind.get(kind, 0) + 1
        if kind in LEGACY_KINDS:
            self.legacy_count += 1
        elif kind in CANONICAL_KINDS:
            self.canonical_count += 1
        else:
            self.other_spark_count += 1

    def maybe_emit(self, logger, *, node: str, service: str, now_ts: float | None = None) -> None:
        now = time.time() if now_ts is None else float(now_ts)
        if now - self.last_emit_ts < self.emit_interval_sec:
            return

        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
            "service": service,
            "node": node,
            "since_ts": self.first_seen_ts,
            "window_sec": round(now - self.first_seen_ts, 3),
            "total": self.counts_total,
            "legacy": self.legacy_count,
            "canonical": self.canonical_count,
            "other": self.other_spark_count,
            "by_kind": dict(self.counts_by_kind),
        }
        logger.info("SPARK_CONTRACT_METRICS %s", json.dumps(payload, sort_keys=True))
        self.last_emit_ts = now
