from __future__ import annotations

import time
from collections import Counter
from typing import Dict


class PadStatsTracker:
    def __init__(self, tick_seconds: int = 15):
        self.counters = Counter()
        self.gauges: Dict[str, float] = {}
        self.drop_reasons = Counter()
        self.tick_seconds = tick_seconds

    def increment_ingested(self) -> None:
        self.counters["ingested"] += 1

    def increment_dropped(self, *, reason: str) -> None:
        self.counters["dropped_total"] += 1
        self.drop_reasons[reason] += 1

    def increment_frames_built(self) -> None:
        self.counters["frames_built"] += 1

    def increment_rpc_requests(self) -> None:
        self.counters["rpc_requests"] += 1

    def increment_rpc_errors(self) -> None:
        self.counters["rpc_errors"] += 1

    def set_queue_depth(self, depth: int) -> None:
        self.gauges["queue_depth"] = depth

    def record_frame_meta(self, *, ts_ms: int, count: int) -> None:
        self.gauges["last_frame_ts_ms"] = ts_ms
        self.gauges["events_in_window"] = count
        self.gauges.setdefault("frame_build_ms", 0.0)

    def record_frame_build_ms(self, duration_ms: float) -> None:
        self.gauges["frame_build_ms"] = duration_ms

    def record_salient(self, salience: float) -> None:
        self.gauges["last_salience"] = salience

    def snapshot(self) -> Dict[str, object]:
        out: Dict[str, object] = {}
        out.update(self.counters)
        out.update(self.gauges)
        out["dropped_by_reason"] = dict(self.drop_reasons)
        out.setdefault("ts_ms", int(time.time() * 1000))
        return out
