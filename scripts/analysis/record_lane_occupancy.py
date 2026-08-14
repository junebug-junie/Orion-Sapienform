#!/usr/bin/env python3
"""Record llama.cpp lane occupancy over time, and report ceiling statistics.

ROADMAP step A1 (`docs/superpowers/specs/2026-08-13-scarcity-ROADMAP.md`). This is an
instrument and it lands alone: it changes no runtime behaviour, publishes nothing to the bus,
and writes no schema.

WHY THIS EXISTS
---------------
The scarcity arc kept measuring the wrong quantity. `nvidia-smi utilization.gpu` samples ~1s
out of every 31s (3% of the timeline), reports "is any kernel resident" rather than "is this
lane full", and under-reports bandwidth-bound LLM decode. The operationally meaningful ceiling
on a llama.cpp worker is its *slot* occupancy, and llama.cpp already publishes that at /slots.

RULES ENFORCED IN CODE, NOT REMEMBERED
--------------------------------------
Each of these exists because the arc got it wrong at least once.

1. FOR A CEILING, THE STATISTIC IS P(all busy), NOT THE MEAN. A lane that averages 11% of
   capacity but is completely full 7.4% of the time is a binding ceiling; the mean describes
   neither of its two states. `report` leads with P(all busy).

2. STATE THE WINDOW -- AND THE WINDOW IS COVERAGE, NOT SPAN. `--out` appends and the file may
   hold several runs, so `last_ts - first_ts` can claim 12 hours for 20 minutes of data. The
   report computes observed coverage by summing inter-sample gaps, discards gaps larger than
   DISCONTINUITY_FACTOR x the median (they are the holes between runs), and reports span,
   coverage and discontinuity count separately. The gate is on coverage.

3. A SAMPLE COUNT IS NOT A WINDOW. 600 samples at --interval 0.1 is 60 seconds. The gate is
   `--min-window-sec` (default 1h) *and* `--min-samples`, both on reachable data.

4. SLOT COUNT MUST BE HELD CONSTANT. A worker restarting from -np 4 to -np 8 mid-window makes
   `busy >= modal_c` count samples from the wide regime as "full" -- reporting a 45% ceiling
   for a lane that was never full. Occupancy statistics are computed over the modal regime
   only; off-modal samples are excluded and reported.

5. AN ABSENT MEASUREMENT IS NOT AN IDLE ONE. Unreachable hosts, `/slots` error envelopes, and
   an EMPTY slot array (llama.cpp serves `[]` while a model loads) are all recorded as
   indeterminate and excluded. Note the live gateway reads `[]` as ZERO FREE SLOTS and blocks
   background dispatch (`priority_admission.py::_free_slot_count`), so scoring it as idle would
   make the instrument disagree with production in the unsafe direction.

6. TIME-WEIGHT, DO NOT COUNT. Polls are sequential and `/slots` latency rises when a worker is
   saturated, so samples thin out during exactly the periods being measured. Counting samples
   understates the ceiling. Every statistic is weighted by the interval each sample represents.

7. REPORT THE THRESHOLD THE DISPATCHER ACTUALLY USES. `priority_admission` admits background
   work only while `free >= reserved_free_slots`, so on a 4-slot lane with 2 reserved,
   background is blocked at `busy >= 2` -- not at 4. Both ceilings are reported.

USAGE
-----
    python3 scripts/analysis/record_lane_occupancy.py record \
        --out /tmp/lane-occupancy/samples.jsonl --interval 1.0 --duration 86400

    python3 scripts/analysis/record_lane_occupancy.py report \
        --in /tmp/lane-occupancy/samples.jsonl

Lane definitions come from LLM_GATEWAY_ROUTE_TABLE_JSON -- the env var if set, otherwise the
first readable --env-file. Routes sharing an upstream URL are polled once and reported as one
lane, since they contend for the same slots.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_ENV_FILES = (
    "services/orion-llm-gateway/.env",
    "services/orion-llm-gateway/.env_example",
)
ROUTE_TABLE_KEY = "LLM_GATEWAY_ROUTE_TABLE_JSON"
DEFAULT_MIN_SAMPLES = 600
DEFAULT_MIN_WINDOW_SEC = 3600.0
SLOTS_TIMEOUT_SEC = 3.0
# A gap larger than this multiple of the median gap is a hole between runs, not observation.
#
# This constant sits between two failure modes that pull in opposite directions, so the value
# matters. Rule 6 (time-weighting) needs a *slow* tick counted at its true duration -- that is
# the whole bias being corrected. Rule 2 (coverage) needs a *between-run* hole discarded. With
# 4 lanes at a 3 s timeout each, a worst-case tick is ~12 s against a 1 s interval, i.e. ~12x;
# a gap between runs is hours, i.e. thousands of x. 30x sits above the first and far below the
# second. An earlier value of 5x was wrong: it classified an ordinary slow tick as a
# discontinuity and silently reverted every statistic to sample-counting.
DISCONTINUITY_FACTOR = 30.0
# Re-read the route table this often during a `record` run. See record()'s docstring: a route
# repointed mid-run otherwise makes the recorder poll a dead address for the rest of the window
# and report a live lane as "0 measured".
DEFAULT_ROUTE_REFRESH_SEC = 300.0


# --------------------------------------------------------------------------- config


def _strip_quotes(value: str) -> str:
    value = value.strip()
    for q in ("'", '"'):
        if len(value) >= 2 and value.startswith(q) and value.endswith(q):
            return value[1:-1]
    return value


def read_route_table(env_files: Sequence[str]) -> Dict[str, dict]:
    """Return the gateway route table, from the environment or the first readable env file."""
    raw = os.environ.get(ROUTE_TABLE_KEY)
    if not raw:
        for path in env_files:
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line.startswith("#") or "=" not in line:
                            continue
                        key, _, value = line.partition("=")
                        if key.strip() == ROUTE_TABLE_KEY:
                            raw = value
                            break
            except OSError:
                continue
            if raw:
                break
    if not raw:
        raise SystemExit(
            f"{ROUTE_TABLE_KEY} not found in the environment or any of: {', '.join(env_files)}"
        )
    try:
        table = json.loads(_strip_quotes(raw))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{ROUTE_TABLE_KEY} is not valid JSON: {exc}") from exc
    if not isinstance(table, dict):
        raise SystemExit(f"{ROUTE_TABLE_KEY} did not parse to an object")
    return table


@dataclass(frozen=True)
class Lane:
    """One llama.cpp upstream. Several gateway routes may share it, and its slots."""

    url: str
    routes: Tuple[str, ...]
    served_by: str
    # Largest reservation among routes sharing this upstream. `priority_admission` holds this
    # many slots free for foreground callers, so it -- not the slot count -- is the ceiling a
    # background route actually meets.
    reserved_free_slots: int = 0

    @property
    def label(self) -> str:
        return f"{'+'.join(self.routes)} @ {self.served_by or self.url}"


def lanes_from_route_table(table: Dict[str, dict]) -> List[Lane]:
    """Collapse routes onto their upstream URL -- routes sharing a URL share its slots."""
    by_url: Dict[str, Dict[str, object]] = {}
    for route, spec in table.items():
        if not isinstance(spec, dict):
            continue
        url = spec.get("url")
        if not isinstance(url, str) or not url:
            continue
        entry = by_url.setdefault(url, {"routes": [], "served_by": "", "reserved": 0})
        entry["routes"].append(route)  # type: ignore[union-attr]
        if not entry["served_by"]:
            entry["served_by"] = str(spec.get("served_by") or "")
        reserved = spec.get("reserved_free_slots")
        if isinstance(reserved, int) and reserved > int(entry["reserved"]):  # type: ignore[arg-type]
            entry["reserved"] = reserved
    return [
        Lane(
            url=url,
            routes=tuple(sorted(e["routes"])),  # type: ignore[arg-type]
            served_by=str(e["served_by"]),
            reserved_free_slots=int(e["reserved"]),  # type: ignore[arg-type]
        )
        for url, e in sorted(by_url.items())
    ]


# --------------------------------------------------------------------------- sampling


@dataclass
class Sample:
    ts: float
    url: str
    reachable: bool
    slots_total: Optional[int] = None
    slots_busy: Optional[int] = None
    error: Optional[str] = None

    def to_json(self) -> str:
        d: Dict[str, object] = {"ts": round(self.ts, 3), "url": self.url, "reachable": self.reachable}
        if self.reachable:
            d["slots_total"] = self.slots_total
            d["slots_busy"] = self.slots_busy
        else:
            d["error"] = self.error
        return json.dumps(d, separators=(",", ":"))


def parse_slots_payload(payload: object) -> Tuple[int, int]:
    """(total, busy) from a llama.cpp /slots body.

    Raises on anything that is not a populated slot list. A dict is an error envelope; an
    empty list is a worker that has not allocated slots yet (model loading). Neither is an
    idle lane, and reporting them as `busy=0` would manufacture idleness for a lane we know
    nothing about -- the exact failure this instrument exists to avoid. The live gateway
    treats `[]` as zero *free* slots, i.e. the opposite of idle.
    """
    if isinstance(payload, dict):
        raise ValueError(f"/slots returned an object, not a slot list: {sorted(payload)[:4]}")
    if not isinstance(payload, list):
        raise ValueError(f"/slots returned {type(payload).__name__}, expected list")
    if not payload:
        raise ValueError("/slots returned an empty slot list (worker not ready); not an idle lane")
    busy = 0
    for slot in payload:
        if isinstance(slot, dict) and slot.get("is_processing"):
            busy += 1
    return len(payload), busy


def poll_lane(lane: Lane, *, timeout: float = SLOTS_TIMEOUT_SEC, now: Optional[float] = None) -> Sample:
    """Never raises. Any failure becomes an unreachable sample, which is excluded from stats."""
    ts = time.time() if now is None else now
    url = lane.url.rstrip("/") + "/slots"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - fixed internal host
            body = json.loads(resp.read().decode("utf-8"))
        total, busy = parse_slots_payload(body)
    except (
        urllib.error.URLError,
        http.client.HTTPException,  # IncompleteRead/BadStatusLine: a host being powered off
        OSError,
        ValueError,  # includes json.JSONDecodeError and parse_slots_payload
        UnicodeDecodeError,
    ) as exc:
        return Sample(ts=ts, url=lane.url, reachable=False, error=f"{type(exc).__name__}: {exc}")
    except Exception as exc:  # noqa: BLE001 - a 24h run must not die on an unforeseen error
        return Sample(ts=ts, url=lane.url, reachable=False, error=f"UNEXPECTED {type(exc).__name__}: {exc}")
    return Sample(ts=ts, url=lane.url, reachable=True, slots_total=total, slots_busy=busy)


def _describe_lanes(lanes: Sequence[Lane], stderr) -> None:
    for lane in lanes:
        reserved = f", reserves {lane.reserved_free_slots}" if lane.reserved_free_slots else ""
        print(f"  {lane.label}  <- {lane.url}{reserved}", file=stderr)


def record(
    lanes: Sequence[Lane],
    out_path: str,
    *,
    interval: float,
    duration: float,
    env_files: Sequence[str] = DEFAULT_ENV_FILES,
    route_refresh_sec: float = DEFAULT_ROUTE_REFRESH_SEC,
    stderr=sys.stderr,
) -> int:
    """Poll every lane at `interval`, re-reading the route table every `route_refresh_sec`.

    THE REFRESH IS NOT A CONVENIENCE. The first version snapshotted the route table once at
    startup and never looked again, which produced exactly the class of confident wrong answer
    this instrument exists to prevent: the `agent` route was repointed from atlas to circe
    mid-run, and the recorder kept polling the old address and reported "0 measured over
    3.67 h" for a lane that was live and serving turns. Over a 24 h gate window that is not a
    stale label, it is a fabricated finding. Caught by Juniper, not by the instrument.

    Set --route-refresh-sec 0 to pin the lane set (useful when replaying a fixed topology).
    """
    if interval <= 0:
        raise SystemExit("--interval must be > 0 (a zero interval hammers /slots)")
    if duration <= 0:
        raise SystemExit("--duration must be > 0")
    if route_refresh_sec < 0:
        raise SystemExit("--route-refresh-sec must be >= 0 (0 pins the lane set)")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    deadline = time.time() + duration
    written = 0
    overruns = 0
    route_changes = 0
    lanes = list(lanes)
    next_refresh = time.time() + route_refresh_sec if route_refresh_sec else float("inf")
    print(
        f"recording {len(lanes)} lane(s) every {interval}s for {duration/3600:.2f}h -> {out_path}"
        + (f" (route table re-read every {route_refresh_sec:.0f}s)" if route_refresh_sec else " (lane set PINNED)"),
        file=stderr,
    )
    _describe_lanes(lanes, stderr)
    with open(out_path, "a", encoding="utf-8") as fh:
        while time.time() < deadline:
            tick = time.time()
            if tick >= next_refresh:
                next_refresh = tick + route_refresh_sec
                try:
                    fresh = lanes_from_route_table(read_route_table(env_files))
                except SystemExit as exc:  # unreadable env mid-run must not end the recording
                    print(f"WARNING: route-table refresh failed, keeping current lanes: {exc}", file=stderr)
                    fresh = []
                if fresh and fresh != lanes:
                    route_changes += 1
                    gone = {l.url for l in lanes} - {l.url for l in fresh}
                    added = {l.url for l in fresh} - {l.url for l in lanes}
                    print(
                        f"ROUTE TABLE CHANGED at {tick:.0f}: "
                        f"{len(gone)} lane(s) removed, {len(added)} added. Statistics for a "
                        f"removed url cover only the period it was routed -- check its coverage "
                        f"against the others before comparing them.",
                        file=stderr,
                    )
                    lanes = fresh
                    _describe_lanes(lanes, stderr)
            for lane in lanes:
                fh.write(poll_lane(lane).to_json() + "\n")
                written += 1
            fh.flush()
            slack = interval - (time.time() - tick)
            if slack > 0:
                time.sleep(slack)
            else:
                overruns += 1
    if route_changes:
        print(f"NOTE: the route table changed {route_changes} time(s) during this run.", file=stderr)
    if overruns:
        print(
            f"WARNING: {overruns} tick(s) took longer than --interval {interval}s. Sampling "
            f"thins out when workers are slow, which is when they are busy -- statistics are "
            f"time-weighted to compensate, but a large overrun count means the real resolution "
            f"is coarser than requested.",
            file=stderr,
        )
    return written


# --------------------------------------------------------------------------- statistics


def erlang_b(servers: int, offered: float) -> float:
    """Blocking probability for M/M/c/c. Iterative form -- no factorial overflow.

    B(1, a) == a/(1+a); at c=1 this equals mean occupancy, which is the whole reason
    utilisation and blocking are not comparable across lanes of different width.
    """
    if servers <= 0:
        return 1.0
    if offered <= 0:
        return 0.0
    inv = 1.0
    for k in range(1, servers + 1):
        inv = 1.0 + inv * k / offered
    return 1.0 / inv


def offered_load_from_carried(servers: int, carried: float) -> Optional[float]:
    """Invert carried = a * (1 - B(c, a)) for a, by bisection.

    `carried` is the mean number of busy servers. Returns None when it is not an attainable
    steady-state carried load, i.e. at or numerically indistinguishable from c.
    """
    if servers <= 0:
        return None
    if carried != carried:  # NaN
        return None
    if carried <= 0:
        return 0.0 if carried == 0 else None
    if carried >= servers:
        return None
    lo, hi = 0.0, max(1.0, float(servers))
    while hi * (1.0 - erlang_b(servers, hi)) < carried:
        hi *= 2.0
        if hi > 1e6:
            return None  # carried is numerically at capacity
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if mid <= lo or mid >= hi:
            break
        if mid * (1.0 - erlang_b(servers, mid)) < carried:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


@dataclass
class LaneStats:
    """Time-weighted occupancy over the modal slot-count regime only."""

    url: str
    n_total: int = 0
    n_reachable: int = 0
    n_indeterminate: int = 0
    n_off_modal: int = 0
    first_ts: Optional[float] = None
    last_ts: Optional[float] = None
    coverage_sec: float = 0.0
    n_discontinuities: int = 0
    median_gap_sec: Optional[float] = None
    slot_counts: Counter = field(default_factory=Counter)
    # busy -> total weight (seconds), modal regime only
    weight_by_busy: Dict[int, float] = field(default_factory=dict)
    total_weight: float = 0.0

    @property
    def span_sec(self) -> float:
        if self.first_ts is None or self.last_ts is None:
            return 0.0
        return self.last_ts - self.first_ts

    @property
    def servers(self) -> Optional[int]:
        if not self.slot_counts:
            return None
        return self.slot_counts.most_common(1)[0][0]

    @property
    def slot_count_changed(self) -> bool:
        return len(self.slot_counts) > 1

    def _weight_where(self, predicate) -> Optional[float]:
        if self.total_weight <= 0:
            return None
        return sum(w for b, w in self.weight_by_busy.items() if predicate(b)) / self.total_weight

    @property
    def mean_busy(self) -> Optional[float]:
        if self.total_weight <= 0:
            return None
        return sum(b * w for b, w in self.weight_by_busy.items()) / self.total_weight

    @property
    def p_any_busy(self) -> Optional[float]:
        return self._weight_where(lambda b: b > 0)

    @property
    def p_all_busy(self) -> Optional[float]:
        """THE headline: time fraction with every slot occupied, modal regime only."""
        c = self.servers
        if c is None or c <= 0:
            return None
        return self._weight_where(lambda b: b >= c)

    def p_admission_blocked(self, reserved: int) -> Optional[float]:
        """Time fraction where a background route would be refused admission.

        `priority_admission` admits only while free >= reserved, i.e. blocks at
        busy >= c - reserved + 1. With reserved=0 this reduces to p_all_busy.
        """
        c = self.servers
        if c is None or c <= 0:
            return None
        if reserved <= 0:
            return self.p_all_busy
        threshold = max(1, c - reserved + 1)
        return self._weight_where(lambda b: b >= threshold)


def _weights_for(timestamps: List[float]) -> Tuple[List[float], float, int, Optional[float]]:
    """Interval each sample represents, discarding between-run holes.

    Returns (weights, coverage_sec, n_discontinuities, median_gap).
    """
    n = len(timestamps)
    if n == 0:
        return [], 0.0, 0, None
    if n == 1:
        return [0.0], 0.0, 0, None
    gaps = [b - a for a, b in zip(timestamps, timestamps[1:])]
    positive = [g for g in gaps if g > 0]
    median_gap = statistics.median(positive) if positive else 0.0
    cutoff = median_gap * DISCONTINUITY_FACTOR if median_gap > 0 else float("inf")
    weights: List[float] = []
    discontinuities = 0
    for g in gaps:
        if g <= 0:
            weights.append(0.0)
        elif g > cutoff:
            discontinuities += 1
            weights.append(median_gap)  # the sample still represents one nominal interval
        else:
            weights.append(g)
    weights.append(median_gap)  # final sample
    return weights, sum(weights), discontinuities, median_gap or None


def accumulate(samples: Iterable[dict]) -> Dict[str, LaneStats]:
    """Two passes per lane: establish the modal slot count, then time-weight that regime."""
    by_url: Dict[str, List[dict]] = {}
    for row in samples:
        url = row.get("url")
        if isinstance(url, str):
            by_url.setdefault(url, []).append(row)

    stats: Dict[str, LaneStats] = {}
    for url, rows in by_url.items():
        rows = sorted(rows, key=lambda r: r.get("ts") if isinstance(r.get("ts"), (int, float)) else 0.0)
        st = LaneStats(url=url)
        st.n_total = len(rows)
        ts_all = [r["ts"] for r in rows if isinstance(r.get("ts"), (int, float))]
        if ts_all:
            st.first_ts, st.last_ts = min(ts_all), max(ts_all)

        usable: List[dict] = []
        for row in rows:
            busy, total = row.get("slots_busy"), row.get("slots_total")
            if (
                not row.get("reachable")
                or not isinstance(busy, int)
                or not isinstance(total, int)
                or total <= 0
            ):
                st.n_indeterminate += 1
                continue
            st.n_reachable += 1
            st.slot_counts[total] += 1
            usable.append(row)

        modal = st.servers
        if modal is None:
            stats[url] = st
            continue

        modal_rows = [r for r in usable if r["slots_total"] == modal]
        st.n_off_modal = len(usable) - len(modal_rows)
        modal_ts = [r["ts"] for r in modal_rows if isinstance(r.get("ts"), (int, float))]
        if len(modal_ts) != len(modal_rows):
            modal_rows = [r for r in modal_rows if isinstance(r.get("ts"), (int, float))]
        weights, coverage, discontinuities, median_gap = _weights_for(modal_ts)
        st.coverage_sec = coverage
        st.n_discontinuities = discontinuities
        st.median_gap_sec = median_gap
        for row, w in zip(modal_rows, weights):
            b = row["slots_busy"]
            st.weight_by_busy[b] = st.weight_by_busy.get(b, 0.0) + w
        st.total_weight = sum(st.weight_by_busy.values())
        stats[url] = st
    return stats


def read_samples(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _fmt_burstiness(observed: float, predicted: float) -> str:
    """Both directions matter: >1 is batched arrivals, <1 is smoother than Poisson."""
    if predicted <= 0:
        return "    burstiness   blocking observed where Poisson predicts ~none: batched arrivals"
    ratio = observed / predicted
    shown = f"{ratio:.2f}" if ratio < 10 else f"{ratio:.0f}"
    if ratio > 1.0:
        return f"    burstiness   {shown}x MORE blocking than Poisson  <- batched arrivals, not volume"
    return f"    burstiness   {shown}x (LESS than Poisson)  <- volume/smooth arrivals, not bursts"


def format_report(
    stats: Dict[str, LaneStats],
    lanes: Dict[str, Lane],
    *,
    min_samples: int,
    min_window_sec: float,
    allow_short: bool,
) -> Tuple[str, bool]:
    """Render the report. Returns (text, ok); ok is False if any lane failed a gate."""
    out: List[str] = []
    ok = True
    for url in sorted(stats):
        st = stats[url]
        lane = lanes.get(url)
        out.append(f"=== {lane.label if lane else url}")
        out.append(f"    url          {url}")
        out.append(
            f"    span         {st.span_sec/3600:.2f} h   "
            f"coverage {st.coverage_sec/3600:.2f} h"
            + (f"   ({st.n_discontinuities} gap(s) between runs discarded)" if st.n_discontinuities else "")
        )
        out.append(
            f"    samples      {st.n_total} total, {st.n_reachable} measured, "
            f"{st.n_indeterminate} indeterminate"
            + (f", {st.n_off_modal} off-modal EXCLUDED" if st.n_off_modal else "")
        )
        if st.median_gap_sec:
            out.append(f"    sample every {st.median_gap_sec:.2f} s (median)")

        if st.n_reachable == 0:
            out.append("    NO MEASUREMENTS in this window (host down, or /slots unavailable).")
            out.append("")
            continue

        if st.slot_count_changed:
            counts = ", ".join(f"{k} slots x{v}" for k, v in sorted(st.slot_counts.items()))
            out.append(
                f"    *** SLOT COUNT CHANGED mid-window ({counts}). The worker was restarted "
                f"with a different -np. Statistics below cover the modal regime "
                f"({st.servers} slots) ONLY. ***"
            )

        short_samples = st.n_reachable < min_samples
        short_window = st.coverage_sec < min_window_sec
        if (short_samples or short_window) and not allow_short:
            ok = False
            why = []
            if short_samples:
                why.append(f"{st.n_reachable} measured samples < --min-samples {min_samples}")
            if short_window:
                why.append(
                    f"{st.coverage_sec/3600:.2f} h coverage < --min-window-sec "
                    f"{min_window_sec/3600:.2f} h"
                )
            out.append(
                f"    REFUSING to report: {'; '.join(why)}. Short windows produced eight wrong "
                f"answers in this arc. Record longer, or pass --allow-short to override."
            )
            out.append("")
            continue
        if short_samples or short_window:
            out.append(
                f"    *** SHORT: {st.n_reachable} samples over {st.coverage_sec/3600:.2f} h. "
                f"Directional only -- do not calibrate on this. ***"
            )

        c = st.servers or 0
        p_all = st.p_all_busy
        p_any = st.p_any_busy
        mean_busy = st.mean_busy or 0.0
        out.append(f"    slots        {c}")
        if p_all is not None:
            out.append(f"    P(all busy)  {100*p_all:.2f}%    <- the ceiling statistic (time-weighted)")
        if lane and lane.reserved_free_slots > 0:
            p_adm = st.p_admission_blocked(lane.reserved_free_slots)
            if p_adm is not None:
                out.append(
                    f"    P(bg blocked){100*p_adm:>7.2f}%    <- background admission ceiling "
                    f"(reserves {lane.reserved_free_slots}, blocks at busy>="
                    f"{max(1, c - lane.reserved_free_slots + 1)})"
                )
        if p_any is not None:
            out.append(f"    P(any busy)  {100*p_any:.2f}%")
        if c:
            out.append(
                f"    mean busy    {mean_busy:.3f} / {c} slots "
                f"({100*mean_busy/c:.1f}% of capacity)  <- NOT the ceiling"
            )
            a = offered_load_from_carried(c, mean_busy)
            if a is None:
                out.append("    offered load unattainable (carried at or above capacity)")
            else:
                pred = erlang_b(c, a)
                out.append(f"    offered load {a:.3f} erlangs")
                out.append(f"    Erlang-B     {100*pred:.3f}%  (blocking if arrivals were Poisson)")
                if c == 1:
                    # At c=1, offered load is inverted FROM mean occupancy and Erlang-B(1,a)
                    # equals that same mean occupancy, so the ratio is ~1.00 by construction
                    # no matter how bursty arrivals are. Printing it invites reading a
                    # tautology as a finding. (It is also why utilisation is an honest
                    # number on a single-slot lane and a misleading one everywhere else.)
                    out.append(
                        "    burstiness   n/a at 1 slot (ratio is ~1.00 by construction; "
                        "P(all busy) IS the utilisation here)"
                    )
                elif p_all:
                    out.append(_fmt_burstiness(p_all, pred))
        hist = ", ".join(
            f"{b}:{w:.0f}s" for b, w in sorted(st.weight_by_busy.items())
        )
        out.append(f"    time by busy {{{hist}}}")
        out.append("")
    return "\n".join(out), ok


# --------------------------------------------------------------------------- cli


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    rec = sub.add_parser("record", help="poll /slots and append JSONL")
    rec.add_argument("--out", required=True)
    rec.add_argument("--interval", type=float, default=1.0)
    rec.add_argument("--duration", type=float, default=86400.0, help="seconds (default 24h)")
    rec.add_argument("--env-file", action="append", default=None)
    rec.add_argument(
        "--route-refresh-sec", type=float, default=DEFAULT_ROUTE_REFRESH_SEC,
        help="re-read the route table this often; 0 pins the lane set",
    )

    rep = sub.add_parser("report", help="compute ceiling statistics from JSONL")
    rep.add_argument("--in", dest="in_path", required=True)
    rep.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)
    rep.add_argument("--min-window-sec", type=float, default=DEFAULT_MIN_WINDOW_SEC)
    rep.add_argument("--allow-short", action="store_true")
    rep.add_argument("--env-file", action="append", default=None)

    args = ap.parse_args(argv)
    env_files = tuple(args.env_file) if args.env_file else DEFAULT_ENV_FILES

    if args.cmd == "record":
        lanes = lanes_from_route_table(read_route_table(env_files))
        if not lanes:
            raise SystemExit("route table produced no lanes")
        n = record(
            lanes, args.out, interval=args.interval, duration=args.duration,
            env_files=env_files, route_refresh_sec=args.route_refresh_sec,
        )
        print(f"wrote {n} samples to {args.out}", file=sys.stderr)
        return 0

    rows = read_samples(args.in_path)
    if not rows:
        raise SystemExit(f"no samples in {args.in_path}")
    lanes: Dict[str, Lane] = {}
    try:
        for lane in lanes_from_route_table(read_route_table(env_files)):
            lanes[lane.url] = lane
    except SystemExit:
        pass  # reporting on an archived file without the env present is fine
    text, ok = format_report(
        accumulate(rows),
        lanes,
        min_samples=args.min_samples,
        min_window_sec=args.min_window_sec,
        allow_short=args.allow_short,
    )
    print(text)
    return 0 if ok else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
