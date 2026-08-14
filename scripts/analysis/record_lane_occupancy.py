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

Two rules from the arc are enforced mechanically here rather than remembered:

1. FOR A CEILING, THE STATISTIC IS P(all busy), NOT THE MEAN. A lane that averages 11% of
   capacity but is completely full 7.4% of the time is a binding ceiling; the mean describes
   neither of its two states. `--report` leads with P(all busy).

2. STATE THE WINDOW, OR DO NOT STATE THE NUMBER. Eight separate short-window errors were made
   in this arc. `--report` refuses to run below --min-samples (default 600) unless
   --allow-short is passed, and stamps the real window on every line either way.

An unreachable upstream is recorded as unreachable and EXCLUDED from occupancy statistics.
Counting a host that is switched off as "0 slots busy" would silently manufacture idleness --
circe is off by choice most of the time, so this is the default case, not an edge case.

USAGE
-----
    # record (foreground; use nohup/systemd/cron for a real 24h run)
    python3 scripts/analysis/record_lane_occupancy.py record \
        --out /tmp/lane-occupancy/samples.jsonl --interval 1.0 --duration 86400

    # report
    python3 scripts/analysis/record_lane_occupancy.py report \
        --in /tmp/lane-occupancy/samples.jsonl

Lane definitions come from LLM_GATEWAY_ROUTE_TABLE_JSON -- the env var if set, otherwise the
first readable --env-file. Routes sharing an upstream URL (chat and agent both point at
circe-worker-1) are polled once and reported as one lane, since they contend for the same
slots.
"""
from __future__ import annotations

import argparse
import json
import os
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
DEFAULT_MIN_SAMPLES = 600  # 10 minutes at 1 Hz
SLOTS_TIMEOUT_SEC = 3.0


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
    table = json.loads(_strip_quotes(raw))
    if not isinstance(table, dict):
        raise SystemExit(f"{ROUTE_TABLE_KEY} did not parse to an object")
    return table


@dataclass(frozen=True)
class Lane:
    """One llama.cpp upstream. Several gateway routes may share it."""

    url: str
    routes: Tuple[str, ...]
    served_by: str

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
        entry = by_url.setdefault(url, {"routes": [], "served_by": ""})
        entry["routes"].append(route)  # type: ignore[union-attr]
        if not entry["served_by"]:
            entry["served_by"] = str(spec.get("served_by") or "")
    return [
        Lane(url=url, routes=tuple(sorted(e["routes"])), served_by=str(e["served_by"]))  # type: ignore[arg-type]
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
        d = {"ts": round(self.ts, 3), "url": self.url, "reachable": self.reachable}
        if self.reachable:
            d["slots_total"] = self.slots_total
            d["slots_busy"] = self.slots_busy
        else:
            d["error"] = self.error
        return json.dumps(d, separators=(",", ":"))


def parse_slots_payload(payload: object) -> Tuple[int, int]:
    """(total, busy) from a llama.cpp /slots body.

    llama.cpp returns a list of slot objects. A dict is an error envelope (e.g. slots
    endpoint disabled) -- raise rather than silently reporting an idle lane.
    """
    if isinstance(payload, dict):
        raise ValueError(f"/slots returned an object, not a slot list: {sorted(payload)[:4]}")
    if not isinstance(payload, list):
        raise ValueError(f"/slots returned {type(payload).__name__}, expected list")
    total = len(payload)
    busy = 0
    for slot in payload:
        if isinstance(slot, dict) and slot.get("is_processing"):
            busy += 1
    return total, busy


def poll_lane(lane: Lane, *, timeout: float = SLOTS_TIMEOUT_SEC, now: Optional[float] = None) -> Sample:
    ts = time.time() if now is None else now
    url = lane.url.rstrip("/") + "/slots"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - fixed internal host
            body = json.loads(resp.read().decode("utf-8"))
        total, busy = parse_slots_payload(body)
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError) as exc:
        return Sample(ts=ts, url=lane.url, reachable=False, error=f"{type(exc).__name__}: {exc}")
    return Sample(ts=ts, url=lane.url, reachable=True, slots_total=total, slots_busy=busy)


def record(
    lanes: Sequence[Lane],
    out_path: str,
    *,
    interval: float,
    duration: float,
    stderr=sys.stderr,
) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    deadline = time.time() + duration
    written = 0
    print(
        f"recording {len(lanes)} lane(s) every {interval}s for {duration/3600:.2f}h -> {out_path}",
        file=stderr,
    )
    for lane in lanes:
        print(f"  {lane.label}  <- {lane.url}", file=stderr)
    with open(out_path, "a", encoding="utf-8") as fh:
        while time.time() < deadline:
            tick = time.time()
            for lane in lanes:
                fh.write(poll_lane(lane).to_json() + "\n")
                written += 1
            fh.flush()
            slack = interval - (time.time() - tick)
            if slack > 0:
                time.sleep(slack)
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

    `carried` is the mean number of busy servers. Returns None when it is not a physically
    attainable carried load (>= c), which would mean the lane never releases a slot.
    """
    if servers <= 0 or carried <= 0:
        return 0.0 if carried == 0 else None
    if carried >= servers:
        return None
    lo, hi = 0.0, max(1.0, float(servers))
    while lo + (hi - lo) / 2 > lo and hi * (1.0 - erlang_b(servers, hi)) < carried:
        hi *= 2.0
        if hi > 1e6:
            return None
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
    url: str
    n_total: int = 0
    n_reachable: int = 0
    n_unreachable: int = 0
    first_ts: Optional[float] = None
    last_ts: Optional[float] = None
    busy_hist: Counter = field(default_factory=Counter)
    slot_counts: Counter = field(default_factory=Counter)

    @property
    def window_sec(self) -> float:
        if self.first_ts is None or self.last_ts is None:
            return 0.0
        return self.last_ts - self.first_ts

    @property
    def servers(self) -> Optional[int]:
        """Modal slot count. A worker restart can change it; the mode is the honest summary."""
        if not self.slot_counts:
            return None
        return self.slot_counts.most_common(1)[0][0]

    @property
    def mean_busy(self) -> Optional[float]:
        if not self.n_reachable:
            return None
        return sum(k * v for k, v in self.busy_hist.items()) / self.n_reachable

    @property
    def p_any_busy(self) -> Optional[float]:
        if not self.n_reachable:
            return None
        return sum(v for k, v in self.busy_hist.items() if k > 0) / self.n_reachable

    @property
    def p_all_busy(self) -> Optional[float]:
        """THE headline. Fraction of reachable samples with every slot occupied."""
        c = self.servers
        if not self.n_reachable or c is None or c <= 0:
            return None
        return sum(v for k, v in self.busy_hist.items() if k >= c) / self.n_reachable


def accumulate(samples: Iterable[dict]) -> Dict[str, LaneStats]:
    stats: Dict[str, LaneStats] = {}
    for row in samples:
        url = row.get("url")
        if not isinstance(url, str):
            continue
        st = stats.setdefault(url, LaneStats(url=url))
        st.n_total += 1
        ts = row.get("ts")
        if isinstance(ts, (int, float)):
            st.first_ts = ts if st.first_ts is None else min(st.first_ts, ts)
            st.last_ts = ts if st.last_ts is None else max(st.last_ts, ts)
        if not row.get("reachable"):
            st.n_unreachable += 1
            continue
        busy, total = row.get("slots_busy"), row.get("slots_total")
        if not isinstance(busy, int) or not isinstance(total, int):
            st.n_unreachable += 1
            continue
        st.n_reachable += 1
        st.busy_hist[busy] += 1
        st.slot_counts[total] += 1
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


def format_report(
    stats: Dict[str, LaneStats],
    labels: Dict[str, str],
    *,
    min_samples: int,
    allow_short: bool,
) -> Tuple[str, bool]:
    """Render the report. Returns (text, ok) -- ok is False if any lane was too short."""
    out: List[str] = []
    ok = True
    for url in sorted(stats):
        st = stats[url]
        label = labels.get(url, url)
        out.append(f"=== {label}")
        out.append(f"    url          {url}")
        hrs = st.window_sec / 3600.0
        out.append(
            f"    window       {hrs:.2f} h   samples={st.n_total} "
            f"(reachable={st.n_reachable}, unreachable={st.n_unreachable})"
        )
        if st.n_reachable == 0:
            out.append("    UNREACHABLE for the whole window -- no occupancy statistics.")
            out.append("")
            continue
        if st.n_reachable < min_samples and not allow_short:
            ok = False
            out.append(
                f"    REFUSING to report: {st.n_reachable} reachable samples < --min-samples "
                f"{min_samples}. Short windows produced eight wrong answers in this arc. "
                f"Record longer, or pass --allow-short to override."
            )
            out.append("")
            continue
        if st.n_reachable < min_samples:
            out.append(
                f"    *** SHORT WINDOW ({st.n_reachable} < {min_samples} samples): "
                f"directional only, do not calibrate on this. ***"
            )
        c = st.servers
        mean_busy = st.mean_busy or 0.0
        p_all = st.p_all_busy
        p_any = st.p_any_busy
        out.append(f"    slots        {c}")
        out.append(
            f"    P(all busy)  {100*p_all:.2f}%    <- the ceiling statistic"
            if p_all is not None
            else "    P(all busy)  n/a"
        )
        out.append(f"    P(any busy)  {100*p_any:.2f}%" if p_any is not None else "")
        out.append(
            f"    mean busy    {mean_busy:.3f} / {c} slots "
            f"({100*mean_busy/c:.1f}% of capacity)  <- NOT the ceiling"
            if c
            else ""
        )
        if c:
            a = offered_load_from_carried(c, mean_busy)
            if a is None:
                out.append("    offered load  unattainable (carried >= slots): lane never idles")
            else:
                pred = erlang_b(c, a)
                out.append(f"    offered load {a:.3f} erlangs")
                out.append(f"    Erlang-B     {100*pred:.3f}%  (blocking if arrivals were Poisson)")
                # Only meaningful once blocking has actually been observed. Printing "0x" for
                # a lane that never filled invites reading "0x burstiness" as a finding when
                # it is just an absence of the event.
                if p_all:
                    if pred > 0:
                        out.append(
                            f"    burstiness   {p_all/pred:.0f}x more blocking than Poisson"
                            f"  <- >1 means batched arrivals, not volume"
                        )
                    else:
                        out.append(
                            "    burstiness   blocking observed where Poisson predicts ~none:"
                            " batched arrivals"
                        )
        hist = ", ".join(f"{k}:{v}" for k, v in sorted(st.busy_hist.items()))
        out.append(f"    distribution {{{hist}}}")
        out.append("")
    return "\n".join(line for line in out if line is not None), ok


# --------------------------------------------------------------------------- cli


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    rec = sub.add_parser("record", help="poll /slots and append JSONL")
    rec.add_argument("--out", required=True)
    rec.add_argument("--interval", type=float, default=1.0)
    rec.add_argument("--duration", type=float, default=86400.0, help="seconds (default 24h)")
    rec.add_argument("--env-file", action="append", default=None)

    rep = sub.add_parser("report", help="compute ceiling statistics from JSONL")
    rep.add_argument("--in", dest="in_path", required=True)
    rep.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)
    rep.add_argument("--allow-short", action="store_true")
    rep.add_argument("--env-file", action="append", default=None)

    args = ap.parse_args(argv)
    env_files = tuple(args.env_file) if args.env_file else DEFAULT_ENV_FILES

    if args.cmd == "record":
        lanes = lanes_from_route_table(read_route_table(env_files))
        if not lanes:
            raise SystemExit("route table produced no lanes")
        n = record(lanes, args.out, interval=args.interval, duration=args.duration)
        print(f"wrote {n} samples to {args.out}", file=sys.stderr)
        return 0

    rows = read_samples(args.in_path)
    if not rows:
        raise SystemExit(f"no samples in {args.in_path}")
    labels: Dict[str, str] = {}
    try:
        for lane in lanes_from_route_table(read_route_table(env_files)):
            labels[lane.url] = lane.label
    except SystemExit:
        pass  # reporting on an archived file without the env present is fine
    text, ok = format_report(
        accumulate(rows), labels, min_samples=args.min_samples, allow_short=args.allow_short
    )
    print(text)
    return 0 if ok else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
