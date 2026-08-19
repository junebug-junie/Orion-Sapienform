#!/usr/bin/env python3
"""Attribute athena's I/O stall to the containers causing it, using cgroup v2 PSI.

ROADMAP step D2 (`docs/superpowers/specs/2026-08-13-scarcity-ROADMAP.md` §5A). This is an
instrument and it lands alone: it changes no runtime behaviour, publishes nothing to the bus,
writes no schema, and only ever reads.

WHY THIS EXISTS, AND WHY IT IS NOT THE INSTRUMENT THE ROADMAP SPECIFIED
----------------------------------------------------------------------
D2 was written as "per-container block I/O over a 24h window" -- rank the containers by bytes
moved. Bytes are the wrong quantity. A container that writes 200 MB in one sequential streak
and a container that issues 200 MB of scattered 4 KB fsyncs move identical bytes and cause
wildly different stalls, and it is the *stall* that D-i is about.

cgroup v2 exposes the stall directly. Every container has `io.pressure`, and the kernel already
integrates it:

    some  = fraction of wall time at least one task in this cgroup was stalled on I/O
    full  = fraction of wall time EVERY runnable task in this cgroup was stalled on I/O

`total=` is a cumulative microsecond counter, so a delta between two samples is the exact stall
time in that interval. That removes the sampling-bias problem that dominated A1: polls do not
need to be frequent or evenly spaced, because nothing is being sampled -- the kernel counted it
all and this just reads the odometer.

Host baseline at first read, 2026-08-19 01:48 (pasted per roadmap §7.5):

    /proc/pressure/io    some avg60=23.61   full avg60=22.29
    /proc/pressure/cpu   some avg60=0.12    full avg60=0.00

athena is stalled on I/O ~22% of wall time and is not CPU-contended at all. The roadmap
estimated "~33 tasks blocked on I/O" by subtracting container CPU from load average; that
derivation is no longer needed, and the direct number is both larger in implication and cheaper
to obtain.

RULES ENFORCED IN CODE, NOT REMEMBERED
--------------------------------------
1. THE STATISTIC IS STALL TIME, NOT BYTES. Bytes rank the wrong thing (above). Bytes are still
   recorded, because they are the natural next question once a stall is attributed, but the
   ranking is by stall.

2. A COUNTER THAT WENT BACKWARDS IS A RESTART, NOT A NEGATIVE STALL. Container restarts reset
   the cgroup counters. A negative delta is discarded as an unusable interval and counted in
   `resets`, never clamped to 0 -- clamping would silently attribute the pre-restart stall to
   nobody.

3. AN ABSENT CONTAINER IS NOT AN IDLE ONE. A container that was not running for part of the
   window has less exposure than one that ran throughout, so each row reports its own observed
   coverage. Ranking a 5-minute container against a 24-hour one on raw totals would be a
   category error.

4. STATE THE WINDOW AS COVERAGE, NOT SPAN. `--out` appends and the file may hold several runs.
   Inter-sample gaps larger than DISCONTINUITY_FACTOR x median are holes between runs and are
   excluded from both the numerator and the denominator.

5. REPORT THE HOST DENOMINATOR. A container stalling 40 s means nothing without the host's own
   stall over the same interval. Every row carries its share of host `full` stall, and the
   shares are reported alongside their sum so an unexplained remainder is visible rather than
   rounded away.

6. `full` IS THE CEILING, `some` IS THE SYMPTOM. `some` rises whenever any one task waits, which
   is normal and constant. `full` is the machine getting nothing done. Both are recorded; the
   report leads with `full`.

USAGE
-----
    python3 scripts/analysis/record_io_attribution.py record \
        --out /tmp/io-attribution/samples.jsonl --interval 10.0 --duration 86400

    python3 scripts/analysis/record_io_attribution.py report \
        --in /tmp/io-attribution/samples.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CGROUP_ROOT = Path("/sys/fs/cgroup")
HOST_IO_PRESSURE = Path("/proc/pressure/io")
HOST_CPU_PRESSURE = Path("/proc/pressure/cpu")

# A gap larger than this multiple of the median inter-sample interval is a hole between runs,
# not a slow sample. Same constant and same reasoning as record_lane_occupancy.py.
DISCONTINUITY_FACTOR = 5.0

_PSI_TOTAL = re.compile(r"^(some|full)\s+.*\btotal=(\d+)", re.MULTILINE)


def _read_psi(path: Path) -> Optional[Dict[str, int]]:
    """Return {"some": µs, "full": µs} or None when the file is absent/unreadable.

    None is a real answer, not an error: PSI is a kernel config option, and a cgroup can vanish
    between listing it and reading it. Rule 3 -- absent is not idle.
    """
    try:
        text = path.read_text()
    except (OSError, ValueError):
        return None
    out = {kind: int(total) for kind, total in _PSI_TOTAL.findall(text)}
    return out or None


def _read_io_bytes(path: Path) -> Optional[Dict[str, int]]:
    """Summed rbytes/wbytes across every device in a cgroup's io.stat."""
    try:
        text = path.read_text()
    except (OSError, ValueError):
        return None
    totals = {"rbytes": 0, "wbytes": 0}
    for line in text.splitlines():
        for field in line.split()[1:]:
            key, _, val = field.partition("=")
            if key in totals and val.isdigit():
                totals[key] += int(val)
    return totals


def _running_containers() -> Dict[str, str]:
    """{container_id_full: name}. Empty dict if docker is unavailable -- not an exception."""
    try:
        proc = subprocess.run(
            ["docker", "ps", "--no-trunc", "--format", "{{.ID}}\t{{.Names}}"],
            capture_output=True, text=True, timeout=30, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    out: Dict[str, str] = {}
    for line in proc.stdout.splitlines():
        cid, _, name = line.partition("\t")
        if cid and name:
            out[cid.strip()] = name.strip()
    return out


def _cgroup_for(cid: str) -> Optional[Path]:
    """Locate a container's cgroup directory across the layouts docker actually uses."""
    candidates = [
        CGROUP_ROOT / "system.slice" / f"docker-{cid}.scope",
        CGROUP_ROOT / "docker" / cid,
        CGROUP_ROOT / "system.slice" / f"docker-{cid}.scope" / "container",
    ]
    for path in candidates:
        if (path / "io.pressure").exists():
            return path
    return None


def sample(names: Dict[str, str]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "ts": time.time(),
        "host_io": _read_psi(HOST_IO_PRESSURE),
        "host_cpu": _read_psi(HOST_CPU_PRESSURE),
        "containers": {},
    }
    for cid, name in names.items():
        cg = _cgroup_for(cid)
        if cg is None:
            continue
        psi = _read_psi(cg / "io.pressure")
        if psi is None:
            continue
        entry: Dict[str, Any] = {"name": name, "some": psi.get("some"), "full": psi.get("full")}
        io_bytes = _read_io_bytes(cg / "io.stat")
        if io_bytes:
            entry.update(io_bytes)
        row["containers"][cid] = entry
    return row


def cmd_record(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    names = _running_containers()
    if not names:
        print("no running containers found (is docker reachable?)", file=sys.stderr)
        return 2
    host = _read_psi(HOST_IO_PRESSURE)
    if host is None:
        print(f"{HOST_IO_PRESSURE} unreadable -- kernel PSI not available, nothing to record",
              file=sys.stderr)
        return 2

    print(f"recording {len(names)} container(s) every {args.interval}s for "
          f"{args.duration / 3600:.2f}h -> {out}")
    print(f"  host io.pressure at start: some={host.get('some')}us full={host.get('full')}us")

    deadline = time.monotonic() + args.duration
    # Container set is re-read periodically: containers restart, and a restart changes the
    # cgroup path even when the name is stable.
    next_refresh = 0.0
    with out.open("a") as fh:
        while time.monotonic() < deadline:
            if time.monotonic() >= next_refresh:
                names = _running_containers() or names
                next_refresh = time.monotonic() + args.refresh_sec
            fh.write(json.dumps(sample(names), separators=(",", ":")) + "\n")
            fh.flush()
            time.sleep(args.interval)
    return 0


def _load(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    rows.sort(key=lambda r: r.get("ts", 0.0))
    return rows


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def cmd_report(args: argparse.Namespace) -> int:
    rows = _load(Path(args.infile))
    if len(rows) < 2:
        print("need at least 2 samples to compute a delta", file=sys.stderr)
        return 2

    gaps = [rows[i]["ts"] - rows[i - 1]["ts"] for i in range(1, len(rows))]
    median_gap = _median(gaps)
    gap_limit = median_gap * DISCONTINUITY_FACTOR if median_gap > 0 else float("inf")

    coverage_s = 0.0
    discontinuities = 0
    host_full_us = 0
    host_some_us = 0
    # cid -> {name, full_us, some_us, rbytes, wbytes, coverage_s, resets}
    per: Dict[str, Dict[str, Any]] = {}

    for prev, cur in zip(rows, rows[1:]):
        dt = cur["ts"] - prev["ts"]
        if dt <= 0 or dt > gap_limit:
            discontinuities += 1
            continue
        coverage_s += dt

        for kind, acc in (("full", "host_full"), ("some", "host_some")):
            a = (prev.get("host_io") or {}).get(kind)
            b = (cur.get("host_io") or {}).get(kind)
            if a is None or b is None or b < a:
                continue
            if acc == "host_full":
                host_full_us += b - a
            else:
                host_some_us += b - a

        pc = prev.get("containers") or {}
        cc = cur.get("containers") or {}
        for cid, cur_entry in cc.items():
            prev_entry = pc.get(cid)
            if prev_entry is None:
                # Rule 3: it was not observed over this interval, so it gets no exposure for it.
                continue
            row = per.setdefault(cid, {
                "name": cur_entry.get("name") or cid[:12],
                "full_us": 0, "some_us": 0, "rbytes": 0, "wbytes": 0,
                "coverage_s": 0.0, "resets": 0,
            })
            row["name"] = cur_entry.get("name") or row["name"]
            reset = False
            deltas: Dict[str, int] = {}
            for key, field in (("full", "full_us"), ("some", "some_us"),
                               ("rbytes", "rbytes"), ("wbytes", "wbytes")):
                a, b = prev_entry.get(key), cur_entry.get(key)
                if a is None or b is None:
                    continue
                if b < a:
                    # Rule 2: counters went backwards -> the container restarted. The interval
                    # is unusable; do NOT clamp to zero, which would silently drop real stall.
                    reset = True
                    break
                deltas[field] = b - a
            if reset:
                row["resets"] += 1
                continue
            for field, value in deltas.items():
                row[field] += value
            row["coverage_s"] += dt

    if coverage_s < args.min_window_sec:
        print(f"REFUSING to report: {coverage_s / 3600:.2f} h coverage < "
              f"--min-window-sec {args.min_window_sec / 3600:.2f} h. "
              f"Short windows produced eight wrong answers in this arc. Record longer, "
              f"or pass --allow-short.", file=sys.stderr)
        if not args.allow_short:
            return 3

    span_s = rows[-1]["ts"] - rows[0]["ts"]
    print(f"samples      {len(rows)} over span {span_s / 3600:.2f} h, "
          f"coverage {coverage_s / 3600:.2f} h, {discontinuities} discontinuity(ies)")
    print(f"sample every {median_gap:.1f} s (median)")
    print()
    host_full_pct = 100.0 * (host_full_us / 1e6) / coverage_s if coverage_s else 0.0
    host_some_pct = 100.0 * (host_some_us / 1e6) / coverage_s if coverage_s else 0.0
    print(f"HOST  io stall  full {host_full_pct:6.2f}%   some {host_some_pct:6.2f}%"
          f"   ({host_full_us / 1e6:.0f} s fully stalled)")
    print("      `full` is the machine getting nothing done; `some` is any one task waiting.")
    print()

    ranked = sorted(per.values(), key=lambda r: r["full_us"], reverse=True)
    print(f"{'container':<38} {'full%':>7} {'some%':>7} {'share':>7} "
          f"{'read':>10} {'write':>10} {'cov':>7} {'rst':>4}")
    print("-" * 96)
    attributed_us = 0
    for row in ranked[: args.top]:
        cov = row["coverage_s"] or 1.0
        full_pct = 100.0 * (row["full_us"] / 1e6) / cov
        some_pct = 100.0 * (row["some_us"] / 1e6) / cov
        share = 100.0 * row["full_us"] / host_full_us if host_full_us else 0.0
        attributed_us += row["full_us"]
        print(f"{row['name'][:38]:<38} {full_pct:7.2f} {some_pct:7.2f} {share:6.1f}% "
              f"{_h(row['rbytes']):>10} {_h(row['wbytes']):>10} "
              f"{cov / 3600:6.2f}h {row['resets']:4d}")
    total_share = 100.0 * attributed_us / host_full_us if host_full_us else 0.0
    print("-" * 96)
    print(f"top {min(args.top, len(ranked))} of {len(ranked)} containers "
          f"account for {total_share:.1f}% of host full-stall time")
    print()
    print("NOTE: container `full` shares need not sum to the host's -- containers stall")
    print("      concurrently, and host-level stall includes non-container tasks. A share far")
    print("      below 100% means the stall is NOT coming from the containers listed.")
    return 0


def _h(n: int) -> str:
    for unit in ("B", "K", "M", "G", "T"):
        if abs(n) < 1024 or unit == "T":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}T"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    rec = sub.add_parser("record")
    rec.add_argument("--out", required=True)
    rec.add_argument("--interval", type=float, default=10.0)
    rec.add_argument("--duration", type=float, default=86400.0)
    rec.add_argument("--refresh-sec", type=float, default=300.0,
                     help="how often to re-read the running container set")
    rec.set_defaults(func=cmd_record)

    rep = sub.add_parser("report")
    rep.add_argument("--in", dest="infile", required=True)
    rep.add_argument("--top", type=int, default=20)
    rep.add_argument("--min-window-sec", type=float, default=3600.0)
    rep.add_argument("--allow-short", action="store_true")
    rep.set_defaults(func=cmd_report)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
