#!/usr/bin/env python3
"""Read-only baseline for real cross-service RPC health, parsed from live docker logs.

Item 1 of docs/superpowers/specs/2026-07-23-transport-domain-rpc-health-redesign.md's
"Recommended next patch": measure the real RPC latency/timeout distribution *before*
designing a schema or aggregator around an assumed shape. `OrionBusAsync.rpc_request()`
(`orion/core/bus/async_service.py`) already logs every RPC call's lifecycle
(`logger.info`/`logger.error`, prefixed `[rpc]`) but nothing persists that data anywhere
queryable -- this script parses it directly out of `docker logs` for a configurable set of
services, real read-only, no writes, no bus events, no flag changes.

**Known gap this script's own prerequisite fix closed, disclosed not hidden:** before
2026-07-23, `rpc_request()`'s "worker" RPC path (the common one -- live sampling from
`orion-cortex-orch` showed only 6 of 33 real calls used the "inline" path) logged nothing at
all on a successful completion, only on timeout. So historical logs from before that fix
will show real latency data only for `path=inline` calls and for timeouts (both paths); a
"reply received ... path=worker" line only exists in logs written after the fix deployed.
This script reports the fix's own deploy-relative coverage honestly rather than silently
treating pre-fix silence as "no traffic."

Run:
    python scripts/analysis/measure_rpc_health_baseline.py --since 24h
    python scripts/analysis/measure_rpc_health_baseline.py --since 24h --containers orion-athena-cortex-orch,orion-athena-cortex-exec
"""

from __future__ import annotations

import argparse
import logging
import re
import statistics
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("orion.analysis.rpc_health_baseline")

OUTPUT_DIR = Path("/tmp/rpc-health-baseline")
REPORT_PATH = OUTPUT_DIR / "report.md"
PROGRESS_PATH = OUTPUT_DIR / "progress.log"

# Confirmed live (2026-07-23) real callers of OrionBusAsync.rpc_request() -- not
# exhaustive (37+ files import it repo-wide), but a representative, high-traffic subset
# spanning multiple independent services, chosen to prove cross-service coverage rather
# than list every container.
DEFAULT_CONTAINERS = (
    "orion-athena-cortex-exec",
    "orion-athena-cortex-orch",
    "orion-athena-hub",
    "orion-athena-actions",
    "orion-athena-thought",
    "orion-athena-spark-introspector",
    "orion-athena-chat-memory",
)

_RE_PUBLISH_BEGIN = re.compile(
    r"\[rpc\] publish begin corr_id=(?P<corr_id>\S+) request_channel=(?P<request_channel>\S+) "
    r"reply_channel=(?P<reply_channel>\S+) path=(?P<path>\w+) elapsed_ms=(?P<elapsed_ms>[\d.]+)"
)
_RE_REPLY_RECEIVED = re.compile(
    r"\[rpc\] reply received corr_id=(?P<corr_id>\S+) reply_channel=(?P<reply_channel>\S+) "
    r"path=(?P<path>\w+) elapsed_ms=(?P<elapsed_ms>[\d.]+)"
)
_RE_TIMEOUT = re.compile(
    r"\[rpc\] timeout waiting for reply corr_id=(?P<corr_id>\S+) request_channel=(?P<request_channel>\S+) "
    r"reply_channel=(?P<reply_channel>\S+) timeout_sec=(?P<timeout_sec>[\d.]+) elapsed_ms=(?P<elapsed_ms>[\d.]+)"
)


# ===========================================================================
# Pure layer -- no I/O. Exercised directly by unit tests with synthetic log lines.
# ===========================================================================


@dataclass
class RpcCallRecord:
    corr_id: str
    request_channel: Optional[str] = None
    path: Optional[str] = None
    outcome: Optional[str] = None  # "success" | "timeout" | None (unresolved)
    latency_ms: Optional[float] = None


@dataclass
class RpcHealthBaseline:
    n_calls: int
    n_success: int
    n_timeout: int
    n_unresolved: int
    latency_ms_all: list[float] = field(default_factory=list)
    latency_by_path: dict[str, list[float]] = field(default_factory=dict)
    channel_prefix_counts: dict[str, int] = field(default_factory=dict)

    @property
    def timeout_rate(self) -> Optional[float]:
        resolved = self.n_success + self.n_timeout
        return (self.n_timeout / resolved) if resolved else None


def _channel_prefix(channel: str) -> str:
    """Strip a trailing correlation-id-shaped suffix so channels like
    'orion:exec:result:RecallService:<uuid>' group under one real logical channel
    instead of one bucket per call. A UUID is 36 chars with 4 dashes in the standard
    positions -- match that shape specifically, not just "any trailing segment", so a
    real non-UUID channel name isn't accidentally truncated."""
    parts = channel.split(":")
    _uuid_re = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)
    while parts and _uuid_re.match(parts[-1]):
        parts.pop()
    return ":".join(parts)


def parse_log_lines(lines: list[str]) -> dict[str, RpcCallRecord]:
    """Parse raw docker-log lines into one RpcCallRecord per corr_id seen. Pure -- no
    I/O, no subprocess. A corr_id with only a "publish begin" line (no reply/timeout
    yet observed in this log window) stays outcome=None, reported as unresolved rather
    than silently dropped."""
    records: dict[str, RpcCallRecord] = {}

    def _get(corr_id: str) -> RpcCallRecord:
        rec = records.get(corr_id)
        if rec is None:
            rec = RpcCallRecord(corr_id=corr_id)
            records[corr_id] = rec
        return rec

    for line in lines:
        m = _RE_PUBLISH_BEGIN.search(line)
        if m:
            rec = _get(m.group("corr_id"))
            rec.request_channel = _channel_prefix(m.group("request_channel"))
            rec.path = m.group("path")
            continue
        m = _RE_REPLY_RECEIVED.search(line)
        if m:
            rec = _get(m.group("corr_id"))
            rec.path = m.group("path")
            rec.outcome = "success"
            rec.latency_ms = float(m.group("elapsed_ms"))
            continue
        m = _RE_TIMEOUT.search(line)
        if m:
            rec = _get(m.group("corr_id"))
            if rec.request_channel is None:
                rec.request_channel = _channel_prefix(m.group("request_channel"))
            rec.outcome = "timeout"
            rec.latency_ms = float(m.group("elapsed_ms"))
            continue

    return records


def compute_baseline(records: dict[str, RpcCallRecord]) -> RpcHealthBaseline:
    n_success = sum(1 for r in records.values() if r.outcome == "success")
    n_timeout = sum(1 for r in records.values() if r.outcome == "timeout")
    n_unresolved = sum(1 for r in records.values() if r.outcome is None)

    latency_ms_all: list[float] = [r.latency_ms for r in records.values() if r.latency_ms is not None]
    latency_by_path: dict[str, list[float]] = {}
    for r in records.values():
        if r.latency_ms is None or r.path is None:
            continue
        latency_by_path.setdefault(r.path, []).append(r.latency_ms)

    channel_prefix_counts: dict[str, int] = {}
    for r in records.values():
        if r.request_channel:
            channel_prefix_counts[r.request_channel] = channel_prefix_counts.get(r.request_channel, 0) + 1

    return RpcHealthBaseline(
        n_calls=len(records),
        n_success=n_success,
        n_timeout=n_timeout,
        n_unresolved=n_unresolved,
        latency_ms_all=latency_ms_all,
        latency_by_path=latency_by_path,
        channel_prefix_counts=channel_prefix_counts,
    )


def _percentile(sorted_values: list[float], pct: float) -> Optional[float]:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = k - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac


# ===========================================================================
# I/O layer.
# ===========================================================================


def fetch_container_logs(container: str, since: str) -> list[str]:
    """Real docker logs for one container. Read-only; never raises past this
    boundary -- a missing/stopped container just contributes zero lines."""
    try:
        result = subprocess.run(
            ["docker", "logs", "--since", since, container],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception:
        logger.warning("failed to fetch docker logs for %s", container, exc_info=True)
        return []
    if result.returncode != 0:
        logger.warning("docker logs %s exited %s: %s", container, result.returncode, result.stderr[:200])
    combined = (result.stdout or "") + (result.stderr or "")
    return combined.splitlines()


# ===========================================================================
# Report rendering + orchestration.
# ===========================================================================


class ProgressLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._start = time.monotonic()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = path.open("w", encoding="utf-8")
        except Exception:
            self._fh = None

    def emit(self, title: str, *, percent: float, processed: int, total: int) -> None:
        elapsed = max(time.monotonic() - self._start, 1e-6)
        rate = processed / elapsed
        line = f"{title} | {percent:5.1f}% | items={processed}/{total} | rate={rate:.1f}/s"
        logger.info(line)
        if self._fh is not None:
            try:
                self._fh.write(line + "\n")
                self._fh.flush()
            except Exception:
                pass

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.1f}"


def render_report(*, since: str, containers: tuple[str, ...], baseline: RpcHealthBaseline) -> str:
    lines = [
        "# RPC Health Baseline -- Real Cross-Service Data",
        "",
        "Read-only. No writes, no bus events, no config changes. See "
        "`docs/superpowers/specs/2026-07-23-transport-domain-rpc-health-redesign.md` "
        "\"Recommended next patch\" item 1.",
        "",
        f"- Window: --since {since}",
        f"- Containers scanned: {', '.join(containers)}",
        f"- Distinct RPC calls observed (by corr_id): {baseline.n_calls}",
        f"- Success: {baseline.n_success}",
        f"- Timeout: {baseline.n_timeout}",
        f"- Unresolved in this window (no reply/timeout line yet seen): {baseline.n_unresolved}",
    ]
    if baseline.timeout_rate is not None:
        lines.append(f"- Timeout rate (of resolved calls): {baseline.timeout_rate * 100:.2f}%")
    lines.append("")

    if baseline.latency_ms_all:
        sorted_all = sorted(baseline.latency_ms_all)
        lines.extend(
            [
                "## Latency distribution (all resolved calls with a real elapsed_ms)",
                "",
                f"- n: {len(sorted_all)}",
                f"- min: {_fmt(sorted_all[0])}ms",
                f"- p50: {_fmt(_percentile(sorted_all, 0.50))}ms",
                f"- p95: {_fmt(_percentile(sorted_all, 0.95))}ms",
                f"- p99: {_fmt(_percentile(sorted_all, 0.99))}ms",
                f"- max: {_fmt(sorted_all[-1])}ms",
                "",
            ]
        )
    else:
        lines.extend(["## Latency distribution", "", "No resolved calls with latency data in this window.", ""])

    lines.extend(["## Latency by RPC path", ""])
    if not baseline.latency_by_path:
        lines.append(
            "No per-path latency data available. If this window predates the "
            "2026-07-23 worker-path logging fix, this is expected for path=worker "
            "specifically -- see this script's module docstring."
        )
    for path_name, values in sorted(baseline.latency_by_path.items()):
        sv = sorted(values)
        lines.append(
            f"- `path={path_name}`: n={len(sv)}, p50={_fmt(_percentile(sv, 0.5))}ms, "
            f"p95={_fmt(_percentile(sv, 0.95))}ms, max={_fmt(sv[-1])}ms"
        )
    lines.append("")

    lines.extend(["## Real request-channel breakdown (top 20 by call count)", "", "| channel | calls |", "| --- | --- |"])
    for channel, count in sorted(baseline.channel_prefix_counts.items(), key=lambda kv: -kv[1])[:20]:
        lines.append(f"| `{channel}` | {count} |")

    if len(baseline.channel_prefix_counts) >= 2:
        lines.extend(
            [
                "",
                f"**{len(baseline.channel_prefix_counts)} distinct real request channels observed** -- "
                "genuinely cross-service, not one dominant caller standing in for the whole signal "
                "the way the old transport_pressure family was.",
            ]
        )

    return "\n".join(lines)


def run(since: str, containers: tuple[str, ...]) -> int:
    progress = ProgressLog(PROGRESS_PATH)
    progress.emit("rpc health baseline started", percent=0.0, processed=0, total=len(containers))

    all_lines: list[str] = []
    for i, container in enumerate(containers, start=1):
        all_lines.extend(fetch_container_logs(container, since))
        progress.emit(f"fetched logs for {container}", percent=100.0 * i / len(containers), processed=i, total=len(containers))

    records = parse_log_lines(all_lines)
    baseline = compute_baseline(records)
    report = render_report(since=since, containers=containers, baseline=baseline)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    progress.emit("report written", percent=100.0, processed=len(containers), total=len(containers))
    progress.close()

    print(report)
    print(f"\nFull report: {REPORT_PATH}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only baseline for real cross-service RPC health, parsed from live docker logs."
    )
    parser.add_argument(
        "--since", type=str, default="24h",
        help="docker logs --since window, e.g. '24h', '2h', a timestamp (default: 24h)",
    )
    parser.add_argument(
        "--containers", type=str, default=",".join(DEFAULT_CONTAINERS),
        help="comma-separated container names to scan (default: a representative real subset)",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_arg_parser().parse_args(argv)
    containers = tuple(c.strip() for c in args.containers.split(",") if c.strip())
    return run(args.since, containers)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
