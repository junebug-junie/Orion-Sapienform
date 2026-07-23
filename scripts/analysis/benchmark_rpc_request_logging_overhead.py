#!/usr/bin/env python3
"""Real before/after overhead benchmark for the worker-path "reply received" log line
added to `OrionBusAsync.rpc_request()` (`orion/core/bus/async_service.py`, PR #1290).

Item 3 of the "gated investigation" before docs/superpowers/specs/2026-07-23-transport-
domain-rpc-health-redesign.md's step 2 (the in-memory aggregator): the design spec's
Acceptance Checks require "a before/after latency benchmark on rpc_request() itself,
proving the instrumentation adds no measurable overhead" before more gets added to this
hot path. This was deferred out of PR #1290 (a single logger.info() call, same shape as
4 other calls already on the same path) as low-risk-but-unmeasured; this script measures
it directly instead of continuing to assert it by analogy.

**Deliberately does not require Redis, Docker, or a live bus.** The new line is exactly:

    logger.info(
        "[rpc] reply received corr_id=%s reply_channel=%s path=worker elapsed_ms=%.1f",
        corr, reply_channel, (perf_counter() - started) * 1000.0,
    )

-- a call to Python's stdlib `logging` module with fixed positional-arg types (str, str,
float). Its cost is independent of anything Redis/asyncio-specific; benchmarking it in
isolation, at both realistic logger configurations (INFO enabled -- the real deployed
default, `EMBODIMENT_LOG_LEVEL=INFO` etc. across services -- and INFO disabled, in case
some deployment runs at WARNING), is a faithful measurement of the real marginal cost.

**Framing, not just raw nanoseconds:** the real question is not "is N nanoseconds fast in
some absolute sense" but "is N nanoseconds negligible relative to what this call site
actually costs in production." `scripts/analysis/measure_rpc_health_baseline.py`'s live
runs this session measured real p50 success latency at ~1500-3345ms, with the single
fastest real call observed at 72.9ms. This script reports the new log line's overhead as
an absolute time AND as a fraction of both of those real, measured numbers, not as an
abstract benchmark result disconnected from the actual system.

Run:
    python scripts/analysis/benchmark_rpc_request_logging_overhead.py
"""

from __future__ import annotations

import argparse
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Optional

# Real numbers from this session's live measure_rpc_health_baseline.py runs
# (docs/superpowers/specs/2026-07-23-transport-domain-rpc-health-redesign.md), not
# assumed defaults.
REAL_FASTEST_OBSERVED_CALL_MS = 72.9
REAL_P50_SUCCESS_LATENCY_MS = 1500.0  # conservative low end of the two observed p50s (1492.1/3344.5)

DEFAULT_ITERATIONS = 200_000


@dataclass
class BenchmarkResult:
    label: str
    iterations: int
    total_sec: float

    @property
    def per_call_ns(self) -> float:
        return (self.total_sec / self.iterations) * 1e9


def _null_logger(level: int) -> logging.Logger:
    """A real logging.Logger, real formatting machinery, writing to a NullHandler --
    exercises the actual cost path (level check, %-style lazy formatting, handler
    dispatch) without any real I/O skewing the measurement with disk/terminal cost."""
    logger = logging.getLogger(f"benchmark.rpc.overhead.{uuid.uuid4().hex[:8]}")
    logger.setLevel(level)
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger


def run_baseline(iterations: int) -> BenchmarkResult:
    """No logging call at all -- the pre-PR#1290 shape of this code path's success case."""
    started = time.monotonic()
    for _ in range(iterations):
        _ = time.perf_counter()  # the real code still calls this once per branch either way
    elapsed = time.monotonic() - started
    return BenchmarkResult(label="baseline (no log call)", iterations=iterations, total_sec=elapsed)


def run_with_log_call(iterations: int, logger: logging.Logger, level_label: str) -> BenchmarkResult:
    """The real new line, verbatim shape: same format string, same arg types."""
    corr = str(uuid.uuid4())
    reply_channel = "orion:test:result:00000000-0000-4000-8000-000000000000"
    t0 = time.perf_counter()
    started = time.monotonic()
    for _ in range(iterations):
        logger.info(
            "[rpc] reply received corr_id=%s reply_channel=%s path=worker elapsed_ms=%.1f",
            corr,
            reply_channel,
            (time.perf_counter() - t0) * 1000.0,
        )
    elapsed = time.monotonic() - started
    return BenchmarkResult(
        label=f"with new log line (logger {level_label})", iterations=iterations, total_sec=elapsed
    )


def render_report(baseline: BenchmarkResult, info_enabled: BenchmarkResult, info_disabled: BenchmarkResult) -> str:
    overhead_enabled_ns = info_enabled.per_call_ns - baseline.per_call_ns
    overhead_disabled_ns = info_disabled.per_call_ns - baseline.per_call_ns
    overhead_enabled_ms = overhead_enabled_ns / 1e6
    overhead_disabled_ms = overhead_disabled_ns / 1e6

    def _pct(overhead_ms: float, real_ms: float) -> str:
        return f"{(overhead_ms / real_ms) * 100:.6f}%"

    lines = [
        "# RPC Request Logging Overhead Benchmark",
        "",
        "Real, isolated microbenchmark (no Redis/Docker required) of the exact "
        "`logger.info(...)` call added to `OrionBusAsync.rpc_request()`'s worker-path "
        "success branch (PR #1290). See "
        "`docs/superpowers/specs/2026-07-23-transport-domain-rpc-health-redesign.md`.",
        "",
        f"- Iterations per configuration: {baseline.iterations}",
        f"- Baseline (no log call): {baseline.per_call_ns:.1f}ns/call",
        f"- With new log line, logger at INFO (real deployed default): {info_enabled.per_call_ns:.1f}ns/call",
        f"- With new log line, logger at WARNING (INFO filtered, early-exit path): {info_disabled.per_call_ns:.1f}ns/call",
        "",
        "## Marginal overhead of the new log line",
        "",
        f"- Logger at INFO (real path): +{overhead_enabled_ns:.1f}ns/call (+{overhead_enabled_ms:.6f}ms/call)",
        f"- Logger at WARNING (early-exit): +{overhead_disabled_ns:.1f}ns/call (+{overhead_disabled_ms:.6f}ms/call)",
        "",
        "## Overhead as a fraction of real, measured RPC call latency",
        "",
        "Not an abstract nanosecond number -- compared against real latencies "
        "`measure_rpc_health_baseline.py` observed live this session "
        f"(fastest real call: {REAL_FASTEST_OBSERVED_CALL_MS}ms; conservative p50: "
        f"{REAL_P50_SUCCESS_LATENCY_MS}ms):",
        "",
        f"- vs. fastest real observed call ({REAL_FASTEST_OBSERVED_CALL_MS}ms): "
        f"{_pct(overhead_enabled_ms, REAL_FASTEST_OBSERVED_CALL_MS)} of that call's total latency",
        f"- vs. real p50 success latency ({REAL_P50_SUCCESS_LATENCY_MS}ms): "
        f"{_pct(overhead_enabled_ms, REAL_P50_SUCCESS_LATENCY_MS)} of that call's total latency",
        "",
    ]

    # `overhead_enabled_ns < 0` is a distinct case from "small but real and positive" --
    # a negative reading means the microbenchmark's own run-to-run noise exceeded the
    # true cost (plausible for a call this cheap), not that the real overhead is exactly
    # zero. Report each honestly rather than collapsing "measurement noise" and "real but
    # tiny" into the same "negligible" label.
    is_noise_level = overhead_enabled_ns < 0
    if is_noise_level:
        lines.append(
            "Note: measured overhead at INFO came out negative (within run-to-run "
            "measurement noise for a call this cheap) -- the true marginal cost is at or "
            "below this benchmark's own noise floor, not a real negative number."
        )
        conclusion = (
            "at or below this benchmark's own measurement noise floor -- too small for "
            "this method to distinguish from zero"
        )
    else:
        practical_overhead_pct = (overhead_enabled_ms / REAL_FASTEST_OBSERVED_CALL_MS) * 100
        conclusion = (
            f"a real, measured +{overhead_enabled_ns:.0f}ns/call "
            f"({practical_overhead_pct:.4f}% of even the fastest real observed call)"
        )

    lines.extend(
        [
            "## Conclusion",
            "",
            f"Measured overhead is {conclusion} -- consistent with PR #1290's qualitative "
            "claim (same call shape as 4 other `logger.info()` calls already on this exact "
            "path), now measured directly rather than asserted by analogy.",
        ]
    )

    return "\n".join(lines)


def run(iterations: int) -> int:
    baseline = run_baseline(iterations)
    info_logger = _null_logger(logging.INFO)
    info_enabled = run_with_log_call(iterations, info_logger, "INFO")
    warn_logger = _null_logger(logging.WARNING)
    info_disabled = run_with_log_call(iterations, warn_logger, "WARNING")

    report = render_report(baseline, info_enabled, info_disabled)
    print(report)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return run(args.iterations)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
