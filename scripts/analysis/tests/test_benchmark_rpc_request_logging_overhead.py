"""Deterministic unit tests for benchmark_rpc_request_logging_overhead.py's report
rendering -- specifically the noise-floor vs. real-but-tiny-overhead distinction found
and fixed during this session's "gated investigation" (real overhead was measured at
+6150ns/call, not noise, but the first draft's conclusion line mislabeled anything under
0.01% as "noise-level" even when the raw ns delta was clearly positive and real).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[1] / "benchmark_rpc_request_logging_overhead.py"
_spec = importlib.util.spec_from_file_location("benchmark_rpc_request_logging_overhead", _MODULE_PATH)
mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules["benchmark_rpc_request_logging_overhead"] = mod
_spec.loader.exec_module(mod)


def _result(label: str, iterations: int, total_sec: float) -> "mod.BenchmarkResult":
    return mod.BenchmarkResult(label=label, iterations=iterations, total_sec=total_sec)


def test_per_call_ns_computed_correctly() -> None:
    r = _result("x", iterations=1000, total_sec=0.001)  # 1000 calls in 1ms -> 1000ns/call
    assert r.per_call_ns == 1000.0


def test_render_report_labels_real_positive_overhead_as_measured_not_noise() -> None:
    """A small-but-real positive overhead (e.g. 0.008% of the fastest real call) must
    not be relabeled "noise-level" just because the percentage is small -- that
    conflated two different claims in an earlier draft, caught during review."""
    baseline = _result("baseline", iterations=200_000, total_sec=200_000 * 85e-9)
    info_enabled = _result("info", iterations=200_000, total_sec=200_000 * (85e-9 + 6000e-9))
    info_disabled = _result("warn", iterations=200_000, total_sec=200_000 * (85e-9 + 200e-9))
    report = mod.render_report(baseline, info_enabled, info_disabled)
    assert "noise-level" not in report
    assert "real, measured" in report
    assert "+6000ns/call" in report or "+6001ns/call" in report or "+5999ns/call" in report


def test_render_report_labels_true_noise_case_distinctly() -> None:
    """When the measured overhead comes out negative (run-to-run noise exceeding the
    true cost), the report must say so plainly, not report a fabricated positive number
    or silently clamp to zero without explanation."""
    baseline = _result("baseline", iterations=200_000, total_sec=200_000 * 200e-9)
    info_enabled = _result("info", iterations=200_000, total_sec=200_000 * 190e-9)  # "faster" than baseline: noise
    info_disabled = _result("warn", iterations=200_000, total_sec=200_000 * 195e-9)
    report = mod.render_report(baseline, info_enabled, info_disabled)
    assert "noise floor" in report
    conclusion_section = report.split("## Conclusion")[1]
    assert "real, measured" not in conclusion_section


def test_render_report_does_not_double_sign_negative_overhead() -> None:
    """The marginal-overhead bullet used to hardcode a literal '+' prefix, which
    rendered a negative delta as '+-10.0ns/call' -- must show a single, correct sign."""
    baseline = _result("baseline", iterations=200_000, total_sec=200_000 * 200e-9)
    info_enabled = _result("info", iterations=200_000, total_sec=200_000 * 190e-9)
    info_disabled = _result("warn", iterations=200_000, total_sec=200_000 * 195e-9)
    report = mod.render_report(baseline, info_enabled, info_disabled)
    assert "+-" not in report
    assert "-10.0ns/call" in report


def test_render_report_exact_zero_overhead_treated_as_noise_floor() -> None:
    baseline = _result("baseline", iterations=200_000, total_sec=200_000 * 200e-9)
    info_enabled = _result("info", iterations=200_000, total_sec=200_000 * 200e-9)  # identical -> exactly 0
    info_disabled = _result("warn", iterations=200_000, total_sec=200_000 * 195e-9)
    report = mod.render_report(baseline, info_enabled, info_disabled)
    assert "noise floor" in report


def test_render_report_includes_both_real_reference_latencies() -> None:
    baseline = _result("baseline", iterations=1000, total_sec=1000 * 85e-9)
    info_enabled = _result("info", iterations=1000, total_sec=1000 * 6000e-9)
    info_disabled = _result("warn", iterations=1000, total_sec=1000 * 250e-9)
    report = mod.render_report(baseline, info_enabled, info_disabled)
    assert f"{mod.REAL_FASTEST_OBSERVED_CALL_MS}ms" in report
    assert f"{mod.REAL_P50_SUCCESS_LATENCY_MS}ms" in report
