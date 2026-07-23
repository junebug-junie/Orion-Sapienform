"""Deterministic unit tests for measure_rpc_health_baseline.py.

No docker, no subprocess. All pure functions operate on synthetic log-line strings,
same module-loading pattern as scripts/analysis/tests/test_measure_self_state_signal_quality.py.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[1] / "measure_rpc_health_baseline.py"
_spec = importlib.util.spec_from_file_location("measure_rpc_health_baseline", _MODULE_PATH)
mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules["measure_rpc_health_baseline"] = mod
_spec.loader.exec_module(mod)


def _worker_success_lines(corr_id: str, elapsed_ms: float, request_channel: str = "orion:test:request") -> list[str]:
    reply_channel = f"orion:test:result:{corr_id}"
    return [
        f"INFO:orion.bus.async:[rpc] publish begin corr_id={corr_id} request_channel={request_channel} "
        f"reply_channel={reply_channel} path=worker elapsed_ms=1.0",
        f"INFO:orion.bus.async:[rpc] publish success corr_id={corr_id} request_channel={request_channel} "
        f"reply_channel={reply_channel} path=worker elapsed_ms=2.0",
        f"INFO:orion.bus.async:[rpc] waiting for reply corr_id={corr_id} reply_channel={reply_channel} "
        f"timeout_sec=60.00 path=worker",
        f"INFO:orion.bus.async:[rpc] reply received corr_id={corr_id} reply_channel={reply_channel} "
        f"path=worker elapsed_ms={elapsed_ms}",
    ]


def _timeout_lines(corr_id: str, elapsed_ms: float, request_channel: str = "orion:test:request") -> list[str]:
    reply_channel = f"orion:test:result:{corr_id}"
    return [
        f"INFO:orion.bus.async:[rpc] publish begin corr_id={corr_id} request_channel={request_channel} "
        f"reply_channel={reply_channel} path=worker elapsed_ms=1.0",
        f"INFO:orion.bus.async:[rpc] waiting for reply corr_id={corr_id} reply_channel={reply_channel} "
        f"timeout_sec=5.00 path=worker",
        f"ERROR:orion.bus.async:[rpc] timeout waiting for reply corr_id={corr_id} request_channel={request_channel} "
        f"reply_channel={reply_channel} timeout_sec=5.00 elapsed_ms={elapsed_ms}",
    ]


def test_parse_log_lines_extracts_success_record() -> None:
    lines = _worker_success_lines("abc-1", 123.4)
    records = mod.parse_log_lines(lines)
    assert "abc-1" in records
    rec = records["abc-1"]
    assert rec.outcome == "success"
    assert rec.path == "worker"
    assert rec.latency_ms == 123.4
    assert rec.request_channel == "orion:test:request"


def test_parse_log_lines_extracts_timeout_record() -> None:
    lines = _timeout_lines("abc-2", 5001.2)
    records = mod.parse_log_lines(lines)
    rec = records["abc-2"]
    assert rec.outcome == "timeout"
    assert rec.latency_ms == 5001.2


def test_parse_log_lines_unresolved_call_has_no_outcome() -> None:
    lines = [
        "INFO:orion.bus.async:[rpc] publish begin corr_id=xyz request_channel=orion:test:request "
        "reply_channel=orion:test:result:xyz path=worker elapsed_ms=1.0",
    ]
    records = mod.parse_log_lines(lines)
    rec = records["xyz"]
    assert rec.outcome is None
    assert rec.latency_ms is None


def test_channel_prefix_strips_trailing_uuid() -> None:
    channel = "orion:exec:result:RecallService:b1a6815d-6ddf-4703-9d5f-35abe5a30f80"
    assert mod._channel_prefix(channel) == "orion:exec:result:RecallService"


def test_channel_prefix_leaves_non_uuid_channel_untouched() -> None:
    channel = "orion:cortex:exec:request:background"
    assert mod._channel_prefix(channel) == channel


def test_compute_baseline_counts_success_and_timeout() -> None:
    lines = _worker_success_lines("a", 100.0) + _timeout_lines("b", 5000.0) + _worker_success_lines("c", 200.0)
    records = mod.parse_log_lines(lines)
    baseline = mod.compute_baseline(records)
    assert baseline.n_calls == 3
    assert baseline.n_success == 2
    assert baseline.n_timeout == 1
    assert baseline.n_unresolved == 0
    assert baseline.timeout_rate == 1 / 3


def test_compute_baseline_success_latency_by_path() -> None:
    lines = _worker_success_lines("a", 100.0)
    records = mod.parse_log_lines(lines)
    baseline = mod.compute_baseline(records)
    assert baseline.success_latency_by_path == {"worker": [100.0]}


def test_compute_baseline_keeps_timeout_elapsed_separate_from_success_latency() -> None:
    """A timeout's elapsed_ms is that caller's own timeout_sec ceiling, not real
    latency -- must never appear in success_latency_ms_all/success_latency_by_path."""
    lines = _worker_success_lines("a", 100.0) + _timeout_lines("b", 420000.0)
    records = mod.parse_log_lines(lines)
    baseline = mod.compute_baseline(records)
    assert baseline.success_latency_ms_all == [100.0]
    assert baseline.timeout_elapsed_ms_all == [420000.0]
    assert 420000.0 not in baseline.success_latency_ms_all
    assert baseline.success_latency_by_path == {"worker": [100.0]}


def test_compute_baseline_channel_prefix_counts() -> None:
    lines = (
        _worker_success_lines("a", 100.0, request_channel="orion:foo:request")
        + _worker_success_lines("b", 100.0, request_channel="orion:foo:request")
        + _worker_success_lines("c", 100.0, request_channel="orion:bar:request")
    )
    records = mod.parse_log_lines(lines)
    baseline = mod.compute_baseline(records)
    assert baseline.channel_prefix_counts == {"orion:foo:request": 2, "orion:bar:request": 1}


def test_percentile_matches_conventional_p50() -> None:
    assert mod._percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.5) == 3.0


def test_percentile_empty_returns_none() -> None:
    assert mod._percentile([], 0.5) is None


def test_render_report_flags_no_latency_data() -> None:
    baseline = mod.compute_baseline({})
    report = mod.render_report(since="24h", containers=("c1",), baseline=baseline)
    assert "No successful calls with latency data" in report


def test_render_report_labels_timeout_elapsed_distinctly_from_latency() -> None:
    lines = _timeout_lines("a", 420000.0)
    records = mod.parse_log_lines(lines)
    baseline = mod.compute_baseline(records)
    report = mod.render_report(since="24h", containers=("c1",), baseline=baseline)
    assert "Timeout elapsed-time (NOT latency" in report
    assert "420000.0" not in report.split("## Success latency distribution")[1].split("## Timeout")[0]


def test_render_report_notes_cross_service_coverage_with_multiple_channels() -> None:
    lines = (
        _worker_success_lines("a", 100.0, request_channel="orion:foo:request")
        + _worker_success_lines("b", 100.0, request_channel="orion:bar:request")
    )
    records = mod.parse_log_lines(lines)
    baseline = mod.compute_baseline(records)
    report = mod.render_report(since="24h", containers=("c1", "c2"), baseline=baseline)
    assert "genuinely cross-service" in report
    assert "2 distinct real request channels" in report


def test_fetch_container_logs_handles_missing_container_gracefully() -> None:
    # A container name that certainly doesn't exist -- must not raise.
    lines = mod.fetch_container_logs("definitely-not-a-real-container-name-xyz", "1h")
    assert isinstance(lines, list)
