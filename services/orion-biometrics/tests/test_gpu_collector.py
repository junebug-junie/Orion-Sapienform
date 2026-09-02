"""Regression coverage for `collect_gpu_stats()`'s process-list attachment (app/utils.py).

Added alongside the Hub Biometrics view's GPU-processes section. `gpu_host_stats.sh`
writes a sibling "<ts>.procs.csv" next to the main "<ts>.csv" -- both end in ".csv",
so the main-file glob must exclude the sibling explicitly rather than relying on
filename ordering, and the process attachment must degrade to an empty list per
row (never raise) when the sibling file is missing or unreadable.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

import app.utils as utils_module
from app.utils import collect_gpu_stats


@pytest.fixture(autouse=True)
def _telemetry_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(utils_module, "TELEMETRY_DIR", str(tmp_path))
    monkeypatch.setattr(utils_module, "LOG_FILE", str(tmp_path / "logs" / "error.log"))
    return tmp_path


def _fake_script(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand in for the real /orion/sensors/gpu_host_stats.sh subprocess call."""

    def _noop_run(*args, **kwargs):
        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr(utils_module.subprocess, "run", _noop_run)
    monkeypatch.setattr(utils_module.time, "sleep", lambda *_a, **_k: None)


def _write_gpu_csv(tmp_path: Path, stem: str) -> None:
    (tmp_path / f"{stem}.csv").write_text(
        "timestamp,gpu_index,gpu_uuid,gpu_name,utilization_gpu,memory_used_mb,memory_total_mb,power_draw_watts\n"
        f"{stem},0,GPU-aaa,Tesla P4,8,512,7680,22.1\n"
        f"{stem},1,GPU-bbb,V100-PCIE-32GB,95,30000,32768,180.4\n",
        encoding="utf-8",
    )


def _write_procs_csv(tmp_path: Path, stem: str) -> None:
    (tmp_path / f"{stem}.procs.csv").write_text(
        "gpu_uuid,pid,process_name,used_memory_mb\n"
        "GPU-bbb,12345,python3,29800\n"
        "GPU-bbb,12399,python3,150\n",
        encoding="utf-8",
    )


def test_processes_attached_to_correct_gpu_row_by_uuid(monkeypatch, tmp_path):
    _fake_script(monkeypatch)
    _write_gpu_csv(tmp_path, "2026-09-02T00:00:00")
    _write_procs_csv(tmp_path, "2026-09-02T00:00:00")

    result = collect_gpu_stats()

    assert result["gpus"][0]["gpu_uuid"] == "GPU-aaa"
    assert result["gpus"][0]["processes"] == []
    assert result["gpus"][1]["gpu_uuid"] == "GPU-bbb"
    assert len(result["gpus"][1]["processes"]) == 2
    assert result["gpus"][1]["processes"][0]["pid"] == "12345"
    assert result["gpus"][1]["processes"][0]["process_name"] == "python3"


def test_missing_procs_file_degrades_to_empty_list_not_raise(monkeypatch, tmp_path):
    _fake_script(monkeypatch)
    _write_gpu_csv(tmp_path, "2026-09-02T00:00:00")
    # Deliberately no .procs.csv written.

    result = collect_gpu_stats()

    assert "error" not in result
    for gpu in result["gpus"]:
        assert gpu["processes"] == []


def test_procs_csv_never_mistaken_for_the_main_gpu_file(monkeypatch, tmp_path):
    """A .procs.csv with a later mtime must not be selected as the main GPU file."""
    _fake_script(monkeypatch)
    _write_gpu_csv(tmp_path, "2026-09-02T00:00:00")
    # procs file written after, so it would win a naive "latest by mtime" scan
    # over files ending in plain ".csv" if the filter didn't exclude it.
    time.sleep(0.01)
    _write_procs_csv(tmp_path, "2026-09-02T00:00:00")

    result = collect_gpu_stats()

    assert result["latest_file"] == "2026-09-02T00:00:00.csv"
    assert len(result["gpus"]) == 2
    assert result["gpus"][0]["gpu_name"] == "Tesla P4"
