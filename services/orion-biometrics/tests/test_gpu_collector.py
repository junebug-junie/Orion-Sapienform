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


def test_real_script_carries_gpu_temperature_into_measurements(monkeypatch, tmp_path):
    """End to end through the real gpu_host_stats.sh (fake nvidia-smi on PATH): the new last
    column reaches collect_gpu_stats() by header name and extract_measurements() per card."""
    import subprocess as real_subprocess

    from orion.telemetry.biometrics_pipeline import extract_measurements

    script = Path(__file__).resolve().parents[3] / "orion" / "sensors" / "gpu_host_stats.sh"
    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake = bindir / "nvidia-smi"
    # Same line shape as circe's real output 2026-09-29.
    fake.write_text(
        "#!/bin/bash\n"
        "case \"$*\" in\n"
        "  *query-gpu*) echo '0, GPU-aaa, Tesla V100-PCIE-32GB, 0, 27888, 32768, 36.91, 39'\n"
        "               echo '3, GPU-ddd, Tesla V100-PCIE-32GB, 100, 15384, 32768, 210.68, 71' ;;\n"
        "  *) : ;;\n"
        "esac\n", encoding="utf-8")
    fake.chmod(0o755)
    out = tmp_path / "out"
    env = {"PATH": f"{bindir}:{os.environ.get('PATH', '')}", "GPU_STATS_OUTDIR": str(out)}
    real_subprocess.run(["bash", str(script)], check=True, env=env)
    monkeypatch.setattr(utils_module, "TELEMETRY_DIR", str(out))
    _fake_script(monkeypatch)

    result = collect_gpu_stats()
    assert [g["temperature_gpu_c"].strip() for g in result["gpus"]] == ["39", "71"]
    m = extract_measurements({"gpu": result})
    assert m["gpu0_temp_c"] == 39.0 and m["gpu3_temp_c"] == 71.0 and m["gpu_temp_c_max"] == 71.0
