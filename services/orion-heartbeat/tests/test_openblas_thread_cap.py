"""Regression: the substrate's tiny matrices must not use a threaded OpenBLAS.

Live 2026-10-30: default OpenBLAS threads made heartbeat burst to 1,600% CPU.
Benchmarked in the heartbeat image: same wall time, 16x the CPU.
"""

from __future__ import annotations

from pathlib import Path


def test_dockerfile_pins_openblas_to_one_thread() -> None:
    dockerfile = (Path(__file__).resolve().parents[1] / "Dockerfile").read_text()
    lines = [line.strip() for line in dockerfile.splitlines()]
    assert "ENV OPENBLAS_NUM_THREADS=1" in lines
    # Must be set before the app is copied/run, i.e. in the image environment.
    cmd_index = next(i for i, line in enumerate(lines) if line.startswith("CMD"))
    assert lines.index("ENV OPENBLAS_NUM_THREADS=1") < cmd_index
