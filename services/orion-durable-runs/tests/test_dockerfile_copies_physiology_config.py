"""Every config/physiology file durable-runs reads must be COPYed into its image.

Regression: rest_body.yaml (PR #2992) was read by app/physiology_report.py but never copied, so the
live report silently fell back to its built-in defaults and said "not present yet" (2026-10-30).
"""
from __future__ import annotations

import re
from pathlib import Path

SERVICE = Path(__file__).resolve().parents[1]
REPO = SERVICE.parents[1]


def test_every_physiology_config_read_by_the_app_is_copied() -> None:
    dockerfile = (SERVICE / "Dockerfile").read_text(encoding="utf-8")
    copied = set(re.findall(r"COPY config/physiology/(\S+\.yaml)", dockerfile))
    sources = [*SERVICE.joinpath("app").rglob("*.py"), REPO / "orion/physiology/rest_body.py"]
    read = set()
    for path in sources:
        read.update(re.findall(r"""["'](?:config/physiology/)?(\w+\.yaml)["']""", path.read_text(encoding="utf-8")))
    read &= {p.name for p in (REPO / "config/physiology").glob("*.yaml")}
    assert read, "found no config/physiology reads; the scan is broken"
    assert not read - copied, f"read by durable-runs but not copied into its image: {sorted(read - copied)}"
