"""The visual-baseline schedule file moved to config/reverie/ (ladder retirement PR 6, 2026-11-07).

``load_baseline_policy`` falls back to the model defaults (``enabled=False``) when the file is
missing, so a wrong path would silently stop every waking painting. Pin that the shipped file is
found and read, and that both images that read it copy it from the new path.
"""
from __future__ import annotations

from pathlib import Path

from orion.reverie.baseline import load_baseline_policy

REPO = Path(__file__).resolve().parents[3]


def test_shipped_baseline_file_is_found_and_enabled():
    policy = load_baseline_policy()
    assert policy.enabled is True
    assert policy.interval_sec == 5400
    assert policy.freshness_sec == 900


def test_images_copy_the_baseline_file_from_its_new_home():
    thought = (REPO / "services/orion-thought/Dockerfile").read_text()
    durable = (REPO / "services/orion-durable-runs/Dockerfile").read_text()
    assert "COPY config/reverie ./config/reverie" in thought
    assert "config/reverie/visual_baseline.v1.yaml /app/config/reverie/visual_baseline.v1.yaml" in durable
    assert "config/proposals" not in thought + durable
