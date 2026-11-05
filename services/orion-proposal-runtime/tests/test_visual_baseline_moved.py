"""Ladder retirement PR 2 (2026-11-05): painting has exactly one trigger, and it is not here.

Fails on pre-PR-2 code, where this runtime read orion-thought's /visual-chain/activity on every
tick, ran the baseline schedule on visual_baseline_checkpoint, and emitted the render_scene template.
The schedule now runs in orion-thought (app/painting_self_start.py)."""
from __future__ import annotations

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]


def test_the_runtime_no_longer_schedules_paintings():
    import app.worker as wm
    from app.settings import Settings
    from app.store import ProposalRuntimeStore

    src = Path(wm.__file__).read_text()
    assert "visual-chain/activity" not in src and "load_baseline_policy" not in src
    assert not hasattr(ProposalRuntimeStore, "baseline_eligibility")
    assert "visual_baseline_arousal_enabled" not in Settings.model_fields


def test_the_policy_has_no_painting_template():
    policy = yaml.safe_load((REPO / "config/proposals/proposal_policy.v1.yaml").read_text())
    assert "render_scene" not in policy["proposal_templates"]
