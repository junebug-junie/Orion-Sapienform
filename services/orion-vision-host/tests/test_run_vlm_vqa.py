from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.profiles import VisionProfiles
from app.runner import VisionRunner


def _load_profiles() -> VisionProfiles:
    cfg_path = Path(__file__).resolve().parents[3] / "config" / "vision_profiles.yaml"
    p = VisionProfiles(str(cfg_path))
    p.load()
    return p


def _runner() -> VisionRunner:
    profiles = _load_profiles()
    tmp = tempfile.mkdtemp()
    return VisionRunner(profiles=profiles, enabled_names=["vlm_vqa", "vlm_caption"], cache_dir=tmp)


def test_run_profile_routes_vlm_kind_to_vqa_handler(monkeypatch) -> None:
    """Confirms _run_profile's dispatch, not just that _run_vlm_vqa exists in
    isolation -- a prior version of this kind hit the generic "kind not
    implemented yet" fallback (see runner.py's own comment); this pins that
    the real handler is reached instead."""
    runner = _runner()
    profile = runner.profiles.get_profile("vlm_vqa")

    called = {}

    def fake_vqa(p, request, device, warnings):
        called["hit"] = True
        return {"kind": "vlm", "implemented": True}

    monkeypatch.setattr(runner, "_run_vlm_vqa", fake_vqa)
    result = runner._run_profile(profile, {"question": "is the door open?"}, "cpu", [])
    assert called.get("hit") is True
    assert result["implemented"] is True


def test_run_vlm_vqa_requires_a_question() -> None:
    """Validation happens before image load / model load -- this must raise
    on a missing/blank question without touching the filesystem or any
    model, so it's testable without mocking transformers."""
    runner = _runner()
    profile = runner.profiles.get_profile("vlm_vqa")

    with pytest.raises(ValueError, match="question"):
        runner._run_vlm_vqa(profile, {}, "cpu", [])

    with pytest.raises(ValueError, match="question"):
        runner._run_vlm_vqa(profile, {"question": "   "}, "cpu", [])


def test_run_vlm_vqa_missing_question_error_is_not_misclassified_as_missing_image() -> None:
    """execute()'s own error-code classification is
    `"missing_image_path" if "image_path" in msg else "request_validation"`
    -- the ValueError message for a missing question must not accidentally
    contain the substring "image_path", or it would surface as the wrong
    error_code to callers."""
    runner = _runner()
    profile = runner.profiles.get_profile("vlm_vqa")
    try:
        runner._run_vlm_vqa(profile, {}, "cpu", [])
        assert False, "expected ValueError"
    except ValueError as e:
        assert "image_path" not in str(e)


def test_warm_profiles_kind_allowlist_includes_vlm(monkeypatch) -> None:
    """Review finding, 2026-08-21: adding the `kind == "vlm"` branch to
    `_warm_profile_backend` alone was not enough -- `warm_profiles()`'s own
    loop has a SECOND, separate kind-allowlist tuple that filtered "vlm" out
    before `_warm_profile_backend` was ever called, so flipping a real
    profile's `warm_on_start: true` later would have silently still not
    warmed it. Confirms the real `warm_profiles()` loop (not just the
    allowlist tuple in isolation) actually reaches `_warm_profile_backend`
    for a `kind == "vlm"` profile once both gates (`p.enabled` /
    `p.warm_on_start`) agree to warm it -- vlm_vqa itself still ships with
    `warm_on_start: false` today, so this monkeypatches that one flag to
    exercise the path without changing real config."""
    runner = _runner()
    profile = runner.profiles.get_profile("vlm_vqa")
    assert profile.kind == "vlm"
    # ProfileDef is a plain (non-frozen) dataclass -- direct mutation is
    # fine, only exercises this one test's own loaded VisionProfiles
    # instance, not real config.
    profile.warm_on_start = True

    monkeypatch.setattr(runner, "_is_enabled", lambda name: name == "vlm_vqa")
    # settings.devices already defaults to ["cuda:0"] (VISION_DEFAULT_DEVICE/
    # VISION_DEVICES both default to "cuda:0") -- no monkeypatch needed for
    # warm_profiles() to pass its own `device.startswith("cuda")` guard.

    called = {}

    def fake_warm_backend(p, device):
        called["kind"] = p.kind
        called["name"] = p.name

    monkeypatch.setattr(runner, "_warm_profile_backend", fake_warm_backend)
    warmed = runner.warm_profiles()

    assert called.get("kind") == "vlm"
    assert called.get("name") == "vlm_vqa"
    assert "vlm_vqa" in warmed
