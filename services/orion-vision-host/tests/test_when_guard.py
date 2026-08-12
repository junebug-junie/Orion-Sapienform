from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.when_guard import safe_when

REPO_ROOT = Path(__file__).resolve().parents[3]
PROFILES_YAML = REPO_ROOT / "config" / "vision_profiles.yaml"


def test_missing_key_is_falsy_not_an_error() -> None:
    """
    Regression: the live bug. Every task logged
      [PIPE] when eval failed expr=request.is_video == True
      err='types.SimpleNamespace' object has no attribute 'is_video'
    because the baseline request never carries `is_video`. A guard on an
    absent flag must evaluate cleanly to False, not raise.
    """
    assert safe_when("request.is_video == true", {"image_path": "/x.jpg"}) is False


def test_missing_key_negated_guard_is_true() -> None:
    # An absent flag reads as None, so `!= true` holds. This is the branch a
    # swallowed AttributeError got wrong: it returned False for both polarities.
    assert safe_when("request.is_video != true", {}) is True


def test_present_true_flag_runs_step() -> None:
    assert safe_when("request.is_video == true", {"is_video": True}) is True


def test_present_false_flag_skips_step() -> None:
    assert safe_when("request.want_embeddings == true", {"want_embeddings": False}) is False


def test_empty_guard_always_runs() -> None:
    assert safe_when("", {}) is True


def test_malformed_expression_still_skips_and_does_not_raise() -> None:
    assert safe_when("request.is_video ===", {}) is False


def test_builtins_are_not_reachable_from_a_guard() -> None:
    assert safe_when("__import__('os').system('true') == 0", {}) is False


def test_true_substring_in_a_key_is_not_mangled() -> None:
    # Whole-word replacement only; `want_true_color` must survive intact.
    assert safe_when("request.want_true_color == true", {"want_true_color": True}) is True


def test_every_guard_in_shipped_profiles_evaluates_cleanly() -> None:
    """
    Contract check against the real config: no `when:` expression in any
    enabled pipeline may error on a bare request. Catches a new guard being
    added that names a flag nothing sends.
    """
    cfg = yaml.safe_load(PROFILES_YAML.read_text())
    guards = [
        step["when"]
        for pipe in cfg.get("pipelines", [])
        for step in pipe.get("steps", [])
        if step.get("when")
    ]
    assert guards, "expected shipped pipelines to carry `when:` guards"

    baseline_request = {"image_path": "/tmp/frame.jpg", "stream_id": "cam0"}
    for expr in guards:
        # False is fine (flag not requested); an exception is not.
        assert safe_when(expr, baseline_request) is False, expr
