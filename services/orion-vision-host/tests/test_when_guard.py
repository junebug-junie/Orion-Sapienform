from __future__ import annotations

import sys
from pathlib import Path

import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import when_guard
from app.when_guard import safe_when

REPO_ROOT = Path(__file__).resolve().parents[3]
PROFILES_YAML = REPO_ROOT / "config" / "vision_profiles.yaml"


class _CaptureLogs:
    """Collect loguru output emitted inside the block."""

    def __init__(self) -> None:
        self.messages: list[str] = []
        self._sink_id: int | None = None

    def __enter__(self) -> "_CaptureLogs":
        self._sink_id = logger.add(self.messages.append, level="DEBUG")
        return self

    def __exit__(self, *exc: object) -> None:
        if self._sink_id is not None:
            logger.remove(self._sink_id)

    @property
    def text(self) -> str:
        return "\n".join(self.messages)


def test_absent_flag_evaluates_cleanly_and_logs_nothing() -> None:
    """
    Regression for the live bug. Every task logged
      [PIPE] when eval failed expr=request.is_video == True
      err='types.SimpleNamespace' object has no attribute 'is_video'
    347 times per 30 minutes, because the baseline request never carries
    `is_video`.

    Asserting on the return value alone is not enough -- the old code also
    returned False here, by raising and swallowing. The distinguishing
    evidence is that nothing is logged, so this test fails against the
    SimpleNamespace implementation it was written for.
    """
    with _CaptureLogs() as logs:
        result = safe_when("request.is_video == true", {"image_path": "/x.jpg"})
    assert result is False
    assert "when eval failed" not in logs.text
    assert logs.text == ""


def test_absent_flag_negated_guard_is_true() -> None:
    # An absent flag reads as None, so `!= true` holds. The old code returned
    # False for both polarities because the exception short-circuited it.
    assert safe_when("request.is_video != true", {}) is True


def test_present_true_flag_runs_step() -> None:
    assert safe_when("request.is_video == true", {"is_video": True}) is True


def test_present_false_flag_skips_step() -> None:
    assert safe_when("request.want_embeddings == true", {"want_embeddings": False}) is False


def test_empty_guard_always_runs() -> None:
    assert safe_when("", {}) is True


def test_unknown_flag_warns_once_so_a_rename_is_not_silent() -> None:
    """
    A guard naming a flag nothing sends is indistinguishable from a
    legitimately-unset one at the value level, so an unrecognised name warns.
    Once per process, not per task -- per-task was the original bug's cost.
    """
    when_guard._warned_unknown_flags.discard("want_capton")
    with _CaptureLogs() as first:
        safe_when("request.want_capton == true", {"want_caption": True})
    with _CaptureLogs() as second:
        safe_when("request.want_capton == true", {"want_caption": True})

    assert "want_capton" in first.text
    assert second.text == "", "should warn once per process, not per task"


def test_known_flags_do_not_warn() -> None:
    with _CaptureLogs() as logs:
        for flag in sorted(when_guard.KNOWN_GUARD_FLAGS):
            safe_when(f"request.{flag} == true", {})
    assert logs.text == ""


def test_malformed_expression_skips_and_does_not_raise() -> None:
    with _CaptureLogs() as logs:
        assert safe_when("request.is_video ===", {}) is False
    assert "when eval failed" in logs.text


def test_guard_language_rejects_calls_and_arbitrary_names() -> None:
    """
    Guards come from repo-controlled YAML, but the evaluator is restricted to
    the guard language rather than relying on `{"__builtins__": {}}`, which is
    not a sandbox: `().__class__.__bases__[0].__subclasses__()` escapes it.
    """
    escapes = [
        "__import__('os').system('true') == 0",
        "().__class__.__bases__[0].__subclasses__() == 1",
        "request.want_caption.__class__ == 1",
        "open('/etc/passwd') == 1",
    ]
    for expr in escapes:
        assert safe_when(expr, {}) is False, expr


def test_string_literals_are_not_rewritten() -> None:
    # The earlier textual true/false substitution turned the literal "true"
    # into "True" and broke this comparison.
    assert safe_when('request.mode == "true"', {"mode": "true"}) is True


def test_every_guard_in_shipped_profiles_evaluates_cleanly() -> None:
    """
    Contract check against the real config: no `when:` expression in any
    shipped pipeline may error, and none may name an unrecognised flag.
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
        with _CaptureLogs() as logs:
            safe_when(expr, baseline_request)
        assert "when eval failed" not in logs.text, expr
        assert "unknown request flag" not in logs.text, expr
