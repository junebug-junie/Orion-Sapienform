"""Tests for LLM surface instability metacog advisory trigger."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

_guard_path = Path(__file__).resolve().parent / "_mind_import_guard.py"


def _mind_prep() -> None:
    spec = importlib.util.spec_from_file_location("_mind_guard_lazy", _guard_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ensure_orion_mind_app()


@pytest.fixture(autouse=True)
def _prep() -> None:
    _mind_prep()


def _unc(**overrides: Any) -> dict[str, Any]:
    base = {
        "schema_version": "v1",
        "available": True,
        "token_count_observed": 20,
        "low_logprob_token_count": 1,
        "unstable_span_count": 0,
        "mean_top1_margin": 0.9,
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize(
    "unc,expected_emit,expected_detail",
    [
        (_unc(available=False), False, "unavailable"),
        (_unc(token_count_observed=0), False, "no_tokens"),
        (_unc(unstable_span_count=1), True, "unstable_span"),
        (_unc(mean_top1_margin=0.5), True, "low_mean_margin"),
        (_unc(low_logprob_token_count=5, token_count_observed=20), True, "high_low_logprob_ratio"),
        (_unc(), False, "stable"),
    ],
)
def test_should_emit_llm_surface_instability(
    unc: dict[str, Any],
    expected_emit: bool,
    expected_detail: str,
) -> None:
    from app.uncertainty_metacog import should_emit_llm_surface_instability

    emit, detail = should_emit_llm_surface_instability(unc)
    assert emit is expected_emit
    assert detail == expected_detail


def test_maybe_publish_skips_when_metacog_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.phase_telemetry import MindPhaseTelemetry
    from app.settings import settings
    from app import uncertainty_metacog

    monkeypatch.setattr(settings, "MIND_LLM_UNCERTAINTY_METACOG_ENABLED", False)
    called = False

    def _fake_publish(*_args: Any, **_kwargs: Any) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(uncertainty_metacog, "_run_blocking", _fake_publish)
    telemetry = MindPhaseTelemetry(
        phase_name="semantic_synthesis",
        route="quick",
        llm_uncertainty=_unc(unstable_span_count=2),
    )
    uncertainty_metacog.maybe_publish_llm_surface_instability_trigger(telemetry)
    assert called is False
