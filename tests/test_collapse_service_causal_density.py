from __future__ import annotations

import json
from pathlib import Path

import pytest

from orion.collapse.service import (
    CollapseMirrorStore,
    create_entry_from_v2,
    score_causal_density_with_self_state,
)
from orion.schemas.self_state import SelfStateDimensionV1, SelfStateV1


def _store(tmp_path: Path) -> CollapseMirrorStore:
    return CollapseMirrorStore(str(tmp_path / "collapse_mirror_store.json"))


def _base_entry_payload(*, observer: str, source_service: str | None, numeric_overrides: dict | None = None) -> dict:
    payload = {
        "observer": observer,
        "trigger": "test trigger",
        "type": "test",
        "emergent_entity": "test entity",
        "summary": "test summary",
        "mantra": "test mantra",
        "numeric_sisters": {
            "valence": 0.1,
            "arousal": 0.1,
            "clarity": 0.1,
            "overload": 0.1,
            "risk_score": 0.1,
        },
    }
    if numeric_overrides:
        payload["numeric_sisters"].update(numeric_overrides)
    if source_service:
        payload["source_service"] = source_service
    return payload


def _steady_self_state() -> SelfStateV1:
    return SelfStateV1(
        self_state_id="ss-1",
        generated_at="2026-07-01T00:00:00+00:00",
        source_field_tick_id="tick-1",
        source_field_generated_at="2026-07-01T00:00:00+00:00",
        source_attention_frame_id="frame-1",
        source_attention_generated_at="2026-07-01T00:00:00+00:00",
        overall_condition="steady",
        overall_intensity=0.3,
        overall_confidence=0.8,
        dimensions={"execution_pressure": SelfStateDimensionV1(dimension_id="execution_pressure", score=0.2, confidence=0.8)},
        prediction_error_scores={"execution_pressure": 0.02},
        trajectory_condition="stable",
    )


def _unstable_self_state() -> SelfStateV1:
    return SelfStateV1(
        self_state_id="ss-2",
        generated_at="2026-07-01T00:05:00+00:00",
        source_field_tick_id="tick-2",
        source_field_generated_at="2026-07-01T00:05:00+00:00",
        source_attention_frame_id="frame-2",
        source_attention_generated_at="2026-07-01T00:05:00+00:00",
        overall_condition="unstable",
        overall_intensity=0.9,
        overall_confidence=0.7,
        dimensions={"execution_pressure": SelfStateDimensionV1(dimension_id="execution_pressure", score=0.85, confidence=0.7)},
        prediction_error_scores={"execution_pressure": 0.62},
        trajectory_condition="degrading",
    )


def test_strict_lane_score_unchanged_with_or_without_self_state(tmp_path, monkeypatch):
    import orion.collapse.service as svc
    monkeypatch.setattr(svc, "_get_store", lambda: _store(tmp_path))

    entry = create_entry_from_v2(
        _base_entry_payload(observer="Juniper", source_service=None, numeric_overrides={"risk_score": 0.9}),
    )

    without_self_state = score_causal_density_with_self_state(entry.event_id, self_state=None)
    score_a = without_self_state.causal_density.score

    with_self_state = score_causal_density_with_self_state(entry.event_id, self_state=_unstable_self_state())
    score_b = with_self_state.causal_density.score

    assert score_a == score_b, "strict-lane entries must not be affected by self_state at all"


def test_metacog_lane_high_self_report_but_steady_self_state_pulls_score_down(tmp_path, monkeypatch):
    import orion.collapse.service as svc
    monkeypatch.setattr(svc, "_get_store", lambda: _store(tmp_path))

    entry = create_entry_from_v2(
        _base_entry_payload(
            observer="Orion",
            source_service="metacog",
            numeric_overrides={"valence": 0.95, "arousal": 0.95, "risk_score": 0.95},
        ),
    )
    self_report_only = score_causal_density_with_self_state(entry.event_id, self_state=None)
    blended = score_causal_density_with_self_state(entry.event_id, self_state=_steady_self_state())

    assert blended.causal_density.score < self_report_only.causal_density.score
    assert blended.causal_density.label in {"salient", "ambient"}


def test_metacog_lane_modest_self_report_but_severe_self_state_pulls_score_up(tmp_path, monkeypatch):
    import orion.collapse.service as svc
    monkeypatch.setattr(svc, "_get_store", lambda: _store(tmp_path))

    entry = create_entry_from_v2(
        _base_entry_payload(
            observer="Orion",
            source_service="metacog",
            numeric_overrides={"valence": 0.2, "arousal": 0.2, "risk_score": 0.2},
        ),
    )
    self_report_only = score_causal_density_with_self_state(entry.event_id, self_state=None)
    blended = score_causal_density_with_self_state(entry.event_id, self_state=_unstable_self_state())

    assert blended.causal_density.score > self_report_only.causal_density.score


def test_metacog_lane_accepts_dict_self_state_like_real_caller(tmp_path, monkeypatch):
    """The real production caller (ScoreCausalDensityVerb.execute) passes a plain
    dict read back from JSONB storage, never a SelfStateV1 instance. This exercises
    the _coerce_self_state dict path end-to-end."""
    import orion.collapse.service as svc
    monkeypatch.setattr(svc, "_get_store", lambda: _store(tmp_path))

    entry = create_entry_from_v2(
        _base_entry_payload(
            observer="Orion",
            source_service="metacog",
            numeric_overrides={"valence": 0.2, "arousal": 0.2, "risk_score": 0.2},
        ),
    )
    self_report_only = score_causal_density_with_self_state(entry.event_id, self_state=None)
    blended = score_causal_density_with_self_state(
        entry.event_id, self_state=_unstable_self_state().model_dump(mode="json")
    )

    assert blended.causal_density.score > self_report_only.causal_density.score


def test_metacog_lane_malformed_self_state_falls_back_to_self_report_only(tmp_path, monkeypatch):
    """Schema drift or garbage payloads must fail open to pure self-report, never raise."""
    import orion.collapse.service as svc
    monkeypatch.setattr(svc, "_get_store", lambda: _store(tmp_path))

    entry = create_entry_from_v2(
        _base_entry_payload(
            observer="Orion",
            source_service="metacog",
            numeric_overrides={"valence": 0.2, "arousal": 0.2, "risk_score": 0.2},
        ),
    )
    self_report_only = score_causal_density_with_self_state(entry.event_id, self_state=None)

    malformed_dict = score_causal_density_with_self_state(
        entry.event_id, self_state={"garbage": "not a valid self state", "extra_field": 123}
    )
    assert malformed_dict.causal_density.score == self_report_only.causal_density.score

    malformed_str = score_causal_density_with_self_state(
        entry.event_id, self_state="not json at all"
    )
    assert malformed_str.causal_density.score == self_report_only.causal_density.score
