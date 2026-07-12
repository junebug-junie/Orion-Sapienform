from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import pytest

import check_inner_state_registry as gate
from orion.self_state.inner_state_registry import (
    Cadence,
    CompositionStatus,
    InnerStateSignal,
)


def test_rot_check_passes_against_real_registry() -> None:
    # The regression this gate exists to prevent: if this ever fails, a
    # REGISTRY entry's schema has gone stale (renamed/deleted/no longer a
    # BaseModel) and nobody noticed.
    assert gate.rot_check() == []


def test_rot_check_fails_on_a_non_basemodel_schema(monkeypatch) -> None:
    class NotAModel:
        pass

    stale_entry = InnerStateSignal(
        signal_id="test.stale",
        schema=NotAModel,  # type: ignore[arg-type]
        producer_service="test",
        cadence=Cadence.PER_TICK,
        composition_status=CompositionStatus.COMPOSED,
    )
    monkeypatch.setattr(gate, "REGISTRY", gate.REGISTRY + (stale_entry,))

    failures = gate.rot_check()

    assert any("test.stale" in f for f in failures)


def test_new_duplicate_heuristic_passes_against_real_channels_yaml() -> None:
    # Confirms the current, real orion/bus/channels.yaml + orion/schemas/
    # registry.py produce zero unaccounted inner-state-keyword matches, given
    # today's REGISTRY + _EXTRA_COVERED_SCHEMA_NAMES triage.
    assert gate.new_duplicate_heuristic_check(channels_file=gate.DEFAULT_CHANNELS_FILE) == []


def test_new_duplicate_heuristic_fails_on_an_unregistered_new_channel(tmp_path) -> None:
    fixture = tmp_path / "channels.yaml"
    fixture.write_text(
        """
channels:
  - name: "orion:test:mood_state"
    kind: "event"
    schema_id: "MoodStateV1"
    producer_services: ["orion-test-service"]
    consumer_services: ["orion-test-consumer"]
    stability: "experimental"
    since: "2026-07-12"
"""
    )

    failures = gate.new_duplicate_heuristic_check(channels_file=fixture)

    assert any("MoodStateV1" in f for f in failures)


def test_new_duplicate_heuristic_ignores_a_channel_with_no_keyword_match(tmp_path) -> None:
    fixture = tmp_path / "channels.yaml"
    fixture.write_text(
        """
channels:
  - name: "orion:test:unrelated"
    kind: "event"
    schema_id: "UnrelatedPayloadV1"
    producer_services: ["orion-test-service"]
    consumer_services: ["orion-test-consumer"]
    stability: "experimental"
    since: "2026-07-12"
"""
    )

    failures = gate.new_duplicate_heuristic_check(channels_file=fixture)

    assert failures == []


def test_innerstate_signal_requires_duplicate_of_when_duplicate() -> None:
    with pytest.raises(ValueError, match="DUPLICATE status requires duplicate_of"):
        InnerStateSignal(
            signal_id="test.bad",
            schema=None,
            producer_service="test",
            cadence=Cadence.PER_TICK,
            composition_status=CompositionStatus.DUPLICATE,
        )


def test_innerstate_signal_requires_shadow_reason_when_shadow() -> None:
    with pytest.raises(ValueError, match="SHADOW status requires shadow_reason"):
        InnerStateSignal(
            signal_id="test.bad",
            schema=None,
            producer_service="test",
            cadence=Cadence.PER_TICK,
            composition_status=CompositionStatus.SHADOW,
        )


def test_main_exits_zero_against_real_repo() -> None:
    assert gate.main([]) == 0
