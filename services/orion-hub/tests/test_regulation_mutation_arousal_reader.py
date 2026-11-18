"""R3 reader 6 in Hub: the scheduled self-modification cycle reads arousal; the manual one never does.

Named test_regulation_* so it is collected after the curiosity tests (Hub's conftest purges
`scripts.*` per test; see the #2624 report).
"""
from __future__ import annotations

import pytest

from orion.regulation.arousal_view import UNKNOWN_DISABLED, ArousalView
from orion.substrate.mutation_queue import SubstrateMutationStore
from scripts import api_routes


@pytest.fixture
def captured_workers(monkeypatch):
    seen = []
    real = api_routes.SubstrateAdaptationWorker

    class Capture(real):  # type: ignore[misc, valid-type]
        def __init__(self, *a, **kw):
            seen.append(kw)
            super().__init__(*a, **kw)

    monkeypatch.setattr(api_routes, "SubstrateAdaptationWorker", Capture)
    monkeypatch.setattr(api_routes, "SUBSTRATE_MUTATION_STORE", SubstrateMutationStore())
    monkeypatch.setattr(api_routes, "substrate_autonomy_runtime_supported", lambda: (True, "supported"))
    monkeypatch.setattr(api_routes, "_mutation_arousal_level", lambda: None)  # never a live Redis read
    return seen


def test_scheduled_cycle_wires_the_reader(monkeypatch, captured_workers):
    monkeypatch.setenv("SUBSTRATE_AUTONOMY_ENABLED", "true")
    monkeypatch.setenv("SUBSTRATE_AUTONOMY_APPLY_ENABLED", "false")
    payload = api_routes.execute_substrate_mutation_scheduled_cycle(telemetry_override=[])
    assert payload["status"] == "completed"
    assert captured_workers and captured_workers[-1].get("arousal_level") is api_routes._mutation_arousal_level
    assert payload["summary"]["proposals_held_arousal"] == 0


def test_must_not_manual_cycle_never_reads_arousal(monkeypatch, captured_workers):
    """An operator-triggered cycle is Juniper's own act; arousal must not hold it."""
    api_routes._execute_substrate_mutation_cycle(request=api_routes.SubstrateMutationExecuteRequest())
    assert captured_workers and "arousal_level" not in captured_workers[-1]


def _patch_read(monkeypatch, view):
    import orion.regulation.arousal_read as ar

    calls = []

    def fake(url, *, enabled, reader, max_age_sec, **_kw):
        calls.append((enabled, reader, max_age_sec))
        return view if enabled else UNKNOWN_DISABLED

    monkeypatch.setattr(ar, "read_arousal_sync", fake)
    return calls


def test_reader_returns_strained(monkeypatch):
    calls = _patch_read(monkeypatch, ArousalView(level="strained", reason="strained"))
    monkeypatch.delenv("SUBSTRATE_AUTONOMY_AROUSAL_ENABLED", raising=False)  # ships on
    assert api_routes._mutation_arousal_level() == "strained"
    assert calls[-1][0] is True and calls[-1][1] == "mutation_pressure"


def test_reader_flag_off_and_unknown_are_none(monkeypatch):
    _patch_read(monkeypatch, ArousalView(level="strained", reason="strained"))
    monkeypatch.setenv("SUBSTRATE_AUTONOMY_AROUSAL_ENABLED", "false")
    assert api_routes._mutation_arousal_level() is None
    _patch_read(monkeypatch, ArousalView(level="unknown", reason="stale"))
    monkeypatch.setenv("SUBSTRATE_AUTONOMY_AROUSAL_ENABLED", "true")
    assert api_routes._mutation_arousal_level() is None


def test_settings_ship_the_flag_on():
    from scripts.settings import settings

    assert type(settings).model_fields["SUBSTRATE_AUTONOMY_AROUSAL_ENABLED"].default is True
