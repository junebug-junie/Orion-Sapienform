"""Causal parent miss notes (spec §7.B)."""
from datetime import datetime, timezone

from orion.signals.causal_helpers import with_missed_parent_notes
from orion.signals.models import OrganClass, OrionSignalV1
from orion.signals.registry import ORGAN_REGISTRY


def _sig(organ_id: str, *, oclass: OrganClass = OrganClass.endogenous) -> OrionSignalV1:
    now = datetime.now(timezone.utc)
    return OrionSignalV1(
        signal_id="s1",
        organ_id=organ_id,
        organ_class=oclass,
        signal_kind="mesh_health",
        dimensions={"level": 0.5},
        observed_at=now,
        emitted_at=now,
    )


def test_no_note_when_parents_present():
    prior = {"biometrics": _sig("biometrics", oclass=OrganClass.exogenous)}
    eq = _sig("equilibrium")
    out = with_missed_parent_notes(eq, prior, ORGAN_REGISTRY)
    assert not any("missed causal link" in n for n in out.notes)


def test_note_when_parent_missing():
    eq = _sig("equilibrium")
    out = with_missed_parent_notes(eq, {}, ORGAN_REGISTRY)
    assert any("missed causal link" in n for n in out.notes)
    assert "biometrics" in out.notes[-1]
