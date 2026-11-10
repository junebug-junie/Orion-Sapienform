"""Regression: the action-outcome surprise stamp stays retired.

Removed 2026-11-10 (prediction-error suite rebuild, retire step 1). Every emitter
used to stamp a success/fail proxy or the ambient bus_synaptic prediction error onto
ActionOutcomeRefV1/ActionOutcomeEmitV1.surprise. Nothing read it. The field stays at
its 0.0 default only because action_outcomes.surprise is NOT NULL.
"""

from __future__ import annotations

from pathlib import Path

from orion.autonomy.models import ActionOutcomeEmitV1, ActionOutcomeRefV1

_REPO = Path(__file__).resolve().parents[3]


def test_outcome_surprise_defaults_to_zero() -> None:
    ref = ActionOutcomeRefV1(action_id="a", kind="k", summary="s", success=False)
    assert ref.surprise == 0.0
    assert ActionOutcomeEmitV1.from_outcome(subject="orion", outcome=ref).surprise == 0.0


def test_stamp_reader_module_is_gone_and_nothing_imports_it() -> None:
    assert not (_REPO / "orion" / "substrate" / "bus_synaptic_surprise.py").exists()
    needle = "orion.substrate." + "bus_synaptic_surprise"
    this = Path(__file__).resolve()
    offenders = []
    for root in ("orion", "services", "scripts"):
        for path in (_REPO / root).rglob("*.py"):
            if path.resolve() == this or "node_modules" in path.parts:
                continue
            if needle in path.read_text(encoding="utf-8", errors="ignore"):
                offenders.append(str(path.relative_to(_REPO)))
    assert offenders == []
