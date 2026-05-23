"""Field operators — small functions that mutate gradient state.

These are deliberately simple. No tensors, no learned weights. The harness
records before/after deltas and aggregates them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Iterable

from orion.schema_kernel import clamp_gradient

from .molecules import SubstrateMoleculeV1


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class GradientDelta:
    """Snapshot of a gradient mutation, useful for the experiment harness."""

    molecule_id: str
    before: dict[str, float]
    after: dict[str, float]
    cause: str

    def changed_keys(self) -> tuple[str, ...]:
        keys = []
        for key, value in self.after.items():
            if self.before.get(key, 0.0) != value:
                keys.append(key)
        return tuple(keys)


GradientObserver = Callable[[GradientDelta], None]


def _apply(
    molecule: SubstrateMoleculeV1,
    mutation: Callable[[dict[str, float]], dict[str, float]],
    *,
    cause: str,
    observer: GradientObserver | None,
) -> GradientDelta:
    before = dict(molecule.gradients)
    new_state = mutation(dict(before))
    new_state = {key: clamp_gradient(value) for key, value in new_state.items()}
    molecule.gradients = new_state
    molecule.touch(_utcnow())
    delta = GradientDelta(
        molecule_id=molecule.molecule_id,
        before=before,
        after=dict(new_state),
        cause=cause,
    )
    if observer is not None:
        observer(delta)
    return delta


def reinforce_molecule(
    molecule: SubstrateMoleculeV1,
    *,
    salience_step: float = 0.1,
    coherence_step: float = 0.05,
    observer: GradientObserver | None = None,
) -> GradientDelta:
    """Repeated access nudges salience + coherence upward."""

    def _mut(state: dict[str, float]) -> dict[str, float]:
        state["salience"] = state.get("salience", 0.0) + salience_step
        state["coherence"] = state.get("coherence", 0.0) + coherence_step
        return state

    return _apply(molecule, _mut, cause="reinforce", observer=observer)


def decay_molecule(
    molecule: SubstrateMoleculeV1,
    *,
    salience_step: float = 0.02,
    coherence_step: float = 0.01,
    observer: GradientObserver | None = None,
) -> GradientDelta:
    """Untouched molecules slowly lose salience/coherence."""

    def _mut(state: dict[str, float]) -> dict[str, float]:
        state["salience"] = state.get("salience", 0.0) - salience_step
        state["coherence"] = state.get("coherence", 0.0) - coherence_step
        return state

    return _apply(molecule, _mut, cause="decay", observer=observer)


def amplify_contradiction(
    molecule: SubstrateMoleculeV1,
    *,
    contradiction_step: float = 0.15,
    salience_step: float = 0.1,
    observer: GradientObserver | None = None,
) -> GradientDelta:
    """A contradiction signal lights up both fields."""

    def _mut(state: dict[str, float]) -> dict[str, float]:
        state["contradiction"] = state.get("contradiction", 0.0) + contradiction_step
        state["salience"] = state.get("salience", 0.0) + salience_step
        return state

    return _apply(molecule, _mut, cause="contradiction", observer=observer)


def stabilize_coherence(
    molecule: SubstrateMoleculeV1,
    *,
    coherence_step: float = 0.1,
    contradiction_decay: float = 0.05,
    observer: GradientObserver | None = None,
) -> GradientDelta:
    """Reinforcement after a reconciliation: coherence up, contradiction down."""

    def _mut(state: dict[str, float]) -> dict[str, float]:
        state["coherence"] = state.get("coherence", 0.0) + coherence_step
        state["contradiction"] = state.get("contradiction", 0.0) - contradiction_decay
        return state

    return _apply(molecule, _mut, cause="stabilize", observer=observer)


# -- traversal -----------------------------------------------------------------


def find_resonant_molecules(
    molecules: Iterable[SubstrateMoleculeV1],
    *,
    gradients: list[str],
    threshold: float,
) -> list[SubstrateMoleculeV1]:
    """Return molecules whose *summed* pressure across the requested gradient
    keys clears the threshold. Order is descending by total pressure.

    The MVP keeps this O(n). A future indexer can swap in.
    """

    scored: list[tuple[float, SubstrateMoleculeV1]] = []
    for molecule in molecules:
        total = sum(molecule.gradient(key) for key in gradients)
        if total >= threshold:
            scored.append((total, molecule))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [molecule for _, molecule in scored]
