"""In-memory registry for the kernel vocabulary.

The registry is intentionally lightweight: it tracks what atoms, predicates,
and molecule kinds are legal. It does not persist anything itself.
"""

from __future__ import annotations

from threading import RLock
from typing import Iterable

from .atom import DEFAULT_ATOMS, ConceptAtomV1
from .relation import DEFAULT_PREDICATES


class SchemaKernelRegistry:
    """Holds the active atom vocabulary, predicate vocabulary, and known
    molecule kinds. Organs read from this registry to discover what they can
    legally emit.
    """

    def __init__(
        self,
        *,
        atoms: Iterable[ConceptAtomV1] | None = None,
        predicates: Iterable[str] | None = None,
        molecule_kinds: Iterable[str] | None = None,
    ) -> None:
        self._lock = RLock()
        self._atoms: dict[str, ConceptAtomV1] = {}
        self._predicates: set[str] = set()
        self._molecule_kinds: set[str] = set()

        for atom in atoms or ():
            self.register_atom(atom)
        for predicate in predicates or ():
            self.register_predicate(predicate)
        for kind in molecule_kinds or ():
            self.register_molecule_kind(kind)

    # -- atoms -----------------------------------------------------------------

    def register_atom(self, atom: ConceptAtomV1) -> None:
        with self._lock:
            self._atoms[atom.key] = atom

    def atom(self, key: str) -> ConceptAtomV1 | None:
        with self._lock:
            return self._atoms.get(key)

    def has_atom(self, key: str) -> bool:
        with self._lock:
            return key in self._atoms

    def atoms(self) -> tuple[ConceptAtomV1, ...]:
        with self._lock:
            return tuple(self._atoms.values())

    # -- predicates ------------------------------------------------------------

    def register_predicate(self, predicate: str) -> None:
        with self._lock:
            self._predicates.add(predicate)

    def has_predicate(self, predicate: str) -> bool:
        with self._lock:
            return predicate in self._predicates

    def predicates(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._predicates))

    # -- molecule kinds --------------------------------------------------------

    def register_molecule_kind(self, kind: str) -> None:
        with self._lock:
            self._molecule_kinds.add(kind)

    def has_molecule_kind(self, kind: str) -> bool:
        with self._lock:
            return kind in self._molecule_kinds

    def molecule_kinds(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._molecule_kinds))


def default_registry() -> SchemaKernelRegistry:
    """Build a registry seeded with the canonical defaults.

    Molecule kinds expected by the MVP organs are pre-registered so emit calls
    do not blow up before the first run.
    """

    return SchemaKernelRegistry(
        atoms=DEFAULT_ATOMS,
        predicates=DEFAULT_PREDICATES,
        molecule_kinds=(
            "observation",
            "claim",
            "pressure",
            "contradiction",
        ),
    )
