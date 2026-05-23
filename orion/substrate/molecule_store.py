"""Append-only molecule persistence.

The MVP persists to JSONL. A future iteration may swap in Postgres or the
existing graph store — the interface is intentionally narrow.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import RLock
from typing import Iterable, Iterator

from .molecules import SubstrateMoleculeV1


class MoleculeJsonlStore:
    """An in-memory + append-only-file molecule store.

    Designed for inspectability. Every write is a single JSON line; the in-memory
    map mirrors disk state so the traversal/operators stay fast.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._molecules: dict[str, SubstrateMoleculeV1] = {}
        if self._path.exists():
            self._load()

    # -- internals -------------------------------------------------------------

    def _load(self) -> None:
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                molecule = SubstrateMoleculeV1.model_validate(data)
                self._molecules[molecule.molecule_id] = molecule

    def _append(self, molecule: SubstrateMoleculeV1) -> None:
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(molecule.to_jsonable(), default=str))
            handle.write("\n")

    # -- public --------------------------------------------------------------

    @property
    def path(self) -> Path:
        return self._path

    def add(self, molecule: SubstrateMoleculeV1) -> SubstrateMoleculeV1:
        with self._lock:
            self._molecules[molecule.molecule_id] = molecule
            self._append(molecule)
            return molecule

    def get(self, molecule_id: str) -> SubstrateMoleculeV1 | None:
        with self._lock:
            return self._molecules.get(molecule_id)

    def all(self) -> list[SubstrateMoleculeV1]:
        with self._lock:
            return list(self._molecules.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._molecules)

    def __iter__(self) -> Iterator[SubstrateMoleculeV1]:
        return iter(self.all())

    def filter(
        self,
        *,
        molecule_kind: str | None = None,
        organ: str | None = None,
    ) -> list[SubstrateMoleculeV1]:
        result: list[SubstrateMoleculeV1] = []
        for molecule in self.all():
            if molecule_kind and molecule.molecule_kind != molecule_kind:
                continue
            if organ and molecule.provenance.get("organ") != organ:
                continue
            result.append(molecule)
        return result

    def extend(self, molecules: Iterable[SubstrateMoleculeV1]) -> None:
        for molecule in molecules:
            self.add(molecule)
