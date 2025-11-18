# orion-cognition/packs_loader.py

from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import yaml


class CognitionPack:
    """Represents a pack as defined in packs/*.yaml."""

    def __init__(self, name: str, label: str, description: str, verbs: List[str]):
        self.name = name
        self.label = label
        self.description = description
        self.verbs = verbs

    def __repr__(self) -> str:
        return f"<CognitionPack {self.name} ({len(self.verbs)} verbs)>"


class PackManager:
    """
    Manages cognitive packs.

    Responsibilities:
    - Load packs from packs/*.yaml
    - List packs
    - List verbs within a pack
    - Validate that pack verbs exist in verbs/*.yaml
    - Load packs (return consolidated verb list)
    """

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.packs_dir = base_dir / "packs"
        self.verbs_dir = base_dir / "verbs"
        self._packs: Dict[str, CognitionPack] = {}

    # ------------------------------
    # PACK LOADING
    # ------------------------------

    def load_packs(self, reload: bool = False) -> None:
        if self._packs and not reload:
            return

        self._packs.clear()

        if not self.packs_dir.exists():
            raise FileNotFoundError(f"packs directory not found: {self.packs_dir}")

        for path in self.packs_dir.glob("*.yaml"):
            with path.open("r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)

            pack = CognitionPack(
                name=raw["name"],
                label=raw.get("label", raw["name"]),
                description=raw.get("description", ""),
                verbs=raw.get("verbs", []),
            )

            self._packs[pack.name] = pack

    # ------------------------------
    # INSPECTION
    # ------------------------------

    def list_packs(self) -> List[str]:
        if not self._packs:
            self.load_packs()
        return list(self._packs.keys())

    def get_pack(self, pack_name: str) -> CognitionPack:
        if not self._packs:
            self.load_packs()
        try:
            return self._packs[pack_name]
        except KeyError:
            available = ", ".join(self._packs.keys())
            raise KeyError(
                f"Pack '{pack_name}' not found. Available packs: {available}"
            )

    def get_pack_verbs(self, pack_name: str) -> List[str]:
        pack = self.get_pack(pack_name)
        return pack.verbs

    # ------------------------------
    # VALIDATION
    # ------------------------------

    def verify_pack(self, pack_name: str) -> Dict[str, List[str]]:
        """
        Validate that all verbs in the pack exist in verbs/*.yaml.
        Returns a dict with:
            {"missing": [...], "present": [...]}
        """
        pack = self.get_pack(pack_name)
        verb_files = set(path.stem for path in self.verbs_dir.glob("*.yaml"))

        missing = [v for v in pack.verbs if v not in verb_files]
        present = [v for v in pack.verbs if v in verb_files]

        return {"missing": missing, "present": present}

    # ------------------------------
    # LOADING PACKS FOR USE
    # ------------------------------

    def load_verb_set(self, pack_names: List[str]) -> List[str]:
        """
        Given one or more pack names, return consolidated list of unique verbs.
        Sorted for stability.
        """
        all_verbs: set[str] = set()

        for pack_name in pack_names:
            pack = self.get_pack(pack_name)
            all_verbs.update(pack.verbs)

        return sorted(all_verbs)
