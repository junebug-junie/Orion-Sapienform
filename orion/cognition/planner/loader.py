# orion-cognition/planner/loader.py

import yaml
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .models import VerbConfig


class VerbRegistry:
    """
    Loads and caches VerbConfig objects from verbs/*.yaml
    """

    def __init__(self, verbs_dir: Path):
        self.verbs_dir = verbs_dir
        self._verbs: Dict[str, VerbConfig] = {}

    def load(self, reload: bool = False) -> None:
        if self._verbs and not reload:
            return

        self._verbs.clear()

        if not self.verbs_dir.exists():
            raise FileNotFoundError(f"Verbs directory not found: {self.verbs_dir}")

        for path in self.verbs_dir.glob("*.yaml"):
            with path.open("r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
            verb = VerbConfig(**raw)
            self._verbs[verb.name] = verb

    def get(self, verb_name: str) -> VerbConfig:
        if not self._verbs:
            self.load()

        try:
            return self._verbs[verb_name]
        except KeyError as e:
            raise KeyError(
                f"Verb '{verb_name}' not found. Available: {list(self._verbs.keys())}"
            ) from e

    def list(self, reload: bool = False) -> List[VerbConfig]:
        """
        Return all registered verbs. Use reload=True to refresh the cache.
        """
        if reload or not self._verbs:
            self.load(reload=reload)
        return list(self._verbs.values())

    def filter(
        self,
        *,
        tags: Optional[Iterable[str]] = None,
        category: Optional[str] = None,
        names: Optional[Iterable[str]] = None,
    ) -> List[VerbConfig]:
        """
        Lightweight filtering helper used by supervisors/planners.
        """
        if not self._verbs:
            self.load()

        name_set = set(names or [])
        tag_set = {t.lower() for t in (tags or [])}
        filtered: List[VerbConfig] = []

        for verb in self._verbs.values():
            if name_set and verb.name not in name_set:
                continue
            if category and verb.category and verb.category != category:
                continue
            if tag_set:
                vtags = {t.lower() for t in (verb.tags or [])}
                if not (vtags & tag_set):
                    continue
            filtered.append(verb)

        return filtered
