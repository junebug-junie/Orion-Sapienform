"""Load ``config/integration/organs.yaml`` (the organ registry). The parse and its rules are pure, in
``orion.temporal_self.now.parse_organs``; this module is the one place that reads the file."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from orion.temporal_self.now import OrganRegistry, parse_organs

REPO_ROOT = Path(__file__).resolve().parents[2]
ORGANS_PATH = REPO_ROOT / "config" / "integration" / "organs.yaml"

_CACHE: dict[str, OrganRegistry] = {}


def load_organs(path: Optional[Path] = None) -> OrganRegistry:
    """The registry from disk, parsed and validated (cached per path)."""
    import yaml

    p = Path(path) if path is not None else ORGANS_PATH
    key = str(p)
    if key not in _CACHE:
        _CACHE[key] = parse_organs(yaml.safe_load(p.read_text(encoding="utf-8")))
    return _CACHE[key]
