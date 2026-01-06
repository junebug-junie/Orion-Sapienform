from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

try:
    from app.settings import settings
except ImportError:  # pragma: no cover - fallback for test harness
    from settings import settings  # type: ignore


DEFAULT_PROFILES_DIR = Path(__file__).resolve().parents[2] / "orion" / "recall" / "profiles"


def _read_profile(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        if path.suffix in {".yaml", ".yml"}:
            return yaml.safe_load(f) or {}
        return json.load(f)


@lru_cache(maxsize=16)
def load_profiles() -> Dict[str, Dict[str, Any]]:
    profiles: Dict[str, Dict[str, Any]] = {}
    base_dir = DEFAULT_PROFILES_DIR
    if base_dir.exists():
        for p in base_dir.glob("*.y*ml"):
            data = _read_profile(p)
            name = data.get("profile") or p.stem
            profiles[name] = data
    if settings.RECALL_DEFAULT_PROFILE and settings.RECALL_DEFAULT_PROFILE not in profiles:
        profiles[settings.RECALL_DEFAULT_PROFILE] = {
            "profile": settings.RECALL_DEFAULT_PROFILE,
            "vector_top_k": settings.RECALL_DEFAULT_MAX_ITEMS,
            "rdf_top_k": 0,
            "max_per_source": 4,
            "max_total_items": settings.RECALL_DEFAULT_MAX_ITEMS,
            "render_budget_tokens": 256,
            "enable_query_expansion": True,
        }
    return profiles


def get_profile(name: str | None) -> Dict[str, Any]:
    profiles = load_profiles()
    if name and name in profiles:
        return profiles[name]
    return profiles.get(settings.RECALL_DEFAULT_PROFILE, {})
