from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

try:
    from app.settings import settings
except ImportError:  # pragma: no cover - fallback for test harness
    from settings import settings  # type: ignore


def _read_profile(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        if path.suffix in {".yaml", ".yml"}:
            return yaml.safe_load(f) or {}
        return json.load(f)


def _find_profiles_dir() -> Optional[Path]:
    """Find orion/recall/profiles in both repo and container layouts.

    Container layout:
      /app/app/profiles.py   (this file)
      /app/orion/recall/profiles/*.yaml  (copied by Dockerfile)

    Repo layout:
      <repo>/services/orion-recall/app/profiles.py
      <repo>/orion/recall/profiles/*.yaml
    """
    env = os.getenv("RECALL_PROFILES_DIR", "").strip()
    if env:
        p = Path(env)
        if p.exists():
            return p

    here = Path(__file__).resolve()

    # Walk upwards a few levels and look for "<parent>/orion/recall/profiles"
    for parent in here.parents:
        cand = parent / "orion" / "recall" / "profiles"
        if cand.exists():
            return cand

    # last-ditch common container guesses
    for cand in (Path("/app/orion/recall/profiles"), Path("/orion/recall/profiles")):
        if cand.exists():
            return cand

    return None


@lru_cache(maxsize=16)
def load_profiles() -> Dict[str, Dict[str, Any]]:
    profiles: Dict[str, Dict[str, Any]] = {}

    base_dir = _find_profiles_dir()
    if base_dir and base_dir.exists():
        for p in sorted(base_dir.glob("*.y*ml")):
            data = _read_profile(p)
            name = data.get("profile") or p.stem
            profiles[str(name)] = data

    # If default profile wasn't found on disk, synthesize a sane fallback
    if settings.RECALL_DEFAULT_PROFILE and settings.RECALL_DEFAULT_PROFILE not in profiles:
        profiles[settings.RECALL_DEFAULT_PROFILE] = {
            "profile": settings.RECALL_DEFAULT_PROFILE,
            "vector_top_k": settings.RECALL_DEFAULT_MAX_ITEMS,
            "rdf_top_k": 0,  # stays 0 if profiles aren't mounted; avoids surprises
            "max_per_source": 4,
            "max_total_items": settings.RECALL_DEFAULT_MAX_ITEMS,
            "render_budget_tokens": 256,
            "enable_query_expansion": True,
            "enable_sql_timeline": True,
            "relevance": {
                "backend_weights": {
                    "vector": 1.0,
                    "sql_timeline": 0.9,
                    "sql_chat": 0.6,
                    "rdf_chat": 0.5,
                    "rdf": 0.4,
                },
                "score_weight": 0.7,
                "text_similarity_weight": 0.15,
                "recency_weight": 0.1,
                "enable_recency": False,
                "recency_half_life_hours": 72,
                "session_boost": 0.1,
            },
        }

    return profiles


def get_profile(name: str | None) -> Dict[str, Any]:
    profiles = load_profiles()
    if name and name in profiles:
        return profiles[name]
    return profiles.get(settings.RECALL_DEFAULT_PROFILE, {})
