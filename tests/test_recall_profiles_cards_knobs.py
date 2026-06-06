"""Every shipped recall profile must declare cards_top_k for the memory-cards rail."""

from __future__ import annotations

from pathlib import Path

import yaml


def _profiles_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "orion" / "recall" / "profiles"


def test_all_recall_profiles_define_cards_top_k() -> None:
    missing: list[str] = []
    for path in sorted(_profiles_dir().glob("*.y*ml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        topk = int(data.get("cards_top_k", 0) or 0)
        if topk <= 0:
            missing.append(path.name)
    assert not missing, f"profiles missing cards_top_k: {missing}"


def test_all_recall_profiles_define_cards_backend_weight() -> None:
    missing: list[str] = []
    for path in sorted(_profiles_dir().glob("*.y*ml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        top = data.get("backend_weights") if isinstance(data.get("backend_weights"), dict) else {}
        rel = data.get("relevance") if isinstance(data.get("relevance"), dict) else {}
        rel_bw = rel.get("backend_weights") if isinstance(rel.get("backend_weights"), dict) else {}
        merged = {**top, **rel_bw}
        try:
            wt = float(merged.get("cards", 0.0) or 0.0)
        except Exception:
            wt = 0.0
        if wt <= 0.0:
            missing.append(path.name)
    assert not missing, f"profiles missing backend_weights.cards: {missing}"
