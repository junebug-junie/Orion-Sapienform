from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_hub_crystallization_ui_renders_provenance_and_turn_scores() -> None:
    ui = (REPO_ROOT / "services" / "orion-hub" / "static" / "js" / "memory-crystallization-ui.js").read_text(
        encoding="utf-8"
    )
    assert "function renderProvenance" in ui
    assert "window_novelty_max" in ui
    assert "window_significance_max" in ui
    assert "gate_reasons" in ui
    assert "function renderEvidence" in ui
    assert "ev.note" in ui
    assert "User:" in ui
    assert "Orion:" in ui


def test_repository_round_trips_provenance_column() -> None:
    repo = (REPO_ROOT / "orion" / "memory" / "crystallization" / "repository.py").read_text(encoding="utf-8")
    sql = (REPO_ROOT / "orion" / "core" / "storage" / "sql" / "memory_crystallizations.sql").read_text(
        encoding="utf-8"
    )
    assert "provenance jsonb" in sql
    assert 'provenance=_parse_jsonb(row.get("provenance"))' in repo
    assert "_jsonb(crystallization.provenance or {})" in repo
