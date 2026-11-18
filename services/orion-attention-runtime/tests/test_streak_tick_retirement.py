"""Retirement means no runtime producer, subscriber, model, boot DDL or config."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_retired_telemetry_has_no_live_runtime_or_config_references():
    paths = [REPO / "orion/schemas/registry.py", REPO / "orion/schemas/field_goal.py",
             REPO / "orion/bus/channels.yaml"]
    for service in ("orion-attention-runtime", "orion-sql-writer"):
        root = REPO / "services" / service
        paths.extend((root / "app").rglob("*.py"))
        paths.extend(root / name for name in (".env_example", "docker-compose.yml"))
    tokens = ("DominanceStreakTick", "debug.attention.streak_tick", "debug:attention:streak_tick",
              "goal_provenance_streak_ticks", "GOAL_PROVENANCE_STREAK_TICK")
    for path in paths:
        source = path.read_text()
        for token in tokens:
            assert token not in source, f"retired runtime path {token}: {path.relative_to(REPO)}"
    assert not (REPO / "services/orion-sql-writer/app/models/dominance_streak_tick.py").exists()
    assert not (REPO / "scripts/analysis/measure_goal_provenance_streak_distribution.py").exists()
