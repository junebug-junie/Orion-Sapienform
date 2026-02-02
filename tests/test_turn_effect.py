import os
from pathlib import Path

import pytest

from orion.schemas.telemetry.turn_effect import (
    compute_deltas_from_turn_effect,
    evaluate_turn_effect_alert,
    should_emit_turn_effect_alert,
    summarize_turn_effect,
    turn_effect_from_spark_meta,
)


def test_turn_effect_extraction_and_summary():
    spark_meta = {
        "phi_before": {"valence": 0.2, "energy": 0.4, "coherence": 0.6, "novelty": 0.1, "junk": 0.9},
        "phi_after": {"valence": 0.3, "energy": 0.5, "coherence": 0.4, "novelty": 0.2},
        "phi_post_before": {"valence": 0.3, "energy": 0.5, "coherence": 0.4, "novelty": 0.2},
        "phi_post_after": {"valence": 0.1, "energy": 0.7, "coherence": 0.5, "novelty": 0.4},
    }
    effect = turn_effect_from_spark_meta(spark_meta)
    assert effect is not None
    assert effect["user"]["valence"] == pytest.approx(0.1)
    assert effect["assistant"]["energy"] == pytest.approx(0.2)
    assert effect["turn"]["coherence"] == pytest.approx(-0.1)
    assert "evidence" in effect
    evidence = effect["evidence"]
    assert evidence["phi_before"]["valence"] == pytest.approx(0.2)
    assert set(evidence["phi_before"].keys()) == {"valence", "energy", "coherence", "novelty"}

    summary = summarize_turn_effect(effect)
    assert "user:" in summary
    assert "assistant:" in summary
    assert "turn:" in summary


def test_compute_deltas_from_turn_effect():
    effect = {
        "user": {"valence": "0.1", "energy": 0.2},
        "assistant": {"coherence": -0.3},
        "turn": {"novelty": 0.4},
    }
    deltas = compute_deltas_from_turn_effect(effect)
    assert deltas["user"]["valence"] == 0.1
    assert deltas["user"]["energy"] == 0.2
    assert deltas["assistant"]["coherence"] == -0.3
    assert deltas["turn"]["novelty"] == 0.4


def test_turn_effect_cli_resolution():
    from scripts.print_recent_turn_effects import _format_dry_run, _resolve_db_url

    env = {
        "ORION_SQL_URL": "postgresql://env_user:env_pass@env-host:5432/env_db",
        "POSTGRES_URI": "postgresql://pg_user:pg_pass@pg-host:5432/pg_db",
        "DATABASE_URL": "postgresql://db_user:db_pass@db-host:5432/db_db",
    }
    url = _resolve_db_url(cli_db_url="postgresql://cli:pass@cli-host:5432/cli_db", sqlite_path=None)
    assert url.startswith("postgresql://cli:pass@cli-host:5432/cli_db")

    original_env = dict(os.environ)
    os.environ.update(env)
    try:
        url = _resolve_db_url(cli_db_url=None, sqlite_path=None)
        assert url == env["ORION_SQL_URL"]
        url = _resolve_db_url(cli_db_url=None, sqlite_path=Path("local.db"))
        assert url == "sqlite:///local.db"
        output = _format_dry_run(url, "SELECT 1")
        assert "driver=sqlite" in output
        assert "query=SELECT 1" in output
    finally:
        os.environ.clear()
        os.environ.update(original_env)


def test_turn_effect_alert_decision_and_cooldown():
    effect = {"turn": {"coherence": -0.4, "valence": -0.1, "novelty": 0.2}}
    alert = evaluate_turn_effect_alert(
        effect,
        coherence_drop=0.25,
        valence_drop=0.25,
        novelty_spike=0.35,
    )
    assert alert is not None
    assert alert["metric"] == "coherence_drop"

    now = 100.0
    assert should_emit_turn_effect_alert(None, now, 120) is True
    assert should_emit_turn_effect_alert(now, now + 1, 120) is False
    assert should_emit_turn_effect_alert(now, now + 120, 120) is True
