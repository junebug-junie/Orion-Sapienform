"""The R3 readers eval: each reader's hold rule, joined to a level timeline."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from measure_arousal_reader_replay import Timeline, export_sql, held, reader_impact  # noqa: E402

T0 = datetime(2026, 10, 11, 0, 0, tzinfo=timezone.utc)
KW = dict(curiosity_base_sec=1800.0, engaged_multiplier=2.0)


def test_hold_rules_match_the_shipped_branches():
    for reader in ("dream", "curiosity", "self_inquiry", "outreach"):
        assert held(reader, "idle", 10.0, **KW) is None
        assert held(reader, "unknown", 10.0, **KW) is None
        assert held(reader, "strained", 99999.0, **KW) == "arousal_strained"
    assert held("dream", "engaged", None, **KW) == "arousal_engaged"
    assert held("self_inquiry", "engaged", None, **KW) == "arousal_engaged"
    assert held("outreach", "engaged", 10.0, **KW) is None
    assert held("curiosity", "engaged", 1900.0, **KW) == "arousal_engaged_cooldown"
    assert held("curiosity", "engaged", 3700.0, **KW) is None
    assert held("curiosity", "engaged", None, **KW) is None
    assert held("curiosity", "engaged", 1900.0, curiosity_base_sec=1800.0, engaged_multiplier=1.0) is None


def test_readers_4_to_6_hold_only_while_strained():
    for reader in ("visual_baseline", "reverie_chain", "mutation"):
        assert held(reader, "strained", None, **KW) == "arousal_strained"
        for level in ("engaged", "idle", "unknown", None):
            assert held(reader, level, 10.0, **KW) is None


def test_impact_counts_per_day_and_uses_earlier_runs_only_for_gaps():
    tl = Timeline([(T0, "idle"), (T0 + timedelta(hours=2), "strained"), (T0 + timedelta(hours=3), "engaged"),
                   (T0 + timedelta(hours=4), "idle")])
    decisions = [
        ("curiosity", T0 - timedelta(minutes=40)),          # before the window: seeds the gap only
        ("curiosity", T0 + timedelta(minutes=10)),          # idle
        ("curiosity", T0 + timedelta(hours=2, minutes=5)),  # strained
        ("curiosity", T0 + timedelta(hours=3, minutes=5)),  # engaged, 60 min gap = 2 x 30 min -> not held
        ("curiosity", T0 + timedelta(hours=3, minutes=30)),  # engaged, 25 min gap -> held
        ("outreach", T0 + timedelta(hours=3, minutes=10)),  # engaged -> not held
        ("dream", T0 + timedelta(hours=3, minutes=20)),     # engaged -> held
    ]
    r = reader_impact(decisions, tl, T0, T0 + timedelta(days=1), **KW)
    day = r["per_day"][T0.date().isoformat()]
    assert day["curiosity"] == {"decisions": 4, "held": 2}
    assert day["outreach"] == {"decisions": 1, "held": 0}
    assert day["dream"] == {"decisions": 1, "held": 1}
    assert r["held_reasons"]["curiosity"] == {"arousal_strained": 1, "arousal_engaged_cooldown": 1}


def test_before_the_first_level_is_unknown():
    assert Timeline([(T0, "strained")]).level_at(T0 - timedelta(seconds=1)) == "unknown"


def test_export_is_one_read_only_transaction():
    sql = export_sql(T0, T0 + timedelta(days=1))
    assert sql.startswith("BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY;")
    assert sql.rstrip().endswith("ROLLBACK;") and sql.count("ROLLBACK;") == 1
    assert "arousal_transition" in sql and "durable_admission_runs" in sql
    for table in ("reverie_visual_attempt", "substrate_reverie_chain", "substrate_mutation_proposal"):
        assert table in sql


def test_dream_hold_is_bounded_by_the_level_age():
    assert held("dream", "strained", None, **KW, level_age_sec=5 * 3600.0) == "arousal_strained"
    assert held("dream", "strained", None, **KW, level_age_sec=6 * 3600.0) is None
    tl = Timeline([(T0, "strained", T0 - timedelta(hours=7))])
    r = reader_impact([("dream", T0 + timedelta(minutes=1))], tl, T0, T0 + timedelta(days=1), **KW)
    assert r["totals"]["dream"] == {"decisions": 1, "held": 0}
