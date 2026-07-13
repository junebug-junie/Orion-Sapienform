from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import check_daily_schedule_collisions as collisions  # noqa: E402


def _write_env_example(tmp_path: Path, *, pulse: tuple[int, int], world: tuple[int, int], metacog: tuple[int, int]) -> Path:
    path = tmp_path / ".env_example"
    path.write_text(
        "\n".join([
            "# comment line, should be ignored",
            "",
            f"ACTIONS_DAILY_PULSE_HOUR_LOCAL={pulse[0]}",
            f"ACTIONS_DAILY_PULSE_MINUTE_LOCAL={pulse[1]}",
            f"ACTIONS_WORLD_PULSE_HOUR_LOCAL={world[0]}",
            f"ACTIONS_WORLD_PULSE_MINUTE_LOCAL={world[1]}",
            f"ACTIONS_DAILY_METACOG_HOUR_LOCAL={metacog[0]}",
            f"ACTIONS_DAILY_METACOG_MINUTE_LOCAL={metacog[1]}",
            "",
        ]),
        encoding="utf-8",
    )
    return path


def test_time_of_day_distance_basic():
    assert collisions._time_of_day_distance_minutes(0, 0) == 0
    assert collisions._time_of_day_distance_minutes(60, 90) == 30
    assert collisions._time_of_day_distance_minutes(90, 60) == 30


def test_time_of_day_distance_wraps_around_midnight():
    # 23:50 vs 00:05 is 15 minutes apart, not ~23h35m.
    late = 23 * 60 + 50
    early = 5
    assert collisions._time_of_day_distance_minutes(late, early) == 15


def test_load_cadences_includes_synthetic_daily_journal_entry(tmp_path):
    env_path = _write_env_example(tmp_path, pulse=(8, 30), world=(6, 0), metacog=(20, 15))
    cadence_minutes = collisions._load_cadences(env_path)
    assert set(cadence_minutes) == {"Daily Pulse", "World Pulse", "Daily Metacog", "Daily Journal"}
    # Daily Journal must mirror Daily Pulse exactly (it reuses Daily Pulse's env keys).
    assert cadence_minutes["Daily Journal"] == cadence_minutes["Daily Pulse"]


def test_load_cadences_raises_on_missing_key(tmp_path):
    env_path = tmp_path / ".env_example"
    env_path.write_text("ACTIONS_DAILY_PULSE_HOUR_LOCAL=8\n", encoding="utf-8")
    try:
        collisions._load_cadences(env_path)
        assert False, "expected KeyError for missing minute/other cadence keys"
    except KeyError:
        pass


def test_find_collisions_detects_pair_within_threshold():
    cadence_minutes = {
        "Daily Pulse": 8 * 60 + 30,
        "Daily Journal": 8 * 60 + 30,
        "World Pulse": 6 * 60,
        "Daily Metacog": 20 * 60 + 15,
    }
    found = collisions._find_collisions(cadence_minutes, threshold_minutes=30)
    pairs = {frozenset((c["a"], c["b"])) for c in found}
    assert frozenset(("Daily Pulse", "Daily Journal")) in pairs


def test_find_collisions_empty_when_well_separated():
    cadence_minutes = {
        "Daily Pulse": 8 * 60,
        "Daily Journal": 8 * 60,  # still reuses Daily Pulse -- collides with it by construction
        "World Pulse": 6 * 60,
        "Daily Metacog": 20 * 60,
    }
    # Widen the gaps except Daily Journal/Daily Pulse (which always collide, that's the
    # known reuse) and use a tiny threshold to prove non-adjacent pairs aren't flagged.
    found = collisions._find_collisions(cadence_minutes, threshold_minutes=5)
    pairs = {frozenset((c["a"], c["b"])) for c in found}
    assert pairs == {frozenset(("Daily Pulse", "Daily Journal"))}


def test_main_report_only_exits_zero_even_with_collision(tmp_path):
    # Same hour/minute for pulse and metacog -- guaranteed collision.
    env_path = _write_env_example(tmp_path, pulse=(8, 30), world=(6, 0), metacog=(8, 30))
    exit_code = collisions.main(["--env-example", str(env_path)])
    assert exit_code == 0


def test_main_fail_on_collision_exits_one(tmp_path):
    env_path = _write_env_example(tmp_path, pulse=(8, 30), world=(6, 0), metacog=(8, 30))
    exit_code = collisions.main(["--env-example", str(env_path), "--fail-on-collision"])
    assert exit_code == 1


def test_main_fail_on_collision_exits_zero_when_below_threshold(tmp_path):
    # Daily Journal always mirrors Daily Pulse exactly (0m apart, see module docstring),
    # so a literal zero-collision config isn't realizable -- instead prove that a
    # tighter-than-zero threshold (impossible to trip) still exits 0.
    env_path = _write_env_example(tmp_path, pulse=(8, 30), world=(6, 0), metacog=(20, 15))
    exit_code = collisions.main(["--env-example", str(env_path), "--fail-on-collision", "--threshold-minutes", "-1"])
    assert exit_code == 0


def test_main_exits_two_on_missing_env_example(tmp_path):
    missing_path = tmp_path / "does_not_exist.env_example"
    exit_code = collisions.main(["--env-example", str(missing_path)])
    assert exit_code == 2


def test_main_json_output_contains_collisions(tmp_path, capsys):
    env_path = _write_env_example(tmp_path, pulse=(8, 30), world=(6, 0), metacog=(20, 15))
    exit_code = collisions.main(["--env-example", str(env_path), "--json"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert '"a": "Daily Journal"' in captured.out or '"a": "Daily Pulse"' in captured.out


def test_real_env_example_has_known_daily_pulse_journal_collision():
    """True positive against the actual repo config: Daily Journal reuses Daily
    Pulse's hour/minute (services/orion-actions/app/main.py's journal_should_run call
    passes settings.actions_daily_pulse_hour_local/minute_local), so this must always
    be flagged as a collision until someone deliberately reschedules one of them. If
    this test starts failing because the real .env_example changed, that means the
    known reuse was fixed (or the values drifted) -- update the "known" documentation
    (this test, and the module docstring) rather than just silencing the assertion.
    """
    env_path = _REPO_ROOT / "services" / "orion-actions" / ".env_example"
    cadence_minutes = collisions._load_cadences(env_path)
    found = collisions._find_collisions(cadence_minutes, threshold_minutes=30)
    pairs = {frozenset((c["a"], c["b"])) for c in found}
    assert frozenset(("Daily Pulse", "Daily Journal")) in pairs
