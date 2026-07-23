from __future__ import annotations

from pathlib import Path

from orion.field.channel_glossary import (
    CLEAN_VERDICTS,
    LIVE_VARIANCE_THRESHOLD,
    SUBNORMAL_CUTOFF,
    _glossary_path_candidates,
    classify_channel_series,
    load_glossary,
)


def test_load_glossary_has_34_channels_matching_field_digester_channels_py():
    """29 + the 5 FCC-motor channels added 2026-07-23 (harness_step_load,
    tool_failure_streak_pressure, avg_step_chars_pressure, compliance_deficit,
    turn_incompletion -- see docs/superpowers/specs/2026-07-23-fcc-motor-field-digester-signals-design.md)."""
    glossary = load_glossary()
    entries = glossary["entries"]
    assert len(entries) == 34
    names = {e.channel for e in entries}
    assert "cpu_pressure" in names
    assert "reliability_pressure" in names
    # transport_pressure/contract_pressure are the two node+capability overlaps.
    overlap = [e for e in entries if set(e.level) == {"node", "capability"}]
    assert {e.channel for e in overlap} == {"transport_pressure", "contract_pressure"}


def test_glossary_path_candidates_prefers_orion_repo_root_env_var(monkeypatch):
    # Regression: Hub's Dockerfile only `COPY orion /app/orion` -- config/
    # (a sibling of orion/ on disk) never lands under /app, so a naive
    # Path(__file__).resolve().parents[2] resolves to /app inside the Hub
    # container and 404s on the real file. ORION_REPO_ROOT (compose default
    # /repo) must be checked, and checked first, not just the bare-checkout
    # parents[2] fallback.
    monkeypatch.setenv("ORION_REPO_ROOT", "/some/mounted/repo")
    candidates = _glossary_path_candidates()
    assert candidates[0] == Path("/some/mounted/repo/config/field/field_channel_glossary.v1.yaml")


def test_glossary_path_candidates_includes_repo_and_mnt_fallbacks(monkeypatch):
    monkeypatch.delenv("ORION_REPO_ROOT", raising=False)
    candidates = [str(c) for c in _glossary_path_candidates()]
    assert any(c.startswith("/repo/") for c in candidates)
    assert any(c.startswith("/mnt/scripts/Orion-Sapienform/") for c in candidates)


def test_load_glossary_categories_cover_all_seven_semantic_groups():
    glossary = load_glossary()
    assert len(glossary["categories"]) == 7
    used_categories = {e.category for e in glossary["entries"]}
    assert used_categories <= set(glossary["categories"].keys())


def test_classify_never_produced_on_empty_series():
    assert classify_channel_series([]) == "never_produced"


def test_classify_dead_when_all_values_subnormal_or_zero():
    assert classify_channel_series([0.0, 0.0, 0.0]) == "dead"
    assert classify_channel_series([SUBNORMAL_CUTOFF / 10, 0.0, SUBNORMAL_CUTOFF / 100]) == "dead"


def test_classify_quiet_when_present_but_low_variance():
    values = [0.5, 0.51, 0.49, 0.50, 0.52]
    assert (max(values) - sorted(values)[len(values) // 2]) <= LIVE_VARIANCE_THRESHOLD
    assert classify_channel_series(values) == "quiet"


def test_classify_live_when_genuine_variance():
    values = [0.1, 0.4, 0.15, 0.6, 0.2, 0.05, 0.55]
    assert classify_channel_series(values) == "live"


def test_classify_ratchet_suspect_when_monotonic_non_decreasing_with_real_climb():
    values = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5]
    assert classify_channel_series(values) == "ratchet_suspect"


def test_classify_not_ratchet_suspect_when_monotonic_but_flat():
    # Monotonic non-decreasing but the net climb never exceeds the variance
    # threshold -- correctly falls through to quiet, not a false ratchet flag.
    values = [0.0, 0.01, 0.01, 0.02, 0.02]
    assert classify_channel_series(values) == "quiet"


def test_classify_two_point_up_step_is_not_ratchet_suspect():
    # With only 2 samples, "non-decreasing" is true whenever the second
    # value isn't lower than the first -- a coin flip for any noisy-but-
    # healthy channel, not a real monotonicity signal. A single up-step
    # (e.g. a channel that was quiet all window and got one real reading
    # near the end) must not be flagged as a suspected one-way ratchet.
    assert classify_channel_series([0.0, 0.3]) == "live"


def test_classify_ratchet_suspect_requires_minimum_sample_count():
    from orion.field.channel_glossary import RATCHET_MIN_SAMPLES

    short_climb = [round(i * 0.5 / (RATCHET_MIN_SAMPLES - 2), 4) for i in range(RATCHET_MIN_SAMPLES - 1)]
    assert len(short_climb) < RATCHET_MIN_SAMPLES
    assert classify_channel_series(short_climb) != "ratchet_suspect"


def test_clean_verdicts_excludes_broken_categories():
    assert CLEAN_VERDICTS == {"live", "quiet"}
    assert "dead" not in CLEAN_VERDICTS
    assert "ratchet_suspect" not in CLEAN_VERDICTS
    assert "never_produced" not in CLEAN_VERDICTS
