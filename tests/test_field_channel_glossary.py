from __future__ import annotations

from pathlib import Path

from orion.field.channel_glossary import (
    CLEAN_VERDICTS,
    LIVE_VARIANCE_THRESHOLD,
    SUBNORMAL_CUTOFF,
    _glossary_path_candidates,
    classify_channel_series,
    load_glossary,
    resolve_channel_entry,
)


def test_load_glossary_has_48_channels_matching_field_digester_channels_py():
    """23 + the 5 FCC-motor channels added 2026-07-23 (harness_step_load,
    tool_failure_streak_pressure, avg_step_chars_pressure, compliance_deficit,
    turn_incompletion -- see docs/superpowers/specs/2026-07-23-fcc-motor-field-digester-signals-design.md)
    + context_gathering_ratio added 2026-07-24
    + power_pressure/disk_capacity_pressure/fan_pressure added 2026-07-25 (real
    iLO/BMC hardware telemetry piggybacked onto biometrics' heartbeat)
    + tension_deviation_pressure added 2026-08-16 -- NOT a raw channel (the
    other entries before 2026-08-23 are), a derived scalar included in the
    glossary anyway for liveness observability. See that entry's own comment for why.
    + 7 cabinet sensor channels added 2026-08-23 (6 activity + staleness)
    + 2 cabinet ambient audio channels added 2026-08-24 (activity + staleness)
    + 1 node-qualified prediction_error entry added 2026-09-03 (version 2,
    node:substrate.bus_synaptic only -- see that entry's own comment for why
    node:substrate.vision's qualified entry is deliberately NOT added yet)
    -- this reuses channel=prediction_error, so it adds a row but not a new
    distinct channel name; the digester's 48 raw/derived channels are
    unchanged, only prediction_error now has a qualified variant alongside
    its pre-existing bare entry."""
    glossary = load_glossary()
    entries = glossary["entries"]
    assert len(entries) == 49
    names = {e.channel for e in entries}
    assert len(names) == 48, "a node-qualified entry must not introduce a new distinct channel name"
    assert "cpu_pressure" in names
    assert "reliability_pressure" in names
    assert "tension_deviation_pressure" in names
    assert "cabinet_climate_activity" in names
    assert "cabinet_sensor_staleness" in names
    assert "cabinet_ambient_audio_activity" in names
    assert "cabinet_ambient_audio_staleness" in names
    # stream_backlog_pressure/contract_pressure are the two node+capability overlaps.
    overlap = [e for e in entries if set(e.level) == {"node", "capability"}]
    assert {e.channel for e in overlap} == {"stream_backlog_pressure", "contract_pressure"}


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


def test_resolve_channel_entry_prefers_node_qualified_over_bare():
    """The whole point of version 2: bus_synaptic's qualified entry must
    describe the fraction, not the generic bare "recent prediction missed
    reality" meaning."""
    bus = resolve_channel_entry("prediction_error", "node:substrate.bus_synaptic")
    bare = resolve_channel_entry("prediction_error")
    assert bus is not None and bare is not None
    assert bus.node == "node:substrate.bus_synaptic"
    assert bare.node is None
    assert bus.meaning != bare.meaning
    assert bus.trend_source


def test_resolve_channel_entry_falls_back_to_bare_entry():
    # No node given at all -- must match every pre-version-2 caller's
    # behavior byte-identically.
    entry = resolve_channel_entry("prediction_error")
    assert entry is not None
    assert entry.node is None

    # A node given, but no qualified entry exists for it -- falls back to
    # bare, does not raise or return None. node:substrate.vision is
    # deliberately still unqualified as of this patch (see that entry's own
    # comment in the glossary YAML) -- this is the real, current case for
    # it, not a placeholder.
    for node in ("node:substrate.chat", "node:substrate.vision"):
        entry = resolve_channel_entry("prediction_error", node)
        assert entry is not None
        assert entry.node is None


def test_resolve_channel_entry_unknown_channel_returns_none():
    assert resolve_channel_entry("not_a_real_channel") is None


def test_resolve_channel_entry_ordinary_channels_are_unaffected():
    """A channel with no node-qualified variant at all resolves exactly as
    it always did, whether or not a node is passed."""
    entry = resolve_channel_entry("cpu_pressure", "node:substrate.execution")
    assert entry is not None
    assert entry.node is None
    assert entry.channel == "cpu_pressure"


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


def test_self_state_dimension_matches_channel_dimension_map_exactly():
    """Fail-closed parity gate between the glossary's `self_state_dimension` and
    `orion/field/pressure.py::CHANNEL_DIMENSION_MAP`, added 2026-08-11.

    The glossary is not a comment. `orion/field/channel_glossary.py:141` loads
    this field and `services/orion-hub/scripts/field_channel_glossary_routes.py:128`
    renders it verbatim in Hub's Field Channel Glossary panel, so a stale entry
    tells an operator that a channel feeds a dimension it does not feed, and
    invites attributing a real movement to a channel contributing nothing.

    Nothing asserted this before, and it had already drifted twice over:

    - Six entries (`availability`, `available_capacity`, `confidence`,
      `expected_offline_suppression`, `field_coherence_warning` -> `coherence`;
      `prediction_error` -> `uncertainty`) documented routes deliberately NOT
      reproduced in the 2026-07-22 SelfStateV1 burn. `orion/field/pressure.py`'s
      own module docstring says so explicitly: those old policy routes "produced
      values nothing ever read". The glossary kept advertising them for three
      weeks.
    - `thermal_pressure` -> `resource_pressure` was made false by Patch A on
      2026-08-11 (see the tombstone at CHANNEL_DIMENSION_MAP).

    All seven were removed in that patch. This test is why they cannot come
    back silently: per CLAUDE.md 0A, the fix for config drift is a failing gate,
    not a louder comment. Deliberately exact equality, not a subset check --
    a channel that routes to a dimension but is undocumented here is the same
    defect in the other direction.
    """
    from orion.field.pressure import CHANNEL_DIMENSION_MAP

    glossary = load_glossary()
    documented = {
        e.channel: e.self_state_dimension
        for e in glossary["entries"]
        if e.self_state_dimension
    }
    assert documented == dict(CHANNEL_DIMENSION_MAP)
