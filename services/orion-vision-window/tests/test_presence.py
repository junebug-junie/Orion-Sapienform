"""Tests for embodied presence: `is someone at the camera, and for how long`.

Every threshold fixture is hand-computed with the arithmetic in a comment, per
this repo's standing rule that a green test proves the formula, not just
agreement with the code that wrote it.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.presence import PresenceRegistry, PresenceTracker  # noqa: E402


def _tracker(grace_sec: float = 120.0) -> PresenceTracker:
    return PresenceTracker(grace_sec=grace_sec)


PERSON = frozenset({"person"})
EMPTY: frozenset[str] = frozenset()


# -- the state machine -------------------------------------------------------


def test_first_sighting_is_present_with_since_zero() -> None:
    t = _tracker()
    snap = t.observe(PERSON, now=0.0)
    assert snap["state"] == "present"
    assert snap["since_sec"] == 0.0
    assert snap["last_seen_sec"] == 0.0


def test_continuous_presence_accumulates_since_sec() -> None:
    """The whole point: "you've been at that desk five hours" is this number."""
    t = _tracker()
    t.observe(PERSON, now=0.0)
    t.observe(PERSON, now=1800.0)
    snap = t.observe(PERSON, now=18000.0)   # 5 hours
    assert snap["state"] == "present"
    assert snap["since_sec"] == 18000.0
    assert snap["last_seen_sec"] == 0.0


def test_absence_before_any_sighting_has_no_last_seen() -> None:
    """Never having seen someone is a different fact from having seen them and
    lost them -- last_seen_sec must be None, not a fabricated 0 or a huge
    number that looks like real data."""
    t = _tracker()
    snap = t.observe(EMPTY, now=100.0)
    assert snap["state"] == "absent"
    assert snap["last_seen_sec"] is None
    assert snap["subject"] == "none"


def test_a_brief_gap_reads_as_recent_not_absent() -> None:
    """Hand-computed: grace_sec=120. Seen at t=0, gone at t=60 -- 60 <= 120."""
    t = _tracker(grace_sec=120.0)
    t.observe(PERSON, now=0.0)
    snap = t.observe(EMPTY, now=60.0)
    assert snap["state"] == "recent"
    assert snap["last_seen_sec"] == 60.0


def test_a_gap_past_grace_reads_as_absent() -> None:
    """Hand-computed: 121 > 120 = grace_sec."""
    t = _tracker(grace_sec=120.0)
    t.observe(PERSON, now=0.0)
    snap = t.observe(EMPTY, now=121.0)
    assert snap["state"] == "absent"


def test_exactly_at_the_grace_boundary_is_still_recent() -> None:
    """The precise boundary, not just a value safely inside it. A first draft
    of this suite tested at 60s against a 120s grace -- both <= and < agree
    there, so an off-by-one on the comparison operator passed unnoticed. This
    is last_seen_sec == grace_sec exactly: <= reads recent, < reads absent."""
    t = _tracker(grace_sec=120.0)
    t.observe(PERSON, now=0.0)
    snap = t.observe(EMPTY, now=120.0)
    assert snap["state"] == "recent", "grace_sec boundary should be inclusive"


def test_reappearing_during_the_grace_window_resets_to_present() -> None:
    t = _tracker(grace_sec=120.0)
    t.observe(PERSON, now=0.0)
    t.observe(EMPTY, now=60.0)                # recent
    snap = t.observe(PERSON, now=90.0)         # back
    assert snap["state"] == "present"
    assert snap["last_seen_sec"] == 0.0


def test_since_sec_resets_on_a_real_state_transition() -> None:
    """The clock is per-STATE, not per-sighting: present -> recent -> absent
    each start their own count."""
    t = _tracker(grace_sec=100.0)
    t.observe(PERSON, now=0.0)
    r1 = t.observe(EMPTY, now=10.0)            # -> recent, since_sec=0
    assert r1["state"] == "recent" and r1["since_sec"] == 0.0
    r2 = t.observe(EMPTY, now=50.0)             # still recent, since_sec=40
    assert r2["state"] == "recent" and r2["since_sec"] == 40.0
    r3 = t.observe(EMPTY, now=200.0)            # -> absent, since_sec=0
    assert r3["state"] == "absent" and r3["since_sec"] == 0.0


def test_subject_stays_unknown_no_identity_wired() -> None:
    """Honest placeholder: person != Juniper until identity_face exists.
    'none' only when nothing has ever been seen at all."""
    t = _tracker()
    assert t.observe(PERSON, now=0.0)["subject"] == "unknown"
    assert t.observe(EMPTY, now=1.0)["subject"] == "unknown"   # recent, still someone


def test_ignores_labels_other_than_the_subject() -> None:
    t = _tracker()
    snap = t.observe(frozenset({"chair", "desk"}), now=0.0)
    assert snap["state"] == "absent"


# -- the registry: per-stream isolation + write throttling -------------------


def test_streams_are_tracked_independently() -> None:
    reg = PresenceRegistry(grace_sec=120.0, write_min_interval_sec=0.0)
    reg.record("cam0", PERSON, now=0.0)
    reg.record("carbon", EMPTY, now=0.0)
    snap_cam0 = reg.record("cam0", PERSON, now=1.0)
    snap_carbon = reg.record("carbon", EMPTY, now=1.0)
    assert snap_cam0["state"] == "present"
    assert snap_carbon["state"] == "absent"


def test_write_is_rate_limited_independently_per_stream() -> None:
    """Hand-computed: write_min_interval=5. First call always due (last_write
    defaults to 0.0, so t=0 - 0.0 = 0 >= 0 only when interval<=0; use t=10 as
    the first call so it clears a 5s interval unambiguously)."""
    reg = PresenceRegistry(grace_sec=120.0, write_min_interval_sec=5.0)
    first = reg.record("cam0", PERSON, now=10.0)
    assert first is not None, "first write for a stream must not be suppressed"
    second = reg.record("cam0", PERSON, now=12.0)   # 2s later, < 5s interval
    assert second is None
    third = reg.record("cam0", PERSON, now=16.0)    # 6s after the first write
    assert third is not None


def test_a_second_streams_writes_are_not_blocked_by_the_first() -> None:
    reg = PresenceRegistry(grace_sec=120.0, write_min_interval_sec=5.0)
    reg.record("cam0", PERSON, now=10.0)
    snap = reg.record("carbon", PERSON, now=10.1)
    assert snap is not None, "one stream's rate limit must not gate another's"


def test_record_never_raises_on_a_bad_input() -> None:
    """Best-effort contract: a presence bug must never break a window flush."""
    reg = PresenceRegistry(grace_sec=120.0, write_min_interval_sec=0.0)
    result = reg.record("cam0", None, now=0.0)   # type: ignore[arg-type]
    assert result is None
