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


def test_subject_stays_unknown_with_no_identity_hint() -> None:
    """No identity_hint given at all -- the honest default, same as before
    identity_face was wired in. 'none' only when nothing has ever been
    seen at all."""
    t = _tracker()
    assert t.observe(PERSON, now=0.0)["subject"] == "unknown"
    assert t.observe(EMPTY, now=1.0)["subject"] == "unknown"   # recent, still someone


def test_ignores_labels_other_than_the_subject() -> None:
    t = _tracker()
    snap = t.observe(frozenset({"chair", "desk"}), now=0.0)
    assert snap["state"] == "absent"


# -- identity_hint narrowing --------------------------------------------------


def test_probable_identity_hint_narrows_subject() -> None:
    t = _tracker()
    snap = t.observe(PERSON, now=0.0, identity_hint={"subject": "juniper", "state": "probable"})
    assert snap["subject"] == "juniper"


def test_possible_identity_hint_narrows_subject() -> None:
    t = _tracker()
    snap = t.observe(PERSON, now=0.0, identity_hint={"subject": "juniper", "state": "possible"})
    assert snap["subject"] == "juniper"


def test_unsure_identity_hint_does_not_narrow_subject() -> None:
    """An 'unsure' hint is the same honesty-preserving no-op as no hint at
    all -- this is presence.py's own contract, not a re-test of
    identity_hint_from_artifact (which already filters unsure out before a
    hint ever reaches here; this asserts the tracker's own defense in
    depth, in case a future caller passes one through anyway)."""
    t = _tracker()
    snap = t.observe(PERSON, now=0.0, identity_hint={"subject": "juniper", "state": "unsure"})
    assert snap["subject"] == "unknown"


def test_identity_hint_does_not_narrow_subject_none() -> None:
    """Nobody believed present -- an identity hint (stale, arrived late for
    someone who already left) must not manufacture a sighting."""
    t = _tracker()
    snap = t.observe(EMPTY, now=0.0, identity_hint={"subject": "juniper", "state": "probable"})
    assert snap["subject"] == "none"


def test_identity_hint_with_subject_unknown_is_a_no_op() -> None:
    t = _tracker()
    snap = t.observe(PERSON, now=0.0, identity_hint={"subject": "unknown", "state": "probable"})
    assert snap["subject"] == "unknown"


# -- identity_uncertain -------------------------------------------------------


def test_uncertain_confidence_sets_identity_uncertain_when_present() -> None:
    t = _tracker()
    snap = t.observe(PERSON, now=0.0, identity_confidence="uncertain")
    assert snap["identity_uncertain"] is True


def test_confirmed_confidence_does_not_set_identity_uncertain() -> None:
    t = _tracker()
    snap = t.observe(PERSON, now=0.0, identity_confidence="confirmed")
    assert snap["identity_uncertain"] is False


def test_no_confidence_at_all_is_not_uncertain() -> None:
    """The subsystem simply not running/no fresh read must render as silence,
    not as a false 'I don't recognize you' -- this is the exact distinction
    Juniper asked for: broken/not-running must never speak up."""
    t = _tracker()
    snap = t.observe(PERSON, now=0.0, identity_confidence=None)
    assert snap["identity_uncertain"] is False


def test_uncertain_confidence_while_only_recent_is_not_uncertain() -> None:
    """Asking about someone who already stepped out of frame is exactly the
    awkwardness identity_uncertain exists to avoid -- only present_now
    qualifies, never 'recent'."""
    t = _tracker(grace_sec=120.0)
    t.observe(PERSON, now=0.0, identity_confidence="uncertain")
    snap = t.observe(EMPTY, now=60.0, identity_confidence="uncertain")  # recent, not present
    assert snap["state"] == "recent"
    assert snap["identity_uncertain"] is False


def test_uncertain_confidence_while_absent_is_not_uncertain() -> None:
    t = _tracker()
    snap = t.observe(EMPTY, now=0.0, identity_confidence="uncertain")
    assert snap["state"] == "absent"
    assert snap["identity_uncertain"] is False


def test_registry_current_snapshot_reflects_last_observe_regardless_of_write_gate() -> None:
    """current_snapshot() must return the fresh subject on every call, not
    just on the Postgres-write-rate-limited cadence record() itself gates."""
    reg = PresenceRegistry(grace_sec=120.0, write_min_interval_sec=999.0)  # write never due again after first
    # First call's own due-check is (now - 0.0) >= 999.0 -- last_write_at
    # defaults to 0.0, so now=0.0 would spuriously read as NOT due (same
    # gotcha test_write_is_rate_limited_independently_per_stream's own
    # comment names). now=1000.0 clears it unambiguously.
    first = reg.record("cam0", PERSON, now=1000.0, identity_hint={"subject": "juniper", "state": "probable"})
    assert first is not None, "first call for a stream is always due"
    second = reg.record("cam0", PERSON, now=1001.0)  # no hint this call, write suppressed by the gate
    assert second is None
    # Despite record() returning None (write not due) and this call passing
    # no hint, current_snapshot() must reflect this LATEST observe() call --
    # which correctly dropped the subject back to "unknown" since no hint
    # was given this time. Proves it reads the tracker's live state, not a
    # cached copy of the last WRITE.
    assert reg.current_snapshot("cam0")["subject"] == "unknown"


def test_registry_current_snapshot_none_before_any_record() -> None:
    reg = PresenceRegistry(grace_sec=120.0)
    assert reg.current_snapshot("never-seen") is None


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


def test_concurrent_streams_do_not_bleed_into_each_other() -> None:
    """Interleaved calls, not just sequential ones -- proves isolation under
    the real access pattern: the RPC path (_handle_rpc) dispatches concurrent
    asyncio tasks per stream, unlike the periodic drain loop, which is
    sequential. Simulates that interleaving directly.
    """
    reg = PresenceRegistry(grace_sec=120.0, write_min_interval_sec=0.0)

    # Interleave observations for two streams as if two concurrent tasks were
    # each mid-flight: cam0 present, carbon absent, cam0 absent, carbon
    # present -- crossing back and forth rather than completing one stream
    # before starting the other.
    reg.record("cam0", PERSON, now=0.0)
    reg.record("carbon", EMPTY, now=0.0)
    reg.record("cam0", EMPTY, now=1.0)
    reg.record("carbon", PERSON, now=1.0)
    reg.record("cam0", EMPTY, now=125.0)     # past grace -> absent
    snap_carbon = reg.record("carbon", PERSON, now=125.0)
    snap_cam0 = reg.record("cam0", EMPTY, now=126.0)

    assert snap_carbon["state"] == "present"
    assert snap_carbon["since_sec"] == 124.0    # carbon has been present since t=1
    assert snap_cam0["state"] == "absent"
    assert snap_cam0["last_seen_sec"] == 126.0  # cam0 last seen at t=0
