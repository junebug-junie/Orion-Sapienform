"""Tests for the object-permanence sweep. Every threshold fixture is
hand-computed, with the arithmetic in a comment, per this repo's rule that a
green test proves the formula rather than agreement with the code that wrote
it.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.vision_object_permanence import InventoryRow, apply_sweep

T0 = datetime(2026, 8, 22, 0, 0, 0, tzinfo=timezone.utc)


def _row(label, first_seen, last_seen, count=1, state="present", state_since=None):
    return InventoryRow(
        stream_id="cam0", label=label, first_seen_at=first_seen, last_seen_at=last_seen,
        last_count=count, state=state, state_since=state_since or first_seen,
    )


# -- arrival ------------------------------------------------------------


def test_first_ever_sighting_is_an_arrival() -> None:
    r = apply_sweep(stream_id="cam0", window_max_counts={"box": 1}, existing={}, now=T0)
    assert "box" in r.updated
    assert r.updated["box"].first_seen_at == T0
    assert r.updated["box"].state == "present"
    assert len(r.transitions) == 1 and r.transitions[0].kind == "arrived"


def test_a_returning_label_keeps_its_original_first_seen_at() -> None:
    """Establishment duration must survive a departure-then-return, or the
    graduated threshold resets to the floor every time something briefly
    leaves and comes back."""
    original_first_seen = T0 - timedelta(days=3)
    existing = {"box": _row("box", original_first_seen, T0 - timedelta(hours=2),
                             state="departed", state_since=T0 - timedelta(hours=2))}
    r = apply_sweep(stream_id="cam0", window_max_counts={"box": 1}, existing=existing, now=T0)
    assert r.updated["box"].first_seen_at == original_first_seen
    assert r.updated["box"].state == "present"
    assert "returned" in r.transitions[0].detail


# -- quiet refresh vs count change --------------------------------------


def test_same_count_is_a_quiet_refresh_no_transition_logged() -> None:
    """The common case, every sweep. Must not spam."""
    existing = {"box": _row("box", T0 - timedelta(hours=1), T0 - timedelta(minutes=30), count=2)}
    r = apply_sweep(stream_id="cam0", window_max_counts={"box": 2}, existing=existing, now=T0)
    assert r.updated["box"].last_seen_at == T0
    assert r.transitions == []


def test_a_changed_count_is_logged_and_does_not_reset_first_seen() -> None:
    existing = {"box": _row("box", T0 - timedelta(hours=5), T0 - timedelta(minutes=30), count=2)}
    r = apply_sweep(stream_id="cam0", window_max_counts={"box": 1}, existing=existing, now=T0)
    assert r.updated["box"].last_count == 1
    assert r.updated["box"].first_seen_at == T0 - timedelta(hours=5)
    assert r.transitions[0].kind == "count_changed" and r.transitions[0].detail == "2 -> 1"


# -- the graduated threshold, hand-computed ------------------------------


def test_short_lived_object_departs_after_the_floor_not_the_fraction() -> None:
    """Established for 100s. 10% of that is 10s -- far below the 3600s floor
    (min_absence_sec default). The floor must win, or a coffee cup detected
    once would be declared departed on the very next sweep."""
    first_seen = T0 - timedelta(seconds=100)
    last_seen = T0 - timedelta(seconds=50)   # established_sec = 50 at time of loss
    existing = {"cup": _row("cup", first_seen, last_seen)}
    # gap = 3601s, just over the 3600s floor -> should depart
    r = apply_sweep(stream_id="cam0", window_max_counts={}, existing=existing,
                     now=last_seen + timedelta(seconds=3601))
    assert r.updated["cup"].state == "departed"
    assert r.transitions[0].kind == "departed"


def test_short_lived_object_survives_a_gap_under_the_floor() -> None:
    first_seen = T0 - timedelta(seconds=100)
    last_seen = T0 - timedelta(seconds=50)
    existing = {"cup": _row("cup", first_seen, last_seen)}
    # gap = 3599s, just under the 3600s floor -> must NOT depart yet
    r = apply_sweep(stream_id="cam0", window_max_counts={}, existing=existing,
                     now=last_seen + timedelta(seconds=3599))
    assert r.updated["cup"].state == "present"
    assert r.transitions == []


def test_long_established_object_gets_the_proportional_grace() -> None:
    """Established 100 days. 10% = 10 days -- far past the 24h ceiling, so
    this proves the ceiling actually CLAMPS something bigger, not merely that
    the fraction happens to equal it.

    First draft used established_sec=10 days, where 10% is EXACTLY 24h -- the
    ceiling and the raw fraction landed on the same number by coincidence, so
    removing the ceiling clamp entirely did not change the computed threshold
    and the mutation test for it passed for the wrong reason. Caught by
    mutation-testing this file against itself, not by inspection.
    """
    established_sec = 100 * 24 * 3600.0
    first_seen = T0 - timedelta(seconds=established_sec)
    last_seen = T0
    existing = {"desk": _row("desk", first_seen, last_seen)}

    # 23h59m59s of absence: under the 24h ceiling -> still present
    still_there = apply_sweep(stream_id="cam0", window_max_counts={}, existing=existing,
                               now=last_seen + timedelta(hours=24) - timedelta(seconds=1))
    assert still_there.updated["desk"].state == "present"

    # 24h + 1s of absence: over the ceiling -> departed
    gone = apply_sweep(stream_id="cam0", window_max_counts={}, existing=existing,
                        now=last_seen + timedelta(hours=24, seconds=1))
    assert gone.updated["desk"].state == "departed"


def test_a_medium_established_object_uses_the_fraction_not_a_clamp() -> None:
    """Established 50000s (~13.9h). 10% = 5000s, between the 3600s floor and
    the 86400s ceiling -- this is the case that actually exercises the
    fraction itself rather than one of the two bounds.

    First draft of this test used established_sec=20000, where 10% = 2000s --
    BELOW the 3600s floor, so the threshold there is silently clamped to 3600
    regardless of the fraction, and the "just_under/just_over 2000" boundary
    was testing the floor by accident, not the fraction. Caught by hand-
    checking the arithmetic rather than trusting the test passed.
    """
    established_sec = 50000.0
    first_seen = T0 - timedelta(seconds=established_sec)
    last_seen = T0
    existing = {"bag": _row("bag", first_seen, last_seen)}

    just_under = apply_sweep(stream_id="cam0", window_max_counts={}, existing=existing,
                              now=last_seen + timedelta(seconds=4999))
    assert just_under.updated["bag"].state == "present"

    just_over = apply_sweep(stream_id="cam0", window_max_counts={}, existing=existing,
                             now=last_seen + timedelta(seconds=5001))
    assert just_over.updated["bag"].state == "departed"


def test_a_departed_label_does_not_get_re_logged_every_sweep() -> None:
    existing = {"box": _row("box", T0 - timedelta(days=2), T0 - timedelta(days=1),
                             state="departed", state_since=T0 - timedelta(days=1))}
    r = apply_sweep(stream_id="cam0", window_max_counts={}, existing=existing, now=T0)
    assert r.updated["box"].state == "departed"
    assert r.transitions == []


def test_never_tracked_and_never_seen_produces_no_row() -> None:
    r = apply_sweep(stream_id="cam0", window_max_counts={}, existing={}, now=T0)
    assert r.updated == {} and r.transitions == []


# -- multi-label independence ---------------------------------------------


def test_labels_within_one_stream_do_not_interfere() -> None:
    """chair reappears this window; table stays silent but is established long
    enough (30 days) that a 10-hour gap is nowhere near its own threshold
    (10% of 30 days = 72h > 10h). Each label's own history decides its own
    fate -- chair's reappearance must not affect table's grace period, and
    table's silence must not affect chair's refresh.
    """
    existing = {
        "chair": _row("chair", T0 - timedelta(hours=10), T0 - timedelta(hours=10), count=1),
        "table": _row("table", T0 - timedelta(days=40), T0 - timedelta(hours=10), count=1),
    }
    r = apply_sweep(stream_id="cam0", window_max_counts={"chair": 1}, existing=existing, now=T0)
    assert r.updated["chair"].last_seen_at == T0
    assert r.updated["chair"].state == "present"
    assert r.updated["table"].last_seen_at == T0 - timedelta(hours=10)   # untouched
    assert r.updated["table"].state == "present"                        # within its own grace
    kinds = {t.label: t.kind for t in r.transitions}
    assert "chair" not in kinds, "same count on reappearance must be a quiet refresh"
    assert "table" not in kinds, "still within grace must not log anything"
