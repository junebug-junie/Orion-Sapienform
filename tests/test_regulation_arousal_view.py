"""A reader's view of the arousal reading: only a fresh, parseable, known level counts."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orion.regulation.arousal_view import (
    UNKNOWN_DISABLED,
    arousal_view,
    engaged_cooldown_sec,
)
from orion.schemas.regulation import ArousalReadingV1, RegulationStateV1

NOW = datetime(2026, 10, 12, 15, 0, tzinfo=timezone.utc)


def state_json(level: str, *, age_sec: float = 30.0, reasons=None) -> str:
    at = NOW - timedelta(seconds=age_sec)
    reading = ArousalReadingV1(
        arousal_level=level, since=at - timedelta(minutes=5), observed_at=at,
        reasons=list(reasons or [f"test_{level}"]),
    )
    return RegulationStateV1(generated_at=at, arousal=reading).model_dump_json()


@pytest.mark.parametrize("level", ["engaged", "idle", "strained"])
def test_fresh_known_levels_pass_through(level):
    v = arousal_view(state_json(level), now=NOW)
    assert v.level == level and v.known and v.age_sec == 30.0


@pytest.mark.parametrize(
    "raw,reason",
    [
        (None, "absent"),
        ("{not json", "unparseable"),
        (b'{"schema_version": "wrong"}', "unparseable"),
    ],
)
def test_absent_or_malformed_is_unknown(raw, reason):
    v = arousal_view(raw, now=NOW)
    assert v.level == "unknown" and v.reason == reason


def test_stale_is_unknown_even_if_strained():
    v = arousal_view(state_json("strained", age_sec=361), now=NOW, max_age_sec=360)
    assert v.level == "unknown" and v.reason == "stale"


def test_future_stamped_is_unknown():
    v = arousal_view(state_json("idle", age_sec=-301), now=NOW)
    assert v.level == "unknown" and v.reason == "future_stamped"


def test_producer_unknown_and_disabled_stay_unknown():
    v = arousal_view(state_json("unknown", reasons=["disabled"]), now=NOW)
    assert v.level == "unknown" and v.reason == "producer_unknown:disabled"


def test_engaged_cooldown_only_when_engaged_and_multiplier_above_one():
    engaged = arousal_view(state_json("engaged"), now=NOW)
    assert engaged_cooldown_sec(100.0, engaged, 2.0) == 200.0
    assert engaged_cooldown_sec(100.0, engaged, 1.0) is None
    assert engaged_cooldown_sec(100.0, engaged, 0.5) is None
    for level in ("idle", "strained", "unknown"):
        assert engaged_cooldown_sec(100.0, arousal_view(state_json(level), now=NOW), 2.0) is None
    assert engaged_cooldown_sec(100.0, UNKNOWN_DISABLED, 2.0) is None
