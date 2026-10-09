"""#2534 decision 2 (approved 2026-10-07): attention novelty must not count a
channel going dark (measured -> key gone) or coming back as a change.

Before: capability:vision pressure 0.85, then the frame router dies and the
key is dropped as unmeasured; the proxy fell 0.85 -> 0.0 and novelty read 0.85,
so a dead eye looked like the most surprising thing in the field. These fail
on main (build_attention_frame had no previous_field).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from orion.attention.field_attention.builder import build_attention_frame
from orion.attention.field_attention.policy import load_attention_policy
from orion.schemas.field_state import FieldStateV1

REPO = Path(__file__).resolve().parents[1]
POLICY = load_attention_policy(REPO / "config" / "attention" / "field_attention_policy.v1.yaml")
T0 = datetime(2026, 10, 7, 12, 0, tzinfo=timezone.utc)


def _field(tick_id: str, at: datetime, vision: dict[str, float], athena: dict[str, float]) -> FieldStateV1:
    return FieldStateV1(
        tick_id=tick_id,
        generated_at=at,
        node_vectors={"node:athena": athena},
        capability_vectors={"capability:vision": vision, "capability:storage": {"pressure": 0.2}},
    )


def _all_targets(frame):
    return {
        t.target_id: t
        for bucket in (frame.dominant_targets, frame.suppressed_targets, frame.capability_targets, frame.node_targets)
        for t in bucket
    }


def _pair(before: FieldStateV1, after: FieldStateV1, *, with_previous_field: bool = True):
    prev = build_attention_frame(field=before, policy=POLICY, now=before.generated_at)
    # second frame needs a real prior frame; build it twice so the prior has entries
    prev = build_attention_frame(field=before, policy=POLICY, previous_frame=prev, now=before.generated_at)
    return build_attention_frame(
        field=after,
        policy=POLICY,
        previous_frame=prev,
        previous_field=before if with_previous_field else None,
        now=after.generated_at,
    )


BEFORE = _field("t1", T0, {"pressure": 0.85, "reliability_pressure": 0.1}, {"cpu_pressure": 0.3, "rpc_timeout_pressure": 0.4})


def test_capability_going_dark_is_not_novel() -> None:
    after = _field("t2", T0 + timedelta(seconds=2), {"reliability_pressure": 0.1}, {"cpu_pressure": 0.3, "rpc_timeout_pressure": 0.4})
    frame = _pair(BEFORE, after)
    assert _all_targets(frame)["capability:vision"].novelty_score == 0.0
    vision = next(
        t for b in (frame.capability_targets, frame.suppressed_targets) for t in b if t.target_id == "capability:vision"
    )
    assert any(r.startswith("novelty_common_channels_only went_dark=['pressure']") for r in vision.reasons)
    # without the previous field (legacy path) the outage reads as 0.75 of news
    legacy = _pair(BEFORE, after, with_previous_field=False)
    assert _all_targets(legacy)["capability:vision"].novelty_score > 0.7


def test_channel_coming_back_is_not_novel_but_a_real_move_still_is() -> None:
    dark = _field("t1", T0, {"reliability_pressure": 0.1}, {"cpu_pressure": 0.3})
    back = _field("t2", T0 + timedelta(seconds=2), {"pressure": 0.85, "reliability_pressure": 0.6}, {"cpu_pressure": 0.3, "rpc_timeout_pressure": 0.9})
    targets = _all_targets(_pair(dark, back))
    # vision: reliability measured in both moved 0.1 -> 0.6; pressure's return is ignored
    assert abs(targets["capability:vision"].novelty_score - 0.5) < 1e-9
    # host node: rpc_timeout_pressure returning is not news; cpu unchanged
    assert targets["node:athena"].novelty_score == 0.0


def test_unchanged_measurement_set_matches_legacy_exactly() -> None:
    after = _field("t2", T0 + timedelta(seconds=2), {"pressure": 0.5, "reliability_pressure": 0.1}, {"cpu_pressure": 0.6, "rpc_timeout_pressure": 0.4})
    new = {k: t.novelty_score for k, t in _all_targets(_pair(BEFORE, after)).items()}
    old = {k: t.novelty_score for k, t in _all_targets(_pair(BEFORE, after, with_previous_field=False)).items()}
    assert new == old


def test_previous_field_from_another_tick_is_ignored() -> None:
    after = _field("t2", T0 + timedelta(seconds=2), {"reliability_pressure": 0.1}, {"cpu_pressure": 0.3})
    wrong = BEFORE.model_copy(update={"tick_id": "not-the-previous-frames-tick"})
    prev = build_attention_frame(field=BEFORE, policy=POLICY, now=T0)
    frame = build_attention_frame(field=after, policy=POLICY, previous_frame=prev, previous_field=wrong, now=after.generated_at)
    assert _all_targets(frame)["capability:vision"].novelty_score > 0.7
