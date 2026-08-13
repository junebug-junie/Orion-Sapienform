"""Round-trip test for perception_staleness through the FalkorDB codec.

Regression for a defect found only by deploying and querying the node: the
codec's closed allowlist plus its two hardcoded encode/decode dicts silently
dropped this key, so node:substrate.vision persisted with the channel NULL
while the in-process cache made it look present. The same trap swallowed
prediction_error_evidence_event_ids on 2026-08-11.

It matters more for this key than for most: the availability channel exists
because a silent bus channel freezes its gap_zscore instead of raising it, so
losing it in transit leaves capability:vision reading a dead camera as calm.
"""

from __future__ import annotations

from orion.substrate.falkor_codec import (
    DYNAMICS_METADATA_KEYS,
    _dynamics_properties_from_metadata,
)


def test_key_is_in_the_closed_allowlist() -> None:
    assert "perception_staleness" in DYNAMICS_METADATA_KEYS


def test_encoder_promotes_the_value_to_a_native_property() -> None:
    props = _dynamics_properties_from_metadata({"perception_staleness": 0.42})
    assert props["perception_staleness"] == 0.42


def test_encoder_keeps_a_real_zero_rather_than_dropping_it() -> None:
    """
    0.0 is the healthy reading and must persist as a value, not vanish into
    "absent" -- a falsy-value filter here would erase exactly the case that
    proves the eye is alive.
    """
    props = _dynamics_properties_from_metadata({"perception_staleness": 0.0})
    assert props["perception_staleness"] == 0.0
    assert props["perception_staleness"] is not None


def test_absent_stays_absent_and_is_not_coerced_to_calm() -> None:
    # "No vision reading yet" must not encode as "measured, and fine".
    props = _dynamics_properties_from_metadata({})
    assert props["perception_staleness"] is None


def test_saturated_value_survives_encoding() -> None:
    props = _dynamics_properties_from_metadata({"perception_staleness": 1.0})
    assert props["perception_staleness"] == 1.0


def test_malformed_value_fails_open_to_absent() -> None:
    props = _dynamics_properties_from_metadata({"perception_staleness": "not-a-number"})
    assert props["perception_staleness"] is None


def test_prediction_error_still_encodes_alongside_it() -> None:
    props = _dynamics_properties_from_metadata(
        {"prediction_error": 0.3, "perception_staleness": 0.9}
    )
    assert props["prediction_error"] == 0.3
    assert props["perception_staleness"] == 0.9
