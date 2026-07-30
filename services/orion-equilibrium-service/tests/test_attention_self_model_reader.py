"""Row-parsing tests for the generative gates' Postgres reader.

The connection handling itself needs a live Postgres, but `parse_confidence_samples`
is the part that decides what the detectors actually see, and it has two
non-obvious contracts worth locking down: it must return oldest -> newest
regardless of the query's own ordering, and it must drop rows that carry no
usable confidence value rather than pass a hole downstream.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.attention_self_model_reader import parse_confidence_samples

T0 = datetime(2026, 7, 30, 12, 0, 0, tzinfo=timezone.utc)
TICK = timedelta(seconds=30)


def test_newest_first_rows_are_reordered_oldest_to_newest() -> None:
    """The live query uses `ORDER BY generated_at DESC LIMIT n`, but both
    detectors require ascending order -- getting this backwards would silently
    invert every recovery into a decline."""
    rows = [
        ({"prediction_error_confidence": 0.93}, T0 + 2 * TICK),
        ({"prediction_error_confidence": 0.80}, T0 + 1 * TICK),
        ({"prediction_error_confidence": 0.66}, T0),
    ]
    samples = parse_confidence_samples(rows)
    assert [s.value for s in samples] == [0.66, 0.80, 0.93]
    assert [s.generated_at for s in samples] == [T0, T0 + TICK, T0 + 2 * TICK]


def test_json_string_payload_is_parsed() -> None:
    rows = [('{"prediction_error_confidence": 0.91}', T0)]
    samples = parse_confidence_samples(rows)
    assert len(samples) == 1
    assert samples[0].value == 0.91


def test_rows_without_a_usable_confidence_are_dropped() -> None:
    rows = [
        ({"prediction_error_confidence": 0.91}, T0),
        ({}, T0 + TICK),  # field absent
        ({"prediction_error_confidence": None}, T0 + 2 * TICK),
        ({"prediction_error_confidence": "0.9"}, T0 + 3 * TICK),  # wrong type
        ({"prediction_error_confidence": True}, T0 + 4 * TICK),  # bool is not a reading
        ({"prediction_error_confidence": float("nan")}, T0 + 5 * TICK),
        ({"prediction_error_confidence": float("inf")}, T0 + 6 * TICK),
        ({"prediction_error_confidence": 0.94}, T0 + 7 * TICK),
    ]
    samples = parse_confidence_samples(rows)
    assert [s.value for s in samples] == [0.91, 0.94]


def test_null_timestamp_and_malformed_payloads_are_dropped() -> None:
    rows = [
        ({"prediction_error_confidence": 0.91}, None),
        ("not json at all", T0),
        (["unexpected", "list"], T0 + TICK),
        (None, T0 + 2 * TICK),
        ({"prediction_error_confidence": 0.92}, T0 + 3 * TICK),
    ]
    samples = parse_confidence_samples(rows)
    assert [s.value for s in samples] == [0.92]


def test_naive_timestamp_is_treated_as_utc() -> None:
    naive = datetime(2026, 7, 30, 12, 0, 0)
    samples = parse_confidence_samples([({"prediction_error_confidence": 0.9}, naive)])
    assert samples[0].generated_at.tzinfo is not None
    assert samples[0].generated_at == T0


def test_empty_input_returns_empty_list() -> None:
    assert parse_confidence_samples([]) == []
