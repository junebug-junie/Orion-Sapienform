from __future__ import annotations

import logging

import pytest

from orion.graph.property_guard import (
    METADATA_MAX_BYTES,
    METADATA_MAX_KEYS,
    PropertyCathedralError,
    sanitize_metadata,
)


def test_clean_metadata_passes_unchanged() -> None:
    metadata = {"source": "test", "confidence": 0.9, "tags": ["a", "b"]}
    sanitized, rejected = sanitize_metadata(metadata)
    assert sanitized == metadata
    assert rejected == []


def test_sixteenth_key_kept_seventeenth_dropped() -> None:
    metadata = {f"k{i}": i for i in range(17)}
    sanitized, rejected = sanitize_metadata(metadata)
    assert len(sanitized) == METADATA_MAX_KEYS
    assert sanitized == {f"k{i}": i for i in range(METADATA_MAX_KEYS)}
    assert rejected == ["k16"]


def test_insertion_order_retained_not_sorted() -> None:
    metadata = {"z": 1, "a": 2, "m": 3}
    extra = {f"extra_{i}": i for i in range(14)}
    metadata = {**metadata, **extra, "last": 99}
    sanitized, rejected = sanitize_metadata(metadata)
    assert list(sanitized.keys()) == list(metadata.keys())[:METADATA_MAX_KEYS]
    assert "last" in rejected


def test_oversized_metadata_drops_keys_from_end() -> None:
    big_value = "x" * (METADATA_MAX_BYTES // 2)
    metadata = {"keep": "small", "drop_me": big_value, "also_drop": big_value}
    sanitized, rejected = sanitize_metadata(metadata)
    assert _byte_size(sanitized) <= METADATA_MAX_BYTES
    assert "drop_me" in rejected or "also_drop" in rejected
    assert sanitized.get("keep") == "small"


def test_fail_closed_raises_on_excess_keys() -> None:
    metadata = {f"k{i}": i for i in range(17)}
    with pytest.raises(PropertyCathedralError, match="max keys"):
        sanitize_metadata(metadata, fail_closed=True)


def test_fail_closed_raises_on_oversize() -> None:
    metadata = {"payload": "x" * (METADATA_MAX_BYTES + 1)}
    with pytest.raises(PropertyCathedralError, match="max bytes"):
        sanitize_metadata(metadata, fail_closed=True)


def test_non_dict_fail_open_returns_empty() -> None:
    sanitized, rejected = sanitize_metadata(["not", "a", "dict"])
    assert sanitized == {}
    assert rejected == ["<non-dict>"]


def test_non_dict_fail_closed_raises() -> None:
    with pytest.raises(PropertyCathedralError, match="must be a dict"):
        sanitize_metadata("nope", fail_closed=True)


def test_rejection_logged(caplog: pytest.LogCaptureFixture) -> None:
    metadata = {f"k{i}": i for i in range(17)}
    with caplog.at_level(logging.WARNING):
        sanitize_metadata(metadata)
    assert any("property_cathedral_rejected" in r.message for r in caplog.records)
    assert any("k16" in r.message for r in caplog.records)


def _byte_size(metadata: dict) -> int:
    import json

    payload = json.dumps(metadata, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return len(payload.encode("utf-8"))
