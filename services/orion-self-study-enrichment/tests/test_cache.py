from __future__ import annotations

import sys
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.cache import content_hash, read_cached, write_cached  # noqa: E402


def test_content_hash_deterministic():
    assert content_hash("hello") == content_hash("hello")
    assert content_hash("hello") != content_hash("world")


def test_write_then_read_roundtrip(tmp_path):
    key = content_hash("some prompt text")
    write_cached(tmp_path, key, {"summary": "grounded summary"})
    result = read_cached(tmp_path, key)
    assert result == {"summary": "grounded summary"}


def test_read_cached_missing_returns_none(tmp_path):
    assert read_cached(tmp_path, "deadbeef") is None


def test_write_cached_uses_two_level_fanout(tmp_path):
    key = content_hash("x")
    path = write_cached(tmp_path, key, {"a": 1})
    assert path.parent.name == key[:2]
