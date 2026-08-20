"""Unit tests for orion.reverie.visual_storage — content-addressed writes.

Deterministic, filesystem-only. Always writes to a temp dir passed as
`base_dir`, never the real /mnt/storage-lukewarm mount (CLAUDE.md discipline
plus visual_storage's own docstring warning).
"""
from __future__ import annotations

import hashlib

import pytest

from orion.reverie.visual_storage import (
    UnsupportedImageError,
    load_visual_artifact,
    sniff_image,
    store_visual_artifact,
)

# 1x1 pixel PNG (real magic bytes + IHDR chunk, minimal valid image).
_PNG_1X1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d494844520000000100000001080600000"
    "01f15c4890000000a49444154789c6300010000050001"
    "0d0a2db40000000049454e44ae426082"
)

_NOT_AN_IMAGE = b"this is definitely not an image, just plain text bytes"


def test_sniff_image_recognises_real_png():
    result = sniff_image(_PNG_1X1)
    assert result is not None
    mime, width, height = result
    assert mime == "image/png"
    assert (width, height) == (1, 1)


def test_sniff_image_rejects_non_image_bytes():
    assert sniff_image(_NOT_AN_IMAGE) is None


def test_store_writes_content_addressed_file(tmp_path):
    result = store_visual_artifact(_PNG_1X1, base_dir=tmp_path)

    expected_sha = hashlib.sha256(_PNG_1X1).hexdigest()
    assert result.sha256 == expected_sha
    assert result.mime == "image/png"
    assert result.bytes == len(_PNG_1X1)
    assert result.width == 1
    assert result.height == 1

    written = tmp_path / f"{expected_sha}.png"
    assert written.exists()
    assert written.read_bytes() == _PNG_1X1
    # No leftover temp/part files after a successful write.
    assert list(tmp_path.glob("*.part")) == []


def test_store_is_idempotent_on_duplicate_content(tmp_path):
    first = store_visual_artifact(_PNG_1X1, base_dir=tmp_path)
    files_after_first = sorted(tmp_path.iterdir())

    second = store_visual_artifact(_PNG_1X1, base_dir=tmp_path)
    files_after_second = sorted(tmp_path.iterdir())

    assert first.sha256 == second.sha256
    assert first.path == second.path
    # Storing the same bytes twice must not create a second file.
    assert files_after_first == files_after_second


def test_store_rejects_unsupported_bytes(tmp_path):
    with pytest.raises(UnsupportedImageError):
        store_visual_artifact(_NOT_AN_IMAGE, base_dir=tmp_path)
    # No file was written for the rejected content.
    assert list(tmp_path.iterdir()) == [] if tmp_path.exists() else True


def test_store_rejects_empty_bytes(tmp_path):
    with pytest.raises(ValueError):
        store_visual_artifact(b"", base_dir=tmp_path)


def test_mime_is_sniffed_not_trusted(tmp_path):
    # There is no caller-supplied mime parameter at all -- the only way to
    # get a mime out of store_visual_artifact is via real magic-byte
    # sniffing. This test pins that contract: passing non-image bytes never
    # silently succeeds with a guessed/trusted mime.
    with pytest.raises(UnsupportedImageError):
        store_visual_artifact(_NOT_AN_IMAGE, base_dir=tmp_path)


def test_load_round_trips_stored_bytes(tmp_path):
    stored = store_visual_artifact(_PNG_1X1, base_dir=tmp_path)
    loaded = load_visual_artifact(stored.sha256, base_dir=tmp_path)
    assert loaded == _PNG_1X1


def test_load_raises_for_missing_artifact(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_visual_artifact("0" * 64, base_dir=tmp_path)
