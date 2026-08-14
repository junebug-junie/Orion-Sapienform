"""Store + sniffing tests for chat attachments.

Fixtures are built byte-by-byte here rather than checked in as binaries, so the
expected width/height are derived from bytes this file wrote on purpose -- an
assertion that passes because the encoder and the decoder happen to agree is
not evidence.
"""

from __future__ import annotations

import hashlib
import os
import struct
import sys
import zlib
from pathlib import Path

import pytest
from fastapi import HTTPException

REPO_ROOT = Path(__file__).resolve().parents[3]
HUB_ROOT = Path(__file__).resolve().parents[1]
for candidate in (str(REPO_ROOT), str(HUB_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

for key, value in {
    "CHANNEL_VOICE_TRANSCRIPT": "orion:voice:transcript",
    "CHANNEL_VOICE_LLM": "orion:voice:llm",
    "CHANNEL_VOICE_TTS": "orion:voice:tts",
    "CHANNEL_COLLAPSE_INTAKE": "orion:collapse:intake",
    "CHANNEL_COLLAPSE_TRIAGE": "orion:collapse:triage",
}.items():
    os.environ.setdefault(key, value)

from scripts import chat_attachments as ca  # noqa: E402


# ───────────────────────────────────────────────────────────────
# Hand-built fixtures
# ───────────────────────────────────────────────────────────────

def make_png(width: int, height: int) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + b"\xdc\x14\x14" * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def make_gif(width: int, height: int) -> bytes:
    return b"GIF89a" + struct.pack("<HH", width, height) + b"\x80\x00\x00" + b"\x00" * 8


def make_jpeg(width: int, height: int) -> bytes:
    # SOI, a COM segment to prove the walker skips segments, then SOF0.
    comment = b"\xff\xfe" + struct.pack(">H", 2 + 4) + b"orio"
    sof0 = b"\xff\xc0" + struct.pack(">H", 11) + b"\x08" + struct.pack(">HH", height, width) + b"\x01\x01\x11\x00"
    return b"\xff\xd8" + comment + sof0 + b"\xff\xd9"


def make_webp_vp8x(width: int, height: int) -> bytes:
    payload = b"WEBP" + b"VP8X" + struct.pack("<I", 10) + b"\x00\x00\x00\x00"
    payload += (width - 1).to_bytes(3, "little") + (height - 1).to_bytes(3, "little")
    return b"RIFF" + struct.pack("<I", len(payload)) + payload


# ───────────────────────────────────────────────────────────────
# Sniffing
# ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "builder,mime,width,height",
    [
        (make_png, "image/png", 64, 48),
        (make_gif, "image/gif", 12, 34),
        (make_jpeg, "image/jpeg", 800, 600),
        (make_webp_vp8x, "image/webp", 1920, 1080),
    ],
)
def test_sniff_reads_format_and_dimensions(builder, mime, width, height):
    assert ca.sniff_image(builder(width, height)) == (mime, width, height)


def test_sniff_rejects_non_image():
    assert ca.sniff_image(b"#!/bin/sh\nrm -rf /\n") is None
    assert ca.sniff_image(b"") is None
    assert ca.sniff_image(b"%PDF-1.7\n") is None


def test_sniff_rejects_png_signature_with_truncated_ihdr():
    assert ca.sniff_image(b"\x89PNG\r\n\x1a\n" + b"\x00" * 4) is None


# ───────────────────────────────────────────────────────────────
# Store
# ───────────────────────────────────────────────────────────────

@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(ca.settings, "HUB_CHAT_ATTACHMENT_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(ca.settings, "HUB_CHAT_ATTACHMENT_MAX_BYTES", 4096, raising=False)
    monkeypatch.setattr(
        ca.settings,
        "HUB_CHAT_ATTACHMENT_ALLOWED_MIMES",
        "image/png,image/jpeg,image/webp,image/gif",
        raising=False,
    )
    monkeypatch.setattr(ca.settings, "HUB_CHAT_ATTACHMENT_PUBLIC_BASE", "http://orion-hub:8000", raising=False)
    return tmp_path


def test_store_round_trip_is_content_addressed(store):
    data = make_png(8, 8)
    ref = ca.store_bytes(data, filename="shot.png")

    assert ref.sha256 == hashlib.sha256(data).hexdigest()
    assert ref.mime == "image/png"
    assert ref.bytes == len(data)
    assert (ref.width, ref.height) == (8, 8)
    assert ref.filename == "shot.png"
    assert ref.source_url == f"http://orion-hub:8000/api/chat/attachments/{ref.sha256}"

    loaded, mime = ca.load_bytes(ref.sha256)
    assert loaded == data
    assert mime == "image/png"


def test_store_is_idempotent_on_identical_content(store):
    data = make_png(8, 8)
    first = ca.store_bytes(data, filename="a.png")
    second = ca.store_bytes(data, filename="b.png")

    assert first.sha256 == second.sha256
    # One image on disk, not two, plus its mime sidecar.
    assert sorted(p.suffix for p in store.iterdir()) == [".bin", ".mime"]


def test_declared_mime_is_ignored_in_favour_of_magic_bytes(store):
    """A .png filename on JPEG bytes stores as image/jpeg, not image/png."""
    ref = ca.store_bytes(make_jpeg(4, 4), filename="lying.png")
    assert ref.mime == "image/jpeg"


def test_oversize_upload_is_refused(store):
    with pytest.raises(HTTPException) as exc:
        ca.store_bytes(make_png(64, 64) + b"\x00" * 4096)
    assert exc.value.status_code == 413


def test_non_image_upload_is_refused(store):
    with pytest.raises(HTTPException) as exc:
        ca.store_bytes(b"<svg onload=alert(1)></svg>")
    assert exc.value.status_code == 415


def test_disabled_mime_is_refused_even_when_sniffable(store, monkeypatch):
    monkeypatch.setattr(ca.settings, "HUB_CHAT_ATTACHMENT_ALLOWED_MIMES", "image/png", raising=False)
    with pytest.raises(HTTPException) as exc:
        ca.store_bytes(make_gif(4, 4))
    assert exc.value.status_code == 415


def test_empty_upload_is_refused(store):
    with pytest.raises(HTTPException) as exc:
        ca.store_bytes(b"")
    assert exc.value.status_code == 400


@pytest.mark.parametrize(
    "bad",
    ["../../etc/passwd", "a" * 63, "a" * 65, "A" * 64, "../" + "a" * 61, "", "g" * 64],
)
def test_path_traversal_and_malformed_ids_are_refused(store, bad):
    with pytest.raises(HTTPException) as exc:
        ca.load_bytes(bad)
    assert exc.value.status_code == 400


def test_missing_attachment_is_404(store):
    with pytest.raises(HTTPException) as exc:
        ca.load_bytes("b" * 64)
    assert exc.value.status_code == 404


def test_load_resniffs_when_mime_sidecar_is_missing(store):
    ref = ca.store_bytes(make_gif(6, 6))
    (store / f"{ref.sha256}.mime").unlink()
    _, mime = ca.load_bytes(ref.sha256)
    assert mime == "image/gif"
