"""HTTP round-trip for the attachment endpoints, through the real router."""

from __future__ import annotations

import hashlib
import os
import struct
import sys
import zlib
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

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


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(ca.settings, "HUB_CHAT_ATTACHMENT_DIR", str(tmp_path), raising=False)
    app = FastAPI()
    app.include_router(ca.router)
    with TestClient(app) as c:
        yield c


def test_upload_then_fetch_round_trip(client):
    data = make_png(16, 12)
    resp = client.post(
        "/api/chat/attachments",
        files={"file": ("shot.png", data, "image/png")},
    )
    assert resp.status_code == 200, resp.text
    ref = resp.json()

    assert ref["sha256"] == hashlib.sha256(data).hexdigest()
    assert ref["mime"] == "image/png"
    assert ref["bytes"] == len(data)
    assert ref["width"] == 16 and ref["height"] == 12
    assert ref["kind"] == "image"
    assert ref["source_url"].endswith(f"/api/chat/attachments/{ref['sha256']}")

    got = client.get(f"/api/chat/attachments/{ref['sha256']}")
    assert got.status_code == 200
    assert got.content == data
    assert got.headers["content-type"].startswith("image/png")
    assert "immutable" in got.headers["cache-control"]
    assert got.headers["x-content-type-options"] == "nosniff"


def test_upload_rejects_a_non_image(client):
    resp = client.post(
        "/api/chat/attachments",
        files={"file": ("payload.png", b"<svg onload=alert(1)></svg>", "image/png")},
    )
    assert resp.status_code == 415


def test_fetch_rejects_a_traversal_id(client):
    assert client.get("/api/chat/attachments/..%2F..%2Fetc%2Fpasswd").status_code in (400, 404)
    assert client.get("/api/chat/attachments/nope").status_code == 400


def test_fetch_unknown_sha_is_404(client):
    assert client.get(f"/api/chat/attachments/{'e' * 64}").status_code == 404


def test_oversize_upload_is_refused(client, monkeypatch):
    monkeypatch.setattr(ca.settings, "HUB_CHAT_ATTACHMENT_MAX_BYTES", 64, raising=False)
    resp = client.post(
        "/api/chat/attachments",
        files={"file": ("big.png", make_png(64, 64), "image/png")},
    )
    assert resp.status_code == 413
