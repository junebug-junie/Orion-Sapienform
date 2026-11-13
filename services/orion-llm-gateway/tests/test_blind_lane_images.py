"""Images placed on a blind lane become a text note (design 2026-11-13-camera-look-design.md).

The Claude CLI's Read tool returns a camera frame as an image block inside a tool_result. The pool
may place that call on a lane with no vision projector; there the image must become a note the
model can read, and on a sighted lane it must pass through untouched.
"""
from __future__ import annotations

import copy
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app import vision
from app.main import app
from app.settings import settings
from app.vision import VisionCapability

IMAGE = {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "/9j/AA=="}}


def _read_step() -> dict:
    """A Claude CLI step after `Read /mnt/orion-fcc/look/.../walkway-1.jpg`."""
    return {
        "model": "agent", "max_tokens": 64, "stream": False,
        "messages": [
            {"role": "user", "content": "Look at the walkway."},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Read",
                                               "input": {"file_path": "/mnt/orion-fcc/look/x.jpg"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": [IMAGE]}]},
        ],
    }


def test_count_and_replace_reach_inside_tool_results() -> None:
    body = _read_step()
    body["messages"][0]["content"] = [IMAGE, {"type": "text", "text": "and this"}]
    original = copy.deepcopy(body)
    assert vision.count_images(body, anthropic=True) == 2
    out, n = vision.replace_images_with_note(body, anthropic=True, reason="no mmproj")
    assert n == 2 and body == original  # the input is never changed
    assert vision.count_images(out, anthropic=True) == 0
    note = out["messages"][2]["content"][0]["content"][0]
    assert note["type"] == "text" and "cannot see images (no mmproj)" in note["text"]
    assert out["messages"][1] is body["messages"][1]  # untouched messages pass as they were


def test_openai_image_url_parts() -> None:
    body = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "x"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]}]}
    assert vision.count_images(body, anthropic=True) == 0
    out, n = vision.replace_images_with_note(body, anthropic=False, reason="r")
    assert n == 1 and out["messages"][0]["content"][1]["type"] == "text"


def test_no_images_returns_the_same_body() -> None:
    body = {"messages": [{"role": "user", "content": "hi"}]}
    out, n = vision.replace_images_with_note(body, anthropic=True, reason="r")
    assert n == 0 and out is body


def _post_client() -> AsyncMock:
    response = MagicMock(status_code=200, content=b'{"id":"m","content":[{"type":"text","text":"ok"}]}',
                         headers={"content-type": "application/json"})
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.__aexit__.return_value = False
    client.post = AsyncMock(return_value=response)
    return client


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, fake_pool) -> TestClient:
    monkeypatch.setattr(settings, "llm_gateway_anthropic_passthrough_enabled", True)
    return TestClient(app)


@pytest.mark.parametrize("sees", [False, True])
@patch("app.passthrough_proxy.httpx.AsyncClient")
def test_passthrough_strips_images_only_on_a_blind_lane(mock_cls: MagicMock, client: TestClient, sees: bool) -> None:
    upstream = _post_client()
    mock_cls.return_value = upstream
    probed: list[str] = []

    def probe(base_url: str) -> VisionCapability:
        probed.append(base_url)
        return VisionCapability(vision=sees, source="props", detail=None if sees else "worker reports modalities.vision=false")

    with patch("app.passthrough_proxy.resolve_vision_capability", side_effect=probe):
        response = client.post("/v1/messages", headers={"anthropic-version": "2023-06-01"}, json=_read_step())

    assert response.status_code == 200
    assert probed == ["http://pool-agent:8015"]  # the lane the pool granted, not a configured guess
    sent = upstream.post.await_args.kwargs["json"]["messages"][2]["content"][0]["content"][0]
    if sees:
        assert sent == IMAGE
    else:
        assert sent["type"] == "text" and "say you could not see it" in sent["text"]


@patch("app.passthrough_proxy.httpx.AsyncClient")
def test_text_only_call_never_probes(mock_cls: MagicMock, client: TestClient) -> None:
    mock_cls.return_value = _post_client()
    with patch("app.passthrough_proxy.resolve_vision_capability") as probe:
        response = client.post("/v1/messages", json={"model": "agent", "max_tokens": 8,
                                                     "messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 200
    probe.assert_not_called()
