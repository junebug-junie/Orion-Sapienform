"""Bus-free by design (mocked bus, mocked percept-store fetch) -- covers
capture_and_assess(): the live-capture bridge added 2026-08-22
(POST /v1/juniper/affect/capture_and_assess, called by Hub's "Affect check"
toggle). See test_trigger.py for the pre-existing manual-path coverage this
builds on; that file's FakeBus supports only one configured RPC reply,
insufficient here since capture_and_assess makes two DIFFERENT RPC calls
(retina, then the worker) -- this file's FakeBus keys replies by the
request channel instead.
"""
from __future__ import annotations

import os
import time
from typing import Any

import pytest

from app.main import JuniperAffectiveStateService, PerceptFetchError, settings
from orion.schemas.affectgpt import AffectGptAssessResultPayload
from orion.schemas.vision import RetinaClipCaptureResultPayload


class FakeEnvelope:
    def __init__(self, payload: dict):
        self.payload = payload


class FakeDecoded:
    def __init__(self, ok: bool, envelope: Any = None, error: str | None = None):
        self.ok = ok
        self.envelope = envelope
        self.error = error


class FakeBus:
    """Routes rpc_request replies by request_channel, since capture_and_assess
    calls two different channels (retina, then the worker) in one flow."""

    def __init__(self):
        self.enabled = True
        self.published: list[tuple[str, Any]] = []
        self._replies: dict[str, dict] = {}
        self._raises: dict[str, Exception] = {}
        self.calls: list[str] = []

    def set_reply(self, channel: str, payload: dict) -> None:
        self._replies[channel] = payload

    def set_raises(self, channel: str, exc: Exception) -> None:
        self._raises[channel] = exc

    async def rpc_request(self, request_channel, envelope, *, reply_channel, timeout_sec):
        self.calls.append(request_channel)
        if request_channel in self._raises:
            raise self._raises[request_channel]
        return {"data": request_channel}

    @property
    def codec(self):
        outer = self

        class _Codec:
            def decode(self, data):
                payload = outer._replies.get(data)
                if payload is None:
                    return FakeDecoded(ok=False, error=f"no reply configured for {data}")
                return FakeDecoded(ok=True, envelope=FakeEnvelope(payload))

        return _Codec()

    async def publish(self, channel, envelope):
        self.published.append((channel, envelope))


@pytest.fixture(autouse=True)
def scratch_dir(tmp_path, monkeypatch):
    """Point AFFECTGPT_SCRATCH_DIR at a real tmp_path instead of the
    production /mnt/scripts/... default -- capture_and_assess() actually
    writes files here."""
    monkeypatch.setattr(settings, "AFFECTGPT_SCRATCH_DIR", str(tmp_path))
    return tmp_path


def _svc_with_fake_bus() -> tuple[JuniperAffectiveStateService, FakeBus]:
    svc = JuniperAffectiveStateService()
    bus = FakeBus()
    svc.bus = bus
    return svc, bus


@pytest.mark.asyncio
async def test_full_round_trip_fetches_and_assesses(monkeypatch, scratch_dir):
    svc, bus = _svc_with_fake_bus()
    bus.set_reply(
        settings.CHANNEL_RETINA_CLIP_INTAKE,
        RetinaClipCaptureResultPayload(
            ok=True, video_sha256="a" * 64, audio_sha256="b" * 64, duration_sec=8.0
        ).model_dump(),
    )
    bus.set_reply(
        settings.CHANNEL_AFFECTGPT_INTAKE,
        AffectGptAssessResultPayload(ok=True, raw_response="sad, contemplative").model_dump(),
    )

    fetched: list[str] = []

    def _fake_fetch(self, sha256, dest_path):
        fetched.append(sha256)
        with open(dest_path, "wb") as f:
            f.write(b"fake-bytes-for-" + sha256.encode())
        return dest_path

    monkeypatch.setattr(JuniperAffectiveStateService, "_fetch_percept", _fake_fetch)

    capture, result, event = await svc.capture_and_assess(subtitle="hello")

    assert capture.ok is True
    assert result.ok is True
    assert result.raw_response == "sad, contemplative"
    assert event.ok is True
    assert fetched == ["a" * 64, "b" * 64]
    assert bus.calls == [settings.CHANNEL_RETINA_CLIP_INTAKE, settings.CHANNEL_AFFECTGPT_INTAKE]
    assert len(bus.published) == 1
    assert bus.published[0][0] == "orion:affectgpt:assessment"
    # video_path/audio_path the worker was actually called with must live
    # under AFFECTGPT_SCRATCH_DIR (the shared volume), not the default /tmp.
    assert event.input_ref["video_path"].startswith(str(scratch_dir))
    assert event.input_ref["audio_path"].startswith(str(scratch_dir))


@pytest.mark.asyncio
async def test_capture_failure_never_calls_the_worker(monkeypatch, scratch_dir):
    svc, bus = _svc_with_fake_bus()
    bus.set_reply(
        settings.CHANNEL_RETINA_CLIP_INTAKE,
        RetinaClipCaptureResultPayload(
            ok=False, error="a capture is already in progress", error_code="busy"
        ).model_dump(),
    )

    def _fetch_should_not_run(self, sha256, dest_path):
        raise AssertionError("_fetch_percept must not run after a failed capture")

    monkeypatch.setattr(JuniperAffectiveStateService, "_fetch_percept", _fetch_should_not_run)

    capture, result, event = await svc.capture_and_assess()

    assert capture.ok is False
    assert result.ok is False
    assert result.error_code == "busy"
    assert "capture failed" in result.error
    assert event.ok is False
    # Only retina's channel was ever called -- the worker channel never was.
    assert bus.calls == [settings.CHANNEL_RETINA_CLIP_INTAKE]
    assert len(bus.published) == 1


@pytest.mark.asyncio
async def test_percept_fetch_failure_never_calls_the_worker(monkeypatch, scratch_dir):
    svc, bus = _svc_with_fake_bus()
    bus.set_reply(
        settings.CHANNEL_RETINA_CLIP_INTAKE,
        RetinaClipCaptureResultPayload(
            ok=True, video_sha256="a" * 64, audio_sha256="b" * 64, duration_sec=8.0
        ).model_dump(),
    )

    def _fetch_raises(self, sha256, dest_path):
        raise PerceptFetchError(f"hash mismatch for {sha256[:12]}")

    monkeypatch.setattr(JuniperAffectiveStateService, "_fetch_percept", _fetch_raises)

    capture, result, event = await svc.capture_and_assess()

    assert capture.ok is True
    assert result.ok is False
    assert result.error_code == "fetch_failed"
    assert event.ok is False
    assert bus.calls == [settings.CHANNEL_RETINA_CLIP_INTAKE]


@pytest.mark.asyncio
async def test_scratch_dir_is_cleaned_up_after(monkeypatch, scratch_dir):
    svc, bus = _svc_with_fake_bus()
    bus.set_reply(
        settings.CHANNEL_RETINA_CLIP_INTAKE,
        RetinaClipCaptureResultPayload(
            ok=True, video_sha256="a" * 64, audio_sha256="b" * 64, duration_sec=8.0
        ).model_dump(),
    )
    bus.set_reply(
        settings.CHANNEL_AFFECTGPT_INTAKE,
        AffectGptAssessResultPayload(ok=True, raw_response="ok").model_dump(),
    )

    seen_dirs: list[str] = []

    def _fake_fetch(self, sha256, dest_path):
        seen_dirs.append(os.path.dirname(dest_path))
        with open(dest_path, "wb") as f:
            f.write(b"x")
        return dest_path

    monkeypatch.setattr(JuniperAffectiveStateService, "_fetch_percept", _fake_fetch)

    await svc.capture_and_assess()

    assert seen_dirs, "_fetch_percept was never called"
    # Nothing survives the with-block -- same "nothing survives past a temp
    # dir" discipline as retina's own clip_capture.py.
    assert not os.path.exists(seen_dirs[0]), "capture temp dir survived past capture_and_assess()"


@pytest.mark.asyncio
async def test_gather_waits_for_slow_sibling_fetch_before_cleanup(monkeypatch, scratch_dir):
    """The exact race return_exceptions=True exists to prevent (review
    finding, 2026-08-22): plain asyncio.gather() re-raises the instant ONE
    side fails while the other's asyncio.to_thread call is still a real,
    running OS thread -- the old code returned from inside the
    `with tempfile.TemporaryDirectory(...)` block right then, so __exit__'s
    rmtree could race a sibling thread still mid-write into that same
    directory. This uses a REAL time.sleep inside a REAL thread (not a mock)
    to prove capture_and_assess() doesn't return until the slow side has
    actually finished, not just that no exception escaped.
    """
    svc, bus = _svc_with_fake_bus()
    bus.set_reply(
        settings.CHANNEL_RETINA_CLIP_INTAKE,
        RetinaClipCaptureResultPayload(
            ok=True, video_sha256="a" * 64, audio_sha256="b" * 64, duration_sec=8.0
        ).model_dump(),
    )

    slow_finished = {"done": False}

    def _fetch(self, sha256, dest_path):
        if sha256 == "a" * 64:
            raise PerceptFetchError("fast failure")
        time.sleep(0.3)  # a real OS thread sleep, standing in for a slow write
        slow_finished["done"] = True
        with open(dest_path, "wb") as f:
            f.write(b"ok")
        return dest_path

    monkeypatch.setattr(JuniperAffectiveStateService, "_fetch_percept", _fetch)

    capture, result, event = await svc.capture_and_assess()

    assert slow_finished["done"], (
        "capture_and_assess() returned before the slow sibling fetch actually "
        "finished -- the exact race this fix exists to prevent"
    )
    assert result.ok is False
    assert result.error_code == "fetch_failed"


@pytest.mark.asyncio
async def test_fetch_percept_verifies_hash_and_rejects_mismatch(monkeypatch, tmp_path):
    """Chain-of-custody check: bytes returned by percept-store must
    actually hash to the sha256 that was asked for."""
    svc, _ = _svc_with_fake_bus()
    monkeypatch.setattr(settings, "PERCEPT_STORE_BASE_URL", "http://store/percepts")

    class _FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"not the bytes you asked for"

    monkeypatch.setattr("app.main.urllib.request.urlopen", lambda *a, **k: _FakeResp())

    with pytest.raises(PerceptFetchError, match="hash mismatch"):
        svc._fetch_percept("a" * 64, str(tmp_path / "out.bin"))


@pytest.mark.asyncio
async def test_fetch_percept_rejects_when_base_url_unset(monkeypatch, tmp_path):
    svc, _ = _svc_with_fake_bus()
    monkeypatch.setattr(settings, "PERCEPT_STORE_BASE_URL", "")

    with pytest.raises(PerceptFetchError, match="PERCEPT_STORE_BASE_URL"):
        svc._fetch_percept("a" * 64, str(tmp_path / "out.bin"))


@pytest.mark.asyncio
async def test_fetch_percept_sends_configured_token_header(monkeypatch, tmp_path):
    """Review finding, 2026-08-22: PERCEPT_STORE_TOKEN didn't exist on this
    service at all before -- enabling percept-store's own auth would have
    silently 401'd every fetch with no way to configure a credential here."""
    svc, _ = _svc_with_fake_bus()
    monkeypatch.setattr(settings, "PERCEPT_STORE_BASE_URL", "http://store/percepts")
    monkeypatch.setattr(settings, "PERCEPT_STORE_TOKEN", "s3cr3t")

    import hashlib as _hashlib

    body = b"real bytes"
    sha = _hashlib.sha256(body).hexdigest()
    captured: dict = {}

    class _FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return body

    def _fake_urlopen(req, timeout):
        captured["token"] = req.headers.get("X-orion-percept-token")
        return _FakeResp()

    monkeypatch.setattr("app.main.urllib.request.urlopen", _fake_urlopen)

    svc._fetch_percept(sha, str(tmp_path / "out.bin"))

    assert captured["token"] == "s3cr3t"


@pytest.mark.asyncio
async def test_fetch_percept_sends_no_token_header_when_unset(monkeypatch, tmp_path):
    svc, _ = _svc_with_fake_bus()
    monkeypatch.setattr(settings, "PERCEPT_STORE_BASE_URL", "http://store/percepts")
    monkeypatch.setattr(settings, "PERCEPT_STORE_TOKEN", "")

    import hashlib as _hashlib

    body = b"real bytes"
    sha = _hashlib.sha256(body).hexdigest()
    captured: dict = {}

    class _FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return body

    def _fake_urlopen(req, timeout):
        captured["token"] = req.headers.get("X-orion-percept-token")
        return _FakeResp()

    monkeypatch.setattr("app.main.urllib.request.urlopen", _fake_urlopen)

    svc._fetch_percept(sha, str(tmp_path / "out.bin"))

    assert captured["token"] is None
