"""Bus-free by design (mocked bus, mocked percept-store fetch) -- covers
capture_and_assess(): the live-capture bridge added 2026-08-22
(POST /v1/juniper/affect/capture_and_assess, called by Hub's "Affect check"
toggle). See test_trigger.py for the pre-existing manual-path coverage this
builds on; that file's FakeBus supports only one configured RPC reply,
insufficient here since capture_and_assess makes two DIFFERENT RPC calls
(retina, then the worker) -- this file's FakeBus keys replies by the
request channel instead.

**Every test in this file pins AFFECT_BACKEND="affectgpt"** (see the autouse
fixture below). These tests describe the AffectGPT round trip specifically --
two RPC legs, a worker reply, an audio blob fetched alongside the video -- and
none of that is what a capture does by default since the 2026-08-26 vision
cutover. Pinning is deliberate rather than rewriting them to the new path:
the affectgpt path still exists as a rollback and still needs coverage, and
the vision path gets its own file (test_vision_backend.py) instead of these
being contorted to cover both.

The pin is explicit and per-file, NOT a conftest default, so it can never
silently mask the default flipping back. test_vision_backend.py asserts the
unpinned default independently.
"""
from __future__ import annotations

import os
import time
from typing import Any

import pytest

from app.main import (
    CaptureAndAssessRequest,
    JuniperAffectiveStateService,
    PerceptFetchError,
    _normalize_trigger,
    settings,
)
from orion.schemas.affectgpt import AffectGptAssessResultPayload
from orion.schemas.vision import RetinaClipCaptureResultPayload
from pydantic import ValidationError


@pytest.fixture(autouse=True)
def _pin_affectgpt_backend(monkeypatch):
    """See module docstring -- this file covers the affectgpt rollback path."""
    monkeypatch.setattr(settings, "AFFECT_BACKEND", "affectgpt", raising=False)


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
        # One entry per rpc_request call, in order -- lets a test assert
        # the SAME correlation_id was used across both legs of one attempt
        # (retina RPC, then worker RPC).
        self.correlation_ids_used: list[str] = []
        # One entry per rpc_request call -- lets a test assert on exactly
        # what was sent (e.g. target_stream_id on the retina leg).
        self.request_payloads: list[dict] = []

    def set_reply(self, channel: str, payload: dict) -> None:
        self._replies[channel] = payload

    def request_payload_for(self, channel: str) -> dict | None:
        """The payload actually sent on `channel`. Reads the existing parallel
        calls/request_payloads lists rather than adding a second recording
        mechanism that could disagree with them."""
        for sent_channel, payload in zip(self.calls, self.request_payloads):
            if sent_channel == channel:
                return payload
        return None

    def set_raises(self, channel: str, exc: Exception) -> None:
        self._raises[channel] = exc

    async def rpc_request(self, request_channel, envelope, *, reply_channel, timeout_sec):
        self.calls.append(request_channel)
        self.correlation_ids_used.append(str(envelope.correlation_id))
        self.request_payloads.append(envelope.payload)
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
    # Default trigger, no explicit correlation_id passed in -- still gets
    # a real one (generated fresh each attempt), just not caller-supplied.
    assert event.trigger == "manual"
    assert event.correlation_id


@pytest.mark.asyncio
async def test_subtitle_source_and_transcript_thread_through_to_the_event(monkeypatch, scratch_dir):
    """The worker's own subtitle_source/transcript telemetry (2026-08-22,
    Whisper auto-transcription) must reach the published event, not just
    raw_response -- a consumer needs to tell "real transcribed speech" apart
    from "no subtitle at all" to interpret a hedging raw_response correctly."""
    svc, bus = _svc_with_fake_bus()
    bus.set_reply(
        settings.CHANNEL_RETINA_CLIP_INTAKE,
        RetinaClipCaptureResultPayload(
            ok=True, video_sha256="a" * 64, audio_sha256="b" * 64, duration_sec=8.0
        ).model_dump(),
    )
    bus.set_reply(
        settings.CHANNEL_AFFECTGPT_INTAKE,
        AffectGptAssessResultPayload(
            ok=True,
            raw_response="the speaker sounds anxious",
            subtitle_source="transcribed",
            transcript="I don't know what to do",
        ).model_dump(),
    )
    monkeypatch.setattr(
        JuniperAffectiveStateService,
        "_fetch_percept",
        lambda self, sha256, dest_path: (open(dest_path, "wb").write(b"x"), dest_path)[1],
    )

    _, result, event = await svc.capture_and_assess()

    assert result.subtitle_source == "transcribed"
    assert result.transcript == "I don't know what to do"
    assert event.subtitle_source == "transcribed"
    assert event.transcript == "I don't know what to do"


@pytest.mark.asyncio
async def test_ambient_trigger_shares_one_correlation_id_across_both_rpc_legs(
    monkeypatch, scratch_dir
):
    """Juniper's ask, 2026-08-22: 'ensure the data model has good ability to
    be correlative with other components in the mesh.' capture_and_assess()
    generates ONE id and threads it through the retina RPC, the worker RPC,
    and the published event -- not three independently-generated ids that
    happen to describe the same real-world attempt but can't be joined."""
    svc, bus = _svc_with_fake_bus()
    bus.set_reply(
        settings.CHANNEL_RETINA_CLIP_INTAKE,
        RetinaClipCaptureResultPayload(
            ok=True, video_sha256="a" * 64, audio_sha256="b" * 64, duration_sec=8.0
        ).model_dump(),
    )
    bus.set_reply(
        settings.CHANNEL_AFFECTGPT_INTAKE,
        AffectGptAssessResultPayload(ok=True, raw_response="calm").model_dump(),
    )

    def _fake_fetch(self, sha256, dest_path):
        with open(dest_path, "wb") as f:
            f.write(b"x")
        return dest_path

    monkeypatch.setattr(JuniperAffectiveStateService, "_fetch_percept", _fake_fetch)

    capture, result, event = await svc.capture_and_assess(trigger="ambient")

    assert event.trigger == "ambient"
    assert len(bus.correlation_ids_used) == 2
    assert bus.correlation_ids_used[0] == bus.correlation_ids_used[1], (
        "retina RPC and worker RPC used different correlation_ids for the "
        "same attempt -- they should be joinable via one id"
    )
    assert event.correlation_id == bus.correlation_ids_used[0]


@pytest.mark.asyncio
async def test_unrecognized_trigger_value_is_clamped_to_manual(monkeypatch, scratch_dir):
    """trigger is caller-supplied (via the HTTP body) -- an unrecognized
    string here must not reach JuniperMultimodalAffectV1's Literal field and
    raise an unhandled ValidationError deep inside _wrap_event."""
    svc, bus = _svc_with_fake_bus()
    bus.set_reply(
        settings.CHANNEL_RETINA_CLIP_INTAKE,
        RetinaClipCaptureResultPayload(
            ok=False, error="unreachable", error_code="timeout"
        ).model_dump(),
    )

    capture, result, event = await svc.capture_and_assess(trigger="not-a-real-value")

    assert event.trigger == "manual"


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


# --- _normalize_trigger / CaptureAndAssessRequest (review finding, 2026-08-22) --
# The trigger clamp used to be duplicated verbatim in two places
# (capture_and_assess() and _wrap_event()) with no shared function backing
# either; the HTTP endpoint accepted an untyped dict with no real
# validation. Both fixed via one helper + one pydantic request model.


def test_normalize_trigger_clamps_anything_it_does_not_recognize():
    # Renamed 2026-08-25: the set of accepted labels is no longer just
    # {"manual", "ambient"} (chat_turn_pre/chat_turn_post were added for
    # Hub's per-turn affect bracket), so the old name asserted something
    # that is no longer true. Every assertion below still holds and is
    # still the point: unrecognized input clamps, and never raises --
    # including unhashable input, which is what caught the set-membership
    # regression introduced alongside those new labels.
    assert _normalize_trigger("ambient") == "ambient"
    assert _normalize_trigger("manual") == "manual"
    assert _normalize_trigger("not-a-real-value") == "manual"
    assert _normalize_trigger(None) == "manual"
    assert _normalize_trigger(123) == "manual"
    assert _normalize_trigger(["ambient"]) == "manual"


def test_capture_and_assess_request_rejects_unrecognized_trigger():
    with pytest.raises(ValidationError):
        CaptureAndAssessRequest(trigger="not-a-real-value")


def test_capture_and_assess_request_defaults_to_manual():
    req = CaptureAndAssessRequest()
    assert req.trigger == "manual"
    assert req.subtitle == ""
    assert req.user_message is None


def test_capture_and_assess_request_accepts_ambient():
    req = CaptureAndAssessRequest(trigger="ambient", subtitle="hi")
    assert req.trigger == "ambient"
    assert req.subtitle == "hi"


@pytest.mark.asyncio
async def test_capture_clip_via_retina_sends_the_configured_target_stream_id(
    monkeypatch, scratch_dir
):
    """Juniper's explicit instruction, 2026-08-22: 'I want this to only run
    on my carbon webcam.' The shared bus channel has no built-in per-
    instance routing, so this field is the only thing stopping any other
    retina instance from responding -- it must actually be sent, and it
    must come from AFFECT_TARGET_STREAM_ID, not a hardcoded literal."""
    svc, bus = _svc_with_fake_bus()
    bus.set_reply(
        settings.CHANNEL_RETINA_CLIP_INTAKE,
        RetinaClipCaptureResultPayload(ok=False, error="unreachable").model_dump(),
    )

    await svc.capture_and_assess()

    assert bus.request_payloads[0]["target_stream_id"] == settings.AFFECT_TARGET_STREAM_ID


@pytest.mark.asyncio
async def test_capture_clip_via_retina_respects_a_reconfigured_target_stream_id(
    monkeypatch, scratch_dir
):
    monkeypatch.setattr(settings, "AFFECT_TARGET_STREAM_ID", "some-other-camera")
    svc, bus = _svc_with_fake_bus()
    bus.set_reply(
        settings.CHANNEL_RETINA_CLIP_INTAKE,
        RetinaClipCaptureResultPayload(ok=False, error="unreachable").model_dump(),
    )

    await svc.capture_and_assess()

    assert bus.request_payloads[0]["target_stream_id"] == "some-other-camera"


# ==========================================================================
# The VISION backend through capture_and_assess (2026-08-26).
#
# The autouse fixture above pins affectgpt for every test in this file; these
# override it explicitly. Review finding 7.1: nothing covered capture_and_assess
# on the vision path at all -- most importantly, deleting the conditional that
# skips the audio fetch broke no test, even though "Juniper's voice never
# crosses a host boundary" is the headline privacy claim of the whole change.
# ==========================================================================


class _StubVisionResult:
    def __init__(self):
        from orion.schemas.affectgpt import AffectReadV1

        self.affect = AffectReadV1(
            valence=-0.1,
            arousal=0.2,
            primary_affect="neutral and contemplative",
            cues=["gaze directed downwards in all frames"],
            confidence=0.85,
            cannot_tell=[],
        )
        self.raw_response = '{"primary_affect": "neutral and contemplative"}'
        self.face_detection = {
            "frames_total": 231,
            "frames_detected": 231,
            "detection_rate": 1.0,
            "frames_sampled": 5,
        }
        self.frames_used = 5
        self.timings = {"total_s": 7.1}
        self.model = "/models/gguf/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf"


@pytest.fixture
def vision_backend(monkeypatch):
    monkeypatch.setattr(settings, "AFFECT_BACKEND", "vision", raising=False)


@pytest.mark.asyncio
async def test_vision_backend_never_fetches_the_audio_blob(
    monkeypatch, scratch_dir, vision_backend
):
    """THE privacy regression test. The vision path must fetch video only --
    the audio blob stays in percept-store and ages out on its own retention."""
    svc, bus = _svc_with_fake_bus()
    bus.set_reply(
        settings.CHANNEL_RETINA_CLIP_INTAKE,
        RetinaClipCaptureResultPayload(
            ok=True, video_sha256="a" * 64, audio_sha256="b" * 64, duration_sec=8.0
        ).model_dump(),
    )
    fetched: list[str] = []

    def _fake_fetch(self, sha256, dest_path):
        fetched.append(sha256)
        with open(dest_path, "wb") as f:
            f.write(b"clip")
        return dest_path

    monkeypatch.setattr(JuniperAffectiveStateService, "_fetch_percept", _fake_fetch)

    async def _fake_assess(bus_, *, video_path, transcript, settings):
        return _StubVisionResult()

    monkeypatch.setattr("app.main.assess_via_vision", _fake_assess)

    capture, result, event = await svc.capture_and_assess()

    assert result.ok is True
    assert fetched == ["a" * 64], f"audio blob was fetched: {fetched}"


@pytest.mark.asyncio
async def test_vision_backend_records_no_audio_path_in_the_event(
    monkeypatch, scratch_dir, vision_backend
):
    """A path for a file that was never downloaded would put a claim in the
    durable record contradicting the property above."""
    svc, bus = _svc_with_fake_bus()
    bus.set_reply(
        settings.CHANNEL_RETINA_CLIP_INTAKE,
        RetinaClipCaptureResultPayload(
            ok=True, video_sha256="a" * 64, audio_sha256="b" * 64, duration_sec=8.0
        ).model_dump(),
    )
    monkeypatch.setattr(
        JuniperAffectiveStateService,
        "_fetch_percept",
        lambda self, sha, dest: (open(dest, "wb").write(b"clip"), dest)[1],
    )

    async def _fake_assess(bus_, *, video_path, transcript, settings):
        return _StubVisionResult()

    monkeypatch.setattr("app.main.assess_via_vision", _fake_assess)

    _capture, _result, event = await svc.capture_and_assess()

    assert "audio_path" not in (event.input_ref or {})
    assert (event.input_ref or {}).get("video_path")


@pytest.mark.asyncio
async def test_vision_backend_sets_backend_and_structured_affect(
    monkeypatch, scratch_dir, vision_backend
):
    svc, bus = _svc_with_fake_bus()
    bus.set_reply(
        settings.CHANNEL_RETINA_CLIP_INTAKE,
        RetinaClipCaptureResultPayload(
            ok=True, video_sha256="a" * 64, audio_sha256="b" * 64, duration_sec=8.0
        ).model_dump(),
    )
    monkeypatch.setattr(
        JuniperAffectiveStateService,
        "_fetch_percept",
        lambda self, sha, dest: (open(dest, "wb").write(b"clip"), dest)[1],
    )

    async def _fake_assess(bus_, *, video_path, transcript, settings):
        return _StubVisionResult()

    monkeypatch.setattr("app.main.assess_via_vision", _fake_assess)

    _capture, _result, event = await svc.capture_and_assess()

    assert event.backend == "vision"
    assert event.source == "vision"
    assert event.affect is not None
    assert event.affect.primary_affect == "neutral and contemplative"
    assert event.frames_used == 5
    # No Whisper on this path, so this can never be "transcribed".
    assert event.subtitle_source == "none"


@pytest.mark.asyncio
async def test_vision_read_failure_is_ok_false_not_a_calm_read(
    monkeypatch, scratch_dir, vision_backend
):
    """A failed read must never be indistinguishable from a genuine calm one."""
    from app.vision_backend import VisionAffectError

    svc, bus = _svc_with_fake_bus()
    bus.set_reply(
        settings.CHANNEL_RETINA_CLIP_INTAKE,
        RetinaClipCaptureResultPayload(
            ok=True, video_sha256="a" * 64, audio_sha256="b" * 64, duration_sec=8.0
        ).model_dump(),
    )
    monkeypatch.setattr(
        JuniperAffectiveStateService,
        "_fetch_percept",
        lambda self, sha, dest: (open(dest, "wb").write(b"clip"), dest)[1],
    )

    async def _boom(bus_, *, video_path, transcript, settings):
        raise VisionAffectError("gateway said no", error_code="empty_completion")

    monkeypatch.setattr("app.main.assess_via_vision", _boom)

    _capture, result, event = await svc.capture_and_assess()

    assert result.ok is False
    assert result.error_code == "empty_completion"
    assert event.affect is None
    assert event.backend == "vision"


@pytest.mark.asyncio
async def test_capture_failure_is_attributed_to_the_selected_backend(
    monkeypatch, scratch_dir, vision_backend
):
    """Review finding 2.1: both early-return failure branches used to fall
    through to _wrap_event's backend="affectgpt" default, so a retina outage
    under AFFECT_BACKEND=vision persisted a row blaming a backend that was
    never invoked -- defeating the reason the column exists."""
    svc, bus = _svc_with_fake_bus()
    bus.set_reply(
        settings.CHANNEL_RETINA_CLIP_INTAKE,
        RetinaClipCaptureResultPayload(
            ok=False, error="camera busy", error_code="busy"
        ).model_dump(),
    )

    _capture, result, event = await svc.capture_and_assess()

    assert result.ok is False
    assert event.backend == "vision"
    assert event.source == "vision"


@pytest.mark.asyncio
async def test_vision_backend_asks_retina_not_to_arm_the_microphone(
    monkeypatch, scratch_dir, vision_backend
):
    """Juniper's report, 2026-08-26: the mic button produced two divorced audio
    recordings, and feeding the second one anything real would have meant
    repeating herself into a much quieter mic. The vision path must therefore
    not arm the mic at all -- not "arm it and discard the result"."""
    svc, bus = _svc_with_fake_bus()
    bus.set_reply(
        settings.CHANNEL_RETINA_CLIP_INTAKE,
        RetinaClipCaptureResultPayload(
            ok=True, video_sha256="a" * 64, audio_sha256=None, duration_sec=8.0
        ).model_dump(),
    )
    monkeypatch.setattr(
        JuniperAffectiveStateService,
        "_fetch_percept",
        lambda self, sha, dest: (open(dest, "wb").write(b"clip"), dest)[1],
    )

    async def _fake_assess(bus_, *, video_path, transcript, settings):
        return _StubVisionResult()

    monkeypatch.setattr("app.main.assess_via_vision", _fake_assess)

    await svc.capture_and_assess()

    sent = bus.request_payload_for(settings.CHANNEL_RETINA_CLIP_INTAKE)
    assert sent is not None, "no retina clip request was published"
    assert sent.get("want_audio") is False


@pytest.mark.asyncio
async def test_affectgpt_backend_still_asks_for_audio(monkeypatch, scratch_dir):
    """The rollback path Whispers the clip's own wav, so it genuinely needs
    one. Asserted explicitly so the flag can never be flipped globally by
    accident."""
    monkeypatch.setattr(settings, "AFFECT_BACKEND", "affectgpt", raising=False)
    svc, bus = _svc_with_fake_bus()
    bus.set_reply(
        settings.CHANNEL_RETINA_CLIP_INTAKE,
        RetinaClipCaptureResultPayload(
            ok=True, video_sha256="a" * 64, audio_sha256="b" * 64, duration_sec=8.0
        ).model_dump(),
    )
    bus.set_reply(
        settings.CHANNEL_AFFECTGPT_INTAKE,
        AffectGptAssessResultPayload(ok=True, raw_response="calm").model_dump(),
    )
    monkeypatch.setattr(
        JuniperAffectiveStateService,
        "_fetch_percept",
        lambda self, sha, dest: (open(dest, "wb").write(b"clip"), dest)[1],
    )

    await svc.capture_and_assess()

    sent = bus.request_payload_for(settings.CHANNEL_RETINA_CLIP_INTAKE)
    assert sent.get("want_audio") is True


@pytest.mark.asyncio
async def test_real_transcript_reaches_the_vision_read(
    monkeypatch, scratch_dir, vision_backend
):
    """The other half of the merge: the affect read is grounded in the words
    Juniper ALREADY said into the browser mic, so no second recording of her
    voice needs to exist for the read to have any speech context at all."""
    svc, bus = _svc_with_fake_bus()
    bus.set_reply(
        settings.CHANNEL_RETINA_CLIP_INTAKE,
        RetinaClipCaptureResultPayload(
            ok=True, video_sha256="a" * 64, audio_sha256=None, duration_sec=8.0
        ).model_dump(),
    )
    monkeypatch.setattr(
        JuniperAffectiveStateService,
        "_fetch_percept",
        lambda self, sha, dest: (open(dest, "wb").write(b"clip"), dest)[1],
    )
    seen = {}

    async def _capture_transcript(bus_, *, video_path, transcript, settings):
        seen["transcript"] = transcript
        return _StubVisionResult()

    monkeypatch.setattr("app.main.assess_via_vision", _capture_transcript)

    await svc.capture_and_assess(subtitle="I'm feeling really tired.")

    assert seen["transcript"] == "I'm feeling really tired."
