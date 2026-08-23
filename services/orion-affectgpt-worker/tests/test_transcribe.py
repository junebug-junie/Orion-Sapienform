"""Model-free: exercises the silence gate and result plumbing against real
synthetic .wav files and a fake Whisper model object, mirroring
test_face_extract.py's convention of building real media with stdlib/cv2
rather than mocking file I/O. The actual `whisper` package is never
imported/loaded here -- `load_whisper_model` (the only thing that touches
it) is not called by these tests.
"""
from __future__ import annotations

import struct
import wave

import pytest

from app.transcribe import measure_wav_peak, resolve_subtitle, transcribe_audio


def _write_wav(path: str, samples: list[int], *, sample_rate: int = 16000) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(struct.pack("<h", s) for s in samples))


@pytest.fixture
def silent_wav(tmp_path):
    path = str(tmp_path / "silent.wav")
    _write_wav(path, [0] * 1600)  # 0.1s of true silence
    return path


@pytest.fixture
def loud_wav(tmp_path):
    path = str(tmp_path / "loud.wav")
    # A real, well-above-threshold square wave -- not actual speech, but
    # loud enough that the silence gate must pass it through to Whisper.
    _write_wav(path, [10000, -10000] * 800)
    return path


class _FakeWhisperModel:
    """Records whether .transcribe() was ever called -- the silence gate's
    whole job is deciding whether this happens at all."""

    def __init__(self, text: str = "hello there"):
        self.text = text
        self.called_with: list[str] = []

    def transcribe(self, path, **kwargs):
        self.called_with.append(path)
        return {"text": self.text}


def test_measure_wav_peak_reads_real_samples(loud_wav, silent_wav):
    rms, peak, error = measure_wav_peak(loud_wav)
    assert peak == 10000
    assert rms > 0
    assert error is None

    rms, peak, error = measure_wav_peak(silent_wav)
    assert peak == 0
    assert rms == 0.0
    assert error is None


def test_measure_wav_peak_distinguishes_unreadable_from_real_silence():
    """Review finding, 2026-08-22: an unreadable file used to return the
    exact same (0.0, 0) as genuine silence, with nothing to tell a corrupt
    upstream fetch apart from a quiet room."""
    rms, peak, error = measure_wav_peak("/nonexistent/path/does-not-exist.wav")
    assert (rms, peak) == (0.0, 0)
    assert error is not None


def test_transcribe_audio_skips_whisper_on_near_silent_clip(silent_wav):
    model = _FakeWhisperModel()
    result = transcribe_audio(model, silent_wav, peak_threshold=50)

    assert result.text == ""
    assert result.meta["silence_gate"] == "rejected"
    assert model.called_with == []  # Whisper must never actually run


def test_transcribe_audio_runs_whisper_anyway_when_peak_cannot_be_measured():
    """Review finding, 2026-08-22: an unreadable wav must NOT be silently
    treated as near-silent -- run Whisper anyway (its own decode will raise
    if the file is truly unusable, caught separately) rather than dropping
    a real upstream failure into the same bucket as routine silence-gating."""
    model = _FakeWhisperModel(text="ran anyway")
    result = transcribe_audio(model, "/nonexistent/path.wav", peak_threshold=50)

    assert result.text == "ran anyway"
    assert model.called_with == ["/nonexistent/path.wav"]
    assert "peak_measure_error" in result.meta
    assert "silence_gate" not in result.meta


def test_transcribe_audio_runs_whisper_on_a_real_signal(loud_wav):
    model = _FakeWhisperModel(text="I am feeling fine")
    result = transcribe_audio(model, loud_wav, peak_threshold=50)

    assert result.text == "I am feeling fine"
    assert result.meta["silence_gate"] == "passed"
    assert model.called_with == [loud_wav]


def test_transcribe_audio_strips_whitespace_only_output(loud_wav):
    model = _FakeWhisperModel(text="   \n  ")
    result = transcribe_audio(model, loud_wav, peak_threshold=50)
    assert result.text == ""


def test_transcribe_audio_never_raises_on_model_error(loud_wav):
    class _BrokenModel:
        def transcribe(self, path, **kwargs):
            raise RuntimeError("cuda out of memory")

    result = transcribe_audio(_BrokenModel(), loud_wav, peak_threshold=50)
    assert result.text == ""
    assert "cuda out of memory" in result.meta["error"]


# --- resolve_subtitle: the actual "what does the model see" decision -------
# Review finding, 2026-08-22: model_runtime.assess()'s "caller subtitle
# always wins, Whisper never overwrites it" comment had no test proving it.


def test_resolve_subtitle_caller_text_always_wins_whisper_never_called(loud_wav):
    class _ExplodingModel:
        def transcribe(self, path, **kwargs):
            raise AssertionError("Whisper must never run when the caller sent real subtitle text")

    subtitle, source, transcript, meta = resolve_subtitle(
        "the caller's real subtitle",
        whisper_model=_ExplodingModel(),
        audio_path=loud_wav,
        peak_threshold=50,
    )
    assert subtitle == "the caller's real subtitle"
    assert source == "caller"
    assert transcript is None
    assert meta is None


def test_resolve_subtitle_whitespace_only_is_treated_as_empty(loud_wav):
    """Review finding, 2026-08-22: a whitespace-only subtitle (e.g. a stray
    " ") used to be treated as real caller text via plain truthiness,
    skipping Whisper and silently reproducing the exact degraded-mode
    failure this feature exists to fix while falsely reporting
    subtitle_source="caller"."""
    model = _FakeWhisperModel(text="real transcript")
    subtitle, source, transcript, meta = resolve_subtitle(
        "   \n  ",
        whisper_model=model,
        audio_path=loud_wav,
        peak_threshold=50,
    )
    assert subtitle == "real transcript"
    assert source == "transcribed"
    assert model.called_with == [loud_wav]


def test_resolve_subtitle_no_whisper_model_falls_back_to_none(loud_wav):
    subtitle, source, transcript, meta = resolve_subtitle(
        "", whisper_model=None, audio_path=loud_wav, peak_threshold=50
    )
    assert subtitle == ""
    assert source == "none"
    assert transcript is None
    assert meta is None


def test_resolve_subtitle_transcribed_path_reports_meta(loud_wav):
    model = _FakeWhisperModel(text="a real transcript")
    subtitle, source, transcript, meta = resolve_subtitle(
        "", whisper_model=model, audio_path=loud_wav, peak_threshold=50
    )
    assert subtitle == "a real transcript"
    assert source == "transcribed"
    assert transcript == "a real transcript"
    assert meta["silence_gate"] == "passed"


def test_resolve_subtitle_near_silent_with_whisper_available_reports_none(silent_wav):
    model = _FakeWhisperModel(text="should not be used")
    subtitle, source, transcript, meta = resolve_subtitle(
        "", whisper_model=model, audio_path=silent_wav, peak_threshold=50
    )
    assert subtitle == ""
    assert source == "none"
    assert transcript is None
    assert meta["silence_gate"] == "rejected"
