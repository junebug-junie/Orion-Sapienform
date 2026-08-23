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

from app.transcribe import measure_wav_peak, transcribe_audio


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
    rms, peak = measure_wav_peak(loud_wav)
    assert peak == 10000
    assert rms > 0

    rms, peak = measure_wav_peak(silent_wav)
    assert peak == 0
    assert rms == 0.0


def test_measure_wav_peak_returns_zero_for_unreadable_path():
    rms, peak = measure_wav_peak("/nonexistent/path/does-not-exist.wav")
    assert (rms, peak) == (0.0, 0)


def test_transcribe_audio_skips_whisper_on_near_silent_clip(silent_wav):
    model = _FakeWhisperModel()
    result = transcribe_audio(model, silent_wav, peak_threshold=50)

    assert result.text == ""
    assert result.meta["silence_gate"] == "rejected"
    assert model.called_with == []  # Whisper must never actually run


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
