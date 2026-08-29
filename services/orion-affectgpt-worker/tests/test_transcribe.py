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


# ==========================================================================
# no_speech_prob filter (2026-08-26)
#
# The amplitude gate at peak=50 PASSED a real clip measured at peak=114 /
# rms=8.68 and Whisper returned a fully-formed sentence about Egyptians on a
# turn where Juniper had actually said "I'm feeling really tired." A
# downstream model then read her affect off the invented sentence.
#
# Amplitude cannot catch that -- 0.15% of full scale is still "loud enough"
# numerically, and a hallucination is perfectly well-formed output, so there
# is nothing about the STRING to match on either. The model's own
# per-segment confidence is the only signal that separates them.
# ==========================================================================

from app.transcribe import keep_only_speech_segments


def _seg(text, prob):
    return {"text": text, "no_speech_prob": prob}


def test_confident_silence_segments_are_dropped_entirely():
    """The exact shape of the live failure: one confident-silence segment
    carrying a fluent, entirely fabricated sentence."""
    result = {
        "text": " Thanks for the light. Thanks for the eyesight.",
        "segments": [_seg(" Thanks for the light. Thanks for the eyesight.", 0.94)],
    }
    text, meta = keep_only_speech_segments(result, 0.6)
    assert text == ""
    assert meta["segments_total"] == 1
    assert meta["segments_kept"] == 0
    assert meta["max_no_speech_prob_seen"] == 0.94


def test_real_speech_is_kept():
    result = {
        "text": " I'm feeling really tired.",
        "segments": [_seg(" I'm feeling really tired.", 0.02)],
    }
    text, meta = keep_only_speech_segments(result, 0.6)
    assert text == "I'm feeling really tired."
    assert meta["segments_kept"] == 1


def test_mixed_clip_keeps_only_the_speech():
    result = {
        "text": " Hello there. Thanks for watching.",
        "segments": [_seg(" Hello there.", 0.05), _seg(" Thanks for watching.", 0.91)],
    }
    text, _ = keep_only_speech_segments(result, 0.6)
    assert text == "Hello there."


def test_threshold_is_honoured_not_hardcoded():
    result = {"text": " maybe", "segments": [_seg(" maybe", 0.5)]}
    assert keep_only_speech_segments(result, 0.6)[0] == "maybe"
    assert keep_only_speech_segments(result, 0.4)[0] == ""


def test_missing_segments_falls_back_to_raw_text():
    """Absent evidence is not evidence of silence -- a model returning no
    segment structure must not have its output silently discarded."""
    text, meta = keep_only_speech_segments({"text": " hello"}, 0.6)
    assert text == "hello"
    assert meta["no_speech_filter"] == "unavailable"


def test_malformed_segment_entries_do_not_crash():
    result = {"text": " hi", "segments": [None, "junk", _seg(" hi", 0.1)]}
    assert keep_only_speech_segments(result, 0.6)[0] == "hi"


def test_non_numeric_no_speech_prob_is_treated_as_speech():
    """A garbage prob must not silently discard real speech -- fail toward
    keeping what the model transcribed, since the amplitude gate already ran."""
    result = {"text": " hi", "segments": [{"text": " hi", "no_speech_prob": "bad"}]}
    assert keep_only_speech_segments(result, 0.6)[0] == "hi"


def test_segments_present_but_all_unparseable_keeps_the_raw_text():
    """Regression: a non-empty `segments` list whose entries are not dicts (a
    Whisper variant returning objects, a serialization change) used to fall
    through the loop, collect no probs, and return "" -- silently discarding a
    real transcript while `meta` claimed the filter had run. The gate would be
    the last thing anyone suspected, because its own telemetry said it worked.
    """
    result = {
        "text": "I'm feeling really tired.",
        "segments": ["not-a-dict", 42, None],
    }
    text, meta = keep_only_speech_segments(result, 0.6)
    assert text == "I'm feeling really tired."
    assert meta["no_speech_filter"] == "unavailable"
    assert meta["reason"] == "no_parseable_segments"
