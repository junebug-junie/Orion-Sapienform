"""Optional Whisper transcription -- populates ``subtitle`` from
``audio_path`` when a caller doesn't supply real text.

Juniper's own ask, 2026-08-22 ("HIT IT"), after this service's README's own
documented finding (from earlier live testing that day) that an empty
``subtitle`` produces materially worse-grounded model output than real
transcript text -- and after Hub's ambient/manual capture paths were
confirmed to NEVER send one (they always default to ``""``).

Deliberately NOT a new cross-service dependency on ``orion-whisper-tts``
(the repo's only other Whisper user): that service is bus-only, its own
producer set is declared as ``orion-hub`` alone in
``orion/bus/channels.yaml``, and its actual deployment host/GPU pinning is
unconfirmed -- adding this worker as a second, undeclared producer would
couple a synchronous assessment path to another service's uptime/GPU
contention with no established guarantee it even shares this GPU.
``orion-affectgpt-worker`` already carries a full CUDA/torch stack on circe
GPU2 with ~11-14GB of confirmed headroom under AffectGPT's own ~18.4GB peak
(see README) -- loading Whisper "base" (~1GB VRAM) into the SAME
already-warm, single-request-locked process is the smaller, more contained
seam.

The silence-gate technique (measure int16 peak, skip Whisper below a
threshold) is deliberately copied from ``services/orion-whisper-tts/app/
stt.py``'s ``STTEngine`` -- that service already found, in production, that
Whisper hallucinates text from near-silent audio if you don't gate it. No
format-canonicalization step is needed here the way that service needs one:
retina's ``clip_capture.py`` always writes 16kHz mono 16-bit PCM
(``-ac 1 -ar 16000``, ffmpeg's default wav codec is ``pcm_s16le``), so
``audio_path`` is already in exactly the shape ``wave.open`` and
``whisper.transcribe`` both expect -- no browser-upload format variability
to normalize.
"""
from __future__ import annotations

import struct
import wave
from dataclasses import dataclass
from typing import Any, Optional

from loguru import logger

# NOT stdlib logging -- confirmed live 2026-08-22: main.py's start() calls
# `loguru.logger.remove()` then adds its own single sink, but never touches
# Python's stdlib logging module at all. A stdlib logging.getLogger(__name__)
# call here is silently invisible (Python's lastResort handler only prints
# WARNING+, and even then in a different, unformatted style than the rest of
# this service's timestamped loguru output) -- caught by an actual live
# request (subtitle_source came back "transcribed" in the response, proving
# this code ran) that produced ZERO matching log lines. loguru is a global
# singleton shared with app/main.py's own `from loguru import logger`, so
# this needs no separate configuration to show up in the same sink.


@dataclass
class TranscribeResult:
    text: str
    meta: dict[str, Any]


def measure_wav_peak(path: str) -> tuple[float, int, Optional[str]]:
    """(rms, peak, error) of a 16-bit PCM wav's samples. ``error`` is None on
    a real, successful measurement; a string on anything unreadable (missing
    file, corrupt wav, wrong sample width). Review finding, 2026-08-22: a
    genuinely silent clip and an unreadable one used to both collapse to the
    identical (0.0, 0) with nothing to tell them apart -- both then produced
    the same "near_silent_skip" log line and subtitle_source="none", hiding
    a real upstream bug (e.g. a truncated percept-store fetch) behind what
    looked like routine silence-gating. Callers now get a real signal to
    decide what "unreadable" should mean instead of it being silently
    conflated with "measured, and it was quiet."""
    try:
        peak = 0
        sum_sq = 0.0
        count = 0
        with wave.open(path, "rb") as wf:
            if wf.getsampwidth() != 2:
                return 0.0, 0, f"unsupported sample width {wf.getsampwidth()} (expected 2)"
            while True:
                frames = wf.readframes(8192)
                if not frames:
                    break
                for i in range(0, len(frames), 2):
                    sample = struct.unpack_from("<h", frames, i)[0]
                    abs_sample = abs(sample)
                    if abs_sample > peak:
                        peak = abs_sample
                    sum_sq += sample * sample
                    count += 1
        if count == 0:
            return 0.0, 0, "wav contained no samples"
        return (sum_sq / count) ** 0.5, peak, None
    except Exception as exc:  # noqa: BLE001 -- a bad/missing wav is a caller error elsewhere, not this gate's job
        logger.warning(f"[TRANSCRIBE] wav_peak_measure_failed error={exc}")
        return 0.0, 0, str(exc)


def load_whisper_model(model_name: str, device: str):
    """Imported lazily (inside the function, not at module top-level) so
    importing this module never requires ``openai-whisper`` to be
    installed/loadable unless transcription is actually enabled -- matches
    ``model_runtime.py``'s own pattern of deferring the vendored AffectGPT
    imports until ``load()`` actually runs."""
    import whisper

    model = whisper.load_model(model_name, device=device)
    logger.info(f"[TRANSCRIBE] whisper_model_loaded model={model_name} device={device}")
    return model


def transcribe_audio(
    model,
    audio_path: str,
    *,
    peak_threshold: int,
    language: str = "en",
) -> TranscribeResult:
    """Never raises -- any failure (missing file, corrupt wav, Whisper
    internal error) returns empty text with the error recorded in meta,
    matching every other boundary in this service ("never crash on a bad
    clip"). Caller (model_runtime.assess) falls back to subtitle="" on an
    empty result, identical to today's caller-supplies-nothing behavior."""
    try:
        rms, peak, measure_error = measure_wav_peak(audio_path)
        meta: dict[str, Any] = {
            "peak": peak,
            "rms": round(rms, 2),
            "peak_threshold": peak_threshold,
        }
        if measure_error is not None:
            # Can't tell "silent" from "unreadable" without a real
            # measurement -- run Whisper anyway rather than silently
            # rejecting as near-silent (review finding, 2026-08-22).
            # Whisper's own decode will raise on a truly corrupt file,
            # which the outer except below still catches.
            meta["peak_measure_error"] = measure_error
            logger.warning(
                f"[TRANSCRIBE] peak_measure_failed_running_anyway error={measure_error}"
            )
        elif peak < peak_threshold:
            meta["silence_gate"] = "rejected"
            logger.info(
                f"[TRANSCRIBE] near_silent_skip peak={peak} threshold={peak_threshold}"
            )
            return TranscribeResult(text="", meta=meta)
        else:
            meta["silence_gate"] = "passed"

        result = model.transcribe(
            audio_path,
            language=language,
            fp16=False,
            condition_on_previous_text=False,
            no_speech_threshold=0.35,
        )
        text = (result.get("text") or "").strip()
        meta["text_len"] = len(text)
        logger.info(f"[TRANSCRIBE] transcribed len={len(text)}")
        return TranscribeResult(text=text, meta=meta)
    except Exception as exc:  # noqa: BLE001 -- transcription is advisory; never break the assessment
        logger.warning(f"[TRANSCRIBE] transcribe_failed error={exc}")
        return TranscribeResult(text="", meta={"error": str(exc)})


def resolve_subtitle(
    subtitle: str,
    *,
    whisper_model,
    audio_path: str,
    peak_threshold: int,
    language: str = "en",
) -> tuple[str, str, Optional[str], Optional[dict[str, Any]]]:
    """The full "what subtitle does the model actually see" decision,
    pulled out of model_runtime.assess() into a pure function that needs
    no GPU/vendored-AffectGPT state -- only whisper_model (or None) and a
    wav path -- so it's directly unit-testable (review finding, 2026-08-22:
    the "caller subtitle always wins, Whisper never overwrites it" guarantee
    had no test proving it; a future `and`/`or` typo in the condition could
    have silently started overwriting real caller text with Whisper output).

    Returns (effective_subtitle, subtitle_source, transcript, transcribe_meta).

    ``subtitle`` is stripped before the truthiness check -- review finding,
    2026-08-22: a whitespace-only subtitle (e.g. a stray " ") used to be
    treated as real caller text, skipping Whisper entirely and silently
    reproducing the exact degraded-mode failure this feature exists to fix,
    while falsely reporting subtitle_source="caller".
    """
    stripped = (subtitle or "").strip()
    if stripped:
        return stripped, "caller", None, None
    if whisper_model is None:
        return "", "none", None, None
    stt_result = transcribe_audio(
        whisper_model, audio_path, peak_threshold=peak_threshold, language=language
    )
    if stt_result.text:
        return stt_result.text, "transcribed", stt_result.text, stt_result.meta
    return "", "none", None, stt_result.meta
