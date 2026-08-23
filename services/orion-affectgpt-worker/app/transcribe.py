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

import logging
import struct
import wave
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class TranscribeResult:
    text: str
    meta: dict[str, Any]


def measure_wav_peak(path: str) -> tuple[float, int]:
    """(rms, peak) of a 16-bit PCM wav's samples. Returns (0.0, 0) on any
    format this can't read (e.g. not 16-bit) rather than raising -- this is
    a gate, not a hard requirement; a measurement failure should fall through
    to "try Whisper anyway", not block transcription outright."""
    try:
        peak = 0
        sum_sq = 0.0
        count = 0
        with wave.open(path, "rb") as wf:
            if wf.getsampwidth() != 2:
                return 0.0, 0
            while True:
                frames = wf.readframes(8192)
                if not frames:
                    break
                for i in range(0, len(frames) - 1, 2):
                    sample = struct.unpack_from("<h", frames, i)[0]
                    abs_sample = abs(sample)
                    if abs_sample > peak:
                        peak = abs_sample
                    sum_sq += sample * sample
                    count += 1
        if count == 0:
            return 0.0, 0
        return (sum_sq / count) ** 0.5, peak
    except Exception as exc:  # noqa: BLE001 -- a bad/missing wav is a caller error elsewhere, not this gate's job
        logger.warning(f"[TRANSCRIBE] wav_peak_measure_failed error={exc}")
        return 0.0, 0


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
        rms, peak = measure_wav_peak(audio_path)
        meta: dict[str, Any] = {
            "peak": peak,
            "rms": round(rms, 2),
            "peak_threshold": peak_threshold,
        }
        if peak < peak_threshold:
            meta["silence_gate"] = "rejected"
            logger.info(
                f"[TRANSCRIBE] near_silent_skip peak={peak} threshold={peak_threshold}"
            )
            return TranscribeResult(text="", meta=meta)
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
