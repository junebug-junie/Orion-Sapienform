# services/orion-whisper-tts/app/stt.py

from __future__ import annotations

import base64
import logging
import struct
import subprocess
import tempfile
import wave
from pathlib import Path

import torch
import whisper

logger = logging.getLogger(__name__)

# Client float peak gate in Hub app.js uses VOICE_CLIENT_PEAK_MIN (~0.003).
# int16 peak below this skips Whisper (saves GPU; matches silent WAV uploads).
STT_NEAR_SILENT_PEAK_INT16 = 200


class STTEngine:
    def __init__(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model("base", device=device)
        logger.info("[STT] Whisper model loaded on %s", device)

    def _measure_wav_levels(self, path: str) -> tuple[float, int]:
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
                for i in range(0, len(frames), 2):
                    sample = struct.unpack_from("<h", frames, i)[0]
                    abs_sample = abs(sample)
                    if abs_sample > peak:
                        peak = abs_sample
                    sum_sq += sample * sample
                    count += 1
        if count == 0:
            return 0.0, 0
        return (sum_sq / count) ** 0.5, peak

    def _probe_duration_sec(self, path: str) -> float:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            return 0.0
        try:
            return float((proc.stdout or "").strip())
        except ValueError:
            return 0.0

    def _to_wav(self, src_path: str) -> str:
        out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        out.close()
        proc = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                src_path,
                "-ar",
                "16000",
                "-ac",
                "1",
                out.name,
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg convert failed: {proc.stderr[-500:] if proc.stderr else proc.returncode}"
            )
        return out.name

    def transcribe(
        self,
        audio_b64: str,
        language: str = "en",
        audio_format: str = "wav",
    ) -> tuple[str, dict]:
        audio_bytes = base64.b64decode(audio_b64)
        if not audio_bytes:
            logger.warning("[STT] empty audio payload format=%s", audio_format)
            return "", {"peak": 0, "rms": 0.0, "duration_sec": 0.0}
        logger.info("[STT] audio_bytes=%d format=%s", len(audio_bytes), audio_format)

        fmt = (audio_format or "wav").lower().strip()
        suffix = ".webm" if "webm" in fmt else ".wav" if "wav" in fmt else f".{fmt}"

        src_path: str | None = None
        wav_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_in:
                temp_in.write(audio_bytes)
                temp_in.flush()
                src_path = temp_in.name

            if suffix == ".wav":
                wav_path = src_path
            else:
                wav_path = self._to_wav(src_path)

            wav_size = Path(wav_path).stat().st_size if wav_path else 0
            duration_sec = self._probe_duration_sec(wav_path)
            rms, peak = self._measure_wav_levels(wav_path)
            logger.info(
                "[STT] wav_bytes=%d duration_sec=%.2f peak=%d rms=%.1f format=%s",
                wav_size,
                duration_sec,
                peak,
                rms,
                fmt,
            )
            meta = {
                "peak": peak,
                "rms": round(rms, 2),
                "duration_sec": round(duration_sec, 3),
            }
            if peak < STT_NEAR_SILENT_PEAK_INT16:
                logger.warning("[STT] near-silent input peak=%d rms=%.1f", peak, rms)
                return "", meta
            result = self.model.transcribe(
                wav_path,
                language=language,
                fp16=False,
                condition_on_previous_text=False,
                no_speech_threshold=0.35,
            )
            text = (result.get("text") or "").strip()
            logger.info("[STT] transcribed len=%d format=%s", len(text), fmt)
            return text, meta
        finally:
            for p in {src_path, wav_path}:
                if p:
                    try:
                        Path(p).unlink(missing_ok=True)
                    except OSError:
                        pass
