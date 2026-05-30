# services/orion-whisper-tts/app/stt.py

from __future__ import annotations

import base64
import logging
import os
import struct
import subprocess
import tempfile
import wave
from pathlib import Path

import torch
import whisper

logger = logging.getLogger(__name__)

# Client float peak warn threshold in Hub app.js uses VOICE_CLIENT_PEAK_MIN (~0.00025).
# int16 peak below STT_NEAR_SILENT_PEAK_INT16 skips Whisper (saves GPU).
_DEFAULT_NEAR_SILENT_PEAK_INT16 = 50


def _peak_threshold() -> int:
    raw = os.environ.get("STT_NEAR_SILENT_PEAK_INT16")
    if raw is not None:
        try:
            return max(1, min(32767, int(raw)))
        except ValueError:
            logger.warning(
                "[STT] invalid STT_NEAR_SILENT_PEAK_INT16=%r; using default %d",
                raw,
                _DEFAULT_NEAR_SILENT_PEAK_INT16,
            )
    try:
        from .settings import settings

        return max(1, min(32767, int(settings.stt_near_silent_peak_int16)))
    except Exception:
        return _DEFAULT_NEAR_SILENT_PEAK_INT16


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

    def _canonicalize_wav(self, src_path: str) -> str:
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
                "-sample_fmt",
                "s16",
                out.name,
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            stderr_tail = (proc.stderr or "")[-500:]
            raise RuntimeError(f"ffmpeg convert failed: {stderr_tail or proc.returncode}")
        return out.name

    @staticmethod
    def _client_suggests_speech(client_audio_meta: dict | None) -> bool:
        if not isinstance(client_audio_meta, dict):
            return False
        raw = client_audio_meta.get("source_peak")
        if raw is None:
            raw = client_audio_meta.get("peak")
        try:
            return float(raw) >= 0.00025
        except (TypeError, ValueError):
            return False

    def transcribe(
        self,
        audio_b64: str,
        language: str = "en",
        audio_format: str = "wav",
        client_audio_meta: dict | None = None,
    ) -> tuple[str, dict]:
        fmt = (audio_format or "wav").lower().strip()
        peak_threshold = _peak_threshold()
        audio_bytes = base64.b64decode(audio_b64)
        if not audio_bytes:
            logger.warning("[STT] empty audio payload format=%s", fmt)
            return "", {
                "input_bytes": 0,
                "input_format": fmt,
                "wav_bytes": 0,
                "duration_sec": 0.0,
                "peak": 0,
                "rms": 0.0,
                "peak_threshold": peak_threshold,
                "silence_gate": "rejected",
            }
        logger.info("[STT] audio_bytes=%d format=%s", len(audio_bytes), fmt)

        suffix = ".webm" if "webm" in fmt else ".wav" if "wav" in fmt else f".{fmt}"

        src_path: str | None = None
        wav_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_in:
                temp_in.write(audio_bytes)
                temp_in.flush()
                src_path = temp_in.name

            input_peak = 0
            if suffix == ".wav":
                _, input_peak = self._measure_wav_levels(src_path)

            wav_path = self._canonicalize_wav(src_path)

            wav_size = Path(wav_path).stat().st_size if wav_path else 0
            duration_sec = self._probe_duration_sec(wav_path)
            rms, peak = self._measure_wav_levels(wav_path)
            logger.info(
                "[STT] wav_bytes=%d duration_sec=%.2f peak=%d rms=%.1f format=%s threshold=%d",
                wav_size,
                duration_sec,
                peak,
                rms,
                fmt,
                peak_threshold,
            )
            meta = {
                "input_bytes": len(audio_bytes),
                "input_format": fmt,
                "input_peak": input_peak,
                "wav_bytes": wav_size,
                "duration_sec": round(duration_sec, 3),
                "peak": peak,
                "rms": round(rms, 2),
                "peak_threshold": peak_threshold,
                "silence_gate": "passed",
            }
            if peak < peak_threshold:
                if (
                    input_peak >= peak_threshold
                    or self._client_suggests_speech(client_audio_meta)
                ):
                    logger.warning(
                        "[STT] canonical peak=%d below threshold=%d but "
                        "input_peak=%d or client meta suggests speech; running Whisper",
                        peak,
                        peak_threshold,
                        input_peak,
                    )
                    meta["silence_gate"] = "passed_client_override"
                else:
                    meta["silence_gate"] = "rejected"
                    logger.warning(
                        "[STT] near-silent input peak=%d input_peak=%d rms=%.1f threshold=%d",
                        peak,
                        input_peak,
                        rms,
                        peak_threshold,
                    )
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
