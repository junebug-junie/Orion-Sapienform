# services/orion-whisper-tts/app/tts.py
from __future__ import annotations

import base64
import io
import logging
import tempfile
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .settings import Settings, settings

logger = logging.getLogger(__name__)


@dataclass
class TTSOutput:
    audio_b64: str
    content_type: str = "audio/wav"
    metadata: dict = field(default_factory=dict)
    duration_sec: Optional[float] = None


@dataclass
class SynthesisPlan:
    kwargs: dict[str, Any]
    metadata: dict[str, Any]


def _ensure_torch_load_compat() -> None:
    """PyTorch 2.6+ defaults weights_only=True; Coqui XTTS checkpoints need False."""
    import torch

    if getattr(_ensure_torch_load_compat, "_patched", False):
        return
    _orig_load = torch.load

    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_load(*args, **kwargs)

    torch.load = _load  # type: ignore[method-assign]
    _ensure_torch_load_compat._patched = True


def _is_xtts_model(model_name: str) -> bool:
    return "xtts" in model_name.lower()


def _resolve_speaker_wav_path(candidate: str, profile_dir: str) -> Path:
    root = Path(profile_dir).resolve()
    raw = Path(candidate)
    resolved = (raw if raw.is_absolute() else root / raw).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"TTS speaker_wav must be under {root}, got {candidate!r}"
        ) from exc
    if not resolved.is_file():
        raise FileNotFoundError(
            f"TTS speaker_wav not found: {resolved} (profile_dir={root})"
        )
    return resolved


def resolve_synthesis_plan(
    cfg: Settings,
    *,
    voice_id: Optional[str],
    language: Optional[str],
    options: Optional[dict],
) -> SynthesisPlan:
    opts = dict(options or {})
    lang = language or opts.pop("language", None) or cfg.tts_default_language
    split = bool(opts.pop("split_sentences", cfg.tts_split_sentences))

    speaker_wav: Optional[str] = opts.pop("speaker_wav", None)
    speaker: Optional[str] = opts.pop("speaker", None)
    speaker_wav_used = False

    if speaker_wav:
        resolved = _resolve_speaker_wav_path(speaker_wav, cfg.tts_voice_profile_dir)
        speaker_wav = str(resolved)
        speaker_wav_used = True
    elif cfg.tts_default_speaker_wav:
        resolved = _resolve_speaker_wav_path(
            cfg.tts_default_speaker_wav,
            cfg.tts_voice_profile_dir,
        )
        speaker_wav = str(resolved)
        speaker_wav_used = True
    elif voice_id:
        looks_like_file = (
            voice_id.startswith("/")
            or voice_id.endswith(".wav")
            or Path(voice_id).suffix.lower() == ".wav"
        )
        profile_candidate = Path(cfg.tts_voice_profile_dir) / voice_id
        if looks_like_file or profile_candidate.exists():
            resolved = _resolve_speaker_wav_path(voice_id, cfg.tts_voice_profile_dir)
            speaker_wav = str(resolved)
            speaker_wav_used = True
        else:
            speaker = voice_id
    elif cfg.tts_default_speaker:
        speaker = cfg.tts_default_speaker

    meta: dict[str, Any] = {
        "backend": cfg.tts_backend,
        "model_name": cfg.tts_model_name,
        "language": lang,
        "voice_id": voice_id,
        "speaker": speaker,
        "speaker_wav_basename": Path(speaker_wav).name if speaker_wav else None,
        "speaker_wav_used": speaker_wav_used,
        "split_sentences": split,
    }

    kwargs: dict[str, Any] = {}
    if _is_xtts_model(cfg.tts_model_name):
        kwargs["language"] = lang
        kwargs["split_sentences"] = split
        if speaker_wav:
            kwargs["speaker_wav"] = speaker_wav
        elif speaker:
            kwargs["speaker"] = speaker
        else:
            raise ValueError(
                "XTTS requires a speaker: set TTS_DEFAULT_SPEAKER, "
                "TTS_DEFAULT_SPEAKER_WAV, voice_id, or options.speaker_wav"
            )
    else:
        if speaker:
            kwargs["speaker"] = speaker
        if speaker_wav:
            kwargs["speaker_wav"] = speaker_wav

    for key in opts:
        logger.warning("[TTS] Ignoring unsupported option key=%s", key)

    return SynthesisPlan(kwargs=kwargs, metadata=meta)


def _validate_xtts_defaults(cfg: Settings) -> None:
    if not _is_xtts_model(cfg.tts_model_name):
        return
    has_speaker = bool(cfg.tts_default_speaker)
    has_wav = bool(cfg.tts_default_speaker_wav)
    if has_speaker or has_wav:
        if has_wav:
            _resolve_speaker_wav_path(cfg.tts_default_speaker_wav, cfg.tts_voice_profile_dir)
        return
    raise RuntimeError(
        "XTTS model configured but no default voice: set TTS_DEFAULT_SPEAKER "
        "(built-in name) or TTS_DEFAULT_SPEAKER_WAV (reference .wav under "
        f"TTS_VOICE_PROFILE_DIR={cfg.tts_voice_profile_dir})"
    )


class CoquiBackend:
    def __init__(self, cfg: Settings):
        _ensure_torch_load_compat()
        from TTS.api import TTS

        self.cfg = cfg
        _validate_xtts_defaults(cfg)
        logger.info(
            "[TTS] Loading coqui backend=%s model=%s use_gpu=%s language=%s "
            "default_speaker=%s default_speaker_wav=%s",
            cfg.tts_backend,
            cfg.tts_model_name,
            cfg.tts_use_gpu,
            cfg.tts_default_language,
            cfg.tts_default_speaker,
            cfg.tts_default_speaker_wav,
        )
        self.tts = TTS(cfg.tts_model_name, gpu=cfg.tts_use_gpu)
        logger.info("[TTS] Coqui model loaded.")

    def synthesize(
        self,
        text: str,
        *,
        voice_id: Optional[str],
        language: Optional[str],
        options: Optional[dict],
    ) -> TTSOutput:
        if not text:
            return TTSOutput(audio_b64="", metadata={"empty_text": True})

        plan = resolve_synthesis_plan(
            self.cfg,
            voice_id=voice_id,
            language=language,
            options=options,
        )
        started = time.perf_counter()

        speaker_wav_path = plan.kwargs.get("speaker_wav")
        logger.info(
            "[TTS] tts_to_file language=%s speaker_set=%s speaker_wav_set=%s "
            "speaker_wav_exists=%s split_sentences=%s",
            plan.kwargs.get("language"),
            bool(plan.kwargs.get("speaker")),
            bool(speaker_wav_path),
            bool(speaker_wav_path and Path(str(speaker_wav_path)).is_file()),
            plan.kwargs.get("split_sentences"),
        )
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            call_kwargs = {"text": text, "file_path": f.name, **plan.kwargs}
            try:
                self.tts.tts_to_file(**call_kwargs)
            except Exception as exc:
                logger.error(
                    "[TTS] synthesis failed model=%s language=%s speaker=%s "
                    "speaker_wav=%s speaker_wav_exists=%s error=%s",
                    self.cfg.tts_model_name,
                    plan.kwargs.get("language"),
                    plan.kwargs.get("speaker"),
                    speaker_wav_path,
                    bool(speaker_wav_path and Path(str(speaker_wav_path)).is_file()),
                    exc,
                    exc_info=True,
                )
                raise RuntimeError(
                    f"Coqui synthesis failed for model={self.cfg.tts_model_name}: {exc}. "
                    f"language={plan.kwargs.get('language')!r} speaker={plan.kwargs.get('speaker')!r} "
                    f"speaker_wav={speaker_wav_path!r}"
                ) from exc
            f.seek(0)
            audio_bytes = f.read()

        duration_sec: Optional[float] = None
        try:
            with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
                duration_sec = wf.getnframes() / float(wf.getframerate())
        except Exception:
            pass

        synthesis_ms = int((time.perf_counter() - started) * 1000)
        metadata = {
            **plan.metadata,
            "synthesis_ms": synthesis_ms,
            "gpu_enabled": self.cfg.tts_use_gpu,
        }
        return TTSOutput(
            audio_b64=base64.b64encode(audio_bytes).decode("utf-8"),
            content_type="audio/wav",
            metadata=metadata,
            duration_sec=duration_sec,
        )


class TTSEngine:
    """Facade: select backend by TTS_BACKEND."""

    def __init__(self):
        backend = (settings.tts_backend or "coqui").strip().lower()
        if backend == "coqui":
            self._backend = CoquiBackend(settings)
            self.backend_name = "coqui"
        else:
            raise RuntimeError(
                f"Unsupported TTS_BACKEND={settings.tts_backend!r}. Supported: coqui"
            )
        logger.info(
            "[TTS] Engine ready backend=%s model=%s use_gpu=%s language=%s "
            "default_speaker=%s default_speaker_wav=%s",
            self.backend_name,
            settings.tts_model_name,
            settings.tts_use_gpu,
            settings.tts_default_language,
            settings.tts_default_speaker,
            settings.tts_default_speaker_wav,
        )

    def synthesize_to_b64(
        self,
        text: str,
        *,
        voice_id: Optional[str] = None,
        language: Optional[str] = None,
        options: Optional[dict] = None,
    ) -> TTSOutput:
        return self._backend.synthesize(
            text,
            voice_id=voice_id,
            language=language,
            options=options,
        )
