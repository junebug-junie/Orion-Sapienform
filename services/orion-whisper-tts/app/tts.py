# services/orion-whisper-tts/app/tts.py
from __future__ import annotations

import base64
import io
import logging
import re
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


def _speaker_wav_refs_exist(speaker_wav: Any) -> bool:
    """Do all configured reference files exist? Safe for a list or a string.

    `speaker_wav` is a list under multi-reference, and `Path(str(<list>))` is
    never a real path -- the previous expression therefore logged
    speaker_wav_exists=False on every multi-reference synthesis, an absent
    reading asserting a cause, and the first thing an operator reads during an
    outage. Above ~85 references it was worse than wrong: the stringified list
    exceeds NAME_MAX and `is_file()` RAISES ENAMETOOLONG, which is not in
    pathlib's ignored errnos. At the failure-logging site that would replace
    the real synthesis error with a bogus one.
    """
    refs = (
        speaker_wav
        if isinstance(speaker_wav, list)
        else ([speaker_wav] if speaker_wav else [])
    )
    return bool(refs) and all(Path(p).is_file() for p in refs)


def _natural_key(path: Path) -> tuple:
    """Sort key that orders chunk_2 before chunk_10.

    Plain `sorted()` is lexicographic, so `chunk_1, chunk_10, chunk_11, chunk_2`
    -- and order is load-bearing here, because `gpt_cond_len` caps the
    concatenation at ~30s and the earliest files are the ones that reach the
    prosody latent. A recording chunked past nine pieces would otherwise be
    reordered silently, with nothing to notice but a changed voice.
    """
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part)
        for part in re.split(r"(\d+)", path.name)
        if part != ""
    )


def _resolve_speaker_wav_refs(candidate: str, profile_dir: str) -> list[Path]:
    """Resolve a speaker reference to one or more wav files.

    A DIRECTORY means multi-reference. XTTS's `get_conditioning_latents` accepts
    a list and does two different things with it: it MEANS the per-file speaker
    embeddings, and concatenates the audio for the GPT latent. The mean is the
    part that matters -- averaging several clips of the same person cancels the
    per-clip codec and room artifacts that a single clip bakes into the clone.

    This is also the only way to use more than one clip's worth of a recording,
    because each reference is independently truncated to `max_ref_len` (30s in
    the shipped checkpoint). A single 100s file is silently cut to its first
    30s; seven ~14s files are not.

    Order is sorted and therefore stable, which matters: `gpt_cond_len` (30s)
    caps the concatenation, so the earliest files decide the prosody latent
    while every file contributes equally to the speaker embedding.

    A single file behaves exactly as before -- one path, in a one-element list.
    """
    root = Path(profile_dir).resolve()
    raw = Path(candidate)
    resolved = (raw if raw.is_absolute() else root / raw).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"TTS speaker_wav must be under {root}, got {candidate!r}"
        ) from exc
    if resolved.is_dir():
        # The profile root itself must not resolve to "every voice on the host,
        # blended". `relative_to` succeeds when resolved == root, and a bus
        # request carrying options.speaker_wav="." is reachable from any
        # producer on the intake channel -- that would silently average every
        # reference in the directory into one composite speaker embedding.
        if resolved == root:
            raise ValueError(
                f"TTS speaker_wav must name a specific reference or "
                f"subdirectory, not the profile root itself: {root}"
            )
        refs: list[Path] = []
        for p in sorted(resolved.glob("*.wav"), key=_natural_key):
            if not p.is_file():
                continue  # e.g. a directory literally named something.wav
            # glob does NOT resolve symlinks and is_file() follows them, so a
            # symlink planted inside the directory would otherwise hand an
            # out-of-root file straight to the audio decoder. The containment
            # check above covers the candidate, not its children.
            try:
                p.resolve().relative_to(root)
            except ValueError as exc:
                raise ValueError(
                    f"TTS speaker_wav reference escapes {root}: {p}"
                ) from exc
            refs.append(p)
        if not refs:
            raise FileNotFoundError(
                f"TTS speaker_wav directory holds no .wav files: {resolved} "
                f"(profile_dir={root})"
            )
        return refs
    if not resolved.is_file():
        raise FileNotFoundError(
            f"TTS speaker_wav not found: {resolved} (profile_dir={root})"
        )
    return [resolved]


def _resolve_speaker_wav_path(candidate: str, profile_dir: str) -> Path:
    """Single-file view of the above, for callers that only need existence."""
    return _resolve_speaker_wav_refs(candidate, profile_dir)[0]


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
    # Populated whenever a reference resolves. A directory yields several paths
    # (multi-reference); a plain file yields exactly one, as before.
    speaker_wav_refs: list[Path] = []
    speaker_wav_source: Optional[str] = None

    if speaker_wav:
        speaker_wav_refs = _resolve_speaker_wav_refs(
            speaker_wav, cfg.tts_voice_profile_dir
        )
        speaker_wav_source = speaker_wav
        speaker_wav = str(speaker_wav_refs[0])
        speaker_wav_used = True
    elif speaker:
        # An explicit per-request `options.speaker` outranks the HOST DEFAULT
        # reference wav. Before this, `speaker` was popped from options above
        # and then silently discarded whenever TTS_DEFAULT_SPEAKER_WAV was set
        # (which is the live config on every host), because the kwargs builder
        # below prefers speaker_wav. The caller asked for a named voice and got
        # the cloned one, with no error and nothing in the metadata to say so.
        #
        # Consequence found live 2026-08-30: with a reference voice configured
        # there was NO way to request a built-in XTTS speaker over the bus at
        # all, so a voice-quality A/B against the built-in speakers could not
        # be run through the normal path.
        #
        # Deliberately narrow. This branch can regress nothing, because the
        # value it now honours was previously discarded -- no caller can depend
        # on it being ignored. `voice_id` is intentionally NOT moved above the
        # host default in the same patch: it is already routed below and Hub
        # sends it, so re-ranking it would change live behaviour rather than
        # un-break dead behaviour.
        pass
    elif cfg.tts_default_speaker_wav:
        speaker_wav_refs = _resolve_speaker_wav_refs(
            cfg.tts_default_speaker_wav,
            cfg.tts_voice_profile_dir,
        )
        speaker_wav_source = cfg.tts_default_speaker_wav
        speaker_wav = str(speaker_wav_refs[0])
        speaker_wav_used = True
    elif voice_id:
        looks_like_file = (
            voice_id.startswith("/")
            or voice_id.endswith(".wav")
            or Path(voice_id).suffix.lower() == ".wav"
        )
        profile_candidate = Path(cfg.tts_voice_profile_dir) / voice_id
        if looks_like_file or profile_candidate.exists():
            speaker_wav_refs = _resolve_speaker_wav_refs(
                voice_id, cfg.tts_voice_profile_dir
            )
            speaker_wav_source = voice_id
            speaker_wav = str(speaker_wav_refs[0])
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
        # Under multi-reference, name the DIRECTORY. Reporting the first chunk
        # would make a 7-file voice read as an ordinary single-file one to any
        # consumer or human skimming the reply metadata.
        "speaker_wav_basename": (
            Path(speaker_wav).parent.name
            if len(speaker_wav_refs) > 1
            else (Path(speaker_wav).name if speaker_wav else None)
        ),
        "speaker_wav_used": speaker_wav_used,
        "speaker_wav_count": len(speaker_wav_refs),
        "speaker_wav_basenames": (
            [p.name for p in speaker_wav_refs] if len(speaker_wav_refs) > 1 else None
        ),
        "speaker_wav_source": speaker_wav_source,
        "split_sentences": split,
    }

    kwargs: dict[str, Any] = {}
    if _is_xtts_model(cfg.tts_model_name):
        kwargs["language"] = lang
        kwargs["split_sentences"] = split
        if speaker_wav:
            # Pass a bare string for the single-reference case so nothing about
            # the existing path changes, and a list only when there really are
            # several references. XTTS accepts either.
            kwargs["speaker_wav"] = (
                [str(p) for p in speaker_wav_refs]
                if len(speaker_wav_refs) > 1
                else speaker_wav
            )
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
            # Non-XTTS backends are not known to accept a list here, so keep
            # them on the single-path form they have always been given.
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
        speaker_wav_exists = _speaker_wav_refs_exist(speaker_wav_path)
        logger.info(
            "[TTS] tts_to_file language=%s speaker_set=%s speaker_wav_set=%s "
            "speaker_wav_exists=%s split_sentences=%s",
            plan.kwargs.get("language"),
            bool(plan.kwargs.get("speaker")),
            bool(speaker_wav_path),
            speaker_wav_exists,
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
                    speaker_wav_exists,
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
