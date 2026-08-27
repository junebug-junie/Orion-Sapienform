# services/orion-whisper-tts/app/settings.py

from typing import Optional

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Identity
    service_name: str = Field("whisper-tts", env="SERVICE_NAME")
    service_version: str = Field("0.1.0", env="SERVICE_VERSION")

    # Bus
    orion_bus_url: str = Field("redis://localhost:6379/0", env="ORION_BUS_URL")
    orion_bus_enabled: bool = Field(True, env="ORION_BUS_ENABLED")
    orion_bus_enforce_catalog: bool = Field(False, env="ORION_BUS_ENFORCE_CATALOG")

    # Channels
    channel_tts_intake: str = Field(
        "orion:tts:intake",
        env="CHANNEL_TTS_INTAKE",
    )

    # Timeouts (mainly here for symmetry; this service is mostly bus-based)
    connect_timeout_sec: int = Field(10, env="CONNECT_TIMEOUT_SEC")
    read_timeout_sec: int = Field(60, env="READ_TIMEOUT_SEC")

    # TTS config
    tts_backend: str = Field("coqui", env="TTS_BACKEND")
    tts_model_name: str = Field(
        "tts_models/multilingual/multi-dataset/xtts_v2",
        env="TTS_MODEL_NAME",
    )
    tts_use_gpu: bool = Field(True, env="TTS_USE_GPU")
    tts_default_language: str = Field("en", env="TTS_DEFAULT_LANGUAGE")
    tts_default_speaker: Optional[str] = Field(None, env="TTS_DEFAULT_SPEAKER")
    tts_default_speaker_wav: Optional[str] = Field(None, env="TTS_DEFAULT_SPEAKER_WAV")
    tts_split_sentences: bool = Field(True, env="TTS_SPLIT_SENTENCES")
    tts_voice_profile_dir: str = Field("/models/voices", env="TTS_VOICE_PROFILE_DIR")
    whisper_tts_stt_timeout_sec: float = Field(
        90.0,
        env="WHISPER_TTS_STT_TIMEOUT_SEC",
    )
    whisper_tts_synth_timeout_sec: float = Field(
        120.0,
        env="WHISPER_TTS_SYNTH_TIMEOUT_SEC",
    )
    stt_near_silent_peak_int16: int = Field(
        50,
        env="STT_NEAR_SILENT_PEAK_INT16",
    )
    # Reject a transcription when Whisper's OWN per-segment no_speech_prob
    # says there was no speech there. The peak gate above is a cheap
    # pre-filter on amplitude; this is the model's own judgement, and it is
    # what actually stops fabricated text.
    #
    # Why this exists (2026-08-26): the sibling gate in
    # orion-affectgpt-worker, set to the identical peak=50, PASSED a clip
    # measured at peak=114 / rms=8.68 (-49 dBFS), and Whisper returned
    # "Thanks for the light, Egyptians. Thanks for the eyesight, thanks for
    # the thanks, this was a long time ago." on a turn where Juniper had
    # actually said "I'm feeling really tired." A downstream model then read
    # her affect off that invented sentence. An amplitude gate alone cannot
    # catch this -- 0.15% of full scale is still "loud enough" numerically.
    #
    # 0.6 rather than Whisper's own internal 0.35 default: this is a
    # post-hoc reject on returned segments, not the decoder's suppression
    # threshold, so it should only discard segments the model is fairly
    # confident are silence. Tune via env if real speech ever gets dropped.
    stt_max_no_speech_prob: float = Field(
        0.6,
        ge=0.0,
        le=1.0,
        env="STT_MAX_NO_SPEECH_PROB",
    )

    # CUDA liveness watchdog (app/cuda_watchdog.py). Real incident,
    # 2026-08-26: a docker+nvidia-container-toolkit staleness quirk
    # ("Failed to initialize NVML: Unknown Error") left this container's
    # torch.cuda.is_available() silently False mid-uptime -- Coqui TTS
    # hard-crashed on its first real request after that, while STT (which
    # has a CPU fallback and had already initialized on CUDA) kept working.
    # A plain container restart fixed it. This watchdog detects that
    # transition and restarts the process itself, so the "restart:
    # unless-stopped" compose policy recovers automatically instead of the
    # failure sitting silent until a human notices a broken voice reply.
    cuda_watchdog_enabled: bool = Field(True, env="CUDA_WATCHDOG_ENABLED")
    # gt=0, not ge=0: review finding, 2026-08-26 -- 0 or negative turns the
    # loop's own asyncio.sleep into an unthrottled busy-loop hammering the
    # NVML/driver layer every event-loop tick, which is a plausible way to
    # WORSEN a real staleness condition rather than detect it.
    cuda_watchdog_poll_sec: float = Field(30.0, gt=0, env="CUDA_WATCHDOG_POLL_SEC")
    # Consecutive failed checks required before restarting -- absorbs a
    # single transient NVML hiccup rather than restarting on one bad poll.
    # ge=1: review finding, 2026-08-26 -- 0 makes should_trigger_restart(1, 0)
    # True on the very first check, silently defeating the whole point of a
    # debounce threshold.
    cuda_watchdog_failure_threshold: int = Field(
        2, ge=1, env="CUDA_WATCHDOG_FAILURE_THRESHOLD"
    )

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
