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
    cuda_watchdog_poll_sec: float = Field(30.0, env="CUDA_WATCHDOG_POLL_SEC")
    # Consecutive failed checks required before restarting -- absorbs a
    # single transient NVML hiccup rather than restarting on one bad poll.
    cuda_watchdog_failure_threshold: int = Field(
        2, env="CUDA_WATCHDOG_FAILURE_THRESHOLD"
    )

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
