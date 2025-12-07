# services/orion-whisper-tts/app/settings.py

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Identity
    service_name: str = Field("whisper-tts", env="SERVICE_NAME")
    service_version: str = Field("0.1.0", env="SERVICE_VERSION")

    # Bus
    orion_bus_url: str = Field("redis://localhost:6379/0", env="ORION_BUS_URL")
    orion_bus_enabled: bool = Field(True, env="ORION_BUS_ENABLED")

    # Channels
    channel_tts_intake: str = Field(
        "orion:tts:intake",
        env="CHANNEL_TTS_INTAKE",
    )

    # Timeouts (mainly here for symmetry; this service is mostly bus-based)
    connect_timeout_sec: int = Field(10, env="CONNECT_TIMEOUT_SEC")
    read_timeout_sec: int = Field(60, env="READ_TIMEOUT_SEC")

    # TTS config
    tts_model_name: str = Field(
        "tts_models/en/ljspeech/tacotron2-DDC",
        env="TTS_MODEL_NAME",
    )
    tts_use_gpu: bool = Field(True, env="TTS_USE_GPU")

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
