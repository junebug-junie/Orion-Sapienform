from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model_path: str = Field(..., env="LLAMACPP_MODEL_PATH")
    n_gpu_layers: int = Field(-1, env="LLAMACPP_N_GPU_LAYERS")
    n_ctx: int = Field(2048, env="LLAMACPP_CTX_SIZE")

    host: str = Field("0.0.0.0", env="LLAMACPP_HOST")
    port: int = Field(8005, env="LLAMACPP_PORT")


settings = Settings()
