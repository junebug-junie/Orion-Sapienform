# settings.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_host: str = "0.0.0.0"
    app_port: int = 9000

    # Note: This URL must match your docker service name (llm-gpu-sync-test-llama-cpp)
    llamacpp_url: str = "http://llm-gpu-sync-test-llama-cpp:8080/v1/chat/completions"

    # This is the missing field causing your crash
    model_alias: str = "Active-GGUF-Model"

    model_config = {
        "env_prefix": "",
        "case_sensitive": False,
        "extra": "ignore" 
    }

settings = Settings()
