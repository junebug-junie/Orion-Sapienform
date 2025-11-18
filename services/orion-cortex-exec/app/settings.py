 # orion-cortex-exec/app/settings.py

import os
from pydantic import BaseSettings


class Settings(BaseSettings):
    NODE_NAME: str = "orion-athena"

    # bus connection
    ORION_BUS_URL: str = "redis://orion-redis:6379/0"
    ORION_BUS_ENABLED: bool = True

    # channels
    EXEC_REQUEST_PREFIX: str = "orion.exec.request"
    EXEC_RESULT_PREFIX: str = "orion.exec.result"

    # timeouts
    STEP_TIMEOUT_MS: int = 8000

    class Config:
        env_prefix = "ORION_CORTEX_"


settings = Settings()

