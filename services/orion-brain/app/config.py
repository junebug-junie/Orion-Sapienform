import os

def _b(name, default="false"):
    return os.getenv(name, str(default)).lower() in ("1","true","yes","y")

SERVICE_NAME = os.getenv("SERVICE_NAME", "orion-brain")
REDIS_URL    = os.getenv("REDIS_URL")  # e.g. redis://orion-redis:6379/0

EVENTS_ENABLE = _b("EVENTS_ENABLE", "true")
EVENTS_STREAM = os.getenv("EVENTS_STREAM", "orion:evt:gateway")

BUS_OUT_ENABLE = _b("BUS_OUT_ENABLE", "true")
BUS_OUT_STREAM = os.getenv("BUS_OUT_STREAM", "orion:bus:out")
