import os

def _b(name: str, default="false") -> bool:
    """Convert environment variable to boolean with lenient parsing."""
    return os.getenv(name, str(default)).lower() in ("1", "true", "yes", "y")

# 🧠 Core service settings
SERVICE_NAME = os.getenv("SERVICE_NAME", "orion-brain")
PORT = int(os.getenv("PORT", "8088"))

# 🧩 Redis + Bus
REDIS_URL = os.getenv("REDIS_URL", "redis://orion-bus-core:6379/0")
EVENTS_ENABLE = _b("EVENTS_ENABLE", "true")
BUS_OUT_ENABLE = _b("BUS_OUT_ENABLE", "true")

# 🛰️ Bus stream names (logical channels)
EVENTS_STREAM = os.getenv("EVENTS_STREAM", "orion:evt:gateway")
BUS_OUT_STREAM = os.getenv("BUS_OUT_STREAM", "orion:bus:out")

# 🕐 Redis connection wait parameters (for startup health)
REDIS_WAIT_ATTEMPTS = int(os.getenv("REDIS_WAIT_ATTEMPTS", "10"))
REDIS_WAIT_DELAY = int(os.getenv("REDIS_WAIT_DELAY", "2"))

# 🤖 LLM Backend routing
BACKENDS = [b.strip() for b in os.getenv("BACKENDS", "").split(",") if b.strip()]
SELECTION_POLICY = os.getenv("SELECTION_POLICY", "least_conn")
HEALTH_INTERVAL = int(os.getenv("HEALTH_INTERVAL_SEC", "5"))
CONNECT_TIMEOUT = int(os.getenv("CONNECT_TIMEOUT_SEC", "10"))
READ_TIMEOUT = int(os.getenv("READ_TIMEOUT_SEC", "600"))
