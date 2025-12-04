import os

def _b(name: str, default="false") -> bool:
    """Convert environment variable to boolean with lenient parsing."""
    return os.getenv(name, str(default)).lower() in ("1", "true", "yes", "y")

# 🧠 Core service settings
PROJECT = os.getenv("PROJECT", "brain")
SERVICE_NAME = os.getenv("SERVICE_NAME", "orion-brain")
SERVICE_VERSION = os.getenv("SERVICE_VERSION", "0.2.0")
PORT = int(os.getenv("PORT", "8088"))
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b-instruct-q8_0")

# 🧩 Redis + Bus
ORION_BUS_URL = os.getenv("ORION_BUS_URL", f"redis://{PROJECT}-bus-core:6379/0")
ORION_BUS_ENABLED = _b("ORION_BUS_ENABLED", "true")

CHANNEL_CHAT_HISTORY_LOG = os.getenv("CHANNEL_CHAT_HISTORY_LOG", "orion:chat:history:log")
# 🛰️ Bus stream names (logical channels)
CHANNEL_BRAIN_INTAKE = os.getenv("CHANNEL_BRAIN_INTAKE", "orion:brain:intake")
CHANNEL_BRAIN_OUT = os.getenv("CHANNEL_BRAIN_OUT", "orion:brain:out")
CHANNEL_BRAIN_STATUS = os.getenv("CHANNEL_BRAIN_STATUS", "orion:brain:status")
CHANNEL_BRAIN_STREAM = os.getenv("CHANNEL_BRAIN_STREAM", "orion:brain:stream")
CHANNEL_CORTEX_EXEC_INTAKE = os.getenv("CHANNEL_CORTEX_EXEC_INTAKE", "orion-exec:request:LLMGatewayService")

CHANNEL_CHAT_HISTORY_LOG = os.getenv("SUBSCRIBE_CHANNEL_CHAT", "orion:chat:history:log")
CHANNEL_DREAM_TRIGGER = os.getenv("CHANNEL_DREAM_TRIGGER", "orion:dream:trigger")

# LLM Gateway bus channels (align these with Hub + llm-gateway service)
CHANNEL_LLM_GATEWAY_EXEC_INTAKE = os.getenv(
    "CHANNEL_LLM_GATEWAY_EXEC_INTAKE",
    "orion-exec:request:LLMGatewayService",
)
CHANNEL_LLM_GATEWAY_REPLY_PREFIX = os.getenv(
    "CHANNEL_LLM_GATEWAY_REPLY_PREFIX",
    "orion:llm-gateway:chat:reply",
)

# Spark / meta-cognition channels
CHANNEL_SPARK_INTROSPECT_CANDIDATE = os.getenv("CHANNEL_SPARK_INTROSPECT_CANDIDATE", "orion:spark:introspect:candidate")

# Optional: cross-domain routing hooks
CHANNEL_VOICE_TRANSCRIPT = os.getenv("CHANNEL_VOICE_TRANSCRIPT", "orion:voice:transcript")
CHANNEL_VOICE_LLM = os.getenv("CHANNEL_VOICE_LLM", "orion:voice:llm")
CHANNEL_COLLAPSE_INTAKE = os.getenv("CHANNEL_COLLAPSE_INTAKE", "orion:collapse:intake")
CHANNEL_TTS_INTAKE = os.getenv("CHANNEL_TTS_INTAKE", "orion:tts:intake")

# 🕐 Redis connection wait parameters
REDIS_WAIT_ATTEMPTS = int(os.getenv("REDIS_WAIT_ATTEMPTS", "10"))
REDIS_WAIT_DELAY = int(os.getenv("REDIS_WAIT_DELAY", "2"))

# 🤖 LLM Backend routing
BACKENDS = [b.strip() for b in os.getenv("BACKENDS", "").split(",") if b.strip()]
SELECTION_POLICY = os.getenv("SELECTION_POLICY", "least_conn")
HEALTH_INTERVAL = int(os.getenv("HEALTH_INTERVAL_SEC", "5"))
CONNECT_TIMEOUT = int(os.getenv("CONNECT_TIMEOUT_SEC", "10"))
READ_TIMEOUT = int(os.getenv("READ_TIMEOUT_SEC", "600"))

