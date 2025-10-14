import logging
import sys
from app.settings import settings

# Configure a shared logger for the entire service.
# This ensures all log messages have a consistent format and destination.
logger = logging.getLogger(settings.SERVICE_NAME)

# Avoid adding duplicate handlers if the module is reloaded
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
