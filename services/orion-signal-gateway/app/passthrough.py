"""Validator for self-hardened organ signals arriving directly on signals.* channels."""
import logging
from typing import Optional

from orion.signals.models import OrionSignalV1
from orion.signals.registry import ORGAN_REGISTRY

logger = logging.getLogger(__name__)


class PassthroughValidator:
    """Validates OrionSignalV1 signals from self-hardened organs."""

    def validate(self, payload: dict) -> Optional[OrionSignalV1]:
        """
        Validate and return the signal, or None if invalid.
        Rejects: unknown organ_id, missing required fields, schema validation errors.
        """
        try:
            signal = OrionSignalV1.model_validate(payload)
        except Exception as exc:
            logger.warning(f"Passthrough signal schema validation failed: {exc}")
            return None

        if signal.organ_id not in ORGAN_REGISTRY:
            logger.warning(f"Passthrough signal rejected: unknown organ_id={signal.organ_id}")
            return None

        return signal
