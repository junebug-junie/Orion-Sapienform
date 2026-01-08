from __future__ import annotations

import logging
from typing import Any, Dict

from .catalog_loader import load_channel_catalog

logger = logging.getLogger("orion.bus.catalog")


class ChannelCatalogEnforcer:
    def __init__(self, *, enforce: bool = False, catalog: Dict[str, Dict[str, Any]] | None = None) -> None:
        self.enforce = enforce
        self._catalog = catalog

    def _ensure_catalog(self) -> Dict[str, Dict[str, Any]]:
        if self._catalog is None:
            self._catalog = load_channel_catalog()
        return self._catalog

    def validate(self, channel: str) -> None:
        catalog = self._ensure_catalog()
        if channel in catalog:
            return
        message = f"Channel not found in catalog: {channel}"
        if self.enforce:
            raise ValueError(message)
        logger.warning(message)


enforcer = ChannelCatalogEnforcer()
