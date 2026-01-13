from __future__ import annotations

import logging
from typing import Any, Dict

from .catalog_loader import load_channel_catalog

logger = logging.getLogger("orion.bus.catalog")


class ChannelCatalogEnforcer:
    def __init__(self, *, enforce: bool = False, catalog: Dict[str, Dict[str, Any]] | None = None) -> None:
        self.enforce = enforce
        self._catalog = catalog
        self._validated = False

    def _ensure_catalog(self) -> Dict[str, Dict[str, Any]]:
        if self._catalog is None:
            self._catalog = load_channel_catalog()
        if not self._validated:
            self._validate_catalog(self._catalog)
            self._validated = True
        return self._catalog

    def _validate_catalog(self, catalog: Dict[str, Dict[str, Any]]) -> None:
        for name in catalog.keys():
            if "*" not in name:
                continue
            if name.count("*") > 1 or not name.endswith("*"):
                raise ValueError(f"Invalid wildcard channel entry: {name}")
            if name == "*":
                raise ValueError("Wildcard channel entry must include a prefix before '*'.")

    def _entry_for(self, channel: str) -> Dict[str, Any] | None:
        catalog = self._ensure_catalog()
        if channel in catalog:
            return catalog[channel]
        best_match: Dict[str, Any] | None = None
        best_prefix_len = -1
        for name, entry in catalog.items():
            if not name.endswith("*"):
                continue
            prefix = name[:-1]
            if channel.startswith(prefix) and len(prefix) > best_prefix_len:
                best_prefix_len = len(prefix)
                best_match = entry
        return best_match

    def validate(self, channel: str) -> None:
        if self._entry_for(channel):
            return
        message = f"Channel not found in catalog: {channel}"
        if self.enforce:
            raise ValueError(message)
        logger.warning(message)

    def entry_for(self, channel: str) -> Dict[str, Any] | None:
        return self._entry_for(channel)


enforcer = ChannelCatalogEnforcer()
