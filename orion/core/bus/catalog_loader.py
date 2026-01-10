from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger("orion.bus.catalog")

# Historically the repo has used both filenames:
#   - orion/bus/channels.yaml  (plural)
#   - orion/bus/channel.yaml   (singular)
# We accept either so the catalog can't silently disappear.
_BUS_DIR = Path(__file__).resolve().parents[2] / "bus"
CATALOG_CANDIDATES = [
    _BUS_DIR / "channels.yaml",
    _BUS_DIR / "channel.yaml",
]


def load_channel_catalog(path: Path | None = None) -> Dict[str, Dict[str, Any]]:
    target = path
    if target is None:
        for candidate in CATALOG_CANDIDATES:
            if candidate.exists():
                target = candidate
                break
        else:
            target = CATALOG_CANDIDATES[0]

    if not target.exists():
        logger.warning("Channel catalog not found at %s", target)
        return {}

    try:
        with target.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    except Exception as exc:
        logger.warning("Failed to load channel catalog from %s: %s", target, exc)
        return {}

    channels = raw.get("channels", [])
    catalog: Dict[str, Dict[str, Any]] = {}
    if not isinstance(channels, list):
        logger.warning("Channel catalog is missing 'channels' list in %s", target)
        return catalog

    for entry in channels:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not name:
            continue
        catalog[name] = entry
    return catalog
