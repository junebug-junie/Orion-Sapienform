"""Read the status.json the portal fetcher writes. Missing or unreadable -> None."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from orion.energy.importer_status import PortalStatus, parse_portal_status

logger = logging.getLogger("orion-energy.portal_status")


def read_portal_status(path: Path) -> Optional[PortalStatus]:
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        logger.warning("energy_portal_status_unreadable path=%s error=%s", path, exc)
        return None
    if not isinstance(raw, dict):
        logger.warning("energy_portal_status_invalid path=%s error=not a JSON object", path)
        return None
    try:
        return parse_portal_status(raw)
    except ValueError as exc:
        logger.warning("energy_portal_status_invalid path=%s error=%s", path, exc)
        return None
