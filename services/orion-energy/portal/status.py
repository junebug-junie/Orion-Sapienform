"""status.json: what the fetcher last tried. orion-energy turns it into importer status."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from orion.energy.importer_status import PortalStatus, parse_portal_status, portal_status_dict

from .fetch import PortalOutcome


def read_status(path: Path) -> Optional[PortalStatus]:
    try:
        return parse_portal_status(json.loads(path.read_text()))
    except (OSError, ValueError):
        return None


def write_status(path: Path, outcome: PortalOutcome, *, now: datetime) -> PortalStatus:
    previous = read_status(path)
    status = PortalStatus(
        state=outcome.state,
        reason=outcome.reason,
        last_attempt_at=now,
        last_success_at=now if outcome.state == "ok" else (previous.last_success_at if previous else None),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.part")
    part.write_text(json.dumps(portal_status_dict(status)))
    part.rename(path)
    return status
