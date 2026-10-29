"""Heat sensitivities as data, not code (body economy Patch B2, Organ 2 Budget).

``config/physiology/heat_sensitivity.yaml`` is WRITTEN by
``scripts/analysis/fit_heat_sensitivity.py --write-config``. Nobody hand-edits it. Per host it holds:

* the short-run (2 h step response) sensitivity in degC per 100 W, with its 95% interval;
* ``n``, ``fitted_at`` and the exact fit command;
* the host's baseline watts (p10 of hourly means in the fit window): Orion's heat share is
  charged on the watts above it.

The cabinet reading is used raw. There is no sensor offset: an earlier version lowered every
reading after 2026-10-17 00:35 by 2 C, on the belief that the sensor moved during the vision
bring-up. The sensor never moved (Juniper, 2026-10-29); the ~2 C rise is real heat. A document that
still carries a non-empty ``measurement_offsets`` list is refused as malformed (heat reads
``no_reading`` with that reason), so the offset cannot come back silently.

A host entry older than ``max_age_days`` (30), or missing, reads as ``no_reading`` for that host's
heat cost, never 0. Pure parsing plus one file read; no clock (the caller passes ``now``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from orion.physiology.prices import usable_sensitivity

SCHEMA = "heat_sensitivity.v1"
DEFAULT_MAX_AGE_DAYS = 30.0


@dataclass(frozen=True)
class HostSensitivity:
    host: str
    point: float
    lo: float
    hi: float
    horizon: str
    n: int
    fitted_at: datetime
    regressor: str
    baseline_watts: float
    fit_command: str

    @property
    def usable(self) -> Optional[float]:
        return usable_sensitivity(self.point, self.lo, self.hi)


@dataclass(frozen=True)
class HeatSensitivityConfig:
    hosts: Mapping[str, HostSensitivity]
    fitted_at: Optional[datetime] = None
    window: tuple[str, str] = ("", "")


def _dt(v: Any, what: str) -> datetime:
    if isinstance(v, datetime):
        d = v
    elif isinstance(v, str):
        d = datetime.fromisoformat(v.replace("Z", "+00:00"))
    else:
        raise ValueError(f"{what}: not a timestamp: {v!r}")
    return d.replace(tzinfo=timezone.utc) if d.tzinfo is None else d.astimezone(timezone.utc)


def _num(v: Any, what: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        raise ValueError(f"{what}: not a finite number: {v!r}")
    return float(v)


def parse_heat_sensitivity(doc: Any) -> HeatSensitivityConfig:
    """YAML document -> config. Raises ValueError on anything malformed (the caller turns that
    into a heat ``no_reading`` with the reason, never into a sensitivity of 0)."""
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        raise ValueError(f"not a {SCHEMA} document")
    hosts: dict[str, HostSensitivity] = {}
    for name, e in (doc.get("hosts") or {}).items():
        if not isinstance(e, dict):
            raise ValueError(f"hosts.{name}: not a mapping")
        ci = e.get("ci95")
        if not (isinstance(ci, list) and len(ci) == 2):
            raise ValueError(f"hosts.{name}.ci95: need [lo, hi]")
        hosts[str(name)] = HostSensitivity(
            host=str(name), point=_num(e.get("point"), f"hosts.{name}.point"),
            lo=_num(ci[0], f"hosts.{name}.ci95[0]"), hi=_num(ci[1], f"hosts.{name}.ci95[1]"),
            horizon=str(e.get("horizon") or ""), n=int(_num(e.get("n"), f"hosts.{name}.n")),
            fitted_at=_dt(e.get("fitted_at"), f"hosts.{name}.fitted_at"),
            regressor=str(e.get("regressor") or ""),
            baseline_watts=_num(e.get("baseline_watts_p10"), f"hosts.{name}.baseline_watts_p10"),
            fit_command=str(e.get("fit_command") or ""))
    if doc.get("measurement_offsets"):
        raise ValueError("measurement_offsets: removed (the cabinet sensor never moved; the 10-17 rise is "
                         "real heat). Refit with fit_heat_sensitivity.py --write-config")
    window = doc.get("window") or ["", ""]
    return HeatSensitivityConfig(
        hosts=hosts,
        fitted_at=_dt(doc["fitted_at"], "fitted_at") if doc.get("fitted_at") else None,
        window=(str(window[0]), str(window[1])))


def load_heat_sensitivity(path: str | Path) -> tuple[Optional[HeatSensitivityConfig], Optional[str]]:
    """(config, None) or (None, reason). Never raises: a missing or broken file is a reason."""
    import yaml

    p = Path(path)
    try:
        text = p.read_text()
    except FileNotFoundError:
        return None, f"missing:{p.name}"
    except OSError as exc:
        return None, f"read_failed:{p.name}:{type(exc).__name__}"
    try:
        return parse_heat_sensitivity(yaml.safe_load(text)), None
    except Exception as exc:  # noqa: BLE001 -- malformed config is a no_reading reason
        return None, f"malformed:{p.name}:{str(exc)[:120]}"


def host_sensitivity(cfg: Optional[HeatSensitivityConfig], host: str, now: datetime, *,
                     max_age_days: float = DEFAULT_MAX_AGE_DAYS
                     ) -> tuple[Optional[HostSensitivity], Optional[float], Optional[str]]:
    """(entry, usable degC per 100 W, None) or (entry|None, None, reason). Missing or older than
    ``max_age_days`` is a reason, never a sensitivity of 0."""
    if cfg is None:
        return None, None, "no_config"
    e = cfg.hosts.get(host)
    if e is None:
        return None, None, f"not_fitted:{host}"
    age = now - e.fitted_at
    if age > timedelta(days=max_age_days):
        return e, None, f"stale:{host}:fitted {e.fitted_at.date().isoformat()}"
    v = e.usable
    if v is None:
        return e, None, f"unusable:{host}"
    return e, v, None


__all__ = [
    "DEFAULT_MAX_AGE_DAYS",
    "HeatSensitivityConfig",
    "HostSensitivity",
    "SCHEMA",
    "host_sensitivity",
    "load_heat_sensitivity",
    "parse_heat_sensitivity",
]
