"""Run-cap override: read and judge a runtime hold on one family's start gate (Family C, 2026-11-16).

Design: docs/superpowers/specs/2026-11-16-family-c-first-experiment-economy-native.md ("Actuator: a run-cap
TTL override"). Contract: ``orion/schemas/physiology_run_cap.py``. Bounds: ``config/physiology/
run_cap_overrides.yaml``.

``judge`` is pure: the raw key value, its Redis TTL, the clock and the family's bound -> obey or ignore, with
a named reason. A key is OBEYED only when every rule holds:

* the family is listed in the bounds file (else ``not_configured``);
* the key has a positive TTL (``no_ttl``: a key with no expiry could hold the family forever);
* it parses as ``RunCapOverrideV1`` for this family and knob (``malformed`` / ``family_mismatch`` /
  ``knob_mismatch``);
* its value is in the bound's ``allowed`` list (``not_allowed``);
* ``set_at`` is not in the future (``future_set_at``, 60 s clock slack) and ``now - set_at <= max_hold_s``
  (``stale_set_at``). The writer never moves ``set_at``, so a writer stuck refreshing still lets go.

An obeyed key's ``retry_after_sec`` is its Redis TTL (never ``expires_at``): the town's self-start loop
holds a refusal in memory for that long without re-checking, so a longer retry would outlive a dead writer.

``read`` does the two Redis calls (GET, then TTL). Any Redis error raises; the caller decides.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import yaml

from orion.schemas.physiology_run_cap import RunCapOverrideV1, run_cap_key

logger = logging.getLogger("orion.physiology.run_cap_override")

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "physiology" / "run_cap_overrides.yaml"
FUTURE_SLACK = timedelta(seconds=60)

OBEYED = "obeyed"
IGNORED_REASONS = ("absent", "not_configured", "no_ttl", "malformed", "family_mismatch", "knob_mismatch",
                   "not_allowed", "future_set_at", "stale_set_at")


@dataclass(frozen=True)
class RunCapBound:
    family: str
    knob: str
    env_value: Any
    allowed: tuple
    max_hold_s: float


@dataclass(frozen=True)
class OverrideVerdict:
    """``active``: the controller must hold. ``reason``: ``obeyed`` or one of IGNORED_REASONS."""
    active: bool
    reason: str
    retry_after_sec: Optional[float] = None
    override: Optional[RunCapOverrideV1] = None

    @property
    def refused(self) -> bool:
        """A key existed and was NOT obeyed (counted as ``run_cap_override_refused``)."""
        return not self.active and self.reason not in ("absent", "not_configured")


def parse_bounds(doc: Mapping[str, Any] | None) -> dict[str, RunCapBound]:
    out: dict[str, RunCapBound] = {}
    for family, body in ((doc or {}).get("families") or {}).items():
        body = body or {}
        knob, allowed, max_hold = body.get("knob"), body.get("allowed"), body.get("max_hold_s")
        if not knob or not isinstance(allowed, list) or not allowed or not isinstance(max_hold, (int, float)) \
                or isinstance(max_hold, bool) or max_hold <= 0:
            raise ValueError(f"run_cap_overrides: {family} needs knob, a non-empty allowed list and max_hold_s > 0")
        out[str(family)] = RunCapBound(str(family), str(knob), body.get("env_value"), tuple(allowed), float(max_hold))
    return out


def load_bounds(path: str | Path | None = None) -> dict[str, RunCapBound]:
    """The bounds file. Missing or unreadable: {} (nothing is overridable; every key is ignored)."""
    p = Path(path) if path else DEFAULT_PATH
    try:
        return parse_bounds(yaml.safe_load(p.read_text(encoding="utf-8")))
    except FileNotFoundError:
        logger.warning("run_cap_overrides_missing path=%s (no family is overridable)", p)
        return {}
    except Exception as exc:  # noqa: BLE001 -- a bad file never makes the controller hold
        logger.warning("run_cap_overrides_unreadable path=%s err=%s: %s", p, type(exc).__name__, exc)
        return {}


def _text(v: Any) -> Optional[str]:
    if v is None:
        return None
    return v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v)


def judge(raw: Any, ttl_sec: Optional[float], now: datetime, family: str,
          bound: Optional[RunCapBound]) -> OverrideVerdict:
    """Pure. ``ttl_sec``: Redis TTL of the key (-1 no expiry, -2 / None gone)."""
    text = _text(raw)
    if text is None or ttl_sec is None or ttl_sec == -2:
        return OverrideVerdict(False, "absent")
    if bound is None:
        return OverrideVerdict(False, "not_configured")
    if ttl_sec <= 0:
        return OverrideVerdict(False, "no_ttl")
    try:
        ov = RunCapOverrideV1.model_validate(json.loads(text))
    except Exception:  # noqa: BLE001
        return OverrideVerdict(False, "malformed")
    if ov.family != family:
        return OverrideVerdict(False, "family_mismatch", override=ov)
    if ov.knob != bound.knob:
        return OverrideVerdict(False, "knob_mismatch", override=ov)
    if ov.value not in bound.allowed:
        return OverrideVerdict(False, "not_allowed", override=ov)
    if ov.set_at > now + FUTURE_SLACK:
        return OverrideVerdict(False, "future_set_at", override=ov)
    if (now - ov.set_at).total_seconds() > bound.max_hold_s:
        return OverrideVerdict(False, "stale_set_at", override=ov)
    return OverrideVerdict(True, OBEYED, retry_after_sec=float(max(1.0, ttl_sec)), override=ov)


async def read(redis: Any, family: str, bounds: Mapping[str, RunCapBound],
               clock: Optional[Callable[[], datetime]] = None) -> OverrideVerdict:
    """GET then TTL of the family's key, judged. The clock is read after both reads (#3134: an age is never
    measured against a clock older than the value it ages). Raises on Redis errors."""
    key = run_cap_key(family)
    raw = await redis.get(key)
    if raw is None:
        return OverrideVerdict(False, "absent")
    ttl = await redis.ttl(key)
    at = clock() if clock is not None else datetime.now(timezone.utc)
    return judge(raw, float(ttl) if ttl is not None else None, at, family, bounds.get(family))


__all__ = ["DEFAULT_PATH", "IGNORED_REASONS", "OBEYED", "OverrideVerdict", "RunCapBound", "judge", "load_bounds",
           "parse_bounds", "read"]
