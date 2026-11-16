"""The one HTTP client for Orion's now (integration patch 3): read it, and record what a run cited.

Spec: docs/superpowers/specs/2026-11-15-orion-integration-and-a-life-design.md, Part A (A3, A4). The routes are
orion-durable-runs ``GET /temporal-self/now`` and ``POST /temporal-self/now/use``; the pure rules are
``orion.temporal_self.now``. Readers: Prime (services/orion-mind/app/prime_loop.py), reveries
(services/orion-thought/app/temporal_self_cue.py) and curiosity (services/orion-hub/scripts/curiosity_now.py).

Rules every reader gets from here, so no reader can forget one:

* **Never raises, short timeout.** A failed read is ``NowRead(now=None, error=<why>)``: the organ goes on with
  "no now" and logs the reason. A thought, a reverie, a curiosity pick or a chat turn never waits on it past
  ``timeout_sec``.
* **A real run reads without ``since``.** That is what moves the organ's cursor (``experience_shown``
  ``advances_cursor``). ``since`` is only for an inspection read (probes, evals); it never moves the cursor.
* **No cites, no call.** A run that read the now and cited nothing posts nothing: the shown row with no use
  rows is the record of "read and cited nothing".

Stdlib only (urllib), so every service image can use it; async callers use ``aread_now`` / ``arecord_use``.
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Iterable, Optional

from orion.schemas.temporal_self import ExperienceUseResultV1, TemporalNowV1

NOW_PATH = "/temporal-self/now"
USE_PATH = "/temporal-self/now/use"
DEFAULT_TIMEOUT_SEC = 3.0

Getter = Callable[[str, float], Any]
Poster = Callable[[str, dict[str, Any], float], Any]

__all__ = ["DEFAULT_TIMEOUT_SEC", "NOW_PATH", "NowRead", "USE_PATH", "aread_now", "arecord_use", "read_now",
           "record_use"]


@dataclass(frozen=True)
class NowRead:
    """What one read gave. ``now`` None: there is no now this run, and ``error`` says why."""

    now: Optional[TemporalNowV1]
    error: Optional[str] = None

    @property
    def shown_ids(self) -> list[str]:
        return [i.event_id for i in self.now.items] if self.now is not None else []


def _http_get(url: str, timeout: float) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 -- internal service URL
        return json.loads(resp.read())


def _http_post(url: str, body: dict[str, Any], timeout: float) -> Any:
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 -- internal service URL
        return json.loads(resp.read())


def _why(exc: BaseException) -> str:
    """A short, loggable reason. HTTP errors keep the status (404 = the now is switched off)."""
    if isinstance(exc, urllib.error.HTTPError):
        return f"http {exc.code}"
    reason = getattr(exc, "reason", None)
    if isinstance(exc, TimeoutError) or isinstance(reason, TimeoutError):
        return "timeout"
    return f"{type(exc).__name__}: {str(exc)[:120]}"


def read_now(base_url: str, organ: str, *, run_ref: Optional[str] = None, since: Optional[datetime] = None,
             timeout_sec: float = DEFAULT_TIMEOUT_SEC, get: Getter = _http_get) -> NowRead:
    """GET the now for ``organ``. ``run_ref``: the reading run's own id, stored on the shown row.
    ``since``: inspection reads only (it never moves the organ's cursor). Never raises."""
    if not (base_url or "").strip():
        return NowRead(None, "no durable-runs URL configured")
    params = {"organ": organ}
    if run_ref:
        params["run_ref"] = run_ref[:300]
    if since is not None:
        params["since"] = since.isoformat()
    url = f"{base_url.rstrip('/')}{NOW_PATH}?{urllib.parse.urlencode(params)}"
    try:
        body = get(url, timeout_sec)
    except Exception as exc:  # noqa: BLE001 -- a reader never fails its organ
        return NowRead(None, _why(exc))
    try:
        return NowRead(TemporalNowV1.model_validate(body))
    except Exception as exc:  # noqa: BLE001
        return NowRead(None, f"not a TemporalNowV1 ({type(exc).__name__})")


def record_use(base_url: str, shown_id: str, cites: Iterable[str], citing_ref: str, *,
               timeout_sec: float = DEFAULT_TIMEOUT_SEC,
               post: Poster = _http_post) -> tuple[Optional[ExperienceUseResultV1], Optional[str]]:
    """POST a run's cites against the shown row it read. ``(None, None)`` when there was nothing to cite
    (no call is made). Durable-runs checks each cite against the shown row (at most 3 per row). Never raises."""
    ids = [str(c) for c in dict.fromkeys(cites) if c]
    if not ids:
        return None, None
    if not (base_url or "").strip() or not shown_id or not citing_ref:
        return None, "missing durable-runs URL, shown_id or citing_ref"
    body = {"shown_id": shown_id, "citing_ref": citing_ref[:300], "cites": ids[:32]}
    try:
        data = post(f"{base_url.rstrip('/')}{USE_PATH}", body, timeout_sec)
    except Exception as exc:  # noqa: BLE001
        return None, _why(exc)
    try:
        return ExperienceUseResultV1.model_validate(data), None
    except Exception as exc:  # noqa: BLE001
        return None, f"not an ExperienceUseResultV1 ({type(exc).__name__})"


async def aread_now(base_url: str, organ: str, *, run_ref: Optional[str] = None, since: Optional[datetime] = None,
                    timeout_sec: float = DEFAULT_TIMEOUT_SEC, get: Getter = _http_get) -> NowRead:
    """``read_now`` off the event loop, bounded as a whole (DNS included). Never raises."""
    try:
        return await asyncio.wait_for(asyncio.to_thread(read_now, base_url, organ, run_ref=run_ref, since=since,
                                                        timeout_sec=timeout_sec, get=get), timeout_sec + 0.5)
    except asyncio.TimeoutError:
        return NowRead(None, "timeout")
    except Exception as exc:  # noqa: BLE001
        return NowRead(None, _why(exc))


async def arecord_use(base_url: str, shown_id: str, cites: Iterable[str], citing_ref: str, *,
                      timeout_sec: float = DEFAULT_TIMEOUT_SEC,
                      post: Poster = _http_post) -> tuple[Optional[ExperienceUseResultV1], Optional[str]]:
    """``record_use`` off the event loop, bounded as a whole. Never raises."""
    cites = list(cites)
    try:
        return await asyncio.wait_for(asyncio.to_thread(record_use, base_url, shown_id, cites, citing_ref,
                                                        timeout_sec=timeout_sec, post=post), timeout_sec + 0.5)
    except asyncio.TimeoutError:
        return None, "timeout"
    except Exception as exc:  # noqa: BLE001
        return None, _why(exc)
