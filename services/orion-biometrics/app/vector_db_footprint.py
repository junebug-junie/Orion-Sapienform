"""Vector DB (Chroma) footprint gauge, piggybacked onto the SystemHealthV1 heartbeat.

Publishes ``details.vector_db`` on this node's heartbeat:

- ``dir_bytes`` / ``sqlite_bytes``: on-disk size of Chroma's persist directory and of its
  ``chroma.sqlite3`` (read through the existing read-only ``/host_mnt/graphdb`` mount).
- ``alert_bytes`` / ``over_alert``: the growth alarm. Crossing it also logs a WARNING once
  per crossing.
- ``collection_counts``: rows per collection from Chroma's documented HTTP API.
- ``unpoliced_collections``: live collections with no entry in
  ``orion/memory/vector_collection_policy.py``.

Measure only, never delete: per that policy no collection qualifies for retention.

Runs as a slow background poller (like ``IloPoller``) because a directory walk plus
~10 HTTP calls does not belong in the heartbeat's bounded per-tick budget. A node whose
Chroma data dir is not mounted (atlas, circe) reports ``error="not_present"`` and makes
no HTTP calls.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from orion.memory.vector_collection_policy import unpoliced

logger = logging.getLogger("orion-biometrics.vector_db")


def dir_size_bytes(path: str) -> int:
    """Sum of apparent file sizes under ``path`` (no symlink following)."""
    total = 0
    stack = [path]
    while stack:
        current = stack.pop()
        with os.scandir(current) as it:
            for entry in it:
                try:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(entry.path)
                    elif entry.is_file(follow_symlinks=False):
                        total += entry.stat(follow_symlinks=False).st_size
                except FileNotFoundError:
                    # Chroma can rotate a file between scandir and stat.
                    continue
    return total


def _get_json(url: str, timeout: float) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 -- internal URL from settings
        return json.loads(resp.read().decode("utf-8"))


def fetch_collection_counts(base_url: str, timeout: float) -> Dict[str, int]:
    """Row count per collection via Chroma's v1 HTTP API (0.4.x server)."""
    base = base_url.rstrip("/")
    collections = _get_json(f"{base}/api/v1/collections", timeout)
    counts: Dict[str, int] = {}
    for col in collections or []:
        name = str(col.get("name") or "")
        cid = str(col.get("id") or "")
        if not name or not cid:
            continue
        counts[name] = int(_get_json(f"{base}/api/v1/collections/{cid}/count", timeout))
    return counts


@dataclass
class VectorDbFootprint:
    dir_bytes: Optional[int] = None
    sqlite_bytes: Optional[int] = None
    alert_bytes: Optional[int] = None
    over_alert: bool = False
    collection_counts: Dict[str, int] = field(default_factory=dict)
    unpoliced_collections: List[str] = field(default_factory=list)
    error: Optional[str] = None
    sampled_at: Optional[str] = None

    def as_details(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "dir_bytes": self.dir_bytes,
            "sqlite_bytes": self.sqlite_bytes,
            "alert_bytes": self.alert_bytes,
            "over_alert": self.over_alert,
            "collection_counts": dict(self.collection_counts),
            "unpoliced_collections": list(self.unpoliced_collections),
            "sampled_at": self.sampled_at,
        }
        if self.error:
            out["error"] = self.error
        return out


def sample_footprint(data_path: str, base_url: str, alert_bytes: int, timeout: float) -> VectorDbFootprint:
    snap = VectorDbFootprint(alert_bytes=alert_bytes, sampled_at=datetime.now(timezone.utc).isoformat())
    if not data_path or not os.path.isdir(data_path):
        snap.error = "not_present"
        return snap
    try:
        snap.dir_bytes = dir_size_bytes(data_path)
        sqlite_path = os.path.join(data_path, "chroma.sqlite3")
        if os.path.isfile(sqlite_path):
            snap.sqlite_bytes = os.path.getsize(sqlite_path)
    except Exception as exc:  # disk read failure must not kill the heartbeat
        snap.error = f"dir_size_failed: {exc}"
        return snap
    snap.over_alert = bool(alert_bytes > 0 and snap.dir_bytes >= alert_bytes)
    if base_url:
        try:
            snap.collection_counts = fetch_collection_counts(base_url, timeout)
            snap.unpoliced_collections = unpoliced(list(snap.collection_counts))
        except Exception as exc:
            # Size is still real without counts; keep it and say why counts are missing.
            snap.error = f"collection_counts_failed: {type(exc).__name__}: {exc}"
    return snap


class VectorDbFootprintPoller:
    def __init__(
        self,
        *,
        enabled: bool,
        data_path: str,
        base_url: str,
        alert_bytes: int,
        interval_sec: float = 300.0,
        timeout_sec: float = 5.0,
    ) -> None:
        self._enabled = enabled
        self._data_path = data_path
        self._base_url = base_url
        self._alert_bytes = alert_bytes
        self._interval_sec = interval_sec
        self._timeout_sec = timeout_sec
        self._snapshot = VectorDbFootprint(error=None if enabled else "disabled")
        self._was_over = False
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def details(self) -> Dict[str, Any]:
        if not self._enabled:
            return {}
        return {"vector_db": self._snapshot.as_details()}

    def poll_once(self) -> VectorDbFootprint:
        snap = sample_footprint(self._data_path, self._base_url, self._alert_bytes, self._timeout_sec)
        if snap.over_alert and not self._was_over:
            logger.warning(
                "vector_db_over_alert dir_bytes=%s alert_bytes=%s path=%s counts=%s",
                snap.dir_bytes, snap.alert_bytes, self._data_path, snap.collection_counts,
            )
        elif self._was_over and not snap.over_alert and snap.dir_bytes is not None:
            logger.info("vector_db_back_under_alert dir_bytes=%s alert_bytes=%s", snap.dir_bytes, snap.alert_bytes)
        if snap.dir_bytes is not None:
            self._was_over = snap.over_alert
        if snap.unpoliced_collections:
            logger.warning("vector_db_unpoliced_collections names=%s", snap.unpoliced_collections)
        self._snapshot = snap
        return snap

    async def start_background(self) -> None:
        if not self._enabled or self._task is not None:
            return
        self._task = asyncio.create_task(self._poll_loop(), name="vector-db-footprint-poller")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=2.0)
            except Exception:
                self._task.cancel()
            self._task = None

    async def _poll_loop(self) -> None:
        # Poll first, then sleep: a crash loop must still produce a reading.
        while not self._stop.is_set():
            try:
                await asyncio.to_thread(self.poll_once)
            except Exception as exc:
                logger.warning("vector_db_footprint_poll_failed error=%s", exc)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._interval_sec)
            except asyncio.TimeoutError:
                pass
