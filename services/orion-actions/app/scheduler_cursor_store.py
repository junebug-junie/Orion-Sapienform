from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from threading import RLock
from typing import Dict

logger = logging.getLogger("orion-actions.scheduler_cursor")

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def scheduler_cursor_completed_local_date(
    *, forced_date: str | None, window_request_date: str, scheduled_local_date: str
) -> str:
    """Persisted cursor date: forced-window label when ACTIONS_DAILY_RUN_ONCE_DATE is set, else scheduler 'today'."""
    if forced_date is not None and str(forced_date).strip():
        return window_request_date
    return scheduled_local_date


def resolve_scheduler_cursor_store_path(path: str | None, *, workflow_schedule_store_path: str) -> Path:
    raw = str(path or "").strip()
    if raw:
        candidate = Path(raw).expanduser()
        if candidate.exists() and candidate.is_dir():
            return candidate / "scheduler_cursors.json"
        if raw.endswith("/"):
            return candidate / "scheduler_cursors.json"
        return candidate
    wf = Path(workflow_schedule_store_path or "").expanduser()
    parent = wf.parent if wf.parent.parts else Path("/tmp/orion-actions")
    return parent / "scheduler_cursors.json"


class SchedulerCursorStore:
    """Durable last-completed local calendar dates for built-in daily scheduler jobs."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._cursors: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text() or "{}")
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "scheduler_cursor_store_load_failed path=%s error=%s",
                self._path,
                exc.__class__.__name__,
            )
            return
        if not isinstance(raw, dict):
            return
        out: Dict[str, str] = {}
        for k, v in raw.items():
            if not isinstance(k, str) or not isinstance(v, str):
                continue
            if not _ISO_DATE.match(v.strip()):
                continue
            out[k.strip()] = v.strip()
        self._cursors = out

    def _persist(self) -> None:
        data = dict(sorted(self._cursors.items()))
        temp = self._path.with_suffix(".tmp")
        temp.write_text(json.dumps(data, indent=2, sort_keys=True))
        temp.replace(self._path)

    def get(self, job_key: str) -> str | None:
        with self._lock:
            v = self._cursors.get(job_key)
            return v

    def all(self) -> Dict[str, str]:
        with self._lock:
            return dict(self._cursors)

    def set_last_completed(self, job_key: str, local_date: str) -> None:
        if not _ISO_DATE.match(local_date):
            return
        with self._lock:
            self._cursors[job_key] = local_date
            self._persist()
