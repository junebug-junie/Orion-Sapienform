from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from pathlib import Path

from orion.autonomy.models import ActionOutcomeRefV1

logger = logging.getLogger("orion.autonomy.action_outcomes")

DEFAULT_STORE_PATH = "/tmp/orion-action-outcomes.json"
_MAX_OUTCOMES = 12

# Process-level engine cache keyed by database URL (SQLAlchemy engines are pooled).
_ENGINE_CACHE: dict[str, object] = {}


def _store_path() -> Path:
    return Path(os.getenv("ORION_ACTION_OUTCOME_STORE_PATH", DEFAULT_STORE_PATH))


def _db_url() -> str | None:
    url = os.getenv("ORION_ACTION_OUTCOME_DB_URL", "").strip()
    return url or None


def _get_engine(url: str):
    from sqlalchemy import create_engine

    engine = _ENGINE_CACHE.get(url)
    if engine is None:
        # setdefault keeps the cache race-free if two callers build concurrently:
        # the first inserted engine wins and any extra is discarded.
        engine = _ENGINE_CACHE.setdefault(url, create_engine(url, pool_pre_ping=True))
    return engine


def _load_from_sql(subject: str) -> list[ActionOutcomeRefV1]:
    """Read the most recent outcomes for a subject from the shared SQL store.

    Returns chronological order (oldest first) to match the file-backed store.
    """
    from sqlalchemy import text

    url = _db_url()
    if not url:
        raise RuntimeError("ORION_ACTION_OUTCOME_DB_URL is not set")
    engine = _get_engine(url)
    query = text(
        """
        SELECT action_id, kind, summary, success, surprise, observed_at
        FROM action_outcomes
        WHERE subject = :subject
        ORDER BY observed_at DESC NULLS LAST, created_at DESC
        LIMIT :limit
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(query, {"subject": subject, "limit": _MAX_OUTCOMES}).mappings().all()

    out: list[ActionOutcomeRefV1] = []
    for row in reversed(rows):  # oldest-first, matching append semantics
        try:
            out.append(
                ActionOutcomeRefV1(
                    action_id=row["action_id"],
                    kind=row["kind"],
                    summary=row["summary"],
                    success=row["success"],
                    surprise=row["surprise"] if row["surprise"] is not None else 0.0,
                    observed_at=row["observed_at"],
                )
            )
        except Exception:
            continue
    return out


@contextmanager
def _store_lock(path: Path):
    """Exclusive lock for read-modify-write on the outcome store (best-effort)."""
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):
            pass
        yield
    finally:
        try:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except (ImportError, OSError):
            pass
        lock_file.close()


def _load_raw() -> dict[str, list[dict]]:
    path = _store_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): v for k, v in data.items() if isinstance(v, list)}


def _save_raw(data: dict[str, list[dict]]) -> None:
    path = _store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    tmp.replace(path)


def load_action_outcomes(subject: str) -> list[ActionOutcomeRefV1]:
    if _db_url():
        try:
            return _load_from_sql(subject)
        except Exception as exc:
            logger.warning(
                "action_outcome_sql_read_failed subject=%s error=%s; falling back to file store",
                subject,
                exc,
            )
    raw = _load_raw()
    bucket = raw.get(subject, [])
    out: list[ActionOutcomeRefV1] = []
    for item in bucket:
        if not isinstance(item, dict):
            continue
        try:
            out.append(ActionOutcomeRefV1.model_validate(item))
        except Exception:
            continue
    return out


def append_action_outcome(subject: str, outcome: ActionOutcomeRefV1) -> None:
    path = _store_path()
    with _store_lock(path):
        data = _load_raw()
        bucket = data.get(subject, [])
        if not isinstance(bucket, list):
            bucket = []
        bucket.append(outcome.model_dump(mode="json"))
        data[subject] = bucket[-_MAX_OUTCOMES:]
        _save_raw(data)
