"""Regression: the degraded-classify retry must baseline against the turn
immediately before the one being retried, not the session's 20th-oldest turn."""

import importlib.util
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SERVICE_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load(rel_path: str, name: str):
    for key in list(sys.modules):
        if key == "app" or key.startswith("app."):
            del sys.modules[key]
    sys.path.insert(0, str(SERVICE_ROOT))
    path = SERVICE_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


retry_mod = _load("app/retry_degraded_classifies.py", "memory_consolidation_retry_degraded")
classify_mod = sys.modules["app.classify"]

_T0 = datetime(2026, 9, 1, tzinfo=timezone.utc)


def _row(i: int, *, session_id: str = "orion_journal") -> dict:
    return {
        "correlation_id": f"c{i:02d}",
        "prompt": f"prompt {i}",
        "response": f"response {i}",
        "spark_meta": "{}",
        "session_id": session_id,
        "created_at": _T0 + timedelta(minutes=i),
    }


class _ChatHistoryPool:
    """Enough of an asyncpg pool to run ``_prior_turns_for`` against an
    in-memory chat_history_log. It honours the query's own ORDER BY direction
    and LIMIT, so the sort direction written in the SQL decides which rows come
    back -- the exact thing this regression is about."""

    def __init__(self, rows: list[dict]):
        self.rows = rows

    async def fetch(self, query: str, session_id: str, created_at: datetime):
        matched = [
            r for r in self.rows if r["session_id"] == session_id and r["created_at"] < created_at
        ]
        direction = re.search(r"ORDER BY created_at (ASC|DESC)", query).group(1)
        matched.sort(key=lambda r: r["created_at"], reverse=direction == "DESC")
        limit = int(re.search(r"LIMIT (\d+)", query).group(1))
        return matched[:limit]


@pytest.mark.asyncio
async def test_prior_turns_are_the_most_recent_twenty_oldest_first():
    rows = [_row(i) for i in range(30)] + [_row(99, session_id="other_session")]
    pool = _ChatHistoryPool(rows)

    prior = await retry_mod._prior_turns_for(
        pool, session_id="orion_journal", created_at=_T0 + timedelta(minutes=30)
    )

    assert [t["correlation_id"] for t in prior] == [f"c{i:02d}" for i in range(10, 30)]


@pytest.mark.asyncio
async def test_retry_baseline_is_the_turn_just_before_the_retried_one(monkeypatch):
    """Old query (ASC LIMIT 20) made the baseline c19 for a 31st turn."""
    history = [_row(i) for i in range(30)]
    retried = _row(30)
    retried["spark_meta"] = '{"turn_change_appraisal": {"turn_change_status": "degraded"}}'
    pool = _ChatHistoryPool(history)
    pool_fetch = pool.fetch

    async def fetch(query: str, *args):
        if not args:  # the retry candidate scan
            return [retried]
        return await pool_fetch(query, *args)

    pool.fetch = fetch
    seen: dict = {}

    async def fake_classify_turn(bus, *, turn, prior_turns, settings):
        seen["baseline"] = classify_mod._prior_turn_baseline(prior_turns)
        return {"memory_classify_status": "ok"}

    monkeypatch.setattr(retry_mod, "classify_turn", fake_classify_turn)
    monkeypatch.setattr(retry_mod, "publish_spark_meta_patch", AsyncMock())
    monkeypatch.setattr(retry_mod, "_retry_counts", {})

    await retry_mod.retry_degraded_classifies(
        pool=pool, bus=AsyncMock(), window_store=AsyncMock(), suggest_runner=AsyncMock()
    )

    mode, text, prior_corr = seen["baseline"]
    assert mode == "prior_turn"
    assert prior_corr == "c29"
    assert "prompt 29" in text
