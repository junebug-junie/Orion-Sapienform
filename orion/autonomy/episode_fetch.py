from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Awaitable, Callable

from orion.autonomy.action_outcomes import append_action_outcome
from orion.autonomy.models import ActionOutcomeRefV1

_FETCH_KIND = "web.fetch.readonly"


@dataclass(frozen=True)
class EpisodeFetchRequest:
    subject: str
    goal_artifact_id: str
    spawned_correlation_id: str
    query: str
    max_articles: int = 2


async def default_fetch_backend(query: str, *, max_articles: int) -> dict:
    raise RuntimeError("episode_fetch_backend_not_configured")


def _build_summary(result: dict) -> str:
    urls = result.get("urls") or []
    if isinstance(urls, list) and urls:
        return f"fetched {len(urls)} article(s)"
    error = result.get("error")
    if error:
        return f"fetch failed: {error}"
    return "fetch returned no articles"


async def execute_readonly_fetch(
    req: EpisodeFetchRequest,
    *,
    fetch_backend: Callable[..., Awaitable[dict]] = default_fetch_backend,
) -> ActionOutcomeRefV1:
    action_id = f"fetch-{req.spawned_correlation_id}-{uuid.uuid4().hex[:8]}"
    observed_at = datetime.now(timezone.utc)
    success = False
    surprise = 1.0
    summary = "fetch failed"

    try:
        result = await fetch_backend(req.query, max_articles=req.max_articles)
        if not isinstance(result, dict):
            result = {}
        success = bool(result.get("success"))
        summary = _build_summary(result)
        surprise = 0.0 if success else 1.0
    except Exception as exc:
        summary = f"fetch failed: {exc}"

    outcome = ActionOutcomeRefV1(
        action_id=action_id,
        kind=_FETCH_KIND,
        summary=summary,
        success=success,
        surprise=surprise,
        observed_at=observed_at,
    )
    append_action_outcome(subject=req.subject, outcome=outcome)
    return outcome
