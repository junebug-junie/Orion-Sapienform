from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Awaitable, Callable

from orion.autonomy.action_outcomes import append_action_outcome
from orion.autonomy.models import ActionOutcomeRefV1, FetchedArticleRefV1
from orion.autonomy.salience import score_article_salience

_FETCH_KIND = "web.fetch.readonly"


@dataclass(frozen=True)
class EpisodeFetchRequest:
    subject: str
    goal_artifact_id: str
    spawned_correlation_id: str
    query: str
    max_articles: int = 2
    gap_terms: tuple[str, ...] = ()


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


def _parse_articles(result: dict, *, gap_terms: set[str]) -> list[FetchedArticleRefV1]:
    """Build scored article refs from a backend result.

    Prefers the rich `articles` list; falls back to bare `urls` for older backends.
    """
    parsed: list[FetchedArticleRefV1] = []
    raw = result.get("articles")
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            title = str(item.get("title") or "")
            description = str(item.get("description") or "")
            salience = score_article_salience(f"{title} {description}", gap_terms)
            parsed.append(
                FetchedArticleRefV1(url=url, title=title, description=description, salience=salience)
            )
    if parsed:
        return parsed
    urls = result.get("urls")
    if isinstance(urls, list):
        for candidate in urls:
            url = str(candidate or "").strip()
            if url:
                parsed.append(FetchedArticleRefV1(url=url))
    return parsed


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
    articles: list[FetchedArticleRefV1] = []
    aggregate_salience = 0.0

    try:
        result = await fetch_backend(req.query, max_articles=req.max_articles)
        if not isinstance(result, dict):
            result = {}
        success = bool(result.get("success"))
        summary = _build_summary(result)
        surprise = 0.0 if success else 1.0
        articles = _parse_articles(result, gap_terms=set(req.gap_terms))
        aggregate_salience = max((a.salience for a in articles), default=0.0)
    except Exception as exc:
        summary = f"fetch failed: {exc}"

    outcome = ActionOutcomeRefV1(
        action_id=action_id,
        kind=_FETCH_KIND,
        summary=summary,
        success=success,
        surprise=surprise,
        observed_at=observed_at,
        query=req.query,
        articles=articles,
        salience=aggregate_salience,
    )
    append_action_outcome(subject=req.subject, outcome=outcome)
    return outcome
