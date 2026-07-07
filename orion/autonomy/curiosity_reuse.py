from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

from orion.autonomy.models import ActionOutcomeRefV1, FetchedArticleRefV1
from orion.autonomy.salience import iter_gap_section_labels
from orion.core.schemas.frontier_curiosity import FrontierInvocationSignalV1
from orion.schemas.world_pulse import CuriosityFollowupV1

_FETCH_KIND = "web.fetch.readonly"


def select_reusable_followup(
    followups: Sequence[CuriosityFollowupV1],
    curiosity_signals: Sequence[FrontierInvocationSignalV1],
) -> CuriosityFollowupV1 | None:
    """Return the followup whose section matches the first gap-section label the
    reactive loop would act on, and only if it actually carries articles.

    Matching mirrors `build_readonly_fetch_query` (first `iter_gap_section_labels`).
    `iter_gap_section_labels` strips the `section:` prefix and turns underscores
    into spaces, so `followup.section` is normalized the same way before compare.
    """
    labels = list(iter_gap_section_labels(curiosity_signals))
    if not labels or not followups:
        return None
    wanted = labels[0].strip().lower()
    for followup in followups:
        if followup.section.replace("_", " ").strip().lower() == wanted and followup.articles:
            return followup
    return None


def outcome_from_followup(followup: CuriosityFollowupV1, *, run_id: str) -> ActionOutcomeRefV1:
    """Rebuild an ActionOutcomeRefV1 from a world-pulse curiosity followup so the
    reactive episode-journal path can reuse it without a second fetch."""
    articles = [
        FetchedArticleRefV1(
            url=a.url, title=a.title, description=a.description, salience=a.salience
        )
        for a in followup.articles
    ]
    return ActionOutcomeRefV1(
        action_id=followup.action_id or f"wp-followup-{run_id}",
        kind=_FETCH_KIND,
        summary=f"reused {len(articles)} article(s) from world-pulse gap fetch",
        success=True,
        surprise=0.0,
        observed_at=datetime.now(timezone.utc),
        query=followup.query,
        articles=articles,
        salience=max((a.salience for a in articles), default=0.0),
    )
