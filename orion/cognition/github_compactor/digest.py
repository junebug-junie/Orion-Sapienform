from __future__ import annotations

import json
from uuid import NAMESPACE_URL, uuid5

from orion.cognition.github_compactor.constants import (
    CARD_SUMMARY_MAX_CHARS,
    DIGEST_INPUT_BODY_MAX_CHARS,
    JOURNAL_BODY_MAX_CHARS,
    JOURNAL_TITLE_MAX_CHARS,
    MAX_DIGEST_INPUT_PRS,
)
from orion.schemas.actions.github_compactor import GithubCompactorDigestV1

_COMPACTOR_JOURNAL_ENTRY_NS = NAMESPACE_URL


def trim_github_compactor_input(fetch_payload: dict, *, max_items: int = MAX_DIGEST_INPUT_PRS) -> dict:
    """Bound digest LLM input size while preserving total merge count metadata."""
    if not isinstance(fetch_payload, dict):
        return {}
    trimmed = dict(fetch_payload)
    items = list(fetch_payload.get("items") or [])
    total = len(items)
    compact_items = []
    for item in items[:max_items]:
        if not isinstance(item, dict):
            continue
        body = str(item.get("body") or "").strip()
        if len(body) > DIGEST_INPUT_BODY_MAX_CHARS:
            body = body[:DIGEST_INPUT_BODY_MAX_CHARS].rstrip() + "…"
        compact_items.append(
            {
                "number": item.get("number"),
                "title": item.get("title"),
                "body": body,
                "merged_at": item.get("merged_at"),
                "url": item.get("url"),
            }
        )
    trimmed["items"] = compact_items
    if total > max_items:
        trimmed["items_truncated_for_digest"] = True
        trimmed["items_total"] = total
    trimmed.pop("grouped_summary", None)
    return trimmed


def assert_digest_within_budget(digest: GithubCompactorDigestV1) -> None:
    if len(digest.card_summary) > CARD_SUMMARY_MAX_CHARS:
        raise ValueError("compactor_output_over_budget:card_summary")
    if len(digest.journal_title) > JOURNAL_TITLE_MAX_CHARS:
        raise ValueError("compactor_output_over_budget:journal_title")
    if len(digest.journal_body) > JOURNAL_BODY_MAX_CHARS:
        raise ValueError("compactor_output_over_budget:journal_body")


def build_quiet_day_digest(*, repo: str, window_label: str) -> GithubCompactorDigestV1:
    repo_label = (repo or "unknown repo").strip()
    return GithubCompactorDigestV1(
        card_summary=f"No merged PRs in {window_label} for {repo_label}.",
        journal_title=f"Repo development digest — {window_label}",
        journal_body=(
            f"No merges were found for {repo_label} during {window_label}. "
            "Previous repo development snapshot card was left unchanged."
        ),
        pr_refs=[],
    )


def parse_github_compactor_digest_json(raw: str) -> GithubCompactorDigestV1:
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("compactor_digest_not_object")
    return GithubCompactorDigestV1.model_validate(payload)


def stable_github_compactor_journal_entry_id(
    *,
    workflow_id: str,
    calendar_date: str,
    repo: str,
) -> str:
    payload = "|".join([workflow_id.strip(), calendar_date.strip(), repo.strip()])
    return str(uuid5(_COMPACTOR_JOURNAL_ENTRY_NS, payload))
