from __future__ import annotations

import json
from uuid import NAMESPACE_URL, uuid5

from orion.cognition.github_compactor.constants import (
    CARD_SUMMARY_MAX_CHARS,
    JOURNAL_BODY_MAX_CHARS,
    JOURNAL_TITLE_MAX_CHARS,
)
from orion.schemas.actions.github_compactor import GithubCompactorDigestV1

_COMPACTOR_JOURNAL_ENTRY_NS = NAMESPACE_URL


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
