from __future__ import annotations

from uuid import UUID

import pytest
from pydantic import ValidationError

from orion.cognition.github_compactor.constants import (
    CARD_SUMMARY_MAX_CHARS,
    JOURNAL_BODY_MAX_CHARS,
    JOURNAL_TITLE_MAX_CHARS,
)
from orion.cognition.github_compactor.digest import (
    assert_digest_within_budget,
    build_quiet_day_digest,
    parse_github_compactor_digest_json,
    stable_github_compactor_journal_entry_id,
    trim_github_compactor_input,
)
from orion.schemas.actions.github_compactor import GithubCompactorDigestV1


def test_github_compactor_digest_v1_rejects_empty_card_summary() -> None:
    with pytest.raises(ValidationError):
        GithubCompactorDigestV1(
            card_summary="",
            journal_title="Title",
            journal_body="Body",
            pr_refs=["#1"],
        )


def test_assert_digest_within_budget_accepts_in_limit() -> None:
    digest = GithubCompactorDigestV1(
        card_summary="a" * CARD_SUMMARY_MAX_CHARS,
        journal_title="b" * JOURNAL_TITLE_MAX_CHARS,
        journal_body="c" * JOURNAL_BODY_MAX_CHARS,
        pr_refs=["#1"],
    )
    assert_digest_within_budget(digest)


def test_assert_digest_within_budget_rejects_over_limit() -> None:
    digest = GithubCompactorDigestV1(
        card_summary="a" * (CARD_SUMMARY_MAX_CHARS + 1),
        journal_title="title",
        journal_body="body",
        pr_refs=[],
    )
    with pytest.raises(ValueError, match="compactor_output_over_budget"):
        assert_digest_within_budget(digest)


def test_build_quiet_day_digest() -> None:
    digest = build_quiet_day_digest(repo="acme/widgets", window_label="2026-07-08")
    assert "No merges" in digest.journal_body
    assert digest.pr_refs == []


def test_stable_github_compactor_journal_entry_id_is_deterministic() -> None:
    a = stable_github_compactor_journal_entry_id(
        workflow_id="github_compactor_pass",
        calendar_date="2026-07-08",
        repo="acme/widgets",
    )
    b = stable_github_compactor_journal_entry_id(
        workflow_id="github_compactor_pass",
        calendar_date="2026-07-08",
        repo="acme/widgets",
    )
    assert a == b
    UUID(a)


def test_parse_github_compactor_digest_json() -> None:
    raw = '{"card_summary":"Card","journal_title":"Title","journal_body":"Body","pr_refs":["#9"]}'
    digest = parse_github_compactor_digest_json(raw)
    assert digest.card_summary == "Card"
    assert digest.pr_refs == ["#9"]


def test_trim_github_compactor_input_caps_items_for_digest() -> None:
    payload = {
        "repo": "acme/widgets",
        "merged_pr_count": 20,
        "items": [
            {"number": i, "title": f"PR {i}", "body": "x" * 900, "touched_paths": ["a"] * 50}
            for i in range(20)
        ],
    }
    trimmed = trim_github_compactor_input(payload, max_items=12)
    assert len(trimmed["items"]) == 12
    assert trimmed["items"][0]["body"] == "x" * 600 + "…"
    assert "grouped_summary" not in trimmed
    assert trimmed["items_truncated_for_digest"] is True
    assert trimmed["items_total"] == 20
    assert trimmed["merged_pr_count"] == 20
