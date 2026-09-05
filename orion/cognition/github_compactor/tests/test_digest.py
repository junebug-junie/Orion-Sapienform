from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest
import yaml
from pydantic import ValidationError

from orion.cognition.github_compactor.constants import (
    CARD_SUMMARY_MAX_CHARS,
    DIGEST_INPUT_BODY_MAX_CHARS,
    DIGEST_ORCH_RPC_TIMEOUT_SEC,
    DIGEST_VERB_TIMEOUT_MS,
    JOURNAL_BODY_MAX_CHARS,
    JOURNAL_TITLE_MAX_CHARS,
    MAX_DIGEST_INPUT_PRS,
)
from orion.cognition.github_compactor.digest import (
    fit_digest_within_budget,
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


def test_fit_digest_within_budget_returns_in_limit_digest_unchanged() -> None:
    digest = GithubCompactorDigestV1(
        card_summary="a" * CARD_SUMMARY_MAX_CHARS,
        journal_title="b" * JOURNAL_TITLE_MAX_CHARS,
        journal_body="c" * JOURNAL_BODY_MAX_CHARS,
        pr_refs=["#1"],
    )
    fitted, trimmed = fit_digest_within_budget(digest)
    assert trimmed == []
    assert fitted is digest


def test_fit_digest_within_budget_repairs_over_limit_instead_of_raising() -> None:
    """An over-long card_summary must not fail the workflow.

    This is the exact live failure mode: 5 `compactor_output_over_budget:card_summary`
    failures on github_compactor_pass (2026-08-27, 2026-08-30), each discarding a
    complete digest and feeding the scheduler's retry path.
    """
    digest = GithubCompactorDigestV1(
        card_summary="a" * (CARD_SUMMARY_MAX_CHARS + 1),
        journal_title="title",
        journal_body="body",
        pr_refs=["#1"],
    )
    fitted, trimmed = fit_digest_within_budget(digest)
    assert trimmed == ["card_summary"]
    assert len(fitted.card_summary) == CARD_SUMMARY_MAX_CHARS
    # Untouched fields survive, and so does non-prose content.
    assert fitted.journal_title == "title"
    assert fitted.journal_body == "body"
    assert fitted.pr_refs == ["#1"]


def test_fit_digest_within_budget_trims_each_over_limit_field() -> None:
    digest = GithubCompactorDigestV1(
        card_summary="a" * (CARD_SUMMARY_MAX_CHARS + 50),
        journal_title="b" * (JOURNAL_TITLE_MAX_CHARS + 50),
        journal_body="c" * (JOURNAL_BODY_MAX_CHARS + 50),
        pr_refs=[],
    )
    fitted, trimmed = fit_digest_within_budget(digest)
    assert trimmed == ["card_summary", "journal_body", "journal_title"]
    assert len(fitted.card_summary) == CARD_SUMMARY_MAX_CHARS
    assert len(fitted.journal_title) == JOURNAL_TITLE_MAX_CHARS
    assert len(fitted.journal_body) == JOURNAL_BODY_MAX_CHARS


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


def test_digest_budget_supports_high_volume_merge_days() -> None:
    """~30 merges/day must fit the digest input + journal body + wall-clock budget.

    Live failure 2026-09-05: timeout_ms=90000 + MAX_DIGEST_INPUT_PRS=8 meant the
    8k-token digest timed out empty (invalid_json) while most daily PRs never
    reached the LLM. Chat can wait; the digest must finish.
    """
    assert MAX_DIGEST_INPUT_PRS >= 30
    assert JOURNAL_BODY_MAX_CHARS >= 8000
    assert DIGEST_VERB_TIMEOUT_MS >= 600_000
    assert DIGEST_ORCH_RPC_TIMEOUT_SEC >= DIGEST_VERB_TIMEOUT_MS / 1000.0

    verb_path = (
        Path(__file__).resolve().parents[3]
        / "cognition"
        / "verbs"
        / "github_compactor_digest_v1.yaml"
    )
    verb = yaml.safe_load(verb_path.read_text(encoding="utf-8"))
    assert int(verb["timeout_ms"]) == DIGEST_VERB_TIMEOUT_MS
    assert int(verb["timeout_ms"]) >= 600_000

    prompt_path = (
        Path(__file__).resolve().parents[3]
        / "cognition"
        / "prompts"
        / "github_compactor_digest_v1.j2"
    )
    prompt = prompt_path.read_text(encoding="utf-8")
    assert f"(max {JOURNAL_BODY_MAX_CHARS} chars)" in prompt
    assert f"(max {CARD_SUMMARY_MAX_CHARS} chars)" in prompt
    assert f"(max {JOURNAL_TITLE_MAX_CHARS} chars)" in prompt


def test_trim_github_compactor_input_caps_items_for_digest() -> None:
    payload = {
        "repo": "acme/widgets",
        "merged_pr_count": 20,
        "items": [
            {"number": i, "title": f"PR {i}", "body": "x" * (DIGEST_INPUT_BODY_MAX_CHARS + 300), "touched_paths": ["a"] * 50}
            for i in range(20)
        ],
    }
    trimmed = trim_github_compactor_input(payload, max_items=12)
    assert len(trimmed["items"]) == 12
    assert trimmed["items"][0]["body"] == "x" * DIGEST_INPUT_BODY_MAX_CHARS + "…"
    assert trimmed["items"][0]["truncated"] is True
    assert trimmed["item_content_truncated"] is True
    assert "grouped_summary" not in trimmed
    assert trimmed["items_truncated_for_digest"] is True
    assert trimmed["items_total"] == 20
    assert trimmed["merged_pr_count"] == 20


def test_trim_github_compactor_input_default_keeps_high_volume_day() -> None:
    payload = {
        "repo": "acme/widgets",
        "merged_pr_count": 35,
        "items": [{"number": i, "title": f"PR {i}", "body": "ok"} for i in range(35)],
    }
    trimmed = trim_github_compactor_input(payload)
    assert len(trimmed["items"]) == MAX_DIGEST_INPUT_PRS
    assert trimmed["items_total"] == 35
    assert trimmed["merged_pr_count"] == 35
    assert trimmed["items_truncated_for_digest"] is True


def test_trim_github_compactor_input_preserves_short_bodies_untruncated() -> None:
    payload = {
        "repo": "acme/widgets",
        "items": [{"number": 1, "title": "Short PR", "body": "A short body, well within budget."}],
    }
    trimmed = trim_github_compactor_input(payload)
    assert "truncated" not in trimmed["items"][0]
    assert "item_content_truncated" not in trimmed
