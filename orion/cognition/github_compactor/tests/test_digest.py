from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import pytest
import yaml
from pydantic import ValidationError

from orion.cognition.compactor.constants import DIGEST_INPUT_CHAR_BUDGET
from orion.cognition.github_compactor.constants import (
    CARD_SUMMARY_MAX_CHARS,
    DIGEST_ORCH_RPC_TIMEOUT_SEC,
    DIGEST_VERB_TIMEOUT_MS,
    JOURNAL_TITLE_MAX_CHARS,
    PR_BODY_MAX_CHARS,
)
from orion.cognition.github_compactor.digest import (
    build_github_compactor_digest_inputs,
    build_github_compactor_merge_input,
    concatenate_github_partial_digests,
    filter_items_to_window,
    fit_digest_within_budget,
    build_quiet_day_digest,
    parse_github_compactor_digest_json,
    stable_github_compactor_journal_entry_id,
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
        journal_body="c" * 8000,
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


def test_fit_digest_within_budget_trims_card_fields_but_never_journal_body() -> None:
    long_body = "c" * 60_000  # was trimmed to 8000 before; now stored verbatim
    digest = GithubCompactorDigestV1(
        card_summary="a" * (CARD_SUMMARY_MAX_CHARS + 50),
        journal_title="b" * (JOURNAL_TITLE_MAX_CHARS + 50),
        journal_body=long_body,
        pr_refs=[],
    )
    fitted, trimmed = fit_digest_within_budget(digest)
    assert trimmed == ["card_summary", "journal_title"]
    assert len(fitted.card_summary) == CARD_SUMMARY_MAX_CHARS
    assert len(fitted.journal_title) == JOURNAL_TITLE_MAX_CHARS
    assert fitted.journal_body == long_body


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


def test_parse_github_compactor_digest_json_accepts_raw_control_characters() -> None:
    # Live: `invalid_json:Invalid control character` failed two complete digests.
    raw = '{"card_summary":"Card","journal_title":"T","journal_body":"line 1\nline 2\tx","pr_refs":[]}'
    assert parse_github_compactor_digest_json(raw).journal_body == "line 1\nline 2\tx"


def test_parse_github_compactor_digest_json_empty_completion_token() -> None:
    for raw in ("", "   \n", "```json\n```"):
        with pytest.raises(ValueError, match="compactor_digest_empty_completion"):
            parse_github_compactor_digest_json(raw)


def test_parse_github_compactor_digest_json_strips_code_fence() -> None:
    raw = '```json\n{"card_summary":"C","journal_title":"T","journal_body":"B","pr_refs":[]}\n```'
    assert parse_github_compactor_digest_json(raw).card_summary == "C"


def test_digest_wall_clock_budget_matches_verb_yaml() -> None:
    assert DIGEST_VERB_TIMEOUT_MS >= 600_000
    assert DIGEST_ORCH_RPC_TIMEOUT_SEC >= DIGEST_VERB_TIMEOUT_MS / 1000.0
    root = Path(__file__).resolve().parents[3] / "cognition"
    for verb_name in ("github_compactor_digest_v1", "chat_history_compactor_digest_v1"):
        verb = yaml.safe_load((root / "verbs" / f"{verb_name}.yaml").read_text(encoding="utf-8"))
        assert int(verb["timeout_ms"]) == DIGEST_VERB_TIMEOUT_MS
    prompt = (root / "prompts" / "github_compactor_digest_v1.j2").read_text(encoding="utf-8")
    assert f"(max {CARD_SUMMARY_MAX_CHARS} chars)" in prompt
    assert f"(max {JOURNAL_TITLE_MAX_CHARS} chars)" in prompt
    assert "NO length limit" in prompt


def _pr(number: int, body: str = "ok", merged_at: str = "2026-09-28T18:00:00Z") -> dict:
    return {"number": number, "title": f"PR {number}", "body": body, "merged_at": merged_at, "touched_paths": ["a"] * 50}


def test_digest_inputs_cover_all_forty_prs_without_count_cap() -> None:
    payload = {"repo": "acme/widgets", "items": [_pr(i, body="word " * 1800) for i in range(40)]}
    inputs, stats = build_github_compactor_digest_inputs(payload)
    numbers = [item["number"] for gi in inputs for item in gi["items"]]
    assert numbers == list(range(40))
    assert stats == {
        "total_count": 40,
        "covered_count": 40,
        "input_truncated": False,
        "truncated_pr_numbers": [],
        "chunk_count": len(inputs),
    }
    assert len(inputs) > 1
    for gi in inputs:
        assert len(json.dumps(gi["items"])) <= DIGEST_INPUT_CHAR_BUDGET
        assert gi["merged_pr_count_total"] == 40
        assert gi["chunk_count"] == len(inputs)
        assert "touched_paths" not in gi["items"][0]


def test_digest_inputs_single_call_when_day_fits() -> None:
    inputs, stats = build_github_compactor_digest_inputs({"repo": "r", "items": [_pr(1), _pr(2)]})
    assert len(inputs) == 1
    assert "chunk_count" not in inputs[0]
    assert stats["covered_count"] == 2


def test_digest_inputs_keep_long_bodies_up_to_safety_cap() -> None:
    body = "word " * 4000  # 20k chars: the old 1500/2000 caps cut this to a fragment
    inputs, stats = build_github_compactor_digest_inputs({"items": [_pr(1, body=body)]})
    assert inputs[0]["items"][0]["body"] == body.strip()
    assert stats["input_truncated"] is False
    huge = "word " * (PR_BODY_MAX_CHARS // 4)
    inputs, stats = build_github_compactor_digest_inputs({"items": [_pr(2, body=huge)]})
    assert inputs[0]["items"][0]["truncated"] is True
    assert stats["input_truncated"] is True
    assert stats["truncated_pr_numbers"] == [2]


def test_digest_inputs_honor_fetch_side_truncation_flag() -> None:
    item = _pr(3)
    item["body_truncated"] = True
    _inputs, stats = build_github_compactor_digest_inputs({"items": [item]})
    assert stats["input_truncated"] is True


def test_filter_items_to_window_is_inclusive_on_merged_at() -> None:
    start = datetime(2026, 9, 28, 6, 0, tzinfo=timezone.utc)
    end = datetime(2026, 9, 29, 5, 59, 59, 999999, tzinfo=timezone.utc)
    items = [
        _pr(1, merged_at="2026-09-28T05:59:59Z"),
        _pr(2, merged_at="2026-09-28T06:00:00Z"),
        _pr(3, merged_at="2026-09-29T05:59:59Z"),
        _pr(4, merged_at="2026-09-29T06:00:00Z"),
        {"number": 5, "merged_at": None},
    ]
    kept = filter_items_to_window(items, window_start=start, window_end=end)
    assert [i["number"] for i in kept] == [2, 3]


def test_merge_input_and_concatenation_keep_every_ref() -> None:
    parts = [
        GithubCompactorDigestV1(card_summary="a", journal_title="t", journal_body="body a", pr_refs=["#1", "#2"]),
        GithubCompactorDigestV1(card_summary="b", journal_title="t", journal_body="body b", pr_refs=["#3"]),
    ]
    merged_in = build_github_compactor_merge_input(base_input={"repo": "r", "merged_pr_count_total": 3, "items": []}, partial_digests=parts)
    assert "items" not in merged_in
    assert [p["pr_refs"] for p in merged_in["partial_digests"]] == [["#1", "#2"], ["#3"]]
    joined = concatenate_github_partial_digests(parts, window_label="2026-09-28")
    assert joined.pr_refs == ["#1", "#2", "#3"]
    assert "body a" in joined.journal_body and "body b" in joined.journal_body


def test_chunk_budget_measures_the_rendered_prompt() -> None:
    """Budget must count what tojson(indent=2) renders (ensure_ascii + htmlsafe escapes)."""
    from orion.cognition.planner.prompt_renderer import PromptRenderer

    body = ("It's a PR — with <b>arrows</b> → & quotes. " * 700)
    inputs, _ = build_github_compactor_digest_inputs({"repo": "r", "items": [_pr(i, body=body) for i in range(12)]})
    assert len(inputs) > 1
    renderer = PromptRenderer(Path(__file__).resolve().parents[2] / "prompts")
    empty = len(renderer.render("github_compactor_digest_v1.j2", {"metadata": {"github_compactor_input": {}}}))
    for gi in inputs:
        rendered = renderer.render("github_compactor_digest_v1.j2", {"metadata": {"github_compactor_input": gi}})
        # Template text + window metadata aside, the rendered items fit the budget (10% nesting slack).
        assert len(rendered) - empty <= DIGEST_INPUT_CHAR_BUDGET * 1.1
