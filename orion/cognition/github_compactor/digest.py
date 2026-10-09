from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from orion.cognition.compactor.budget import fit_fields_within_budget
from orion.cognition.compactor.chunking import chunk_items_by_char_budget
from orion.cognition.compactor.constants import DIGEST_INPUT_CHAR_BUDGET
from orion.cognition.compactor.digest import parse_compactor_digest_json
from orion.cognition.compactor.truncate import truncate_at_word_boundary
from orion.cognition.github_compactor.constants import (
    CARD_SUMMARY_MAX_CHARS,
    JOURNAL_TITLE_MAX_CHARS,
    PR_BODY_MAX_CHARS,
)
from orion.schemas.actions.github_compactor import GithubCompactorDigestV1

_COMPACTOR_JOURNAL_ENTRY_NS = NAMESPACE_URL


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def filter_items_to_window(items: list, *, window_start: datetime | None, window_end: datetime | None) -> list:
    """Keep PRs whose merged_at falls inside [window_start, window_end].

    Orch re-applies the window after fetch so a scheduled calendar-day run stays
    correct even against an exec that predates window-bounded fetch (it returns
    a rolling now-N-days list instead).
    """
    if window_start is None and window_end is None:
        return list(items)
    kept = []
    for item in items:
        if not isinstance(item, dict):
            continue
        merged = _parse_ts(item.get("merged_at"))
        if merged is None:
            continue
        if window_start is not None and merged < window_start:
            continue
        if window_end is not None and merged > window_end:
            continue
        kept.append(item)
    return kept


def _compact_item(item: dict) -> tuple[dict, bool]:
    body = str(item.get("body") or "").strip()
    # Fetch already capped at PR_BODY_MAX_CHARS; re-applied here only so a
    # payload from another producer cannot blow a chunk.
    body, body_truncated = truncate_at_word_boundary(body, PR_BODY_MAX_CHARS)
    body_truncated = body_truncated or bool(item.get("body_truncated"))
    compact = {
        "number": item.get("number"),
        "title": item.get("title"),
        "body": body,
        "merged_at": item.get("merged_at"),
        "url": item.get("url"),
    }
    services = item.get("inferred_services")
    if isinstance(services, list) and services:
        compact["inferred_services"] = services
    if body_truncated:
        compact["truncated"] = True
    return compact, body_truncated


def build_github_compactor_digest_inputs(
    fetch_payload: dict,
    *,
    budget_chars: int = DIGEST_INPUT_CHAR_BUDGET,
) -> tuple[list[dict], dict[str, Any]]:
    """Every merged PR -> one or more digest input payloads, each within `budget_chars`.

    Returns ``(inputs, stats)``. ``inputs`` has one payload when the whole day
    fits (single digest call) and several when it does not (map step; the orch
    then merges the partial digests). No PR is ever dropped: ``stats`` reports
    ``total_count`` == ``covered_count`` by construction, and ``input_truncated``
    is true only if some PR body hit the PR_BODY_MAX_CHARS safety cap.
    """
    if not isinstance(fetch_payload, dict):
        return [], {"total_count": 0, "covered_count": 0, "input_truncated": False, "chunk_count": 0}
    raw_items = [item for item in (fetch_payload.get("items") or []) if isinstance(item, dict)]
    compact_items: list[dict] = []
    truncated_numbers: list[Any] = []
    for item in raw_items:
        compact, truncated = _compact_item(item)
        compact_items.append(compact)
        if truncated:
            truncated_numbers.append(item.get("number"))
    chunks = chunk_items_by_char_budget(compact_items, budget_chars=budget_chars) if compact_items else []
    base = {
        key: fetch_payload.get(key)
        for key in ("repo", "lookback_days", "window_mode", "window_start_utc", "window_end_utc", "calendar_date")
        if fetch_payload.get(key) is not None
    }
    inputs: list[dict] = []
    for index, chunk in enumerate(chunks, start=1):
        payload = dict(base)
        payload["merged_pr_count_total"] = len(compact_items)
        payload["items"] = chunk
        if len(chunks) > 1:
            payload["chunk_index"] = index
            payload["chunk_count"] = len(chunks)
        if any(item.get("truncated") for item in chunk):
            payload["item_content_truncated"] = True
        inputs.append(payload)
    stats = {
        "total_count": len(raw_items),
        "covered_count": sum(len(chunk) for chunk in chunks),
        "input_truncated": bool(truncated_numbers),
        "truncated_pr_numbers": truncated_numbers,
        "chunk_count": len(chunks),
    }
    return inputs, stats


def build_github_compactor_merge_input(
    *,
    base_input: dict,
    partial_digests: list[GithubCompactorDigestV1],
) -> dict:
    """Reduce step input: the chunk digests to be merged into one day digest."""
    payload = {
        key: base_input.get(key)
        for key in ("repo", "lookback_days", "window_mode", "window_start_utc", "window_end_utc", "calendar_date", "merged_pr_count_total")
        if base_input.get(key) is not None
    }
    payload["partial_digests"] = [
        {
            "chunk_index": index,
            "card_summary": digest.card_summary,
            "journal_title": digest.journal_title,
            "journal_body": digest.journal_body,
            "pr_refs": list(digest.pr_refs or []),
        }
        for index, digest in enumerate(partial_digests, start=1)
    ]
    return payload


def concatenate_github_partial_digests(
    partial_digests: list[GithubCompactorDigestV1],
    *,
    window_label: str,
) -> GithubCompactorDigestV1:
    """Deterministic reduce used only when every merge call failed.

    Every chunk digest is real model output over real PRs, so joining them
    loses no coverage; it is recorded as ``merge_mode=concatenated`` so it is
    never mistaken for a merged narrative.
    """
    refs: list[str] = []
    for digest in partial_digests:
        for ref in digest.pr_refs or []:
            if ref not in refs:
                refs.append(ref)
    body = "\n\n".join(
        f"Part {index} of {len(partial_digests)}\n\n{digest.journal_body.strip()}"
        for index, digest in enumerate(partial_digests, start=1)
    )
    summary = " ".join(d.card_summary.strip() for d in partial_digests if d.card_summary.strip())
    return GithubCompactorDigestV1(
        card_summary=summary or f"Repo development digest — {window_label}",
        journal_title=f"Repo development digest — {window_label}",
        journal_body=body,
        pr_refs=refs,
    )


def fit_digest_within_budget(
    digest: GithubCompactorDigestV1,
) -> tuple[GithubCompactorDigestV1, list[str]]:
    """Trim the memory-card fields (card_summary, journal_title) to their caps.

    ``journal_body`` is deliberately NOT trimmed: it is stored in full and
    embedded verbatim in the daily letter. Returns ``(digest,
    trimmed_field_names)``; callers surface a non-empty list as run evidence.
    """
    fitted, trimmed = fit_fields_within_budget(
        {
            "card_summary": (digest.card_summary, CARD_SUMMARY_MAX_CHARS),
            "journal_title": (digest.journal_title, JOURNAL_TITLE_MAX_CHARS),
        }
    )
    if not trimmed:
        return digest, []
    return digest.model_copy(update=fitted), trimmed


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
    return parse_compactor_digest_json(raw, GithubCompactorDigestV1)


def stable_github_compactor_journal_entry_id(
    *,
    workflow_id: str,
    calendar_date: str,
    repo: str,
) -> str:
    payload = "|".join([workflow_id.strip(), calendar_date.strip(), repo.strip()])
    return str(uuid5(_COMPACTOR_JOURNAL_ENTRY_NS, payload))
