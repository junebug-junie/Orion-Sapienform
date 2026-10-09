"""Curiosity runs as orion-introspect items. Pure: payloads in, items out.

The run join is `orion/curiosity/run_story.py`'s, read through
`curiosity_run_store.read_run_payload` / `read_runs_payload` -- the same story
the Curiosity tab shows. Nothing here re-derives a run's status or line.

Labels (spec, "Curiosity decisions"): a run with a write-up is `unsettled`
(what Orion concluded then, not established fact); a run without one --
failed, cancelled, wrote nothing -- is a short `record` of what happened, so
failures stay in Orion's view of their own history. Open self-questions are
`record`.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Optional

from orion.curiosity.run_story import LINE_LABELS
from orion.schemas.introspect import (
    CURIOSITY_FULL_JSON_BUDGET,
    DEFAULT_TEXT_CAP,
    SHORT_FIELD_CAP,
    IntrospectItemV1,
    clip_json_text,
    clip_text,
)

# Budgets in SERIALIZED JSON characters (quotes, backslashes and newlines
# double when escaped). Five list items -- text + error + prior claim +
# unknown_reason + reach-out decision + fixed fields -- stay under the 12,000
# MCP result budget; a test pins the worst case through the real MCP server.
LIST_TEXT_JSON_BUDGET = 1000
SHORT_JSON_BUDGET = 160

RUN_KIND = "curiosity_run"
SELF_QUESTION_KIND = "self_question"
INDEX_TEXT_CHARS = 1800

# Only a heading that is exactly "Answer" (any level, any case). Live
# 2026-10-09 only 1 of 371 curiosity write-ups has one; the rest are prose,
# so the opening is what a list shows.
_ANSWER_HEADING = re.compile(r"^#{1,6}[ \t]+answer[ \t]*:?[ \t]*$", re.IGNORECASE | re.MULTILINE)
_ANY_HEADING = re.compile(r"^#{1,6}[ \t]+\S", re.MULTILINE)

OUTCOME_FIELDS = ("turn_ok", "n_tested", "n_moved", "n_formed", "unknown_reason")


def answer_section(body: str | None) -> str:
    """The `## Answer` section of a write-up, else the whole (stripped) body."""
    text = (body or "").strip()
    match = _ANSWER_HEADING.search(text)
    if match is None:
        return text
    rest = text[match.end():]
    nxt = _ANY_HEADING.search(rest)
    section = (rest[: nxt.start()] if nxt else rest).strip()
    return section or text


def index_text(body: str | None) -> str:
    return clip_text(answer_section(body), INDEX_TEXT_CHARS)[0]


def _ms_to_dt(value: Any) -> Optional[datetime]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return datetime.fromtimestamp(int(value) / 1000.0, tz=timezone.utc)
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def occurred_at(run: dict[str, Any]) -> Optional[datetime]:
    """Started, else finished. None when the run carries neither clock."""
    return _ms_to_dt(run.get("started_at")) or _ms_to_dt(run.get("finished_at"))


def journal_clock(story: dict[str, Any]) -> Optional[datetime]:
    """When the run's newest write-up was saved, from the story timeline.

    Runs from before the admission path (live: 46 of 371 write-ups,
    2026-08-26..09-14) have only a graph clock, and some graph nodes carry
    none; the write-up's own clock keeps such a run from being dropped.
    """
    stamps = [
        it.get("at") for it in (story.get("timeline") or [])
        if isinstance(it, dict) and it.get("kind") == "journal" and it.get("at") is not None
    ]
    return _ms_to_dt(max(stamps)) if stamps else None


def _short(value: Any) -> str:
    text = clip_text(str(value or ""), SHORT_FIELD_CAP)[0]
    return clip_json_text(text, SHORT_JSON_BUDGET)[0]


def _list_text(text: str) -> tuple[str, bool]:
    body, cut = clip_text(text, DEFAULT_TEXT_CAP)
    body, cut_json = clip_json_text(body, LIST_TEXT_JSON_BUDGET)
    return body, cut or cut_json


def _prior(prior: Any) -> Optional[dict[str, Any]]:
    if not isinstance(prior, dict):
        return None
    return {"claim": _short(prior.get("claim")), "from": prior.get("from"), "to": prior.get("to")}


def _record_sentence(run: dict[str, Any]) -> str:
    label = LINE_LABELS.get(str(run.get("line") or ""), "Curiosity")
    status = str(run.get("status") or "unknown")
    text = f"{label} run {status} with no write-up"
    error = _short(run.get("error"))
    if error:
        text += f"; error: {error}"
    else:
        text += f"; outcome: {run.get('outcome_kind') or 'unknown'}"
    return text + "."


def run_item(
    story: dict[str, Any],
    outcome: Optional[dict[str, Any]] = None,
    *,
    full: bool = False,
    extra: Optional[dict[str, Any]] = None,
) -> Optional[IntrospectItemV1]:
    """One run from a `read_run_payload` story. None only when it has no
    clock at all: no start, no finish, and no write-up.

    `full` (one run by id) returns the write-up up to
    CURIOSITY_FULL_JSON_BUDGET serialized characters; a list shows the Answer
    section (or opening) at DEFAULT_TEXT_CAP.
    """
    run = story.get("run") or {}
    when = occurred_at(run)
    clock_from = None
    if when is None:
        when, clock_from = journal_clock(story), "write_up"
    run_id = str(run.get("run_id") or "")
    if when is None or not run_id:
        return None
    body = str(story.get("journal_body") or "").strip()
    if body:
        status = "unsettled"
        if full:
            text, truncated = clip_json_text(body, CURIOSITY_FULL_JSON_BUDGET)
        else:
            text, truncated = _list_text(answer_section(body))
    else:
        status, (text, truncated) = "record", _list_text(_record_sentence(run))
    reach = run.get("reach_out") if isinstance(run.get("reach_out"), dict) else {}
    fields: dict[str, Any] = {
        "line": run.get("line"),
        "status": run.get("status"),
        "error": _short(run.get("error")) or None,
        "hops": run.get("hops"),
        "findings": run.get("findings"),
        "revisions": run.get("revisions"),
        "prior_touched": _prior(run.get("prior_touched")),
        "outcome_kind": run.get("outcome_kind"),
        "reach_out": _short(reach.get("decision")) or None,
        "has_write_up": bool(body),
    }
    if outcome:
        fields.update({k: outcome.get(k) for k in OUTCOME_FIELDS})
        if fields.get("unknown_reason") is not None:
            fields["unknown_reason"] = _short(fields["unknown_reason"])
    if clock_from is not None:
        fields["clock_from"] = clock_from
    fields.update(extra or {})
    return IntrospectItemV1(
        id=run_id, occurred_at=when, kind=RUN_KIND, epistemic_status=status,
        text=text, truncated=truncated, extra=fields,
    )


def self_question_item(row: dict[str, Any]) -> IntrospectItemV1:
    created = row["created_at"]
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    last = row.get("last_asked_at")
    text, truncated = _list_text(str(row.get("text") or ""))
    return IntrospectItemV1(
        id=str(row["question_id"]), occurred_at=created, kind=SELF_QUESTION_KIND,
        epistemic_status="record", text=text, truncated=truncated,
        extra={
            "family": _short(row.get("family")),
            "ask_count": row.get("ask_count"),
            "last_asked_at": last.isoformat() if isinstance(last, datetime) else None,
            "pinned": bool(row.get("pinned")),
        },
    )
