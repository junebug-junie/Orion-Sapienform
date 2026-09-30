"""Gather one day of Orion's activity from Postgres, read-only, full length.

Pure async SQL over an asyncpg-style connection (``fetch`` / ``fetchrow``), so Hub
can call it directly and tests can pass a fake. Every statement is a SELECT.

Each source is read independently: a failed query records
``sources[name] = error`` and leaves that section empty, so one broken table
names its gap in the letter instead of cancelling the letter. Table and column
names were checked against the live database on 2026-09-30.

Sources:

* curiosity -- completed ``curiosity.investigate`` / ``self_sense_eval`` finish events
  (substrate_durable_run_state), their journal entries (source_ref ``curiosity:<run>``),
  curiosity_run_outcomes, failed runs, and self_sense_eval_log answers.
* readings -- verified world_pulse_read_seed rows (orion/world_pulse_read/introspect.py,
  full text) and ``world_pulse_read_stage2:*`` journal entries.
* dreams -- narrative dreams and OFFERED hypotheses only (orion/dream/introspect_sql.py,
  the blind rule).
* reveries -- substrate_reverie_thought / substrate_reverie_chain (text) and
  reverie_visual_artifact (image references; no bytes).
* compactors -- the chat and GitHub compactor journal entries for the day, by their
  stable ids (or their stable source_ref), full length.
* world pulse -- the day's world_pulse_digest, if any.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from typing import Any, Awaitable, Callable

from orion.cognition.chat_history_compactor.digest import stable_chat_compactor_journal_entry_id
from orion.cognition.compactor.index import build_compactor_index
from orion.cognition.github_compactor.digest import stable_github_compactor_journal_entry_id
from orion.dream.introspect_sql import H_COLS, HYPOTHESIS_BLIND_WHERE, NARRATIVE_WINDOW_SQL
from orion.orion_day.window import orion_day_window
from orion.schemas.orion_day import (
    ORION_DAY_TIMEZONE,
    CuriosityFailedRunV1,
    CuriosityRunItemV1,
    DreamHypothesisV1,
    DreamNarrativeV1,
    JournalTextV1,
    OrionDayMaterialV1,
    OrionDaySourceStatusV1,
    ReadingItemV1,
    ReverieChainV1,
    ReverieThoughtV1,
    SelfSenseAnswerV1,
    VisualReverieV1,
    WorldPulseDigestItemV1,
    WorldPulseDigestV1,
)
from orion.world_pulse_read.introspect import reading_items_between

logger = logging.getLogger(__name__)

CURIOSITY_WORKFLOWS = ["curiosity.investigate", "self_sense_eval"]
# Workflow ids the compactor passes write under (services/orion-cortex-orch/app/workflow_runtime.py).
CHAT_COMPACTOR_WORKFLOW_ID = "chat_history_compactor_pass"
GITHUB_COMPACTOR_WORKFLOW_ID = "github_compactor_pass"
DEFAULT_GITHUB_REPO = "junebug-junie/Orion-Sapienform"
CURIOSITY_JOURNAL_PREFIX = "curiosity:"

CURIOSITY_FINISH_SQL = """
SELECT DISTINCT ON (s.run_id) s.run_id, s.workflow, s.generated_at, s.detail::jsonb AS detail
FROM substrate_durable_run_state s
WHERE s.status = 'completed' AND s.node = 'finish'
  AND s.workflow = ANY($3::text[])
  AND s.generated_at >= $1 AND s.generated_at < $2
ORDER BY s.run_id, s.generated_at DESC
"""

CURIOSITY_FAILED_SQL = """
SELECT DISTINCT ON (s.run_id) s.run_id, s.workflow, s.generated_at, s.detail::jsonb->>'error' AS error
FROM substrate_durable_run_state s
WHERE s.status = 'failed'
  AND s.workflow = ANY($3::text[])
  AND s.generated_at >= $1 AND s.generated_at < $2
ORDER BY s.run_id, s.generated_at DESC
"""

CURIOSITY_JOURNALS_SQL = """
SELECT entry_id, created_at, title, body, source_ref
FROM journal_entries
WHERE (source_ref LIKE 'curiosity:%' AND created_at >= $1 AND created_at < $2)
   OR entry_id = ANY($3::text[])
ORDER BY created_at, entry_id
"""

CURIOSITY_OUTCOMES_SQL = """
SELECT run_id, completed_at, turn_ok, realized_nats, unknown_reason,
       n_tested, n_moved, n_formed, per_prior
FROM curiosity_run_outcomes
WHERE run_id = ANY($1::text[])
"""

SELF_SENSE_SQL = """
SELECT run_id, question_key, question, answer_text, answer_source,
       self_label_score, grounded_record_score, created_at
FROM self_sense_eval_log
WHERE created_at >= $1 AND created_at < $2
ORDER BY created_at, entry_id
"""

JOURNALS_BY_PREFIX_SQL = """
SELECT entry_id, created_at, title, body, source_ref
FROM journal_entries
WHERE source_ref LIKE $3 AND created_at >= $1 AND created_at < $2
ORDER BY created_at, entry_id
"""

# A compactor's DAY digest for day D is written after D ends (06:00 the next morning). The
# GitHub compactor's rolling mode labels a run with its own date and shares the day mode's
# stable id, so a rolling run made DURING D would otherwise stand in for D's digest.
JOURNAL_BY_ID_OR_REF_SQL = """
SELECT entry_id, created_at, title, body, source_ref
FROM journal_entries
WHERE (entry_id = $1 OR source_ref = $2) AND created_at >= $3
ORDER BY (entry_id = $1) DESC, created_at DESC
LIMIT 1
"""

# Offered hypotheses (the blind rule, orion/dream/introspect_sql.py) AND only those whose offering
# curiosity run completed: "offered" alone does not mean Orion saw one (a failed run keeps its
# offer), and the letter must not be the first time Orion sees a hypothesis.
DREAM_HYPOTHESES_SEEN_SQL = f"""
SELECT {H_COLS} FROM dream_hypothesis h
WHERE {HYPOTHESIS_BLIND_WHERE} AND h.offered_at >= $1 AND h.offered_at < $2
  AND EXISTS (SELECT 1 FROM substrate_durable_run_state s
              WHERE s.run_id = h.offered_run_id AND s.status = 'completed')
ORDER BY h.offered_at, h.hypothesis_id
"""
READINGS_LIMIT = 2000

REVERIE_THOUGHTS_SQL = """
SELECT thought_id, created_at, salience, interpretation, expectation, expectation_verdict,
       thought_json->>'chain_id' AS chain_id,
       COALESCE(thought_json->'hollow' = 'true'::jsonb, false) AS hollow
FROM substrate_reverie_thought
WHERE created_at >= $1 AND created_at < $2
ORDER BY created_at, thought_id
"""

REVERIE_CHAINS_SQL = """
SELECT chain_id, created_at, theme_key, terminal_reason, ema_salience,
       CASE WHEN jsonb_typeof(chain_json->'thought_ids') = 'array'
            THEN jsonb_array_length(chain_json->'thought_ids') ELSE 0 END AS thought_count
FROM substrate_reverie_chain
WHERE created_at >= $1 AND created_at < $2
ORDER BY created_at, chain_id
"""

VISUAL_REVERIES_SQL = """
SELECT a.sha256, a.chain_id, a.step_index, a.created_at, a.mime, a.width, a.height,
       a.bytes, a.path, a.description, c.theme_key
FROM reverie_visual_artifact a
LEFT JOIN reverie_visual_chain c ON c.chain_id = a.chain_id
WHERE a.created_at >= $1 AND a.created_at < $2
ORDER BY a.created_at, a.sha256
"""

WORLD_PULSE_DIGEST_SQL = """
SELECT run_id, date, title, executive_summary, payload_json
FROM world_pulse_digest
WHERE date = $1
ORDER BY created_at DESC
LIMIT 1
"""


def _obj(raw: Any) -> Any:
    if isinstance(raw, (str, bytes)):
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return None
    return raw


def _text_field(value: Any) -> str | None:
    """A detail field that is either plain text or ``{"text": ...}``."""
    value = _obj(value) if isinstance(value, str) and value.lstrip().startswith("{") else value
    if isinstance(value, dict):
        value = value.get("text")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    return None


def _aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _journal(row: Any) -> JournalTextV1:
    return JournalTextV1(
        entry_id=str(row["entry_id"]),
        created_at=_aware(row["created_at"]),
        title=row["title"],
        body=row["body"] or "",
        source_ref=row["source_ref"],
    )


def _error(exc: BaseException) -> OrionDaySourceStatusV1:
    return OrionDaySourceStatusV1(status="error", error=f"{type(exc).__name__}: {exc}"[:300])


def _status(count: int) -> OrionDaySourceStatusV1:
    return OrionDaySourceStatusV1(status="ok" if count else "empty", count=count)


async def _read(sources: dict, name: str, reader: Callable[[], Awaitable[Any]], default: Any) -> Any:
    try:
        value = await reader()
    except Exception as exc:  # noqa: BLE001 -- one broken source names its gap, never cancels the letter
        logger.warning("orion_day_gather_source_failed source=%s err=%s", name, exc)
        sources[name] = _error(exc)
        return default
    count = len(value) if isinstance(value, list) else (1 if value is not None else 0)
    sources[name] = _status(count)
    return value


async def gather_curiosity(conn: Any, start: datetime, end: datetime) -> list[CuriosityRunItemV1]:
    finishes = await conn.fetch(CURIOSITY_FINISH_SQL, start, end, CURIOSITY_WORKFLOWS)
    details = {str(r["run_id"]): (r, _obj(r["detail"]) or {}) for r in finishes}
    wanted_ids = [str(d.get("journal_entry_id")) for _, d in details.values() if d.get("journal_entry_id")]
    journals = await conn.fetch(CURIOSITY_JOURNALS_SQL, start, end, wanted_ids)
    by_run: dict[str, Any] = {}
    for j in journals:
        ref = j["source_ref"] or ""
        if ref.startswith(CURIOSITY_JOURNAL_PREFIX):
            by_run[ref[len(CURIOSITY_JOURNAL_PREFIX):]] = j
    for run_id, (_, detail) in details.items():
        want = detail.get("journal_entry_id")
        if want and run_id not in by_run:
            match = next((j for j in journals if str(j["entry_id"]) == str(want)), None)
            if match is not None:
                by_run[run_id] = match
    run_ids = sorted(set(details) | set(by_run))
    outcomes = {str(r["run_id"]): r for r in await conn.fetch(CURIOSITY_OUTCOMES_SQL, run_ids)} if run_ids else {}

    items: list[CuriosityRunItemV1] = []
    for run_id in run_ids:
        row, detail = details.get(run_id, (None, {}))
        journal = by_run.get(run_id)
        workflow = row["workflow"] if row is not None else None
        if workflow == "self_sense_eval":
            continue  # its substance is the self_sense_eval_log answers, gathered separately
        outcome_row = outcomes.get(run_id)
        outcome = None
        if outcome_row is not None:
            outcome = {
                "turn_ok": outcome_row["turn_ok"],
                "realized_nats": outcome_row["realized_nats"],
                "unknown_reason": outcome_row["unknown_reason"],
                "n_tested": outcome_row["n_tested"],
                "n_moved": outcome_row["n_moved"],
                "n_formed": outcome_row["n_formed"],
                "per_prior": _obj(outcome_row["per_prior"]) or [],
            }
        items.append(CuriosityRunItemV1(
            run_id=run_id,
            workflow=workflow or "curiosity.investigate",
            line=detail.get("line"),
            self_question_family=detail.get("self_question_family"),
            completed_at=_aware(row["generated_at"]) if row is not None else _aware(journal["created_at"]),
            journal_entry_id=str(journal["entry_id"]) if journal is not None else detail.get("journal_entry_id"),
            journal_title=journal["title"] if journal is not None else None,
            journal_body=(journal["body"] or "") if journal is not None else "",
            finding_text=None if journal is not None else _text_field(detail.get("finding_text")),
            continue_line=_bool(detail.get("continue_line")),
            reach_out=_bool(detail.get("reach_out")),
            reach_out_why=_text_field(detail.get("reach_out_why")),
            self_definition_text=_text_field(detail.get("self_definition")),
            lived_answer_text=_text_field(detail.get("lived_answer")),
            outcome=outcome,
        ))
    items.sort(key=lambda i: (i.completed_at or datetime.min.replace(tzinfo=timezone.utc), i.run_id))
    return items


async def gather_curiosity_failed(conn: Any, start: datetime, end: datetime) -> list[CuriosityFailedRunV1]:
    rows = await conn.fetch(CURIOSITY_FAILED_SQL, start, end, CURIOSITY_WORKFLOWS)
    items = [CuriosityFailedRunV1(run_id=str(r["run_id"]), workflow=r["workflow"],
                                  failed_at=_aware(r["generated_at"]), error=r["error"]) for r in rows]
    return sorted(items, key=lambda i: (i.failed_at, i.run_id))


async def gather_self_sense(conn: Any, start: datetime, end: datetime) -> list[SelfSenseAnswerV1]:
    rows = await conn.fetch(SELF_SENSE_SQL, start, end)
    return [SelfSenseAnswerV1(
        run_id=r["run_id"], question_key=r["question_key"], question=r["question"] or "",
        answer_text=r["answer_text"] or "", answer_source=r["answer_source"],
        self_label_score=r["self_label_score"], grounded_record_score=r["grounded_record_score"],
        created_at=_aware(r["created_at"]),
    ) for r in rows]


async def gather_readings(conn: Any, start: datetime, end: datetime) -> list[ReadingItemV1]:
    items = await reading_items_between(conn, since=start, until=end, text_cap=None, limit=READINGS_LIMIT)
    return [ReadingItemV1(
        seed_id=i.id, occurred_at=i.occurred_at, title=i.extra.get("title") or None,
        url=i.extra.get("url") or None, why_now=i.extra.get("why_now") or None, learned=i.text,
        reading_status=i.extra.get("reading_status"), source_read=bool(i.extra.get("source_read")),
    ) for i in items]


async def gather_journals_by_prefix(conn: Any, start: datetime, end: datetime, prefix: str) -> list[JournalTextV1]:
    rows = await conn.fetch(JOURNALS_BY_PREFIX_SQL, start, end, prefix + "%")
    return [_journal(r) for r in rows]


async def gather_dream_narratives(conn: Any, start: datetime, end: datetime) -> list[DreamNarrativeV1]:
    return [DreamNarrativeV1(
        id=int(r["id"]), dream_date=r["dream_date"], occurred_at=_aware(r["occurred_at"]),
        tldr=r["tldr"], themes=_obj(r["themes"]), narrative=r["narrative"],
    ) for r in await conn.fetch(NARRATIVE_WINDOW_SQL, start, end)]


async def gather_dream_hypotheses(conn: Any, start: datetime, end: datetime) -> list[DreamHypothesisV1]:
    return [DreamHypothesisV1(
        hypothesis_id=str(r["hypothesis_id"]), cycle_id=r["cycle_id"], claim=r["claim"] or "",
        why=r["why"], offered_at=_aware(r["occurred_at"]),
    ) for r in await conn.fetch(DREAM_HYPOTHESES_SEEN_SQL, start, end)]


async def gather_reverie_thoughts(conn: Any, start: datetime, end: datetime) -> list[ReverieThoughtV1]:
    return [ReverieThoughtV1(
        thought_id=str(r["thought_id"]), chain_id=r["chain_id"], created_at=_aware(r["created_at"]),
        salience=r["salience"], interpretation=r["interpretation"] or "", expectation=r["expectation"],
        expectation_verdict=r["expectation_verdict"], hollow=bool(r["hollow"]),
    ) for r in await conn.fetch(REVERIE_THOUGHTS_SQL, start, end)]


async def gather_reverie_chains(conn: Any, start: datetime, end: datetime) -> list[ReverieChainV1]:
    return [ReverieChainV1(
        chain_id=str(r["chain_id"]), created_at=_aware(r["created_at"]), theme_key=r["theme_key"],
        terminal_reason=r["terminal_reason"], ema_salience=r["ema_salience"],
        thought_count=int(r["thought_count"] or 0),
    ) for r in await conn.fetch(REVERIE_CHAINS_SQL, start, end)]


async def gather_visual_reveries(conn: Any, start: datetime, end: datetime) -> list[VisualReverieV1]:
    return [VisualReverieV1(
        sha256=str(r["sha256"]), chain_id=r["chain_id"], step_index=r["step_index"],
        created_at=_aware(r["created_at"]), mime=r["mime"], width=r["width"], height=r["height"],
        bytes=r["bytes"], path=r["path"], description=r["description"], theme_key=r["theme_key"],
    ) for r in await conn.fetch(VISUAL_REVERIES_SQL, start, end)]


async def gather_chat_compactor(conn: Any, letter_date: date, written_after: datetime) -> JournalTextV1 | None:
    index = build_compactor_index(kind="chat_history_log", mode="day", calendar_date=letter_date.isoformat())
    entry_id = stable_chat_compactor_journal_entry_id(workflow_id=CHAT_COMPACTOR_WORKFLOW_ID, compactor_index=index)
    row = await conn.fetchrow(JOURNAL_BY_ID_OR_REF_SQL, entry_id, f"{CHAT_COMPACTOR_WORKFLOW_ID}:{index}", written_after)
    return _journal(row) if row is not None else None


async def gather_github_compactor(conn: Any, letter_date: date, repo: str, written_after: datetime) -> JournalTextV1 | None:
    day = letter_date.isoformat()
    entry_id = stable_github_compactor_journal_entry_id(
        workflow_id=GITHUB_COMPACTOR_WORKFLOW_ID, calendar_date=day, repo=repo)
    row = await conn.fetchrow(JOURNAL_BY_ID_OR_REF_SQL, entry_id, f"{GITHUB_COMPACTOR_WORKFLOW_ID}:{day}:{repo}",
                              written_after)
    return _journal(row) if row is not None else None


async def gather_world_pulse_digest(conn: Any, letter_date: date) -> WorldPulseDigestV1 | None:
    row = await conn.fetchrow(WORLD_PULSE_DIGEST_SQL, letter_date.isoformat())
    if row is None:
        return None
    payload = _obj(row["payload_json"]) or {}
    items = []
    for raw in payload.get("items") or []:
        if not isinstance(raw, dict) or not raw.get("title"):
            continue
        items.append(WorldPulseDigestItemV1(
            title=str(raw["title"]), category=raw.get("category"),
            summary=raw.get("summary"), why_it_matters=raw.get("why_it_matters"),
        ))
    return WorldPulseDigestV1(
        run_id=str(row["run_id"]), date=str(row["date"]), title=row["title"],
        executive_summary=row["executive_summary"], items=items,
    )


async def gather_orion_day(
    conn: Any,
    letter_date: date,
    *,
    now: datetime | None = None,
    github_repo: str = DEFAULT_GITHUB_REPO,
    tz_name: str = ORION_DAY_TIMEZONE,
) -> OrionDayMaterialV1:
    """Read everything Orion did on ``letter_date`` (local ``tz_name`` day). Read-only."""
    window = orion_day_window(letter_date, tz_name=tz_name)
    start, end = window.window_start, window.window_end
    sources: dict[str, OrionDaySourceStatusV1] = {}
    curiosity = await _read(sources, "curiosity_runs", lambda: gather_curiosity(conn, start, end), [])
    failed = await _read(sources, "curiosity_failed", lambda: gather_curiosity_failed(conn, start, end), [])
    self_sense = await _read(sources, "self_sense", lambda: gather_self_sense(conn, start, end), [])
    readings = await _read(sources, "readings", lambda: gather_readings(conn, start, end), [])
    if len(readings) >= READINGS_LIMIT:
        sources["readings"] = sources["readings"].model_copy(update={"truncated": True})
    reading_journals = await _read(
        sources, "reading_journals",
        lambda: gather_journals_by_prefix(conn, start, end, "world_pulse_read_stage2:"), [])
    narratives = await _read(sources, "dream_narratives", lambda: gather_dream_narratives(conn, start, end), [])
    hypotheses = await _read(sources, "dream_hypotheses", lambda: gather_dream_hypotheses(conn, start, end), [])
    thoughts = await _read(sources, "reverie_thoughts", lambda: gather_reverie_thoughts(conn, start, end), [])
    chains = await _read(sources, "reverie_chains", lambda: gather_reverie_chains(conn, start, end), [])
    visuals = await _read(sources, "visual_reveries", lambda: gather_visual_reveries(conn, start, end), [])
    chat = await _read(sources, "chat_compactor", lambda: gather_chat_compactor(conn, letter_date, end), None)
    github = await _read(sources, "github_compactor", lambda: gather_github_compactor(conn, letter_date, github_repo, end), None)
    digest = await _read(sources, "world_pulse_digest", lambda: gather_world_pulse_digest(conn, letter_date), None)
    return OrionDayMaterialV1(
        letter_date=letter_date,
        timezone=tz_name,
        window_start=start,
        window_end=end,
        gathered_at=now or datetime.now(timezone.utc),
        curiosity_runs=curiosity,
        curiosity_failed=failed,
        self_sense=self_sense,
        readings=readings,
        reading_journals=reading_journals,
        dream_narratives=narratives,
        dream_hypotheses=hypotheses,
        reverie_thoughts=thoughts,
        reverie_chains=chains,
        visual_reveries=visuals,
        chat_compactor=chat,
        github_compactor=github,
        world_pulse_digest=digest,
        sources=sources,
    )
