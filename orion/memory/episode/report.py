"""The daily old-vs-new memory report (spec Stage 1 acceptance 11). Read-only.

For each shadow episode closed in the window: the legacy crystallization rows its turns produced
(the old intake) next to the shadow distiller's memories (the new writer), with the validator's
rejections and the run's cost. No notification is sent; Juniper reads it if she wants to.

PRIVACY: the report quotes Juniper's conversation. It is written to a local volume
(MEMORY_EPISODE_REPORT_DIR), never committed and never published on the bus.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable
from zoneinfo import ZoneInfo

EPISODES_SQL = """
SELECT episode_id, started_at, last_turn_at, closed_at, close_reason, close_lag_sec, episode_status,
       skip_reason, juniper_turn_count, command_turn_count, turns
FROM memory_episode_shadow
WHERE status = 'closed' AND closed_at >= $1 AND closed_at < $2 AND source_platform IS NULL
ORDER BY closed_at
"""
OLD_ROWS_SQL = """
SELECT DISTINCT c.crystallization_id::text AS id, c.kind, c.status, c.summary
FROM memory_crystallization_sources s JOIN memory_crystallizations c USING (crystallization_id)
WHERE s.source_kind = 'chat_turn' AND s.source_id = ANY($1::text[])
ORDER BY c.kind, c.summary
"""
NEW_ROWS_SQL = """
SELECT memory_id::text AS id, purpose, voice, channel, statement, stakes, stakes_reason, confirmation_state
FROM episode_memory WHERE episode_id = $1 ORDER BY purpose, statement
"""
EVENTS_SQL = """
SELECT op, reason, count(*) AS n FROM episode_memory_event
WHERE episode_id = $1 AND op IN ('rejected_invalid', 'downgraded_voice', 'stakes_raised', 'stakes_reason_missing',
                                    'stakes_reason_set', 'stakes_uncategorized', 'evidence_dropped')
GROUP BY op, reason ORDER BY op, reason
"""
# Closed (not skipped) episodes in the window whose distill run has not finished yet: while any
# remain, the report is provisional and is rewritten on the next pass.
UNDISTILLED_SQL = """
SELECT count(*) AS n FROM memory_episode_shadow e
WHERE e.status = 'closed' AND e.episode_status = 'closed' AND e.source_platform IS NULL
  AND e.closed_at >= $1 AND e.closed_at < $2
  AND NOT EXISTS (SELECT 1 FROM episode_distill_run d WHERE d.episode_id = e.episode_id)
"""
RUN_SQL = """
SELECT run_id, model, prompt_tokens, completion_tokens, llm_latency_ms, hold_wait_ms, coverage
FROM episode_distill_run WHERE episode_id = $1
"""

# Memory Stage 2 referents (orion/memory/referents): what resolution did in the window.
REFERENT_NODES_SQL = """
SELECT promotion_state, admitted_by, count(*) AS n FROM referent_alias
WHERE alias_class = 'key' AND created_at >= $1 AND created_at < $2
GROUP BY 1, 2 ORDER BY 1, 2
"""
REFERENT_ALIASES_SQL = """
SELECT alias_class, promotion_state, admitted_by, count(*) AS n FROM referent_alias
WHERE alias_class <> 'key' AND created_at >= $1 AND created_at < $2
GROUP BY 1, 2, 3 ORDER BY 1, 2, 3
"""
IDENTITY_QUESTIONS_SQL = """
SELECT text, scope, answer_via, source_refs->0->>'reason' AS reason FROM memory_tension_shadow
WHERE status = 'open' AND source_refs->0->>'reason' IN ('referent_identity', 'alias_collision', 'label_collision')
ORDER BY created_at
"""
HELD_CLAIMS_SQL = """
SELECT count(*) AS n FROM substrate_graph_journal p
WHERE p.event_kind = 'proposal' AND p.actor = 'memory.referents' AND p.recorded_at >= $1 AND p.recorded_at < $2
  AND NOT EXISTS (SELECT 1 FROM substrate_graph_journal d WHERE d.event_kind = 'decision' AND d.proposal_id = p.proposal_id)
"""

_VOICE_LABEL = {
    "juniper_said": "Juniper said",
    "worked_out_together": "worked out together",
    "orion_thought": "Orion thought",
    "orion_read": "Orion read",
    "orion_self_knowledge": "Orion self-knowledge",
}


def _turn_ids(turns: Any) -> list[str]:
    import json

    items = json.loads(turns) if isinstance(turns, str) else (turns or [])
    return [str(t.get("correlation_id")) for t in items if isinstance(t, dict)]


def _fmt_lag(sec: Any) -> str:
    if sec is None:
        return "n/a"
    sec = float(sec)
    return f"{sec / 3600:.1f} h" if sec >= 3600 else f"{sec / 60:.0f} min"


def render_episode(ep: dict[str, Any], old: Iterable[dict], new: Iterable[dict], events: Iterable[dict],
                   run: dict[str, Any] | None, tz: ZoneInfo) -> list[str]:
    started = ep["started_at"].astimezone(tz).strftime("%Y-%m-%d %H:%M")
    ended = ep["last_turn_at"].astimezone(tz).strftime("%H:%M")
    lines = [
        f"## Episode {started}-{ended} ({ep['episode_id'][:8]})",
        "",
        f"{ep.get('juniper_turn_count') or 0} turns from Juniper, {ep.get('command_turn_count') or 0} commands. "
        f"Closed by `{ep.get('close_reason')}`, {_fmt_lag(ep.get('close_lag_sec'))} after its last turn."
        + (f" Skipped: {ep.get('skip_reason')}." if ep.get("episode_status") == "skipped" else ""),
        "",
    ]
    old = list(old)
    lines.append(f"**Old intake ({len(old)} crystallization rows):**")
    lines += [f"- [{o['kind']}, {o['status']}] {o['summary']}" for o in old] or ["- (none)"]
    lines.append("")
    new = list(new)
    if run is None and ep.get("episode_status") != "skipped":
        lines.append("**New writer:** not distilled yet (no run recorded).")
    else:
        lines.append(f"**New writer ({len(new)} memories):**")
        for m in new:
            flag = " (unconfirmed)" if m["confirmation_state"] == "pending_confirmation" else ""
            stakes = f", high: {m['stakes_reason'] or 'no category'}" if m["stakes"] == "high" else ""
            lines.append(f"- [{m['purpose']}, {_VOICE_LABEL.get(m['voice'], m['voice'])}/{m['channel']}{stakes}]"
                         f"{flag} {m['statement']}")
        if not new:
            lines.append("- (none)")
    ev = list(events)
    if ev:
        lines.append("")
        lines.append("Validator: " + "; ".join(f"{e['op']} {e['reason']} x{e['n']}" for e in ev) + ".")
    if run:
        lines.append(
            f"Cost: {run.get('prompt_tokens')} tokens in, {run.get('completion_tokens')} out, "
            f"model {run.get('llm_latency_ms')} ms, hold wait {run.get('hold_wait_ms')} ms, "
            f"coverage {run.get('coverage')}."
        )
    lines.append("")
    return lines


async def render_referents(conn: Any, *, start: datetime, end: datetime) -> list[str]:
    """The referent step's own trace: nodes minted, names admitted (and by which rule), the
    identity questions still open, and co-occurrence claims held back for review."""
    try:
        nodes = [dict(r) for r in await conn.fetch(REFERENT_NODES_SQL, start, end)]
        aliases = [dict(r) for r in await conn.fetch(REFERENT_ALIASES_SQL, start, end)]
        questions = [dict(r) for r in await conn.fetch(IDENTITY_QUESTIONS_SQL)]
        held = int((await conn.fetchrow(HELD_CLAIMS_SQL, start, end))["n"])
    except Exception as exc:  # noqa: BLE001 - the episode report must not depend on Stage 2 tables
        return ["## Referents", "", f"Unavailable ({type(exc).__name__}): the Stage 2 migrations may not be applied.", ""]
    lines = ["## Referents", ""]
    lines.append("Things: " + ("; ".join(f"{r['n']} {r['promotion_state']} ({r['admitted_by']})" for r in nodes)
                               or "none new") + ".")
    lines.append("Names: " + ("; ".join(f"{r['n']} {r['alias_class']} {r['promotion_state']} ({r['admitted_by']})"
                                        for r in aliases) or "none new") + ".")
    lines.append(f"Co-occurrence claims held for review (not walkable): {held}.")
    lines.append(f"Open identity questions ({len(questions)}):")
    lines += [f"- [{q['reason']}, ask via {q['answer_via']}] {q['text']}" for q in questions] or ["- (none)"]
    lines.append("")
    return lines


async def build_report(conn: Any, *, start: datetime, end: datetime, tz_name: str = "America/Denver") -> str:
    """Markdown for shadow episodes closed in [start, end). ``conn`` is an asyncpg connection/pool."""
    tz = ZoneInfo(tz_name)
    episodes = [dict(r) for r in await conn.fetch(EPISODES_SQL, start, end)]
    lines = [
        f"# Memory: old intake vs new episode writer, {start.astimezone(tz):%Y-%m-%d %H:%M} to "
        f"{end.astimezone(tz):%Y-%m-%d %H:%M} ({tz_name})",
        "",
        f"{len(episodes)} episodes closed. SHADOW: the new memories are not used by anything yet.",
        "",
    ]
    totals = {"old": 0, "new": 0}
    for ep in episodes:
        ids = _turn_ids(ep.get("turns"))
        old = [dict(r) for r in await conn.fetch(OLD_ROWS_SQL, ids)] if ids else []
        new = [dict(r) for r in await conn.fetch(NEW_ROWS_SQL, ep["episode_id"])]
        events = [dict(r) for r in await conn.fetch(EVENTS_SQL, ep["episode_id"])]
        run = await conn.fetchrow(RUN_SQL, ep["episode_id"])
        totals["old"] += len(old)
        totals["new"] += len(new)
        lines += render_episode(ep, old, new, events, dict(run) if run else None, tz)
    lines += await render_referents(conn, start=start, end=end)
    lines.insert(4, f"Totals: {totals['old']} old crystallization rows, {totals['new']} new memories.")
    lines.insert(5, "")
    return "\n".join(lines).rstrip() + "\n"
