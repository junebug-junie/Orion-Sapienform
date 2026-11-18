#!/usr/bin/env python3
"""Backfill-distill the Hub chat that predates the episode tracker into episode_memory (issue #3537).

``episode_memory`` starts 2026-10-04 because the episode tracker (memory-consolidation's
``memory_episode_shadow``) started on 2026-10-02. Juniper's Hub chat from July to then was never
grouped into episodes or distilled. This script does that once, on the live code path:

* **Which turns.** Every ``chat_history_log`` row older than the tracker's first episode that the
  live path would have published as a memory turn (sql-writer ``_memory_turn_initiator``: Juniper
  wrote a prompt, or Orion's own unsolicited outreach; never a Claude-room row; a non-empty
  response; a direct Hub conversation, no external platform), minus rows any tracked episode already
  holds and smoke/deploy-check sessions.
* **Grouping.** The tracker's own Rule 3 (``episode_shadow.episode_boundary`` and
  ``build_closed_event``, imported, not copied), fed the same per-session wall-clock phase
  (``orion.situational.context.classify_conversation_phase``). Old turns carry no LLM boundary
  score, so a ``resumed_thread`` turn never splits (what the live rule does with no score).
* **Distill.** ``orion.memory.episode.distill`` prompt + parser, ``validate_distillation``,
  ``store.persist_episode`` with the live referent policy: the same rows (memories, evidence,
  ``episode_memory_referent``, events, questions, aliases, the run row) a live distill writes.
  No write-time supersede: that looks at rows written earlier, and the one-time pass
  (``scripts/memory_supersede_pass.py``) covers the backfilled rows at the end.
* **Time.** Every row is written as of its episode's last turn (``persist_episode(now=...)``), so
  ``created_at`` keeps episode order. The supersede pass orders by ``created_at``: a July row
  stamped "today" would look newer than a November correction and could retire it.
  ``memory_episode_shadow.created_at`` keeps the true write time.
* **GPU.** One ``operator_agent`` hold (``orion.evals.leased_llm``, issue #3391): the same agent-role
  model the ``memory_distill`` route serves, queued behind all of Orion's own work, recallable by
  any owner, given back every ``--max-hold-sec`` so Orion's acts interleave.
* **Idempotent and resumable.** Deterministic episode ids; an episode with an
  ``episode_distill_run`` row is skipped; every insert is ON CONFLICT DO NOTHING; each model answer
  is cached (mode 600) so a resumed run never re-asks.

Private files live in ``/tmp/memory-backfill-jul-sep/`` (dir 700, files 600). Nothing here prints
chat or memory text: progress, report and CSV carry ids and counts only.

    plan      snapshot the targets, group them, write episodes.json      (no DB writes, no GPU)
    dry-run   distill ~10 exchanges, validate, referent coverage          (no DB writes)
    run       distill + persist every episode (resumable)
    gate      per-week referent coverage of the written rows (exit 1 = below the live band)
    report    report.md + before_after.csv
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import csv
import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Optional

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OUT = Path("/tmp/memory-backfill-jul-sep")
ISSUE = "3537"
RUN_PREFIX = f"backfill{ISSUE}-"
HOLDER = f"operator:memory-backfill-{ISSUE}"
EPISODE_NS = uuid.UUID("6b1f3c1e-3537-4d2b-9a8e-6d656d6f7279")
SNAPSHOT_MAX_ROWS = 100_000
TZ = "America/Denver"
# Live values (services/orion-memory-consolidation/.env_example, checked in the running container).
BOUNDARY_SETTINGS = SimpleNamespace(
    MEMORY_EPISODE_ORION_CLOSE_GAP_SEC=10800,
    MEMORY_EPISODE_ORION_REPLY_WINDOW_SEC=86400,
    MEMORY_BOUNDARY_OVERRIDE_THRESHOLD=0.92,
    MEMORY_WINDOW_FALLBACK_GAP_SEC=5400,
)
# Live distill call (orion-durable-runs: MEMORY_EPISODE_DISTILL_*; runner._call_frozen_evidence_llm).
MAX_TOKENS = 8192
TIMEOUT_SEC = 600.0
TEMPERATURE = 0.2
MAX_ATTEMPTS = 4          # episode_distill_graph.EPISODE_DISTILL_MIN_ATTEMPTS
# Spec band (core-memory-pages design, "Pair detection"; #3537 comments): live weeks since 10-05.
BAND_ANY_MIN = 0.97
BAND_OTHER_MIN = 0.68
GATE_MIN_WEEK_ROWS = 10   # a smaller week is reported, and pooled into the overall check
SELF_KEYS = frozenset({"person:juniper", "person:orion"})   # orion.memory.referents.resolve
SMOKE_SESSION = re.compile(r"smoke|deploy-check", re.IGNORECASE)

TARGETS_SQL = """
SELECT id, correlation_id, source, session_id, prompt, response, created_at, client_meta
FROM chat_history_log
WHERE created_at < %(cutoff)s
ORDER BY created_at, correlation_id
"""
TRACKED_IDS_SQL = """
SELECT DISTINCT t->>'correlation_id' AS cid
FROM memory_episode_shadow e, jsonb_array_elements(e.turns) t
"""
CUTOFF_SQL = "SELECT min(started_at) AS cutoff FROM memory_episode_shadow"
# The turn that would have closed the last backfilled episode: the first one the tracker saw.
FIRST_TRACKED_SQL = """
SELECT t->>'correlation_id' AS correlation_id, (t->>'at')::timestamptz AS at
FROM memory_episode_shadow e, jsonb_array_elements(e.turns) t
ORDER BY (t->>'at')::timestamptz LIMIT 1
"""
INSERT_SHADOW_SQL = """
INSERT INTO memory_episode_shadow
  (episode_id, source_platform, status, episode_status, skip_reason, boundary_rule, turns, started_at,
   last_turn_at, closed_at, closing_correlation_id, close_reason, phase_at_close, boundary_score_at_close,
   close_lag_sec, juniper_turn_count, command_turn_count, closed_event_published_at)
VALUES (%(episode_id)s, NULL, 'closed', %(episode_status)s, %(skip_reason)s, 'v2', %(turns)s, %(started_at)s,
        %(last_turn_at)s, %(closed_at)s, %(closing_turn_id)s, %(close_reason)s, %(phase_at_close)s, NULL,
        %(close_lag_sec)s, %(juniper_turn_count)s, %(command_turn_count)s, now())
ON CONFLICT (episode_id) DO NOTHING
"""
# closed_event_published_at = now(): memory-consolidation's re-publisher (unpublished_closed) never
# sends these, so the live distiller (and its write-time supersede) never runs on them.
DONE_SQL = "SELECT episode_id FROM episode_distill_run WHERE episode_id = ANY(%s)"
COVERAGE_SQL = """
SELECT date_trunc('week', m.created_at AT TIME ZONE 'UTC')::date AS week,
       count(*) AS n,
       count(*) FILTER (WHERE EXISTS (SELECT 1 FROM episode_memory_referent r WHERE r.memory_id = m.memory_id))
         AS any_ref,
       count(*) FILTER (WHERE EXISTS (SELECT 1 FROM episode_memory_referent r WHERE r.memory_id = m.memory_id
                                      AND NOT (r.referent_key = ANY(%(self_keys)s)))) AS other_ref
FROM episode_memory m
WHERE m.run_id LIKE %(run_like)s
GROUP BY 1 ORDER BY 1
"""
COUNTS_SQL = {
    "episode_memory": "SELECT count(*) AS n FROM episode_memory",
    "episode_memory_backfill": "SELECT count(*) AS n FROM episode_memory WHERE run_id LIKE %(run_like)s",
    "episode_memory_pre_oct": "SELECT count(*) AS n FROM episode_memory WHERE created_at < '2026-10-02'",
    "episode_memory_referent_backfill": (
        "SELECT count(*) AS n FROM episode_memory_referent r JOIN episode_memory m USING (memory_id) "
        "WHERE m.run_id LIKE %(run_like)s"),
    "episode_memory_evidence_backfill": (
        "SELECT count(*) AS n FROM episode_memory_evidence e JOIN episode_memory m USING (memory_id) "
        "WHERE m.run_id LIKE %(run_like)s"),
    "episode_distill_run_backfill": "SELECT count(*) AS n FROM episode_distill_run WHERE run_id LIKE %(run_like)s",
    "memory_episode_shadow_pre_cutoff": (
        "SELECT count(*) AS n FROM memory_episode_shadow WHERE started_at < '2026-10-02 08:44:39+00'"),
    "rejected_invalid_events_backfill": (
        "SELECT count(*) AS n FROM episode_memory_event WHERE op = 'rejected_invalid' "
        "AND evidence->>'run_id' LIKE %(run_like)s"),
    "pending_confirmation_high_backfill": (
        "SELECT count(*) AS n FROM episode_memory WHERE run_id LIKE %(run_like)s "
        "AND stakes = 'high' AND confirmation_state = 'pending_confirmation'"),
    "follow_up_unexpired_backfill": (
        "SELECT count(*) AS n FROM episode_memory WHERE run_id LIKE %(run_like)s "
        "AND purpose = 'follow_up' AND (expires_at IS NULL OR expires_at > now())"),
    "superseded_rows": "SELECT count(*) AS n FROM episode_memory WHERE status = 'superseded'",
}


# ----------------------------------------------------------------------------------- private files

def private_dir(path: Path) -> Path:
    os.umask(0o077)
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    return path


def write_private(path: Path, text: str) -> None:
    private_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.chmod(tmp, 0o600)
    tmp.replace(path)


def default_dsn() -> str:
    if os.environ.get("ORION_PAGES_DSN"):
        return os.environ["ORION_PAGES_DSN"]
    for base in (REPO, Path("/mnt/scripts/Orion-Sapienform")):
        env = base / "services/orion-durable-runs/.env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("POSTGRES_URI="):
                    return line.split("=", 1)[1].strip().replace("@orion-athena-sql-db:5432", "@localhost:55432")
    raise SystemExit("no DSN: set ORION_PAGES_DSN")


def _utc(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# ----------------------------------------------------------------------------------- live rules

def memory_turn_initiator(prompt: Any, client_meta: Any) -> Optional[str]:
    """sql-writer ``_memory_turn_initiator`` (services/orion-sql-writer/app/worker.py), the rule that
    decides which chat rows become memory turns. Kept equal by test_backfill_jul_sep_script."""
    meta = client_meta
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:  # noqa: BLE001
            meta = None
    if isinstance(meta, dict) and meta.get("room_claude"):
        return None
    if str(prompt or "").strip():
        return "juniper"
    # Live checks the RAW value here, not the parsed one: a JSON-string client_meta is never outreach.
    if isinstance(client_meta, dict) and client_meta.get("unsolicited") is True:
        return "orion"
    return None


def source_platform(client_meta: Any) -> Optional[str]:
    """sql-writer ``_chat_source_platform``: None = a direct Hub conversation."""
    meta = client_meta
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:  # noqa: BLE001
            return None
    if not isinstance(meta, dict) or not isinstance(meta.get("external_room"), dict):
        return None
    platform = meta["external_room"].get("platform")
    return str(platform) if platform else None


def select_targets(rows: Iterable[dict[str, Any]], tracked_ids: set[str]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """chat rows -> the memory turns to backfill (time order), plus why each other row was left out."""
    kept: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    seen: set[str] = set()
    for r in rows:
        cid = str(r.get("correlation_id") or "")
        initiator = memory_turn_initiator(r.get("prompt"), r.get("client_meta"))
        if not cid:
            skip("no_correlation_id")
        elif initiator is None:
            skip("not_a_memory_turn")
        elif not str(r.get("response") or "").strip():
            skip("no_response")
        elif source_platform(r.get("client_meta")) is not None:
            skip("external_platform")
        elif cid in tracked_ids:
            skip("already_in_a_tracked_episode")
        elif SMOKE_SESSION.search(str(r.get("session_id") or "")):
            skip("smoke_session")
        elif cid in seen:
            skip("duplicate_correlation_id")
        else:
            seen.add(cid)
            kept.append({**r, "initiated_by": initiator})
    kept.sort(key=lambda r: (_utc(r["created_at"]), str(r["correlation_id"])))
    return kept, skipped


def _load_live_boundary():
    """The tracker's own boundary code (services/orion-memory-consolidation/app/episode_shadow.py).
    Its package is named ``app``, like every service's, so it is imported with the service on the
    path and the ``app`` entries are put back afterwards."""
    svc = str(REPO / "services" / "orion-memory-consolidation")
    saved = {k: v for k, v in sys.modules.items() if k == "app" or k.startswith("app.")}
    for k in saved:
        del sys.modules[k]
    sys.path.insert(0, svc)
    try:
        import app.episode_shadow as shadow  # noqa: PLC0415
        return shadow
    finally:
        sys.path.remove(svc)
        for k in [k for k in sys.modules if k == "app" or k.startswith("app.")]:
            del sys.modules[k]
        sys.modules.update(saved)


def episode_id_for(first_correlation_id: str) -> str:
    return str(uuid.uuid5(EPISODE_NS, f"backfill{ISSUE}|{first_correlation_id}"))


def run_id_for(episode_id: str) -> str:
    return f"{RUN_PREFIX}{episode_id}"


@dataclass
class _Open:
    episode_id: str
    turns: list[dict[str, Any]]
    started_at: datetime
    last_turn_at: datetime


def group_episodes(targets: list[dict[str, Any]], *, closing_turn: Optional[dict[str, Any]] = None,
                   tz_name: str = TZ, shadow_mod: Any = None) -> list[dict[str, Any]]:
    """Targets (time order) -> closed episodes, exactly as the tracker's observe_turn would group
    them: one open episode for direct Hub chat, Rule 3 + Orion's-own-message rules at each turn.
    ``closing_turn`` ({"correlation_id", "at"}) is the first turn the live tracker saw; it closes the
    last episode (the tracker started fresh, so it never joined)."""
    from orion.situational.context import classify_conversation_phase

    sh = shadow_mod or _load_live_boundary()
    last_juniper_at: dict[str, datetime] = {}   # session -> Juniper's last turn (session_turn_state)
    out: list[dict[str, Any]] = []
    cur: Optional[_Open] = None

    def shadow_turn(r: dict[str, Any]):
        at = _utc(r["created_at"])
        session = str(r.get("session_id") or "global")
        phase = classify_conversation_phase(last_juniper_at.get(session), at, tz_name)
        if r["initiated_by"] == "juniper":
            last_juniper_at[session] = at
        return sh.ShadowTurn(
            correlation_id=str(r["correlation_id"]), at=at, phase_change=phase["phase_change"],
            delta_user_seconds=phase["delta_user_seconds"], phase_source="backfill_recomputed",
            boundary_score=None, is_command=sh.is_workflow_command_turn(r.get("response")),
            legacy_close_reason=None, initiated_by=r["initiated_by"])

    def close(closing, reason: str) -> None:
        event = sh.build_closed_event(episode_id=cur.episode_id, source_platform=None, turns=cur.turns,
                                      started_at=cur.started_at, closing=closing, close_reason=reason)
        out.append({**event.model_dump(mode="json"), "turns": cur.turns,
                    "last_turn_at": cur.last_turn_at.isoformat()})

    for r in targets:
        st = shadow_turn(r)
        if cur is None:
            cur = _Open(episode_id_for(st.correlation_id), [st.entry(v2_boundary=False, v2_reason="v2:first_turn")],
                        st.at, st.at)
            continue
        gap = (st.at - cur.last_turn_at).total_seconds()
        is_boundary, reason = sh.episode_boundary(cur.turns, st, gap, BOUNDARY_SETTINGS)
        if not is_boundary:
            cur.turns.append(st.entry(v2_boundary=False, v2_reason=reason))
            cur.last_turn_at = max(cur.last_turn_at, st.at)
            continue
        close(st, reason)
        cur = _Open(episode_id_for(st.correlation_id), [st.entry(v2_boundary=True, v2_reason=reason)], st.at, st.at)
    if cur is not None:
        if closing_turn is not None:
            at = _utc(closing_turn["at"])
            closing = sh.ShadowTurn(correlation_id=str(closing_turn["correlation_id"]), at=at, phase_change=None,
                                    delta_user_seconds=None, phase_source="backfill_recomputed", boundary_score=None,
                                    is_command=False, legacy_close_reason=None)
        else:
            closing = sh.ShadowTurn(correlation_id=cur.turns[-1]["correlation_id"], at=cur.last_turn_at,
                                    phase_change=None, delta_user_seconds=None, phase_source="backfill_recomputed",
                                    boundary_score=None, is_command=False, legacy_close_reason=None)
        close(closing, "backfill:tracker_started")
    return out


def referent_coverage(memories: Iterable[Any]) -> dict[str, Any]:
    """memories with ``.referents`` [(key, role)] -> any / non-self coverage (the gate's measure)."""
    n = any_ref = other = 0
    for m in memories:
        keys = {k for k, _ in (m.referents or [])}
        n += 1
        any_ref += bool(keys)
        other += bool(keys - SELF_KEYS)
    return {"n": n, "any": any_ref, "other": other,
            "any_rate": (any_ref / n) if n else None, "other_rate": (other / n) if n else None}


def gate_verdict(weeks: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-week rows {week, n, any_ref, other_ref} -> pass/fail against the live band. Every week with
    at least GATE_MIN_WEEK_ROWS rows must be in band, and so must the pooled total."""
    failures = []
    for w in weeks:
        n = int(w["n"])
        if n < GATE_MIN_WEEK_ROWS:
            continue
        a, o = int(w["any_ref"]) / n, int(w["other_ref"]) / n
        if a < BAND_ANY_MIN or o < BAND_OTHER_MIN:
            failures.append({"week": str(w["week"]), "n": n, "any_rate": round(a, 3), "other_rate": round(o, 3)})
    total = sum(int(w["n"]) for w in weeks)
    pooled_any = sum(int(w["any_ref"]) for w in weeks) / total if total else None
    pooled_other = sum(int(w["other_ref"]) for w in weeks) / total if total else None
    pooled_ok = total > 0 and pooled_any >= BAND_ANY_MIN and pooled_other >= BAND_OTHER_MIN
    return {"pass": bool(pooled_ok and not failures), "rows": total,
            "pooled_any_rate": None if pooled_any is None else round(pooled_any, 3),
            "pooled_other_rate": None if pooled_other is None else round(pooled_other, 3),
            "week_failures": failures,
            "small_weeks": [str(w["week"]) for w in weeks if int(w["n"]) < GATE_MIN_WEEK_ROWS]}


def occurred_after_episode(memories: Iterable[Any], ended_at: datetime, slack: timedelta = timedelta(days=1)) -> int:
    """How many non-follow-up memories claim to have happened after their episode ended (a bad date)."""
    return sum(1 for m in memories if m.occurred_at is not None and m.purpose != "follow_up"
               and _utc(m.occurred_at) > ended_at + slack)


def dry_run_sample(episodes: list[dict[str, Any]], min_exchanges: int = 10) -> list[dict[str, Any]]:
    """Evenly spaced closed episodes across the range until they hold at least ``min_exchanges`` of
    Juniper's turns (deterministic; the earliest July days alone are not representative)."""
    closed = [e for e in episodes if e["episode_status"] == "closed"]
    if not closed:
        return []
    picked: list[dict[str, Any]] = []
    for step in (8, 6, 4, 3, 2, 1):
        picked = []
        total = 0
        for e in closed[::max(1, len(closed) // step)]:
            picked.append(e)
            total += int(e["juniper_turn_count"])
            if total >= min_exchanges:
                return picked
    return picked


# ----------------------------------------------------------------------------------- progress

class Progress:
    def __init__(self, path: Path, total: int, total_turns: int) -> None:
        self.path, self.total, self.total_turns = path, total, total_turns
        self.started = time.monotonic()
        self.done = self.turns = self.errors = 0
        private_dir(path.parent)
        if not path.exists():
            write_private(path, "")

    def line(self, title: str, anomalies: str = "") -> None:
        elapsed = time.monotonic() - self.started
        rate = self.done / elapsed if elapsed > 0 and self.done else 0.0
        eta = (self.total - self.done) / rate if rate > 0 else float("nan")
        pct = 100.0 * self.done / self.total if self.total else 100.0
        msg = (f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {title} | {pct:5.1f}% | "
               f"ETA {eta:,.0f}s | episodes {self.done}/{self.total} turns {self.turns}/{self.total_turns} | "
               f"{rate * 3600:.1f} episodes/h | errors {self.errors}"
               f"{' | ' + anomalies if anomalies else ''}\n")
        with self.path.open("a", encoding="utf-8") as f:
            f.write(msg)
        print(msg, end="", flush=True)


# ----------------------------------------------------------------------------------- db

class OneShotPool:
    """``pool.connection()`` for persist_episode / candidate_referent_keys: a fresh autocommit,
    dict-row async connection per use (the script is sequential)."""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    @contextlib.asynccontextmanager
    async def connection(self):
        import psycopg
        from psycopg.rows import dict_row

        conn = await psycopg.AsyncConnection.connect(self.dsn, autocommit=True, row_factory=dict_row)
        try:
            yield conn
        finally:
            await conn.close()


async def _fetch(pool: OneShotPool, sql: str, params: Any = None) -> list[dict[str, Any]]:
    async with pool.connection() as conn:
        return list(await (await conn.execute(sql, params)).fetchall())


# ----------------------------------------------------------------------------------- distill

def _cache_path(episode_id: str) -> Path:
    return OUT / "answers" / f"{episode_id}.json"


def llm_body(prompt: str) -> dict[str, Any]:
    """The live distill call's options as an OpenAI body (json_object_only, thinking off)."""
    return {"messages": [{"role": "user", "content": prompt}], "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE, "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": False}}


def distill_answer(lane: Any, episode: dict[str, Any], turns: list[Any], candidates: list[str]) -> dict[str, Any]:
    """One attempt: prompt -> gateway (under the hold) -> parsed check. Raises on a failed attempt."""
    from orion.memory.episode.distill import parse_distillation, render_prompt, template_prompt_version

    version = template_prompt_version()
    prompt = render_prompt(episode_id=episode["episode_id"], turns=turns, candidate_referents=candidates)
    started = time.monotonic()
    reply = lane.chat(llm_body(prompt), timeout_sec=TIMEOUT_SEC)
    choice = (reply.get("choices") or [{}])[0]
    text = str((choice.get("message") or {}).get("content") or "")
    if not text.strip():
        raise ValueError("empty_generation")
    if choice.get("finish_reason") == "length":
        raise ValueError(f"distill_truncated_at_max_tokens:{MAX_TOKENS}")
    parse_distillation(text)
    usage = reply.get("usage") or {}
    return {"answer_text": text, "prompt_version": version, "model": reply.get("model"),
            "usage": {k: usage.get(k) for k in ("prompt_tokens", "completion_tokens")},
            "latency_ms": int((time.monotonic() - started) * 1000), "candidates": candidates}


def validated_for(episode: dict[str, Any], turns: list[Any], answer: dict[str, Any]):
    from orion.memory.episode.distill import parse_distillation
    from orion.memory.episode.validate import validate_distillation

    return validate_distillation(parse_distillation(answer["answer_text"]), turns, episode_id=episode["episode_id"],
                                 prompt_version=answer["prompt_version"],
                                 known_referents=answer.get("candidates") or [])


async def load_turns(pool: OneShotPool, episode: dict[str, Any]) -> list[Any]:
    from orion.memory.episode.distill import LOAD_TURNS_SQL, turns_from_rows

    rows = await _fetch(pool, LOAD_TURNS_SQL, (list(episode["turn_ids"]),))
    return turns_from_rows(rows)


def referent_policy():
    """orion-durable-runs ``_referent_policy`` with the live flags (all true in the running container)."""
    from orion.memory.referents.resolve import ReferentPolicy

    return ReferentPolicy(grounding_auto_accept=True, cooccurrence_auto_accept=True)


async def persist(pool: OneShotPool, episode: dict[str, Any], turns: list[Any], answer: dict[str, Any]) -> dict[str, Any]:
    from orion.memory.episode.store import persist_episode
    from orion.memory.episode.validate import coverage

    result = validated_for(episode, turns, answer)
    now = _utc(episode["last_turn_at"])
    async with pool.connection() as conn:
        await conn.execute(INSERT_SHADOW_SQL, shadow_params(episode))
    counts = await persist_episode(
        pool, episode_id=episode["episode_id"], run_id=run_id_for(episode["episode_id"]), result=result,
        model_route="operator_agent", model=answer.get("model"), prompt_version=answer["prompt_version"],
        usage=answer.get("usage") or {}, llm_latency_ms=answer.get("latency_ms"), hold_wait_ms=None,
        coverage=coverage(result, turns)["coverage"], now=now, referent_policy=referent_policy())
    return {"counts": counts, "result": result}


def shadow_params(episode: dict[str, Any]) -> dict[str, Any]:
    from psycopg.types.json import Jsonb

    return {
        "episode_id": episode["episode_id"], "episode_status": episode["episode_status"],
        "skip_reason": episode.get("skip_reason"), "turns": Jsonb(episode["turns"]),
        "started_at": episode["started_at"], "last_turn_at": episode["last_turn_at"],
        "closed_at": episode["closed_at"], "closing_turn_id": episode.get("closing_turn_id"),
        "close_reason": episode.get("close_reason"), "phase_at_close": episode.get("phase_at_close"),
        "close_lag_sec": episode.get("close_lag_sec"), "juniper_turn_count": episode.get("juniper_turn_count"),
        "command_turn_count": episode.get("command_turn_count"),
    }


# ----------------------------------------------------------------------------------- commands

def _load_plan() -> dict[str, Any]:
    path = OUT / "episodes.json"
    if not path.exists():
        raise SystemExit(f"no plan: run `plan` first ({path})")
    return json.loads(path.read_text(encoding="utf-8"))


async def cmd_plan(args: argparse.Namespace) -> int:
    pool = OneShotPool(args.dsn)
    private_dir(OUT)
    cutoff = (await _fetch(pool, CUTOFF_SQL))[0]["cutoff"]
    tracked = {r["cid"] for r in await _fetch(pool, TRACKED_IDS_SQL)}
    rows = await _fetch(pool, TARGETS_SQL, {"cutoff": cutoff})
    if len(rows) > SNAPSHOT_MAX_ROWS:
        raise SystemExit(f"{len(rows)} rows exceed the snapshot limit {SNAPSHOT_MAX_ROWS}: ask Juniper")
    targets, skipped = select_targets(rows, tracked)
    first = await _fetch(pool, FIRST_TRACKED_SQL)
    episodes = group_episodes(targets, closing_turn=first[0] if first else None)
    # Snapshot: the full target rows (private text), so the run can be audited against what it read.
    with (OUT / "targets.jsonl.tmp").open("w", encoding="utf-8") as f:
        os.chmod(OUT / "targets.jsonl.tmp", 0o600)
        for r in targets:
            f.write(json.dumps(r, default=str) + "\n")
    (OUT / "targets.jsonl.tmp").replace(OUT / "targets.jsonl")
    by_status: dict[str, int] = {}
    for e in episodes:
        key = e["episode_status"] if e["episode_status"] == "closed" else f"skipped:{e.get('skip_reason')}"
        by_status[key] = by_status.get(key, 0) + 1
    summary = {
        "cutoff": str(cutoff), "chat_rows_before_cutoff": len(rows), "targets": len(targets),
        "targets_juniper": sum(1 for t in targets if t["initiated_by"] == "juniper"),
        "targets_orion": sum(1 for t in targets if t["initiated_by"] == "orion"),
        "targets_before_oct": sum(1 for t in targets if _utc(t["created_at"]) < datetime(2026, 10, 1, tzinfo=timezone.utc)),
        "skipped": skipped, "episodes": len(episodes), "episodes_by_status": by_status,
        "first_turn": str(targets[0]["created_at"]) if targets else None,
        "last_turn": str(targets[-1]["created_at"]) if targets else None,
    }
    write_private(OUT / "episodes.json", json.dumps({"summary": summary, "episodes": episodes}, default=str, indent=1))
    print(json.dumps(summary, indent=1, default=str))
    return 0


def _hold_loop(args: argparse.Namespace, work: list[dict[str, Any]], handle_episode) -> None:
    """Run ``handle_episode(lane, episode)`` over ``work`` under operator holds that are given back
    every ``max_hold_sec`` (and after a recall), so Orion's own acts interleave."""
    from orion.evals.leased_llm import LaneRecalled, operator_lane_sync

    queue = list(work)
    while queue:
        granted = time.monotonic()
        try:
            with operator_lane_sync("agent", holder=HOLDER, bus_url=args.bus_url, gateway=args.gateway,
                                    queue_wait_sec=args.queue_wait_sec) as lane:
                while queue and time.monotonic() - granted < args.max_hold_sec:
                    handle_episode(lane, queue[0])
                    queue.pop(0)
        except LaneRecalled:
            continue   # the episode at the head of the queue is re-tried under the next hold


def _answer_with_retries(lane: Any, episode: dict[str, Any], turns: list[Any], candidates: list[str],
                         prog: Progress) -> Optional[dict[str, Any]]:
    from orion.evals.leased_llm import LaneRecalled

    cache = _cache_path(episode["episode_id"])
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    last = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            answer = distill_answer(lane, episode, turns, candidates)
            write_private(cache, json.dumps(answer))
            return answer
        except LaneRecalled:
            raise
        except Exception as exc:  # noqa: BLE001 -- a failed attempt: bounded, backed off
            last = f"{type(exc).__name__}: {str(exc)[:160]}"
            prog.line(f"attempt_failed episode={episode['episode_id']} attempt={attempt}",
                      anomalies=f"error={type(exc).__name__}")
            time.sleep(min(60.0, 5.0 * 2 ** (attempt - 1)))
    prog.errors += 1
    _append_error(episode["episode_id"], "distill_failed", last or "")
    return None


def _append_error(episode_id: str, stage: str, detail: str) -> None:
    path = OUT / "errors.jsonl"
    private_dir(OUT)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"episode_id": episode_id, "stage": stage, "detail": detail,
                            "at": datetime.now(timezone.utc).isoformat()}) + "\n")
    os.chmod(path, 0o600)


async def _candidates(pool: OneShotPool) -> list[str]:
    from orion.memory.episode.store import candidate_referent_keys

    return await candidate_referent_keys(pool)


def cmd_dry_run(args: argparse.Namespace) -> int:
    plan = _load_plan()
    sample = dry_run_sample(plan["episodes"], args.exchanges)
    pool = OneShotPool(args.dsn)
    loop = asyncio.new_event_loop()
    candidates = loop.run_until_complete(_candidates(pool))
    prog = Progress(OUT / "progress.log", len(sample), sum(len(e["turn_ids"]) for e in sample))
    prog.line(f"dry_run_started episodes={len(sample)} exchanges={sum(e['juniper_turn_count'] for e in sample)}")
    results: list[dict[str, Any]] = []
    memories: list[Any] = []

    def handle(lane, episode):
        turns = loop.run_until_complete(load_turns(pool, episode))
        answer = _answer_with_retries(lane, episode, turns, candidates, prog)
        prog.done += 1
        prog.turns += len(turns)
        if answer is None:
            prog.line(f"dry_run_episode_failed episode={episode['episode_id']}")
            return
        result = validated_for(episode, turns, answer)
        memories.extend(result.memories)
        late = occurred_after_episode(result.memories, _utc(episode["last_turn_at"]))
        cov = referent_coverage(result.memories)
        results.append({"episode_id": episode["episode_id"], "turns": len(turns),
                        "juniper_turns": episode["juniper_turn_count"], "memories": len(result.memories),
                        "rejections": len(result.rejections), "questions": len(result.questions),
                        "with_referent": cov["any"], "with_other_referent": cov["other"],
                        "occurred_after_episode": late, "model": answer.get("model")})
        anomalies = []
        if not result.memories:
            anomalies.append("no_memories")
        if late:
            anomalies.append(f"occurred_after_episode={late}")
        prog.line(f"dry_run_episode episode={episode['episode_id']} memories={len(result.memories)} "
                  f"rejected={len(result.rejections)}", anomalies=" ".join(anomalies))

    _hold_loop(args, sample, handle)
    cov = referent_coverage(memories)
    verdict = {
        "pass": cov["n"] > 0 and cov["any_rate"] >= BAND_ANY_MIN and cov["other_rate"] >= BAND_OTHER_MIN,
        **cov, "band": {"any_min": BAND_ANY_MIN, "other_min": BAND_OTHER_MIN},
    }
    out = {"episodes": results, "referent_coverage": verdict, "errors": prog.errors}
    write_private(OUT / "dry_run.json", json.dumps(out, indent=1, default=str))
    prog.line(f"dry_run_done memories={cov['n']} any_rate={cov['any_rate']} other_rate={cov['other_rate']} "
              f"gate={'pass' if verdict['pass'] else 'FAIL'}")
    print(json.dumps(out, indent=1, default=str))
    return 0 if verdict["pass"] else 1


def cmd_run(args: argparse.Namespace) -> int:
    plan = _load_plan()
    episodes = plan["episodes"]
    pool = OneShotPool(args.dsn)
    loop = asyncio.new_event_loop()
    done = {r["episode_id"] for r in loop.run_until_complete(
        _fetch(pool, DONE_SQL, ([e["episode_id"] for e in episodes],)))}
    skipped = [e for e in episodes if e["episode_status"] != "closed"]
    todo = [e for e in episodes if e["episode_status"] == "closed" and e["episode_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    # Skipped episodes (Orion-only, command-only) get their tracker row, as live, and no distill.
    for e in skipped:
        async def put(ep=e):
            async with pool.connection() as conn:
                await conn.execute(INSERT_SHADOW_SQL, shadow_params(ep))
        loop.run_until_complete(put())
    candidates = loop.run_until_complete(_candidates(pool))
    prog = Progress(OUT / "progress.log", len(todo), sum(len(e["turn_ids"]) for e in todo))
    prog.line(f"run_started todo={len(todo)} already_done={len(done)} skipped_episodes={len(skipped)}")

    def handle(lane, episode):
        turns = loop.run_until_complete(load_turns(pool, episode))
        anomalies = []
        if len(turns) != len(episode["turn_ids"]):
            anomalies.append(f"turns_loaded={len(turns)}/{len(episode['turn_ids'])}")
        answer = _answer_with_retries(lane, episode, turns, candidates, prog)
        prog.done += 1
        prog.turns += len(turns)
        if answer is None:
            prog.line(f"episode_failed episode={episode['episode_id']}", anomalies="distill_failed")
            return
        try:
            out = loop.run_until_complete(persist(pool, episode, turns, answer))
        except Exception as exc:  # noqa: BLE001 -- recorded; the episode is retried by a re-run
            prog.errors += 1
            _append_error(episode["episode_id"], "persist_failed", f"{type(exc).__name__}: {str(exc)[:160]}")
            prog.line(f"episode_failed episode={episode['episode_id']}", anomalies=f"persist={type(exc).__name__}")
            return
        result, counts = out["result"], out["counts"]
        late = occurred_after_episode(result.memories, _utc(episode["last_turn_at"]))
        cov = referent_coverage(result.memories)
        if not result.memories:
            anomalies.append("no_memories")
        if late:
            anomalies.append(f"occurred_after_episode={late}")
        if cov["n"] and cov["any"] < cov["n"]:
            anomalies.append(f"memories_without_referent={cov['n'] - cov['any']}")
        if "referents_resolved" not in counts and result.memories:
            anomalies.append("referent_step_skipped")
        prog.line(f"episode_done episode={episode['episode_id']} memories={counts['memories']} "
                  f"rejected={counts['rejections']} questions={counts['questions']} "
                  f"referents_resolved={counts.get('referents_resolved')}", anomalies=" ".join(anomalies))

    _hold_loop(args, todo, handle)
    prog.line(f"run_done processed={prog.done} errors={prog.errors}")
    return 0 if prog.errors == 0 else 1


async def _coverage_weeks(pool: OneShotPool) -> list[dict[str, Any]]:
    return await _fetch(pool, COVERAGE_SQL, {"self_keys": sorted(SELF_KEYS), "run_like": f"{RUN_PREFIX}%"})


def cmd_gate(args: argparse.Namespace) -> int:
    pool = OneShotPool(args.dsn)
    weeks = asyncio.run(_coverage_weeks(pool))
    verdict = gate_verdict(weeks)
    out = {"weeks": [{"week": str(w["week"]), "n": w["n"], "any_ref": w["any_ref"], "other_ref": w["other_ref"]}
                     for w in weeks], **verdict,
           "band": {"any_min": BAND_ANY_MIN, "other_min": BAND_OTHER_MIN, "min_week_rows": GATE_MIN_WEEK_ROWS}}
    write_private(OUT / "referent_gate.json", json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))
    return 0 if verdict["pass"] else 1


async def _counts(pool: OneShotPool) -> dict[str, int]:
    out = {}
    for name, sql in COUNTS_SQL.items():
        out[name] = int((await _fetch(pool, sql, {"run_like": f"{RUN_PREFIX}%"}))[0]["n"])
    return out


def cmd_counts(args: argparse.Namespace) -> int:
    """Write the before/after counts (``--label before`` once before ``run``, ``after`` at the end)."""
    counts = asyncio.run(_counts(OneShotPool(args.dsn)))
    path = OUT / "counts.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data[args.label] = {"at": datetime.now(timezone.utc).isoformat(), **counts}
    write_private(path, json.dumps(data, indent=1))
    print(json.dumps(data[args.label], indent=1))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    counts = json.loads((OUT / "counts.json").read_text())
    before, after = counts.get("before", {}), counts.get("after", {})
    with (OUT / "before_after.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "before", "after", "delta"])
        for k in COUNTS_SQL:
            b, a = before.get(k), after.get(k)
            w.writerow([k, b, a, (a - b) if isinstance(a, int) and isinstance(b, int) else ""])
    os.chmod(OUT / "before_after.csv", 0o600)
    print(f"wrote {OUT / 'before_after.csv'}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dsn", default=None)
    p.add_argument("--bus-url", default=os.environ.get("ORION_BUS_URL"))
    p.add_argument("--gateway", default=os.environ.get("ORION_LLM_GATEWAY_URL"))
    p.add_argument("--queue-wait-sec", type=float, default=6 * 3600.0)
    p.add_argument("--max-hold-sec", type=float, default=900.0)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    d = sub.add_parser("dry-run")
    d.add_argument("--exchanges", type=int, default=10)
    r = sub.add_parser("run")
    r.add_argument("--limit", type=int, default=0)
    sub.add_parser("gate")
    c = sub.add_parser("counts")
    c.add_argument("--label", choices=("before", "after"), required=True)
    sub.add_parser("report")
    args = p.parse_args(argv)
    args.dsn = args.dsn or default_dsn()
    if args.cmd == "plan":
        return asyncio.run(cmd_plan(args))
    if args.cmd in ("dry-run", "run") and not args.bus_url:
        raise SystemExit("ORION_BUS_URL is not set: the GPU pool hold needs the bus")
    return {"dry-run": cmd_dry_run, "run": cmd_run, "gate": cmd_gate, "counts": cmd_counts,
            "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
