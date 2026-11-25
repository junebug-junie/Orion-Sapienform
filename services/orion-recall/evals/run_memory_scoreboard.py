#!/usr/bin/env python3
"""Memory scoreboard runner (issue #3536): how well does Orion remember Juniper's real history?

    make eval-memory-scoreboard                      # both scores, today's date
    python services/orion-recall/evals/run_memory_scoreboard.py --phases retrieval

Two scores over the host-side bank (``~/.orion/evals/memory_scoreboard_bank.json``, mode 600):

1. **Retrieval, no model.** Each item replays orion-recall's two chat phases over the real bus,
   the way cortex-exec's ``pcr_chat_memory.py`` runs them on a chat turn: the phase-0 skip gate,
   phase 1 (``chat.continuity.v1``, where lived memory is injected), then phase 3 (purposeful) with
   the intent's profile. Hit = a target id is in either phase's DURABLE MEMORY bundle or in the
   cites of the current ``page:juniper`` core page. The recall service persists one
   ``recall_telemetry`` row per call (session ``memory-scoreboard-eval``); its ``deadline_hit`` is
   read back to mark degraded recalls, which are reported apart from the score.
2. **Answer, end to end.** Each item is sent as a chat turn through the real Hub chat path
   (``POST /api/chat``, mode ``orion``) with ``no_write: true`` on its own eval session, the same
   way ``services/orion-hub/evals/run_self_sense_eval.py`` does it: nothing lands in chat history,
   so Juniper's conversations and the memory distiller never see it. The reply is graded by a local
   model judge, one GPU pool operator hold for the run (``orion/evals/leased_llm.py``; local lanes
   only, through orion-llm-gateway). The turn's own ``recall_telemetry`` row (same correlation id)
   says whether the target was in its prompt.

Output (host-side, mode 600, never committed: it holds replies):
``~/.orion/evals/memory_scoreboard_runs/<YYYY-MM-DD>.json`` (everything) and ``.md`` (per-category
table, counts and qids only), plus one line appended to ``history.jsonl`` so a weekly run can track
change. Exit 0 = ran; the bars are in the report, not the exit code. Exit 2 = could not run.

Env: ORION_BUS_URL (required, redis://<bus-host>:6379/0), POSTGRES_* or --dsn, HUB_BASE_URL.

**Scoring a branch before deploy** (``--lived-local``, issue #3613). The retrieval phase still
calls the deployed recall service, but the lived-memory lines in its continuity bundle (source
``lived_memory``) are replaced by the lived recall of the code on disk, run in-process with the
live ``RECALL_LIVED_MEMORY_*`` settings against the host-side Chroma, vector host and LLM gateway.
Everything else (the other backends, phase 3, the core page) stays live. ``--lived-local-short-gate
off`` runs the same code with the plain content-word floor, the "before" of a short-gate change.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
for p in (str(HERE.parent), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

from memory_scoreboard import (  # noqa: E402
    JUDGE_SYSTEM,
    Item,
    RetrievalObservation,
    build_judge_prompt,
    degraded_reasons,
    iter_record_ids,
    load_items,
    outcome_for,
    page_refs,
    parse_judge_label,
    render_table,
    score_answers,
    score_retrieval,
    search_causes,
    summary_line,
)

SESSION_ID = "memory-scoreboard-eval"
HOLDER = "operator:memory-scoreboard"
DEFAULT_BANK = "~/.orion/evals/memory_scoreboard_bank.json"
DEFAULT_OUT = "~/.orion/evals/memory_scoreboard_runs"
DEFAULT_HUB = "http://127.0.0.1:8080"
RECALL_CHANNEL = "orion:exec:request:RecallService"
# What cortex-exec sends on a full chat turn (recall_telemetry shows verb stance_react for both
# phases). The two profile names mirror services/orion-cortex-exec/app/pcr_chat_memory.py; the
# fixture test fails if they drift.
VERB = "stance_react"
# cortex-exec CHAT_PCR_SKIP_MAX_NOVELTY / CHAT_PCR_SKIP_SHIFT_NOVELTY_FLOOR defaults (drift-tested).
SKIP_MAX_NOVELTY = 0.25
SKIP_SHIFT_NOVELTY_FLOOR = 0.35
CONTINUITY_PROFILE = "chat.continuity.v1"
PROFILE_FOR_INTENT: dict[str, str] = {
    "relational": "chat.belief.relational.v1",
    "semantic": "chat.belief.semantic.v1",
    "procedural": "chat.belief.procedural.v1",
    "open_loop": "chat.belief.open_loop.v1",
    "contradiction": "chat.belief.contradiction.v1",
}
# Phase 3's intent comes from the stance brief and the attention frame, which only exist inside a
# live turn. On real chat turns this week, open_loop was the phase-3 intent on most turns (open
# loops are nearly always present), so that is the default; ``auto`` derives it from the message
# alone (no stance, no attention frame).
DEFAULT_PHASE3_INTENT = "open_loop"
# orion-recall settings.RECALL_LIVED_MEMORY_TIMEOUT_SEC default; the make target loads the real value.
LIVED_TIMEOUT_SEC_DEFAULT = 4.0
PAGE_SQL = ("SELECT version, content FROM self_concept_history WHERE concept_id = 'page:juniper' "
            "AND produced_by = 'core_page_fold' ORDER BY created_at DESC LIMIT 1")
RECORD_SQL = {
    "episode_memory": "SELECT statement FROM episode_memory WHERE memory_id::text = $1",
    "stance": "SELECT summary FROM memory_crystallizations WHERE crystallization_id::text = $1",
    "person_fact": "SELECT value_text FROM person_fact WHERE fact_id = $1",
}


def _dsn(arg: Optional[str]) -> str:
    if arg:
        return arg
    if os.getenv("ORION_SCOREBOARD_DSN"):
        return os.environ["ORION_SCOREBOARD_DSN"]
    pw = os.getenv("POSTGRES_PASSWORD")
    if not pw:
        raise SystemExit("no DB: pass --dsn, or set ORION_SCOREBOARD_DSN or POSTGRES_PASSWORD (repo .env)")
    return (f"postgresql://{os.getenv('POSTGRES_USER', 'postgres')}:{pw}@127.0.0.1:"
            f"{os.getenv('POSTGRES_PORT', '55432')}/{os.getenv('POSTGRES_DB', 'conjourney')}")


def _write_private(path: Path, text: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.chmod(path, 0o600)


# ---------------------------------------------------------------- retrieval

async def _recall_rpc(bus: Any, *, text: str, profile: str, phase: str, intent: str,
                      deadline_ms: int, timeout_sec: float) -> tuple[Dict[str, Any], str]:
    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
    from orion.core.contracts.recall import RecallQueryV1

    corr = str(uuid4())
    reply = f"orion:exec:result:RecallService:{uuid4()}"
    req = RecallQueryV1(
        fragment=text, verb=VERB, session_id=SESSION_ID, profile=profile, profile_explicit=True,
        exclude={"active_turn_ids": [corr], "active_turn_text": text, "active_turn_ts": time.time()},
        reply_to=reply, recall_phase=phase, retrieval_intent=intent, deadline_ms=deadline_ms,
    )
    payload = req.model_dump(mode="json")
    env = BaseEnvelope(kind="recall.query.v1", source=ServiceRef(name="memory-scoreboard", node="host", version="1"),
                       correlation_id=corr, reply_to=reply, payload=payload)
    msg = await bus.rpc_request(RECALL_CHANNEL, env, reply_channel=reply, timeout_sec=timeout_sec)
    decoded = bus.codec.decode(msg.get("data"))
    if not decoded.ok:
        raise RuntimeError(f"decode: {decoded.error}")
    body = decoded.envelope.payload if isinstance(decoded.envelope.payload, dict) else {}
    if body.get("error"):
        raise RuntimeError(str(body["error"]))
    from orion.core.contracts.recall import RecallReplyV1

    RecallReplyV1.model_validate(body)   # same check RecallClient does on a real turn
    return body, corr


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name) or default)
    except ValueError:
        return default


def build_local_lived(pool: Any, bus: Any, args: argparse.Namespace) -> Any:
    """The lived recall of the code on disk, configured like the live service (``--lived-local``)."""
    service = str(HERE.parents[1])
    if service not in sys.path:
        sys.path.insert(0, service)
    from app.lived_memory import LivedMemoryConfig, LivedMemoryRecall
    from app.lived_rewrite import BusRewriter
    from orion.core.bus.bus_schemas import ServiceRef
    from orion.introspect.semantic_index import SearchConfig
    from orion.memory.lived_memory import COLLECTION

    rewrite_timeout = _env_float("RECALL_LIVED_MEMORY_REWRITE_TIMEOUT_SEC", 2.5)
    rewriter = None
    if str(os.getenv("RECALL_LIVED_MEMORY_REWRITE_ENABLED", "true")).lower() not in ("0", "false", "no"):
        rewriter = BusRewriter(bus, channel=os.getenv("CHANNEL_LLM_INTAKE", "orion:exec:request:LLMGatewayService"),
                               route=os.getenv("RECALL_LIVED_MEMORY_REWRITE_ROUTE", "quick_turn"),
                               timeout_sec=rewrite_timeout,
                               source=ServiceRef(name="orion-recall", version="scoreboard-local", node="eval"),
                               tz=os.getenv("RECALL_LIVED_MEMORY_TIMEZONE", "America/Denver"))
    cfg = LivedMemoryConfig(
        search=SearchConfig(chroma_url=args.lived_chroma_url, embed_url=args.lived_embed_url,
                            collection=os.getenv("RECALL_LIVED_MEMORY_COLLECTION", COLLECTION), min_similarity=0.0),
        rerank_url=args.lived_rerank_url,
        min_fit=_env_float("RECALL_LIVED_MEMORY_MIN_FIT", 0.004),
        max_lines=int(_env_float("RECALL_LIVED_MEMORY_MAX_LINES", 4)),
        timeout_sec=_env_float("RECALL_LIVED_MEMORY_TIMEOUT_SEC", 5.5),
        min_query_tokens=int(_env_float("RECALL_LIVED_MEMORY_MIN_QUERY_TOKENS", 5)),
        short_gate=args.lived_local_short_gate == "on",
        rewrite_timeout_sec=rewrite_timeout,
        link_top_k=int(_env_float("RECALL_LIVED_MEMORY_LINK_TOP_K", 2)),
        time_window=str(os.getenv("RECALL_LIVED_MEMORY_TIME_WINDOW_ENABLED", "true")).lower() not in ("0", "false", "no"),
        timezone=os.getenv("RECALL_LIVED_MEMORY_TIMEZONE", "America/Denver"))
    return LivedMemoryRecall(pool, cfg, rewriter=rewriter)


def asked_at(item: Item, tz: str | None = None) -> Optional[datetime]:
    """Noon on the item's ``as_of`` day in Juniper's zone: "now" for its relative time words
    (--lived-local only; the deployed service always reads the real clock)."""
    if not item.as_of:
        return None
    from datetime import date, time as dtime
    from zoneinfo import ZoneInfo

    try:
        day = date.fromisoformat(str(item.as_of)[:10])
    except ValueError:
        return None
    zone = ZoneInfo(tz or os.getenv("RECALL_LIVED_MEMORY_TIMEZONE") or "America/Denver")
    return datetime.combine(day, dtime(12, 0), tzinfo=zone).astimezone(timezone.utc)


def swap_lived_lines(bundle_items: List[Dict[str, Any]], local_ids: List[str]) -> set[str]:
    """A continuity bundle's ids with the deployed service's lived lines replaced by the local ones."""
    kept = {str(i.get("id")) for i in bundle_items if str(i.get("source") or "") != "lived_memory"}
    return kept | set(local_ids)


async def run_retrieval(items: List[Item], texts: Dict[str, str], pool: Any, args: argparse.Namespace) -> Dict[str, Any]:
    from orion.core.bus.async_service import OrionBusAsync
    from orion.memory.recall_skip_gate import recall_skip_gate
    from orion.memory.retrieval_intent import derive_retrieval_intent

    page = await pool.fetchrow(PAGE_SQL)
    pids = page_refs(page["content"] if page else None)
    bus = OrionBusAsync(url=args.bus_url)
    await bus.connect()
    local = build_local_lived(pool, bus, args) if getattr(args, "lived_local", False) else None
    local_rows: Dict[str, Dict[str, Any]] = {}   # qid -> the local lived search's status and trace
    local_traces: Dict[str, Dict[str, Any]] = {}  # qid -> the local lived trace (kept for the report)
    local_ms: Dict[str, List[int]] = {}   # gate -> lived-recall total ms
    gates: Dict[str, int] = {}
    observations: Dict[str, RetrievalObservation] = {}
    corrs: Dict[str, List[str]] = {}
    rpc_errors: Dict[str, str] = {}
    try:
        for n, it in enumerate(items, 1):
            text = texts[it.qid]
            obs = RetrievalObservation(qid=it.qid, page_ids=set(pids))
            observations[it.qid] = obs
            gate = recall_skip_gate(user_message=text, appraisal=None, has_repair_grammar_signal=False,
                                    max_novelty=SKIP_MAX_NOVELTY, shift_novelty_floor=SKIP_SHIFT_NOVELTY_FLOOR)
            if gate.skip:
                obs.skipped = True
                print(f"  [{n}/{len(items)}] {it.qid}: skip gate", flush=True)
                continue
            try:
                body, corr = await _recall_rpc(bus, text=text, profile=CONTINUITY_PROFILE, phase="continuity",
                                               intent="continuity", deadline_ms=args.deadline_ms,
                                               timeout_sec=args.rpc_timeout)
                corrs.setdefault(it.qid, []).append(corr)
                bundle_items = list((body.get("bundle") or {}).get("items") or [])
                if local is not None:
                    from app.lived_memory import status_of, trace_of

                    # A time item's relative words ("last week") refer to the day it was written.
                    out = await local.recall(text, session_id=SESSION_ID, now=asked_at(it))
                    ids = [s.rec.doc_id for s in out.chosen]
                    local_rows[it.qid] = {"lived_status": status_of(out.debug, len(ids), backend=out.backend),
                                          "lived_trace": trace_of(out.debug), "served": len(ids)}
                    local_traces[it.qid] = dict(local_rows[it.qid]["lived_trace"])
                    g = str(out.debug.get("gate") or "none")
                    local_ms.setdefault(g, []).append(int((out.debug.get("timings_ms") or {}).get("total") or 0))
                    gates[g] = gates.get(g, 0) + 1
                    obs.injected_ids |= swap_lived_lines(bundle_items, ids)
                else:
                    obs.injected_ids |= {str(i.get("id")) for i in bundle_items}
                intent = args.phase3_intent
                if intent == "auto":
                    # No stance brief or attention frame outside a live turn; eligible_belief_count is
                    # only in diagnostic replies, so the brain-lane belief default cannot fire here.
                    intent, _rule = derive_retrieval_intent(
                        skip_gate=gate, stance_brief={}, attention_frame=None, appraisal=None,
                        hub_chat_lane="orion", user_message=text)
                if intent in PROFILE_FOR_INTENT:
                    body3, corr3 = await _recall_rpc(bus, text=text, profile=PROFILE_FOR_INTENT[intent],
                                                     phase="purposeful", intent=intent,
                                                     deadline_ms=args.deadline_ms, timeout_sec=args.rpc_timeout)
                    corrs[it.qid].append(corr3)
                    obs.injected_ids |= {str(i.get("id")) for i in (body3.get("bundle") or {}).get("items") or []}
            except Exception as exc:  # noqa: BLE001 -- a failed recall is reported, not fatal
                rpc_errors[it.qid] = type(exc).__name__
            hit = any(t in obs.injected_ids | obs.page_ids for t in it.targets)
            print(f"  [{n}/{len(items)}] {it.qid}: injected={len(obs.injected_ids)} "
                  f"hit={hit if it.targets else '-'} rpc_error={rpc_errors.get(it.qid, '-')}", flush=True)
    finally:
        await bus.close()
    # recall persists recall_telemetry after it replies; read every call's trace back.
    await asyncio.sleep(3.0)
    lived_timeout_ms = 1000.0 * float(os.getenv("RECALL_LIVED_MEMORY_TIMEOUT_SEC") or LIVED_TIMEOUT_SEC_DEFAULT)
    lived_lines = 0
    for it in items:
        obs = observations[it.qid]
        if obs.skipped:
            continue
        cs = corrs.get(it.qid) or []
        rows = [dict(r) for r in await pool.fetch(
            "SELECT profile, deadline_hit, backend_counts, timings_ms, lived_status, lived_trace FROM recall_telemetry "
            "WHERE corr_id = ANY($1::text[])", cs)] if cs else []
        if it.qid in local_rows:
            # The local lived search replaced the deployed one's lines: read its verdict, not theirs.
            lived_lines += int(local_rows[it.qid].pop("served", 0))
            for r in rows:
                if r.get("profile") == CONTINUITY_PROFILE:
                    r.update(local_rows[it.qid])
        else:
            for r in rows:
                if r.get("profile") == CONTINUITY_PROFILE:
                    counts = r.get("backend_counts")
                    counts = json.loads(counts) if isinstance(counts, str) else (counts or {})
                    lived_lines += int(counts.get("lived_memory") or 0)
        if rows and len(rows) < len(cs):
            obs.degraded.append("trace:partial")
        obs.degraded += degraded_reasons(rows, lived_timeout_ms=lived_timeout_ms, rpc_error=rpc_errors.get(it.qid))
    report = score_retrieval(items, observations)
    report["page"] = {"version": page["version"] if page else None, "cites": len(pids)}
    report["lived_memory_lines_injected"] = lived_lines
    report["phase3_intent"] = args.phase3_intent
    if local is not None:
        def pcts(vals: List[int]) -> Dict[str, int] | None:
            ms = sorted(vals)
            return {"p50": ms[len(ms) // 2], "p90": ms[min(len(ms) - 1, int(0.9 * len(ms)))],
                    "max": ms[-1]} if ms else None

        rewrites: Dict[str, int] = {}
        for row in local_traces.values():
            k = str(row.get("rewrite") or "none")
            rewrites[k] = rewrites.get(k, 0) + 1
        report["lived_local"] = {
            "short_gate": args.lived_local_short_gate, "gates": gates,
            # Rewrite outcomes per run (ok/none/timeout/...): the run-to-run noise source (#3674).
            "rewrite": dict(sorted(rewrites.items())),
            # Per item: rewrite outcome, the time window read, pool size. Ids and dates only.
            "traces": {q: {k: v for k, v in t.items() if k in ("rewrite", "window", "rewrite_window", "gate")}
                       | {"candidates": len(t.get("candidate_ids") or [])} for q, t in sorted(local_traces.items())},
            "total_ms": pcts([m for vals in local_ms.values() for m in vals]),
            "total_ms_by_gate": {g: pcts(v) for g, v in sorted(local_ms.items())},
        }
    return report


# ---------------------------------------------------------------- answer

def wait_for_hub(base_url: str, max_wait_sec: float, poll_sec: float = 10.0) -> None:
    """Return once Hub's /health answers 200; raise after ``max_wait_sec``."""
    import requests

    deadline = time.monotonic() + max(0.0, max_wait_sec)
    while True:
        try:
            if requests.get(base_url.rstrip("/") + "/health", timeout=5).status_code == 200:
                return
        except requests.RequestException:
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Hub not healthy after {max_wait_sec:.0f}s")
        time.sleep(poll_sec)


def ask_hub(base_url: str, text: str, timeout_sec: float) -> Dict[str, Any]:
    import requests

    resp = requests.post(
        base_url.rstrip("/") + "/api/chat",
        json={"messages": [{"role": "user", "content": text}], "mode": "orion", "no_write": True},
        headers={"X-Orion-Session-Id": SESSION_ID, "X-Orion-No-Write": "true"},
        timeout=timeout_sec,
    )
    resp.raise_for_status()
    body = resp.json()
    if not isinstance(body, dict):
        raise RuntimeError(f"unexpected /api/chat body type: {type(body).__name__}")
    return body


async def run_answers(items: List[Item], texts: Dict[str, str], records: Dict[str, str], pool: Any,
                      args: argparse.Namespace) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    page = await pool.fetchrow(PAGE_SQL)
    pids = page_refs(page["content"] if page else None)   # the core page rides in every chat prompt
    prior = {t["qid"]: t for t in (getattr(args, "prior_turns", None) or [])}
    turns: List[Dict[str, Any]] = []
    for n, it in enumerate(items, 1):
        old = prior.get(it.qid)
        if old and not old.get("error") and (old.get("reply") or "").strip():
            # --resume: a turn that already has a reply is kept (and its judgment, if any).
            turns.append(dict(old))
            print(f"  [{n}/{len(items)}] {it.qid}: kept from the resumed run", flush=True)
            continue
        t0 = time.monotonic()
        turn: Dict[str, Any] = {"qid": it.qid, "reply": "", "correlation_id": None, "error": None}
        try:
            # A Hub restart mid-run (seen live 2026-11-17: 49 turns lost to connection refused)
            # waits for /health instead of failing every remaining item.
            await asyncio.to_thread(wait_for_hub, args.hub_url, args.hub_wait)
            body = await asyncio.to_thread(ask_hub, args.hub_url, texts[it.qid], args.turn_timeout)
            turn["reply"] = str(body.get("text") or "").strip()
            turn["correlation_id"] = body.get("correlation_id")
        except Exception as exc:  # noqa: BLE001
            turn["error"] = f"{type(exc).__name__}: {exc}"[:300]
        corr = turn["correlation_id"]
        if corr and not turn["reply"]:
            # The body's text can come back empty when delivery fails though the turn produced
            # text (run_self_sense_eval.py); the turn trace is the reliable source.
            for _ in range(6):
                row = await pool.fetchrow(
                    "SELECT run_artifact->>'final_text' AS t FROM harness_turn_trace WHERE correlation_id = $1", corr)
                if row and (row["t"] or "").strip():
                    turn["reply"] = row["t"].strip()
                    break
                await asyncio.sleep(5)
        if corr:
            from orion.memory.recall_skip_gate import recall_skip_gate

            gate = recall_skip_gate(user_message=texts[it.qid], appraisal=None, has_repair_grammar_signal=False,
                                    max_novelty=SKIP_MAX_NOVELTY, shift_novelty_floor=SKIP_SHIFT_NOVELTY_FLOOR)
            rows = []
            for _ in range(1 if gate.skip else 5):   # recall persists its row after it replies
                rows = await pool.fetch(
                    "SELECT profile, selected_ids, deadline_hit, lived_status, lived_trace FROM recall_telemetry "
                    "WHERE corr_id = $1 AND profile = ANY($2::text[])",
                    corr, [CONTINUITY_PROFILE, *PROFILE_FOR_INTENT.values()])
                if rows:
                    break
                await asyncio.sleep(3)
            # Did this turn's own recall run in full? A denial after a skipped or degraded search
            # is still a false denial, reported under its cause (#3536 comment).
            turn["search_causes"] = (["recall:skip_gate"] if not rows and gate.skip
                                     else search_causes([dict(r) for r in rows]))
            ids: set[str] = set()
            for r in rows:
                sel = r["selected_ids"]
                sel = json.loads(sel) if isinstance(sel, str) else (sel or [])
                ids |= {str(x) for x in sel}
            turn["recall_rows"] = len(rows)
            in_page = any(t in pids for t in it.targets)
            turn["target_in_prompt"] = (in_page or any(t in ids for t in it.targets)) if (rows or in_page) and it.targets else None
            if it.chat_id:
                # An implicit item resends a real past message; flag a turn whose recall served
                # that very message's chat row (its old reply rides with it).
                turn["own_message_served"] = it.chat_id in ids
        if not turn["reply"] and not turn["error"]:
            turn["error"] = "empty reply"
        turn["seconds"] = round(time.monotonic() - t0, 1)
        turns.append(turn)
        if getattr(args, "checkpoint", None):
            # Every Hub reply is on disk before the next turn, so --resume always has a file.
            _write_private(args.checkpoint, json.dumps({"turns": turns}, indent=1, default=str))
        print(f"  [{n}/{len(items)}] {it.qid}: {turn['seconds']}s reply_chars={len(turn['reply'])} "
              f"target_in_prompt={turn.get('target_in_prompt')} error={bool(turn['error'])}", flush=True)

    from orion.evals.leased_llm import LaneCallError, LaneRecalled, LaneUnavailable, operator_lane_sync

    outcomes: Dict[str, str] = {}
    by_qid = {it.qid: it for it in items}
    try:
        with operator_lane_sync(args.judge_lane, holder=HOLDER, bus_url=args.bus_url) as lane:
            for turn in turns:
                it = by_qid[turn["qid"]]
                if turn["error"]:
                    outcomes[it.qid] = "error"
                    continue
                if turn.get("judge_label"):
                    outcomes[it.qid] = outcome_for(it, turn["judge_label"])
                    continue
                prompt = build_judge_prompt(it, texts[it.qid], turn["reply"],
                                            [records[t] for t in it.targets if records.get(t)])
                label = None
                for _attempt in range(2):
                    try:
                        data = lane.chat({
                        "messages": [{"role": "system", "content": JUDGE_SYSTEM},
                                     {"role": "user", "content": prompt}],
                        "max_tokens": 200, "temperature": 0.0,
                        "chat_template_kwargs": {"enable_thinking": False},
                        }, timeout_sec=180)
                    except LaneCallError as exc:
                        turn["judge_error"] = str(exc)[:200]
                        continue
                    raw = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
                    label = parse_judge_label(raw)
                    if label:
                        turn["judge_raw"] = raw[:400]
                        break
                turn["judge_label"] = label
                outcomes[it.qid] = outcome_for(it, label)
    except LaneRecalled as exc:
        print(f"judge lane recalled: {exc} -- remaining items unjudged (rerun with --resume)", flush=True)
    except LaneUnavailable as exc:
        print(f"judge lane unavailable: {exc} -- items unjudged (rerun with --resume)", flush=True)
    for turn in turns:
        outcomes.setdefault(turn["qid"], "unjudged")
        turn["outcome"] = outcomes[turn["qid"]]
    report = score_answers(items, outcomes, {t["qid"]: t.get("search_causes") for t in turns})
    report["judge_lane"] = args.judge_lane
    report["target_in_prompt"] = {
        "known": sum(1 for t in turns if t.get("target_in_prompt") is not None),
        "yes": sum(1 for t in turns if t.get("target_in_prompt")),
    }
    return report, turns


# ---------------------------------------------------------------- main

async def _main(args: argparse.Namespace) -> int:
    import asyncpg  # type: ignore

    bank_path = Path(args.bank).expanduser()
    if bank_path.stat().st_mode & 0o077:
        print(f"WARNING: {bank_path} is readable by others; it holds private history (chmod 600)", flush=True)
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    items = load_items(bank)
    if args.only:
        keep = set(args.only.split(","))
        items = [it for it in items if it.qid in keep]
    if args.limit:
        items = items[: args.limit]
    started = datetime.now(timezone.utc)
    run_date = args.date or started.date().isoformat()
    # The dated file is the weekly record: only a whole-bank run of both phases (or a resume of
    # one) writes it and the history line. Anything else gets its own timestamped name.
    full_run = not args.only and not args.limit and (set(args.phases.split(",")) >= {"retrieval", "answer"}
                                                       or bool(args.resume))
    run_name = run_date if full_run else f"{run_date}T{started.strftime('%H%M%S')}.partial"
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(out_dir, 0o700)
    target = out_dir / f"{run_name}.json"
    if target.exists() and not args.force and (not args.resume or Path(args.resume).expanduser().resolve() != target.resolve()):
        print(f"{target} exists; pass --resume {target} to continue it or --force to replace it")
        return 2
    args.checkpoint = out_dir / f"{run_name}.checkpoint.json"
    pool = await asyncpg.create_pool(_dsn(args.dsn), min_size=1, max_size=2)
    try:
        texts: Dict[str, str] = {}
        for it in items:
            if it.chat_id:
                row = await pool.fetchrow("SELECT prompt FROM chat_history_log WHERE id::text = $1", it.chat_id)
                if row is None or not (row["prompt"] or "").strip():
                    print(f"  {it.qid}: chat message {it.chat_id} is gone; item skipped", flush=True)
                    continue
                texts[it.qid] = row["prompt"]
            else:
                texts[it.qid] = str(it.question)
        items = [it for it in items if it.qid in texts]
        records: Dict[str, str] = {}
        missing: List[str] = []
        for rid in iter_record_ids(items):
            kind, _, key = rid.partition(":")
            row = await pool.fetchrow(RECORD_SQL[kind], key)
            if row is None:
                missing.append(rid)
            else:
                records[rid] = str(row[0] or "")
        if missing:
            print(f"WARNING: {len(missing)} bank record id(s) no longer exist: {missing[:5]}", flush=True)
        # An item whose every target is gone can never hit: a bank problem, not a memory one.
        gone = [it.qid for it in items if it.targets and all(t in missing for t in it.targets)]
        if gone:
            print(f"WARNING: {len(gone)} item(s) left out, all targets deleted: {gone}", flush=True)
            items = [it for it in items if it.qid not in gone]
        phases = set(args.phases.split(","))
        retrieval = answers = None
        args.prior_turns = []
        if args.resume:
            prev = json.loads(Path(args.resume).expanduser().read_text(encoding="utf-8"))
            args.prior_turns = prev.get("turns") or []
            if args.rejudge:
                # Keep the replies, drop the judgments: the judge rubric changed, the turns did not.
                args.prior_turns = [{k: v for k, v in t.items() if k not in ("judge_label", "judge_raw", "outcome")}
                                    for t in args.prior_turns]
            if "retrieval" not in phases:
                retrieval = prev.get("retrieval")   # keep the resumed run's retrieval score
        turns: List[Dict[str, Any]] = []
        if "retrieval" in phases:
            print(f"retrieval: {len(items)} items (phase-3 intent {args.phase3_intent})", flush=True)
            retrieval = await run_retrieval(items, texts, pool, args)
        if "answer" in phases:
            print(f"answer: {len(items)} Hub chat turns on session {SESSION_ID} (no_write), judge lane {args.judge_lane}",
                  flush=True)
            answers, turns = await run_answers(items, texts, records, pool, args)
    finally:
        await pool.close()
    table = render_table(retrieval, answers, run_date=run_date, bank_items=len(items))
    full = {"date": run_date, "started_at": started.isoformat(), "bank": str(args.bank), "items": len(items),
            "missing_records": missing, "retrieval": retrieval, "answer": answers, "turns": turns}
    full["left_out_all_targets_deleted"] = gone
    full["own_message_served"] = [t["qid"] for t in turns if t.get("own_message_served")]
    _write_private(target, json.dumps(full, indent=1, default=str))
    _write_private(out_dir / f"{run_name}.md", table + "\n")
    if args.checkpoint.exists():
        args.checkpoint.unlink()
    if full_run:
        hist = out_dir / "history.jsonl"
        # One line per date: a resumed or repeated full run replaces that date's line.
        kept = [ln for ln in (hist.read_text(encoding="utf-8").splitlines() if hist.exists() else [])
                if ln.strip() and json.loads(ln).get("date") != run_date]
        _write_private(hist, "".join(ln + "\n" for ln in kept)
                       + json.dumps(summary_line(run_date, retrieval, answers)) + "\n")
    print("\n" + table)
    print(f"\nwritten: {target} and .md (mode 600){'' if full_run else ' -- partial run, no history line'}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bank", default=DEFAULT_BANK)
    ap.add_argument("--dsn", default=None, help="Postgres DSN (default: ORION_SCOREBOARD_DSN, else POSTGRES_* on 127.0.0.1)")
    ap.add_argument("--bus-url", default=os.getenv("ORION_BUS_URL"))
    ap.add_argument("--hub-url", default=os.getenv("HUB_BASE_URL", DEFAULT_HUB))
    ap.add_argument("--phases", default="retrieval,answer")
    ap.add_argument("--phase3-intent", default=DEFAULT_PHASE3_INTENT, choices=["auto", "none", *PROFILE_FOR_INTENT])
    # A real chat turn waits min(STEP_TIMEOUT_MS, RECALL_RPC_TIMEOUT_SEC) = 90 s (cortex-exec defaults).
    ap.add_argument("--deadline-ms", type=int, default=90000, help="recall deadline per phase")
    ap.add_argument("--rpc-timeout", type=float, default=120.0)
    ap.add_argument("--force", action="store_true", help="replace an existing run file of the same name")
    ap.add_argument("--turn-timeout", type=float, default=900.0, help="seconds per Hub chat turn")
    ap.add_argument("--judge-lane", default="agent", help="orion.evals.leased_llm lane (local only)")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--date", default=None, help="run date label (default: today, UTC)")
    ap.add_argument("--only", default=None, help="comma-separated qids")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", default=None,
                    help="earlier run JSON: keep its replies and judgments, redo failed turns and unjudged items")
    ap.add_argument("--rejudge", action="store_true", help="with --resume: judge every kept reply again")
    ap.add_argument("--lived-local", action="store_true",
                    help="retrieval only: replace the deployed lived-memory lines with the code on disk (#3613)")
    ap.add_argument("--lived-local-short-gate", choices=["on", "off"], default="on",
                    help="with --lived-local: off = the plain content-word floor (the before of #3613)")
    ap.add_argument("--lived-chroma-url", default="http://127.0.0.1:8500")
    ap.add_argument("--lived-embed-url", default="http://127.0.0.1:8320/embedding")
    ap.add_argument("--lived-rerank-url", default="http://127.0.0.1:8320/rerank")
    ap.add_argument("--hub-wait", type=float, default=900.0, help="seconds to wait for Hub /health before a turn")
    args = ap.parse_args(argv)
    if not args.bus_url:
        print("ORION_BUS_URL is not set (redis://<bus-host>:6379/0): recall and the judge's GPU seat go over the real bus.")
        return 2
    if not Path(args.bank).expanduser().exists():
        print(f"bank not found: {args.bank}")
        return 2
    return asyncio.run(_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
