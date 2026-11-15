"""Curiosity finding register: label and novelty for every finished curiosity turn (event wants, step 2).

Design: docs/superpowers/specs/2026-10-16-attention-world-candidates-design.md ("Which findings compete").
Pure parts (prompt, parser, novelty, corpus selection) live in ``orion/curiosity/register.py``.

After a curiosity turn ends, Hub writes ONE row to ``curiosity_finding_register``:

1. **Judge.** One LLM call through orion-llm-gateway on the curiosity lane's own route (Juniper,
   2026-11-15: the lane that wrote the finding judges it), with the pinned prompt v2, temperature 0,
   thinking off. Kept only when the served model passed the judge-agreement eval
   (``register.judge_model_passed``); otherwise ``judge_error = model_not_evaluated:<model>``.
2. **Novelty.** The finding is embedded (vector-host ``/embedding``) and compared with every piece of
   Orion's writing before the run, cached in ``curiosity_novelty_corpus`` (only new text is embedded).
   The corpus must be complete: a partial corpus would read an old idea as new, so any embed failure
   leaves novelty NULL with ``novelty_error``.

Insert-only. A failure here never touches the turn: every entry point swallows and logs.
Log line: ``curiosity_register run=<id> register=<r> novelty=<v> judge=<model>``.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Mapping, Optional, Sequence
from uuid import uuid4

from orion.core.bus.bus_schemas import BaseEnvelope, ChatRequestPayload, LLMMessage
from orion.curiosity.register import (
    CORPUS_VERSION,
    EMBED_CHARS,
    JUDGE_PROMPT_SHA256,
    corpus_before,
    judge_messages,
    judge_model_passed,
    novelty,
    parse_judge_reply,
)

logger = logging.getLogger("orion.hub.curiosity_finding_register")

JUDGE_TIMEOUT_SEC = 180.0
JUDGE_MAX_TOKENS = 400  # 160 cut 21 of 263 eval replies mid-JSON (2026-11-15)
EMBED_TIMEOUT_SEC = 30.0
EMBED_CONCURRENCY = 4
CORPUS_SYNC_BUDGET_SEC = 600.0  # first sync embeds ~1,600 items; later runs embed a handful
EMBED_MODEL = "bge-large-en-v1.5"

INSERT_REGISTER_SQL = """
INSERT INTO curiosity_finding_register (
    run_id, line, question_ref, register, register_reason, judge_model, judge_route, prompt_sha256,
    judge_error, judged_at, novelty, nearest_ref, novelty_error, embed_model, corpus_version, corpus_size,
    answer_chars)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
ON CONFLICT (run_id) DO NOTHING
"""
LETTERS_SQL = "SELECT run_id, created_at, note_md FROM orion_day_letter"
PER_PRIOR_SQL = "SELECT per_prior FROM curiosity_run_outcomes WHERE run_id = $1"
QUESTIONS_SQL = "SELECT question_id, text FROM curiosity_self_questions"
REGISTER_EXISTS_SQL = "SELECT 1 FROM curiosity_finding_register WHERE run_id = $1"
CORPUS_READ_SQL = """
SELECT ref, text_sha256, embedding FROM curiosity_novelty_corpus
 WHERE (ref, text_sha256) IN (SELECT * FROM unnest($1::text[], $2::text[]))
"""
CORPUS_INSERT_SQL = """
INSERT INTO curiosity_novelty_corpus (ref, text_sha256, kind, run_id, written_at, embed_model, embedding)
VALUES ($1, $2, $3, $4, $5, $6, $7)
ON CONFLICT (ref, text_sha256) DO NOTHING
"""


def per_prior_list(raw: Any) -> list[dict]:
    """The outcome row's ``per_prior`` (jsonb arrives as text from asyncpg), [] when absent or malformed."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return []
    return [p for p in raw if isinstance(p, dict)] if isinstance(raw, list) else []


@dataclass(frozen=True)
class JudgeResult:
    register: Optional[str]
    reason: Optional[str]
    error: Optional[str]
    model: Optional[str]


async def judge_finding(
    bus: Any, *, intake_channel: str, gateway_service: str, source_ref: Any, route: str,
    question: str, answer: str, timeout_sec: float = JUDGE_TIMEOUT_SEC,
) -> JudgeResult:
    """One judge call on ``route``. Never raises: every failure is a ``JudgeResult`` with ``error``."""
    corr = str(uuid4())
    reply_channel = f"orion:exec:result:{gateway_service}:{corr}"
    payload = ChatRequestPayload(
        messages=[LLMMessage(role=m["role"], content=m["content"]) for m in judge_messages(question, answer)],
        route=route,
        options={
            "temperature": 0,
            "max_tokens": JUDGE_MAX_TOKENS,
            "llm_route": route,
            "purpose": "classify",
            "skip_spark_candidate_publish": True,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    env = BaseEnvelope(kind="llm.chat.request", source=source_ref, correlation_id=corr, reply_to=reply_channel,
                       payload=payload.model_dump(mode="json"))
    try:
        msg = await bus.rpc_request(intake_channel, env, reply_channel=reply_channel, timeout_sec=timeout_sec)
        decoded = bus.codec.decode(msg.get("data") if isinstance(msg, dict) else msg)
        if not decoded.ok:
            return JudgeResult(None, None, f"call: {decoded.error}"[:200], None)
        body = decoded.envelope.payload or {}
    except Exception as exc:  # noqa: BLE001
        return JudgeResult(None, None, f"call: {type(exc).__name__}: {exc}"[:200], None)
    model = str(body.get("model_used") or "") or None
    if not judge_model_passed(model):
        return JudgeResult(None, None, f"model_not_evaluated: {model or 'unknown'}"[:200], model)
    reg, reason, err = parse_judge_reply(str(body.get("content") or body.get("text") or ""))
    return JudgeResult(reg, reason, err, model)


EmbedFn = Callable[[str], Awaitable[list[float]]]


def http_embedder(client: Any, url: str) -> EmbedFn:
    async def embed(text: str) -> list[float]:
        r = await client.post(url, json={"doc_id": "curiosity-register", "text": text[:EMBED_CHARS]},
                              timeout=EMBED_TIMEOUT_SEC)
        r.raise_for_status()
        vec = r.json().get("embedding")
        if not isinstance(vec, list) or not vec:
            raise ValueError("no embedding in reply")
        return [float(x) for x in vec]
    return embed


async def sync_corpus(pool: Any, items: Sequence[Mapping[str, Any]], embed: EmbedFn, *,
                      budget_sec: float = CORPUS_SYNC_BUDGET_SEC) -> list[tuple[str, list[float]]]:
    """(ref, vector) for every item, embedding and storing the ones not cached yet. Raises when any
    item cannot be embedded within the budget: a partial corpus is not a corpus."""
    refs = [str(i["ref"]) for i in items]
    shas = [str(i["sha"]) for i in items]
    async with pool.acquire() as conn:
        rows = await conn.fetch(CORPUS_READ_SQL, refs, shas)
    have = {(r["ref"], r["text_sha256"]): list(r["embedding"]) for r in rows}
    todo = [i for i in items if (i["ref"], i["sha"]) not in have]
    if todo:
        sem = asyncio.Semaphore(EMBED_CONCURRENCY)

        async def one(it: Mapping[str, Any]) -> None:
            async with sem:
                vec = await embed(str(it["text"]))
            async with pool.acquire() as conn:
                await conn.execute(CORPUS_INSERT_SQL, it["ref"], it["sha"], it["kind"], it.get("run_id"),
                                   it.get("written_at"), EMBED_MODEL, vec)
            have[(it["ref"], it["sha"])] = vec

        await asyncio.wait_for(asyncio.gather(*(one(i) for i in todo)), timeout=budget_sec)
        logger.info("curiosity_register_corpus_embedded new=%s total=%s", len(todo), len(items))
    return [(str(i["ref"]), have[(i["ref"], i["sha"])]) for i in items]


async def finding_novelty(pool: Any, items: Sequence[Mapping[str, Any]], answer: str, embed: EmbedFn, *,
                          run_id: str, started_at: Optional[datetime]) -> tuple[Optional[float], Optional[str], int, Optional[str]]:
    """(novelty, nearest_ref, corpus_size, error). Novelty is None with an error on any failure."""
    before = corpus_before(items, run_id=run_id, started_at=started_at)
    try:
        corpus = await sync_corpus(pool, before, embed)
        vec = await embed(answer)
    except Exception as exc:  # noqa: BLE001
        return None, None, len(before), f"{type(exc).__name__}: {exc}"[:200]
    value, nearest = novelty(vec, corpus)
    return value, nearest, len(corpus), (None if value is not None else "empty_corpus")


async def already_recorded(pool: Any, run_id: str) -> bool:
    async with pool.acquire() as conn:
        return bool(await conn.fetchval(REGISTER_EXISTS_SQL, run_id))


async def record_register(pool: Any, row: Mapping[str, Any]) -> bool:
    async with pool.acquire() as conn:
        status = await conn.execute(
            INSERT_REGISTER_SQL, row["run_id"], row["line"], row.get("question_ref"), row.get("register"),
            row.get("register_reason"), row.get("judge_model"), row.get("judge_route"), JUDGE_PROMPT_SHA256,
            row.get("judge_error"), row.get("judged_at"), row.get("novelty"), row.get("nearest_ref"),
            row.get("novelty_error"), EMBED_MODEL if row.get("novelty") is not None else None, CORPUS_VERSION,
            row.get("corpus_size"), row.get("answer_chars"))
    return str(status).endswith(" 1")


async def score_and_record(
    *, pool: Any, run_id: str, line: str, question: str, question_ref: Optional[str], answer: str,
    items: Sequence[Mapping[str, Any]], started_at: Optional[datetime], judge: Callable[[], Awaitable[JudgeResult]],
    embed: EmbedFn, route: str,
) -> Optional[dict[str, Any]]:
    """Judge + novelty + insert. Returns the row written, or None when the run already had one."""
    if await already_recorded(pool, run_id):
        return None
    t0 = time.monotonic()
    if answer:
        verdict, (nov, nearest, size, nov_err) = await asyncio.gather(
            judge(), finding_novelty(pool, items, answer, embed, run_id=run_id, started_at=started_at))
    else:
        verdict, (nov, nearest, size, nov_err) = JudgeResult(None, None, "no_answer", None), (None, None, 0, "no_answer")
    row = {
        "run_id": run_id, "line": line, "question_ref": question_ref, "register": verdict.register,
        "register_reason": verdict.reason, "judge_model": verdict.model, "judge_route": route,
        "judge_error": verdict.error, "judged_at": datetime.now(timezone.utc),
        "novelty": nov, "nearest_ref": nearest, "novelty_error": nov_err, "corpus_size": size,
        "answer_chars": len(answer),
    }
    written = await record_register(pool, row)
    logger.info(
        "curiosity_register run=%s line=%s register=%s novelty=%s judge=%s error=%s novelty_error=%s "
        "corpus=%s written=%s took_s=%.1f",
        run_id, line, verdict.register, nov, verdict.model, verdict.error, nov_err, size, written,
        time.monotonic() - t0)
    return row if written else None


__all__ = ["LETTERS_SQL", "PER_PRIOR_SQL", "QUESTIONS_SQL", "JudgeResult", "per_prior_list", "already_recorded", "finding_novelty", "http_embedder", "judge_finding", "record_register",
           "score_and_record", "sync_corpus"]
