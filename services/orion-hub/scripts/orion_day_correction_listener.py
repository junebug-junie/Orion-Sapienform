"""Hub's introspect responder for the `record_correction` tool.

In a conversation about an Orion's Day letter, Orion and Juniper establish that a claim was
wrong; Orion records the correction. This responder checks it and writes one
``orion_day_correction`` row -- the only write anywhere in the letter-reread design
(docs/superpowers/specs/2026-10-11-orion-day-letter-reread-design.md, section 4).

Before anything is written:
* the request is a chat turn (a correction is agreed with Juniper, not made up in a run);
* ``ref`` names a numbered note paragraph or carry item of a stored letter, found with the
  same splitters the email and the `orion_day` tool number with (letter_parts.py);
* every ``prior_id`` exists in Orion's own graph, read with ``GRAPH.RO_QUERY``
  (orion.orion_day.corrections.read_named_priors over the curiosity WorldviewReader).
  Hub never writes orion_worldview (the PR #2199 boundary); the prior changes, if at all, in a
  later curiosity run that Orion runs.

Any check that fails is ok=false and nothing is written. The insert is the last step, so an
ok=false reply always means "not recorded". Same reply rules as the other introspect
responders: only to exactly orion:introspect:result:<corr>, and only for the request kind.
"""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import suppress
from datetime import date, datetime, timezone
from typing import Any, Callable

from orion.core.bus.bus_schemas import BaseEnvelope
from orion.introspect.redact import safe_exception_detail
from orion.introspect.transport import RECORD_CORRECTION_REQUEST_CHANNEL, REQUEST_KIND, RESULT_KIND, RESULT_PREFIX
from orion.orion_day.budget import material_ref_records
from orion.orion_day.corrections import (
    INSERT_CORRECTION_SQL,
    MIGRATION,
    correction_id_for,
    evidence_records,
    TRUNCATED,
    read_named_priors_or_reason,
)
from orion.orion_day.letter_parts import find_part, parse_ref, split_carry, split_note
from orion.orion_day.store import fetch_letter
from orion.schemas.introspect import IntrospectItemV1, IntrospectRequestV1, IntrospectResultV1, RecordCorrectionArguments

logger = logging.getLogger("orion-hub.orion_day_correction")

OPERATION = "record_correction"
# Not "not recorded": a connection lost after the INSERT committed lands here too.
UNAVAILABLE = "record_correction_unavailable; answer unknown"
WORLDVIEW_UNREADABLE = (
    "your graph (orion_worldview) could not be read, so the prior ids could not be checked"
)
WORLDVIEW_TRUNCATED = (
    "your graph holds more priors than one bounded read returns, so the prior ids could not be "
    "checked; name no prior ids to record it as context only"
)
_INVALID_CAP = 400
_UNDEFINED_TABLE = "42P01"


class RefusedError(ValueError):
    """A check failed; the message says which, in plain words. Nothing was written."""


def _failed(now: datetime, error: str) -> IntrospectResultV1:
    return IntrospectResultV1(ok=False, operation=OPERATION, as_of=now, error=error)


def _squash(text: str) -> str:
    return " ".join((text or "").split()).casefold()


class OrionDayCorrectionListener:
    def __init__(self, *, pool_provider: Callable[[], Any], reader_provider: Callable[[], Any], source_ref: Any):
        self.pool_provider = pool_provider
        self.reader_provider = reader_provider
        self.source_ref = source_ref
        self.bus: Any = None
        self.task: asyncio.Task | None = None

    async def handle(self, envelope: BaseEnvelope) -> None:
        reply = f"{RESULT_PREFIX}{envelope.correlation_id}"
        if envelope.reply_to != reply or envelope.kind != REQUEST_KIND:
            return
        now = datetime.now(timezone.utc)
        try:
            request = IntrospectRequestV1.model_validate(envelope.payload)
            if request.operation != OPERATION:
                raise ValueError(f"operation {request.operation!r} is not answered here")
            args = RecordCorrectionArguments.model_validate(request.args)
        except ValueError as exc:
            await self._publish(reply, envelope, _failed(now, f"invalid record_correction request: {str(exc)[:_INVALID_CAP]}"))
            return
        try:
            result = await self.record(request, args, now)
        except RefusedError as exc:
            logger.info("orion_day_correction_refused corr=%s ref=%s detail=%s",
                        envelope.correlation_id, args.ref, str(exc)[:_INVALID_CAP])
            result = _failed(now, str(exc)[:_INVALID_CAP])
        except Exception as exc:
            if getattr(exc, "sqlstate", None) == _UNDEFINED_TABLE:
                logger.warning("orion_day_correction_table_missing corr=%s -- apply %s", envelope.correlation_id, MIGRATION)
            logger.warning(
                "introspect_failed op=record_correction corr=%s exc_type=%s detail=%s",
                envelope.correlation_id, type(exc).__name__, safe_exception_detail(exc),
            )
            result = _failed(now, UNAVAILABLE)
        else:
            item = result.items[0]
            logger.info(
                "orion_day_correction_recorded corr=%s correction_id=%s ref=%s priors=%d already_recorded=%s",
                envelope.correlation_id, item.id, args.ref, len(args.prior_ids), item.extra.get("already_recorded"),
            )
        await self._publish(reply, envelope, result)

    async def record(self, request: IntrospectRequestV1, args: RecordCorrectionArguments, now: datetime) -> IntrospectResultV1:
        binding = request.binding
        if binding.invocation_context != "unified_chat":
            raise RefusedError(
                "a correction is recorded from a conversation with Juniper; this is a "
                f"{binding.invocation_context} turn"
            )
        parsed = parse_ref(args.ref)
        if parsed is None:  # the contract's pattern already holds; kept for a hand-built request
            raise RefusedError(f"ref {args.ref!r} is not a letter part id like '2026-10-09 ¶3' or '2026-10-09 carry 5'")
        day, kind, index = parsed
        try:
            letter_day = date.fromisoformat(day)
        except ValueError as exc:
            raise RefusedError(f"ref {args.ref!r} names no real date") from exc
        pool = self.pool_provider()
        if pool is None:
            raise RuntimeError("no memory pool")

        async with pool.acquire() as conn:
            letter = await fetch_letter(conn, letter_day)
        if letter is None:
            raise RefusedError(f"unknown letter {day}: no Orion's Day letter for that date (not found)")
        parts = split_note(letter.note_md) if kind == "paragraph" else split_carry(letter.carry_forward_md)
        part = find_part(parts, kind, index)
        if part is None:
            numbered = sum(1 for p in parts if p.kind == kind)
            label = "note paragraphs" if kind == "paragraph" else "carry items"
            raise RefusedError(f"unknown part {args.ref}: that letter has {numbered} numbered {label} (not found)")

        if args.prior_ids:
            current, reason = await asyncio.to_thread(
                read_named_priors_or_reason, self.reader_provider(), args.prior_ids,
            )
            if current is None:
                raise RefusedError(WORLDVIEW_TRUNCATED if reason == TRUNCATED else WORLDVIEW_UNREADABLE)
            missing = [p for p in args.prior_ids if current.get(p) is None]
            if missing:
                raise RefusedError(
                    f"unknown prior_id {', '.join(missing[:5])}: no :Prior with that id in your graph (not found)"
                )

        chat_corr = binding.parent_trace_id or None
        correction_id = correction_id_for(args.ref, args.claim, args.correction, chat_corr)
        evidence = evidence_records(args.evidence, material_ref_records(letter.material))
        async with pool.acquire() as conn:
            inserted = await conn.fetchval(
                INSERT_CORRECTION_SQL, correction_id, letter.letter_date, args.ref, args.claim,
                args.correction, json.dumps(evidence), list(args.prior_ids), chat_corr,
            )
        item = IntrospectItemV1(
            id=str(correction_id),
            occurred_at=now,
            kind="orion_day_correction",
            epistemic_status="record",
            text=args.correction,
            extra={
                "correction_id": str(correction_id),
                "ref": args.ref,
                "letter_date": letter.letter_date.isoformat(),
                "claim": args.claim,
                # Whether the claim's words appear in that part (whitespace and case aside).
                # False: a paraphrase, so the stored claim is not the letter's own wording.
                "claim_quoted": _squash(args.claim) in _squash(part.text),
                "evidence": evidence,
                "prior_ids": list(args.prior_ids),
                "chat_correlation_id": chat_corr,
                "already_recorded": inserted is None,
                "what_happens_next": (
                    "nothing changes now; a later curiosity run is shown this once and asked to "
                    "re-test the named priors, and whether it did is recorded on this correction"
                    if args.prior_ids
                    else "nothing changes now; a later curiosity run and the next letter see it as context"
                ),
            },
        )
        return IntrospectResultV1(ok=True, operation=OPERATION, as_of=now, total_available=1, items=[item])

    async def _publish(self, reply: str, envelope: BaseEnvelope, result: IntrospectResultV1) -> None:
        await self.bus.publish(reply, BaseEnvelope(
            kind=RESULT_KIND, correlation_id=envelope.correlation_id,
            source=self.source_ref, payload=result.model_dump(mode="json"),
        ))

    async def _run(self) -> None:
        while True:
            try:
                async with self.bus.subscribe(RECORD_CORRECTION_REQUEST_CHANNEL) as pubsub:
                    logger.info("orion_day_correction_listening channel=%s", RECORD_CORRECTION_REQUEST_CHANNEL)
                    async for raw in self.bus.iter_messages(pubsub):
                        try:
                            decoded = self.bus.codec.decode(raw.get("data"))
                            if decoded.ok:
                                await self.handle(decoded.envelope)
                        except Exception:
                            logger.warning("orion_day_correction_message_failed", exc_info=True)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("orion_day_correction_reconnecting", exc_info=True)
                await asyncio.sleep(1)

    async def start(self, bus: Any) -> None:
        self.bus = bus
        self.task = asyncio.create_task(self._run(), name="hub-orion-day-correction")

    async def stop(self) -> None:
        if self.task:
            self.task.cancel()
            with suppress(asyncio.CancelledError):
                await self.task
            self.task = None

