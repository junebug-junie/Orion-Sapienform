"""Core memory pages: the nightly fold of Orion's pages (``page:juniper``, ``page:origin``).

Spec: docs/superpowers/specs/2026-11-13-core-memory-pages-and-concept-growth-design.md. The page
logic (sources, prompt, cite check) is each page's ``FOLD_SPEC`` (``orion/memory/pages/juniper.py``,
``orion/memory/pages/origin.py``, shared parts in ``orion/memory/pages/fold.py``); this module only
decides when to fold, makes the one LLM call, and publishes the passing page. One
``CorePageFolder`` per page, each with its own once-a-day and attempt counts; ``run_folders_forever``
ticks them one after another, so two folds never hold the LLM seat at once.

Every CORE_PAGE_FOLD_POLL_SEC, inside the Denver window [CORE_PAGE_FOLD_HOUR_LOCAL,
CORE_PAGE_FOLD_END_HOUR_LOCAL):

1. A version already written today (Denver date), or published by this process today even if
   orion-sql-writer has not landed it yet -> ``done_today``.
2. CORE_PAGE_FOLD_MAX_ATTEMPTS_PER_DAY LLM calls spent today -> ``attempts_spent``.
3. A current version exists and no record it reads changed since it was written -> ``no_new_sources``.
4. Otherwise: load the records, one LLM call on CORE_PAGE_FOLD_ROUTE (the gateway leases the seat),
   ``check_page``. Pass -> the next version is published to orion-sql-writer (``folded``). Fail ->
   ``rejected`` with the check's errors; the next tick in the window tries again.

The manual trigger (``POST /core-pages/<slug>/fold``) runs step 4 now, at any hour; ``dry_run``
returns the page and its check without publishing. Attempt counts live in memory: a restart can
spend at most one more day's worth.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Awaitable, Callable
from zoneinfo import ZoneInfo

from orion.memory.pages import juniper as _juniper
from orion.memory.pages.registry import FOLD_SPECS, selected_specs  # noqa: F401 -- re-exported for main.py
from orion.memory.pages.fold import (
    CURRENT_PAGE_SQL, MAX_VERSION_SQL, PRODUCED_BY, FoldSpec, current_page_from_row, handle_map,
)

logger = logging.getLogger("orion-durable-runs.core_page_fold")

ZONE = ZoneInfo("America/Denver")
SELF_CONCEPT_CHANNEL = "orion:self_concept:history:write"
SELF_CONCEPT_KIND = "self_concept.history.write.v1"

CallLlm = Callable[[str], Awaitable[dict[str, Any]]]
Publish = Callable[[str, str, Any, uuid.UUID], Awaitable[bool]]


def _aware(value: Any) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def in_window(now: datetime, start_hour: int, end_hour: int) -> bool:
    local = now.astimezone(ZONE)
    return start_hour <= local.hour < end_hour


@dataclass
class FoldResult:
    at: str
    decision: str     # folded | would_write | rejected | llm_failed | done_today | attempts_spent | no_new_sources | ...
    page: str = ""
    dry_run: bool = False
    version: int | None = None
    current_version: int | None = None
    sources: int = 0
    check: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)
    page_text: str | None = None
    stored_text: str | None = None
    latency_ms: int | None = None
    model: str | None = None
    error: str | None = None

    def as_dict(self, *, with_text: bool = True) -> dict[str, Any]:
        out = dict(self.__dict__)
        if not with_text:
            out.pop("page_text", None)
            out.pop("stored_text", None)
        return out


class CorePageFolder:
    def __init__(self, *, settings: Any, pool: Any, call_llm: CallLlm, publish: Publish,
                 now: Callable[[], datetime] | None = None, spec: FoldSpec = _juniper.FOLD_SPEC,
                 lock: asyncio.Lock | None = None):
        self.spec = spec
        self.settings = settings
        self.pool = pool
        self.call_llm = call_llm
        self.publish = publish
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.last: FoldResult | None = None
        self._attempts: dict[date, int] = {}
        # The Denver date this process last published a page. orion-sql-writer writes the row
        # asynchronously; a slow or rejecting writer must not make the next tick fold again.
        self._folded_on: date | None = None
        # Shared by every page's folder (main.py passes one lock), so the nightly loop and a manual
        # trigger for another page never hold the fold seat at the same time.
        self._lock = lock or asyncio.Lock()

    async def _fetch(self) -> tuple[dict[str, list[dict]], dict | None, datetime | None, int]:
        params = {"cid": self.spec.concept_id, "by": PRODUCED_BY}
        rows: dict[str, list[dict]] = {}
        async with self.pool.connection() as conn:
            for name, sql in self.spec.source_sql.items():
                rows[name] = [dict(r) for r in await (await conn.execute(sql)).fetchall()]
            current = await (await conn.execute(CURRENT_PAGE_SQL, params)).fetchone()
            newest = await (await conn.execute(self.spec.newest_sql)).fetchone()
            top = await (await conn.execute(MAX_VERSION_SQL, params)).fetchone()
        return (rows, dict(current) if current else None,
                _aware((newest or {}).get("newest")), int((top or {}).get("v") or 0))

    async def tick_once(self, *, dry_run: bool = False, force: bool = False) -> FoldResult:
        """One decision. ``force``: skip the window / once-a-day / no-change gates (manual trigger).
        ``dry_run``: fold and check, publish nothing, count no attempt."""
        async with self._lock:
            result = await self._tick(dry_run=dry_run, force=force)
            if not dry_run:
                self.last = result
            return result

    async def _tick(self, *, dry_run: bool, force: bool) -> FoldResult:
        now = self.now()
        today = now.astimezone(ZONE).date()
        result = FoldResult(at=now.isoformat(), decision="", page=self.spec.concept_id, dry_run=dry_run)
        s = self.settings
        if not force and not in_window(now, int(s.core_page_fold_hour_local), int(s.core_page_fold_end_hour_local)):
            result.decision = "outside_window"
            return result
        try:
            rows, current_row, newest, top_version = await self._fetch()
        except Exception as exc:  # noqa: BLE001 -- DB away: say so, try next tick
            result.decision, result.error = "db_unreadable", f"{type(exc).__name__}: {exc}"[:300]
            logger.warning("core_page_fold_db_unreadable page=%s %s", self.spec.concept_id, result.error)
            return result
        current = current_page_from_row(current_row)
        written_at = current.created_at if current is not None else None
        result.current_version = current.version if current is not None else None
        if not force:
            if self._folded_on == today or (written_at is not None and written_at.astimezone(ZONE).date() == today):
                result.decision = "done_today"
                return result
            if self._attempts.get(today, 0) >= int(s.core_page_fold_max_attempts_per_day):
                result.decision = "attempts_spent"
                return result
            if written_at is not None and newest is not None and newest <= written_at:
                result.decision = "no_new_sources"
                return result
        sources = self.spec.sources(rows)
        result.sources = len(sources)
        if not sources:
            result.decision = "no_sources"
            return result
        handles = handle_map(sources)
        prompt = self.spec.prompt(sources, current)
        if not dry_run:
            self._attempts = {today: self._attempts.get(today, 0) + 1}
        started = time.monotonic()
        try:
            reply = await self.call_llm(prompt)
        except Exception as exc:  # noqa: BLE001 -- gateway/pool refusal or timeout: next tick retries
            result.decision, result.error = "llm_failed", f"{type(exc).__name__}: {exc}"[:300]
            logger.warning("core_page_fold_llm_failed page=%s %s", self.spec.concept_id, result.error)
            return result
        result.latency_ms = int((time.monotonic() - started) * 1000)
        result.model = reply.get("model")
        check = self.spec.check(str(reply.get("text") or ""), handles)
        result.check = check.as_dict()
        result.errors = list(check.errors)
        result.page_text = check.clean_text
        result.stored_text = check.stored_text
        if not check.ok:
            result.decision = "rejected"
            logger.warning("core_page_fold_rejected %s", json.dumps(result.as_dict(with_text=False)))
            return result
        version = top_version + 1
        result.version = version
        if dry_run:
            result.decision = "would_write"
            return result
        row = self.spec.history_row(check, version=version, now=now)
        corr = uuid.uuid5(uuid.NAMESPACE_URL, f"orion:core_page:{self.spec.concept_id}:v{version}")
        if not await self.publish(SELF_CONCEPT_CHANNEL, SELF_CONCEPT_KIND, row, corr):
            result.decision, result.error = "publish_failed", "bus publish failed"
            return result
        self._folded_on = today
        result.decision = "folded"
        logger.info("core_page_folded %s", json.dumps(result.as_dict(with_text=False)))
        return result

    async def run_forever(self, stop: asyncio.Event) -> None:
        await run_folders_forever([self], stop)

    def health(self) -> dict[str, Any] | None:
        return self.last.as_dict(with_text=False) if self.last else None


async def run_folders_forever(folders: list[CorePageFolder], stop: asyncio.Event) -> None:
    """Tick every page's folder in turn, then sleep CORE_PAGE_FOLD_POLL_SEC. One at a time, so two
    pages never hold the fold seat at once."""
    if not folders:
        return
    interval = max(60.0, float(folders[0].settings.core_page_fold_poll_sec))
    while not stop.is_set():
        for folder in folders:
            if stop.is_set():
                break
            try:
                await folder.tick_once()
            except Exception:  # noqa: BLE001 -- log, try next tick (the window allows several)
                logger.exception("core_page_fold_tick_failed page=%s", folder.spec.concept_id)
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass


def bus_llm_caller(bus: Any, settings: Any, source: Any) -> CallLlm:
    """One tool-free gateway call on ``CORE_PAGE_FOLD_ROUTE``. No ``gpu_lease`` option: the gateway
    takes its own pool lease for the call (config/gpu_pool.yaml routes). Thinking off: the page is
    short prose, and Qwen-family thinking eats ``max_tokens`` on prose."""
    from orion.core.bus.bus_schemas import BaseEnvelope, ChatRequestPayload, LLMMessage

    async def call(prompt: str) -> dict[str, Any]:
        if bus is None:
            raise RuntimeError("no_bus")
        route = str(settings.core_page_fold_route)
        timeout = float(settings.core_page_fold_timeout_sec)
        payload = ChatRequestPayload(
            messages=[LLMMessage(role="user", content=prompt)],
            route=route,
            options={
                "llm_route": route,
                "max_tokens": int(settings.core_page_fold_max_tokens),
                "temperature": 0.3,
                "purpose": "core_page_fold",
                "chat_template_kwargs": {"enable_thinking": False},
                "skip_spark_candidate_publish": True,
                "gateway_read_timeout_sec": timeout,
            },
        )
        corr = uuid.uuid4()
        reply_channel = f"orion:exec:result:LLMGatewayService:{corr}"
        envelope = BaseEnvelope(kind="llm.chat.request", source=source, correlation_id=corr,
                                reply_to=reply_channel, payload=payload.model_dump(mode="json"))
        msg = await bus.rpc_request(settings.llm_intake_channel, envelope, reply_channel=reply_channel,
                                    timeout_sec=timeout + 30.0, require_listener=True)
        decoded = bus.codec.decode(msg.get("data"))
        if not decoded.ok or decoded.envelope is None:
            raise RuntimeError(f"gateway_decode_failed:{decoded.error}")
        result = decoded.envelope.payload if isinstance(decoded.envelope.payload, dict) else {}
        raw = result.get("raw") if isinstance(result.get("raw"), dict) else {}
        if raw.get("error"):
            raise RuntimeError(f"gateway_error:{raw.get('error')}")
        text = str(result.get("content") or result.get("text") or "")
        if not text.strip():
            raise RuntimeError("gateway_empty_text")
        return {"text": text, "model": raw.get("model") or result.get("model")}

    return call


def service_source(settings: Any):
    from orion.core.bus.bus_schemas import ServiceRef

    return ServiceRef(name=settings.service_name, version=settings.service_version, node=settings.node_name)


def bus_publisher(bus: Any, source: Any) -> Publish:
    """Publish one row as a ``BaseEnvelope`` (reconnecting once on a dropped connection). False on failure."""
    from orion.core.bus.bus_schemas import BaseEnvelope
    from orion.core.bus.resilience import publish_with_reconnect

    async def publish(channel: str, kind: str, model: Any, corr: uuid.UUID) -> bool:
        try:
            await publish_with_reconnect(
                bus, channel,
                BaseEnvelope(kind=kind, source=source, correlation_id=corr, payload=model.model_dump(mode="json")),
                log_label="core_page_fold_publish")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("core_page_fold_publish_failed channel=%s err=%s", channel, exc)
            return False

    return publish
