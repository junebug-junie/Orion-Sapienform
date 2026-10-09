"""The must-deliver report for urgent curiosity runs.

Every urgent run ends in a critical Hub + email notice: a final verdict, a
failure, an INCOMPLETE at the overall deadline, or "not investigated" when no
GPU was granted in time. Nothing ends silently.

- `compose_urgent_report` builds the notice from the incident record (Hub's
  `orion:curiosity:urgent:incidents` hash) and the run's finish detail only --
  hardware/pool evidence and Orion's own report, never chat or memory.
- `UrgentReporter.deliver` dedupes per `(incident_id, kind)` in Redis before
  sending, because orion-notify stores `dedupe_key` but never enforces it, and
  retries a refused send with backoff for up to 30 minutes.
- `UrgentReporter.watch` runs the two in-process timers. They do not survive a
  Hub restart; the final/failed notice still goes out because it rides the
  durable run-state event (`CuriosityInvestigation._handle_run_state`).

Plan: docs/superpowers/plans/2026-09-28-urgent-curiosity-plan-3-seeded-urgent-runs.md (Task 6)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Awaitable, Callable, Literal, Optional

from orion.schemas.notify import NotificationRequest

from .curiosity_investigation import URGENT_INCIDENTS_KEY

logger = logging.getLogger("orion-hub.urgent_report")

ReportKind = Literal["final", "failed", "timeout", "no_gpu"]

URGENT_SENT_KEY_PREFIX = "orion:curiosity:urgent:sent:"
URGENT_SENT_TTL_SEC = 7 * 24 * 3600
URGENT_DELIVERY_WINDOW_SEC = 30 * 60
_BACKOFF_SEC = (2, 4, 8, 16, 32, 60)
EVIDENCE_TEXT_CAP = 6000

# Watchdog kinds report on a run that is still going; they must not overwrite
# a terminal status in the incident hash when they land late.
_OPEN_STATUSES = frozenset({"dispatched", "dispatch_unconfirmed", "reported_no_gpu", "reported_timeout"})
_WATCHDOG_KINDS = frozenset({"timeout", "no_gpu"})
_TERMINAL_KINDS = frozenset({"final", "failed"})


def urgent_sent_key(incident_id: str, kind: str) -> str:
    return f"{URGENT_SENT_KEY_PREFIX}{incident_id}:{kind}"


def retry_delay(attempt: int) -> int:
    """Seconds to wait after refused attempt `attempt` (1-based): 2, 4, 8, 16, 32, 60, 60, ..."""
    return _BACKOFF_SEC[min(max(attempt, 1), len(_BACKOFF_SEC)) - 1]


def _label(incident: dict) -> str:
    subject = str(incident.get("subject") or "").strip()
    return subject or str(incident.get("question") or "").strip()[:60]


def _evidence_text(evidence: Any) -> str:
    text = json.dumps(evidence if evidence is not None else {}, indent=2, sort_keys=True, default=str)
    if len(text) <= EVIDENCE_TEXT_CAP:
        return text
    return f"{text[:EVIDENCE_TEXT_CAP]}\n... (truncated, {len(text) - EVIDENCE_TEXT_CAP} more chars)"


UNFINISHED_MARK = "UNFINISHED:"


def compose_urgent_report(
    incident: dict,
    *,
    kind: ReportKind,
    detail: Optional[dict] = None,
    reason: str = "",
) -> NotificationRequest:
    """The notice for one outcome. Reads only the incident record and the finish
    detail's `incident_report` / `report_flag` / `finding_text`."""
    detail = detail or {}
    incident_id = str(incident.get("incident_id") or "")
    run_id = str(incident.get("run_id") or "")
    label = _label(incident)
    report = detail.get("incident_report") if kind == "final" else None
    report = report if isinstance(report, dict) else None
    report_flag = str(detail.get("report_flag") or "") if kind == "final" else ""
    if kind == "final" and report is None and not report_flag:
        report_flag = "no_structured_verdict"

    flags: list[str] = []
    attach_bundle = True
    if kind == "final":
        if report is not None:
            title = f"URGENT: {report.get('is_real')} / {report.get('severity')} — {label}"
            attach_bundle = False
        else:
            title = f"URGENT: no structured verdict — {label}"
        if report_flag:
            flags.append(
                f"FLAG: {report_flag} — Orion did not leave a usable IncidentReport; their own words are below."
            )
    elif kind == "failed":
        title = f"Urgent investigation failed — {label}"
        flags.append(f"investigation failed: {reason or 'unknown'}")
    elif kind == "timeout":
        title = f"URGENT: incomplete — {label}"
        flags.append(
            f"INCOMPLETE: {reason or 'no result yet'}. The run is being stopped at its deadline; "
            "its final report (or its failure) will follow when it ends."
        )
    else:
        title = f"URGENT: not investigated — {label}"
        flags.append(f"not investigated: {reason or 'no GPU granted in time'}. The run is still queued.")
        if incident.get("status") == "dispatch_unconfirmed":
            flags.append("Dispatch was unconfirmed: cortex never confirmed the run was registered.")

    lines: list[str] = list(flags)
    if report is not None:
        confidence = report.get("confidence")
        conf_text = f"{float(confidence):.2f}" if isinstance(confidence, (int, float)) else str(confidence)
        lines.append(f"Verdict: {report.get('is_real')} / severity {report.get('severity')} / confidence {conf_text}")
        lines.append(f"Operator action: {report.get('operator_action')}")
        lines.append(f"Likely cause: {report.get('likely_cause')}")
        cited = [str(item) for item in (report.get("evidence") or [])]
        if cited:
            lines.append("Cited evidence:\n" + "\n".join(f"- {item}" for item in cited))
    if attach_bundle:
        lines.append("Evidence bundle at request time:\n" + _evidence_text(incident.get("evidence")))
    finding_text = str(detail.get("finding_text") or "").strip() if kind == "final" else ""
    if finding_text:
        if detail.get("draft_salvaged"):
            # Next to the words it qualifies, so a real verdict still leads the notice.
            why = str(detail.get("salvaged_from_error") or "").strip()
            lines.append(
                f"{UNFINISHED_MARK} Orion's turn ended before their answer was finalized"
                + (f" ({why})" if why else "")
                + "; the words below are their working draft, unreviewed."
            )
        lines.append(f"Orion's words:\n{finding_text}")
    lines.append(f"Trigger: {incident.get('trigger') or 'unknown'}")
    lines.append(f"Question: {incident.get('question') or ''}")
    lines.append(f"Incident {incident_id} · run {run_id or '-'} · requested {incident.get('requested_at') or '-'}")

    context: dict[str, Any] = {
        "incident_id": incident_id,
        "run_id": run_id,
        "kind": kind,
        "trigger": incident.get("trigger"),
        "subject": incident.get("subject"),
        "report_flag": report_flag or None,
        "draft_salvaged": True if kind == "final" and detail.get("draft_salvaged") else None,
        "is_real": report.get("is_real") if report else None,
        "severity": report.get("severity") if report else None,
    }
    return NotificationRequest(
        source_service="orion-hub",
        event_kind="curiosity.urgent.report",
        severity="critical",
        title=title,
        body_text="\n\n".join(lines),
        context={k: v for k, v in context.items() if v is not None},
        tags=["curiosity", "urgent", kind],
        channels_requested=["in_app", "email"],
        dedupe_key=f"urgent:{incident_id}:{kind}",
        correlation_id=run_id or None,
    )


# --- run progress (watchdog) -------------------------------------------------
#
# Urgent runs always go through durable admission. On that path the bridge
# table `substrate_durable_run_state` receives only the terminal row (live
# 2026-09-28: 87 `finish` + 9 `failed` curiosity rows in 7 days, nothing
# else), so "got past resource_wait" is read from the admission events --
# the same tables and pool `curiosity_run_store` reads.

URGENT_ADMISSION_SQL = "SELECT terminal, control FROM durable_admission_runs WHERE run_id = $1"
URGENT_STATE_SQL = (
    "SELECT node, status, detail::text AS detail FROM substrate_durable_run_state "
    "WHERE run_id = $1 ORDER BY created_at DESC LIMIT 1"
)
URGENT_PROGRESS_EVENTS_SQL = (
    "SELECT 1 AS one FROM durable_resource_events WHERE run_id = $1 AND event = ANY($2::text[]) LIMIT 1"
)
# The terminal outbox event (orion/durable_runs/registry_store.py finish_projection) carries the
# same detail the run-state event does, committed with `terminal` in one transaction.
URGENT_TERMINAL_EVENT_SQL = "SELECT payload->'detail' AS detail FROM durable_resource_events WHERE entry_id = $1"
PAST_RESOURCE_WAIT_EVENTS = ("run.resource_granted", "run.admitted", "run.running", "run.started")
# A completed run whose detail cannot be read yet is re-read once after this long.
MISSED_DETAIL_RETRY_SEC = 60.0
_TERMINAL_STATUSES = ("completed", "failed", "cancelled")


def _json_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _state_terminal(state: Any) -> Optional[str]:
    if state is None:
        return None
    status = str(state["status"] or "")
    if status == "completed" or status == "cancelled":
        return status
    # A non-admitted runner emits resumable `failed` on any node; only the
    # terminal `failed` node ends the run.
    if status == "failed" and str(state["node"] or "") == "failed":
        return "failed"
    return None


async def read_urgent_run_progress(pool: Any, run_id: str) -> Optional[dict[str, Any]]:
    """`None` when no store knows the run; else
    `{"past_resource_wait": bool, "terminal": str | None, "detail": dict}`.

    Raises when there is no pool or a read fails -- the watchdog reports that
    as "run state unreadable" rather than guessing."""
    if pool is None:
        raise RuntimeError("no_postgres_pool")
    async with pool.acquire() as conn:
        admission = await conn.fetchrow(URGENT_ADMISSION_SQL, run_id)
        state = await conn.fetchrow(URGENT_STATE_SQL, run_id)
        if admission is None and state is None:
            return None
        progressed = await conn.fetchrow(URGENT_PROGRESS_EVENTS_SQL, run_id, list(PAST_RESOURCE_WAIT_EVENTS))
        terminal = _state_terminal(state)
        detail = _json_dict(state["detail"]) if terminal is not None else {}
        if terminal is None and admission is not None:
            admitted_terminal = str(admission["terminal"] or "")
            terminal = admitted_terminal if admitted_terminal in _TERMINAL_STATUSES else None
            if terminal is not None:
                # The bridge row lags (sql-writer) or never landed: the outbox event has the detail.
                event = await conn.fetchrow(URGENT_TERMINAL_EVENT_SQL, f"{run_id}:terminal:{terminal}")
                detail = _json_dict(event["detail"]) if event is not None else {}
    past = bool(
        terminal
        or progressed is not None
        or (state is not None and str(state["node"] or "") not in ("", "resource_wait"))
    )
    return {"past_resource_wait": past, "terminal": terminal, "detail": detail}


class UrgentReporter:
    def __init__(
        self,
        *,
        notify: Any,
        redis: Any,
        settings: Any,
        run_state_reader: Callable[[str], Awaitable[Optional[dict[str, Any]]]],
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
        # (incident_id, run_id) -> frees the incident's open key only while it
        # still holds that run (CuriosityInvestigation.release_urgent_open_key_for).
        release_open_key: Optional[Callable[[str, str], Awaitable[None]]] = None,
    ) -> None:
        self._notify = notify
        self._redis = redis
        self._reader = run_state_reader
        self._release_open_key = release_open_key
        self._sleep = sleep
        self._clock = clock
        self.grant_wait_sec = float(getattr(settings, "HUB_CURIOSITY_URGENT_GRANT_WAIT_SEC", 120.0))
        self.timeout_sec = float(getattr(settings, "HUB_CURIOSITY_URGENT_TIMEOUT_SEC", 1200.0))
        self._inflight: set[tuple[str, str]] = set()
        self._terminal: set[str] = set()
        self._tasks: set[asyncio.Task] = set()

    # --- delivery ---------------------------------------------------------------

    async def deliver(self, incident: dict, request: NotificationRequest, *, kind: str) -> bool:
        """True once notify accepted this `(incident, run, kind)` -- now or earlier.

        The sent key is per incident and kind but holds the run id, so a retried
        run for the same incident is still reported."""
        incident_id = str(incident.get("incident_id") or "")
        run_id = str(incident.get("run_id") or "")
        if kind in _TERMINAL_KINDS:
            self._terminal.add(run_id)
        key = (incident_id, run_id, kind)
        if key in self._inflight:
            return True
        self._inflight.add(key)
        try:
            if await self._already_sent(incident_id, kind, run_id):
                logger.info("urgent_report_already_sent incident_id=%s kind=%s", incident_id, kind)
                return True
            started = self._clock()
            attempt = 0
            while True:
                attempt += 1
                detail = await self._send(request)
                if detail is None:
                    await self._mark_sent(incident_id, kind, run_id)
                    await self._set_status(incident, f"reported_{kind}", kind=kind)
                    logger.info(
                        "urgent_report_delivered incident_id=%s kind=%s attempt=%d", incident_id, kind, attempt
                    )
                    return True
                delay = retry_delay(attempt)
                if self._clock() - started + delay > URGENT_DELIVERY_WINDOW_SEC:
                    break
                logger.warning(
                    "urgent_report_retry incident_id=%s kind=%s attempt=%d delay=%ss detail=%s",
                    incident_id, kind, attempt, delay, detail,
                )
                await self._sleep(delay)
            logger.error(
                "urgent_report_undelivered incident_id=%s kind=%s attempts=%d", incident_id, kind, attempt
            )
            await self._set_status(incident, "report_undelivered", kind=kind)
            return False
        finally:
            self._inflight.discard(key)

    async def _send(self, request: NotificationRequest) -> Optional[str]:
        """None when accepted; else why not."""
        try:
            result = await asyncio.to_thread(self._notify.send, request)
        except Exception as exc:  # noqa: BLE001 -- NotifyClient never raises; a fake or a bug might
            return f"{type(exc).__name__}: {exc}"
        if getattr(result, "ok", False):
            return None
        return str(getattr(result, "detail", None) or "not accepted")

    async def _already_sent(self, incident_id: str, kind: str, run_id: str) -> bool:
        if self._redis is None:
            return False
        try:
            held = await self._redis.get(urgent_sent_key(incident_id, kind))
        except Exception:  # noqa: BLE001 -- unsure is "not sent": a duplicate beats silence
            logger.warning("urgent_report_sent_check_failed incident_id=%s kind=%s", incident_id, kind, exc_info=True)
            return False
        if held is None:
            return False
        held = held.decode() if isinstance(held, bytes) else str(held)
        return held == run_id

    async def _mark_sent(self, incident_id: str, kind: str, run_id: str) -> None:
        if self._redis is None:
            return
        try:
            await self._redis.set(urgent_sent_key(incident_id, kind), run_id, ex=URGENT_SENT_TTL_SEC)
        except Exception:  # noqa: BLE001
            logger.warning("urgent_report_mark_sent_failed incident_id=%s kind=%s", incident_id, kind, exc_info=True)

    async def _set_status(self, incident: dict, status: str, *, kind: str) -> None:
        if self._redis is None:
            return
        incident_id = str(incident.get("incident_id") or "")
        try:
            raw = await self._redis.hget(URGENT_INCIDENTS_KEY, incident_id)
            record = json.loads(raw) if raw else dict(incident)
            if str(record.get("run_id") or "") != str(incident.get("run_id") or ""):
                # The record is another run's (a retry of the same incident): not ours to mark.
                return
            if kind in _WATCHDOG_KINDS and str(record.get("status") or "") not in _OPEN_STATUSES:
                return
            record["status"] = status
            await self._redis.hset(URGENT_INCIDENTS_KEY, incident_id, json.dumps(record, default=str))
        except Exception:  # noqa: BLE001
            logger.warning("urgent_report_status_failed incident_id=%s status=%s", incident_id, status, exc_info=True)

    # --- entry points from CuriosityInvestigation.start_urgent ---------------------

    async def close(self) -> None:
        """Cancel the timers and background deliveries (Hub shutdown)."""
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    def _spawn(self, coro: Awaitable[Any]) -> asyncio.Task:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def dispatch_failed(self, incident: dict, reason: str) -> None:
        """Dispatch raised before any run could exist: report now, in the background."""
        request = compose_urgent_report(incident, kind="failed", reason=reason)
        self._spawn(self.deliver(incident, request, kind="failed"))

    def watch(self, incident: dict) -> None:
        """Schedule the no-GPU check (grant wait) and the overall-deadline check."""
        self._spawn(self._check_grant(dict(incident)))
        self._spawn(self._check_deadline(dict(incident)))

    async def _progress(self, run_id: str) -> tuple[Optional[dict[str, Any]], str]:
        try:
            return await self._reader(run_id), ""
        except Exception as exc:  # noqa: BLE001
            return None, f"run state unreadable ({type(exc).__name__}: {str(exc)[:120]})"

    async def _release(self, incident: dict) -> None:
        if self._release_open_key is None:
            return
        incident_id = str(incident.get("incident_id") or "")
        try:
            await self._release_open_key(incident_id, str(incident.get("run_id") or ""))
        except Exception:  # noqa: BLE001 -- the key's own TTL still frees it
            logger.warning("urgent_report_release_failed incident_id=%s", incident_id, exc_info=True)

    async def _check_grant(self, incident: dict) -> None:
        await self._sleep(self.grant_wait_sec)
        run_id = str(incident.get("run_id") or "")
        if run_id in self._terminal:
            return
        progress, error = await self._progress(run_id)
        if progress is None and not error and incident.get("status") == "dispatch_unconfirmed":
            # durable-runs commits the admission row before it publishes the
            # receipt (app/main.py: `await admission.submit` then publish), and the
            # request channel is pub/sub, so no row by now means no run will ever
            # exist. Ends the incident: freed first (delivery can retry for a long
            # time), the deadline check stays quiet, then the failed report.
            self._terminal.add(run_id)
            await self._release(incident)
            reason = f"cortex never registered the run (no run record after {self.grant_wait_sec:.0f} s)"
            request = compose_urgent_report(incident, kind="failed", reason=reason)
            await self.deliver(incident, request, kind="failed")
            return
        if error:
            reason = f"could not confirm a GPU grant after {self.grant_wait_sec:.0f} s: {error}"
        elif progress is None:
            reason = f"no run record found after {self.grant_wait_sec:.0f} s"
        elif progress.get("past_resource_wait") or progress.get("terminal"):
            return
        else:
            reason = f"still waiting for a GPU after {self.grant_wait_sec:.0f} s"
        if run_id in self._terminal:
            return
        await self.deliver(incident, compose_urgent_report(incident, kind="no_gpu", reason=reason), kind="no_gpu")

    async def _check_deadline(self, incident: dict) -> None:
        await self._sleep(self.timeout_sec)
        run_id = str(incident.get("run_id") or "")
        if run_id in self._terminal:
            return
        progress, error = await self._progress(run_id)
        terminal = (progress or {}).get("terminal")
        if terminal == "completed" and not (progress or {}).get("detail"):
            # Completed, but its result is not readable yet. A "no structured verdict"
            # final now would dedupe-block the real one: read once more later.
            await self._sleep(MISSED_DETAIL_RETRY_SEC)
            if run_id in self._terminal:
                return
            progress, error = await self._progress(run_id)
            terminal = (progress or {}).get("terminal")
        if terminal:
            # Ended, but this process never saw the run-state event. Report it
            # from the run store; the sent key stops a later duplicate.
            kind: ReportKind = "final" if terminal == "completed" else "failed"
            detail = (progress or {}).get("detail") or {}
            # The run is over: free the incident before delivery, which can retry for a long time.
            await self._release(incident)
            if kind == "final" and not detail:
                # Still unreadable: say so as "failed", which dedupes separately
                # from "final", so a late real verdict can still go out.
                logger.error(
                    "urgent_report_missed_terminal_unreadable incident_id=%s run=%s",
                    incident.get("incident_id"), run_id,
                )
                reason = f"run {run_id} completed but its result could not be read; check it in the curiosity atlas"
                await self.deliver(incident, compose_urgent_report(incident, kind="failed", reason=reason), kind="failed")
            else:
                reason = str(detail.get("error") or terminal)
                request = compose_urgent_report(incident, kind=kind, detail=detail, reason=reason)
                await self.deliver(incident, request, kind=kind)
            return
        reason = f"no result after {self.timeout_sec:.0f} s"
        if error:
            reason = f"{reason}; {error}"
        elif progress is None:
            reason = f"{reason}; no run record found"
        if run_id in self._terminal:
            return
        await self.deliver(incident, compose_urgent_report(incident, kind="timeout", reason=reason), kind="timeout")
