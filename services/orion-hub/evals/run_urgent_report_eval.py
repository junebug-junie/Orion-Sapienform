#!/usr/bin/env python3
"""Urgent report eval: does every urgent-run outcome end in a critical Hub +
email notice that leads with the right thing?

Plan 3 Task 8 of docs/superpowers/plans/2026-09-28-urgent-curiosity-plan-3-seeded-urgent-runs.md
(spec 5a). Offline -- no Redis, notify, Postgres or graph. Drives the real code:

- Graph rows go through `orion.curiosity.incident_report.read_incident_report`.
- Terminal outcomes (completed / failed / cancelled) go through the real
  `CuriosityInvestigation._handle_run_state` as `orion:durable:run:state`
  envelopes, which picks the report kind and calls the real
  `compose_urgent_report`.
- Timeout, no GPU and unconfirmed dispatch go through the real
  `UrgentReporter.watch` timers with a fake clock and a fake run-state reader.

Per case: severity critical, email requested, the expected flag leads the body,
the verdict line comes before Orion's prose when there is a verdict, the raw
evidence bundle is attached when there is no verdict, and no chat / memory /
journal content planted in the inputs reaches the notice.

    python services/orion-hub/evals/run_urgent_report_eval.py

Exit 0 = every case passed. Exit 1 = at least one failed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import warnings
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_HUB_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _HUB_ROOT.parents[1]
warnings.filterwarnings("ignore", message=r"Field \"model_", category=UserWarning)
# Same order as evals/conftest.py: Hub first so `scripts` is Hub's package.
for _key in [k for k in sys.modules if k == "scripts" or k.startswith("scripts.")]:
    del sys.modules[_key]
for _path in (str(_REPO_ROOT), str(_HUB_ROOT)):
    if _path in sys.path:
        sys.path.remove(_path)
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_HUB_ROOT))

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef  # noqa: E402
from orion.core.bus.codec import OrionCodec  # noqa: E402
from orion.curiosity.incident_report import IncidentReport, read_incident_report  # noqa: E402
from orion.schemas.durable_run import DurableRunStateV1  # noqa: E402
from orion.schemas.notify import NotificationAccepted, NotificationRequest  # noqa: E402
from scripts.curiosity_investigation import URGENT_INCIDENTS_KEY, CuriosityInvestigation  # noqa: E402
from scripts.urgent_report import UrgentReporter  # noqa: E402

SOURCE = ServiceRef(name="orion-hub", version="eval", node="athena")
INCIDENT = "e7" * 16
RUN = "abc123def456"
NOW = datetime(2026, 9, 28, 20, 0, tzinfo=timezone.utc)
PROSE = "Athena's fan reads zero rpm while GPU load is flat, so the heat is real."
# Planted in fields the composer must never read. Any of these in a notice is a leak.
LEAK_SENTINELS = ("SENTINEL_CHAT", "SENTINEL_MEMORY", "SENTINEL_JOURNAL")
EVIDENCE = {
    "hosts": {"athena": {"measurements": {"temp_c_max": 88.0, "fan_rpm": 0}}},
    "pool": {"active_leases": 1, "queued": 0},
    "collected_at": NOW.isoformat(),
}
URGENT = {
    "incident_id": INCIDENT,
    "trigger": "manual",
    "subject": "athena",
    "question": "Why is athena at 88C?",
    "requested_at": NOW.isoformat(),
}
BUNDLE_MARK = "Evidence bundle at request time:"
PROSE_MARK = "Orion's words:"
VERDICT_MARK = "Verdict:"


def _incident(**over: Any) -> dict[str, Any]:
    base = {
        **URGENT,
        "run_id": RUN,
        "requested_by": "juniper",
        "evidence": EVIDENCE,
        "status": "dispatched",
        "chat_history": "SENTINEL_CHAT",
        "memory_cards": ["SENTINEL_MEMORY"],
        "journal_entry": "SENTINEL_JOURNAL",
    }
    base.update(over)
    return base


def _row(**over: Any) -> dict[str, Any]:
    """One `incident_report_for_run_cypher` row (one per evidence entry)."""
    base = {
        "node_id": 7,
        "run_id": RUN,
        "incident_id": INCIDENT,
        "is_real": "real",
        "likely_cause": "radiator fan stalled",
        "evidence": "athena temp_c_max 88.0",
        "severity": "critical",
        "operator_action": "power down athena and check the radiator fan",
        "confidence": 0.82,
        "written_at": 1790000000000,
    }
    base.update(over)
    return base


class _GraphReader:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def query(self, cypher: str) -> list[dict[str, Any]]:
        assert f"r.run_id = '{RUN}'" in cypher
        return self.rows


def _report_detail(report: Optional[IncidentReport]) -> Optional[dict[str, Any]]:
    # Same shape as durable-runs `runner.incident_report_to_detail` (not
    # importable here: both services own a top-level `app` package).
    return None if report is None else {**asdict(report), "evidence": list(report.evidence)}


class _Redis:
    def __init__(self) -> None:
        self.values: dict[str, Any] = {}
        self.hashes: dict[str, dict[str, Any]] = {}

    async def get(self, key):
        return self.values.get(key)

    async def set(self, key, value, *, ex=None, nx=False):
        if nx and key in self.values:
            return None
        self.values[key] = value
        return True

    async def delete(self, *keys):
        return sum(self.values.pop(k, None) is not None for k in keys)

    async def hget(self, key, field):
        value = self.hashes.get(key, {}).get(field)
        return value.encode() if isinstance(value, str) else value

    async def hset(self, key, field, value):
        self.hashes.setdefault(key, {})[field] = value
        return 1

    async def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

    async def hlen(self, key):
        return len(self.hashes.get(key, {}))

    async def hdel(self, key, *fields):
        h = self.hashes.get(key, {})
        return sum(h.pop(f, None) is not None for f in fields)


class _Bus:
    def __init__(self) -> None:
        self.codec = OrionCodec()
        self.redis = _Redis()


class _CapturingReporter:
    def __init__(self) -> None:
        self.delivered: list[tuple[str, NotificationRequest]] = []

    async def deliver(self, incident, request, *, kind):
        self.delivered.append((kind, request))
        return True


class _Notify:
    def __init__(self) -> None:
        self.sent: list[NotificationRequest] = []

    def send(self, request):
        self.sent.append(request)
        return NotificationAccepted(ok=True)


async def _no_sleep(_delay: float) -> None:
    return None


def _terminal(status: str, node: str, detail: dict[str, Any]) -> tuple[str, NotificationRequest]:
    """A durable run-state event through the real Hub handler."""
    bus = _Bus()
    bus.redis.hashes[URGENT_INCIDENTS_KEY] = {INCIDENT: json.dumps(_incident())}
    loop = CuriosityInvestigation(
        enabled=True, tick_interval_sec=60.0, min_cooldown_sec=14400.0, daily_cap=3,
        timeout_sec=8840.0, session_id="orion_curiosity", pool_provider=lambda: None, source_ref=SOURCE,
        kickoff_via_cortex=True, durable_admission_enabled=True, urgent_enabled=True,
    )
    loop._bus = bus
    loop.urgent_reporter = _CapturingReporter()
    reached_out: list[Any] = []

    async def _reach(**kwargs):
        reached_out.append(kwargs)

    loop._maybe_reach_out = _reach  # type: ignore[assignment]
    state = DurableRunStateV1(
        run_id=RUN, workflow="curiosity.investigate", thread_id=RUN, node=node, status=status,
        correlation_id="eval", detail=detail,
    )
    env = BaseEnvelope(kind="durable.run.state.v1", source=SOURCE, payload=state.model_dump(mode="json"))

    async def scenario() -> None:
        await loop._handle_run_state({"data": bus.codec.encode(env)})
        await asyncio.gather(*list(loop._urgent_report_tasks))

    asyncio.run(scenario())
    if reached_out:
        raise AssertionError("urgent run reached out like an ordinary run")
    [(kind, request)] = loop.urgent_reporter.delivered
    return kind, request


def _final(rows: list[dict[str, Any]], **extra: Any) -> tuple[str, NotificationRequest]:
    report, flag = read_incident_report(_GraphReader(rows), RUN)
    detail = {
        "urgent": URGENT,
        "incident_report": _report_detail(report),
        "report_flag": flag,
        "finding_text": PROSE,
        "memory_recall": "SENTINEL_MEMORY",
        "chat_turns": "SENTINEL_CHAT",
        **extra,
    }
    return _terminal("completed", "finish", detail)


# The turn hit its limit (or finalize failed) and Hub handed back Orion's draft
# (run a153451fe423): the notice must still carry their words, marked unfinished.
SALVAGED = {"draft_salvaged": True, "salvaged_from_error": "finalize_reply_deadline"}
UNFINISHED_LINE = "UNFINISHED: Orion's turn ended before their answer was finalized (finalize_reply_deadline)"


def _watched(progress: Optional[dict[str, Any]], want: str, **incident_over: Any) -> tuple[str, NotificationRequest]:
    """The two in-process watchdog timers, clock collapsed to zero."""
    notify = _Notify()

    async def reader(run_id: str):
        return progress

    settings = type("S", (), {"HUB_CURIOSITY_URGENT_GRANT_WAIT_SEC": 120.0, "HUB_CURIOSITY_URGENT_TIMEOUT_SEC": 1200.0})
    reporter = UrgentReporter(notify=notify, redis=None, settings=settings, run_state_reader=reader, sleep=_no_sleep)

    async def scenario() -> None:
        reporter.watch(_incident(**incident_over))
        await asyncio.gather(*list(reporter._tasks))

    asyncio.run(scenario())
    kinds = [req.tags[-1] for req in notify.sent]
    matches = [req for req in notify.sent if req.tags[-1] == want]
    if len(matches) != 1:
        raise AssertionError(f"expected one {want!r} notice, watchdog sent {kinds}")
    return want, matches[0]


# name, runner, expected kind, flag that must open the body (None = verdict
# opens it), bundle required, text that must also appear
CASES: list[tuple[str, Any, str, Optional[str], bool, tuple[str, ...]]] = [
    (
        "valid_report", lambda: _final([_row(), _row(evidence="fan rpm 0")]), "final", None, False,
        ("- athena temp_c_max 88.0", "- fan rpm 0", PROSE),
    ),
    (
        "malformed_report", lambda: _final([_row(is_real="maybe")]), "final", "FLAG: no_structured_verdict", True,
        (PROSE,),
    ),
    (
        "empty_evidence_report", lambda: _final([_row(evidence=None)]), "final", "FLAG: no_structured_verdict", True,
        (PROSE,),
    ),
    (
        "salvaged_draft_no_report", lambda: _final([], **SALVAGED), "final", "FLAG: no_structured_verdict", True,
        (UNFINISHED_LINE, PROSE),
    ),
    (
        "salvaged_draft_with_report", lambda: _final([_row()], **SALVAGED), "final", None, False,
        (UNFINISHED_LINE, PROSE),
    ),
    (
        "failed_run",
        lambda: _terminal("failed", "failed", {"error": "HarnessTurnFailed: empty_generation", "urgent": URGENT}),
        "failed", "investigation failed: HarnessTurnFailed: empty_generation", True, (),
    ),
    (
        "cancelled_run",
        lambda: _terminal("cancelled", "finish", {"error": "cancelled", "urgent": URGENT}),
        "failed", "investigation failed: cancelled", True, (),
    ),
    (
        "timeout",
        lambda: _watched({"past_resource_wait": True, "terminal": None, "detail": {}}, "timeout"),
        "timeout", "INCOMPLETE: no result after 1200 s", True, ("stopped at its deadline",),
    ),
    (
        "no_gpu",
        lambda: _watched({"past_resource_wait": False, "terminal": None, "detail": {}}, "no_gpu"),
        "no_gpu", "not investigated: still waiting for a GPU after 120 s", True, (),
    ),
    (
        "dispatch_unconfirmed",
        # Cortex never registered it: a failed report, never "still queued".
        lambda: _watched(None, "failed", status="dispatch_unconfirmed"),
        "failed", "investigation failed: cortex never registered the run (no run record after 120 s)", True,
        (),
    ),
]


def check_case(
    request: NotificationRequest,
    *,
    kind: str,
    want_kind: str,
    flag: Optional[str],
    bundle: bool,
    must_contain: tuple[str, ...] = (),
) -> list[str]:
    """Every problem with one notice; empty means it passed."""
    problems: list[str] = []
    body = request.body_text or ""
    if kind != want_kind:
        problems.append(f"kind {kind!r} != {want_kind!r}")
    if request.severity != "critical":
        problems.append(f"severity {request.severity!r}")
    if "email" not in (request.channels_requested or []):
        problems.append("email not requested")
    if flag is None:
        if not body.startswith(VERDICT_MARK):
            problems.append("body does not open with the verdict line")
        if VERDICT_MARK not in body or PROSE_MARK not in body:
            problems.append("verdict or prose missing")
        elif body.index(PROSE_MARK) < body.index(VERDICT_MARK):
            problems.append("prose comes before the verdict")
    elif not body.startswith(flag):
        problems.append(f"body does not open with flag {flag!r}")
    missing = [text for text in must_contain if text not in body]
    if missing:
        problems.append(f"missing {missing}")
    if bundle and BUNDLE_MARK not in body:
        problems.append("evidence bundle missing")
    if not bundle and BUNDLE_MARK in body:
        problems.append("evidence bundle attached despite a verdict")
    wire = request.model_dump_json()
    leaked = [s for s in LEAK_SENTINELS if s in wire]
    if leaked or "juniper" in body:
        problems.append(f"leaked {leaked or ['requested_by']}")
    bad_keys = [k for k in request.context if any(w in k for w in ("chat", "memory", "journal"))]
    if bad_keys:
        problems.append(f"context has {bad_keys}")
    return problems


def run_cases() -> list[tuple[str, str, str, list[str]]]:
    """(case, kind, title, problems) per case."""
    results = []
    for name, runner, want_kind, flag, bundle, must_contain in CASES:
        try:
            kind, request = runner()
        except Exception as exc:  # noqa: BLE001 -- a crash is a failed case, not a crashed eval
            results.append((name, "-", "-", [f"{type(exc).__name__}: {exc}"]))
            continue
        problems = check_case(
            request, kind=kind, want_kind=want_kind, flag=flag, bundle=bundle, must_contain=must_contain
        )
        results.append((name, kind, request.title, problems))
    return results


def main() -> int:
    # The malformed cases log a warning by design; the per-case lines say it.
    logging.getLogger("orion.curiosity.incident_report").setLevel(logging.ERROR)
    results = run_cases()
    failures = [name for name, _, _, problems in results if problems]
    for name, kind, title, problems in results:
        status = "PASS" if not problems else "FAIL " + "; ".join(problems)
        flag = next((case[3] for case in CASES if case[0] == name), None) or "verdict"
        print(f"{name:<22} kind={kind:<8} lead={flag[:40]!r:<44} title={title!r} {status}")
    print(f"cases: {len(results)}  failed: {len(failures)}")
    print("VERDICT:", "PASS" if not failures else f"FAIL {failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
