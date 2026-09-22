"""Curiosity Atlas — a read-only operator surface for Orion's world view.

WHY. On 2026-08-27 the accumulation loop went to zero and the only symptom was
one log line — `priors=0/0` — that nobody read for four hours. Everything needed
to see it was already in FalkorDB, Postgres, Redis and docker logs; nothing put
it on one screen. This module is that screen.

NOTHING HERE WRITES TO ORION'S GRAPH, and that is a design constraint rather
than a phase-one scope cut. Hub never writes to `orion_worldview` — Orion authors
its own graph — and a surface that could edit it would need auth thinking, an
audit trail, and a story about what it means for an operator to overwrite a
belief Orion formed. That has not changed and should not.

There is now exactly one POST, and it is narrower than it looks: `/api/run-now`
asks the loop to take a turn sooner than the cooldown would have. It writes no
memory, no prior, no finding. Orion still authors everything the turn produces.
An earlier version of this docstring said "no POST routes here and there should
not be", which was a claim about writes to the graph stated as a claim about
HTTP verbs; a control action is not a write, and conflating them would have
forced an operator button into a module that has nothing to do with this page.

Follows `concept_atlas_routes.py`: JSON GETs plus a standalone page route,
degrading to an honest "unavailable" payload rather than a 500, because this is
an interpretability surface and a broken panel must never take Hub down.

Three reads feed the page (design: docs/superpowers/specs/2026-09-22-curiosity-
tab-redesign-design.md):

- `/api/atlas`   -- priors, revisions, pool totals, self panel, peer briefs.
- `/api/runs`    -- the 14-day sittings strip: one summary per run across the
                    three lines, plus the reach-out tally and the per-line
                    budget read from the same Redis keys the loop writes.
- `/api/run/{id}` -- one run's story: a clock-ordered timeline joined across
                    lifecycle rows, Orion's graph, the journal, the outreach
                    decision and any reply.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, JSONResponse

from orion.curiosity.atlas import read_atlas, to_payload
from orion.curiosity.run_story import LINES
from orion.curiosity.self_panel import read_self_panel
from orion.curiosity.self_panel import to_payload as self_panel_to_payload
from orion.curiosity.worldview import WorldviewReader

logger = logging.getLogger("orion-hub.curiosity_routes")

# Strong references to in-flight forced turns. asyncio only holds a weak
# reference to a running task, so without this the turn can be garbage-collected
# mid-run -- the button would return ok and nothing would happen.
_RUN_NOW_TASKS: set = set()

router = APIRouter(prefix="/curiosity", tags=["curiosity"])

# Same keys the loop itself uses to reach Orion's graph, read through the same
# settings object -- a second copy of the host/port here is how a dashboard ends
# up describing a graph nobody is writing to.
_NO_CACHE = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}


def _build_reader() -> Optional[WorldviewReader]:
    """The same reader the loop builds, or None when the graph half is off.

    None and "unreachable" are deliberately different: a graph that was never
    configured is not an outage, and rendering the two identically is the exact
    confusion `worldview.read_snapshot` exists to prevent.
    """
    try:
        from app.settings import get_settings

        cfg = get_settings()
    except Exception as exc:  # noqa: BLE001 -- a dashboard never 500s on config
        logger.warning("curiosity_atlas_settings_unavailable err=%s", exc)
        return None
    host = getattr(cfg, "HUB_CURIOSITY_GRAPH_HOST", "") or ""
    user = getattr(cfg, "HUB_CURIOSITY_GRAPH_ORION_USER", "") or ""
    password = getattr(cfg, "HUB_CURIOSITY_GRAPH_ORION_PASSWORD", "") or ""
    if not (host and user and password):
        return None
    return WorldviewReader(
        host=host,
        port=int(getattr(cfg, "HUB_CURIOSITY_GRAPH_PORT", 6380) or 6380),
        graph_name=getattr(cfg, "HUB_CURIOSITY_GRAPH_OWN", "orion_worldview"),
    )


def _stamp_to_next(last: Any, cooldown: float) -> tuple[Optional[str], Optional[str]]:
    """(last ISO, next-eligible ISO) from one cooldown key's value."""
    if isinstance(last, (bytes, bytearray)):
        last = last.decode("utf-8", errors="replace")
    if not last:
        return None, None
    if not cooldown:
        return str(last), None
    try:
        stamped = datetime.fromisoformat(str(last))
    except ValueError:
        # The loop writes this key, not Orion, so a malformed value is a
        # real defect worth seeing rather than swallowing.
        logger.warning("curiosity_atlas_bad_cooldown_stamp value=%r", last)
        return str(last), None
    if stamped.tzinfo is None:
        stamped = stamped.replace(tzinfo=timezone.utc)
    return str(last), (stamped + timedelta(seconds=cooldown)).isoformat()


async def _read_schedule() -> dict[str, Any]:
    """Cooldown, daily cap and next-eligible for each of the three lines --
    the half of the picture that lives in Redis rather than the graph.

    The key names are IMPORTED from the loop that writes them, never retyped.
    A dashboard reading `orion:curiosity:count:...` from its own string literal
    would keep rendering a confident 0 forever the day that prefix changes.

    The daily counter is keyed on the OPERATOR'S LOCAL date, and the zone comes
    from the same setting the loop uses (`HUB_ENDOGENOUS_OUTREACH_TZ`), NOT from
    this process's own locale. The container sets no `TZ`, so `datetime.now()
    .astimezone()` here is UTC: between 18:00 and 23:59 in Juniper's zone that
    reads tomorrow's key, finds nothing, and reports `runs_today: 0` while the
    loop is at cap -- the tile would sit at 0 of 3 every evening and the at-cap
    highlight would never fire. Caught in review; the earlier version of this
    docstring claimed to avoid the exact bug it had.

    The top-level fields describe the investigate line (kept for the one
    existing consumer); `lines` carries all three.
    """
    out: dict[str, Any] = {
        "available": False,
        "local_date": None,
        "tz": None,
        "last_investigation_at": None,
        "next_eligible_at": None,
        "runs_today": None,
        "daily_cap": None,
        "cooldown_sec": None,
        "lines": {},
    }
    try:
        from app.settings import get_settings

        from . import main as hub_main
        from .curiosity_investigation import (
            _COOLDOWN_KEY,
            _DAILY_COUNT_KEY_PREFIX,
            _SELF_COOLDOWN_KEY,
            _SELF_DAILY_COUNT_KEY_PREFIX,
            _SENSE_EVAL_COOLDOWN_KEY,
            _SENSE_EVAL_DAILY_COUNT_KEY_PREFIX,
        )

        cfg = get_settings()
        spec = {
            "investigate": (
                _COOLDOWN_KEY, _DAILY_COUNT_KEY_PREFIX,
                "HUB_CURIOSITY_INVESTIGATION_ENABLED",
                "HUB_CURIOSITY_INVESTIGATION_DAILY_CAP",
                "HUB_CURIOSITY_INVESTIGATION_MIN_COOLDOWN_SEC",
            ),
            "self_inquiry": (
                _SELF_COOLDOWN_KEY, _SELF_DAILY_COUNT_KEY_PREFIX,
                "HUB_CURIOSITY_SELF_INQUIRY_ENABLED",
                "HUB_CURIOSITY_SELF_INQUIRY_DAILY_CAP",
                "HUB_CURIOSITY_SELF_INQUIRY_MIN_COOLDOWN_SEC",
            ),
            "self_sense_eval": (
                _SENSE_EVAL_COOLDOWN_KEY, _SENSE_EVAL_DAILY_COUNT_KEY_PREFIX,
                "HUB_CURIOSITY_SELF_SENSE_EVAL_ENABLED",
                "HUB_CURIOSITY_SELF_SENSE_EVAL_DAILY_CAP",
                "HUB_CURIOSITY_SELF_SENSE_EVAL_MIN_COOLDOWN_SEC",
            ),
        }
        assert set(spec) == set(LINES), "a line was added without a budget tile"
        lines: dict[str, dict[str, Any]] = {}
        for line, (_, _, enabled_key, cap_key, cooldown_key) in spec.items():
            lines[line] = {
                "enabled": bool(getattr(cfg, enabled_key, False)),
                "daily_cap": int(getattr(cfg, cap_key, 0) or 0),
                "cooldown_sec": float(getattr(cfg, cooldown_key, 0) or 0),
                "runs_today": None,
                "last_at": None,
                "next_eligible_at": None,
            }
        out["daily_cap"] = lines["investigate"]["daily_cap"]
        out["cooldown_sec"] = lines["investigate"]["cooldown_sec"]
        out["lines"] = lines

        redis = getattr(getattr(hub_main, "bus", None), "redis", None)
        if redis is None:
            return out

        try:
            tz = ZoneInfo(getattr(cfg, "HUB_ENDOGENOUS_OUTREACH_TZ", "UTC") or "UTC")
        except Exception:  # noqa: BLE001 -- same fallback the loop takes
            tz = timezone.utc
        local_date = datetime.now(timezone.utc).astimezone(tz).date().isoformat()
        for line, (cooldown_key_name, count_prefix, _, _, _) in spec.items():
            last = await redis.get(cooldown_key_name)
            count = await redis.get(f"{count_prefix}{local_date}")
            if isinstance(count, (bytes, bytearray)):
                count = count.decode("utf-8", errors="replace")
            last_iso, next_iso = _stamp_to_next(last, lines[line]["cooldown_sec"])
            lines[line]["runs_today"] = int(count) if count else 0
            lines[line]["last_at"] = last_iso
            lines[line]["next_eligible_at"] = next_iso

        out["available"] = True
        out["local_date"] = local_date
        out["tz"] = str(tz)
        out["last_investigation_at"] = lines["investigate"]["last_at"]
        out["runs_today"] = lines["investigate"]["runs_today"]
        out["next_eligible_at"] = lines["investigate"]["next_eligible_at"]
    except Exception as exc:  # noqa: BLE001 -- a dashboard never 500s
        logger.warning("curiosity_atlas_schedule_unavailable err=%s", exc)
    return out


def _get_memory_pg_pool() -> Any:
    """Hub's shared asyncpg pool, or None if it is not up yet. One place to
    change if `app.state.memory_pg_pool` ever moves -- review finding,
    2026-09-09: this lookup was duplicated inline in two readers in this file."""
    try:
        from . import main as hub_main

        state = getattr(getattr(hub_main, "app", None), "state", None)
        return getattr(state, "memory_pg_pool", None)
    except Exception as exc:  # noqa: BLE001
        logger.warning("memory_pg_pool_lookup_failed err=%s", exc)
        return None


def _runs_on_local_date(
    runs: list[dict[str, Any]], local_date: Optional[str], tz_name: Optional[str]
) -> Optional[dict[str, int]]:
    """How many runs per line the STORES show on the SAME calendar day the
    Redis counter keys on -- so a tile can say "counter 3, stores 2: one run
    left no trace".

    Computed here rather than in the browser. The daily counter is keyed in
    `HUB_ENDOGENOUS_OUTREACH_TZ`, while a browser's date is the viewer's own
    zone, and comparing a count from one zone against a count from another
    makes the tile disagree with the counter for reasons that have nothing to
    do with Orion. A run is dated by its start, else its end; an undated run
    is on no day. None when the server could not name the day at all.
    """
    if not local_date:
        return None
    try:
        tz = ZoneInfo(tz_name) if tz_name else timezone.utc
    except Exception:  # noqa: BLE001
        tz = timezone.utc
    out: dict[str, int] = {}
    for run in runs:
        stamp = run.get("started_at") or run.get("finished_at")
        if not stamp:
            continue
        when = datetime.fromtimestamp(int(stamp) / 1000, timezone.utc).astimezone(tz)
        if when.date().isoformat() == local_date:
            line = str(run.get("line") or "")
            out[line] = out.get(line, 0) + 1
    return out


@router.get("/api/atlas")
async def curiosity_atlas_api() -> JSONResponse:
    """Priors, revisions, pool totals, self panel, peer briefs, schedule.

    One read for every panel that is a projection of the priors, so the
    priors list and its sparklines cannot disagree. Runs are NOT here any
    more: `/api/runs` and `/api/run/{id}` own them, bounded by a window
    rather than by the total number of runs Orion has ever taken.
    """
    reader = _build_reader()
    if reader is None:
        payload: dict[str, Any] = {
            "available": False,
            "reason": "graph_not_configured",
        }
    else:
        # `to_payload` and not just `read_atlas`: building the payload walks
        # every prior's trajectory and deep-copies every dataclass, and Hub runs
        # one uvicorn worker -- doing that on the event loop stalls every
        # connected websocket, which is the rule `WorldviewReader` already
        # states for its own blocking call.
        payload = await asyncio.to_thread(lambda: to_payload(read_atlas(reader)))
    payload["schedule"] = await _read_schedule()
    payload["self"] = await _read_self_panel_payload()
    return JSONResponse(content=payload, headers=_NO_CACHE)


def _run_store():
    """Lazy, like `from . import main`: `tests/test_curiosity_atlas.py` loads
    this module by file path with no parent package, where a top-level
    relative import cannot resolve."""
    from . import curiosity_run_store

    return curiosity_run_store


@router.get("/api/runs")
async def curiosity_runs_api(
    days: Optional[int] = None, line: str = "all"
) -> JSONResponse:
    """The sittings strip: one summary per run in the window, newest first,
    across all three lines (or one), plus the reach-out tally and the
    per-line budget. `days` is clamped to 1..90; an unknown `line` is `all`.
    Never 500s: a dead store is named in `stores`, both dead is
    `available: false`."""
    try:
        store = _run_store()
        payload = await store.read_runs_payload(
            pool=_get_memory_pg_pool(), reader=_build_reader(),
            days=store.clamp_days(days), line=store.clamp_line(line),
        )
    except Exception as exc:  # noqa: BLE001 -- a dashboard never 500s
        logger.warning("curiosity_runs_api_failed err=%s", exc)
        payload = {"available": False, "reason": f"{type(exc).__name__}: {str(exc)[:160]}"}
    payload["schedule"] = await _read_schedule()
    payload["schedule"]["runs_seen_today"] = _runs_on_local_date(
        payload.get("runs", []), payload["schedule"].get("local_date"),
        payload["schedule"].get("tz"),
    )
    return JSONResponse(content=payload, headers=_NO_CACHE)


@router.get("/api/run/{run_id}")
async def curiosity_run_api(run_id: str) -> JSONResponse:
    """One run's story. `found: false` for an id no store knows; the id is
    validated before it reaches any query."""
    try:
        payload = await _run_store().read_run_payload(
            pool=_get_memory_pg_pool(), reader=_build_reader(), run_id=run_id
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("curiosity_run_api_failed run=%s err=%s", run_id[:64], exc)
        payload = {"available": False, "reason": f"{type(exc).__name__}: {str(exc)[:160]}"}
    return JSONResponse(content=payload, headers=_NO_CACHE)


async def _read_self_panel_payload() -> dict[str, Any]:
    """Orion's own definition, its version history, and the self-inquiry
    journal -- read directly from Postgres (see `orion.curiosity.self_panel`
    for why this does not go through the `:TurnOutcome`-keyed run list)."""
    view = await read_self_panel(_get_memory_pg_pool())
    return self_panel_to_payload(view)


@router.post("/api/run-now")
async def curiosity_run_now() -> JSONResponse:
    """Take a turn now, skipping the cooldown and the daily cap.

    Every health gate still applies. `enabled` is NOT overridable: a loop
    switched off is a decision already made, and a button that quietly undid it
    would make the switch meaningless.

    The run still counts against today. A forced run that did not would make the
    daily counter lie, and that counter is what the page compares against to
    notice a run that left no trace.
    """
    try:
        from . import main as hub_main

        loop = getattr(hub_main, "curiosity_investigation", None)
    except Exception as exc:  # noqa: BLE001
        logger.warning("curiosity_run_now_import_failed err=%s", exc)
        loop = None

    if loop is None:
        return JSONResponse(
            content={"ok": False, "reason": "loop_not_running"},
            status_code=503,
            headers=_NO_CACHE,
        )

    logger.warning("curiosity_run_now_requested -- operator asked for a turn")
    try:
        # A turn runs for ~20 minutes and this is a browser request. Fire it and
        # return the acceptance, rather than holding an HTTP connection open
        # across the whole turn and reporting a proxy timeout as a failure.
        task = asyncio.create_task(loop.tick(force=True))
        _RUN_NOW_TASKS.add(task)
        task.add_done_callback(_RUN_NOW_TASKS.discard)
    except Exception as exc:  # noqa: BLE001
        logger.warning("curiosity_run_now_failed err=%s", exc)
        return JSONResponse(
            content={"ok": False, "reason": str(exc)[:200]},
            status_code=500,
            headers=_NO_CACHE,
        )
    return JSONResponse(
        content={
            "ok": True,
            "detail": "Turn requested. It takes ~20 minutes; the page will "
                      "show it once the run writes its first node.",
        },
        headers=_NO_CACHE,
    )


@router.post("/api/self-inquiry/run-now")
async def curiosity_self_inquiry_run_now() -> JSONResponse:
    """Take a SELF-INQUIRY turn now (orion/curiosity/self_inquiry.py),
    skipping that line's cooldown and daily cap. Same rules as run-now above:
    `enabled` and every health gate still apply, and the run still counts
    against the self-inquiry budget."""
    try:
        from . import main as hub_main

        loop = getattr(hub_main, "curiosity_investigation", None)
    except Exception as exc:  # noqa: BLE001
        logger.warning("curiosity_self_inquiry_run_now_import_failed err=%s", exc)
        loop = None
    if loop is None:
        return JSONResponse(
            content={"ok": False, "reason": "loop_not_running"},
            status_code=503,
            headers=_NO_CACHE,
        )
    logger.warning("curiosity_self_inquiry_run_now_requested -- operator asked for a self-inquiry turn")
    try:
        task = asyncio.create_task(loop.tick_self_inquiry(force=True))
        _RUN_NOW_TASKS.add(task)
        task.add_done_callback(_RUN_NOW_TASKS.discard)
    except Exception as exc:  # noqa: BLE001
        logger.warning("curiosity_self_inquiry_run_now_failed err=%s", exc)
        return JSONResponse(
            content={"ok": False, "reason": str(exc)[:200]},
            status_code=500,
            headers=_NO_CACHE,
        )
    return JSONResponse(
        content={
            "ok": True,
            "detail": "Self-inquiry turn requested. It takes ~20 minutes; the "
                      "definition appears in self_concept_history once mirrored.",
        },
        headers=_NO_CACHE,
    )


@router.get("")
@router.get("/")
async def curiosity_atlas_page() -> HTMLResponse:
    from .main import TEMPLATES_DIR, build_hub_ui_asset_version

    template = (TEMPLATES_DIR / "curiosity_atlas.html").read_text(encoding="utf-8")
    rendered = template.replace(
        "{{HUB_UI_ASSET_VERSION}}", build_hub_ui_asset_version()
    )
    return HTMLResponse(content=rendered, status_code=200, headers=_NO_CACHE)
