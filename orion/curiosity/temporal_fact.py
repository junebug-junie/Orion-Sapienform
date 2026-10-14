"""How much of today curiosity has already had, as one fact line. Pure; never raises.

Temporal Self rev 4 (#2369), "curiosity gets its day": the kickoff is told how many
curiosity runs finished today and how long they took, from orion-durable-runs' chronicle.
Stated as fact, no steering -- the line never says what to do about it.

MATCH KEY. A curiosity arc's ``subject_ref`` is its own ``run_id`` (one arc per completed
run, ``orion/temporal_self/sources.py:curiosity_run``), so no arc names a topic or thread.
The only exact link between the kickoff and today's arcs is the earlier runs it shows
Orion (``WorldviewSnapshot.recent_runs[*].run_id``, the same id the Hub stamps into
``curiosity_run_outcomes`` and the graph's TurnOutcome). The line says how many of THOSE
ran today and for how long, plus the whole day's curiosity total. It never claims time
spent "on a subject": a run's offered priors are not what it worked on.

COMPLETENESS. ``frame.arcs_today`` is capped (``ARCS_IN_FRAME_CAP``) and drops the oldest
non-reverie arcs first, so on a busy day it silently omits morning curiosity runs (live
2026-10-13: 224 arcs, 64 listed, the day's curiosity run among the dropped). The Hub
therefore passes the uncapped ``/temporal-self/arcs?kind=curiosity`` list. Without it the
frame's own list is used only when it is provably complete; otherwise nothing is said.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable, Mapping, Optional, Sequence

from zoneinfo import ZoneInfo

from orion.temporal_self.day import DEFAULT_TZ, as_utc, day_id_for

FRAME_MAX_AGE_SEC = 900.0
# A frame stamped in the future is clock skew or garbage; allow only a little.
FRAME_MAX_FUTURE_SEC = 60.0
CITATION_CAP = 5


def _minutes(seconds: float) -> str:
    if seconds < 60:
        return "under a minute"
    m = int(round(seconds / 60.0))
    return f"{m} min"


def _dwell(arc: Mapping[str, Any]) -> float:
    try:
        value = float(arc.get("cumulative_dwell_sec") or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return value if value > 0 and value == value and value != float("inf") else 0.0


def _curiosity_arcs(rows: Any, day_id: str) -> Optional[list[Mapping[str, Any]]]:
    if not isinstance(rows, list):
        return None
    out = []
    for arc in rows:
        if not isinstance(arc, Mapping) or arc.get("kind") != "curiosity":
            continue
        if not arc.get("arc_id") or not arc.get("subject_ref"):
            continue
        if arc.get("day_id") != day_id:
            continue
        out.append(arc)
    return out


def _frame_list_is_complete(frame: Mapping[str, Any]) -> bool:
    listed = frame.get("arcs_today")
    if not isinstance(listed, list):
        return False
    total = frame.get("arcs_today_total")
    if isinstance(total, int) and total <= len(listed):
        return True
    # Reverie arcs fill the list only after every non-reverie arc is in it.
    return any(isinstance(a, Mapping) and a.get("kind") == "reverie" for a in listed)


def curiosity_today_fact(
    frame: Any,
    *,
    now: datetime,
    thread_run_ids: Iterable[str] = (),
    curiosity_arcs: Optional[Mapping[str, Any]] = None,
    tz_name: Optional[str] = None,
) -> Optional[str]:
    """The fact line, or None when there is nothing honest to say.

    ``frame``: the ``GET /temporal-self/frame`` dict (freshness and day).
    ``curiosity_arcs``: the ``GET /temporal-self/arcs?kind=curiosity`` dict
    (``{"day_id", "count", "arcs"}``), uncapped; optional.
    ``thread_run_ids``: run ids of the earlier runs the kickoff lists.
    None when: the frame is missing/garbage, older than 15 min (or more than 1 min in the
    future), or for another day than ``now``; or the arc list is not provably complete.
    An arc's day is its run's START day (``sources.curiosity_run``), hence "started today".
    The chronicle lags its clock by its read lag (~5 min), so the line names the record's
    time and never states that a listed run is absent from today.
    """
    try:
        return _fact(frame, now=now, thread_run_ids=thread_run_ids,
                     curiosity_arcs=curiosity_arcs, tz_name=tz_name)
    except Exception:  # noqa: BLE001 -- a fact line is never worth a failed kickoff
        return None


def _fact(
    frame: Any,
    *,
    now: datetime,
    thread_run_ids: Iterable[str],
    curiosity_arcs: Optional[Mapping[str, Any]],
    tz_name: Optional[str],
) -> Optional[str]:
    if not isinstance(frame, Mapping):
        return None
    day_id = frame.get("day_id")
    as_of = as_utc(frame.get("as_of"))
    now_utc = as_utc(now)
    if not isinstance(day_id, str) or not day_id or as_of is None or now_utc is None:
        return None
    age = (now_utc - as_of).total_seconds()
    if age > FRAME_MAX_AGE_SEC or age < -FRAME_MAX_FUTURE_SEC:
        return None
    today = day_id_for(now_utc, tz_name) if tz_name else day_id_for(now_utc)
    if day_id != today:
        return None

    arcs: Optional[list[Mapping[str, Any]]] = None
    if isinstance(curiosity_arcs, Mapping) and curiosity_arcs.get("day_id") == day_id:
        arcs = _curiosity_arcs(curiosity_arcs.get("arcs"), day_id)
    if arcs is None:
        if not _frame_list_is_complete(frame):
            return None
        arcs = _curiosity_arcs(frame.get("arcs_today"), day_id)
    if arcs is None:
        return None

    # One arc per run; a repeated row (re-read) must not double the minutes.
    by_id: dict[str, Mapping[str, Any]] = {}
    for arc in arcs:
        by_id.setdefault(str(arc["arc_id"]), arc)
    ordered = sorted(by_id.values(), key=lambda a: (str(a.get("began_at") or ""), str(a["arc_id"])))

    # The chronicle folds rows a few minutes behind its own clock, so the newest run may
    # not be in it yet: say when the record is from, and never claim an absence.
    zone = ZoneInfo(tz_name or DEFAULT_TZ)
    head = f"TODAY SO FAR (day record as of {as_of.astimezone(zone).strftime('%H:%M')} local):"
    if not ordered:
        return f"{head} no curiosity run started today has finished."

    total_sec = sum(_dwell(a) for a in ordered)
    wanted = {str(r) for r in thread_run_ids if r}
    thread = [a for a in ordered if str(a["subject_ref"]) in wanted]
    thread_sec = sum(_dwell(a) for a in thread)

    # Thread arcs first (most recent first), then the rest most recent first.
    cited: list[str] = []
    for arc in list(reversed(thread)) + [a for a in reversed(ordered) if a not in thread]:
        if len(cited) >= CITATION_CAP:
            break
        cited.append(str(arc["arc_id"]))
    refs = " ".join(f"[arc:{i}]" for i in cited)

    n = len(ordered)
    line = (f"{head} {n} curiosity {'run' if n == 1 else 'runs'} started today and finished, "
            f"{_minutes(total_sec)} in total.")
    k = len(thread)
    if k:
        line += (f" {k} of the earlier runs listed above {'is' if k == 1 else 'are'} among them, "
                 f"{_minutes(thread_sec)} of that.")
    return f"{line} {refs}"


def thread_run_ids_from_view(view: Any) -> Sequence[str]:
    """Run ids of the earlier runs the kickoff's thread section lists. Empty when that
    section is not rendered (fewer than 2 runs, ``kickoff_prompt._thread_section``), so the
    line never points at a list Orion cannot see."""
    try:
        runs = list(getattr(view, "recent_runs", None) or [])
        if len(runs) < 2:
            return []
        return [str(r.run_id) for r in runs if getattr(r, "run_id", "")]
    except Exception:  # noqa: BLE001
        return []


__all__ = ["CITATION_CAP", "FRAME_MAX_AGE_SEC", "curiosity_today_fact", "thread_run_ids_from_view"]
