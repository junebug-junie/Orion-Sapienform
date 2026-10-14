"""Sleep pressure + NREM replay selection. Deterministic (§4): no LLM here.

Candidates are what the day left unprocessed since the last cycle:

  metacog            Orion's own self-observations flagged degraded/critical
                     (surprise: it noticed something going wrong).
  compaction_request themes reverie asked sleep to consolidate (unresolved).
  resonance          themes reverie kept re-igniting faster than its refractory
                     bound allows (emotional charge / rumination).
  crystallization    active memory crystallizations touched since the last
                     cycle, weighted by their own salience.
  arc                closed Temporal Self arcs since the last sleep (bottom of
                     this file). Replay material only, never sleep pressure.

Weights are declared heuristics, stated here once.

Each candidate has a KEY naming the thing it is about, not the row it came
from: metacog's normalized `trigger_reason` (its `summary` is model prose,
reworded every time for the same event), the reverie theme, or the
crystallization id. Rows sharing a key are one candidate at their highest
weight, so 222 copies of one gateway timeout are one item to replay.

Pressure counts only NEW keys: keys present since the last sleep and absent
from the lookback before it (two-process model, Process S, read through the
synaptic homeostasis hypothesis: new encoding builds sleep need, re-meeting a
known thing does not). Right after a sleep the window is empty and pressure
reads exactly 0.0. A chronic problem adds pressure once, on first appearance;
it stays a replay candidate every window it recurs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Optional

from orion.schemas.dream_cycle import MAX_REPLAY_ITEMS, ReplayItemV1

SEVERITY_WEIGHT = {"critical": 1.0, "degraded": 0.6}
COMPACTION_REQUEST_WEIGHT = 0.5
RESONANCE_BASE = 0.3
RESONANCE_PER_VIOLATION = 0.1
TEXT_CAP = 600


def _clip(text: Any, n: int = TEXT_CAP) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _clamp01(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, v))


def _tags(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(t).strip().lower() for t in value if str(t).strip()][:32]
    return []


def row_key(kind: str, row: dict[str, Any]) -> Optional[str]:
    """`kind:<what this is about>`. SQL supplies `dedupe_key` (cycle_store);
    without one, fall back to the row's own identity so it counts as new."""
    key = str(row.get("dedupe_key") or "").strip().lower()
    if not key:
        fallback = {
            "metacog": row.get("id"),
            "compaction_request": row.get("theme"),
            "resonance": row.get("theme_key"),
            "crystallization": row.get("crystallization_id"),
        }.get(kind)
        key = " ".join(str(fallback or "").split()).lower()
    return f"{kind}:{key}" if key else None


def candidate_from_metacog(row: dict[str, Any]) -> Optional[ReplayItemV1]:
    severity = str(row.get("severity") or "").strip().lower()
    weight = SEVERITY_WEIGHT.get(severity)
    text = _clip(row.get("summary"))
    rid = str(row.get("id") or "").strip()
    if weight is None or not text or not rid:
        return None
    kind = str(row.get("trigger_kind") or "unknown").strip()
    return ReplayItemV1(
        ref_id=f"metacog:{rid}",
        source_kind="metacog",
        text=text,
        weight=weight,
        reason=f"metacog flagged {severity} ({kind})",
        tags=_tags(row.get("tags")),
    )


def _loop_name(row: dict[str, Any], theme: str) -> str:
    """What the dream reads for a reverie theme: the open loop's human title when the
    source query found one (cycle_store _title_join), else the bare id. Text only:
    ref_id, tags and the novelty key keep the id."""
    return _clip(row.get("title"), 200) or theme


def candidate_from_compaction_request(row: dict[str, Any]) -> Optional[ReplayItemV1]:
    rid = str(row.get("request_id") or "").strip()
    theme = _clip(row.get("theme"), 200)
    if not rid or not theme:
        return None
    reason_text = _clip(row.get("reason"), 400)
    name = _loop_name(row, theme)
    text = f"{name}: {reason_text}" if reason_text else name
    return ReplayItemV1(
        ref_id=f"compaction_request:{rid}",
        source_kind="compaction_request",
        text=text,
        weight=COMPACTION_REQUEST_WEIGHT,
        reason=f"reverie asked sleep to consolidate {theme!r}",
        tags=[theme.lower()],
    )


def candidate_from_resonance(row: dict[str, Any]) -> Optional[ReplayItemV1]:
    aid = str(row.get("alert_id") or "").strip()
    theme = _clip(row.get("theme_key"), 200)
    if not aid or not theme:
        return None
    violations = int(row.get("violation_count") or 0)
    weight = min(1.0, RESONANCE_BASE + RESONANCE_PER_VIOLATION * max(0, violations))
    return ReplayItemV1(
        ref_id=f"resonance:{aid}",
        source_kind="resonance",
        text=f"a theme reverie kept returning to: {_loop_name(row, theme)}",
        weight=weight,
        reason=f"reverie re-ignited {theme!r} past its refractory bound ({violations} violations)",
        tags=[theme.lower()],
    )


def candidate_from_crystallization(row: dict[str, Any]) -> Optional[ReplayItemV1]:
    cid = str(row.get("crystallization_id") or "").strip()
    subject = _clip(row.get("subject"), 160)
    summary = _clip(row.get("summary"), 440)
    if not cid or not (subject or summary):
        return None
    salience = _clamp01(row.get("salience"))
    if salience <= 0.0:
        return None
    return ReplayItemV1(
        ref_id=f"crystallization:{cid}",
        source_kind="crystallization",
        text=f"{subject}: {summary}" if subject and summary else (subject or summary),
        weight=salience,
        reason=f"active crystallization touched since last sleep (salience {salience:.2f})",
        tags=_tags(row.get("tags")),
    )


_BUILDERS = {
    "metacog": candidate_from_metacog,
    "compaction_request": candidate_from_compaction_request,
    "resonance": candidate_from_resonance,
    "crystallization": candidate_from_crystallization,
}


def keyed_candidates(raw: dict[str, Iterable[dict[str, Any]]]) -> dict[str, ReplayItemV1]:
    """raw: source_kind -> rows, newest first. key -> one candidate per thing:
    the first row's text, the highest weight any row carried. Unusable rows
    are dropped, never invented."""
    out: dict[str, ReplayItemV1] = {}
    for kind, rows in raw.items():
        builder = _BUILDERS.get(kind)
        if builder is None:
            continue
        for row in rows:
            item = builder(row)
            key = row_key(kind, row)
            if item is None or key is None:
                continue
            kept = out.get(key)
            if kept is None:
                out[key] = item
            elif item.weight > kept.weight:
                out[key] = kept.model_copy(update={"weight": item.weight})
    return out


def build_candidates(raw: dict[str, Iterable[dict[str, Any]]]) -> list[ReplayItemV1]:
    return list(keyed_candidates(raw).values())


def prior_keys(raw: dict[str, Iterable[dict[str, Any]]]) -> set[str]:
    """Keys seen in the lookback before the window (same rows, same key rule)."""
    out: set[str] = set()
    for kind, rows in raw.items():
        for row in rows:
            key = row_key(kind, row)
            if key is not None:
                out.add(key)
    return out


def compute_pressure(
    keyed: Mapping[str, ReplayItemV1], seen_before: Iterable[str] = ()
) -> tuple[float, dict[str, int], dict[str, int]]:
    """(pressure, counts, new_counts). counts: distinct things per source in the
    window. pressure and new_counts: only keys absent from `seen_before`."""
    before = set(seen_before)
    total = 0.0
    counts: dict[str, int] = {}
    new_counts: dict[str, int] = {}
    for key, c in keyed.items():
        counts[c.source_kind] = counts.get(c.source_kind, 0) + 1
        if key in before:
            continue
        total += c.weight
        new_counts[c.source_kind] = new_counts.get(c.source_kind, 0) + 1
    return round(total, 6), counts, new_counts


def select_replay(
    candidates: Iterable[ReplayItemV1], k: int, source_caps: Optional[Mapping[str, int]] = None
) -> list[ReplayItemV1]:
    """Top-k by weight, at most ceil(k/2) from any one source.

    The per-source cap keeps one noisy producer (e.g. a burst of degraded
    metacog during an outage) from turning the whole night into one topic --
    recombination needs material from different places to pair.
    `source_caps` tightens (never loosens) that cap for named sources.
    """
    k = max(0, min(int(k), MAX_REPLAY_ITEMS))
    per_source = max(1, (k + 1) // 2)
    caps = {src: min(per_source, max(0, int(n))) for src, n in (source_caps or {}).items()}
    ranked = sorted(candidates, key=lambda c: (-c.weight, c.ref_id))
    picked: list[ReplayItemV1] = []
    used: dict[str, int] = {}
    for c in ranked:
        if len(picked) >= k:
            break
        if used.get(c.source_kind, 0) >= caps.get(c.source_kind, per_source):
            continue
        used[c.source_kind] = used.get(c.source_kind, 0) + 1
        picked.append(c)
    return picked


# --- closed Temporal Self arcs (rev 4 patch 7) --------------------------------
#
# A closed arc is one stretch of the chronology Orion lived through (a concern it
# held, a curiosity it chased, a body state). Replay material ONLY: arcs are added
# after the pressure read (app/cycle.py), so they never touch sleep pressure,
# new_counts, has_candidates or the rest drive.
#
# Weight (declared heuristic, the arc's own two measures of how much it held Orion):
#   dwell   = min(1, ln(1 + dwell_sec/60) / ln(1 + 60))   log: live dwell spans 5 s
#             (reverie p50) to hours (curiosity p90 ~66 min); saturates at 60 min.
#   returns = min(1, attention_returns / 3)                coming back to a thing after
#             it was suspended; three returns is "kept returning".
#   weight  = ARC_WEIGHT_MAX * (dwell + returns) / 2
# ARC_WEIGHT_MAX < 1.0 so a critical metacog (1.0) always outranks an arc. Weight 0
# (no dwell, no return) carries nothing and is dropped, like a 0-salience crystallization.

ARC_WEIGHT_MAX = 0.8
ARC_DWELL_SATURATE_MIN = 60.0
ARC_RETURNS_SATURATE = 3
ARC_CANDIDATES_MAX = 20
ARC_EXCLUDED_KINDS = frozenset({"sleep"})  # a sleep never replays a sleep


def arc_replay_slots(k: int) -> int:
    """At most a third of the replay is arcs. The other sources are things the day
    explicitly flagged (a critical metacog, a theme reverie ASKED sleep to consolidate);
    arcs are the chronology around them. Live 10-12: under the plain ceil(k/2) per-source
    cap, six arcs at >= 0.5 would have displaced all six compaction requests (0.5)."""
    return max(0, int(k)) // 3


def arc_weight(dwell_sec: Any, returns: Any) -> float:
    import math

    try:
        dwell = max(0.0, float(dwell_sec or 0.0))
    except (TypeError, ValueError):
        dwell = 0.0
    try:
        ret = max(0, int(returns or 0))
    except (TypeError, ValueError):
        ret = 0
    d = min(1.0, math.log1p(dwell / 60.0) / math.log1p(ARC_DWELL_SATURATE_MIN))
    r = min(1.0, ret / ARC_RETURNS_SATURATE)
    return round(ARC_WEIGHT_MAX * (d + r) / 2.0, 4)


def _hhmm(value: Any) -> str:
    try:
        return value.astimezone(timezone.utc).strftime("%m-%d %H:%M")
    except Exception:  # noqa: BLE001 -- text only
        return "?"


def candidate_from_arc(row: dict[str, Any]) -> Optional[ReplayItemV1]:
    aid = str(row.get("arc_id") or "").strip()
    kind = str(row.get("kind") or "").strip().lower()
    if not aid or not kind or kind in ARC_EXCLUDED_KINDS:
        return None
    if str(row.get("status") or "") != "closed" or row.get("ended_at") is None:
        return None
    dwell = row.get("cumulative_dwell_sec")
    returns = row.get("attention_returns")
    weight = arc_weight(dwell, returns)
    if weight <= 0.0:
        return None
    subject = _clip(row.get("subject_label"), 200) or _clip(row.get("subject_ref"), 200)
    try:
        dwell_min = float(dwell or 0.0) / 60.0
    except (TypeError, ValueError):
        dwell_min = 0.0
    ret = int(returns or 0)
    return ReplayItemV1(
        ref_id=f"arc:{aid}",
        source_kind="arc",
        text=_clip(
            f"{kind} arc: {subject or '(no subject)'}, {_hhmm(row.get('began_at'))} to "
            f"{_hhmm(row.get('ended_at'))} UTC, held {dwell_min:.0f} min, returned to {ret} times"
        ),
        weight=weight,
        reason=f"closed {kind} arc since last sleep (dwell {dwell_min:.1f} min, {ret} returns)",
        tags=[kind],
    )


def arc_candidates(
    rows: Iterable[dict[str, Any]], since: datetime, until: datetime, *, cap: int = ARC_CANDIDATES_MAX
) -> list[ReplayItemV1]:
    """Closed, non-sleep arcs that ENDED in [since, until), top `cap` by weight (ties:
    newest end, then arc_id). The store filters the same way; this re-checks so a loose
    source can never widen the window.

    One candidate per THING, the same rule as every other source: arcs sharing
    (kind, subject_ref) are one candidate, the heaviest arc (its arc_id is the one cited).
    Live 10-12: node:substrate.bus_synaptic closed three interoception arcs in one window
    and took 3 of the top 5 slots before this."""
    best: dict[tuple[str, str], tuple[ReplayItemV1, Any]] = {}
    for row in rows or ():
        ended = row.get("ended_at")
        try:
            in_window = ended is not None and since <= ended < until
        except TypeError:
            in_window = False
        if not in_window:
            continue
        item = candidate_from_arc(row)
        if item is None:
            continue
        key = (item.tags[0], str(row.get("subject_ref") or item.ref_id).strip().lower())
        kept = best.get(key)
        if kept is None or (item.weight, ended.timestamp(), item.ref_id) > (kept[0].weight, kept[1].timestamp(), kept[0].ref_id):
            best[key] = (item, ended)
    ranked = sorted(best.values(), key=lambda t: (-t[0].weight, -t[1].timestamp(), t[0].ref_id))
    return [item for item, _ in ranked[: max(0, int(cap))]]
