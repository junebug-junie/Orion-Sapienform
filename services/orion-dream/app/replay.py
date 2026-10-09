"""Sleep pressure + NREM replay selection. Deterministic (§4): no LLM here.

Candidates are what the day left unprocessed since the last cycle:

  metacog            Orion's own self-observations flagged degraded/critical
                     (surprise: it noticed something going wrong).
  compaction_request themes reverie asked sleep to consolidate (unresolved).
  resonance          themes reverie kept re-igniting faster than its refractory
                     bound allows (emotional charge / rumination).
  crystallization    active memory crystallizations touched since the last
                     cycle, weighted by their own salience.

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


def candidate_from_compaction_request(row: dict[str, Any]) -> Optional[ReplayItemV1]:
    rid = str(row.get("request_id") or "").strip()
    theme = _clip(row.get("theme"), 200)
    if not rid or not theme:
        return None
    reason_text = _clip(row.get("reason"), 400)
    text = f"{theme}: {reason_text}" if reason_text else theme
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
        text=f"a theme reverie kept returning to: {theme}",
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


def select_replay(candidates: Iterable[ReplayItemV1], k: int) -> list[ReplayItemV1]:
    """Top-k by weight, at most ceil(k/2) from any one source.

    The per-source cap keeps one noisy producer (e.g. a burst of degraded
    metacog during an outage) from turning the whole night into one topic --
    recombination needs material from different places to pair.
    """
    k = max(0, min(int(k), MAX_REPLAY_ITEMS))
    per_source = max(1, (k + 1) // 2)
    ranked = sorted(candidates, key=lambda c: (-c.weight, c.ref_id))
    picked: list[ReplayItemV1] = []
    used: dict[str, int] = {}
    for c in ranked:
        if len(picked) >= k:
            break
        if used.get(c.source_kind, 0) >= per_source:
            continue
        used[c.source_kind] = used.get(c.source_kind, 0) + 1
        picked.append(c)
    return picked
