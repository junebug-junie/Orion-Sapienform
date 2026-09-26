"""Sleep pressure + NREM replay selection. Deterministic (§4): no LLM here.

Candidates are what the day left unprocessed since the last cycle:

  metacog            Orion's own self-observations flagged degraded/critical
                     (surprise: it noticed something going wrong).
  compaction_request themes reverie asked sleep to consolidate (unresolved).
  resonance          themes reverie kept re-igniting faster than its refractory
                     bound allows (emotional charge / rumination).
  crystallization    active memory crystallizations touched since the last
                     cycle, weighted by their own salience.

Weights are declared heuristics, stated here once. Pressure is the SUM of
candidate weights -- the same numbers replay ranks on -- so the trigger and the
selection cannot drift apart. Every source is windowed on `since` (the last
cycle's end), which gives pressure a true rest point: right after a cycle it
reads exactly 0.0, and it only rises as new unprocessed rows arrive.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

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


def build_candidates(raw: dict[str, Iterable[dict[str, Any]]]) -> list[ReplayItemV1]:
    """raw: source_kind -> rows. Unusable rows are dropped, never invented."""
    out: list[ReplayItemV1] = []
    seen: set[str] = set()
    for kind, rows in raw.items():
        builder = _BUILDERS.get(kind)
        if builder is None:
            continue
        for row in rows:
            item = builder(row)
            if item is not None and item.ref_id not in seen:
                seen.add(item.ref_id)
                out.append(item)
    return out


def compute_pressure(candidates: Iterable[ReplayItemV1]) -> tuple[float, dict[str, int]]:
    total = 0.0
    counts: dict[str, int] = {}
    for c in candidates:
        total += c.weight
        counts[c.source_kind] = counts.get(c.source_kind, 0) + 1
    return round(total, 6), counts


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
