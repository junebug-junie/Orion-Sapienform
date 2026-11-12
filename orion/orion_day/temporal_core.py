"""``render_temporal_core``: what the closed day's temporal record adds beyond the chronology block.

Living Prime design #3224, constraint 6. Read-only over the stored ``TemporalSelfDayV1`` (the final
frame and the full arc list, ``temporal_self_day.day_json``) plus the day's own
``temporal_self_event`` rows for the frame's event ids. A fixed template; no model writes it.

Four parts, each line citing what it came from:

* **Threads with Juniper** -- conversation arcs and concern arcs (attention open loops; the arc
  carries its ``carried_from_previous_day`` flag, so a thread that crossed midnight says so).
  ``[arc:<id>]``.
* **What I expected and what happened** -- ``expectations_resolved_today`` /
  ``expectations_pending``, counted by kind and verdict. ``[day:<id>]`` for counts,
  ``[event:<event_id>]`` for the acts whose claim did not hold.
* **Acts and their graded outcomes** -- ``self_change_event_ids``. Live (2026-11-10) these are
  ``action_outcome`` rows (an act and whether its claim held) plus scored ``expectation_verdict``
  rows; the reducer files them as self-change (``SELF_CHANGE_KINDS`` in arcs.py). Grouped by act
  target, one cited event per group.
* **The body's day** -- strain, binding, sleep and company arcs with their spans. ``[arc:<id>]``.

Capped at ``DAY_BLOCK_MAX_CHARS`` like the chronology block, lowest-priority lines dropped first and
counted. ``records`` maps every ref in the text to a compact copy of the row it names, so a citation
resolves after the fact; ``labels`` feeds the citation gate (clock times and labels shown).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Mapping
from zoneinfo import ZoneInfo

from orion.schemas.temporal_self import DAY_BLOCK_MAX_CHARS, TemporalSelfArcV1, TemporalSelfDayBlockV1, TemporalSelfDayV1
from orion.temporal_self.day import DEFAULT_TZ
from orion.temporal_self.day_block import _arc_record
from orion.temporal_self.stance_cue import binding_words, span_words, strain_cause_text

MAX_ACT_GROUPS = 6
MAX_MISSED_ACTS = 4
MAX_COMPANY = 4


def _clock(at: datetime | None, day_id: str, tz: ZoneInfo) -> str:
    if at is None:
        return "?"
    local = at.astimezone(tz)
    return local.strftime("%H:%M") if local.date().isoformat() == day_id else local.strftime("%m-%d %H:%M")


def _upheld(verdict: str | None) -> bool | None:
    """``action_outcome``'s own verdict words (``claim_upheld=true|false``); anything else unknown."""
    if verdict == "claim_upheld=true":
        return True
    if verdict == "claim_upheld=false":
        return False
    return None


def _act_target(event: Mapping[str, Any] | None) -> str:
    payload = (event or {}).get("payload") or {}
    kind = str(payload.get("dispatch_kind") or "").strip()
    target = str(payload.get("target_id") or "").strip()
    return " ".join(x for x in (kind, target) if x) or "an act (its row was not read)"


def render_temporal_core(
    day: TemporalSelfDayV1 | None,
    events: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    tz_name: str = DEFAULT_TZ,
    max_chars: int = DAY_BLOCK_MAX_CHARS,
    already_shown: set[str] | frozenset[str] = frozenset(),
) -> TemporalSelfDayBlockV1 | None:
    """The temporal core, or None when there is no closed day or nothing to say.

    ``already_shown``: arc ids the closed-day block already cites; the body's day skips them, so
    the two blocks do not spend the writer's input saying the same strain twice.

    ``events`` maps ``event_id`` -> ``{"source_kind", "verdict", "payload", "occurred_at"}`` for the
    frame's expectation / self-change ids (a missing row renders as "an act (its row was not read)", never invented)."""
    if day is None:
        return None
    events = events or {}
    tz = ZoneInfo(tz_name)
    day_ref = f"[day:{day.day_id}]"
    frame = day.frame
    arcs = sorted(day.arcs, key=lambda a: (a.began_at, a.arc_id))
    cited_arcs: dict[str, TemporalSelfArcV1] = {}
    cited_events: set[str] = set()
    labels: dict[str, list[str]] = defaultdict(list)

    def ref(arc: TemporalSelfArcV1, *label_texts: str) -> str:
        cited_arcs[arc.arc_id] = arc
        for text in label_texts:
            if text and arc.arc_id not in labels[text]:
                labels[text].append(arc.arc_id)
        return f"[arc:{arc.arc_id}]"

    def eref(event_id: str) -> str:
        cited_events.add(event_id)
        return f"[event:{event_id}]"

    def clock(at: datetime | None) -> str:
        return _clock(at, day.day_id, tz)

    lines: list[tuple[int, str]] = []

    # --- threads with Juniper ---
    talks = [a for a in arcs if a.kind == "conversation"]
    concerns = [a for a in arcs if a.kind == "concern"]
    if talks or concerns:
        lines.append((0, "Threads with Juniper:"))
    for a in talks:
        start, last = clock(a.began_at), clock(a.last_seen_at)
        span = start if start == last else f"{start} to {last}"
        lines.append((1, f"- Talked {span}, {span_words(a.cumulative_dwell_sec)} in the conversation, "
                         f"{a.attention_returns} return{'s' if a.attention_returns != 1 else ''} {ref(a, start)}."))
    for a in concerns:
        name = a.subject_label.strip() or a.subject_ref
        carried = "carried over from the day before, " if a.carried_from_previous_day else ""
        state = "still open at midnight" if a.closed_reason == "day_boundary" else (a.closed_reason or a.status)
        lines.append((2, f"- Open loop {name}: {carried}{a.attention_returns} return"
                         f"{'s' if a.attention_returns != 1 else ''}, {state} {ref(a, a.subject_label.strip())}."))

    # --- expectations ---
    resolved = frame.expectations_resolved_today
    pending = frame.expectations_pending
    if frame.expectations_resolved_total or pending:
        by_kind: dict[str, Counter] = defaultdict(Counter)
        for x in resolved:
            by_kind[x.source_kind][x.verdict or "no verdict"] += 1
        parts = []
        for kind in sorted(by_kind):
            verdicts = ", ".join(f"{n} {v}" for v, n in sorted(by_kind[kind].items(), key=lambda kv: (-kv[1], kv[0])))
            parts.append(f"{kind}: {verdicts}")
        listed = "" if len(resolved) == frame.expectations_resolved_total else f" ({len(resolved)} listed)"
        pend = Counter(x.source_kind for x in pending)
        pend_text = ", ".join(f"{n} {k}" for k, n in sorted(pend.items())) or "none"
        lines.append((1, f"What I expected and what happened: {frame.expectations_resolved_total} resolved{listed}"
                         f" -- {'; '.join(parts) or 'none listed'}. Still pending at midnight: {pend_text} {day_ref}."))
        missed = [x for x in resolved if x.source_kind == "action_outcome" and _upheld(x.verdict) is False]
        if missed:
            shown = "; ".join(f"{_act_target(events.get(x.event_id))} at {clock(x.resolved_at)} {eref(x.event_id)}"
                              for x in missed[:MAX_MISSED_ACTS])
            more = f"; and {len(missed) - MAX_MISSED_ACTS} more {day_ref}" if len(missed) > MAX_MISSED_ACTS else ""
            lines.append((2, f"- Acts whose claim did not hold: {shown}{more}."))

    # --- acts and their graded outcomes (the frame's self-change ids) ---
    if frame.self_change_event_ids:
        groups: dict[str, list[str]] = defaultdict(list)
        for event_id in frame.self_change_event_ids:
            e = events.get(event_id)
            kind = (e or {}).get("source_kind") or event_id.split(":", 1)[0]
            groups[_act_target(e) if kind == "action_outcome" else kind].append(event_id)
        ranked = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        overflow = f", plus {frame.self_change_overflow} over the list cap" if frame.self_change_overflow else ""
        lines.append((2, f"Acts and graded outcomes ({len(frame.self_change_event_ids)}{overflow}), by target {day_ref}:"))
        for target, ids in ranked[:MAX_ACT_GROUPS]:
            held = Counter(_upheld((events.get(i) or {}).get("verdict")) for i in ids)
            tally = ", ".join(f"{n} {w}" for w, n in (("held", held[True]), ("did not hold", held[False]),
                                                      ("ungraded", held[None])) if n)
            lines.append((3, f"- {target}: {len(ids)} ({tally}), e.g. {eref(ids[0])}."))
        if len(ranked) > MAX_ACT_GROUPS:
            lines.append((3, f"- and {len(ranked) - MAX_ACT_GROUPS} more targets {day_ref}."))

    # --- the body's day ---
    body: list[tuple[int, str]] = []
    body_arcs = [a for a in arcs if a.arc_id not in already_shown]
    for a in (x for x in body_arcs if x.kind == "strain"):
        start = clock(a.episode_began_at or a.began_at)
        end = "still strained at midnight" if a.closed_reason == "day_boundary" else f"to {clock(a.ended_at)}"
        causes = strain_cause_text(a.subject_label)
        body.append((2, f"- Strained {start} {end}{f' ({causes})' if causes else ''} {ref(a, start)}."))
    for a in (x for x in body_arcs if x.kind == "binding" and x.binding is not None):
        start = clock(a.episode_began_at or a.began_at)
        end = ("still binding at midnight" if a.closed_reason == "day_boundary"
               else f"last read {clock(a.ended_at)}" if a.closed_reason == "source_stale" else f"to {clock(a.ended_at)}")
        body.append((3, f"- Limit binding: {binding_words(a.binding, short=True)}, {start} {end} {ref(a, start)}."))
    for a in (x for x in body_arcs if x.kind == "sleep"):
        start = clock(a.began_at)
        took = span_words(((a.ended_at or a.last_seen_at) - a.began_at).total_seconds())
        body.append((2, f"- Sleep cycle at {start}, lasting {took} {ref(a, start)}."))
    for a in [x for x in body_arcs if x.kind == "company"][:MAX_COMPANY]:
        start = clock(a.began_at)
        body.append((3, f"- Company: {a.subject_label.strip() or a.subject_ref} from {start} {ref(a, start)}."))
    if body:
        lines.append((1, "The body's day (beyond what the chronology above already lists):" if already_shown
                      else "The body's day:"))
        lines.extend(body)

    if not lines:
        return None
    kept = list(lines)
    dropped = 0
    while len("\n".join(t for _, t in kept)) > max_chars and len(kept) > 1:
        kept.pop(max(range(len(kept)), key=lambda i: (kept[i][0], i)))
        dropped += 1
    if dropped:
        note = f"({dropped} lower-priority line{'s' if dropped != 1 else ''} left out to fit) {day_ref}"
        while len("\n".join([t for _, t in kept] + [note])) > max_chars and len(kept) > 1:
            kept.pop(max(range(len(kept)), key=lambda i: (kept[i][0], i)))
            dropped += 1
            note = f"({dropped} lower-priority lines left out to fit) {day_ref}"
        kept.append((0, note))
    text = "\n".join(t for _, t in kept)[:max_chars]

    shown_arcs = {a_id for a_id in cited_arcs if f"[arc:{a_id}]" in text}
    records: dict[str, dict[str, Any]] = {f"arc:{a_id}": _arc_record(cited_arcs[a_id]) for a_id in sorted(shown_arcs)}
    for event_id in sorted(cited_events):
        if f"[event:{event_id}]" not in text:
            continue
        e = events.get(event_id) or {}
        payload = e.get("payload") or {}
        records[f"event:{event_id}"] = {
            "event_id": event_id, "source_kind": e.get("source_kind") or event_id.split(":", 1)[0],
            "verdict": e.get("verdict"), "occurred_at": e.get("occurred_at"),
            "dispatch_kind": payload.get("dispatch_kind"), "target_id": payload.get("target_id"),
            "recorded": bool(e),
        }
    records[f"day:{day.day_id}"] = {
        "day_id": day.day_id, "closed_at": day.closed_at.isoformat(),
        "expectations_resolved_total": frame.expectations_resolved_total,
        "expectations_pending": len(pending),
        "self_change_event_ids": len(frame.self_change_event_ids) + frame.self_change_overflow,
    }
    return TemporalSelfDayBlockV1(
        day_id=day.day_id, closed_at=day.closed_at, text=text, records=records,
        labels={k: [a for a in v if a in shown_arcs] for k, v in sorted(labels.items()) if any(a in shown_arcs for a in v)},
        lines_dropped=dropped,
    )
