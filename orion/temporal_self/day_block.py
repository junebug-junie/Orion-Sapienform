"""``render_closed_day_block``: one closed day of Orion's own chronology, for Orion's Day letter. Pure.

A fixed template over the stored ``TemporalSelfDayV1`` (``temporal_self_day.day_json``). No model
writes it. Every line cites what it came from: ``[arc:<id>]`` for a line about one arc,
``[day:<day_id>]`` for a count over the whole day (arc totals, expectations, refusals,
instrument warnings). ``records`` carries a compact copy of each cited row so the letter's
citations resolve later without the full day record.

Closed days are frozen (Juniper, 2026-10-11): rows that arrive after the close (memory episodes
run ~13 h late, image attempts ~90 min) are not in the record, and the first line says the
record is as it stood at close.

``immune_findings`` (Temporal Self R4): the day's ``immune_finding`` rows from
``temporal_self_event`` (payloads written by the regulation step's immune sweep,
``orion.regulation.immune``), recorded before the close. Each renders as one plain sentence naming
the gauge, what was wrong with it and when; at most ``MAX_IMMUNE_LINES``, then "and N more".
None or empty renders nothing.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

from orion.schemas.temporal_self import (
    DAY_BLOCK_MAX_CHARS,
    TemporalSelfArcV1,
    TemporalSelfDayBlockV1,
    TemporalSelfDayV1,
)
from orion.temporal_self.day import DEFAULT_TZ
from orion.temporal_self.stance_cue import span_words, strain_cause_text

TOP_ATTENTION = 5
MAX_CURIOSITY_LINES = 10
MAX_IMMUNE_LINES = 5
IMMUNE_KIND_WORDS = {"constant": "constant", "floor_locked": "floor-locked", "replayed": "replayed",
                     "decayed_to_zero": "decayed to zero"}
KIND_ORDER = ("attention", "conversation", "curiosity", "sleep", "imagery", "concern", "reverie", "interoception",
              "company", "strain", "play")


def _clock(at: datetime | None, day_id: str, tz: ZoneInfo) -> str:
    if at is None:
        return "?"
    local = at.astimezone(tz)
    if local.date().isoformat() == day_id:
        return local.strftime("%H:%M")
    return local.strftime("%m-%d %H:%M")


def _arc_record(arc: TemporalSelfArcV1) -> dict[str, Any]:
    return {
        "arc_id": arc.arc_id, "kind": arc.kind, "subject_ref": arc.subject_ref,
        "subject_label": arc.subject_label, "status": arc.status, "closed_reason": arc.closed_reason,
        "began_at": arc.began_at.isoformat(), "ended_at": arc.ended_at.isoformat() if arc.ended_at else None,
        "last_seen_at": arc.last_seen_at.isoformat(), "attention_returns": arc.attention_returns,
        "cumulative_dwell_sec": round(arc.cumulative_dwell_sec, 3), "warnings": list(arc.warnings),
    }


def _gauge_name(node: str, channel: str) -> str:
    """``node:substrate.route`` + ``prediction_error`` -> ``route prediction error``."""
    short = node.rsplit(":", 1)[-1]
    short = short[len("substrate."):] if short.startswith("substrate.") else short
    return f"{short} {channel.replace('_', ' ')}".strip()


def _immune_rows(raw: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    """Valid finding payloads, oldest window first. A malformed row is dropped, never invented."""
    out = []
    for r in raw or ():
        if not isinstance(r, Mapping):
            continue
        try:
            start = datetime.fromisoformat(str(r["window_start"]))
            end = datetime.fromisoformat(str(r["window_end"]))
            kind, node, channel = str(r["kind"]), str(r["node"]), str(r["channel"])
        except (KeyError, TypeError, ValueError):
            continue
        if start.tzinfo is None or end.tzinfo is None or kind not in IMMUNE_KIND_WORDS:
            continue
        out.append({"metric_id": str(r.get("metric_id") or f"{node}.{channel}"), "node": node, "channel": channel,
                    "kind": kind, "start": start, "end": end})
    return sorted(out, key=lambda x: (x["start"], x["metric_id"], x["kind"]))


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def render_closed_day_block(
    day: TemporalSelfDayV1 | None,
    *,
    tz_name: str = DEFAULT_TZ,
    max_chars: int = DAY_BLOCK_MAX_CHARS,
    immune_findings: Sequence[Mapping[str, Any]] | None = None,
) -> TemporalSelfDayBlockV1 | None:
    """The block, or None when there is no closed day (missing renders nothing, never a placeholder)."""
    if day is None:
        return None
    tz = ZoneInfo(tz_name)
    day_ref = f"[day:{day.day_id}]"
    frame = day.frame
    arcs = sorted(day.arcs, key=lambda a: (a.began_at, a.arc_id))
    by_id = {a.arc_id: a for a in arcs}
    cited: dict[str, TemporalSelfArcV1] = {}
    labels: dict[str, list[str]] = defaultdict(list)

    def ref(arc: TemporalSelfArcV1, *label_texts: str) -> str:
        cited[arc.arc_id] = arc
        for text in label_texts:
            if text and arc.arc_id not in labels[text]:
                labels[text].append(arc.arc_id)
        return f"[arc:{arc.arc_id}]"

    def clock(arc: TemporalSelfArcV1, at: datetime | None) -> str:
        return _clock(at, day.day_id, tz)

    # (priority, line). Lower priority number survives the cap longer; lines keep their order.
    lines: list[tuple[int, str]] = []
    closed_local = day.closed_at.astimezone(tz).strftime("%Y-%m-%d %H:%M")
    kinds = Counter(a.kind for a in arcs)
    kind_text = ", ".join(f"{kinds[k]} {k}" for k in KIND_ORDER if kinds.get(k)) or "none"
    lines.append((0, f"Closed day {day.day_id} ({tz_name}, midnight to midnight), as recorded at close, "
                     f"{closed_local} local: rows that arrived later are not in it. "
                     f"{_plural(len(arcs), 'arc')}: {kind_text} {day_ref}."))
    if not arcs:
        lines.append((0, f"No arcs were recorded on this day {day_ref}."))

    talks = [a for a in arcs if a.kind == "conversation"]
    if talks:
        parts = []
        for a in talks:
            start, last = clock(a, a.began_at), clock(a, a.last_seen_at)
            span = start if start == last else f"{start} to {last}"
            parts.append(f"{span} {ref(a, start)}")
        lines.append((1, "Talked with Juniper: " + "; ".join(parts) + "."))

    groups: dict[str, list[TemporalSelfArcV1]] = defaultdict(list)
    for a in arcs:
        if a.kind == "attention":
            groups[a.subject_ref].append(a)
    ranked = sorted(groups.values(), key=lambda g: (-sum(a.cumulative_dwell_sec for a in g), g[0].subject_ref))
    # subject_ref -> (its rendered line, the totals its "[day:]" clause states); kept in the day
    # record only when that exact line survived the cap.
    group_totals: dict[str, tuple[str, dict[str, Any]]] = {}
    if ranked:
        lines.append((1, "Attention, most time first:"))
        for i, group in enumerate(ranked[:TOP_ATTENTION]):
            longest = max(group, key=lambda a: (a.cumulative_dwell_sec, a.arc_id))
            label = longest.subject_label.strip() or longest.subject_ref
            # Each number sits next to the record that holds it: the group's total over the day
            # cites the day, the cited arc's own dwell and returns cite the arc (truth eval, #2623).
            own = (f"{span_words(longest.cumulative_dwell_sec)}, {_plural(longest.attention_returns, 'return')} "
                   f"{ref(longest, longest.subject_label.strip())}")
            if len(group) > 1:
                total = span_words(sum(a.cumulative_dwell_sec for a in group))
                own = f"{total} over {_plural(len(group), 'arc')} {day_ref}; longest {own}"
            line = f"- {label}: {own}."
            if len(group) > 1:
                group_totals[longest.subject_ref] = (line, {
                    "arcs": len(group), "cumulative_dwell_sec": round(sum(a.cumulative_dwell_sec for a in group), 3)})
            lines.append((2 + i, line))

    runs = [a for a in arcs if a.kind == "curiosity"]
    if runs:
        lines.append((2, f"Curiosity runs ({len(runs)}):"))
        for i, a in enumerate(runs[:MAX_CURIOSITY_LINES]):
            took = span_words(((a.ended_at or a.last_seen_at) - a.began_at).total_seconds())
            doubt = f" ({'; '.join(a.warnings)})" if a.warnings else ""
            start = clock(a, a.began_at)
            lines.append((4 + i, f"- {start}, {took}{doubt} {ref(a, start)}."))
        if len(runs) > MAX_CURIOSITY_LINES:
            lines.append((4, f"- and {len(runs) - MAX_CURIOSITY_LINES} more {day_ref}."))

    strains = [a for a in arcs if a.kind == "strain"]
    if strains:
        parts = []
        for a in strains:
            start = clock(a, a.episode_began_at or a.began_at)
            causes = strain_cause_text(a.subject_label)
            end = "still strained at midnight" if a.closed_reason == "day_boundary" else f"to {clock(a, a.ended_at)}"
            parts.append(f"{start} {end}{f' ({causes})' if causes else ''} {ref(a, start)}")
        lines.append((2, "Strained: " + "; ".join(parts) + "."))

    sleeps = [a for a in arcs if a.kind == "sleep"]
    if sleeps:
        lines.append((2, "Sleep cycles: " + "; ".join(f"{clock(a, a.began_at)} {ref(a, clock(a, a.began_at))}" for a in sleeps) + "."))

    images = [a for a in arcs if a.kind == "imagery"]
    if images:
        lines.append((5, f"Image attempts: {len(images)}, first {clock(images[0], images[0].began_at)} "
                         f"{ref(images[0])}, last {clock(images[-1], images[-1].began_at)} {ref(images[-1])}."))

    if frame.expectations_resolved_total or frame.expectations_pending:
        verdicts = Counter(x.verdict or "no verdict" for x in frame.expectations_resolved_today)
        listed = len(frame.expectations_resolved_today)
        detail = ", ".join(f"{n} {v}" for v, n in sorted(verdicts.items(), key=lambda kv: (-kv[1], kv[0])))
        of = f"of the {listed} listed: " if listed != frame.expectations_resolved_total else ""
        lines.append((3, f"Expectations resolved: {frame.expectations_resolved_total} ({of}{detail or 'none listed'}). "
                         f"{len(frame.expectations_pending)} still pending {day_ref}."))

    if frame.constraint_event_ids or frame.constraint_overflow:
        refusals = Counter(e.split(":", 1)[0] for e in frame.constraint_event_ids)
        text = ", ".join(f"{n} {k}" for k, n in sorted(refusals.items(), key=lambda kv: (-kv[1], kv[0])))
        more = f", plus {frame.constraint_overflow} over the list cap" if frame.constraint_overflow else ""
        lines.append((3, f"Refused or made to wait: {text}{more} {day_ref}."))

    threads = [a for a in arcs if a.kind == "concern"]
    # A thread open at close whose arc row is not in the day (a frame/arc mismatch) still renders,
    # cited to the day record and named as missing, never dropped silently.
    orphans = [t for t in frame.open_threads if t.arc_id and t.arc_id not in by_id]
    if threads or orphans:
        parts = [f"{a.subject_label.strip() or a.subject_ref}, {_plural(a.attention_returns, 'return')}, "
                 f"{a.closed_reason or a.status} {ref(a, a.subject_label.strip())}" for a in threads]
        parts += [f"{t.subject_label.strip() or t.subject_ref}, open at close, arc {t.arc_id} not in the day record "
                  f"{day_ref}" for t in orphans]
        lines.append((3, "Conversation threads: " + "; ".join(parts) + "."))

    day_warnings = []
    for w in frame.warnings:
        # Per-arc warnings name their arc ("curiosity arc <id> (...)"): cite it, not the day.
        head = w.split(" ", 3)
        arc = by_id.get(head[2]) if len(head) > 2 and head[1] == "arc" else None
        day_warnings.append(f"{w} {ref(arc)}" if arc is not None else f"{w} {day_ref}")
    if day_warnings:
        lines.append((1, "Instrument warnings: " + " | ".join(day_warnings) + "."))

    # Priority 3 (with expectations and threads): a run of stuck gauges must not evict what Orion
    # attended to and did, which sits at 1-2.
    immune = _immune_rows(immune_findings)
    immune_lines: list[tuple[str, dict[str, Any]]] = []
    for f in immune[:MAX_IMMUNE_LINES]:
        immune_lines.append((f"My {_gauge_name(f['node'], f['channel'])} gauge looked stuck "
                             f"({IMMUNE_KIND_WORDS[f['kind']]}) from {_clock(f['start'], day.day_id, tz)} "
                             f"to {_clock(f['end'], day.day_id, tz)} {day_ref}.", f))
    lines.extend((3, text) for text, _ in immune_lines)
    if len(immune) > MAX_IMMUNE_LINES:
        lines.append((3, f"And {len(immune) - MAX_IMMUNE_LINES} more stuck gauges {day_ref}."))

    kept = list(lines)
    dropped = 0
    while len("\n".join(t for _, t in kept)) > max_chars and len(kept) > 1:
        worst = max(range(len(kept)), key=lambda i: (kept[i][0], i))
        kept.pop(worst)
        dropped += 1
    if dropped:
        note = f"({dropped} lower-priority line{'s' if dropped != 1 else ''} left out to fit) {day_ref}"
        while len("\n".join([t for _, t in kept] + [note])) > max_chars and len(kept) > 1:
            kept.pop(max(range(len(kept)), key=lambda i: (kept[i][0], i)))
            dropped += 1
            note = f"({dropped} lower-priority lines left out to fit) {day_ref}"
        kept.append((0, note))
    text = "\n".join(t for _, t in kept)[:max_chars]

    shown = {a_id for a_id in cited if f"[arc:{a_id}]" in text}
    records: dict[str, dict[str, Any]] = {f"arc:{a_id}": _arc_record(cited[a_id]) for a_id in sorted(shown)}
    records[f"day:{day.day_id}"] = {
        "day_id": day.day_id, "closed_at": day.closed_at.isoformat(), "arcs_total": len(arcs),
        "arcs_by_kind": dict(sorted(kinds.items())),
        "expectations_resolved_total": frame.expectations_resolved_total,
        "expectations_pending": len(frame.expectations_pending),
        "constraint_event_ids": len(frame.constraint_event_ids) + frame.constraint_overflow,
        "warnings": list(frame.warnings),
        # Like attention_groups: only findings whose line survived the cap.
        "immune_findings": [{"metric_id": f["metric_id"], "kind": f["kind"], "window_start": f["start"].isoformat(),
                             "window_end": f["end"].isoformat()}
                            for line, f in immune_lines if line in text.splitlines()],
        "attention_groups": {k: v for k, (line, v) in sorted(group_totals.items()) if line in text.splitlines()},
    }
    return TemporalSelfDayBlockV1(
        day_id=day.day_id, closed_at=day.closed_at, text=text, records=records,
        labels={k: [a for a in v if a in shown] for k, v in sorted(labels.items()) if any(a in shown for a in v)},
        lines_dropped=dropped,
    )
