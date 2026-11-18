"""`dig attention [<window>] [<end>]`: what Orion attended to, as relative rates, read-only.

Issue #3600. World sources (chat, reading, the street) fire rarely by nature, so a raw count of
winning frames measures how often a source happens, not how fairly attention treats it. World-first
attention (``orion/attention/world_first.py``) already judges each source against its OWN 7 days,
so every measure here is conditional on the source itself:

1. Won when eligible: frames where the target is rank 1 in ``dominant_targets``, out of frames
   where it is in ``dominant_targets`` at all; and the contested share (more than one eligible).
2. Eligible against its own activity:
   - event sources (an explicit ``EVENT_SOURCES`` entry): eligible frames out of
     events x the source's own news window / frame interval;
   - every other source: eligible share of frames, against the top-decile design rate
     (``1 - DEFAULT_BAND_CUTS[1]``: eligible means band high or unusual for itself).
3. Not attended: the world-first verdict cause, counted per target.
4. Rest: frames with no eligible target.
5. Drift: each rate against the same rate in the 7 days before the window, as a relative change.

It reads ``substrate_attention_hour_rollup`` (``orion/attention/frame_rollup.py``): orion-attention-runtime
adds each frame's counts there when it saves the frame, so a read is a few hundred small rows in whole
UTC hours, never the ~8 KB ``frame_json`` of 41k frames a day (that read took longer than its 25 s
timeout on a cold cache, 2026-11-18). Every frame counts; nothing is sampled.

Every printed line is a rate with its numerator and denominator, or a
ref (``tests/test_attention.py`` fails on a bare count).
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable

from orion.attention.frame_rollup import (  # noqa: F401 -- re-exported for the eval and tests
    CAUSES,
    OTHER_CAUSE,
    WORLD_FIRST_PREFIX,
    classify,
    classify_verdict,
    verdict_of,
)
from orion.attention.world_first import (
    CHAT_RATE_WINDOW,
    EVENT_ORIENTING_WINDOW_SEC,
    READING_RATE_WINDOW,
)
from orion.substrate.prediction_error_magnitude import DEFAULT_BAND_CUTS

DEFAULT_WINDOW = "24h"
MAX_WINDOW = timedelta(days=30)
BASELINE = timedelta(days=7)
STATEMENT_TIMEOUT = "25s"
# Eligible = band high or unusual for itself = at or above the usual/high cut of its own week.
DESIGN_ELIGIBLE_RATE = 1.0 - DEFAULT_BAND_CUTS[1]
WINDOW_RE = re.compile(r"(\d{1,4})([hd])")


@dataclass(frozen=True)
class EventSource:
    """A target whose eligibility is driven by discrete events Orion records in temporal_self_event."""

    kind: str  # temporal_self_event.source_kind
    window_sec: float  # how long one event keeps the source eligible, from world_first
    window_name: str  # the world_first constant, for the printout
    payload_key: str | None = None  # optional payload_json filter (key, value)
    payload_value: str | None = None
    note: str = ""


# Explicit map, target -> its own event kind. A target not here has no clean event source.
EVENT_SOURCES: dict[str, EventSource] = {
    # The chat candidate counts Juniper's turns in the trailing 15 min (world_first.CHAT_RATE_WINDOW,
    # _CHAT_TURN_WHERE): Orion's outreach rows do not light it, so they are not its activity.
    "world:chat": EventSource("chat_turn", CHAT_RATE_WINDOW.total_seconds(), "CHAT_RATE_WINDOW",
                              "speaker", "juniper", "Juniper's turns"),
    # Loose: world:reading is lit by ACCEPTED reading links in the trailing 15 min, not by every
    # finished reading. reading_turn is the nearest event kind in temporal_self_event.
    "world:reading": EventSource("reading_turn", READING_RATE_WINDOW.total_seconds(), "READING_RATE_WINDOW",
                                 note="loose"),
    "node:substrate.street": EventSource("street_event", EVENT_ORIENTING_WINDOW_SEC,
                                         "EVENT_ORIENTING_WINDOW_SEC"),
}

# --- window -----------------------------------------------------------------------------------

@dataclass(frozen=True)
class Window:
    """The asked window [start, end) and its baseline [base_start, start), in whole UTC hours (the
    rollup's grain: ``parse_window`` snaps the end down to the hour). ``first_hour`` is the oldest
    rollup hour in [base_start, end); ``covered()`` clamps both sides to it."""

    label: str
    start: datetime
    end: datetime
    base_start: datetime
    first_hour: datetime | None = None
    asked_end: datetime | None = None  # an explicit <end> before the snap to whole hours

    def covered(self, first: datetime | None) -> "Window":
        if first is None:
            return replace(self, first_hour=None)
        return replace(self, first_hour=first, start=max(self.start, first), base_start=max(self.base_start, first))

    @property
    def seconds(self) -> float:
        return max(0.0, (self.end - self.start).total_seconds())

    @property
    def base_seconds(self) -> float:
        return max(0.0, (self.start - self.base_start).total_seconds())


def _floor_hour(dt: datetime) -> datetime:
    return dt.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)


def parse_end(text: str | None, now: datetime | None = None) -> datetime:
    if text in (None, "", "now"):
        return (now or datetime.now(timezone.utc)).replace(microsecond=0)
    t = text.strip().replace(" ", "T", 1)
    if t.endswith(("Z", "z")):
        t = t[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(t)
    except ValueError as exc:
        raise ValueError(f"attention: <end> must be an ISO time like 2026-11-17T08:23Z, got {text!r}") from exc
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def parse_window(text: str | None = None, end: str | None = None, now: datetime | None = None) -> Window:
    label = (text or DEFAULT_WINDOW).strip().lower()
    m = WINDOW_RE.fullmatch(label)
    if not m:
        raise ValueError(f"attention: window must be like 24h or 7d, got {text!r}")
    span = timedelta(hours=int(m.group(1))) if m.group(2) == "h" else timedelta(days=int(m.group(1)))
    if span <= timedelta(0) or span > MAX_WINDOW:
        raise ValueError(f"attention: window must be 1h to {MAX_WINDOW.days}d, got {label}")
    asked = parse_end(end, now)
    stop = _floor_hour(asked)  # the rollup counts whole hours; the current hour is still filling
    return Window(label, stop - span, stop, stop - span - BASELINE,
                  asked_end=asked if end not in (None, "", "now") else None)


# --- SQL ----------------------------------------------------------------------------------------

def _ts(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def coverage_sql(w: Window) -> str:
    """The oldest rollup hour in [base_start, end), and the oldest rollup hour at all (``origin``: the
    hour the writer was deployed, only partly counted). Primary-key reads."""
    return f"""SET LOCAL statement_timeout = '{STATEMENT_TIMEOUT}';
SELECT min(hour) FILTER (WHERE hour >= '{_ts(w.base_start)}' AND hour < '{_ts(w.end)}') AS first,
       min(hour) AS origin
  FROM substrate_attention_hour_rollup"""


def frames_sql(w: Window) -> str:
    """The rollup's counts summed per side (cur t = the window, f = its baseline). Row meanings are
    orion/attention/frame_rollup.py's: part 'frame' a/b/c = frames/rest/contested; part 'won' per
    target a/b/c/d = eligible/ranked first/eligible contested/first contested; part 'not' per target
    and cause a = frames judged and not attended."""
    return f"""SET LOCAL statement_timeout = '{STATEMENT_TIMEOUT}';
SELECT part, hour >= '{_ts(w.start)}' AS cur, target, cause, sum(a) AS a, sum(b) AS b, sum(c) AS c, sum(d) AS d
  FROM substrate_attention_hour_rollup
 WHERE hour >= '{_ts(w.base_start)}' AND hour < '{_ts(w.end)}'
 GROUP BY 1, 2, 3, 4"""


def events_sql(w: Window) -> str:
    """Each event source's own activity per side (whole, never sampled): n = events inside the side, and
    lit = seconds of the side covered by the union of its events' news windows. Each event lights
    [t, min(t + window, next event)), so overlapping windows count once; an event just before the side
    that still lights part of it counts for that part."""
    conds, wins = [], []
    for src in EVENT_SOURCES.values():
        c = f"source_kind = '{src.kind}'"
        if src.payload_key:
            c += f" AND payload_json->>'{src.payload_key}' = '{src.payload_value}'"
        conds.append(f"({c})")
        wins.append(f"WHEN '{src.kind}' THEN {src.window_sec:.0f}")
    reach = max(src.window_sec for src in EVENT_SOURCES.values())
    a, b0, b = _ts(w.start), _ts(w.base_start), _ts(w.end)

    def side(cur: str, lo: str, hi: str) -> str:
        return f"""SELECT '{cur}' AS cur, kind, count(*) FILTER (WHERE t >= '{lo}' AND t < '{hi}') AS n,
       coalesce(sum(greatest(0, extract(epoch FROM least(t + w * interval '1 second', coalesce(nt, 'infinity'), '{hi}'::timestamptz)
                                        - greatest(t, '{lo}'::timestamptz)))), 0) AS lit
  FROM s WHERE t < '{hi}' GROUP BY kind"""

    return f"""SET LOCAL statement_timeout = '{STATEMENT_TIMEOUT}';
WITH e AS (
  SELECT source_kind AS kind, occurred_at AS t, CASE source_kind {' '.join(wins)} END AS w
    FROM temporal_self_event
   WHERE occurred_at >= '{b0}'::timestamptz - interval '{reach:.0f} seconds' AND occurred_at < '{b}'
     AND ({' OR '.join(conds)})
), s AS (SELECT kind, t, w, lead(t) OVER (PARTITION BY kind ORDER BY t) AS nt FROM e)
{side('t', a, b)}
UNION ALL
{side('f', b0, a)}"""


def lit_seconds(times: Iterable[datetime], window_sec: float, lo: datetime, hi: datetime) -> float:
    """Python reference for events_sql's ``lit``: seconds of [lo, hi) inside the union of
    [t, t + window) over the event times (each window cut at the next event). The live eval checks
    the SQL against this on real rows."""
    ts = sorted(times)
    total = 0.0
    for i, t in enumerate(ts):
        stop = min(t + timedelta(seconds=window_sec), ts[i + 1] if i + 1 < len(ts) else hi, hi)
        total += max(0.0, (stop - max(t, lo)).total_seconds())
    return total


# --- reading ------------------------------------------------------------------------------------

@dataclass
class Side:
    """One side's counts (cur = the window, base = the kept frames before it). Frame counts are
    whole frames from the rollup; event counts are whole."""

    seconds: float = 0.0
    frames: int = 0
    rest: int = 0
    contested: int = 0
    present: dict[str, int] = field(default_factory=dict)
    won: dict[str, int] = field(default_factory=dict)
    present_contested: dict[str, int] = field(default_factory=dict)
    won_contested: dict[str, int] = field(default_factory=dict)
    causes: dict[str, dict[str, int]] = field(default_factory=dict)  # target -> cause -> frames
    events: dict[str, int] = field(default_factory=dict)  # event kind -> events
    lit: dict[str, float] = field(default_factory=dict)  # event kind -> seconds lit by its news windows (merged)

    @property
    def interval(self) -> float:
        """Seconds per frame."""
        return self.seconds / self.frames if self.frames else 0.0


@dataclass
class AttentionRead:
    window: str
    asked_start: str
    start: str
    end: str
    base_start: str
    cur: Side
    base: Side
    kinds: dict[str, str] = field(default_factory=dict)  # target -> event|continuous|event-written|percentile
    took_sec: float | None = None
    asked_end: str = ""


def _truth(v: str) -> bool:
    return str(v).strip().lower() in ("t", "true", "1")


def parse_first(rows: list[dict[str, str]]) -> datetime | None:
    """The first whole counted hour in range. The rollup's own first hour (``origin``) began counting at
    deploy, partway through, so it is skipped: its frames would be read against a full hour of seconds
    and of events (review 2026-11-18)."""
    raw = (rows[0].get("first") or "").strip() if rows else ""
    if not raw:
        return None
    first = parse_end(raw)
    origin = (rows[0].get("origin") or "").strip()
    if origin and parse_end(origin) == first:
        first += timedelta(hours=1)
    return first


def parse(w: Window, frame_rows: Iterable[dict[str, str]], event_rows: Iterable[dict[str, str]],
          asked_start: datetime | None = None) -> AttentionRead:
    cur, base = Side(w.seconds), Side(w.base_seconds)
    for r in frame_rows:
        side = cur if _truth(r["cur"]) else base
        a, b, c, d = (int(r[k] or 0) for k in ("a", "b", "c", "d"))
        if r["part"] == "frame":
            side.frames, side.rest, side.contested = a, b, c
        elif r["part"] == "won":
            t = r["target"]
            side.present[t], side.won[t], side.present_contested[t], side.won_contested[t] = a, b, c, d
        elif r["part"] == "not":
            causes = side.causes.setdefault(r["target"], {})
            causes[r["cause"]] = causes.get(r["cause"], 0) + a
    for r in event_rows:
        side = cur if _truth(r["cur"]) else base
        side.events[r["kind"]] = side.events.get(r["kind"], 0) + int(r["n"])
        side.lit[r["kind"]] = side.lit.get(r["kind"], 0.0) + float(r.get("lit") or 0.0)
    return AttentionRead(w.label, _ts(asked_start or w.start), _ts(w.start), _ts(w.end), _ts(w.base_start), cur, base)


def target_kind(target: str) -> str:
    """How a target's eligibility is judged: the explicit event map first, then the semantic layer
    (``world_first``'s own functions, so the rule cannot drift from attention's)."""
    if target in EVENT_SOURCES:
        return "event"
    from orion.attention.world_first import (
        node_prediction_error_continuous,
        node_prediction_error_event_written,
    )

    if node_prediction_error_continuous(target):
        return "continuous"
    if node_prediction_error_event_written(target):
        return "event-written"
    return "percentile"


def targets(r: AttentionRead) -> list[str]:
    seen = set(r.cur.present) | set(r.cur.causes) | set(r.base.present) | set(r.base.causes)
    return sorted(seen, key=lambda t: (-r.cur.present.get(t, 0), -r.base.present.get(t, 0), t))


# --- rates --------------------------------------------------------------------------------------

def _rate(num: int, den: int) -> float | None:
    return num / den if den else None


def pct(x: float | None) -> str:
    if x is None:
        return "n/a"
    v = 100.0 * x
    # one decimal under 20% and between 99% and 100%: "100%" must mean none lost
    return f"{v:.1f}%" if v < 20 or 99 < v < 100 else f"{v:.0f}%"


def rate_text(num: int, den: int) -> str:
    """'93% (2,503 of 2,694)': a rate always carries its numerator and denominator."""
    return f"{pct(_rate(num, den))} ({num:,} of {den:,})"


def drift(cur: float | None, base: float | None) -> str:
    """Relative change of a rate against the same rate before the window."""
    if cur is None or base is None:
        return "n/a"
    if base == 0:
        return "+0%" if cur == 0 else "new (prior 0%)"
    return f"{(cur / base - 1.0) * 100:+.0f}%"


@dataclass
class Eligibility:
    rate: float | None
    num: int
    den: int  # expected frames (event) or frames it was judged (other)
    reference: str


def eligibility(side: Side, target: str, kind: str) -> Eligibility:
    present = side.present.get(target, 0)
    src = EVENT_SOURCES.get(target)
    if kind == "event" and src is not None:
        events = side.events.get(src.kind, 0)
        # expected frames = seconds its events lit (windows merged) / seconds per frame
        expected = side.lit.get(src.kind, 0.0) * side.frames / side.seconds if side.seconds else 0.0
        ref = f"{events:,} {src.kind} x {src.window_sec:.0f} s, overlaps merged"
        return Eligibility(present / expected if expected else None, present, round(expected), ref)
    judged = judged_frames(side, target)  # frames it was a candidate at all, not every frame
    return Eligibility(_rate(present, judged), present, judged, f"design {pct(DESIGN_ELIGIBLE_RATE)}")


def judged_frames(side: Side, target: str) -> int:
    return side.present.get(target, 0) + sum(side.causes.get(target, {}).values())


# --- the rule: every number is part of a rate, a duration, a time or a ref -----------------------

# Allowed number shapes on a line: a rate with its numerator and denominator ("91% (3,971 of 4,368)"),
# a rate alone or a relative change ("10.0%", "+24%"), a duration ("900 s", "24h", "2.0d"), a time
# ("2026-11-17T08:23Z"), an event-expectation ref ("2 chat_turn x 900 s": the denominator's parts,
# printed inside the rate's own clause), and ids/paths (refs).
_ALLOWED_NUMBERS = [
    r"n/a \(\d[\d,]* of \d[\d,]*\)",  # before refs: "n/a" holds a slash
    r"\b[a-z_]+:[A-Za-z_][\w.\-:]*",  # refs: node:substrate.perception, world:chat (no digit may lead the id)
    r"(?<![\w/])/[A-Za-z_.][\w./\-]*",  # paths: /tmp/...-24h-1.json (a "834/40846" ratio is not a path)
    r"[+-]?\d[\d,]*(?:\.\d+)?% \(\d[\d,]* of \d[\d,]*\)",
    r"\d[\d,]* [a-z_]+ x \d+ s",
    r"cut at \d+ chars|showing \d+ of \d+ lines",  # dig's own length-cut notes (capped())
    r"\d{4}-\d\d-\d\dT\d\d:\d\d(?::\d\d)?Z?",
    r"[+-]?\d+(?:\.\d+)?%",
    r"\d+(?:\.\d+)? ?(?:s|h|d|min)\b",
]


def bare_counts(line: str) -> list[str]:
    """Numbers on a line that are not part of a rate, a duration, a time or a ref."""
    rest = line
    for pat in _ALLOWED_NUMBERS:
        rest = re.sub(pat, " ", rest)
    return re.findall(r"\d[\d,]*", rest)



# --- render -------------------------------------------------------------------------------------

LEGEND = ("kinds: [event] eligible frames vs its own events x its news window / frame interval ([loose]: "
          "world:reading is lit by accepted reading links, not every reading); [percentile] eligible share of "
          "frames vs the top-decile design; [continuous] same, the dwell rule keeps it below; [event-written] "
          "same, no clean event source, the orienting window keeps it below")


def _span(seconds: float) -> str:
    return f"{seconds / 3600:.0f}h" if seconds < 2 * 86400 else f"{seconds / 86400:.1f}d"


def target_line(r: AttentionRead, t: str) -> str:
    kind = r.kinds.get(t) or target_kind(t)
    c, b = r.cur, r.base
    present = c.present.get(t, 0)
    e_now, e_then = eligibility(c, t, kind), eligibility(b, t, kind)
    if present:
        pc = c.present_contested.get(t, 0)
        won = f"won {rate_text(c.won.get(t, 0), present)}, contested won {rate_text(c.won_contested.get(t, 0), pc)}"
    else:
        won = "never eligible"
    if kind == "event":
        src = EVENT_SOURCES[t]
        elig = f"eligible {rate_text(e_now.num, e_now.den)} of expected, {e_now.reference}"
        if src.note:
            elig += f" [{src.note}]"
    else:
        elig = f"eligible {rate_text(e_now.num, e_now.den)} vs design {pct(DESIGN_ELIGIBLE_RATE)}"
    if not present and not judged_frames(c, t) and kind != "event":
        return f"  {t} [{kind}]: not judged in this window (a candidate only before it)"
    won_then = _rate(b.won.get(t, 0), b.present.get(t, 0))
    won_now = _rate(c.won.get(t, 0), present)
    return f"  {t} [{kind}]: {won}; {elig}; drift won {drift(won_now, won_then)}, eligible {drift(e_now.rate, e_then.rate)}"


def prior_line(r: AttentionRead, t: str) -> str:
    """The prior rates each drift was taken against, with their numerators and denominators (the
    prior may be short or thin, so a drift on a small prior count reads as one)."""
    kind = r.kinds.get(t) or target_kind(t)
    b = r.base
    e = eligibility(b, t, kind)
    return f"  {t}: won {rate_text(b.won.get(t, 0), b.present.get(t, 0))}, eligible {rate_text(e.num, e.den)}"


def cause_line(r: AttentionRead, t: str) -> str:
    causes = r.cur.causes.get(t, {})
    total = sum(causes.values())
    if not total:
        if r.cur.present.get(t, 0):
            return f"  {t}: attended in every frame it was judged, {rate_text(r.cur.present[t], r.cur.present[t])}"
        return f"  {t}: not judged in this window"
    parts = sorted(causes.items(), key=lambda kv: (-kv[1], kv[0]))
    shown = ", ".join(f"{k} {rate_text(n, total)}" for k, n in parts[:2])
    return f"  {t}: {shown}" + (" ..." if len(parts) > 2 else "")


def render(r: AttentionRead) -> list[str]:
    c, b = r.cur, r.base
    lines = [f"attention {r.window} to {r.end[:16]}Z, read-only, relative rates only"
             + (f" (read in {r.took_sec:.1f} s)" if r.took_sec is not None else "")]
    if r.asked_end and r.asked_end[:16] != r.end[:16]:
        lines.append(f"  asked to {r.asked_end[:16]}Z: snapped down to whole UTC hours (the rollup's grain)")
    if r.start != r.asked_start:
        lines.append(f"  counted only from {r.start[:16]}Z: the hourly rollup starts there")
    if not c.frames:
        lines.append("  no attention frames in this window: unread, not calm")
        return lines
    prior = (f"drift = relative change vs the {_span(b.seconds)} before it" if b.frames
             else "drift: n/a, no counted hours before the window yet")
    lines.append(f"  whole UTC hours, every frame, about {c.interval:.1f} s a frame; {prior}")
    lines.append(f"rest (no eligible target): {rate_text(c.rest, c.frames)} of frames; "
                 f"drift {drift(_rate(c.rest, c.frames), _rate(b.rest, b.frames))} (was {rate_text(b.rest, b.frames)})")
    lines.append(f"contested (more than one eligible): {rate_text(c.contested, c.frames)} of frames; "
                 f"drift {drift(_rate(c.contested, c.frames), _rate(b.contested, b.frames))} "
                 f"(was {rate_text(b.contested, b.frames)})")
    lines.append("won = ranked first in the frames it was eligible (contested: of those with more than one eligible); "
                 "eligible against its own activity:")
    order = targets(r)
    never = [t for t in order if not c.present.get(t) and not b.present.get(t)]
    lines += [target_line(r, t) for t in order if t not in never]
    if never:
        lines.append("  never eligible here or before (eligible / frames it was judged): "
                     + ", ".join(f"{t} {rate_text(0, judged_frames(c, t))}" for t in never))
    lines.append(LEGEND)
    lines.append("not attended, top causes (share of the frames it was judged and not attended):")
    lines += [cause_line(r, t) for t in order]
    if b.frames:
        lines.append("prior rates each drift is against:")
        lines += [prior_line(r, t) for t in order if t not in never]
    return lines


def to_file(r: AttentionRead) -> dict:
    out = asdict(r)
    out["rates"] = {}
    for t in targets(r):
        kind = r.kinds.get(t) or target_kind(t)
        row: dict = {"kind": kind}
        for name, side in (("cur", r.cur), ("base", r.base)):
            e = eligibility(side, t, kind)
            row[name] = {
                "won_when_eligible": _rate(side.won.get(t, 0), side.present.get(t, 0)),
                "won_when_contested": _rate(side.won_contested.get(t, 0), side.present_contested.get(t, 0)),
                "eligible": e.rate, "eligible_num": e.num, "eligible_den": e.den, "eligible_ref": e.reference,
            }
        out["rates"][t] = row
    return out


def read(w: Window, query: Callable[[str], list[dict[str, str]]]) -> AttentionRead:
    """Three read-only queries: rollup coverage, the rollup's counts, event counts."""
    asked = w.start
    w = w.covered(parse_first(query(coverage_sql(w))))
    if w.first_hour is None or w.seconds <= 0:
        r = AttentionRead(w.label, _ts(asked), _ts(max(asked, w.start)), _ts(w.end), _ts(w.base_start),
                          Side(), Side(), asked_end=_ts(w.asked_end) if w.asked_end else "")
        return r
    r = parse(w, query(frames_sql(w)), query(events_sql(w)), asked_start=asked)
    r.kinds = {t: target_kind(t) for t in targets(r)}
    r.asked_end = _ts(w.asked_end) if w.asked_end else ""
    return r
