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

One pass over the frames (one query, ``STATEMENT_TIMEOUT``), and the frames are sampled by a hash of
``frame_id`` so a pass reads about one day of frames whatever the window (rates are ratios, so a
uniform sample keeps them; the numerators and denominators printed are sampled frames, and the
header says the sample rate). Every printed line is a rate with its numerator and denominator, or a
ref (``tests/test_attention.py`` fails on a bare count).
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable

from orion.attention.world_first import (
    CHAT_RATE_WINDOW,
    EVENT_ORIENTING_WINDOW_SEC,
    READING_RATE_WINDOW,
)
from orion.substrate.prediction_error_magnitude import DEFAULT_BAND_CUTS

DEFAULT_WINDOW = "24h"
MAX_WINDOW = timedelta(days=30)
BASELINE = timedelta(days=7)
# Frames per pass: about one day for the window (~41k frames, ~6 s live on 2026-11-17), about half a day for
# the baseline (a drift ratio needs far fewer frames than the window's own rates). Past it, sample.
FRAMES_SPAN_PER_PASS = timedelta(days=1)
BASE_FRAMES_SPAN_PER_PASS = timedelta(hours=12)
STATEMENT_TIMEOUT = "25s"
WORLD_FIRST_PREFIX = "world-first:"
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

# World-first verdict causes, first match wins (the verdict text is world_first.judge_candidate's
# reason; "body high for itself, not sustained" is "not sustained", not "high").
CAUSES: tuple[tuple[str, str], ...] = (
    ("absent", "absent%"),
    ("insufficient history", "insufficient_history%"),
    ("past the orienting window", "%orienting window%"),
    ("not sustained", "%not sustained%"),
    ("below its band floor", "%floor%"),
    ("stale reading", "stale%"),
    ("not a measurement", "%not a measurement%"),
    ("no polarity", "no declared polarity%"),
    ("quiet for itself", "%quiet for itself%"),
    ("usual for itself", "%usual for itself%"),
    ("high for itself", "%high for itself%"),
)
OTHER_CAUSE = "other"


def _like(pattern: str, text: str) -> bool:
    rx = "^" + ".*".join(re.escape(p) for p in pattern.split("%")) + "$"
    return re.match(rx, text, re.S) is not None


def verdict_of(reason: str) -> str:
    """'world-first: not attended -- body quiet for itself; ...' -> 'body quiet for itself'."""
    if " -- " not in reason:
        return ""  # same as the SQL: split_part(..., ' -- ', 2) is '' without the separator
    return reason.split(" -- ", 1)[1].split(";", 1)[0].strip()


def classify(reason: str) -> str:
    """The cause class of one not-attended reason (the SQL CASE below is built from the same table)."""
    verdict = verdict_of(reason)
    for label, pattern in CAUSES:
        if _like(pattern, verdict):
            return label
    return OTHER_CAUSE


def _cause_case(col: str) -> str:
    verdict = f"btrim(split_part(split_part({col}, ' -- ', 2), ';', 1))"
    # '_' is a one-character wildcard in LIKE; escape it with '!' (a backslash is refused by ReadOnlyPsql)
    whens = " ".join(f"WHEN {verdict} LIKE '{p.replace('_', '!_')}' ESCAPE '!' THEN '{label}'" for label, p in CAUSES)
    return f"CASE {whens} ELSE '{OTHER_CAUSE}' END"




# --- window -----------------------------------------------------------------------------------

@dataclass(frozen=True)
class Window:
    """The asked window [start, end) and its baseline [base_start, start). ``first_frame`` is the
    oldest frame kept in [base_start, end) (frames are pruned after ~72 h, so the baseline is
    usually shorter than ``BASELINE``); ``covered()`` clamps both sides to it."""

    label: str
    start: datetime
    end: datetime
    base_start: datetime
    first_frame: datetime | None = None

    def covered(self, first: datetime | None) -> "Window":
        if first is None:
            return replace(self, first_frame=None)
        return replace(self, first_frame=first, start=max(self.start, first), base_start=max(self.base_start, first))

    @property
    def seconds(self) -> float:
        return max(0.0, (self.end - self.start).total_seconds())

    @property
    def base_seconds(self) -> float:
        return max(0.0, (self.start - self.base_start).total_seconds())

    @property
    def k_cur(self) -> int:  # sample 1 in k frames (hash of frame_id)
        return _sample_k(self.seconds)

    @property
    def k_base(self) -> int:
        return _sample_k(self.base_seconds, BASE_FRAMES_SPAN_PER_PASS)


def _sample_k(seconds: float, per_pass: timedelta = FRAMES_SPAN_PER_PASS) -> int:
    return max(1, math.ceil(seconds / per_pass.total_seconds() - 1e-9))


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
    stop = parse_end(end, now)
    return Window(label, stop - span, stop, stop - span - BASELINE)


# --- SQL ----------------------------------------------------------------------------------------

def _ts(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def coverage_sql(w: Window) -> str:
    """The oldest kept frame in [base_start, end): an index read, no frame_json."""
    return f"""SET LOCAL statement_timeout = '{STATEMENT_TIMEOUT}';
SELECT min(generated_at) AS first FROM substrate_attention_frames
 WHERE generated_at >= '{_ts(w.base_start)}' AND generated_at < '{_ts(w.end)}'"""


def frames_sql(w: Window) -> str:
    """One pass over the frames of [base_start, end), one aggregate. Per side (cur t = the window,
    f = its baseline): part 'frame': a = frames, b = rest (no eligible target), c = contested
    (more than one eligible); part 'won', per target: a = frames it was eligible, b = rank 1,
    c = eligible and contested, d = rank 1 and contested; part 'not', per target and cause:
    a = frames judged and not attended.

    ``frame_json || '{}'`` detoasts each frame once in the innermost subquery (OFFSET 0 keeps it
    there): three bare references read the toast three times (9.5 s vs 5.5 s live, 2026-11-17)."""
    a, b0, b = _ts(w.start), _ts(w.base_start), _ts(w.end)
    wf = f"'{WORLD_FIRST_PREFIX}%'"
    return f"""SET LOCAL statement_timeout = '{STATEMENT_TIMEOUT}';
SELECT el.part, f.cur, el.target, el.cause, count(*) AS a,
       count(*) FILTER (WHERE (el.part = 'frame' AND f.n = 0) OR (el.part = 'won' AND el.o = 1)) AS b,
       count(*) FILTER (WHERE el.part <> 'not' AND f.n > 1) AS c,
       count(*) FILTER (WHERE el.part = 'won' AND el.o = 1 AND f.n > 1) AS d
  FROM (SELECT cur, j->'dominant_targets' AS dt, j->'suppressed_targets' AS st,
               (SELECT count(*) FROM jsonb_array_elements(j->'dominant_targets') e
                 WHERE e->'reasons'->>0 LIKE {wf}) AS n
          FROM (SELECT generated_at >= '{a}' AS cur, frame_json || '{{}}'::jsonb AS j
                  FROM substrate_attention_frames
                 WHERE generated_at >= '{b0}' AND generated_at < '{b}'
                   AND hashtext(frame_id) % (CASE WHEN generated_at >= '{a}' THEN {w.k_cur} ELSE {w.k_base} END) = 0
                OFFSET 0) raw
        OFFSET 0) f
 CROSS JOIN LATERAL (
       SELECT 'frame' AS part, 0::bigint AS o, '' AS target, '' AS cause
       UNION ALL
       SELECT 'won', x.o, x.t->>'target_id', '' FROM jsonb_array_elements(f.dt) WITH ORDINALITY AS x(t, o)
        WHERE x.t->'reasons'->>0 LIKE {wf}
       UNION ALL
       SELECT 'not', 0, t->>'target_id', {_cause_case("t->'reasons'->>0")} FROM jsonb_array_elements(f.st) AS t
        WHERE t->'reasons'->>0 LIKE {wf}) el
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
    sampled frames (1 in ``k``); event counts are whole."""

    seconds: float = 0.0
    k: int = 1
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
        """Seconds per frame (real frames, not sampled)."""
        return self.seconds / (self.frames * self.k) if self.frames else 0.0


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


def _truth(v: str) -> bool:
    return str(v).strip().lower() in ("t", "true", "1")


def parse_first(rows: list[dict[str, str]]) -> datetime | None:
    raw = (rows[0].get("first") or "").strip() if rows else ""
    return parse_end(raw) if raw else None


def parse(w: Window, frame_rows: Iterable[dict[str, str]], event_rows: Iterable[dict[str, str]],
          asked_start: datetime | None = None) -> AttentionRead:
    cur, base = Side(w.seconds, w.k_cur), Side(w.base_seconds, w.k_base)
    for r in frame_rows:
        side = cur if _truth(r["cur"]) else base
        a, b, c, d = (int(r[k] or 0) for k in ("a", "b", "c", "d"))
        if r["part"] == "frame":
            side.frames, side.rest, side.contested = a, b, c
        elif r["part"] == "won":
            t = r["target"]
            side.present[t], side.won[t], side.present_contested[t], side.won_contested[t] = a, b, c, d
        elif r["part"] == "not":
            side.causes.setdefault(r["target"], {})[r["cause"]] = a
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
    den: int  # expected frames (event) or all frames (other); sampled units
    reference: str


def eligibility(side: Side, target: str, kind: str) -> Eligibility:
    present = side.present.get(target, 0)
    src = EVENT_SOURCES.get(target)
    if kind == "event" and src is not None:
        events = side.events.get(src.kind, 0)
        # expected sampled frames = seconds its events lit (windows merged) / (seconds per sampled frame)
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
# printed inside the rate's own clause), "1 in 4" (the sample rate), and ids/paths (refs).
_ALLOWED_NUMBERS = [
    r"n/a \(\d[\d,]* of \d[\d,]*\)",  # before refs: "n/a" holds a slash
    r"\b[a-z_]+:[A-Za-z_][\w.\-:]*",  # refs: node:substrate.perception, world:chat (no digit may lead the id)
    r"(?<![\w/])/[A-Za-z_.][\w./\-]*",  # paths: /tmp/...-24h-1.json (a "834/40846" ratio is not a path)
    r"[+-]?\d[\d,]*(?:\.\d+)?% \(\d[\d,]* of \d[\d,]*\)",
    r"\d[\d,]* [a-z_]+ x \d+ s",
    r"\b1 in \d+\b",
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
        won = f"won {rate_text(c.won.get(t, 0), present)}, won-contested {rate_text(c.won_contested.get(t, 0), pc)}"
    else:
        won = "never eligible"
    if kind == "event":
        src = EVENT_SOURCES[t]
        elig = f"eligible {rate_text(e_now.num, e_now.den)} of expected: {e_now.reference}"
        if src.note:
            elig += f" [{src.note}]"
    else:
        elig = f"eligible {rate_text(e_now.num, e_now.den)} vs {pct(DESIGN_ELIGIBLE_RATE)} design"
    if not present and not judged_frames(c, t) and kind != "event":
        return f"  {t} [{kind}]: not judged in this window (a candidate only before it)"
    won_then = _rate(b.won.get(t, 0), b.present.get(t, 0))
    won_now = _rate(c.won.get(t, 0), present)
    return (f"  {t} [{kind}]: {won}; {elig}; drift won {drift(won_now, won_then)} "
            f"(was {rate_text(b.won.get(t, 0), b.present.get(t, 0))}), "
            f"eligible {drift(e_now.rate, e_then.rate)} (was {rate_text(e_then.num, e_then.den)})")


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
    if r.start != r.asked_start:
        lines.append(f"  frames kept only from {r.start[:16]}Z: frames are pruned after about 72 h")
    if not c.frames:
        lines.append("  no attention frames in this window: unread, not calm")
        return lines
    sample = "every frame" if c.k == 1 else f"1 in {c.k} frames sampled"
    prior = (f"drift = relative change vs the {_span(b.seconds)} of kept frames before it "
             f"({'every frame' if b.k == 1 else f'1 in {b.k} sampled'})" if b.frames
             else "drift: n/a, no kept frames before the window (frames are pruned after about 72 h)")
    lines.append(f"  {sample}, about {c.interval:.1f} s a frame; {prior}")
    lines.append(f"rest (no eligible target): {rate_text(c.rest, c.frames)} of frames; "
                 f"drift {drift(_rate(c.rest, c.frames), _rate(b.rest, b.frames))} (was {rate_text(b.rest, b.frames)})")
    lines.append(f"contested (more than one eligible): {rate_text(c.contested, c.frames)} of frames; "
                 f"drift {drift(_rate(c.contested, c.frames), _rate(b.contested, b.frames))} "
                 f"(was {rate_text(b.contested, b.frames)})")
    lines.append("won = ranked first in the frames it was eligible (won-contested: of those with more than one eligible); "
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
    """Three read-only queries: frame coverage (index only), one pass over frames, event counts."""
    asked = w.start
    w = w.covered(parse_first(query(coverage_sql(w))))
    if w.first_frame is None or w.seconds <= 0:
        r = AttentionRead(w.label, _ts(asked), _ts(max(asked, w.start)), _ts(w.end), _ts(w.base_start),
                          Side(), Side())
        return r
    r = parse(w, query(frames_sql(w)), query(events_sql(w)), asked_start=asked)
    r.kinds = {t: target_kind(t) for t in targets(r)}
    return r
