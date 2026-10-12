"""Does a sentence that cites Orion's day describe the cited arc correctly? Pure, deterministic.

The citation gate (``citations.py``) proves a cited arc id exists and was shown. It does not prove
the sentence around it is true to that arc. This module checks the facts a sentence states against
the stored arc rows (``temporal_self_arc.arc_json``), using only what can be decided without a
model:

* **kind** -- the renderer's own phrases ("talking with Juniper", "slept", "curiosity run", ...)
  name an arc kind; some cited arc must be of that kind.
* **subject** -- "attention on <X>" must name a cited attention arc's label or subject ref.
* **time** -- "N min ago" / "5 h 13 min ago" / "under a minute ago" must land on a cited arc's
  began/ended/last-seen time; "for N min" on its start; a local clock "HH:MM" on one of them.
* **counts** -- "back to it N times" / "N returns" against ``attention_returns``; "N open threads"
  against the cited concern arcs open at the time.

The phrase table mirrors the stance cue and closed-day block templates (``stance_cue.py``,
``day_block.py``). It is deliberately not grown to catch free paraphrase: a sentence the table
does not parse is **undecidable**, and that count is reported, not hidden. There is no LLM judge.

Arc rows are mutable (an open arc's ``last_seen_at`` and ``attention_returns`` keep moving after
the sentence was written). Where the row has moved past the sentence's time and the claim no
longer matches the current row, the check says undecidable, never mismatch.

Outcome per sentence: ``mismatch`` if any fact contradicts every cited arc, else ``match`` if at
least one fact matched, else ``undecidable``.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Literal, Mapping
from zoneinfo import ZoneInfo

from orion.temporal_self.citations import _ANY_ARC_BRACKET_RE, _SENTENCE_SPLIT_RE
from orion.temporal_self.day import DEFAULT_TZ

Outcome = Literal["match", "mismatch", "undecidable"]

# Stance cue rounds ages down to the minute and its frame may trail the turn by up to 900 s
# (DEFAULT_MAX_AGE_SEC): a claim may read older than the row by that much, never much newer.
ROUNDING_SEC = 60.0
FRAME_LAG_SEC = 900.0
MIN_TOLERANCE_SEC = 120.0
RELATIVE_TOLERANCE = 0.10
CLOCK_TOLERANCE_MIN = 1
SENTENCE_MAX_CHARS = 300

# The renderer templates' own wording -> the arc kind it names (stance_cue.py, day_block.py).
KIND_PHRASES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?:talking|talked|a conversation) with Juniper\b|\bJuniper last spoke\b", re.I), "conversation"),
    (re.compile(r"\b(?:slept|asleep|sleep cycles?)\b", re.I), "sleep"),
    (re.compile(r"\bcuriosity runs?\b", re.I), "curiosity"),
    (re.compile(r"\bimage attempts?\b", re.I), "imagery"),
    (re.compile(r"\battention on\b", re.I), "attention"),
    (re.compile(r"\b(?:open|conversation) threads?\b", re.I), "concern"),
    (re.compile(r"\breverie\b", re.I), "reverie"),
)
_SPAN = r"(?:(?P<h>\d+)\s*h(?:ours?|rs?)?\.?)?\s*(?:(?P<m>\d+)\s*min(?:utes?|s)?\.?)?"
AGO_RE = re.compile(rf"(?<![\w:])(?:{_SPAN}|(?P<under>under a minute))\s+ago\b", re.I)
FOR_RE = re.compile(rf"\bfor\s+{_SPAN}(?=[\s,.;]|$)", re.I)
CLOCK_RE = re.compile(r"(?<![\d:])([01]\d|2[0-3]):([0-5]\d)(?![\d:])(?!\s*UTC)")
RETURNS_RE = re.compile(r"\bback to it (\d+) times?\b|\b(\d+) returns?\b", re.I)
THREADS_RE = re.compile(r"\b(\d+) open threads?\b", re.I)
# A label may hold parentheses ("Juniper is talking (1 message in the last 15 min)"): stop only
# at the template's own separators.
SUBJECT_RE = re.compile(r"\battention on (.+?)(?=\s+for\s|\s*[,;\[]|\.(?:\s|$)|\s+set aside|\s+ended|$)", re.I)
# The word before an age says which time it is: "ended 44 min ago", "Asleep since 2 h ago".
AGO_FIELD = {"ended": ("ended",), "since": ("began",)}


@dataclass(frozen=True)
class ArcFacts:
    arc_id: str
    kind: str
    subject_ref: str
    subject_label: str
    status: str
    began_at: datetime
    ended_at: datetime | None
    last_seen_at: datetime | None
    attention_returns: int

    @classmethod
    def from_row(cls, arc: Mapping[str, Any]) -> "ArcFacts":
        return cls(
            arc_id=str(arc["arc_id"]), kind=str(arc.get("kind") or ""), subject_ref=str(arc.get("subject_ref") or ""),
            subject_label=str(arc.get("subject_label") or ""), status=str(arc.get("status") or ""),
            began_at=_dt(arc["began_at"]), ended_at=_dt(arc.get("ended_at")), last_seen_at=_dt(arc.get("last_seen_at")),
            attention_returns=int(arc.get("attention_returns") or 0),
        )

    def times(self) -> list[tuple[str, datetime]]:
        out = [("began", self.began_at)]
        if self.ended_at is not None:
            out.append(("ended", self.ended_at))
        if self.last_seen_at is not None:
            out.append(("last seen", self.last_seen_at))
        return out

    def moved_after(self, t: datetime) -> bool:
        """The row changed after ``t``: what it said at ``t`` is not stored."""
        return any(seen > t + timedelta(seconds=MIN_TOLERANCE_SEC) for name, seen in self.times() if name != "began")


@dataclass
class SentenceCheck:
    source: str  # imperative | tone | reply
    sentence: str
    cited: list[str]
    attribution: Literal["inline", "evidence_refs"]
    outcome: Outcome
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"source": self.source, "sentence": self.sentence, "cited": self.cited,
                "attribution": self.attribution, "outcome": self.outcome, "reasons": self.reasons}


def _dt(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _span_sec(m: re.Match[str]) -> float | None:
    if m.groupdict().get("under"):
        return 30.0
    h, mins = m.group("h"), m.group("m")
    if h is None and mins is None:
        return None
    return int(h or 0) * 3600.0 + int(mins or 0) * 60.0


def _tolerance(span_sec: float) -> float:
    return max(MIN_TOLERANCE_SEC, RELATIVE_TOLERANCE * span_sec) + ROUNDING_SEC


def _fmt(sec: float) -> str:
    sec = abs(sec)
    return f"{int(sec // 60)} min" if sec < 3600 else f"{sec / 3600:.1f} h"


# A fact check returns (outcome, reason); outcome None means the fact did not apply.
Fact = tuple[Outcome, str]


def _check_kinds(sentence: str, arcs: list[ArcFacts], inline: bool) -> list[Fact]:
    out: list[Fact] = []
    kinds = {a.kind for a in arcs}
    for pattern, kind in KIND_PHRASES:
        m = pattern.search(sentence)
        if m is None:
            continue
        if kind in kinds:
            out.append(("match", f"kind: '{m.group(0)}' names a {kind} arc, and one is cited"))
        elif inline:
            out.append(("mismatch", f"kind: '{m.group(0)}' names a {kind} arc; cited arcs are {sorted(kinds)}"))
        else:
            out.append(("undecidable", f"kind: '{m.group(0)}' names a {kind} arc; none is in evidence_refs"))
    return out


def _check_subject(sentence: str, arcs: list[ArcFacts], inline: bool) -> list[Fact]:
    out: list[Fact] = []
    attention = [a for a in arcs if a.kind == "attention"]
    for m in SUBJECT_RE.finditer(sentence):
        said = m.group(1).strip().lower()
        if not said or not attention:
            continue
        hit = [a for a in attention if said in {a.subject_label.strip().lower(), a.subject_ref.lower()}
               or (a.subject_label and a.subject_label.strip().lower() in said)]
        if hit:
            out.append(("match", f"subject: '{m.group(1).strip()}' is the label of [arc:{hit[0].arc_id}]"))
        else:
            labels = sorted({a.subject_label or a.subject_ref for a in attention})
            out.append(("mismatch" if inline else "undecidable",
                        f"subject: '{m.group(1).strip()}' is not a cited attention arc's label {labels}"))
    return out


def _check_instant(what: str, claimed: datetime, span: float, at: datetime, arcs: list[ArcFacts],
                   fields: Iterable[str] | None, inline: bool) -> Fact:
    tol = _tolerance(span)
    candidates = [(a, name, t) for a in arcs for name, t in a.times() if fields is None or name in fields]
    for a, name, t in candidates:
        # A claim may read older than the row by the frame's lag (the cue saw an older frame).
        if -tol <= (t - claimed).total_seconds() <= tol + FRAME_LAG_SEC:
            return ("match", f"time: {what} -> {claimed:%H:%MZ}, [arc:{a.arc_id}] {name} {t:%H:%MZ}")
    if any(a.moved_after(at) for a in arcs):
        return ("undecidable", f"time: {what} -> {claimed:%H:%MZ}; a cited arc's row moved after the sentence, "
                               "its value at the time is not stored")
    if not inline:
        return ("undecidable", f"time: {what} -> {claimed:%H:%MZ} matches no evidence_refs arc (may name an uncited record)")
    near = min(candidates, key=lambda c: abs((c[2] - claimed).total_seconds()), default=None)
    detail = (f"; nearest [arc:{near[0].arc_id}] {near[1]} {near[2]:%H:%MZ}, off by {_fmt((near[2] - claimed).total_seconds())}"
              if near else "")
    return ("mismatch", f"time: {what} -> {claimed:%H:%MZ} matches no cited arc time{detail}")


def _check_times(sentence: str, at: datetime, arcs: list[ArcFacts], inline: bool, tz: ZoneInfo) -> list[Fact]:
    out: list[Fact] = []
    for m in AGO_RE.finditer(sentence):
        span = _span_sec(m)
        if span is None:
            continue
        before = sentence[:m.start()].split()
        fields = AGO_FIELD.get(before[-1].lower()) if before else None
        out.append(_check_instant(f"'{m.group(0).strip()}'", at - timedelta(seconds=span), span, at, arcs, fields, inline))
    for m in FOR_RE.finditer(sentence):
        span = _span_sec(m)
        if span is None:
            continue
        out.append(_check_instant(f"'{m.group(0).strip()}' (start)", at - timedelta(seconds=span), span, at, arcs,
                                  ("began",), inline))
    for m in CLOCK_RE.finditer(sentence):
        said = int(m.group(1)) * 60 + int(m.group(2))

        def off(t: datetime) -> int:
            local = t.astimezone(tz)
            d = abs(local.hour * 60 + local.minute - said)
            return min(d, 1440 - d)

        hits = [(a, name) for a in arcs for name, t in a.times() if off(t) <= CLOCK_TOLERANCE_MIN]
        if hits:
            a, name = hits[0]
            out.append(("match", f"clock: {m.group(0)} local is [arc:{a.arc_id}] {name}"))
        elif inline and len(arcs) == 1 and not arcs[0].moved_after(at):
            out.append(("mismatch", f"clock: {m.group(0)} local is none of [arc:{arcs[0].arc_id}]'s times"))
        else:
            out.append(("undecidable", f"clock: {m.group(0)} local matches no cited arc time "
                                       "(a clock may name something other than the arc)"))
    return out


def _check_counts(sentence: str, at: datetime, arcs: list[ArcFacts], inline: bool) -> list[Fact]:
    out: list[Fact] = []
    for m in RETURNS_RE.finditer(sentence):
        n = int(m.group(1) or m.group(2))
        have = sorted(a.attention_returns for a in arcs)
        if n in have:
            out.append(("match", f"count: {n} return(s) equals a cited arc's attention_returns"))
        elif inline and all(n > a.attention_returns for a in arcs):
            # Returns only grow: a claim above every current count was never true.
            out.append(("mismatch", f"count: {n} return(s); cited arcs have {have}"))
        elif inline and not any(a.moved_after(at) for a in arcs):
            out.append(("mismatch", f"count: {n} return(s); cited arcs have {have}"))
        else:
            out.append(("undecidable", f"count: {n} return(s); cited arcs have {have} now "
                                       "(rows moved after the sentence, or not cited inline)"))
    for m in THREADS_RE.finditer(sentence):
        n = int(m.group(1))
        open_then = [a for a in arcs if a.kind == "concern" and a.began_at <= at + timedelta(seconds=MIN_TOLERANCE_SEC)
                     and (a.ended_at is None or a.ended_at > at)]
        if len(open_then) == n:
            out.append(("match", f"count: {n} open thread(s), {n} cited concern arc(s) open then"))
        elif len(open_then) > n:
            out.append(("mismatch", f"count: {n} open thread(s), but {len(open_then)} cited concern arcs were open then"))
        else:
            out.append(("undecidable", f"count: {n} open thread(s), {len(open_then)} cited (older cues cited only some)"))
    return out


def check_sentence(
    sentence: str,
    *,
    source: str,
    at: datetime,
    cited: list[str],
    attribution: Literal["inline", "evidence_refs"],
    arcs: Mapping[str, ArcFacts],
    tz_name: str = DEFAULT_TZ,
) -> SentenceCheck:
    text = " ".join(sentence.split())[:SENTENCE_MAX_CHARS]
    missing = [a for a in cited if a not in arcs]
    found = [arcs[a] for a in cited if a in arcs]
    reasons: list[str] = [f"arc row missing: [arc:{a}]" for a in missing]
    if not found:
        return SentenceCheck(source, text, cited, attribution, "undecidable", reasons or ["no cited arc row"])
    inline = attribution == "inline"
    bare = _ANY_ARC_BRACKET_RE.sub("", sentence)
    facts: list[Fact] = []
    facts += _check_kinds(bare, found, inline)
    # A sentence that names a kind ("I slept 3 h ago") is about arcs of that kind: its time and
    # count are checked against those only, never against whatever cited arc happens to fit.
    named = {kind for pattern, kind in KIND_PHRASES if pattern.search(bare)}
    scoped = [a for a in found if a.kind in named] if named else found
    facts += _check_subject(bare, found, inline)
    if scoped:
        facts += _check_times(bare, at, scoped, inline, ZoneInfo(tz_name))
        facts += _check_counts(bare, at, scoped, inline)
    reasons += [r for _, r in facts]
    outcomes = {o for o, _ in facts}
    if "mismatch" in outcomes:
        outcome: Outcome = "mismatch"
    elif "match" in outcomes:
        outcome = "match"
    else:
        outcome = "undecidable"
        if not facts:
            reasons.append("no fact the deterministic check can read (no template phrase, time, clock or count)")
    return SentenceCheck(source, text, cited, attribution, outcome, reasons)


def _inline_ids(sentence: str) -> list[str]:
    out: list[str] = []
    for m in _ANY_ARC_BRACKET_RE.finditer(sentence):
        arc_id = (m.group(1) or m.group(2) or "").lower()
        if re.fullmatch(r"[0-9a-f]{16}", arc_id) and arc_id not in out:
            out.append(arc_id)
    return out


def _has_fact(sentence: str) -> bool:
    return any(p.search(sentence) for p, _ in KIND_PHRASES) or bool(
        AGO_RE.search(sentence) and any(_span_sec(m) is not None for m in AGO_RE.finditer(sentence))
    ) or bool(RETURNS_RE.search(sentence) or THREADS_RE.search(sentence))


def check_decision(
    *,
    at: datetime,
    cited_valid: Iterable[str],
    texts: Mapping[str, str | None],
    arcs: Mapping[str, ArcFacts],
    tz_name: str = DEFAULT_TZ,
) -> tuple[list[SentenceCheck], list[str]]:
    """Check one stance decision. ``texts``: source -> text (``imperative``, ``tone``, ``reply``).

    A sentence with an inline ``[arc:<id>]`` is checked against those arcs. A reply sentence with
    no inline id but a checkable fact is checked against the decision's cited arcs (evidence_refs
    attribution). Imperative/tone sentences without an inline id are instructions to the reply
    writer, not claims, and are skipped. Returns (checks, notes about what could not be read)."""
    at = at if at.tzinfo else at.replace(tzinfo=timezone.utc)
    cited = [c[4:] if c.startswith("arc:") else c for c in cited_valid]
    checks: list[SentenceCheck] = []
    notes: list[str] = []
    for source, text in texts.items():
        if not text:
            notes.append(f"no {source} text stored")
            continue
        for sentence in _SENTENCE_SPLIT_RE.split(text):
            if not sentence.strip():
                continue
            ids = _inline_ids(sentence)
            if ids:
                checks.append(check_sentence(sentence, source=source, at=at, cited=ids, attribution="inline",
                                             arcs=arcs, tz_name=tz_name))
            elif source == "reply" and cited and _has_fact(sentence):
                checks.append(check_sentence(sentence, source=source, at=at, cited=cited,
                                             attribution="evidence_refs", arcs=arcs, tz_name=tz_name))
    if not checks:
        notes.append("no sentence to check: no inline arc citation and no reply sentence stating a checkable day fact")
    return checks, notes


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-ISO-week counts and the weekly number: match / (match + mismatch) over decided sentences."""
    weeks: dict[str, dict[str, int]] = defaultdict(lambda: {"rows": 0, "rows_with_sentences": 0, "sentences": 0,
                                                       "match": 0, "mismatch": 0, "undecidable": 0})
    for row in rows:
        at = _dt(row["created_at"])
        assert at is not None
        year, week, _ = at.isocalendar()
        w = weeks[f"{year}-W{week:02d}"]
        w["rows"] += 1
        if row["sentences"]:
            w["rows_with_sentences"] += 1
        for s in row["sentences"]:
            w["sentences"] += 1
            w[s["outcome"]] += 1
    out = {}
    for key in sorted(weeks):
        w = dict(weeks[key])
        decided = w["match"] + w["mismatch"]
        w["truth_rate"] = round(w["match"] / decided, 3) if decided else None
        w["decided_share"] = round(decided / w["sentences"], 3) if w["sentences"] else None
        out[key] = w
    return out
