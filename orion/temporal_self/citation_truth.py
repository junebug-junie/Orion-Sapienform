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

from orion.temporal_self.citations import _ANY_ARC_BRACKET_RE, _SENTENCE_SPLIT_RE, LABEL_MIN_CHARS
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
    (re.compile(r"\bbeen here since\b", re.I), "company"),
    (re.compile(r"\b(?:strained since|was strained)\b", re.I), "strain"),
)
_SPAN = r"(?:(?P<h>\d+)\s*h(?:ours?|rs?)?\.?)?\s*(?:(?P<m>\d+)\s*min(?:utes?|s)?\.?)?"
AGO_RE = re.compile(rf"(?<![\w:])(?:(?P<under>under a minute)|{_SPAN})\s+ago\b", re.I)
FOR_RE = re.compile(rf"\bfor\s+{_SPAN}(?=[\s,.;]|$)", re.I)
CLOCK_RE = re.compile(r"(?<![\d:])([01]\d|2[0-3]):([0-5]\d)(?![\d:])(?!\s*UTC)")
RETURNS_RE = re.compile(r"\bback to it (\d+) times?\b|\b(\d+) returns?\b", re.I)
THREADS_RE = re.compile(r"\b(\d+) open threads?\b", re.I)
# A label may hold parentheses ("Juniper is talking (1 message in the last 15 min)"): stop only
# at the template's own separators.
SUBJECT_RE = re.compile(r"\battention on (.+?)(?=\s+for\s|\s*[,;\[]|\.(?:\s|$)|\s+set aside|\s+ended|$)", re.I)
# The word before an age says which time it is: "ended 44 min ago", "Asleep since 2 h ago".
AGO_FIELD = {"ended": ("ended",), "since": ("began",)}
# A word before a number that makes it a bound or an estimate, not a template value.
HEDGES: tuple[tuple[str, str], ...] = (
    ("more than", "lower"), ("over", "lower"), ("at least", "lower"),
    ("less than", "upper"), ("under", "upper"), ("within", "upper"), ("at most", "upper"),
    ("about", "approx"), ("around", "approx"), ("nearly", "approx"), ("almost", "approx"),
    ("roughly", "approx"), ("some", "approx"), ("maybe", "approx"),
)


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
    # A strain arc carried over midnight began at the boundary; its episode began earlier.
    episode_began_at: datetime | None = None

    @classmethod
    def from_row(cls, arc: Mapping[str, Any]) -> "ArcFacts":
        return cls(
            arc_id=str(arc["arc_id"]), kind=str(arc.get("kind") or ""), subject_ref=str(arc.get("subject_ref") or ""),
            subject_label=str(arc.get("subject_label") or ""), status=str(arc.get("status") or ""),
            began_at=_dt(arc["began_at"]), ended_at=_dt(arc.get("ended_at")), last_seen_at=_dt(arc.get("last_seen_at")),
            attention_returns=int(arc.get("attention_returns") or 0),
            episode_began_at=_dt(arc.get("episode_began_at")),
        )

    def times(self) -> list[tuple[str, datetime]]:
        out = [("began", self.began_at)]
        if self.ended_at is not None:
            out.append(("ended", self.ended_at))
        if self.last_seen_at is not None:
            out.append(("last seen", self.last_seen_at))
        if self.episode_began_at is not None:
            out.append(("episode began", self.episode_began_at))
        return out

    def moved_after(self, t: datetime) -> bool:
        """The row changed after ``t``: what it said at ``t`` is not stored."""
        return any(seen > t + timedelta(seconds=MIN_TOLERANCE_SEC) for name, seen in self.times() if name not in ("began", "episode began"))


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


def _quote(m: re.Match[str]) -> str:
    return "'" + m.group(0).strip().rstrip(".,;:!?") + "'"


# A fact check returns (outcome, reason).
Fact = tuple[Outcome, str]


def _moved(arcs: list[ArcFacts], at: datetime) -> bool:
    """Some cited row changed after the frame the sentence may have been written from (up to
    FRAME_LAG_SEC before ``at``): its value then is not stored, so a miss cannot be a mismatch."""
    return any(a.moved_after(at - timedelta(seconds=FRAME_LAG_SEC)) for a in arcs)


def _check_subject(sentence: str, found: list[ArcFacts], known: Mapping[str, ArcFacts],
                   inline: bool) -> tuple[list[Fact], list[tuple[int, int]]]:
    """Returns (facts, spans of the subject text, which the kind check must not read)."""
    facts: list[Fact] = []
    spans: list[tuple[int, int]] = []
    attention = [a for a in found if a.kind == "attention"]
    lower = sentence.lower()
    for m in re.finditer(r"\battention on\s+", sentence, re.I):
        start = m.end()
        # 1) A cited label written out after "attention on": a match, whatever separators it holds.
        hit = next((a for a in attention for name in (a.subject_label.strip(), a.subject_ref)
                    if len(name) >= LABEL_MIN_CHARS and lower.startswith(name.lower(), start)), None)
        if hit is not None:
            name = hit.subject_label.strip() if lower.startswith(hit.subject_label.strip().lower(), start) and \
                hit.subject_label.strip() else hit.subject_ref
            spans.append((start, start + len(name)))
            facts.append(("match", f"subject: '{name}' is the label of [arc:{hit.arc_id}]"))
            continue
        sm = SUBJECT_RE.match(sentence, m.start())
        if sm is None:
            continue
        said = sm.group(1).strip()
        spans.append((sm.start(1), sm.end(1)))
        key = said.lower()
        if not attention:
            continue
        if any(len(n) >= LABEL_MIN_CHARS and (n.lower().startswith(key) or n.lower() in key)
               for a in attention for n in (a.subject_label.strip(), a.subject_ref)):
            facts.append(("match", f"subject: '{said}' names a cited attention arc's label"))
            continue
        other = next((a for a in known.values() if a.kind == "attention" and a not in attention
                      and key in {a.subject_label.strip().lower(), a.subject_ref.lower()}), None)
        if other is not None and inline:
            facts.append(("mismatch", f"subject: '{said}' is the label of another arc [arc:{other.arc_id}], "
                                      "not of a cited one"))
        else:
            facts.append(("undecidable", f"subject: '{said}' is not a cited attention arc's label (may be a paraphrase)"))
    return facts, spans


def _check_instant(what: str, claimed: datetime, span: float, at: datetime, arcs: list[ArcFacts],
                   fields: Iterable[str] | None, decisive: bool, hedge: str | None = None) -> Fact:
    tol = _tolerance(span)
    candidates = [(a, name, t) for a in arcs for name, t in a.times() if fields is None or name in fields]
    for a, name, t in candidates:
        d = (t - claimed).total_seconds()  # > 0: the record is newer than the claim
        if hedge == "lower":  # "over 2 h ago": happened at or before the claimed instant
            ok = d <= tol + FRAME_LAG_SEC
        elif hedge == "upper":  # "under 10 min ago": at or after it
            ok = d >= -tol
        elif hedge == "approx":
            ok = -2 * tol <= d <= 2 * tol + FRAME_LAG_SEC
        else:
            # A claim may read older than the row by the frame's lag (the cue saw an older frame).
            ok = -tol <= d <= tol + FRAME_LAG_SEC
        if ok:
            return ("match", f"time: {what} -> {claimed:%H:%MZ}, [arc:{a.arc_id}] {name} {t:%H:%MZ}")
    if hedge is not None:
        return ("undecidable", f"time: {what} is hedged and fits no cited arc time")
    if _moved(arcs, at):
        return ("undecidable", f"time: {what} -> {claimed:%H:%MZ}; a cited arc's row moved after the frame the "
                               "sentence may have read, its value then is not stored")
    if not decisive:
        return ("undecidable", f"time: {what} -> {claimed:%H:%MZ} matches no cited arc time (attribution is not "
                               "one-to-one: evidence_refs, or a cited arc of a kind the sentence does not name)")
    near = min(candidates, key=lambda c: abs((c[2] - claimed).total_seconds()), default=None)
    detail = (f"; nearest [arc:{near[0].arc_id}] {near[1]} {near[2]:%H:%MZ}, off by {_fmt((near[2] - claimed).total_seconds())}"
              if near else "")
    return ("mismatch", f"time: {what} -> {claimed:%H:%MZ} matches no cited arc time{detail}")


def _hedge(sentence: str, start: int) -> str | None:
    before = " ".join(sentence[:start].lower().split()[-2:])
    for words, kind in HEDGES:
        if before.endswith(words):
            return kind
    return None


def _check_times(sentence: str, at: datetime, arcs: list[ArcFacts], decisive: bool, tz: ZoneInfo) -> list[Fact]:
    out: list[Fact] = []
    for m in AGO_RE.finditer(sentence):
        span = _span_sec(m)
        if span is None:
            continue
        before = sentence[:m.start()].split()
        fields = AGO_FIELD.get(before[-1].lower()) if before else None
        hedge = None if m.groupdict().get("under") else _hedge(sentence, m.start())
        out.append(_check_instant(_quote(m), at - timedelta(seconds=span), span, at, arcs, fields, decisive, hedge))
    for m in FOR_RE.finditer(sentence):
        span = _span_sec(m)
        if span is None:
            continue
        # Only the templates' "attention on X for N" / "open for N" mean "since the arc began".
        head = sentence[:m.start()].lower()
        template = bool(re.search(r"\battention on\b[^.;]*$|\bopen$", head.rstrip()))
        out.append(_check_instant(_quote(m) + (" (start)" if template else ""), at - timedelta(seconds=span), span,
                                  at, arcs, ("began",) if template else None, decisive and template,
                                  _hedge(sentence, m.start() + len("for "))))
    now_local = at.astimezone(tz)
    for m in CLOCK_RE.finditer(sentence):
        said = int(m.group(1)) * 60 + int(m.group(2))

        def off(t: datetime) -> int:
            local = t.astimezone(tz)
            d = abs(local.hour * 60 + local.minute - said)
            return min(d, 1440 - d)

        hits = [(a, name) for a in arcs for name, t in a.times() if off(t) <= CLOCK_TOLERANCE_MIN]
        lag_min = int(FRAME_LAG_SEC // 60) + CLOCK_TOLERANCE_MIN
        is_now = any(off(now_local - timedelta(minutes=k)) <= CLOCK_TOLERANCE_MIN for k in range(0, lag_min + 1))
        if hits:
            a, name = hits[0]
            out.append(("match", f"clock: {m.group(0)} local is [arc:{a.arc_id}] {name}"))
        elif is_now:
            out.append(("undecidable", f"clock: {m.group(0)} local is the time of writing or of the frame, not an arc time"))
        elif decisive and len(arcs) == 1 and not _moved(arcs, at):
            out.append(("mismatch", f"clock: {m.group(0)} local is none of [arc:{arcs[0].arc_id}]'s times"))
        else:
            out.append(("undecidable", f"clock: {m.group(0)} local matches no cited arc time "
                                       "(a clock may name something other than the arc)"))
    return out


def _check_counts(sentence: str, at: datetime, arcs: list[ArcFacts], decisive: bool) -> list[Fact]:
    out: list[Fact] = []
    for m in RETURNS_RE.finditer(sentence):
        n = int(m.group(1) or m.group(2))
        have = sorted(a.attention_returns for a in arcs)
        if n in have:
            out.append(("match", f"count: {n} return(s) equals a cited arc's attention_returns"))
        elif decisive and all(n > a.attention_returns for a in arcs):
            # Returns only grow, and a lagging frame only shows fewer: above every current count was never true.
            out.append(("mismatch", f"count: {n} return(s); cited arcs have {have}"))
        elif decisive and not _moved(arcs, at):
            out.append(("mismatch", f"count: {n} return(s); cited arcs have {have}"))
        else:
            out.append(("undecidable", f"count: {n} return(s); cited arcs have {have} now "
                                       "(rows moved since the frame, or attribution is not one-to-one)"))
    for m in THREADS_RE.finditer(sentence):
        n = int(m.group(1))
        open_then = [a for a in arcs if a.kind == "concern" and a.began_at <= at + timedelta(seconds=MIN_TOLERANCE_SEC)
                     and (a.ended_at is None or a.ended_at > at)]
        if len(open_then) == n:
            out.append(("match", f"count: {n} open thread(s), {n} cited concern arc(s) open then"))
        elif decisive and len(open_then) > n:
            out.append(("mismatch", f"count: {n} open thread(s), but {len(open_then)} cited concern arcs were open then"))
        else:
            out.append(("undecidable", f"count: {n} open thread(s), {len(open_then)} cited concern arc(s) open then"))
    return out


def _blank(text: str, spans: list[tuple[int, int]]) -> str:
    chars = list(text)
    for a, b in spans:
        for i in range(a, b):
            chars[i] = " "
    return "".join(chars)


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
    facts, subject_spans = _check_subject(bare, found, arcs, inline)
    # Kind phrases inside a subject label ("attention on curiosity runs backlog") name no kind.
    kind_text = _blank(bare, subject_spans)
    named = {kind: pattern.search(kind_text) for pattern, kind in KIND_PHRASES}
    named = {k: m for k, m in named.items() if m is not None}
    cited_kinds = {a.kind for a in found}
    scoped = [a for a in found if a.kind in named] if named else found
    has_measure = bool([m for m in AGO_RE.finditer(kind_text) if _span_sec(m) is not None]
                       or [m for m in FOR_RE.finditer(kind_text) if _span_sec(m) is not None]
                       or RETURNS_RE.search(kind_text) or THREADS_RE.search(kind_text))
    if named:
        phrases = ", ".join(f"'{m.group(0)}'" for m in named.values())
        if set(named) & cited_kinds:
            facts.append(("match", f"kind: {phrases} names {sorted(named)}, and a cited arc is one"))
        elif inline and len(named) == 1 and has_measure:
            # "Last slept 2 h ago [arc:<attention>]": a timed or counted claim about a kind no cited arc is.
            facts.append(("mismatch", f"kind: {phrases} names {sorted(named)}; cited arcs are {sorted(cited_kinds)}"))
        else:
            facts.append(("undecidable", f"kind: {phrases} names {sorted(named)}; cited arcs are {sorted(cited_kinds)} "
                                         "(ordinary words, no timed claim: not decisive)"))
    # One-to-one: every cited arc is of a kind the sentence names (or it names none).
    decisive = inline and (not named or cited_kinds <= set(named))
    if scoped:
        facts += _check_times(kind_text, at, scoped, decisive, ZoneInfo(tz_name))
        facts += _check_counts(kind_text, at, scoped, decisive)
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
