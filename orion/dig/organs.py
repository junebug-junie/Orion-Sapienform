"""`dig organs <date> [<end>]`: the daily dig roundup -- what each organ did on one Denver day. Read-only.

Spec: docs/superpowers/specs/2026-11-15-orion-integration-and-a-life-design.md, Part A.7. One roundup per
day, no model call. The organ registry (``config/integration/organs.yaml``) names the organs; for each one:

* what it did: experiences recorded (``temporal_self_event``, per source kind) and acts opened and closed
  (``grammar_acts``, its families = act-ledger actors), and its 2 highest-salience experiences quoted with
  their refs (``experience:<id>``; ``act:<id>`` when the row is an act);
* what it read from other organs and cited (``experience_use``, quoted, with refs), and how often it read
  the now and cited nothing (``experience_shown``) -- versus "did not read" (no ``experience_shown`` row);
* who cited it, and closed (another organ used one of its experiences) or write-only;
* breaks: 0 of a kind where the 14 days before had a median of at least 1 a day.

Then the day's experience use rate and closed organs -- the same SQL (``orion/integration/coverage.py``) and the
same arithmetic (``orion.temporal_self.now.use_rate``) as ``scripts/report_integration_coverage.py`` -- and the
notification counts that used to be the "Orion Daily Digest" email (retired with this roundup).

The rule: every line is a count, a quote or a ref (``check_lines``; the renderer runs it on its own output, so a
summary line cannot ship). A quote is the row's own label (``now_label``: the label the now shows) or the
registry's own words.

Two readers share the SQL and the pure ``build``: the dig CLI (ReadOnlyPsql, CSV strings) and the day letter's
gather (``orion/orion_day/gather.py``, an asyncpg connection, typed values). A query that fails is named in an
``UNREAD`` line, never shown as an empty organ.
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime, timedelta, timezone
from typing import Any, Callable, Iterable, Mapping, Optional
from zoneinfo import ZoneInfo

from orion.integration.coverage import NOW_READS_SQL, RATE_CITES_MADE_SQL, RATE_KINDS_SQL, RATE_USES_SQL
from orion.temporal_self.now import OrganRegistry, now_label, use_rate

ZONE = ZoneInfo("America/Denver")
ROUNDUP_FROM = date(2026, 10, 9)  # first chronicled Temporal Self day (== orion.dig.days.TEMPORAL_SELF_FROM)
MAX_DAYS = 7  # `dig organs <start> <end>`: a week of roundups joined
TOP_PER_ORGAN = 2
CITED_SHOWN = 3
HISTORY_DAYS = 14
QUOTE_CHARS = 160
NOTIFY_TITLES = 3
# The Prime thought trace (integration Part B, patch 5): each thought's outcome_note starts
# "woke: need:<name> <ref>". Until patch 5 ships, notes start "woke: quiet: ..." (the 6-hour floor) or
# "woke: <a change>"; both are counted, so the roundup counts thoughts per need the day patch 5 lands.
NEED_PREFIX = "need:"

# The dream experiment's blind rule (orion/schemas/orion_day.py, orion_day_email.py): a hypothesis id is the
# dream scorecard's join key and is never shown. A dream_hypothesis event's id IS ``dream_hypothesis:<id>``, so
# those rows are counted and quoted but never printed with a ref.
BLIND_KINDS = frozenset({"dream_hypothesis"})
BLIND_REF = "(ref withheld: dream experiment blind rule)"
_BLIND_ID_RE = re.compile(r"dream_hypothesis:(?!\(withheld\))[^\]\s\"']+")

# Refs this roundup prints. The letter's ref gate (orion/orion_day/budget.py _REF_RE) knows both kinds.
REF_RE = re.compile(r"\[(?:experience|act):[^\]\s]+\]")
_QUOTE_RE = re.compile(r"\"[^\"]+\"|“[^”]+”")
_DIGIT_RE = re.compile(r"\d")


# --------------------------------------------------------------------------- the line rule


def line_kind(line: str) -> Optional[str]:
    """``ref`` / ``quote`` / ``count`` for a line the roundup may print; None for a line it may not (a
    summary with no number, no quoted words and no ref)."""
    if REF_RE.search(line):
        return "ref"
    if _QUOTE_RE.search(line):
        return "quote"
    if _DIGIT_RE.search(line):
        return "count"
    return None


def check_lines(lines: Iterable[str]) -> None:
    """Raise ValueError on the first line that is not a count, a quote or a ref (blank lines are layout)."""
    for i, line in enumerate(lines):
        if line.strip() and line_kind(line) is None:
            raise ValueError(f"roundup line {i} is not a count, a quote or a ref: {line[:120]!r}")


def _q(text: Any, n: int = QUOTE_CHARS) -> str:
    """Quote a row's own words: one line, cut to ``n``, inner double quotes made single."""
    t = " ".join(str(text or "").split()).replace('"', "'")
    if len(t) > n:
        t = t[: n - 1].rstrip() + "…"
    return f'"{t}"'


# --------------------------------------------------------------------------- the window and the SQL


def day_bounds(day: date) -> tuple[str, str]:
    """The Denver day as a half-open UTC window, ISO strings (safe inside a SQL literal: built from a date)."""
    lo = datetime.combine(day, dtime(0), tzinfo=ZONE).astimezone(timezone.utc)
    hi = datetime.combine(day + timedelta(days=1), dtime(0), tzinfo=ZONE).astimezone(timezone.utc)
    return lo.isoformat(), hi.isoformat()


def _bind(sql: str, **bounds: str) -> str:
    """The coverage report's psycopg-param SQL with the bounds written in, so both read one definition."""
    for k, v in bounds.items():
        sql = sql.replace(f"%({k})s", f"'{v}'::timestamptz")
    return sql


def queries(day: date) -> dict[str, str]:
    """name -> read-only SQL for one Denver day."""
    a, b = day_bounds(day)
    h, _ = day_bounds(day - timedelta(days=HISTORY_DAYS))
    win = f"occurred_at >= '{a}'::timestamptz AND occurred_at < '{b}'::timestamptz"
    acts_win = f"opened_at >= '{a}'::timestamptz AND opened_at < '{b}'::timestamptz"
    # notify_requests.created_at is a timestamp without time zone holding UTC
    nwin = (f"created_at >= ('{a}'::timestamptz AT TIME ZONE 'UTC') "
            f"AND created_at < ('{b}'::timestamptz AT TIME ZONE 'UTC')")
    return {
        "kinds": _bind(RATE_KINDS_SQL, lo=a, hi=b),
        "uses": _bind(RATE_USES_SQL, lo=a, hi=b),
        "cites_made": _bind(RATE_CITES_MADE_SQL, lo=a, hi=b),
        "reads": _bind(NOW_READS_SQL, lo=a, hi=b),
        "top": f"""
SELECT event_id, source_kind, coalesce(source_table, '') AS source_table, coalesce(source_ref, '') AS source_ref,
       coalesce(label, '') AS label, coalesce(verdict, '') AS verdict, salience, occurred_at,
       coalesce(payload_json::text, '{{}}') AS payload
  FROM (SELECT e.*, row_number() OVER (PARTITION BY source_kind
                                       ORDER BY salience DESC NULLS LAST, occurred_at DESC, event_id) AS rn
          FROM temporal_self_event e WHERE {win}) t
 WHERE rn <= {TOP_PER_ORGAN}""",
        "history": f"""
SELECT source_kind, ((occurred_at AT TIME ZONE 'America/Denver')::date)::text AS day, count(*) AS n
  FROM temporal_self_event WHERE occurred_at >= '{h}'::timestamptz AND occurred_at < '{a}'::timestamptz
 GROUP BY 1, 2""",
        "acts": f"""
SELECT actor, verb, count(*) AS opened, count(closed_at) AS closed
  FROM grammar_acts WHERE {acts_win} GROUP BY 1, 2""",
        "prime_wakes": f"""
SELECT act_id, coalesce(outcome_note, '') AS outcome_note
  FROM grammar_acts WHERE actor = 'prime' AND verb = 'infer' AND {acts_win} ORDER BY opened_at""",
        "cited": f"""
SELECT u.event_id, u.citing_organ, coalesce(u.citing_ref, '') AS citing_ref, e.source_kind,
       coalesce(e.label, '') AS label, coalesce(e.verdict, '') AS verdict,
       coalesce(e.payload_json::text, '{{}}') AS payload
  FROM experience_use u JOIN temporal_self_event e ON e.event_id = u.event_id
 WHERE u.cited_at >= '{a}'::timestamptz AND u.cited_at < '{b}'::timestamptz
 ORDER BY u.cited_at""",
        "notify": f"""
SELECT severity, status, count(*) AS n FROM notify_requests WHERE {nwin} GROUP BY 1, 2""",
        "notify_titles": f"""
SELECT severity, title, count(*) AS n FROM notify_requests
 WHERE {nwin} AND severity IN ('critical', 'error')
 GROUP BY 1, 2 ORDER BY 3 DESC, 2 LIMIT {NOTIFY_TITLES}""",
    }


# --------------------------------------------------------------------------- row coercion (CSV strings or typed)


def _int(v: Any) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return 0


def _float(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _payload(v: Any) -> Mapping[str, Any]:
    if isinstance(v, Mapping):
        return v
    try:
        got = json.loads(v or "{}")
    except (TypeError, ValueError):
        return {}
    return got if isinstance(got, Mapping) else {}


def _label(r: Mapping[str, Any]) -> str:
    return now_label(str(r.get("source_kind") or ""), r.get("label") or None, r.get("verdict") or None,
                     _payload(r.get("payload")))


def _ref(r: Mapping[str, Any]) -> str:
    return BLIND_REF if str(r.get("source_kind")) in BLIND_KINDS else f"[experience:{r['event_id']}]"


def _when(v: Any) -> float:
    """occurred_at as epoch seconds, from an asyncpg datetime or a psql text timestamp (same order both ways)."""
    if isinstance(v, datetime):
        return (v if v.tzinfo else v.replace(tzinfo=timezone.utc)).timestamp()
    t = str(v or "").strip().replace(" ", "T", 1)
    if re.search(r"[+-]\d\d$", t):
        t += ":00"
    try:
        d = datetime.fromisoformat(t)
    except ValueError:
        return 0.0
    return (d if d.tzinfo else d.replace(tzinfo=timezone.utc)).timestamp()


def wake_reason(note: str) -> str:
    """``need:<name>`` / ``quiet floor`` / ``a change`` / ``still open`` from a Prime thought's outcome_note."""
    note = (note or "").strip()
    if not note:
        return "still open"
    rest = note[len("woke:"):].strip() if note.lower().startswith("woke:") else note
    if rest.startswith(NEED_PREFIX):
        name = rest[len(NEED_PREFIX):].split()[0] if rest[len(NEED_PREFIX):].split() else "?"
        return f"{NEED_PREFIX}{name.rstrip(':,;')}"
    if rest.lower().startswith("quiet"):
        return "quiet floor"
    return "a change"


# --------------------------------------------------------------------------- the roundup


@dataclass
class Roundup:
    day: str
    lines: list[str] = field(default_factory=list)
    refs: list[str] = field(default_factory=list)
    unread: dict[str, str] = field(default_factory=dict)
    use_rate: Optional[float] = None
    closed_organs: list[str] = field(default_factory=list)
    experiences: int = 0
    organ_experiences: dict[str, int] = field(default_factory=dict)  # organ -> experiences that day


def _organ_families(registry: OrganRegistry) -> dict[str, str]:
    out: dict[str, str] = {}
    for organ, spec in registry.raw.items():
        for fam in (spec or {}).get("families") or []:
            out[str(fam)] = str(organ)
    return out


def _counts(pairs: Iterable[tuple[str, int]]) -> str:
    return ", ".join(f"{k} {n}" for k, n in sorted(pairs, key=lambda kv: (-kv[1], kv[0])) if n)


def build(day: date, rows: Mapping[str, list[Mapping[str, Any]]], registry: OrganRegistry,
          unread: Optional[Mapping[str, str]] = None) -> Roundup:
    """Pure: the roundup from the query rows (missing names read as no rows; ``unread`` names why)."""
    unread = dict(unread or {})
    a, b = day_bounds(day)
    lo, hi = datetime.fromisoformat(a), datetime.fromisoformat(b)
    kinds = {str(r["source_kind"]): _int(r["n"]) for r in rows.get("kinds", [])}
    uses = [(str(r["event_id"]), str(r["source_kind"]), str(r["citing_organ"])) for r in rows.get("uses", [])]
    made = {str(r["citing_organ"]): _int(r["n"]) for r in rows.get("cites_made", [])}
    reads = {str(r["reader_organ"]): (_int(r["reads"]), _int(r["reads_no_cite"]), _int(r.get("reads_unclosed")))
             for r in rows.get("reads", [])}
    rate = use_rate(kinds, uses, window_start=lo, window_end=hi, registry=registry, cites_made=made)
    per_organ = {o.organ: o for o in rate.organs}

    top: dict[str, list[Mapping[str, Any]]] = {}
    for r in rows.get("top", []):
        organ = registry.organ_of.get(str(r["source_kind"]))
        if organ:
            top.setdefault(organ, []).append(r)

    # 14 days before, zero-filled, per kind: a break is 0 today where the median day had >= 1
    hist: dict[str, dict[str, int]] = {}
    for r in rows.get("history", []):
        hist.setdefault(str(r["source_kind"]), {})[str(r["day"])[:10]] = _int(r["n"])
    days_before = [(day - timedelta(days=i)).isoformat() for i in range(1, HISTORY_DAYS + 1)]
    usual = {k: statistics.median([d.get(x, 0) for x in days_before]) for k, d in hist.items()}

    fam_organ = _organ_families(registry)
    acts: dict[str, list[tuple[str, int, int]]] = {}
    unclaimed: dict[str, int] = {}
    for r in rows.get("acts", []):
        actor, verb = str(r["actor"]), str(r["verb"])
        organ = fam_organ.get(actor)
        if organ is None:
            unclaimed[actor] = unclaimed.get(actor, 0) + _int(r["opened"])
        else:
            acts.setdefault(organ, []).append((f"{actor} {verb}", _int(r["opened"]), _int(r["closed"])))

    cited_by_organ: dict[str, list[Mapping[str, Any]]] = {}  # citing organ -> its valid cites today
    cited_of: dict[str, dict[str, int]] = {}  # cited organ -> citing organ -> cites today
    for r in rows.get("cited", []):
        citing = str(r["citing_organ"])
        cited_by_organ.setdefault(citing, []).append(r)
        owner = registry.organ_of.get(str(r["source_kind"]))
        if owner and owner != citing:
            cited_of.setdefault(owner, {})
            cited_of[owner][citing] = cited_of[owner].get(citing, 0) + 1

    wakes: dict[str, int] = {}
    for r in rows.get("prime_wakes", []):
        w = wake_reason(str(r.get("outcome_note") or ""))
        wakes[w] = wakes.get(w, 0) + 1

    out = Roundup(day=day.isoformat(), unread=unread, use_rate=rate.rate, closed_organs=list(rate.closed_organs),
                  experiences=rate.experiences, organ_experiences={o.organ: o.experiences for o in rate.organs})
    L = out.lines
    L.append(f"organs {day.isoformat()} (Denver; {a[:16]}Z to {b[:16]}Z): {len(registry.organs)} organs, "
             f"{rate.experiences} experiences")
    quiet: list[str] = []
    for organ in registry.organs:
        spec = registry.raw.get(organ) or {}
        o = per_organ[organ]
        organ_kinds = registry.kinds_of(organ)
        organ_acts = acts.get(organ, [])
        r_n, r_none, r_open = reads.get(organ, (0, 0, 0))
        mine = cited_by_organ.get(organ, [])
        if not (o.experiences or organ_acts or r_n or mine or o.used_by_other):
            quiet.append(organ)
            continue
        head = f"{organ}: {o.experiences} experiences"
        kc = _counts((k, kinds.get(k, 0)) for k in organ_kinds)
        if kc:
            head += f" ({kc})"
        if organ_acts:
            head += (f"; {sum(x[1] for x in organ_acts)} acts opened, {sum(x[2] for x in organ_acts)} closed "
                     f"({_counts((x[0], x[1]) for x in organ_acts)})")
        L.append(head)
        # highest salience first; unscored after scored (the now's own order, orion/temporal_self/now.py)
        ranked = sorted(top.get(organ, []), key=lambda r: str(r.get("event_id")))
        ranked.sort(key=lambda r: _when(r.get("occurred_at")), reverse=True)
        ranked.sort(key=lambda r: (_float(r.get("salience")) is None, -(_float(r.get("salience")) or 0.0)))
        for r in ranked[:TOP_PER_ORGAN]:
            ref = _ref(r)
            sal = _float(r.get("salience"))
            line = f"  {_q(_label(r))} {ref} " + (f"salience {sal:.2f}" if sal is not None else "salience none")
            if r.get("source_table") == "grammar_acts" and r.get("source_ref") and r["source_kind"] not in BLIND_KINDS:
                line += f" [act:{r['source_ref']}]"  # the source row, when it is an act `dig act` opens
            L.append(line)
        if organ == "prime" and wakes:
            L.append(f"  thoughts by what woke them: {_counts(wakes.items())} "
                     f"(a need counts once Prime writes \"woke: {NEED_PREFIX}<name>\")")
        if spec.get("reads_now"):
            # #3498: a read the organ never closed is one its model never saw (a refusal, a timeout), not a choice
            L.append(f"  read the now {r_n} times; {r_none} of those reads cited nothing; {r_open} never closed "
                     f"(no model call used them); {o.cites_made} valid cites")
        else:
            L.append(f"  did not read the now (registry: {_q(spec.get('exempt_read') or 'no reason given')})")
        for r in mine[:CITED_SHOWN]:
            owner = registry.organ_of.get(str(r["source_kind"]), "?")
            L.append(f"  cited from {owner}: {_q(_label(r))} {_ref(r)}")
        if len(mine) > CITED_SHOWN:
            L.append(f"  {len(mine) - CITED_SHOWN} more cites today")
        by = cited_of.get(organ, {})
        if o.used_by_other:
            L.append(f"  closed: {o.used_by_other} of its {o.experiences} experiences cited by another organ"
                     + (f" (today's cites by {_counts(by.items())})" if by else ""))
        else:
            L.append(f"  write-only: 0 of its {o.experiences} experiences cited by another organ")
        for k in organ_kinds:
            if kinds.get(k, 0) == 0 and usual.get(k, 0) >= 1:
                L.append(f"  break: 0 {k} today; usual {usual[k]:g} a day (median of the {HISTORY_DAYS} days before)")
    if quiet:
        L.append(f"quiet organs (0 experiences, 0 acts, 0 reads, 0 cites): {len(quiet)} ({', '.join(quiet)})")
    rate_txt = "n/a" if rate.rate is None else f"{rate.rate:.2%}"
    L.append(f"experience use rate: {rate.used_by_other} / {rate.experiences} = {rate_txt} (cited by another organ "
             f"within 24 h); closed organs: {len(rate.closed_organs)} of {rate.registered_organs}"
             + (f" ({', '.join(rate.closed_organs)})" if rate.closed_organs else ""))
    if rate.unregistered_kinds:
        L.append(f"unregistered kinds (no organ claims them): {_counts(rate.unregistered_kinds.items())}")
    if unclaimed:
        L.append(f"acts by actors no organ claims: {_counts(unclaimed.items())}")
    sev: dict[str, int] = {}
    status: dict[str, int] = {}
    for r in rows.get("notify", []):
        sev[str(r["severity"])] = sev.get(str(r["severity"]), 0) + _int(r["n"])
        status[str(r["status"])] = status.get(str(r["status"]), 0) + _int(r["n"])
    L.append(f"notifications (the retired Daily Digest's counts): {sum(sev.values())} requests"
             + (f"; by severity {_counts(sev.items())}" if sev else "")
             + (f"; by status {_counts(status.items())}" if status else ""))
    for r in rows.get("notify_titles", []):
        L.append(f"  {r['severity']} x{_int(r['n'])}: {_q(r['title'])}")
    for name, why in sorted(unread.items()):
        L.append(f"UNREAD {name}: {_q(why, 200)} (that part is missing, not empty)")
    # The blind rule on the output itself: a hypothesis id quoted inside some other row's label is withheld too.
    L[:] = [_BLIND_ID_RE.sub("dream_hypothesis:(withheld)", x) for x in L]
    out.lines = L
    check_lines(L)
    seen: dict[str, None] = {}
    for line in L:
        for m in REF_RE.finditer(line):
            seen.setdefault(m.group(0)[1:-1], None)
    out.refs = list(seen)
    return out


# --------------------------------------------------------------------------- readers


def read_sync(day: date, fetch: Callable[[str], list[Mapping[str, Any]]], registry: OrganRegistry) -> Roundup:
    """The dig CLI's reader: ``fetch(sql)`` returns rows or raises (ReadOnlyPsql through psql_rows)."""
    rows: dict[str, list[Mapping[str, Any]]] = {}
    unread: dict[str, str] = {}
    for name, sql in queries(day).items():
        try:
            rows[name] = fetch(sql)
        except Exception as exc:  # noqa: BLE001 -- named in an UNREAD line, never shown as empty
            unread[name] = f"{type(exc).__name__}: {exc}"[:300]
    return build(day, rows, registry, unread)


async def read_async(conn: Any, day: date, registry: OrganRegistry) -> Roundup:
    """The day letter's reader: an asyncpg-style connection (``fetch``)."""
    rows: dict[str, list[Mapping[str, Any]]] = {}
    unread: dict[str, str] = {}
    for name, sql in queries(day).items():
        try:
            rows[name] = [dict(r) for r in await conn.fetch(sql)]
        except Exception as exc:  # noqa: BLE001
            unread[name] = f"{type(exc).__name__}: {exc}"[:300]
    return build(day, rows, registry, unread)


def check_window(start: date, end: date) -> None:
    if end < start:
        raise ValueError(f"organs: {end} is before {start}")
    if end < ROUNDUP_FROM:
        raise ValueError(f"organs: no Temporal Self record before {ROUNDUP_FROM}")
    if (end - start).days + 1 > MAX_DAYS:
        raise ValueError(f"organs: at most {MAX_DAYS} days at a time, got {(end - start).days + 1}")


__all__ = ["Roundup", "build", "check_lines", "check_window", "day_bounds", "line_kind", "queries", "read_async",
           "read_sync", "wake_reason"]
