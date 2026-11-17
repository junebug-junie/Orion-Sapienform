#!/usr/bin/env python3
"""The promise ledger in shadow: what opened, what closed and how, and a blinded precision sample. Read-only.

Spec: docs/superpowers/specs/2026-11-11-intention-ledger-design.md (D2 and the 2026-11-16 amendments).
Issue #3531 reads this before any promise act counts toward a track record or reaches stance.

Over the last ``--days`` (default 28) it reports:

* promise acts (``grammar_acts`` actor ``hub.promise``, signal ``promise_kept``): open, kept (by the
  kind of evidence that closed it), expired, void (no Juniper chat turn in the horizon: censored);
* un-closeable: expired acts the judge was never shown at any Juniper turn in their window (more
  open promises than the judge's view, ``PROMISE_JUDGE_VIEW_MAX``). They are not broken promises.
  Approximate: turn times here are ``chat_history_log.created_at`` (when the reply was stored), not
  the ingress time the resolver uses, so a turn at the very edge of a window can fall either side;
* expired acts whose promising turn got an intention carry reply in the window (Prime took the
  promise on and delivered something; the ledger does not count a carry as a keep in v1), with their act ids;
* keep rate kept / (kept + expired), void and un-closeable left out, with a Beta(1, 1) posterior;
* the self-reinforcement guard: firm promises per JUDGED Juniper turn (judge model pass, not the
  quick gate), with a 90% Poisson score interval, against the frozen replay baseline (``REPLAY_BASELINE``);
  red above 2x. Stored items with no
  ``force``/``done_when`` (written before #3532) are skipped and counted, never guessed;
* ``--sample N``: a blinded sample of stored promise items for hand labels. Each item shows ONLY the
  quote and the reply around it, never the judge's doer/force/act/done_when, so the labeller cannot
  lean on the judge's own reading (amendment 2). Writes a label template with hash keys;
  ``--score FILE`` reads it back and prints precision with a Wilson 95% interval.

    ORION_LIVE_DSN=postgresql://... python3 scripts/report_promise_ledger.py [--days 28] [--json]
    ... --sample 60 --out /tmp/promise-ledger-labels.json      # blinded labelling sheet
    ... --score /tmp/promise-ledger-labels.json                # precision from filled labels
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.grammar.promise_acts import (  # noqa: E402
    PROMISE_ACTOR,
    PROMISE_SIGNAL,
    kept_evidence_kind,
    parse_promise_ref,
    promise_close_kind,
)
from orion.harness.promise_gate import normalize_for_quote_match  # noqa: E402
from orion.regulation.juniper_turns import JUNIPER_INGRESS_PREDICATE, JUNIPER_TURN_PREDICATE  # noqa: E402
from orion.schemas.harness_finalize import (  # noqa: E402
    PROMISE_DONE_WHEN_MIN_CHARS,
    PROMISE_FORCE_COMMITS,
    PROMISE_JUDGE_VIEW_MAX,
)

# Replay of the deployed finder (#3532 rules, same model and lane as live), judged turns only:
# dev 10-13..11-12 (179 judged, 15 firm promises), held-out 09-13..10-13 (86 judged, 0) and
# 11-12..11-16 (24 judged, 8). Firm = gated item with offer false. 23 / 289 = 0.080 per judged turn.
# FROZEN for the whole shadow period (Orion's PR #3561 question 3): set 2026-11-16, before any live
# promise act existed, and pinned by tests/test_report_promise_ledger.py. It is never re-based on
# shadow data: a baseline moved after seeing the rate it judges is no guard. Changing it is a
# Juniper decision recorded in #3531, not an edit here. The slices are uneven (0/86 to 8/24), so the
# report prints a 90% interval on the live rate beside the ratio; red still means ratio > 2x.
REPLAY_BASELINE = (23, 289)
# When orion-harness-governor went live with the #3532 force/done_when rules (post-pull rebuild at
# head d198f79ed, ~/.orion/post_pull_rebuild.log). A judged turn before this could store no firm item,
# so the guard leaves those turns out of its denominator (they would pull the live rate toward 0).
FINDER_LIVE_AT = datetime(2026, 11, 16, 18, 5, tzinfo=timezone.utc)
GUARD_RATIO_RED = 2.0
# The guard cannot go red before this many judged turns (Orion's PR #3561 second-round question 3).
# At the frozen rate 0.0796, 60 turns expect 4.8 firm promises; red (ratio > 2) then needs 10 or more,
# which pure noise gives about 2.4% of the time (Poisson). At 50 turns red needs 8 (about 5%), at 20
# turns 4 (about 8%).
GUARD_MIN_JUDGED_TURNS = 60
# The keep rate's ceiling until #3595 lands: the share of firm promises a later chat reply could show
# kept. Hand count of the 23 firm replay promises (one reader, not blind; #3595): 9 chat-showable,
# 8 delivered by a run or outside chat, 6 abstentions or standing states. A keep rate near this
# number is the most the reply-only close can show, not a sign of broken promises.
KEEP_RATE_CEILING = (9, 23)
# Two firm promises in different turns of one session, open at the same time, that may be one
# promise said twice in other words: same act or done_when text, or this many shared characters.
DUPLICATE_SPAN_CHARS = 24
JUNIPER_SAID_CHARS = 280
CARRY_SOURCE = "intention_carry"
CONTEXT_CHARS = 280

ACTS_SQL = """
SELECT act_id, test_before, correlation_id, opened_at, horizon_at, closed_at, outcome_source, outcome_note
  FROM grammar_acts
 WHERE actor = %(actor)s AND signal_id = %(signal)s AND opened_at >= %(since)s
 ORDER BY opened_at
"""
JUNIPER_STARTS_SQL = f"""
SELECT created_at FROM chat_history_log WHERE {JUNIPER_TURN_PREDICATE} AND created_at >= %(since)s ORDER BY created_at
"""
def qualify(predicate: str, alias: str, columns: Sequence[str]) -> str:
    """``predicate`` with each bare column name prefixed by ``alias.`` (whole words only, never inside a
    quoted string). Fails loudly if a column is missing, so a changed predicate cannot be mis-qualified."""
    parts = re.split(r"('(?:[^']|'')*')", predicate)
    for col in columns:
        pat = re.compile(rf"(?<![\w.]){re.escape(col)}\b")
        if not any(pat.search(x) for i, x in enumerate(parts) if i % 2 == 0):
            raise ValueError(f"column {col!r} not in predicate {predicate!r}")
        parts = [x if i % 2 else pat.sub(f"{alias}.{col}", x) for i, x in enumerate(parts)]
    return "".join(parts)


_JUNIPER_C = qualify(JUNIPER_TURN_PREDICATE, "c", ("prompt", "client_meta"))
_JUNIPER_P = qualify(JUNIPER_TURN_PREDICATE, "p", ("prompt", "client_meta"))
_INGRESS_K = qualify(JUNIPER_INGRESS_PREDICATE, "k", ("stage", "seq", "raw"))
# prev_correlation_id / prev_is_juniper: the session's previous chat_history_log row of ANY kind, the
# Hub's own rule for which reply's offers the judge is shown (orion/grammar/promise_view.py PREVIOUS_TURN_SQL).
JUDGED_SQL = f"""
SELECT DISTINCT ON (t.correlation_id) t.correlation_id, t.created_at,
       t.run_artifact -> 'reflection' ->> 'reflection_source' AS reflection_source,
       t.run_artifact -> 'reflection' -> 'promises_made' AS made,
       t.run_artifact -> 'reflection' -> 'offers_accepted' AS accepted,
       t.run_artifact ->> 'final_text' AS final_text,
       c.session_id, c.prompt AS juniper_said,
       EXISTS (SELECT 1 FROM cockpit_turn_sighting k
                WHERE k.correlation_id = t.correlation_id AND {_INGRESS_K}) AS from_browser,
       pv.prev_correlation_id, pv.prev_is_juniper
  FROM harness_turn_trace t
  JOIN chat_history_log c ON c.correlation_id = t.correlation_id
  LEFT JOIN LATERAL (
       SELECT p.correlation_id AS prev_correlation_id, ({_JUNIPER_P}) AS prev_is_juniper
         FROM chat_history_log p
        WHERE p.session_id = c.session_id AND p.correlation_id IS DISTINCT FROM c.correlation_id
          AND p.created_at < c.created_at
        ORDER BY p.created_at DESC LIMIT 1) pv ON true
 WHERE t.created_at >= %(since)s AND {_JUNIPER_C}
 ORDER BY t.correlation_id
"""
CARRY_SQL = """
SELECT client_meta ->> 'concern_id' AS concern_id, created_at
  FROM chat_history_log
 WHERE client_meta ->> 'source' = %(source)s AND created_at >= %(since)s
"""


def _aware(ts: Any) -> Optional[datetime]:
    if not isinstance(ts, datetime):
        return None
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, mid - half), min(1.0, mid + half))


def poisson_rate_interval(k: int, n: int, z: float = 1.645) -> tuple[float, float]:
    """Score interval for a count rate k events over n turns (a turn can hold several promises, so this
    is not a proportion and Wilson does not apply: it would take sqrt of a negative once k > n)."""
    if n <= 0:
        return (0.0, float("inf"))
    half = z * math.sqrt(k + z * z / 4)
    return (max(0.0, (k + z * z / 2 - half) / n), (k + z * z / 2 + half) / n)


def beta_summary(kept: int, broken: int) -> dict[str, float]:
    """Beta(1, 1) prior: posterior mean and an approximate 90% interval (normal approximation of the
    Beta, clipped; exact enough for a report line, not used as a gate)."""
    a, b = kept + 1, broken + 1
    mean = a / (a + b)
    sd = math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
    return {"mean": round(mean, 3), "lo90": round(max(0.0, mean - 1.645 * sd), 3),
            "hi90": round(min(1.0, mean + 1.645 * sd), 3), "n": kept + broken}


def unclosable_ids(acts: Sequence[Mapping[str, Any]], juniper_starts: Sequence[datetime],
                   view_max: int = PROMISE_JUDGE_VIEW_MAX) -> set[str]:
    """Expired acts that, at every Juniper turn inside their window, were not among the oldest
    ``view_max`` open promises: the judge never saw them, so they could not be marked kept."""
    out: set[str] = set()
    starts = sorted(t for t in (_aware(s) for s in juniper_starts) if t is not None)
    for act in acts:
        if promise_close_kind(act.get("outcome_source"), act.get("outcome_note"), act.get("closed_at") is not None) != "expired":
            continue
        opened, horizon = _aware(act.get("opened_at")), _aware(act.get("horizon_at"))
        if opened is None or horizon is None:
            continue
        seen = False
        for t in (s for s in starts if opened < s <= horizon):
            open_now = [a for a in acts if _open_at(a, t)]
            shown = {a["act_id"] for a in sorted(open_now, key=lambda a: _aware(a.get("opened_at")))[:view_max]}
            if act["act_id"] in shown:
                seen = True
                break
        if not seen:
            out.add(str(act["act_id"]))
    return out


def _open_at(act: Mapping[str, Any], t: datetime) -> bool:
    opened, horizon, closed = _aware(act.get("opened_at")), _aware(act.get("horizon_at")), _aware(act.get("closed_at"))
    return opened is not None and opened <= t and (horizon is None or horizon > t) and (closed is None or closed > t)


def close_summary(acts: Sequence[Mapping[str, Any]], *, unclosable: set[str],
                  carry_concerns: Mapping[str, list[datetime]] = {}) -> dict[str, Any]:
    from orion.prime.handback import turn_concern_id

    kinds: Counter[str] = Counter()
    evidence: Counter[str] = Counter()
    expired_with_carry: list[str] = []
    for act in acts:
        kind = promise_close_kind(act.get("outcome_source"), act.get("outcome_note"), act.get("closed_at") is not None)
        if kind == "expired" and str(act["act_id"]) in unclosable:
            kind = "unclosable"
        kinds[kind] += 1
        if kind == "kept":
            evidence[kept_evidence_kind(act.get("outcome_note")) or "unknown"] += 1
        if kind == "expired":
            parsed = parse_promise_ref(str(act.get("test_before") or ""))
            opened, horizon = _aware(act.get("opened_at")), _aware(act.get("horizon_at"))
            if parsed and opened and horizon and any(
                    opened < t <= horizon for t in carry_concerns.get(turn_concern_id(parsed[0]), [])):
                expired_with_carry.append(str(act["act_id"]))
    kept, expired = kinds.get("kept", 0), kinds.get("expired", 0)
    return {"acts": len(acts), "open": kinds.get("open", 0), "kept": kept, "expired": expired,
            "void": kinds.get("void", 0), "unclosable": kinds.get("unclosable", 0),
            "kept_by_evidence": dict(evidence), "expired_with_carry_reply": len(expired_with_carry),
            "expired_with_carry_reply_acts": expired_with_carry,
            "keep_rate": beta_summary(kept, expired)}


def _made(raw: Any) -> list[dict]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return []
    return [x for x in raw if isinstance(x, dict)] if isinstance(raw, list) else []


def _firm(item: Mapping[str, Any]) -> bool:
    return item.get("force") == PROMISE_FORCE_COMMITS and len(str(item.get("done_when") or "").strip()) >= PROMISE_DONE_WHEN_MIN_CHARS


def guard_summary(turns: Sequence[Mapping[str, Any]], baseline: tuple[int, int] = REPLAY_BASELINE,
                  finder_live_at: datetime = FINDER_LIVE_AT,
                  min_judged: int = GUARD_MIN_JUDGED_TURNS) -> dict[str, Any]:
    """Firm (non-offer) promises per JUDGED Juniper turn vs the replay baseline, counting only turns
    from ``finder_live_at`` on (``turns_before_finder`` counts the rest) and only turns Juniper sent
    from a Hub browser surface (``turns_not_from_browser`` counts the rest: a coding session's message
    posted over the Hub API, #3602). Items written before #3532 (no force/done_when) are counted in
    ``skipped_pre_force`` and left out of both sides. Red needs at least ``GUARD_MIN_JUDGED_TURNS``
    judged turns (``armed``)."""
    def _before(t: Mapping[str, Any]) -> bool:
        at = _aware(t.get("created_at"))
        return at is not None and at < finder_live_at

    early = [t for t in turns if _before(t)]
    not_browser = [t for t in turns if not _before(t) and t.get("from_browser", True) is False]
    judged = [t for t in turns if t.get("reflection_source") == "substrate_informed_pass" and not _before(t)
              and t.get("from_browser", True) is not False]
    firm = skipped = offers = 0
    for t in judged:
        for item in _made(t.get("made")):
            if "force" not in item and "done_when" not in item:
                skipped += 1
            elif _firm(item):
                if item.get("offer") is True:
                    offers += 1
                else:
                    firm += 1
    rate = firm / len(judged) if judged else None
    base = baseline[0] / baseline[1]
    ratio = (rate / base) if rate is not None and base > 0 else None
    lo, hi = poisson_rate_interval(firm, len(judged)) if judged else (None, None)
    for t in early:
        skipped += sum(1 for item in _made(t.get("made")) if "force" not in item and "done_when" not in item)
    armed = len(judged) >= min_judged
    return {"judged_turns": len(judged),
            "turns_not_judged": len(turns) - len(early) - len(not_browser) - len(judged),
            "turns_before_finder": len(early), "turns_not_from_browser": len(not_browser),
            "firm_promises": firm,
            "offers": offers, "skipped_pre_force": skipped,
            "rate": None if rate is None else round(rate, 4),
            "rate_poisson90": None if lo is None else [round(lo, 4), round(hi, 4)],
            "baseline": f"{baseline[0]}/{baseline[1]} replay",
            "baseline_rate": round(base, 4), "ratio": None if ratio is None else round(ratio, 2),
            "min_judged_turns": min_judged, "armed": armed,
            "red": bool(armed and ratio is not None and ratio > GUARD_RATIO_RED)}


def _judged_firm(turns: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for t in turns:
        if t.get("reflection_source") != "substrate_informed_pass":
            continue
        for i, item in enumerate(_made(t.get("made"))):
            if _firm(item) and item.get("offer") is not True and str(item.get("quote") or "").strip():
                out.append({"ref": f"{str(t['correlation_id'])[:8]}#{i}", "corr": str(t["correlation_id"]),
                            "session": t.get("session_id"), "at": _aware(t.get("created_at")), **item})
    return out


def _flat(text: Any) -> str:
    return normalize_for_quote_match(str(text or "")).casefold()


def duplicate_candidates(turns: Sequence[Mapping[str, Any]], *, window: timedelta = timedelta(hours=72),
                         span_chars: int = DUPLICATE_SPAN_CHARS) -> list[dict[str, Any]]:
    """Pairs of firm promises from different turns of one session, made within ``window`` of each other
    (both could be open at once), that may be one promise said twice in other words: the same ``act``,
    the same ``done_when``, or a shared run of at least ``span_chars`` characters in the quotes. Only
    word-for-word repeats are skipped at open time (``REPEAT_MIN_CHARS``); this lists the rest for a
    hand look, it closes nothing."""
    from difflib import SequenceMatcher

    items = sorted(_judged_firm(turns), key=lambda x: (x["at"] or datetime.min.replace(tzinfo=timezone.utc)))
    pairs: list[dict[str, Any]] = []
    for i, a in enumerate(items):
        for b in items[i + 1:]:
            if a["corr"] == b["corr"] or not a.get("session") or a.get("session") != b.get("session"):
                continue
            if a["at"] is None or b["at"] is None:
                continue  # no time: cannot tell whether both were open at once
            if b["at"] - a["at"] > window:
                break  # sorted by time: every later b is further away
            why = []
            if _flat(a.get("act")) and _flat(a.get("act")) == _flat(b.get("act")):
                why.append("same_act")
            if _flat(a.get("done_when")) and _flat(a.get("done_when")) == _flat(b.get("done_when")):
                why.append("same_done_when")
            qa, qb = _flat(a.get("quote")), _flat(b.get("quote"))
            m = SequenceMatcher(None, qa, qb, autojunk=False).find_longest_match(0, len(qa), 0, len(qb))
            if m.size >= span_chars:
                why.append(f"shared_span_{m.size}")
            if why:
                pairs.append({"a": a["ref"], "b": b["ref"], "why": why})
    return pairs


def offers_summary(turns: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Accepted offers in the stored judged turns, checked against the reply the Hub showed the judge:
    the session's previous chat row of any kind (``prev_correlation_id``, Hub's PREVIOUS_TURN_SQL rule).
    Matching keys on the normalized STORED offer quote: the gate (rule 2) already swapped the judge's own
    text for the stored quote it matched. So ``accepted_unmatched`` (the Hub's per-turn count of the same
    name) means the view and the stored offers disagreed. Counted apart: ``accepted_prev_not_juniper``
    (the previous row was not a Juniper turn, so the Hub showed no offers: also a disagreement) and
    ``accepted_no_prior`` (the previous row is outside the window or missing)."""
    by_corr = {str(t["correlation_id"]): t for t in turns}
    accepted = unmatched = no_prior = not_juniper = 0
    unmatched_refs: list[str] = []
    for cur in turns:
        quotes = [q for q in _accepted(cur.get("accepted")) if q.strip()]
        if not quotes:
            continue
        accepted += len(quotes)
        if cur.get("prev_correlation_id") and cur.get("prev_is_juniper") is False:
            not_juniper += len(quotes)
            continue
        prev = by_corr.get(str(cur.get("prev_correlation_id") or ""))
        if prev is None:
            no_prior += len(quotes)
            continue
        offered = {normalize_for_quote_match(str(x.get("quote") or "")) for x in _made(prev.get("made"))
                   if x.get("offer") is True and _firm(x)}
        for q in quotes:
            if normalize_for_quote_match(q) not in offered:
                unmatched += 1
                unmatched_refs.append(str(cur["correlation_id"])[:8])
    return {"accepted": accepted, "accepted_unmatched": unmatched, "accepted_prev_not_juniper": not_juniper,
            "accepted_no_prior": no_prior, "accepted_unmatched_turns": unmatched_refs}


def _accepted(raw: Any) -> list[str]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return []
    return [x for x in raw if isinstance(x, str)] if isinstance(raw, list) else []


def item_key(correlation_id: str, index: int, quote: str) -> str:
    raw = f"{correlation_id}|{index}|{normalize_for_quote_match(quote)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _context(final_text: str, quote: str) -> str:
    text = " ".join(str(final_text or "").split())
    q = " ".join(str(quote or "").split())
    at = text.find(q)
    if at < 0:
        return text[: 2 * CONTEXT_CHARS]
    lo, hi = max(0, at - CONTEXT_CHARS), min(len(text), at + len(q) + CONTEXT_CHARS)
    return ("..." if lo else "") + text[lo:hi] + ("..." if hi < len(text) else "")


def _clip(text: Any, cap: int = JUNIPER_SAID_CHARS) -> str:
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= cap else flat[:cap] + "..."


def blinded_sample(turns: Sequence[Mapping[str, Any]], n: int, *, seed: int = 3531) -> list[dict[str, Any]]:
    """Stored firm items from judged Juniper turns, n at random. Each shows the quote, the reply around it
    and Juniper's message that reply answered (``juniper_said``, so an offer she asked for reads as one):
    no doer, force, act, done_when or offer flag (the judge's own reading)."""
    pool: list[dict[str, Any]] = []
    for t in turns:
        if t.get("reflection_source") != "substrate_informed_pass":
            continue
        for i, item in enumerate(_made(t.get("made"))):
            if not _firm(item) or not str(item.get("quote") or "").strip():
                continue
            pool.append({"key": item_key(str(t["correlation_id"]), i, item["quote"]),
                         "quote": item["quote"], "juniper_said": _clip(t.get("juniper_said")),
                         "reply_context": _context(t.get("final_text") or "", item["quote"]),
                         "label": None, "why": ""})
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:n]


def score_labels(items: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    labelled = [x for x in items if x.get("label") in (True, False)]
    k = sum(1 for x in labelled if x["label"] is True)
    lo, hi = wilson(k, len(labelled))
    return {"labelled": len(labelled), "true": k, "precision": round(k / len(labelled), 3) if labelled else None,
            "wilson95": [round(lo, 3), round(hi, 3)]}


def _rows(conn: Any, sql: str, params: Mapping[str, Any]) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]


def gather(dsn: str, since: datetime) -> dict[str, Any]:
    import psycopg
    from psycopg.rows import dict_row

    with psycopg.connect(dsn, row_factory=dict_row, autocommit=True) as conn:
        acts = _rows(conn, ACTS_SQL, {"actor": PROMISE_ACTOR, "signal": PROMISE_SIGNAL, "since": since})
        starts = [r["created_at"] for r in _rows(conn, JUNIPER_STARTS_SQL, {"since": since})]
        turns = _rows(conn, JUDGED_SQL, {"since": since})
        carry: dict[str, list[datetime]] = {}
        for r in _rows(conn, CARRY_SQL, {"source": CARRY_SOURCE, "since": since}):
            if r.get("concern_id") and _aware(r.get("created_at")):
                carry.setdefault(str(r["concern_id"]), []).append(_aware(r["created_at"]))
    return {"acts": acts, "starts": starts, "turns": turns, "carry": carry}


def report(data: Mapping[str, Any]) -> dict[str, Any]:
    unclosable = unclosable_ids(data["acts"], data["starts"])
    # Same turns the guard counts: from the finder going live, sent from a Hub browser surface (#3602).
    # offers_summary still looks up the previous row among all rows, so a session turn can be a "previous".
    since_finder = [t for t in data["turns"] if (_aware(t.get("created_at")) or FINDER_LIVE_AT) >= FINDER_LIVE_AT]
    juniper_only = [t for t in since_finder if t.get("from_browser", True) is not False]
    ledger = close_summary(data["acts"], unclosable=unclosable, carry_concerns=data["carry"])
    ledger["keep_rate_ceiling"] = {"name": "chat-showable share", "kept_could_show": KEEP_RATE_CEILING[0],
                                   "of": KEEP_RATE_CEILING[1],
                                   "share": round(KEEP_RATE_CEILING[0] / KEEP_RATE_CEILING[1], 2),
                                   "source": "replay hand count, one reader, not blind (#3595)"}
    return {"ledger": ledger, "guard": guard_summary(data["turns"]), "offers": offers_summary(juniper_only),
            "duplicate_candidates": duplicate_candidates(juniper_only)}


def _ids(ids: Sequence[str], cap: int = 10) -> str:
    if not ids:
        return ""
    more = len(ids) - cap
    return f" ({', '.join(ids[:cap])}{f' +{more} more' if more > 0 else ''})"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dsn", default=os.getenv("ORION_LIVE_DSN") or os.getenv("POSTGRES_URI"))
    ap.add_argument("--days", type=float, default=28.0)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--sample", type=int, default=0, help="write a blinded labelling sheet of N stored items")
    ap.add_argument("--out", default="/tmp/promise-ledger-labels.json")
    ap.add_argument("--score", help="score a filled labelling sheet (no database needed)")
    args = ap.parse_args(argv)
    if args.score:
        print(json.dumps(score_labels(json.loads(Path(args.score).read_text())), indent=2))
        return 0
    if not args.dsn:
        print("set ORION_LIVE_DSN (or POSTGRES_URI, or --dsn); the report only reads", file=sys.stderr)
        return 2
    since = datetime.now(timezone.utc) - timedelta(days=args.days)
    data = gather(args.dsn, since)
    out = report(data)
    if args.sample:
        sheet = blinded_sample(data["turns"], args.sample)
        Path(args.out).write_text(json.dumps(sheet, indent=2))
        out["sample"] = {"written": len(sheet), "path": args.out}
    if args.json:
        print(json.dumps(out, indent=2, default=str))
        return 0
    led, g = out["ledger"], out["guard"]
    print(f"promise acts since {since:%Y-%m-%d %H:%M} UTC: {led['acts']} "
          f"(open {led['open']}, kept {led['kept']}, expired {led['expired']}, void {led['void']}, "
          f"un-closeable {led['unclosable']})")
    print(f"kept by evidence: {led['kept_by_evidence'] or '-'}; expired with a carry reply: {led['expired_with_carry_reply']}"
          + _ids(led["expired_with_carry_reply_acts"]))
    kr = led["keep_rate"]
    print(f"keep rate (Beta(1,1)) over kept {led['kept']} + expired {led['expired']} (void {led['void']} and "
          f"un-closeable {led['unclosable']} left out): mean {kr['mean']}, 90% {kr['lo90']}-{kr['hi90']}, n={kr['n']}; "
          f"ceiling ({led['keep_rate_ceiling']['name']}) {led['keep_rate_ceiling']['share']} "
          f"({KEEP_RATE_CEILING[0]}/{KEEP_RATE_CEILING[1]}, {led['keep_rate_ceiling']['source']})")
    ci = g["rate_poisson90"]
    print(f"guard: {g['firm_promises']} firm promises over {g['judged_turns']} judged Juniper turns = {g['rate']}"
          f"{f' (90% {ci[0]}-{ci[1]})' if ci else ''} "
          f"vs frozen baseline {g['baseline_rate']} ({g['baseline']}): ratio {g['ratio']}{' RED' if g['red'] else ''}"
          f"{'' if g['armed'] else f' (not armed: red needs {GUARD_MIN_JUDGED_TURNS} judged turns)'}; "
          f"skipped pre-#3532 items {g['skipped_pre_force']}; turns not judged {g['turns_not_judged']}; "
          f"turns before the finder went live ({FINDER_LIVE_AT:%Y-%m-%d %H:%MZ}) left out {g['turns_before_finder']}; "
          f"turns not from a Hub browser surface left out {g['turns_not_from_browser']}")
    of = out["offers"]
    print(f"offers accepted: {of['accepted']}; accepted_unmatched {of['accepted_unmatched']}"
          f"{_ids(of['accepted_unmatched_turns'])}; previous row not a Juniper turn {of['accepted_prev_not_juniper']}; "
          f"previous row outside the window {of['accepted_no_prior']}")
    dups = out["duplicate_candidates"]
    print(f"duplicate candidates (same session, within 72 h; same act or done_when, or a shared span of "
          f">= {DUPLICATE_SPAN_CHARS} chars): {len(dups)}"
          + _ids([f"{d['a']}~{d['b']} {'+'.join(d['why'])}" for d in dups]))
    if args.sample:
        print(f"blinded sample: {out['sample']['written']} items -> {out['sample']['path']} (fill label true/false, then --score)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
