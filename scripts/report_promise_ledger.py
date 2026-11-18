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
* keep rate kept / (kept + expired), void and un-closeable left out, with a Beta(1, 1) posterior; this is the
  headline over the graded settle class, with each void kind (#3595) and later carry deliveries beside it;
* Prime concern vs act: how often Prime's promise concerns and the ledger's acts disagree (counted only);
* the self-reinforcement guard: firm promises per JUDGED Juniper turn (judge model pass, not the
  quick gate), with a 90% Poisson score interval, against the frozen replay baseline (``REPLAY_BASELINE``);
  red above 2x. Stored items with no
  ``force``/``done_when`` (written before #3532) are skipped and counted, never guessed;
* expected vs observed: the firm promises the frozen baseline rate predicts for the judged browser turns
  (``judged x 0.080``) beside the count seen, and how likely a count that low is by chance;
* the browser window: Juniper's browser turns since the finder went live and the last one (its correlation
  id and age). Hub reads the promise view only on those turns (#3641), so a stretch with no browser turn
  logs no ``promise_view``/``promise_ledger`` line by design: the report prints the gap so it is not read
  as "the ledger was quiet";
* the lane split: every turn whose reflection stored promise items since the finder went live, by lane
  (Juniper's browser turns vs each other session), because only the browser lane can open a ledger act;
* Prime by exact key: for each of those turns, Prime's concern for it (``turn_concern_id(corr)``: open,
  closed, or refused by a loop-guard bound) and the ``intention.carry`` replies keyed on the same concern
  id. Prime state is Redis (``orion:prime:state``/``orion:prime:turns``): read with the ``redis`` package
  when it is installed, or from ``--prime-state``/``--prime-turns`` files (``redis-cli GET`` output);
  else the Prime column prints ``not read``;
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
    promise_void_kind,
)
from orion.harness.promise_gate import normalize_for_quote_match  # noqa: E402
from orion.regulation.juniper_turns import JUNIPER_INGRESS_PREDICATE, JUNIPER_TURN_PREDICATE  # noqa: E402
from orion.schemas.promise_horizon import PROMISE_HORIZON_SEC  # noqa: E402
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
# The same baseline without the 2 judged turns that were coding-session notes posted to /api/chat
# (6cf88d2b, 9f846e8e on 11-15, each with 1 firm promise; #3602). Printed beside the frozen one
# (Orion's PR #3561 third-round question 2); red is still judged against REPLAY_BASELINE until
# Juniper re-freezes it (#3531).
REPLAY_BASELINE_CLEAN = (21, 287)
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
# The keep rate's ceiling, printed beside the headline: the share of firm promises a later chat reply could show
# kept (rows stored before #3595 carry no settle class, so this hand count still frames them). Hand count
# of the 23 firm replay promises (one reader, not blind; #3595): 9 chat-showable,
# 8 delivered by a run or outside chat, 6 abstentions or standing states. A keep rate near this
# number is the most the reply-only close can show, not a sign of broken promises. The 6 hold the
# abstentions ("I won't float anything back up while you're asleep", "I'll stop muttering about bus
# error": 2 of the 23, a second reading, 2026-11-17), so the 9 never counts one as chat-showable.
# Whether an abstention is a promise at all is #3595.
KEEP_RATE_CEILING = (9, 23)
# The ceiling on the clean basis (REPLAY_BASELINE_CLEAN): both coding-session promises fall in the 8
# delivered outside chat (a record label; a read of proposals that arrive by session thread), so the
# numerator stays 9.
KEEP_RATE_CEILING_CLEAN = (9, 21)
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
# Duplicate candidates are looked for inside one promise horizon: a repeat while the earlier act could still be
# open (#3595: the one 72 h promise clock).
PROMISE_WINDOW = timedelta(seconds=PROMISE_HORIZON_SEC)


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
# Every turn (any lane) whose reflection stored promise items, with its ingress session and whether a Hub
# browser surface sent it. Starts from the few item-bearing traces; the ingress hop is a primary-key probe.
STORED_SQL = f"""
SELECT t.correlation_id, t.created_at, t.workflow,
       t.run_artifact -> 'reflection' -> 'promises_made' AS made,
       (SELECT k.raw ->> 'session_id' FROM cockpit_turn_sighting k
         WHERE k.correlation_id = t.correlation_id AND k.stage = 'ingress' AND k.seq = 0) AS session_id,
       (SELECT k.raw ? 'mode' FROM cockpit_turn_sighting k
         WHERE k.correlation_id = t.correlation_id AND k.stage = 'ingress' AND k.seq = 0) AS has_mode,
       EXISTS (SELECT 1 FROM cockpit_turn_sighting k
                WHERE k.correlation_id = t.correlation_id AND {_INGRESS_K}) AS from_browser
  FROM harness_turn_trace t
 WHERE t.created_at >= %(since)s
   AND t.run_artifact -> 'reflection' -> 'promises_made' @> '[{{}}]'
 ORDER BY t.created_at
"""
ROUNDUP_SESSION_PREFIX = "orion_session-roundup-"
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


def poisson_cdf(k: int, mean: float) -> float:
    """P(X <= k) for X ~ Poisson(mean): how likely a count this low is when the rate is the baseline."""
    if mean <= 0:
        return 1.0
    term = total = math.exp(-mean)
    for i in range(1, k + 1):
        term *= mean / i
        total += term
    return min(1.0, total)


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
    void_kinds: Counter[str] = Counter()
    void_later_carry: list[str] = []
    expired_with_carry: list[str] = []
    for act in acts:
        kind = promise_close_kind(act.get("outcome_source"), act.get("outcome_note"), act.get("closed_at") is not None)
        if kind == "expired" and str(act["act_id"]) in unclosable:
            kind = "unclosable"
        kinds[kind] += 1
        if kind == "kept":
            evidence[kept_evidence_kind(act.get("outcome_note")) or "unknown"] += 1
        if kind == "void":
            void_kinds[promise_void_kind(act.get("outcome_note")) or "no_juniper_turn"] += 1
            # #3595 rec 2: a carry run that delivered a void act later is a note here, never a reopen.
            parsed = parse_promise_ref(str(act.get("test_before") or ""))
            opened = _aware(act.get("opened_at"))
            if parsed and opened and any(t > opened for t in carry_concerns.get(turn_concern_id(parsed[0]), [])):
                void_later_carry.append(str(act["act_id"]))
        if kind == "expired":
            parsed = parse_promise_ref(str(act.get("test_before") or ""))
            opened, horizon = _aware(act.get("opened_at")), _aware(act.get("horizon_at"))
            if parsed and opened and horizon and any(
                    opened < t <= horizon for t in carry_concerns.get(turn_concern_id(parsed[0]), [])):
                expired_with_carry.append(str(act["act_id"]))
    kept, expired = kinds.get("kept", 0), kinds.get("expired", 0)
    closed = kept + expired + kinds.get("void", 0) + kinds.get("unclosable", 0)
    return {"acts": len(acts), "open": kinds.get("open", 0), "kept": kept, "expired": expired,
            "void": kinds.get("void", 0), "unclosable": kinds.get("unclosable", 0),
            "void_share": round(kinds.get("void", 0) / closed, 3) if closed else None,
            "kept_by_evidence": dict(evidence), "expired_with_carry_reply": len(expired_with_carry),
            "expired_with_carry_reply_acts": expired_with_carry,
            # Settle classes (#3595): which void each void close is. The keep rate below is over the graded
            # class only (kept + expired: a chat-showable promise, or a conditional one whose condition arose);
            # these voids sit beside it, never in it.
            "void_by_kind": dict(void_kinds),
            # The judge's settle-class labels decide which unkept promises are graded at all: their share of closed
            # acts is printed so a drift toward "run"/"absence"/conditions (which shrinks n) is visible (#3595 review).
            "settle_class_void_share": (round(sum(v for k, v in void_kinds.items() if k != "no_juniper_turn") / closed, 3)
                                        if closed else None),
            "void_with_later_carry": len(void_later_carry), "void_with_later_carry_acts": void_later_carry,
            "keep_rate": beta_summary(kept, expired)}


def prime_vs_acts(acts: Sequence[Mapping[str, Any]], prime_state: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    """Prime's promise concerns against the ledger's acts, keyed by the promising turn (#3595 rec 3, #3531 point 6).
    In shadow the two records may disagree by design; this only counts the disagreement. A concern is a promise
    concern when its ``event_ref`` is ``turn:<corr>``; an act belongs to turn ``corr`` by its ``test_before`` ref.
    ``concern_no_act``: Prime holds a promise concern for a turn that opened no act (a pre-#3532 promise, an offer,
    a non-Juniper turn). ``act_no_concern``: an act whose turn Prime holds no open or recorded concern for (refused
    by a loop-guard bound, or aged out). ``concern_open_act_closed``: Prime still holds it while the act closed.
    None when Prime's state was not read."""
    if prime_state is None:
        return {"read": False}
    from orion.prime.handback import turn_concern_id

    by_corr: dict[str, list[Mapping[str, Any]]] = {}
    for act in acts:
        parsed = parse_promise_ref(str(act.get("test_before") or ""))
        if parsed:
            by_corr.setdefault(parsed[0], []).append(act)
    open_turns = {str(c.get("event_ref"))[len("turn:"):]: str(c.get("concern_id"))
                  for c in prime_state.get("concerns") or [] if str(c.get("event_ref") or "").startswith("turn:")}
    recorded = {str(r.get("concern_id")) for r in prime_state.get("concern_record") or []}
    concern_no_act = sorted(cid for corr, cid in open_turns.items() if corr not in by_corr)
    act_no_concern, open_act_closed = [], []
    for corr, corr_acts in by_corr.items():
        cid = turn_concern_id(corr)
        if corr not in open_turns and cid not in recorded:
            act_no_concern.extend(str(a["act_id"]) for a in corr_acts)
        elif corr in open_turns and all(a.get("closed_at") is not None for a in corr_acts):
            open_act_closed.append(open_turns[corr])
    return {"read": True, "concern_no_act": len(concern_no_act), "concern_no_act_ids": concern_no_act,
            "act_no_concern": len(act_no_concern), "act_no_concern_ids": act_no_concern,
            "concern_open_act_closed": len(open_act_closed), "concern_open_act_closed_ids": sorted(open_act_closed)}


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
    judged turns (``armed``). The clean basis (``ratio_clean``) is always ``REPLAY_BASELINE_CLEAN``, whatever
    ``baseline`` is passed: it is a fixed side number, not a second guard."""
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
    clean = REPLAY_BASELINE_CLEAN[0] / REPLAY_BASELINE_CLEAN[1]
    lo, hi = poisson_rate_interval(firm, len(judged)) if judged else (None, None)
    for t in early:
        skipped += sum(1 for item in _made(t.get("made")) if "force" not in item and "done_when" not in item)
    armed = len(judged) >= min_judged
    expected = len(judged) * base
    return {"judged_turns": len(judged),
            "expected_firm": round(expected, 2),
            "p_at_most_observed": round(poisson_cdf(firm, expected), 3) if judged else None,
            "p_at_least_observed": (round(1.0 - poisson_cdf(firm - 1, expected), 3) if firm else 1.0) if judged else None,
            "turns_not_judged": len(turns) - len(early) - len(not_browser) - len(judged),
            "turns_before_finder": len(early), "turns_not_from_browser": len(not_browser),
            "firm_promises": firm,
            "offers": offers, "skipped_pre_force": skipped,
            "rate": None if rate is None else round(rate, 4),
            "rate_poisson90": None if lo is None else [round(lo, 4), round(hi, 4)],
            "baseline": f"{baseline[0]}/{baseline[1]} replay",
            "baseline_rate": round(base, 4), "ratio": None if ratio is None else round(ratio, 2),
            "baseline_clean": f"{REPLAY_BASELINE_CLEAN[0]}/{REPLAY_BASELINE_CLEAN[1]} replay, coding-session turns out",
            "baseline_clean_rate": round(clean, 4),
            "ratio_clean": None if rate is None else round(rate / clean, 2),
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


def duplicate_candidates(turns: Sequence[Mapping[str, Any]], *, window: timedelta = PROMISE_WINDOW,
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


def duplicate_window_pairs(turns: Sequence[Mapping[str, Any]], *, window: timedelta = PROMISE_WINDOW) -> int:
    """How many pairs of firm promises ``duplicate_candidates`` compares at all: different turns of one
    session, made within ``window`` of each other, before any similarity test. This, not the candidate
    count, sizes the double-open risk (Orion's PR #3598 third-round question 5)."""
    items = sorted((x for x in _judged_firm(turns) if x["at"] is not None and x.get("session")),
                   key=lambda x: x["at"])
    n = 0
    for i, a in enumerate(items):
        for b in items[i + 1:]:
            if b["at"] - a["at"] > window:
                break
            if a["corr"] != b["corr"] and a["session"] == b["session"]:
                n += 1
    return n


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


def lane_of(row: Mapping[str, Any]) -> str:
    """``juniper_browser`` for a turn a Hub browser surface sent (the only lane that can open a ledger act),
    else its ingress session (round-ups folded into one), else its workflow."""
    if row.get("from_browser"):
        return "juniper_browser"
    session = str(row.get("session_id") or "")
    if session.startswith(ROUNDUP_SESSION_PREFIX):
        return "orion_session-roundup"
    return session or str(row.get("workflow") or "") or "unknown"


def _promise_items(raw: Any) -> list[int]:
    """Indices of the stored items Prime's hand-back counts as promises (``_counted_promises``: a dict with a
    non-empty quote, not an offer)."""
    from orion.prime.handback import _counted_promises

    return _counted_promises({"promises_made": _made(raw)})[0]


def lane_split(stored: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Item-bearing turns by lane: turns, items, firm non-offer items."""
    lanes: dict[str, dict[str, int]] = {}
    for row in stored:
        made = _made(row.get("made"))
        if not made:
            continue
        lane = lanes.setdefault(lane_of(row), {"turns": 0, "items": 0, "firm": 0})
        lane["turns"] += 1
        lane["items"] += len(made)
        lane["firm"] += sum(1 for x in made if _firm(x) and x.get("offer") is not True)
    ordered = dict(sorted(lanes.items(), key=lambda kv: (-kv[1]["turns"], kv[0])))
    return {"turns": sum(v["turns"] for v in lanes.values()), "items": sum(v["items"] for v in lanes.values()),
            "browser_turns": lanes.get("juniper_browser", {}).get("turns", 0), "by_lane": ordered}


def browser_window(turns: Sequence[Mapping[str, Any]], now: datetime,
                   finder_live_at: datetime = FINDER_LIVE_AT) -> dict[str, Any]:
    """Juniper's browser turns since the finder went live (logged turns), and the last one: the only turns
    Hub reads the promise view on, so the gap after the last one has no promise lines by design."""
    browser = sorted(((at, t) for t in turns
                      if t.get("from_browser") is True and (at := _aware(t.get("created_at"))) is not None
                      and at >= finder_live_at), key=lambda x: x[0])
    judged = sum(1 for _, t in browser if t.get("reflection_source") == "substrate_informed_pass")
    if not browser:
        return {"browser_turns": 0, "judged": 0, "last_at": None, "last_corr": None, "hours_since_last": None}
    at, last = browser[-1]
    return {"browser_turns": len(browser), "judged": judged, "last_at": at.isoformat(timespec="seconds"),
            "last_corr": str(last["correlation_id"])[:8],
            "hours_since_last": round((now - at).total_seconds() / 3600.0, 1)}


def prime_by_ref(stored: Sequence[Mapping[str, Any]], carry: Mapping[str, list[datetime]],
                 prime_state: Optional[Mapping[str, Any]], prime_turns: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    """For each item-bearing turn with a non-offer promise: what Prime did with it, keyed on the exact concern
    id the hand-back derives from the turn (``turn_concern_id(corr)``, the id the carry run and its chat row
    carry), and the carry replies on that id. ``who`` is Prime's own ``classify_who`` on the ingress session,
    ``mode`` flag and browser ``surface_context`` (#3602): ``None`` means the hand-back never reads that session (``not_handed_back``). Prime status
    is ``not_read`` when no Prime state was given. ``refused`` is visible only for turns still in Prime's
    recent-turn window (``MAX_RECENT_TURNS``) and a closed concern only while in ``concern_record``
    (``MAX_CONCERN_RECORD``), so ``no_concern_or_aged_out`` holds both "opened nothing" and "aged out"."""
    from orion.prime.handback import classify_who, turn_concern_id

    open_ids = {str(c.get("concern_id")) for c in (prime_state or {}).get("concerns") or []}
    closed = {str(c.get("concern_id")): str(c.get("end") or "closed")
              for c in (prime_state or {}).get("concern_record") or []}
    recent = {str(t.get("correlation_id")): t for t in (prime_turns or {}).get("recent") or []}
    status: Counter[str] = Counter()
    by_lane: dict[str, Counter[str]] = {}
    carries = 0
    rows: list[dict[str, Any]] = []
    for row in stored:
        if not _promise_items(row.get("made")):
            continue
        corr = str(row["correlation_id"])
        cid = turn_concern_id(corr)
        # The ingress raw as classify_who reads it: ``mode`` and, for a Hub browser turn, ``surface_context`` (#3602).
        raw = {**({"mode": True} if row.get("has_mode") else {}),
               **({"surface_context": {"from_browser": True}} if row.get("from_browser") else {})}
        who = classify_who(row.get("session_id"), raw)
        if who is None:
            st = "not_handed_back"
        elif prime_state is None:
            st = "not_read"
        elif cid in open_ids:
            st = "concern_open"
        elif cid in closed:
            st = f"concern_{closed[cid]}"
        elif corr in recent and recent[corr].get("thread_refused"):
            st = "refused"
        else:
            st = "no_concern_or_aged_out"
        n_carry = len(carry.get(cid, []))
        carries += n_carry
        status[st] += 1
        by_lane.setdefault(lane_of(row), Counter())[st] += 1
        rows.append({"corr": corr[:8], "lane": lane_of(row), "prime_who": who, "concern_id": cid, "prime": st,
                     "carries": n_carry,
                     "refused_why": (recent.get(corr) or {}).get("thread_refused")})
    return {"turns": len(rows), "status": dict(status), "by_lane": {k: dict(v) for k, v in by_lane.items()},
            "carry_replies": carries, "rows": rows}


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
        stored = _rows(conn, STORED_SQL, {"since": max(since, FINDER_LIVE_AT)})
        carry: dict[str, list[datetime]] = {}
        for r in _rows(conn, CARRY_SQL, {"source": CARRY_SOURCE, "since": since}):
            if r.get("concern_id") and _aware(r.get("created_at")):
                carry.setdefault(str(r["concern_id"]), []).append(_aware(r["created_at"]))
    return {"acts": acts, "starts": starts, "turns": turns, "carry": carry, "stored": stored}


PRIME_STATE_KEY = "orion:prime:state"
PRIME_TURNS_KEY = "orion:prime:turns"


def read_prime(redis_url: Optional[str], state_file: Optional[str] = None,
               turns_file: Optional[str] = None) -> tuple[Optional[dict], Optional[dict], str]:
    """(state, turns, how) from files when given, else from Redis with the ``redis`` package. Never raises:
    a missing package or an unreachable Redis returns (None, None, why) and the report prints ``not read``."""
    if state_file:
        try:
            state = json.loads(Path(state_file).read_text())
            turns = json.loads(Path(turns_file).read_text()) if turns_file else None
            return state, turns, f"files {state_file}{f', {turns_file}' if turns_file else ''}"
        except (OSError, ValueError) as exc:
            return None, None, f"not read ({type(exc).__name__}: {exc})"
    if not redis_url:
        return None, None, "not read (no ORION_BUS_URL and no --prime-state)"
    try:
        import redis  # type: ignore[import-not-found]
    except ImportError:
        return None, None, "not read (no redis package: pass --prime-state/--prime-turns from redis-cli GET)"
    try:
        client = redis.Redis.from_url(redis_url, socket_timeout=3, socket_connect_timeout=3)
        raw_state, raw_turns = client.get(PRIME_STATE_KEY), client.get(PRIME_TURNS_KEY)
        state = json.loads(raw_state) if raw_state else None
        turns = json.loads(raw_turns) if raw_turns else None
        return state, turns, ("redis" if state is not None else f"not read ({PRIME_STATE_KEY} empty)")
    except Exception as exc:  # noqa: BLE001 - a report line, never a crash
        return None, None, f"not read ({type(exc).__name__})"


def report(data: Mapping[str, Any], now: Optional[datetime] = None) -> dict[str, Any]:
    unclosable = unclosable_ids(data["acts"], data["starts"])
    # Same turns the guard counts: from the finder going live, sent from a Hub browser surface (#3602).
    # offers_summary still looks up the previous row among all rows, so a session turn can be a "previous".
    since_finder = [t for t in data["turns"] if (_aware(t.get("created_at")) or FINDER_LIVE_AT) >= FINDER_LIVE_AT]
    juniper_only = [t for t in since_finder if t.get("from_browser", True) is not False]
    ledger = close_summary(data["acts"], unclosable=unclosable, carry_concerns=data["carry"])
    ledger["keep_rate_ceiling"] = {"name": "chat-showable share", "kept_could_show": KEEP_RATE_CEILING[0],
                                   "of": KEEP_RATE_CEILING[1],
                                   "share": round(KEEP_RATE_CEILING[0] / KEEP_RATE_CEILING[1], 2),
                                   "clean_kept_could_show": KEEP_RATE_CEILING_CLEAN[0],
                                   "clean_of": KEEP_RATE_CEILING_CLEAN[1],
                                   "clean_share": round(KEEP_RATE_CEILING_CLEAN[0] / KEEP_RATE_CEILING_CLEAN[1], 2),
                                   "source": "replay hand count, one reader, not blind (#3595)"}
    stored = data.get("stored") or []
    return {"ledger": ledger, "guard": guard_summary(data["turns"]), "offers": offers_summary(juniper_only),
            "duplicate_candidates": duplicate_candidates(juniper_only),
            "duplicate_window_pairs": duplicate_window_pairs(juniper_only),
            "browser_window": browser_window(data["turns"], now or datetime.now(timezone.utc)),
            "lanes": lane_split(stored),
            "prime": {**prime_by_ref(stored, data["carry"], data.get("prime_state"), data.get("prime_turns")),
                      "read": data.get("prime_read") or "not read"},
            "prime_vs_acts": prime_vs_acts(data["acts"], data.get("prime_state"))}


def _or_dash(x: Any) -> str:
    return "-" if x is None else str(x)


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
    ap.add_argument("--prime-state", help="orion:prime:state JSON (redis-cli GET), instead of reading Redis")
    ap.add_argument("--prime-turns", help="orion:prime:turns JSON (redis-cli GET), with --prime-state")
    args = ap.parse_args(argv)
    if args.prime_turns and not args.prime_state:
        ap.error("--prime-turns needs --prime-state")
    if args.score:
        print(json.dumps(score_labels(json.loads(Path(args.score).read_text())), indent=2))
        return 0
    if not args.dsn:
        print("set ORION_LIVE_DSN (or POSTGRES_URI, or --dsn); the report only reads", file=sys.stderr)
        return 2
    since = datetime.now(timezone.utc) - timedelta(days=args.days)
    data = gather(args.dsn, since)
    data["prime_state"], data["prime_turns"], data["prime_read"] = read_prime(
        os.getenv("ORION_BUS_URL"), args.prime_state, args.prime_turns)
    out = report(data)
    start = max(since, FINDER_LIVE_AT)
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
    vs = led.get("void_share")
    print(f"void share of closed acts: {'-' if vs is None else vs} (void = no Juniper turn started in the "
          "horizon: a logged Juniper chat row from any surface, or a Hub browser ingress hop; act_ledger "
          "_JUNIPER_REPLY_SQL)")
    print(f"kept by evidence: {led['kept_by_evidence'] or '-'}; expired with a carry reply: {led['expired_with_carry_reply']}"
          + _ids(led["expired_with_carry_reply_acts"]))
    print(f"void by kind: {led['void_by_kind'] or '-'} (no_juniper_turn = she wrote nothing in 72 h; not_due = a "
          "condition no judged reply saw arise; not_chat = delivered by a run or outside chat; absence_unchecked = "
          "an abstention, no breach check yet; all censored, none graded, #3595)")
    print(f"settle-class void share of closed acts (not graded by the judge's label; watch for drift): "
          f"{_or_dash(led['settle_class_void_share'])}")
    print(f"void acts a carry run for the same turn delivered after the act opened (a note, never a reopen): "
          f"{led['void_with_later_carry']}"
          + _ids(led["void_with_later_carry_acts"]))
    kr = led["keep_rate"]
    print(f"HEADLINE keep rate over the graded class (chat-showable, or conditional with its condition met): "
          f"kept {led['kept']} / (kept + expired {led['expired']}); settle-class voids beside it, not in it")
    print(f"keep rate (Beta(1,1)) over kept {led['kept']} + expired {led['expired']} (void {led['void']} and "
          f"un-closeable {led['unclosable']} left out): mean {kr['mean']}, 90% {kr['lo90']}-{kr['hi90']}, n={kr['n']}; "
          f"ceiling ({led['keep_rate_ceiling']['name']}) {led['keep_rate_ceiling']['share']} "
          f"({KEEP_RATE_CEILING[0]}/{KEEP_RATE_CEILING[1]}; clean {led['keep_rate_ceiling']['clean_share']} "
          f"{KEEP_RATE_CEILING_CLEAN[0]}/{KEEP_RATE_CEILING_CLEAN[1]}; {led['keep_rate_ceiling']['source']})")
    ci = g["rate_poisson90"]
    print(f"guard: {g['firm_promises']} firm promises over {g['judged_turns']} judged Juniper turns = {g['rate']}"
          f"{f' (90% {ci[0]}-{ci[1]})' if ci else ''} "
          f"vs frozen baseline {g['baseline_rate']} ({g['baseline']}): ratio {g['ratio']}{' RED' if g['red'] else ''}"
          f" [clean baseline {g['baseline_clean_rate']} ({g['baseline_clean']}): ratio {g['ratio_clean']}]"
          f"{'' if g['armed'] else f' (not armed: red needs {GUARD_MIN_JUDGED_TURNS} judged turns)'}; "
          f"skipped pre-#3532 items {g['skipped_pre_force']}; turns not judged {g['turns_not_judged']}; "
          f"turns before the finder went live ({FINDER_LIVE_AT:%Y-%m-%d %H:%MZ}) left out {g['turns_before_finder']}; "
          f"turns not from a Hub browser surface left out {g['turns_not_from_browser']}")
    print(f"expected vs observed: {g['expected_firm']} firm promises expected over {g['judged_turns']} judged "
          f"browser turns at the frozen rate {g['baseline_rate']}; observed {g['firm_promises']}; "
          f"P(a count this low by chance) {_or_dash(g['p_at_most_observed'])}, "
          f"P(a count this high by chance) {_or_dash(g['p_at_least_observed'])}")
    bw = out["browser_window"]
    if bw["last_at"]:
        print(f"browser window: {bw['browser_turns']} Juniper browser turns since {start:%Y-%m-%d %H:%MZ} "
              f"({bw['judged']} judged); last {bw['last_at']} ({bw['last_corr']}), {bw['hours_since_last']} h "
              "before this report. Hub reads the promise view only on browser turns, so no promise_view/"
              "promise_ledger line is expected in that gap")
    else:
        print(f"browser window: 0 Juniper browser turns since {start:%Y-%m-%d %H:%MZ}: no promise_view/"
              "promise_ledger line is expected")
    ln = out["lanes"]
    print(f"stored promise items since {start:%Y-%m-%d %H:%MZ}: {ln['items']} on {ln['turns']} turns, "
          f"{ln['browser_turns']} of them Juniper browser turns (the only lane that opens a ledger act); by lane: "
          + (", ".join(f"{k} {v['turns']}t/{v['items']}i/{v['firm']}f" for k, v in ln["by_lane"].items()) or "-"))
    pr = out["prime"]
    print(f"Prime by exact key (turn_concern_id) on {pr['turns']} item-bearing turns with a non-offer promise "
          f"[{pr['read']}]: {pr['status'] or '-'}; intention.carry replies on those keys {pr['carry_replies']}"
          + _ids([f"{r['corr']} {r['lane']} who={r['prime_who']} {r['prime']}" for r in pr["rows"]]))
    print("  (who= is Prime's CURRENT classify_who rule applied to every row: a turn handed back as chat_juniper "
          "before #3602 deployed reads not_handed_back here if it had no browser surface)")
    pv = out["prime_vs_acts"]
    if pv.get("read"):
        print(f"Prime concern vs act (shadow: may disagree, counted only; Prime's concerns are its state now, acts only "
              f"those opened in the window): concern with no act {pv['concern_no_act']}"
              + _ids(pv["concern_no_act_ids"]) + f"; act with no concern {pv['act_no_concern']}"
              + _ids(pv["act_no_concern_ids"]) + f"; concern still open while its act closed "
              f"{pv['concern_open_act_closed']}" + _ids(pv["concern_open_act_closed_ids"]))
    else:
        print("Prime concern vs act: not read (no Prime state)")
    of = out["offers"]
    print(f"offers accepted: {of['accepted']}; accepted_unmatched {of['accepted_unmatched']}"
          f"{_ids(of['accepted_unmatched_turns'])}; previous row not a Juniper turn {of['accepted_prev_not_juniper']}; "
          f"previous row outside the window {of['accepted_no_prior']}")
    dups = out["duplicate_candidates"]
    print(f"duplicate candidates (same session, within 72 h; same act or done_when, or a shared span of "
          f">= {DUPLICATE_SPAN_CHARS} chars): {len(dups)} of {out['duplicate_window_pairs']} same-session "
          f"pairs within 72 h" + _ids([f"{d['a']}~{d['b']} {'+'.join(d['why'])}" for d in dups]))
    if args.sample:
        print(f"blinded sample: {out['sample']['written']} items -> {out['sample']['path']} (fill label true/false, then --score)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
