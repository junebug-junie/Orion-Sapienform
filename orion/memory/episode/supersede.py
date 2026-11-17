"""Corrections supersede what they correct (issue #3587; spec "Corrections: supersede at write time").

When a new ``episode_memory`` row corrects an older one, the older row is retired, never deleted:

1. the old row: ``status = 'superseded'``, ``confirmation_state = 'corrected'``;
2. the correction row: ``supersedes_memory_id = <old memory_id>`` (when it points at nothing yet);
3. one ``episode_memory_event`` row: op ``superseded``, actor, reason, and in ``evidence`` the
   old row's prior ``confirmation_state`` (what undo restores) and the check's reason.

The same shape the confirmation loop writes for Juniper's revisions (``confirmation.py``), so every
reader that filters ``status = 'active'`` drops the old row.

How a pair is found, in two steps:

- **Candidates (deterministic).** Older rows the lived-memory retriever may serve
  (``orion.memory.lived_memory.EPISODE_SQL``, the serving gate), ranked by meaning (cosine in the
  ``orion_lived_memory`` index, the retriever's own vectors) and required to share something
  specific with the new row: a referent key few rows carry, or a rare word. At most ``K`` per row.
- **The check (local model, memory_distill route).** A strict yes/no per pair: "does B correct
  A?". Only ``corrects=yes`` with ``sure=yes`` and a reason goes on to a second question in a
  different framing: "can A and B both be true, each at its own date?". The pair links only when
  that answer is "no". Anything else -- no, unsure, unparseable, a failed call -- links nothing. A wrong supersede hides a true memory; a
  missed one leaves both visible, which is the state before this module existed.

Never superseded automatically: rows Juniper confirmed (her word outranks a model's judgment),
follow-up rows (a to-do is closed, not corrected) and Orion's own views (a changed view is growth;
the earlier one stays as history). Undo: :func:`undo_link`.

Idempotency. A pair is handled once its ``superseded`` event exists: event_id =
uuid5(SUPERSEDE_NAMESPACE, "superseded|<old_id>|<new_id>") (:func:`superseded_event_id`). Applying
an already-superseded pair is a no-op: the old row is no longer active, so nothing is updated and no
second event is written (the insert is also ON CONFLICT DO NOTHING on that id). A pair with a
``supersede_undone`` event is never re-applied.
"""
from __future__ import annotations

import json
import math
import re
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Iterable, Mapping, Optional, Sequence

from orion.memory.lived_memory import EPISODE_SQL

ACTOR_DISTILL = "memory.episode_distill"
ACTOR_PASS = "memory.supersede_pass"
OP_SUPERSEDED = "superseded"
OP_UNDONE = "supersede_undone"
CHECK_VERSION = "supersede_check.v2"
SUPERSEDE_NAMESPACE = uuid.UUID("6f1d3c52-2b8e-4f7a-9d41-3587a0c0ffee")

# Candidate limits. bge-large cosines are compressed (live 2026-11-17: median pair 0.56, 90th
# percentile 0.65, 99th 0.73), so the floor sits near the 80th percentile and the "shares
# something specific" rule does the real narrowing.
K_PER_ROW = 3
DENSE_FLOOR = 0.60
DENSE_ALONE = 0.80        # this close needs no shared rare word or referent
RARE_REFERENT_MAX_ROWS = 25  # person:juniper (106 rows) or person:orion (40) say nothing specific
RARE_TOKEN_MAX_ROWS = 8
NO_DENSE_MIN_SHARED = 2   # without vectors: two shared rare words, or one plus a rare referent

# Never superseded by the model: Juniper's confirmed word, to-dos, and Orion's own views. A later
# view that disagrees is growth, not a correction: the earlier view stays as history. (Dry run
# 2026-11-17: the only sure-yes pair on a view was a changed opinion, not a fact.)
PROTECTED_CONFIRMATION_STATES = frozenset({"confirmed"})
# A row Juniper is still being asked about is hers to settle, not the model's: never retired while
# pending/unconfirmed (her answer would be dropped as already_resolved). And only a correction that
# is settled (auto: low stakes, or confirmed) may retire anything: a pending high-stakes correction
# she later rejects would leave the old row hidden with nothing in its place.
OPEN_CONFIRMATION_STATES = frozenset({"pending_confirmation", "unconfirmed"})
CORRECTION_STATES = frozenset({"auto", "confirmed"})
SKIPPED_PURPOSES = frozenset({"follow_up"})                     # neither side
NOT_SUPERSEDABLE_PURPOSES = SKIPPED_PURPOSES | {"orion_view"}   # never the old side

# The live rows a correction may retire: exactly what recall may serve (EPISODE_SQL), plus what
# the candidate step needs. %s-free, so psycopg and asyncpg both run it.
LIVE_ROWS_SQL = f"""
SELECT l.record_id AS memory_id, l.text AS statement, l.purpose, l.created_at, l.occurred_at,
       m.episode_id, m.confirmation_state, m.supersedes_memory_id::text AS supersedes_memory_id,
       coalesce((SELECT array_agg(DISTINCT r.referent_key) FROM episode_memory_referent r
                 WHERE r.memory_id = m.memory_id), ARRAY[]::text[]) AS referents
FROM ({EPISODE_SQL}) l
JOIN episode_memory m ON m.memory_id::text = l.record_id
ORDER BY l.created_at, l.record_id
"""

_STOP = frozenset(
    """the and but for nor yet with without about over under after before again also just only very
    really not yes that this these those there here what which who whom whose when where why how
    was were been being does did doing have has had having will would shall should can could may
    might must let get got she her hers him his they them their its our ours you your yours mine
    myself told tell tells said says asked asks noted note notes think thought want wants wanted
    juniper orion i'm i've i'd it's that's then than from into onto like some any all each every
    more most other such own same both few much many one two now still even ever well""".split()
)
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’-]*")


def tokens(text: str) -> set[str]:
    out = set()
    for raw in _WORD.findall(text or ""):
        tok = raw.lower().replace("’", "'")
        if tok.endswith("'s"):
            tok = tok[:-2]
        tok = tok.strip("'-")
        if len(tok) >= 3 and tok not in _STOP:
            out.add(tok)
    return out


@dataclass(frozen=True)
class MemoryRow:
    memory_id: str
    statement: str
    purpose: str
    created_at: Optional[datetime]
    confirmation_state: str = "auto"
    episode_id: Optional[str] = None
    supersedes_memory_id: Optional[str] = None
    referents: frozenset[str] = frozenset()
    occurred_at: Optional[datetime] = None

    @property
    def date(self) -> str:
        when = self.occurred_at or self.created_at
        return when.date().isoformat() if when else "undated"


def _aware(value: Any) -> Optional[datetime]:
    if not isinstance(value, datetime):
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def row_from_db(r: Mapping[str, Any]) -> MemoryRow:
    return MemoryRow(
        memory_id=str(r["memory_id"]),
        statement=" ".join(str(r.get("statement") or "").split()),
        purpose=str(r.get("purpose") or ""),
        created_at=_aware(r.get("created_at")),
        confirmation_state=str(r.get("confirmation_state") or "auto"),
        episode_id=r.get("episode_id"),
        supersedes_memory_id=(str(r["supersedes_memory_id"]) if r.get("supersedes_memory_id") else None),
        referents=frozenset(str(k) for k in (r.get("referents") or [])),
        occurred_at=_aware(r.get("occurred_at")),
    )


@dataclass
class Corpus:
    """The live rows and how many of them carry each word / referent (what makes a share specific)."""
    rows: list[MemoryRow]
    token_df: Counter = field(default_factory=Counter)
    referent_df: Counter = field(default_factory=Counter)
    _tokens: dict[str, set[str]] = field(default_factory=dict)

    @classmethod
    def build(cls, rows: Iterable[MemoryRow]) -> "Corpus":
        c = cls(rows=list(rows))
        for r in c.rows:
            toks = tokens(r.statement)
            c._tokens[r.memory_id] = toks
            c.token_df.update(toks)
            c.referent_df.update(r.referents)
        return c

    def toks(self, row: MemoryRow) -> set[str]:
        got = self._tokens.get(row.memory_id)
        return got if got is not None else tokens(row.statement)


@dataclass(frozen=True)
class Candidate:
    old: MemoryRow
    dense: Optional[float]
    shared_rare_tokens: tuple[str, ...]
    shared_rare_referents: tuple[str, ...]


def can_be_superseded(row: MemoryRow) -> bool:
    return (row.confirmation_state not in PROTECTED_CONFIRMATION_STATES
            and row.confirmation_state not in OPEN_CONFIRMATION_STATES
            and row.confirmation_state != "rejected"
            and row.purpose not in NOT_SUPERSEDABLE_PURPOSES)


def can_correct(row: MemoryRow) -> bool:
    return (row.purpose not in SKIPPED_PURPOSES and bool(row.statement)
            and row.confirmation_state in CORRECTION_STATES)


def candidates_for(new: MemoryRow, corpus: Corpus, *, dense: Optional[Mapping[str, float]] = None,
                   k: int = K_PER_ROW, excluded: frozenset[tuple[str, str]] = frozenset()) -> list[Candidate]:
    """Older live rows ``new`` might correct, best first, at most ``k``. ``dense`` maps memory_id to
    cosine with ``new`` (None when the index is unavailable: then shared words/referents alone)."""
    if not can_correct(new):
        return []
    new_toks = corpus.toks(new)
    out: list[tuple[float, Candidate]] = []
    for old in corpus.rows:
        if old.memory_id == new.memory_id or not can_be_superseded(old):
            continue
        if new.episode_id and old.episode_id == new.episode_id:
            continue  # siblings from one episode (e.g. a re-distill) never retire each other
        if (old.memory_id, new.memory_id) in excluded:
            continue  # a pair someone undid is never proposed again
        if new.created_at and old.created_at and old.created_at >= new.created_at:
            continue  # a correction is newer than what it corrects
        rare_toks = tuple(sorted(t for t in new_toks & corpus.toks(old)
                                 if corpus.token_df.get(t, 0) <= RARE_TOKEN_MAX_ROWS))
        rare_refs = tuple(sorted(k_ for k_ in new.referents & old.referents
                                 if corpus.referent_df.get(k_, 0) <= RARE_REFERENT_MAX_ROWS))
        sim = None if dense is None else dense.get(old.memory_id)
        if dense is not None:
            if sim is None or sim < DENSE_FLOOR:
                continue
            if sim < DENSE_ALONE and not rare_toks and not rare_refs:
                continue
            score = sim + 0.02 * len(rare_toks) + 0.02 * len(rare_refs)
        else:
            if len(rare_toks) + len(rare_refs) < NO_DENSE_MIN_SHARED:
                continue
            score = len(rare_toks) + 1.5 * len(rare_refs)
        out.append((score, Candidate(old=old, dense=sim, shared_rare_tokens=rare_toks,
                                     shared_rare_referents=rare_refs)))
    out.sort(key=lambda x: (-x[0], x[1].old.memory_id))
    return [c for _, c in out[: max(0, int(k))]]


# --- the check -------------------------------------------------------------------------------

CHECK_PROMPT = """You check whether a newer memory CORRECTS an older one. Both are memories Orion
(an AI) wrote about conversations with Juniper. Be strict: a wrong "yes" hides a true memory.

A (older, {a_date}): {a}
B (newer, {b_date}): {b}

B corrects A only when B says that something A states is wrong, or gives a different value for
the same attribute of the same thing (same person, pet, place, object, or count), so that A should
no longer be believed.

Answer "no" when any of these hold:
- both can be true (B adds detail, narrows, or follows up on A);
- A was true at its own time and B is about a later or different time (a trip, a plan, a
  status that changed);
- they are about different things that only share a name or topic;
- B repeats or agrees with A;
- you are not sure.

Reply with one JSON object and nothing else:
{{"corrects": "yes" or "no", "sure": "yes" or "no", "reason": "<one short sentence>"}}
"""


# The second question, asked only after a sure "yes": a different framing of the same claim. On
# live pairs (2026-11-17) the first question alone said a sure "yes" to an overhead figure versus
# a whole-host figure (different attributes). Both must agree before anything is retired.
BOTH_TRUE_PROMPT = """Two memories Orion (an AI) wrote. Read each as true at its own date.

A ({a_date}): {a}
B ({b_date}): {b}

Could A and B both be true -- A at its date and B at its date -- about the same world? Different
times, different measurements, different parts of a thing, or extra detail all mean "yes". Say
"no" only if B states that what A says is (or was) wrong.

Reply with one JSON object and nothing else:
{{"both_true": "yes" or "no", "reason": "<one short sentence>"}}
"""


def build_check_prompt(old: MemoryRow, new: MemoryRow) -> str:
    return CHECK_PROMPT.format(a_date=old.date, a=old.statement, b_date=new.date, b=new.statement)


def build_both_true_prompt(old: MemoryRow, new: MemoryRow) -> str:
    return BOTH_TRUE_PROMPT.format(a_date=old.date, a=old.statement, b_date=new.date, b=new.statement)


@dataclass(frozen=True)
class CheckVerdict:
    corrects: bool
    sure: bool
    reason: str
    error: Optional[str] = None
    # The second question's answer: None = not asked (the first said no / unsure / failed).
    both_true: Optional[bool] = None
    both_true_reason: str = ""

    @property
    def first_yes(self) -> bool:
        return self.corrects and self.sure and bool(self.reason) and self.error is None

    @property
    def accepted(self) -> bool:
        return self.first_yes and self.both_true is False


def _yes(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    v = str(value or "").strip().lower()
    if v in ("yes", "true"):
        return True
    if v in ("no", "false"):
        return False
    return None


def _json_object(text: str) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(text or "").strip(), flags=re.MULTILINE).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return None, "no_json_object"
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None, "json_invalid"
    if not isinstance(data, dict):
        return None, "json_not_object"
    return data, None


def parse_check(text: str) -> CheckVerdict:
    """The first answer -> a verdict. Anything but one well-formed object is a refusal."""
    data, error = _json_object(text)
    if data is None:
        return CheckVerdict(False, False, "", error=error)
    corrects, sure = _yes(data.get("corrects")), _yes(data.get("sure"))
    reason = " ".join(str(data.get("reason") or "").split())[:400]
    if corrects is None or sure is None:
        return CheckVerdict(False, False, reason, error="bad_answer_value")
    return CheckVerdict(corrects, sure, reason)


def parse_both_true(text: str) -> tuple[Optional[bool], str]:
    """The second answer -> (both_true, reason). Unparseable = None (treated as "yes": no link)."""
    data, _ = _json_object(text)
    if data is None:
        return None, ""
    return _yes(data.get("both_true")), " ".join(str(data.get("reason") or "").split())[:400]


AskFn = Callable[[str], Awaitable[str]]
"""prompt -> the model's raw answer text (raises on a failed call)."""


@dataclass(frozen=True)
class Link:
    old_id: str
    new_id: str
    reason: str
    dense: Optional[float] = None
    shared: tuple[str, ...] = ()
    check_version: str = CHECK_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "shared": list(self.shared)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Link":
        return cls(old_id=str(d["old_id"]), new_id=str(d["new_id"]), reason=str(d.get("reason") or ""),
                   dense=d.get("dense"), shared=tuple(d.get("shared") or ()),
                   check_version=str(d.get("check_version") or CHECK_VERSION))


@dataclass
class CheckRecord:
    """One asked pair (kept for the pass report; statements are never stored in git or events)."""
    old_id: str
    new_id: str
    verdict: CheckVerdict
    dense: Optional[float]
    shared: tuple[str, ...]


async def run_check(old: MemoryRow, new: MemoryRow, ask: "AskFn",
                    budget: Optional[list[int]] = None) -> CheckVerdict:
    """Question 1 ("does B correct A?"); only on a sure yes, question 2 ("can both be true?").
    Linked only when Q1 = yes + sure and Q2 = no. A failed call is a refusal, never a link."""
    try:
        verdict = parse_check(await ask(build_check_prompt(old, new)))
    except Exception as exc:  # noqa: BLE001
        return CheckVerdict(False, False, "", error=f"call_failed:{type(exc).__name__}")
    if not verdict.first_yes:
        return verdict
    if budget is not None:
        if budget[0] <= 0:
            return replace(verdict, both_true=None, both_true_reason="budget_spent")
        budget[0] -= 1
    try:
        both, why = parse_both_true(await ask(build_both_true_prompt(old, new)))
    except Exception as exc:  # noqa: BLE001
        return replace(verdict, both_true=None, both_true_reason=f"call_failed:{type(exc).__name__}")
    return replace(verdict, both_true=both, both_true_reason=why)


async def check_pairs(new: MemoryRow, cands: Sequence[Candidate], ask: AskFn, *,
                      budget: Optional[list[int]] = None) -> tuple[list[Link], list[CheckRecord]]:
    """Ask the check for each candidate. ``budget`` is a one-item mutable counter of calls left
    (shared across rows); at 0 no more calls are made. A failed call is a refusal, never a link."""
    links: list[Link] = []
    records: list[CheckRecord] = []
    for c in cands:
        if budget is not None:
            if budget[0] <= 0:
                break
            budget[0] -= 1
        verdict = await run_check(c.old, new, ask, budget)
        shared = c.shared_rare_referents + c.shared_rare_tokens
        records.append(CheckRecord(c.old.memory_id, new.memory_id, verdict, c.dense, shared))
        if (verdict.error or "").startswith("call_failed") or verdict.both_true_reason.startswith("call_failed"):
            # The model call failed (a lost hold, an operator pause, a dead gateway): stop asking.
            if budget is not None:
                budget[0] = 0
            break
        if verdict.accepted:
            links.append(Link(old_id=c.old.memory_id, new_id=new.memory_id, reason=verdict.reason,
                              dense=None if c.dense is None else round(float(c.dense), 4), shared=shared))
    return links, records


def dedupe_links(links: Iterable[Link]) -> list[Link]:
    """One link per old row (the first found), and never a row that is itself retired by the same
    plan acting as a correction of another (a chain inside one plan is ambiguous: keep the first)."""
    out: list[Link] = []
    seen_old: set[str] = set()
    for link in links:
        if link.old_id in seen_old or link.old_id == link.new_id:
            continue
        if any(link.new_id == l.old_id or link.old_id == l.new_id for l in out):
            continue
        seen_old.add(link.old_id)
        out.append(link)
    return out


# --- writes (psycopg async connection, inside the caller's transaction) -----------------------

def superseded_event_id(old_id: str, new_id: str) -> str:
    return str(uuid.uuid5(SUPERSEDE_NAMESPACE, f"superseded|{old_id}|{new_id}"))


async def _fetchone(conn: Any, sql: str, params: Sequence[Any]) -> Optional[dict[str, Any]]:
    cur = await conn.execute(sql, params)
    row = await cur.fetchone()
    if row is None:
        return None
    return row if isinstance(row, dict) else dict(zip([d.name for d in cur.description], row))


async def apply_link(conn: Any, link: Link, *, actor: str, now: Optional[datetime] = None,
                     episode_id: Optional[str] = None) -> Optional[dict[str, Any]]:
    """Write the three marks for one pair. Re-checks live state first: the old row must still be
    active and supersedable, the new row active, and the pair never undone before. Returns what
    was written, or None when nothing was (the caller counts it as skipped)."""
    from psycopg.types.json import Jsonb

    now = now or datetime.now(timezone.utc)
    # Both rows locked in one statement, in id order (the pass and the distiller may run together).
    cur = await conn.execute("SELECT memory_id::text AS memory_id, status, confirmation_state, purpose, episode_id, "
                             "supersedes_memory_id::text AS supersedes_memory_id FROM episode_memory "
                             "WHERE memory_id = ANY(%s::uuid[]) ORDER BY memory_id FOR UPDATE",
                             ([link.old_id, link.new_id],))
    locked = {}
    for r in await cur.fetchall():
        r = r if isinstance(r, dict) else dict(zip([d.name for d in cur.description], r))
        locked[r["memory_id"]] = r
    old, new = locked.get(link.old_id), locked.get(link.new_id)
    if not old or not new or link.old_id == link.new_id or old["status"] != "active" or new["status"] != "active":
        return None
    if not can_be_superseded(MemoryRow(memory_id=link.old_id, statement="", purpose=old["purpose"],
                                       created_at=None, confirmation_state=old["confirmation_state"])):
        return None
    if new["confirmation_state"] not in CORRECTION_STATES:
        return None
    if old["episode_id"] and old["episode_id"] == new["episode_id"]:
        return None
    undone = await _fetchone(conn, "SELECT 1 AS x FROM episode_memory_event WHERE op = %s AND memory_id = %s "
                                   "AND evidence->>'superseded_by' = %s LIMIT 1",
                             (OP_UNDONE, link.old_id, link.new_id))
    if undone:
        return None  # Juniper (or an operator) put this pair back; never re-apply it
    await conn.execute("UPDATE episode_memory SET status = 'superseded', confirmation_state = 'corrected', "
                       "updated_at = %s WHERE memory_id = %s AND status = 'active'", (now, link.old_id))
    pointer_set = False
    if new["supersedes_memory_id"] is None:
        await conn.execute("UPDATE episode_memory SET supersedes_memory_id = %s, updated_at = %s "
                           "WHERE memory_id = %s AND supersedes_memory_id IS NULL", (link.old_id, now, link.new_id))
        pointer_set = True
    evidence = {"superseded_by": link.new_id, "prior_confirmation_state": old["confirmation_state"],
                "pointer_set": pointer_set, "check_version": link.check_version, "dense": link.dense,
                "shared": list(link.shared)}
    await conn.execute(
        "INSERT INTO episode_memory_event (event_id, memory_id, op, actor, episode_id, evidence, reason, created_at) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (event_id) DO NOTHING",
        (superseded_event_id(link.old_id, link.new_id), link.old_id, OP_SUPERSEDED, actor,
         episode_id or new.get("episode_id"), Jsonb(evidence), link.reason[:400], now),
    )
    return evidence


async def undo_link(conn: Any, *, old_id: str, new_id: str, actor: str, note: str = "",
                    now: Optional[datetime] = None) -> Optional[dict[str, Any]]:
    """Put a pair back: the old row active again with its prior confirmation_state, the new row's
    pointer cleared (only if it points at old_id), and one ``supersede_undone`` event. A later pass
    never re-applies an undone pair. Returns what was restored, or None when the pair is not
    superseded by this mechanism."""
    from psycopg.types.json import Jsonb

    now = now or datetime.now(timezone.utc)
    old = await _fetchone(conn, "SELECT status FROM episode_memory WHERE memory_id = %s FOR UPDATE", (old_id,))
    # Only the change this pair made can be undone: the row's latest status event must be this
    # pair's ``superseded`` (not a later revision by Juniper, another correction, or an undo).
    last = await _fetchone(conn, "SELECT op, evidence FROM episode_memory_event WHERE memory_id = %s "
                                 "AND op IN ('superseded', 'supersede_undone', 'revised', 'rejected') "
                                 "ORDER BY created_at DESC, event_id DESC LIMIT 1", (old_id,))
    if not old or old["status"] != "superseded" or not last or last["op"] != OP_SUPERSEDED:
        return None
    evidence = last["evidence"] if isinstance(last["evidence"], dict) else json.loads(last["evidence"] or "{}")
    if evidence.get("superseded_by") != new_id:
        return None
    prior = str(evidence.get("prior_confirmation_state") or "auto")
    await conn.execute("UPDATE episode_memory SET status = 'active', confirmation_state = %s, updated_at = %s "
                       "WHERE memory_id = %s AND status = 'superseded'", (prior, now, old_id))
    if evidence.get("pointer_set"):  # only clear a pointer this link set
        await conn.execute("UPDATE episode_memory SET supersedes_memory_id = NULL, updated_at = %s "
                           "WHERE memory_id = %s AND supersedes_memory_id = %s", (now, new_id, old_id))
    restored = {"superseded_by": new_id, "restored_confirmation_state": prior, "note": note[:400]}
    await conn.execute(
        "INSERT INTO episode_memory_event (event_id, memory_id, op, actor, episode_id, evidence, reason, created_at) "
        "VALUES (%s, %s, %s, %s, NULL, %s, %s, %s)",
        (str(uuid.uuid4()), old_id, OP_UNDONE, actor, Jsonb(restored), "supersede_undone", now),
    )
    return restored


# --- write time (the distiller) --------------------------------------------------------------

def cosine(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if not na or not nb or len(a) != len(b):
        return None
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


DenseFn = Callable[[str], Awaitable[Optional[dict[str, float]]]]
"""statement -> {memory_id: cosine} for episode_memory rows near it, or None (index unavailable)."""


async def plan_for_new_rows(new_rows: Sequence[MemoryRow], corpus: Corpus, ask: AskFn, *,
                            dense: Optional[DenseFn] = None, max_checks: int = 12,
                            excluded: frozenset[tuple[str, str]] = frozenset()) -> dict[str, Any]:
    """The distiller's step: candidates and checks for this episode's new rows. Returns a
    checkpointable dict: links (to apply in the persist transaction), counts, and any error."""
    budget = [max(0, int(max_checks))]
    links: list[Link] = []
    asked = refused = 0
    dense_unavailable = False
    for new in new_rows:
        scores = None
        if dense is not None and not dense_unavailable:
            try:
                scores = await dense(new.statement)
            except Exception:  # noqa: BLE001 -- fall back to shared words/referents
                scores = None
            if scores is None:
                dense_unavailable = True
        found, records = await check_pairs(new, candidates_for(new, corpus, dense=scores, excluded=excluded), ask,
                                           budget=budget)
        asked += len(records)
        refused += sum(1 for r in records if r.verdict.error)
        links.extend(found)
    return {"links": [l.as_dict() for l in dedupe_links(links)], "asked": asked, "check_errors": refused,
            "dense": "unavailable" if dense_unavailable else ("on" if dense is not None else "off"),
            "budget_left": budget[0]}


def index_dense(cfg: Any, *, n: int = 16) -> DenseFn:
    """A DenseFn over the lived-memory index (``orion_lived_memory``, the retriever's vectors):
    embed the statement, take the n nearest episode_memory docs. ``cfg`` is a
    ``orion.introspect.semantic_index.SearchConfig``. None when the index cannot answer."""
    async def run(statement: str) -> Optional[dict[str, float]]:
        import httpx

        from orion.introspect.semantic_index import SearchUnavailableError, embed, nearest
        from orion.memory.lived_memory import KIND_EPISODE, split_doc_id

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                vector, _ = await embed(client, cfg, statement, doc_prefix="supersede-check")
                hits = await nearest(client, cfg, vector, n, where={"kind": KIND_EPISODE})
        except SearchUnavailableError:
            return None
        out: dict[str, float] = {}
        for doc, sim in hits:
            parts = split_doc_id(doc)
            if parts and parts[0] == KIND_EPISODE:
                out[parts[1]] = float(sim)
        return out

    return run


def rows_from_validation(result: Any, *, now: datetime, episode_id: Optional[str] = None) -> list[MemoryRow]:
    """ValidationResult memories -> MemoryRow (created now, so every live row is older)."""
    return [
        MemoryRow(memory_id=m.memory_id, statement=" ".join(m.statement.split()), purpose=m.purpose,
                  created_at=now, confirmation_state=m.confirmation_state, episode_id=episode_id,
                  referents=frozenset(k for k, _ in m.referents), occurred_at=m.occurred_at)
        for m in result.memories
    ]


UNDONE_PAIRS_SQL = ("SELECT memory_id::text AS old_id, evidence->>'superseded_by' AS new_id "
                    "FROM episode_memory_event WHERE op = 'supersede_undone'")


async def undone_pairs(conn: Any) -> frozenset[tuple[str, str]]:
    """Pairs someone put back: never proposed or applied again."""
    if hasattr(conn, "fetch"):
        rows = [dict(r) for r in await conn.fetch(UNDONE_PAIRS_SQL)]
    else:
        cur = await conn.execute(UNDONE_PAIRS_SQL)
        rows = [r if isinstance(r, dict) else dict(zip([d.name for d in cur.description], r))
                for r in await cur.fetchall()]
    return frozenset((str(r["old_id"]), str(r["new_id"])) for r in rows)


async def load_corpus(conn: Any) -> Corpus:
    """Live rows through the serving gate (psycopg dict rows or asyncpg records)."""
    if hasattr(conn, "fetch"):
        rows = [dict(r) for r in await conn.fetch(LIVE_ROWS_SQL)]
    else:
        cur = await conn.execute(LIVE_ROWS_SQL)
        rows = [r if isinstance(r, dict) else dict(zip([d.name for d in cur.description], r))
                for r in await cur.fetchall()]
    return Corpus.build(row_from_db(r) for r in rows)
