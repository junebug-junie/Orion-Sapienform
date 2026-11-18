"""Corrections supersede what they correct (issues #3587, #3659, #3660; spec "Corrections: supersede at write time").

When a new ``episode_memory`` row updates or corrects an older one, the older row is retired, never
deleted, with bi-temporal validity (Zep/Graphiti style: when it stopped being true, and why):

1. the old row: ``status = 'superseded'``, ``invalid_at``, ``invalidated_by`` (the new row) and
   ``invalid_reason``:
   - ``update``: A was true and is no longer current; ``invalid_at`` = the new row's time.
     ``confirmation_state`` is kept (A was right at its date). Past-tense reads still reach it
     (``orion.memory.lived_memory.EPISODE_VALID_AT_SQL``).
   - ``correction``: A was never true; ``invalid_at`` = A's own time, ``confirmation_state =
     'corrected'``. Never served again.
2. the new row: ``supersedes_memory_id = <old memory_id>`` (when it points at nothing yet);
3. one ``episode_memory_event`` row: op ``superseded``, actor, reason, and in ``evidence`` the
   relation, ``invalid_at``, the votes, and the old row's prior ``confirmation_state`` (what undo
   restores).

Every reader that serves ``status = 'active'`` serves only what is currently true.

How a pair is found:

- **Candidates (deterministic).** Older rows the lived-memory retriever may serve
  (``orion.memory.lived_memory.EPISODE_SQL``, the serving gate), each needing BOTH a cosine at or
  above ``DENSE_FLOOR`` in the ``orion_lived_memory`` index AND a shared rare word or rare referent.
  At most ``K`` per row.
- **The classifier (local model, memory_distill route, JSON mode).** Mem0-style: "how does B
  relate to A?" -- update, correction, refine, compatible, or unrelated -- with "same attribute"
  and a confidence. Sampled ``VOTES`` times above temperature 0 (self-consistency). A pair acts only
  on a unanimous update or correction, every vote same-attribute and at least ``MIN_CONFIDENCE``.
  Anything else abstains; abstaining is always safe (both rows stay visible).
- **Juniper's own words.** A new row in Juniper's voice backed by a verified quote from her own
  chat prompt outranks the model's abstention: when every vote says update or correction (any
  confidence) and exactly one candidate qualifies (a clear target), it acts.

Never acts degraded (issue #3659). Write time applies a link only when the meaning index answered
for every new row, every model answer parsed, and the latest precision eval for this check version
passed (:func:`eval_gate`; ``scripts/memory_supersede_eval.py``). Otherwise the plan proposes and
holds its links, with the reasons in ``refused_because``. There is no word-only fallback for
applying; the pass's word-only candidates (``--no-dense`` or a dead index) are proposals its
``apply`` refuses. Every run leaves one append-only ``memory_supersede_run`` row (ids, legs,
votes, verdicts; never statements) -- :func:`record_run`, issue #3660.

Never superseded automatically: rows Juniper confirmed (her word outranks a model's judgment),
rows still being asked about, follow-up rows (a to-do is closed, not corrected) and Orion's own
views (a changed view is growth; the earlier one stays as history). Undo: :func:`undo_link`
(restores status and state, clears the validity end and the pointer).

Idempotency. A pair is handled once its ``superseded`` event exists: event_id =
uuid5(SUPERSEDE_NAMESPACE, "superseded|<old_id>|<new_id>") (:func:`superseded_event_id`). Applying
an already-superseded pair is a no-op: the old row is no longer active, so nothing is updated and no
second event is written (the insert is also ON CONFLICT DO NOTHING on that id). A pair with a
``supersede_undone`` event is never re-applied.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Iterable, Mapping, Optional, Sequence

from orion.memory.lived_memory import EPISODE_SQL

logger = logging.getLogger(__name__)

ACTOR_DISTILL = "memory.episode_distill"
ACTOR_PASS = "memory.supersede_pass"
OP_SUPERSEDED = "superseded"
OP_UNDONE = "supersede_undone"
CHECK_VERSION = "supersede_classify.v3.5"
SUPERSEDE_NAMESPACE = uuid.UUID("6f1d3c52-2b8e-4f7a-9d41-3587a0c0ffee")

# Candidate limits. bge-large cosines are compressed (live 2026-11-17: median pair 0.56, 90th
# percentile 0.65, 99th 0.73), so the floor sits near the 80th percentile and the "shares
# something specific" rule does the real narrowing.
K_PER_ROW = 3
DENSE_FLOOR = 0.60
# No "close enough alone" leg (issue #3659): a cosine alone never makes a pair. Live 11-17, most
# pairs at cosine >= 0.80 shared no word at all (different events on one topic).
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
       m.episode_id, m.confirmation_state, m.supersedes_memory_id::text AS supersedes_memory_id, m.voice,
       (m.voice = 'juniper_said' AND EXISTS (SELECT 1 FROM episode_memory_evidence e
            WHERE e.memory_id = m.memory_id AND e.source_kind = 'chat_prompt' AND e.verified)) AS juniper_quoted,
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
    voice: str = ""
    # Juniper's own words back this row: voice juniper_said with a verified quote from her chat
    # prompt (episode_memory_evidence source_kind chat_prompt). Her explicit correction outranks
    # a model's abstention (issue #3659).
    juniper_quoted: bool = False

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
        voice=str(r.get("voice") or ""),
        juniper_quoted=bool(r.get("juniper_quoted")),
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
    # How many live rows carry each shared token / referent (the "rare" leg's evidence), and this
    # candidate's place in its row's top K (1-based). Audit only (issue #3660).
    token_rows: tuple[int, ...] = ()
    referent_rows: tuple[int, ...] = ()
    rank: int = 0

    def legs(self, k: int = K_PER_ROW) -> dict[str, Any]:
        """Which candidate legs fired, without any text (the per-watched-row trace, #3660)."""
        return {"cosine": None if self.dense is None else round(float(self.dense), 4),
                "rare_tokens": len(self.shared_rare_tokens), "rare_token_rows": list(self.token_rows),
                "rare_referents": len(self.shared_rare_referents), "rare_referent_rows": list(self.referent_rows),
                "rank": self.rank, "k": k}


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
    cosine with ``new``. Every candidate needs both legs: a cosine at or above the floor AND a
    shared rare word or rare referent. ``dense`` None (index unavailable) = shared words/referents
    alone -- proposals only; nothing found that way is ever applied (issue #3659)."""
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
            if sim is None or sim < DENSE_FLOOR or (not rare_toks and not rare_refs):
                continue
            score = sim + 0.02 * len(rare_toks) + 0.02 * len(rare_refs)
        else:
            if len(rare_toks) + len(rare_refs) < NO_DENSE_MIN_SHARED:
                continue
            score = len(rare_toks) + 1.5 * len(rare_refs)
        out.append((score, Candidate(old=old, dense=sim, shared_rare_tokens=rare_toks,
                                     shared_rare_referents=rare_refs,
                                     token_rows=tuple(corpus.token_df.get(t, 0) for t in rare_toks),
                                     referent_rows=tuple(corpus.referent_df.get(r, 0) for r in rare_refs))))
    out.sort(key=lambda x: (-x[0], x[1].old.memory_id))
    return [replace(c, rank=i + 1) for i, (_, c) in enumerate(out[: max(0, int(k))])]


# --- answer parsing ---------------------------------------------------------------------------

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


AskFn = Callable[..., Awaitable[str]]
"""ask(prompt, *, temperature) -> the model's raw answer text (raises on a failed call)."""


# --- the classifier (check v3, issue #3659) ----------------------------------------------------
#
# Mem0-style classification instead of a yes/no: the local model names how B relates to A. Asked
# VOTES times at a temperature above 0 (self-consistency); a pair acts only on a unanimous,
# high-confidence update or correction whose checklist agrees (same referent, same attribute,
# B says A changed, a kind that relation can retire). Anything else abstains, and
# abstaining is always safe: both rows stay visible, which is the state before this module.

REL_UPDATE = "update"            # A was true; B says it has since changed (A no longer current)
REL_CORRECTION = "correction"    # A was never true
REL_REFINE = "refine"            # B adds detail to A; both stay
REL_COMPATIBLE = "compatible"    # both true, each at its own date
REL_UNRELATED = "unrelated"
RELATIONS = (REL_UPDATE, REL_CORRECTION, REL_REFINE, REL_COMPATIBLE, REL_UNRELATED)
ACTING_RELATIONS = frozenset({REL_UPDATE, REL_CORRECTION})

VOTES = 3
VOTE_TEMPERATURE = 0.7
MIN_CONFIDENCE = 0.8

# What A describes. Only a lasting state can be UPDATED (a later change makes it no longer current);
# an event, a plan, a measurement or count at a moment, an explicitly past period, or a view stays
# true at its own date. Eval 2026-11-18: without this field the model called a plan-then-what-
# happened pair an "update" and two counts on different days a "correction", unanimously.
A_KINDS = ("lasting_state", "event", "plan", "measurement", "past_period", "view")
UPDATABLE_KINDS = frozenset({"lasting_state"})
CORRECTABLE_KINDS = frozenset({"lasting_state", "event", "measurement"})

CLASSIFY_PROMPT = """You compare two memories that Orion (an AI) wrote about conversations with
Juniper. Decide how the NEWER memory B relates to the OLDER memory A. A wrong "update" or
"correction" hides a true memory, so when in doubt choose "compatible".

A (older, about {a_date}): {a}
B (newer, about {b_date}): {b}

First, what kind of thing does A describe?
- "lasting_state": something true now and ongoing as of A's date (where someone lives, a job, a
  pet's name or species, a preference, how something is set up).
- "event": one thing that happened once.
- "plan": something someone intends or is about to do.
- "measurement": a number, count, or reading at a moment or for a window of time.
- "past_period": something A itself places in an earlier period ("in 2019", "as a child").
- "view": an opinion or interpretation.

Second, does B itself say that A is wrong, or that A's state has since changed? Answer "yes" only
when B's own words say so (for example "not X", "actually", "was a mistake", "moved", "no longer",
"sold", "now"). A different event, a different day or window, or more detail is "no".

Third, is B about the SAME SPECIFIC THING as A, and the SAME ATTRIBUTE of it?
- "same_referent": the exact same item, not just the same project, host, person or topic word: the
  same PR or fix, the same pet, the same measured quantity over the same scope and time window.
  A slice of a measurement (one part, one overnight window, an overhead share) and a whole total
  are different referents. If A names a specific item (a PR or issue number, a named fix, a named
  pet or person) and B does not repeat that same name or number, answer "no": a general statement
  in B ("nothing is deployed yet", "my numbers were off") is not about A's item. If B could be
  about a different item, answer "no".
- "same_attribute": the same property of that item (species, home town, count, deployed or not).

Then pick exactly one relation:
- "correction": B says A was wrong from the start: a different value for the same attribute of
  the same specific referent, for the same time or occasion.
- "update": A was a lasting state, and B says that same attribute of the same referent has since
  changed for good.
- "refine": B adds detail to A or narrows it; both stay true.
- "compatible": both can be true, each at its own date. Examples: a trip or visit versus where
  someone lives; a plan versus what happened; counts or readings for different days or windows;
  different parts, people, or times; B repeats or agrees with A.
- "unrelated": different things that only share a word, a name, or a topic.

Reply with one JSON object and nothing else:
{{"a_kind": "<one of the six kinds>", "b_says_a_wrong_or_changed": "yes" or "no", "same_referent": "yes" or "no", "same_attribute": "yes" or "no", "relation": "<one of the five>", "confidence": <0.0 to 1.0>, "reason": "<one short sentence>"}}
"""


# A numbered item A names (PR #2968, issue 41): a correction or update must name the same number.
# Deterministic, after the votes (v3.3, 2026-11-18: the model said "same referent: yes" 3/3 for a
# general "nothing is deployed yet" against a named PR's deploy, even when the prompt said "no").
_NUMBERED = re.compile(r"(?:(?<!#)#|\b(?:prs?|pull requests?|issues?|tickets?)[\s#-]*)(\d{2,})\b",
                       re.IGNORECASE)


NUMBERED_ITEM_NOT_IN_B = "numbered_item_not_in_b"
NUMBERED_JUNIPER_ONLY = "numbered_item_juniper_path_only"


def numbered_items(text: str) -> frozenset[str]:
    return frozenset(str(int(n)) for n in _NUMBERED.findall(text or ""))


def same_numbered_item(old: "MemoryRow", new: "MemoryRow") -> bool:
    """False when A names numbered items and B names none of them (no shared referent possible)."""
    a = numbered_items(old.statement)
    return not a or bool(a & numbered_items(new.statement))


def build_classify_prompt(old: MemoryRow, new: MemoryRow) -> str:
    return CLASSIFY_PROMPT.format(a_date=old.date, a=old.statement, b_date=new.date, b=new.statement)


@dataclass(frozen=True)
class Vote:
    relation: Optional[str]
    same_attribute: Optional[bool]
    confidence: float
    reason: str = ""
    error: Optional[str] = None
    a_kind: Optional[str] = None
    b_says_changed: Optional[bool] = None
    # Same specific item, not just a shared project/host/topic word (v3.2, 2026-11-18: Juniper
    # rejected two unanimous v3.1 "corrections": an overnight overhead slice vs whole-host totals,
    # and one fix's deploy vs a different fix two weeks later).
    same_referent: Optional[bool] = None

    @property
    def acting(self) -> bool:
        """An update or correction whose own checklist agrees: same specific referent, same
        attribute, B's words say A is wrong or changed, and A is a kind that relation can retire."""
        if self.error is not None or self.relation not in ACTING_RELATIONS:
            return False
        if self.same_attribute is not True or self.b_says_changed is not True or self.same_referent is not True:
            return False
        kinds = UPDATABLE_KINDS if self.relation == REL_UPDATE else CORRECTABLE_KINDS
        return self.a_kind in kinds

    def audit(self) -> dict[str, Any]:
        return {"relation": self.relation, "same_attribute": self.same_attribute, "a_kind": self.a_kind,
                "b_says_changed": self.b_says_changed, "same_referent": self.same_referent, "confidence": round(self.confidence, 3),
                "reason": self.reason, "error": self.error}


def parse_vote(text: str) -> Vote:
    data, error = _json_object(text)
    if data is None:
        return Vote(None, None, 0.0, error=error)
    rel = str(data.get("relation") or "").strip().lower()
    same = _yes(data.get("same_attribute"))
    changed = _yes(data.get("b_says_a_wrong_or_changed"))
    kind = re.sub(r"[\s-]+", "_", str(data.get("a_kind") or "").strip().lower())
    reason = " ".join(str(data.get("reason") or "").split())[:300]
    raw_conf = data.get("confidence")
    try:
        conf = float(raw_conf) if not isinstance(raw_conf, bool) else float("nan")
    except (TypeError, ValueError):
        conf = float("nan")
    if not (0.0 <= conf <= 1.0):  # also catches NaN
        return Vote(rel or None, same, 0.0, reason, error="bad_confidence", a_kind=kind or None, b_says_changed=changed,
                    same_referent=_yes(data.get("same_referent")))
    if rel not in RELATIONS or same is None:
        return Vote(rel or None, same, conf, reason, error="bad_answer_value", a_kind=kind or None,
                    b_says_changed=changed, same_referent=_yes(data.get("same_referent")))
    # A malformed checklist field is not an error: such a vote can never act (Vote.acting needs a
    # known kind and a yes), so it abstains without holding the rest of the plan.
    return Vote(rel, same, conf, reason, a_kind=kind if kind in A_KINDS else None,
                b_says_changed=changed, same_referent=_yes(data.get("same_referent")))


@dataclass(frozen=True)
class Classification:
    """The self-consistent verdict for one pair. ``acts`` = the strict rule; ``juniper_acts`` =
    the rule for Juniper's own explicit words (every vote acting, confidence not required); the
    clear-target check for that path happens per new row in :func:`check_pairs`."""
    votes: tuple[Vote, ...]
    relation: Optional[str]
    acts: bool
    juniper_acts: bool
    abstain: str = ""          # why not (empty when acting)

    @property
    def error(self) -> Optional[str]:
        return next((v.error for v in self.votes if v.error), None)

    def audit(self) -> dict[str, Any]:
        return {"relation": self.relation, "acts": self.acts, "juniper_acts": self.juniper_acts,
                "abstain": self.abstain, "votes": [v.audit() for v in self.votes]}


def _majority(votes: Sequence[Vote]) -> Optional[str]:
    counts = Counter(v.relation for v in votes if v.relation)
    if not counts:
        return None
    (top, n), *rest = counts.most_common()
    return top if not rest or rest[0][1] < n else None


async def classify_pair(old: MemoryRow, new: MemoryRow, ask: "AskFn", *, votes: int = VOTES,
                        temperature: float = VOTE_TEMPERATURE, min_confidence: float = MIN_CONFIDENCE,
                        budget: Optional[list[int]] = None) -> Classification:
    """Up to ``votes`` samples. Stops early on the first vote that cannot lead to acting (any
    non-acting answer ends the strict path; for Juniper's words a non-acting answer ends hers
    too). A failed call or a malformed answer is an error vote: the pair abstains and the caller
    treats the plan as degraded."""
    # Juniper's own quoted words may correct a numbered item without restating its number: her
    # path still needs every vote acting, same_referent included; the strict path never bypasses.
    numbered_ok = same_numbered_item(old, new)
    if not numbered_ok and not new.juniper_quoted:
        # A names a numbered item B never mentions: no vote could act, so none is asked (no budget spent).
        return Classification(votes=(), relation=None, acts=False, juniper_acts=False,
                              abstain=NUMBERED_ITEM_NOT_IN_B)
    prompt = build_classify_prompt(old, new)
    got: list[Vote] = []
    for _ in range(max(1, int(votes))):
        if budget is not None:
            if budget[0] <= 0:
                break
            budget[0] -= 1
        try:
            vote = parse_vote(await ask(prompt, temperature=temperature))
        except Exception as exc:  # noqa: BLE001
            vote = Vote(None, None, 0.0, error=f"call_failed:{type(exc).__name__}")
        got.append(vote)
        if not vote.acting:
            break
    rel = _majority(got)
    complete = len(got) == max(1, int(votes))
    all_acting = complete and all(v.acting for v in got)
    unanimous = all_acting and len({v.relation for v in got}) == 1
    strict = unanimous and all(v.confidence >= min_confidence for v in got) and numbered_ok
    juniper = all_acting and new.juniper_quoted
    if any(v.error for v in got):
        why = "error"
    elif not complete and all(v.acting for v in got):
        why = "budget_spent"
    elif not all_acting:
        last = got[-1]
        why = f"checklist:{last.relation}" if last.relation in ACTING_RELATIONS else f"vote:{last.relation}"
    elif not numbered_ok:
        why = NUMBERED_JUNIPER_ONLY   # the model path cannot act; only Juniper's path may
    elif not unanimous:
        why = "split_update_correction"
    elif not strict:
        why = "low_confidence"
    else:
        why = ""
    # Juniper's path: when the votes split between update and correction, the old row was still
    # not current; record it as an update (the gentler mark: it stays reachable as history).
    relation = rel if rel in ACTING_RELATIONS else (REL_UPDATE if all_acting else rel)
    return Classification(votes=tuple(got), relation=relation, acts=strict, juniper_acts=juniper and not strict,
                          abstain=why if not strict else "")


@dataclass(frozen=True)
class Link:
    old_id: str
    new_id: str
    reason: str
    dense: Optional[float] = None
    shared: tuple[str, ...] = ()
    check_version: str = CHECK_VERSION
    relation: str = REL_CORRECTION     # update | correction: decides invalid_at (see apply_link)
    path: str = "model"                # model (unanimous, confident) | juniper (her explicit words)
    votes: tuple[str, ...] = ()        # each vote's relation, for the event's evidence

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "shared": list(self.shared), "votes": list(self.votes)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Link":
        rel = str(d.get("relation") or REL_CORRECTION)
        return cls(old_id=str(d["old_id"]), new_id=str(d["new_id"]), reason=str(d.get("reason") or ""),
                   dense=d.get("dense"), shared=tuple(d.get("shared") or ()),
                   check_version=str(d.get("check_version") or CHECK_VERSION),
                   relation=rel if rel in ACTING_RELATIONS else REL_CORRECTION,
                   path=str(d.get("path") or "model"), votes=tuple(d.get("votes") or ()))


@dataclass
class CheckRecord:
    """One asked pair. ``audit()`` is what the run manifest keeps: ids, legs, votes, verdict; never
    statements."""
    old_id: str
    new_id: str
    verdict: Classification
    candidate: Optional[Candidate] = None
    linked: bool = False

    @property
    def dense(self) -> Optional[float]:
        return None if self.candidate is None else self.candidate.dense

    @property
    def shared(self) -> tuple[str, ...]:
        c = self.candidate
        return () if c is None else c.shared_rare_referents + c.shared_rare_tokens

    def audit(self) -> dict[str, Any]:
        return {"old_id": self.old_id, "new_id": self.new_id,
                "legs": self.candidate.legs() if self.candidate else None,
                **self.verdict.audit(), "linked": self.linked}


def check_failed(v: Classification) -> bool:
    """A model call failed or an answer did not parse."""
    return v.error is not None


async def check_pairs(new: MemoryRow, cands: Sequence[Candidate], ask: AskFn, *,
                      budget: Optional[list[int]] = None,
                      votes: int = VOTES) -> tuple[list[Link], list[CheckRecord]]:
    """Classify each candidate. ``budget`` is a one-item mutable counter of model calls left
    (shared across rows). A failed call or malformed answer stops asking (the plan is then held).

    Links: every candidate with a unanimous, confident update/correction. Juniper's explicit words
    (``new.juniper_quoted``) outrank abstention: when no candidate passes the strict rule and exactly
    ONE candidate drew only update/correction votes, that one links (path "juniper"). Two or more
    such candidates is no clear target: nothing links. "Exactly one" is among the top ``K_PER_ROW``
    candidates asked to the end (a budget cut disables the path)."""
    links: list[Link] = []
    records: list[CheckRecord] = []
    juniper: list[tuple[Candidate, Classification]] = []
    for c in cands:
        if budget is not None and budget[0] <= 0:
            break
        verdict = await classify_pair(c.old, new, ask, votes=votes, budget=budget)
        rec = CheckRecord(c.old.memory_id, new.memory_id, verdict, c)
        records.append(rec)
        if check_failed(verdict):
            if budget is not None:
                budget[0] = 0
            break
        if verdict.acts:
            links.append(_link(c, new, verdict, "model"))
            rec.linked = True
        elif verdict.juniper_acts:
            juniper.append((c, verdict))
    # A clear target needs every candidate asked to the end: a budget cut could hide a second one.
    complete = len(records) == len(cands) and all(r.verdict.abstain != "budget_spent" for r in records)
    if not links and complete and len(juniper) == 1 and not any(check_failed(r.verdict) for r in records):
        c, verdict = juniper[0]
        links.append(_link(c, new, verdict, "juniper"))
        next(r for r in records if r.candidate is c).linked = True
    return links, records


def _link(c: Candidate, new: MemoryRow, v: Classification, path: str) -> Link:
    reason = next((x.reason for x in v.votes if x.reason), "")
    return Link(old_id=c.old.memory_id, new_id=new.memory_id, reason=reason,
                dense=None if c.dense is None else round(float(c.dense), 4),
                shared=c.shared_rare_referents + c.shared_rare_tokens,
                relation=v.relation if v.relation in ACTING_RELATIONS else REL_UPDATE, path=path,
                votes=tuple(str(x.relation) for x in v.votes))


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


def invalid_at_for(relation: str, old: Mapping[str, Any], new: Mapping[str, Any], now: datetime) -> datetime:
    """When the old fact stopped being true (bi-temporal valid time).

    update: it WAS true until the newer memory's time (``occurred_at``, else when it was written).
    correction: it was never true, so it is invalid from its own time."""
    old_start = _aware(old.get("occurred_at")) or _aware(old.get("created_at")) or now
    if relation == REL_UPDATE:
        return _aware(new.get("occurred_at")) or _aware(new.get("created_at")) or now
    return old_start


async def apply_link(conn: Any, link: Link, *, actor: str, now: Optional[datetime] = None,
                     episode_id: Optional[str] = None) -> Optional[dict[str, Any]]:
    """Write the marks for one pair. Re-checks live state first: the old row must still be
    active and supersedable, the new row active, and the pair never undone before. Returns what
    was written, or None when nothing was (the caller counts it as skipped).

    Marks: the old row ``status = 'superseded'`` with ``invalid_at`` / ``invalidated_by`` /
    ``invalid_reason`` (and ``confirmation_state = 'corrected'`` only for a correction: an update's
    old row was right at its time); the new row's pointer; one ``superseded`` event."""
    from psycopg.types.json import Jsonb

    now = now or datetime.now(timezone.utc)
    # Both rows locked in one statement, in id order (the pass and the distiller may run together).
    cur = await conn.execute("SELECT memory_id::text AS memory_id, status, confirmation_state, purpose, episode_id, "
                             "supersedes_memory_id::text AS supersedes_memory_id, occurred_at, created_at "
                             "FROM episode_memory WHERE memory_id = ANY(%s::uuid[]) ORDER BY memory_id FOR UPDATE",
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
    relation = link.relation if link.relation in ACTING_RELATIONS else REL_CORRECTION
    invalid_at = invalid_at_for(relation, old, new, now)
    old_start = _aware(old.get("occurred_at")) or _aware(old.get("created_at")) or now
    if relation == REL_UPDATE and invalid_at <= old_start:
        return None  # "since changed" cannot point backwards: an empty validity window would hide history
    state = "corrected" if relation == REL_CORRECTION else old["confirmation_state"]
    await conn.execute("UPDATE episode_memory SET status = 'superseded', confirmation_state = %s, invalid_at = %s, "
                       "invalidated_by = %s, invalid_reason = %s, updated_at = %s "
                       "WHERE memory_id = %s AND status = 'active'",
                       (state, invalid_at, link.new_id, relation, now, link.old_id))
    pointer_set = False
    if new["supersedes_memory_id"] is None:
        await conn.execute("UPDATE episode_memory SET supersedes_memory_id = %s, updated_at = %s "
                           "WHERE memory_id = %s AND supersedes_memory_id IS NULL", (link.old_id, now, link.new_id))
        pointer_set = True
    evidence = {"superseded_by": link.new_id, "prior_confirmation_state": old["confirmation_state"],
                "pointer_set": pointer_set, "check_version": link.check_version, "dense": link.dense,
                "shared": list(link.shared), "relation": relation, "invalid_at": invalid_at.isoformat(),
                "path": link.path, "votes": list(link.votes)}
    await conn.execute(
        "INSERT INTO episode_memory_event (event_id, memory_id, op, actor, episode_id, evidence, reason, created_at) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (event_id) DO NOTHING",
        (superseded_event_id(link.old_id, link.new_id), link.old_id, OP_SUPERSEDED, actor,
         episode_id or new.get("episode_id"), Jsonb(evidence), link.reason[:400], now),
    )
    return evidence


async def undo_link(conn: Any, *, old_id: str, new_id: str, actor: str, note: str = "",
                    now: Optional[datetime] = None) -> Optional[dict[str, Any]]:
    """Put a pair back: the old row active again with its prior confirmation_state and no
    validity end, the new row's pointer cleared (only if it points at old_id), and one
    ``supersede_undone`` event. A later pass never re-applies an undone pair. Returns what was
    restored, or None when the pair is not superseded by this mechanism."""
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
    await conn.execute("UPDATE episode_memory SET status = 'active', confirmation_state = %s, invalid_at = NULL, "
                       "invalidated_by = NULL, invalid_reason = NULL, updated_at = %s "
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


# Why a plan's links are held instead of applied (issue #3659). A degraded plan never acts: the
# candidate search lost its meaning leg, or a model answer failed, or the precision eval has not
# passed for this check version. Held links are kept in the plan for the audit trail only.
REFUSED_DENSE_OFF = "dense_off"                  # no meaning index configured
REFUSED_DENSE_UNAVAILABLE = "dense_unavailable"  # the index failed or could not answer for a row
REFUSED_CHECK_ERRORS = "check_errors"            # a model call failed or an answer did not parse
REFUSED_EVAL_GATE = "eval_gate"                  # no passing precision eval for this check version


async def plan_for_new_rows(new_rows: Sequence[MemoryRow], corpus: Corpus, ask: AskFn, *,
                            dense: Optional[DenseFn] = None, max_checks: int = 12,
                            excluded: frozenset[tuple[str, str]] = frozenset(),
                            apply_gate: tuple[bool, str] = (False, "eval_gate_not_checked")) -> dict[str, Any]:
    """The distiller's step: candidates and classifications for this episode's new rows. Returns a
    checkpointable dict: ``links`` (to apply in the persist transaction), counts, why apply was
    refused (if it was), and ``decisions`` (every asked pair's ids, legs and votes, no text: the run
    manifest, #3660). ``max_checks`` caps MODEL CALLS (each pair takes 1 to ``VOTES`` calls).

    Never acts degraded (issue #3659): without a working meaning index there is no word-only
    fallback. When the index is off or fails, no more pairs are asked and nothing is applied.
    When any model answer fails or does not parse, nothing is applied. When ``apply_gate`` is
    closed (no passing precision eval for this check version), the plan still asks and proposes,
    but its links go to ``held_links``, not ``links``. ``apply`` is "on" only when none of these
    hold; the persist step applies nothing unless it reads ``apply == "on"``."""
    budget = [max(0, int(max_checks))]
    links: list[Link] = []
    decisions: list[dict[str, Any]] = []
    asked = refused = 0
    dense_ms = dense_calls = 0
    reasons: list[str] = []
    if dense is None:
        reasons.append(REFUSED_DENSE_OFF)
    else:
        for new in new_rows:
            if budget[0] <= 0:
                break  # no calls left: an index fault on a row never asked must not hold proven links
            started = time.monotonic()
            try:
                scores = await dense(new.statement)
            except Exception as exc:  # noqa: BLE001 -- no fallback: a degraded search never acts
                scores, why = None, f"{REFUSED_DENSE_UNAVAILABLE}:{type(exc).__name__}"
            else:
                why = REFUSED_DENSE_UNAVAILABLE
            finally:
                dense_ms += int((time.monotonic() - started) * 1000)
                dense_calls += 1
            if scores is None:
                reasons.append(why)
                break
            found, records = await check_pairs(new, candidates_for(new, corpus, dense=scores, excluded=excluded),
                                               ask, budget=budget)
            asked += len(records)
            refused += sum(1 for r in records if check_failed(r.verdict))
            decisions.extend(r.audit() for r in records)
            links.extend(found)
    if refused:
        reasons.append(REFUSED_CHECK_ERRORS)
    gate_open, gate_why = apply_gate
    if not gate_open:
        reasons.append(f"{REFUSED_EVAL_GATE}:{gate_why}")
    planned = [l.as_dict() for l in dedupe_links(links)]
    dense_state = "off" if dense is None else (
        "unavailable" if any(r.startswith(REFUSED_DENSE_UNAVAILABLE) for r in reasons) else "on")
    return {"links": [] if reasons else planned, "held_links": planned if reasons else [],
            "apply": "refused" if reasons else "on", "refused_because": reasons,
            "asked": asked, "check_errors": refused, "dense": dense_state, "budget_left": budget[0],
            "check_version": CHECK_VERSION, "decisions": decisions,
            # The meaning-index lookups: how many, total wall time, and how many retries fired
            # (index_dense exposes its retry count; a failed retry shows as refused_because dense_unavailable).
            "dense_calls": dense_calls, "dense_ms": dense_ms,
            "dense_retries": int(getattr(dense, "retries", lambda: 0)()) if dense is not None else 0,
            # Why each asked pair did not link, counted (the numbered-item rule shows as its own reason).
            "abstained": dict(Counter(d["abstain"] for d in decisions if d.get("abstain") and not d.get("linked")))}


RECORD_RUN_SQL = """
INSERT INTO memory_supersede_run (run_id, source, episode_id, check_version, started_at, finished_at, dense, apply,
                                  refused_because, counts, decisions, applied)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (run_id) DO NOTHING
"""


async def record_run(conn: Any, *, run_id: str, source: str, plan: Mapping[str, Any],
                     applied: Sequence[Mapping[str, Any]] = (), episode_id: Optional[str] = None,
                     started_at: Optional[datetime] = None, finished_at: Optional[datetime] = None) -> None:
    """One append-only ``memory_supersede_run`` row (issue #3660): ids, legs, votes and verdicts
    of every asked pair, why apply was refused, and what was written. Never statements. The same
    run_id twice writes nothing (a re-persist after a crash)."""
    from psycopg.types.json import Jsonb

    counts = {k: plan.get(k) for k in ("asked", "check_errors", "budget_left") if k in plan}
    counts["links"] = len(plan.get("links") or [])
    counts["held_links"] = len(plan.get("held_links") or [])
    counts["applied"] = len(applied)
    for k in ("dense_calls", "dense_ms", "dense_retries"):
        if k in plan:
            counts[k] = plan[k]
    # Every path (nightly distill, pass dry-run, pass apply with per-pair decisions) gets the same
    # count, derived from decisions when the caller did not give one.
    abstained = plan.get("abstained")
    if abstained is None:
        abstained = dict(Counter(d.get("abstain") for d in (plan.get("decisions") or [])
                                 if isinstance(d, Mapping) and d.get("abstain") and not d.get("linked")))
    if abstained:
        counts["abstained"] = dict(abstained)
    # The hand-run pass apply asks no model: its per-pair decisions carry what apply did instead
    # ("applied", "already_handled", "undone_never_reapplied", ...). Counted the same way.
    outcomes = Counter(d.get("outcome") for d in (plan.get("decisions") or [])
                       if isinstance(d, Mapping) and d.get("outcome"))
    if outcomes:
        counts["outcomes"] = dict(outcomes)
    if plan.get("error"):
        counts["error"] = str(plan["error"])
    await conn.execute(RECORD_RUN_SQL, (
        run_id, source, episode_id, str(plan.get("check_version") or CHECK_VERSION), started_at,
        finished_at or datetime.now(timezone.utc), plan.get("dense"), plan.get("apply") or "refused",
        list(plan.get("refused_because") or ([f"node_error:{plan['error']}"] if plan.get("error") else [])),
        Jsonb(counts), Jsonb(list(plan.get("decisions") or [])), Jsonb(list(applied))))


def links_to_apply(plan: Optional[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The only links the persist step may apply: a plan that says ``apply == "on"`` (so a
    checkpoint written before #3659, or a degraded plan, applies nothing)."""
    if not plan or plan.get("apply") != "on" or plan.get("refused_because"):
        return []
    # Defense in depth: only links from the current check version, with a relation it names.
    return [l for l in (plan.get("links") or [])
            if l.get("check_version") == CHECK_VERSION and l.get("relation") in ACTING_RELATIONS]


# The precision eval's verdict, read before write-time apply (issue #3659). The table is created by
# the eval's migration; until it exists (or while its latest run for this check version did not
# pass), write time proposes only.
# A pass expires: the model behind the memory_distill route can change under a fixed check version.
EVAL_MAX_AGE_DAYS = 30
EVAL_ROUTE = "memory_distill"   # the route write time asks on
EVAL_GATE_SQL = ("SELECT passed, check_version, finished_at FROM memory_supersede_eval_run "
                 "WHERE check_version = %s AND votes = %s AND route = %s AND finished_at IS NOT NULL "
                 "ORDER BY finished_at DESC LIMIT 1")


async def eval_gate(conn: Any, *, check_version: str = CHECK_VERSION, votes: int = VOTES,
                    now: Optional[datetime] = None) -> tuple[bool, str]:
    """(open, why). Open only when the most recent finished eval run for ``check_version`` at the
    production vote count and route passed, less than ``EVAL_MAX_AGE_DAYS`` ago. Any read failure (table
    missing, permissions, a dead connection) reads as closed."""
    try:
        row = await _fetchone(conn, EVAL_GATE_SQL, (check_version, int(votes), EVAL_ROUTE))
    except Exception as exc:  # noqa: BLE001 -- unknown is closed
        return False, f"eval_unreadable:{type(exc).__name__}"
    if row is None:
        return False, "eval_missing"
    if not row.get("passed"):
        return False, "eval_failed"
    finished = _aware(row.get("finished_at"))
    if finished is None or (now or datetime.now(timezone.utc)) - finished > timedelta(days=EVAL_MAX_AGE_DAYS):
        return False, "eval_stale"
    return True, "eval_passed"


DENSE_ATTEMPTS = 2
DENSE_TIMEOUT_SEC = 40.0
DENSE_RETRY_SLEEP_SEC = 5.0


def index_dense(cfg: Any, *, n: int = 16) -> DenseFn:
    """A DenseFn over the lived-memory index (``orion_lived_memory``, the retriever's vectors):
    embed the statement, take the n nearest episode_memory docs. ``cfg`` is a
    ``orion.introspect.semantic_index.SearchConfig``. None when the index cannot answer."""
    retried = [False]   # one retry per plan (index_dense is built per plan), not per row
    retry_count = [0]

    async def run(statement: str) -> Optional[dict[str, float]]:
        import httpx

        from orion.introspect.semantic_index import SearchUnavailableError, embed, nearest
        from orion.memory.lived_memory import KIND_EPISODE, split_doc_id

        hits = None
        for attempt in range(DENSE_ATTEMPTS):
            try:
                # The bound covers the whole attempt (embed + every Chroma call), not each phase.
                async with asyncio.timeout(DENSE_TIMEOUT_SEC):
                    async with httpx.AsyncClient(timeout=DENSE_TIMEOUT_SEC) as client:
                        vector, _ = await embed(client, cfg, statement, doc_prefix="supersede-check")
                        hits = await nearest(client, cfg, vector, n, where={"kind": KIND_EPISODE})
                break
            except (SearchUnavailableError, TimeoutError) as exc:
                # The plan only records "dense_unavailable"; the log keeps why (issue #3693: a live
                # write-time run refused this way with no cause on record).
                logger.warning("supersede_dense_unavailable attempt=%s error=%s: %s", attempt + 1,
                               type(exc).__name__, str(exc)[:300])
                # 2026-11-18 11:10Z: Chroma's collection lookup stalled 25 s once (20 s timeout) and the
                # night's check refused as dense_unavailable. ONE retry per plan, only for a transient
                # fault; still no fallback if it fails.
                transient = isinstance(exc, TimeoutError) or "unavailable:" in str(exc)
                if attempt + 1 < DENSE_ATTEMPTS and transient and not retried[0]:
                    retried[0] = True
                    retry_count[0] += 1
                    await asyncio.sleep(DENSE_RETRY_SLEEP_SEC)
                    continue
                break
        if hits is None:
            return None
        out: dict[str, float] = {}
        for doc, sim in hits:
            parts = split_doc_id(doc)
            if parts and parts[0] == KIND_EPISODE:
                out[parts[1]] = float(sim)
        return out

    run.retries = lambda: retry_count[0]  # type: ignore[attr-defined]  # read by plan_for_new_rows
    return run


def rows_from_validation(result: Any, *, now: datetime, episode_id: Optional[str] = None) -> list[MemoryRow]:
    """ValidationResult memories -> MemoryRow (created now, so every live row is older)."""
    return [
        MemoryRow(memory_id=m.memory_id, statement=" ".join(m.statement.split()), purpose=m.purpose,
                  created_at=now, confirmation_state=m.confirmation_state, episode_id=episode_id,
                  referents=frozenset(k for k, _ in m.referents), occurred_at=m.occurred_at, voice=m.voice,
                  juniper_quoted=(m.voice == "juniper_said"
                                  and any(e.source_kind == "chat_prompt" and e.verified for e in m.evidence)))
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
