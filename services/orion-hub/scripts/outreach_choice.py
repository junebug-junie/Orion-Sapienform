"""The outreach choice point: Orion picks which topic to bring to Juniper, or says nothing.

Personality spec rev 4, move 1 (docs/superpowers/specs/2026-10-16-orion-dynamic-personality-graph-design.md):
at a choice point Orion is shown the top options a funded act would take and picks one, or picks
"none". The background outreach tick used to compose one message from whatever material it had.
Now the SAME compose call (no new model call) shows Orion up to ``max_options`` grounded topics
plus "say nothing". Orion's reply starts with ``Pick: <n>``; ``Pick: 0`` or PASS declines.

Every gate the tick already had still runs before this point (enabled, turn in flight, recent chat,
quiet hours, daily cap, cooldown, rest drive, arousal, the body-economy bid). Those gates decide
whether the tick may compose at all. The per-topic gates below decide which topics may be shown:

* a topic whose subject Juniper already got (or Orion already declined) in the last 7 days is not
  shown, unless its version changed (``filter_unsaid``). A subject is the source's own id (the
  daydream id, the read's seed id), never a hash of the message text. This is what the old tick
  lacked: on 2026-11-11..12 it sent the same "graph first" habit 3 times, because the habit rode in
  as free prompt material while only the daydream had an id in the ledger.
* each topic source applies its own gates (the readings door's spacing, its told ledger).

Pure: no DB, no bus, no clock reads (callers pass ``now``).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from orion.schemas.personality import (
    CHOICE_OPTION_TEXT_MAX,
    CHOICE_SITUATION_MAX_KEYS,
    CHOICE_SITUATION_TEXT_MAX,
    ChoiceOptionV1,
    PersonalityChoiceV1,
)

# Decision-row reasons this module's choice point produces (``endogenous_outreach_decisions.reason``).
# Orion was shown topics and chose to say nothing (``Pick: 0`` or PASS). Nothing is sent.
ORION_DECLINED_REASON = "orion_declined"
# Orion wrote a message to Juniper but no ``Pick:`` line, with two or more topics shown: we cannot
# tell which subject it is about, so it is not sent (the subject ledger would be wrong).
NO_PICK_REASON = "no_pick"
# ``Pick: <n>`` named a number that was not shown.
INVALID_PICK_REASON = "invalid_pick"

# Same 7-day window as the tick's content-novelty ledger.
SUBJECT_LOOKBACK_DAYS = 7
DEFAULT_MAX_OPTIONS = 4
MIN_OPTIONS, MAX_OPTIONS = 1, 5

# Reasons whose row says "Orion already decided about this subject" (read by ``fold_said_subjects``).
# sent / no_why_juniper: the chosen subject. orion_declined / no_pick / invalid_pick: every subject shown.
CHOSEN_SUBJECT_REASONS = frozenset({"sent", "no_why_juniper"})
# no_pick / invalid_pick: Orion wrote something we could not read as a pick. Counted as decided for
# the shown subjects, so a model that keeps dropping the Pick line does not recompose the same
# topics on every tick.
DECLINED_SUBJECT_REASONS = frozenset({ORION_DECLINED_REASON, NO_PICK_REASON, INVALID_PICK_REASON})


@dataclass(frozen=True)
class OutreachTopic:
    """One thing Orion could bring to Juniper this tick.

    ``subject``: the source's stable id (``daydream:<id>``, ``reading:<seed_id>``). It is also the
    option key in the choice record. ``version``: what makes the same subject new again (a read's
    finding ids: a belief moved days later is a new version). "" when the subject never changes.
    ``option_text``: what the option says, as Orion saw it (<= 400 chars; the personality reducer
    embeds it). ``prompt_text``: the fuller line the compose prompt shows. ``observed_at``: when the
    material appeared; the shown order is freshest first. ``appendix``: stored record text delivered
    under Orion's message when this topic is picked (a read's article summary). ``finding_ids``:
    recorded on the decision row so the readings door treats them as told. ``on_sent``: called
    once after delivery (the readings door updates its in-process told set)."""

    subject: str
    kind: str
    option_text: str
    prompt_text: str
    version: str = ""
    observed_at: Optional[datetime] = None
    appendix: str = ""
    finding_ids: Tuple[str, ...] = ()
    seed_id: str = ""
    on_sent: Optional[Callable[[], None]] = field(default=None, compare=False, repr=False)
    # Called once when Orion declined every topic at the tick (the readings door marks the read told,
    # as it does for its own orion_passed).
    on_declined: Optional[Callable[[], None]] = field(default=None, compare=False, repr=False)


def version_of(ids: Iterable[str]) -> str:
    """Stable version string for a set of ids (order-free). "" for no ids."""
    clean = sorted({str(i).strip() for i in ids if str(i or "").strip()})
    if not clean:
        return ""
    return hashlib.sha256("|".join(clean).encode("utf-8")).hexdigest()[:16]


def clip(text: str, cap: int) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= cap else s[: cap - 1].rstrip() + "…"


def filter_unsaid(
    topics: Sequence[OutreachTopic], said: Mapping[str, Set[str]]
) -> Tuple[List[OutreachTopic], List[str]]:
    """(topics still worth showing, subjects dropped). A topic is dropped when its subject is in
    ``said`` with the same version. A new version (material change) is shown again."""
    keep: List[OutreachTopic] = []
    dropped: List[str] = []
    for t in topics:
        if t.version in said.get(t.subject, set()):
            dropped.append(t.subject)
            continue
        keep.append(t)
    return keep, dropped


def rank_topics(topics: Sequence[OutreachTopic], max_options: int) -> List[OutreachTopic]:
    """Freshest first (no timestamp last), one per subject, at most ``max_options``.

    The shown order is the producer's ranking: ``market_rank`` 1 is the freshest topic. There is
    no market price for these yet; the personality reducer weighs a pick of rank 1 at 0.25."""
    k = max(MIN_OPTIONS, min(MAX_OPTIONS, int(max_options)))
    floor = datetime.min.replace(tzinfo=timezone.utc)
    seen: Set[str] = set()
    unique: List[OutreachTopic] = []
    for t in topics:
        if t.subject and t.subject not in seen:
            seen.add(t.subject)
            unique.append(t)
    ordered = sorted(
        unique,
        key=lambda t: (t.observed_at is not None, t.observed_at or floor),
        reverse=True,
    )
    return ordered[:k]


def choice_block(topics: Sequence[OutreachTopic]) -> List[str]:
    """Prompt lines: the numbered topics, "0. Say nothing", and how to answer."""
    lines = [
        "What you could bring to Juniper right now. Each is something real from your own day. "
        "Pick at most one -- the one you actually want her to hear about, which need not be the "
        "first -- or none:",
    ]
    lines.extend(f"{i}. {t.prompt_text}" for i, t in enumerate(topics, start=1))
    lines.append("0. Say nothing to her this time.")
    lines.append("")
    lines.append(
        'Start your reply with one line: "Pick: <number>". "Pick: 0" means you say nothing to '
        "Juniper now; then write nothing else. Otherwise write the message after that line, "
        "about the one thing you picked only."
    )
    return lines


# "Pick: 2", "**Pick:** 2", "My pick — 2", "Pick: say nothing". Without a separator the value must
# end the line ("Pick 2"), so a sentence like "Pick 3 things matter here" is not a pick.
_PICK_LINE = re.compile(
    r"^[\s*_`>#-]*(?:my\s+)?pick[\s*_`]*(?P<sep>[:=\-\u2013\u2014])?[\s*_`\[(#]*(?:option\s*|number\s*|#)?"
    r"(?P<val>\d+|none|nothing|say\s+nothing|zero)\b(?P<rest>.*)$",
    re.IGNORECASE,
)
# Any line that starts like a pick line. With one topic shown, a reply holding such a line that
# did not parse is not an implicit pick: its Pick line would reach Juniper.
_PICK_ATTEMPT = re.compile(r"^\W*(?:my\s+)?pick\b", re.IGNORECASE)


@dataclass(frozen=True)
class Pick:
    """``kind``: "chosen" (``index`` 0-based), "declined", "missing", "invalid". ``body``: the reply
    with the Pick line removed. ``shape``: "line" (an explicit Pick line), "implicit" (one topic,
    no Pick line, a message), "pass" (PASS), "zero" (Pick: 0 / none)."""

    kind: str
    index: Optional[int]
    body: str
    shape: str


def parse_pick(text: str, n_options: int, *, is_pass: Callable[[str], bool]) -> Pick:
    """Read Orion's pick from the compose reply. The first line matching ``Pick: <n>`` wins and is
    removed from the body. PASS anywhere a PASS counts (``is_pass``) is a decline. With exactly one
    topic shown and no Pick line, a message means Orion picked it (still a choice: speak or not)."""
    raw = str(text or "").strip()
    lines = raw.splitlines()
    for i, line in enumerate(lines):
        m = _PICK_LINE.match(line.strip())
        if not m or (not m.group("sep") and m.group("rest").strip(" .)]*_`")):
            continue
        body = "\n".join(lines[:i] + lines[i + 1:]).strip()
        val = m.group("val").lower()
        if not val.isdigit() or int(val) == 0:
            return Pick("declined", None, body, "zero")
        n = int(val)
        if not 1 <= n <= n_options:
            return Pick("invalid", None, body, "line")
        if body and is_pass(body):
            return Pick("declined", None, body, "pass")
        return Pick("chosen", n - 1, body, "line")
    if raw and is_pass(raw):
        return Pick("declined", None, raw, "pass")
    if n_options == 1 and raw and not any(_PICK_ATTEMPT.match(line) for line in lines):
        return Pick("chosen", 0, raw, "implicit")
    return Pick("missing", None, raw, "none")


def _option(t: OutreachTopic, rank: int) -> ChoiceOptionV1:
    return ChoiceOptionV1(key=t.subject[:200], text=clip(t.option_text, CHOICE_OPTION_TEXT_MAX) or t.subject[:200],
                          market_rank=rank)


def _flat_situation(situation: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in situation.items():
        if len(out) >= CHOICE_SITUATION_MAX_KEYS:
            break
        key = str(k)[:64]
        if not key:
            continue
        if v is None or isinstance(v, (bool, int, float)):
            out[key] = v
        else:
            out[key] = clip(str(v), CHOICE_SITUATION_TEXT_MAX)
    return out


def build_choice(
    *,
    decision_key: str,
    topics: Sequence[OutreachTopic],
    pick: Pick,
    occurred_at: datetime,
    situation: Mapping[str, Any],
    act_id: Optional[str],
    correlation_id: Optional[str],
) -> Optional[PersonalityChoiceV1]:
    """The ``PersonalityChoiceV1`` for one compose decision, or None when there was no readable
    choice (missing / invalid pick). ``decision_key`` is unique per compose (the want id), so a
    re-publish is the same row (``choice_id = outreach:<decision_key>``).

    Declined: every shown topic is an alternative. Chosen: the others are. With one topic shown and
    a pick, there is no turned-down topic, so "say nothing" is the alternative Orion turned down."""
    if pick.kind not in ("chosen", "declined") or not topics:
        return None
    options = [_option(t, i) for i, t in enumerate(topics, start=1)]
    if pick.kind == "declined":
        chosen = None
        alternatives = options
    else:
        assert pick.index is not None
        chosen = options[pick.index]
        alternatives = [o for i, o in enumerate(options) if i != pick.index]
        if not alternatives:
            alternatives = [ChoiceOptionV1(key="say_nothing", text="Say nothing to Juniper this time.",
                                           market_rank=None)]
    return PersonalityChoiceV1(
        choice_id=f"outreach:{decision_key}"[:300],
        lane="outreach",
        occurred_at=occurred_at,
        chosen=chosen,
        alternatives=alternatives,
        declined_all=pick.kind == "declined",
        situation=_flat_situation(situation),
        act_id=(act_id or None) and str(act_id)[:300],
        correlation_id=(correlation_id or None) and str(correlation_id)[:300],
    )


def choice_record(topics: Sequence[OutreachTopic], pick: Pick, choice_id: Optional[str]) -> Dict[str, Any]:
    """What the decision row stores under ``result_json.choice``: every shown subject and version
    (the subject ledger reads these back), the pick, and the published choice id."""
    chosen = topics[pick.index] if pick.kind == "chosen" and pick.index is not None else None
    return {
        "choice_id": choice_id,
        "shown": [
            {"subject": t.subject, "version": t.version, "kind": t.kind, "market_rank": i}
            for i, t in enumerate(topics, start=1)
        ],
        "pick": pick.kind,
        "pick_shape": pick.shape,
        "chosen_subject": chosen.subject if chosen else None,
        "chosen_version": chosen.version if chosen else None,
        "chosen_rank": (pick.index + 1) if chosen is not None and pick.index is not None else None,
        "declined_all": pick.kind == "declined",
    }


def fold_said_subjects(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Set[str]]:
    """Fold decision rows into ``{subject: {versions}}`` Orion already decided about. Pure.

    Each row is ``{"reason": ..., "choice": <result_json.choice>, "daydream_id": ...}``.
    * sent / no_why_juniper with a choice: the chosen subject (Juniper got it, or Orion wrote it).
    * orion_declined / no_pick / invalid_pick: every subject shown (Orion looked at them and said
      nothing, or answered in a way we could not read; either way not shown again for 7 days).
    * a tick row from before the choice point (no choice) with a daydream id: that daydream
      (version ""), so the old ledger and the new one agree.
    Malformed rows are skipped: a bad historical row cannot poison the ledger."""
    said: Dict[str, Set[str]] = {}

    def add(subject: Any, version: Any) -> None:
        s = str(subject or "").strip()
        if s:
            said.setdefault(s, set()).add(str(version or ""))

    for row in rows:
        if not isinstance(row, Mapping):
            continue
        reason = str(row.get("reason") or "")
        choice = row.get("choice")
        if isinstance(choice, Mapping):
            if reason in CHOSEN_SUBJECT_REASONS and choice.get("chosen_subject"):
                add(choice.get("chosen_subject"), choice.get("chosen_version"))
            elif reason in DECLINED_SUBJECT_REASONS:
                for item in choice.get("shown") or []:
                    if isinstance(item, Mapping):
                        add(item.get("subject"), item.get("version"))
            continue
        did = str(row.get("daydream_id") or "").strip()
        if did and reason in (CHOSEN_SUBJECT_REASONS | DECLINED_SUBJECT_REASONS | {"orion_passed"}):
            add(did, "")
    return said
