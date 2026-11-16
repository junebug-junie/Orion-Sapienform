"""Intention ledger D0: plain-code gate on the 5b judge's promise fields.

The judge names promises; this module decides which of them are real. Nothing here is a
model call or a word list. Five rules (docs/superpowers/specs/2026-11-11-intention-ledger-design.md):

0. A ``promises_made`` item must name Orion as the one who does the act
   (``PromiseV1.doer == "orion"``). The judge classifies every item it lists; an ask for
   Juniper to act, a hand-off to someone else, and a forecast about a system are dropped
   here, before the cap, so they cannot crowd out a real promise. A missing or unknown
   doer is dropped too (fail closed).
0b. A ``promises_made`` item must have a completion that could fail to happen
   (``PromiseV1.done_when``: what a later reply or run would show when the act is done). The
   judge writes null when nothing Orion later does could break the line: a presence or
   readiness line ("I'll be here when you get back") describes a state Orion is in anyway,
   so it commits to nothing (Searle's preparatory condition for a promise: the act is not
   one the speaker would do in the normal course of events). A missing, null, or
   placeholder-length done_when is dropped (fail closed).
1. A ``promises_made`` quote must be literally in the reply, after whitespace and
   quote-mark normalization. A quote the judge paraphrased or invented is dropped.
2. An ``offers_accepted`` entry must be one of the offer quotes the harness showed the
   judge (Orion's previous-reply offers). Anything else is dropped.
3. A ``promises_kept`` entry must be one of the open-promise ids the harness showed the
   judge. Anything else is dropped.

Offers (``PromiseV1.offer``) are recorded but are not counted as promises until a later
judge call accepts them (``counted_promise_quotes``).

Flag: ORION_PROMISE_LEDGER_ENABLED (default on). Off: the judge is not asked for the
fields and the gate clears them.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from orion.schemas.harness_finalize import (
    PROMISE_DONE_WHEN_MIN_CHARS,
    PROMISE_DOER_ORION,
    PROMISE_DOERS,
    PROMISE_ITEMS_MAX,
    PROMISE_QUOTE_MAX_CHARS,
    FinalizeReflectionV1,
    OpenPromiseRefV1,
    PromiseV1,
)

logger = logging.getLogger("orion.harness.promise_gate")

PROMISE_FIELDS: tuple[str, ...] = ("promises_made", "offers_accepted", "promises_kept")
# A quote shorter than this is a fragment ("I'll"), not a statement of what Orion will do.
PROMISE_QUOTE_MIN_CHARS = 12

_QUOTE_MARKS = str.maketrans({
    "‘": "'", "’": "'", "‛": "'", "′": "'",
    "“": '"', "”": '"', "‟": '"', "″": '"',
})
_WS = re.compile(r"\s+")
_DASH = re.compile(r"\s*(?:\u2014|\u2013|--)\s*")


def promise_ledger_enabled() -> bool:
    raw = os.environ.get("ORION_PROMISE_LEDGER_ENABLED", "true")
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def normalize_for_quote_match(text: str) -> str:
    """Whitespace runs to one space; curly quote marks, dashes and the ellipsis character to
    their plain forms; backticks removed (models rewrite these when they copy). Case and
    words are kept."""
    t = (text or "").translate(_QUOTE_MARKS).replace("\u2026", "...")
    t = t.replace("`", "")  # code marks: 11-03 a real offer was dropped for losing them
    t = _DASH.sub("-", t)
    return _WS.sub(" ", t).strip()


def quote_in_text(quote: str, text: str) -> bool:
    q = normalize_for_quote_match(quote)
    return len(q) >= PROMISE_QUOTE_MIN_CHARS and q in normalize_for_quote_match(text)


@dataclass
class PromiseGateResult:
    reflection: FinalizeReflectionV1
    kept_made: int = 0
    dropped_not_in_text: list[str] = field(default_factory=list)
    # Items whose doer is not Orion, as (doer, quote); doer None = the judge gave none.
    dropped_not_orion: list[tuple[str | None, str]] = field(default_factory=list)
    # Orion's items with no completion that could fail (done_when null or a placeholder).
    dropped_no_completion: list[str] = field(default_factory=list)
    dropped_too_long: int = 0
    dropped_too_short: int = 0
    dropped_unknown_offer: list[str] = field(default_factory=list)
    dropped_unknown_kept: list[str] = field(default_factory=list)
    dropped_over_cap: int = 0

    @property
    def dropped_total(self) -> int:
        return (len(self.dropped_not_in_text) + len(self.dropped_not_orion) + len(self.dropped_no_completion)
                + self.dropped_too_long + self.dropped_too_short
                + len(self.dropped_unknown_offer)
                + len(self.dropped_unknown_kept) + self.dropped_over_cap)


def _opt_str(value: Any) -> str | None:
    return value.strip() or None if isinstance(value, str) else None


def has_completion(promise: PromiseV1) -> bool:
    """Rule 0b: the judge named how a later reply or run would show the act done."""
    return len((promise.done_when or "").strip()) >= PROMISE_DONE_WHEN_MIN_CHARS


def split_promise_fields(raw: dict[str, Any]) -> dict[str, Any]:
    """Pop the promise fields off a judge payload so a malformed list cannot fail
    FinalizeReflectionV1 validation and sink the alignment verdict."""
    return {name: raw.pop(name) for name in PROMISE_FIELDS if name in raw}


def coerce_promise_fields(raw: dict[str, Any]) -> dict[str, Any]:
    """Lenient parse of popped promise fields. Wrong shapes become empty, never an error."""
    def _list(name: str) -> list[Any]:
        value = raw.get(name)
        return value if isinstance(value, list) else []

    made: list[PromiseV1] = []
    for item in _list("promises_made"):
        if isinstance(item, dict) and isinstance(item.get("quote"), str):
            doer = item.get("doer")
            made.append(PromiseV1(quote=item["quote"], offer=item.get("offer") is True,
                                  doer=doer if doer in PROMISE_DOERS else None,
                                  act=_opt_str(item.get("act")), done_when=_opt_str(item.get("done_when"))))
    accepted = [x for x in _list("offers_accepted") if isinstance(x, str)]
    kept = [x for x in _list("promises_kept") if isinstance(x, str)]
    return {"promises_made": made, "offers_accepted": accepted, "promises_kept": kept}


def gate_promises(
    reflection: FinalizeReflectionV1,
    *,
    reply_text: str,
    pending_offers: Iterable[str] = (),
    open_promises: Iterable[OpenPromiseRefV1 | str] = (),
    enabled: bool | None = None,
    correlation_id: str = "",
) -> PromiseGateResult:
    """Apply the five rules. Returns a copy of the reflection with only surviving entries."""
    if enabled is None:
        enabled = promise_ledger_enabled()
    if not enabled:
        cleared = reflection.model_copy(update={"promises_made": [], "offers_accepted": [], "promises_kept": []})
        return PromiseGateResult(reflection=cleared)

    result = PromiseGateResult(reflection=reflection)
    made: list[PromiseV1] = []
    seen: set[str] = set()
    for promise in reflection.promises_made:
        if promise.doer != PROMISE_DOER_ORION:
            result.dropped_not_orion.append((promise.doer, promise.quote))
            continue
        if not has_completion(promise):
            result.dropped_no_completion.append(promise.quote)
            continue
        norm = normalize_for_quote_match(promise.quote)
        if len(promise.quote) > PROMISE_QUOTE_MAX_CHARS:
            result.dropped_too_long += 1
            continue
        if len(norm) < PROMISE_QUOTE_MIN_CHARS:
            result.dropped_too_short += 1
            continue
        if not quote_in_text(promise.quote, reply_text):
            result.dropped_not_in_text.append(promise.quote)
            continue
        if norm in seen:
            continue
        if len(made) >= PROMISE_ITEMS_MAX:
            result.dropped_over_cap += 1
            continue
        seen.add(norm)
        made.append(promise)
    result.kept_made = len(made)

    offers = {normalize_for_quote_match(q): q for q in pending_offers}
    accepted: list[str] = []
    for entry in reflection.offers_accepted:
        original = offers.get(normalize_for_quote_match(entry))
        if original is None:
            result.dropped_unknown_offer.append(entry)
        elif original not in accepted:
            accepted.append(original)

    open_ids = {p if isinstance(p, str) else p.id for p in open_promises}
    kept: list[str] = []
    for entry in reflection.promises_kept:
        if entry not in open_ids:
            result.dropped_unknown_kept.append(entry)
        elif entry not in kept:
            kept.append(entry)

    result.reflection = reflection.model_copy(
        update={"promises_made": made, "offers_accepted": accepted, "promises_kept": kept}
    )
    if result.dropped_total or made or accepted or kept:
        logger.info(
            "promise_gate corr=%s made=%d offers=%d accepted=%d kept=%d dropped_not_in_text=%d dropped_not_orion=%d "
            "dropped_no_completion=%d "
            "dropped_too_long=%d dropped_too_short=%d dropped_unknown_offer=%d dropped_unknown_kept=%d dropped_over_cap=%d",
            correlation_id, len(made), sum(1 for p in made if p.offer), len(accepted), len(kept),
            len(result.dropped_not_in_text), len(result.dropped_not_orion), len(result.dropped_no_completion),
            result.dropped_too_long, result.dropped_too_short, len(result.dropped_unknown_offer),
            len(result.dropped_unknown_kept), result.dropped_over_cap,
        )
    return result


def counted_promise_quotes(reflection: FinalizeReflectionV1, *, accepted_later: Iterable[str] = ()) -> list[str]:
    """The promises this reply counts for the ledger: firm promises, plus offers a later
    judge call accepted (Juniper's 2026-11-12 answer 1). Offers alone never count."""
    accepted = {normalize_for_quote_match(q) for q in accepted_later}
    return [p.quote for p in reflection.promises_made
            if not p.offer or normalize_for_quote_match(p.quote) in accepted]
