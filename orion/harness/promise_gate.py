"""Intention ledger D0: plain-code gate on the 5b judge's promise fields.

The judge names promises; this module decides which of them are real. Nothing here is a
model call or a word list. Three rules (docs/superpowers/specs/2026-11-11-intention-ledger-design.md):

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


def promise_ledger_enabled() -> bool:
    raw = os.environ.get("ORION_PROMISE_LEDGER_ENABLED", "true")
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def normalize_for_quote_match(text: str) -> str:
    """Whitespace runs to one space, curly quote marks to straight ones. Case is kept."""
    return _WS.sub(" ", (text or "").translate(_QUOTE_MARKS)).strip()


def quote_in_text(quote: str, text: str) -> bool:
    q = normalize_for_quote_match(quote)
    return len(q) >= PROMISE_QUOTE_MIN_CHARS and q in normalize_for_quote_match(text)


@dataclass
class PromiseGateResult:
    reflection: FinalizeReflectionV1
    kept_made: int = 0
    dropped_not_in_text: list[str] = field(default_factory=list)
    dropped_too_long: int = 0
    dropped_unknown_offer: list[str] = field(default_factory=list)
    dropped_unknown_kept: list[str] = field(default_factory=list)
    dropped_over_cap: int = 0

    @property
    def dropped_total(self) -> int:
        return (len(self.dropped_not_in_text) + self.dropped_too_long + len(self.dropped_unknown_offer)
                + len(self.dropped_unknown_kept) + self.dropped_over_cap)


def split_promise_fields(raw: dict[str, Any]) -> dict[str, Any]:
    """Pop the promise fields off a judge payload so a malformed list cannot fail
    FinalizeReflectionV1 validation and sink the alignment verdict."""
    return {name: raw.pop(name) for name in PROMISE_FIELDS if name in raw}


def coerce_promise_fields(raw: dict[str, Any]) -> dict[str, Any]:
    """Lenient parse of popped promise fields. Wrong shapes become empty, never an error."""
    made: list[PromiseV1] = []
    for item in raw.get("promises_made") or []:
        if isinstance(item, dict) and isinstance(item.get("quote"), str):
            made.append(PromiseV1(quote=item["quote"], offer=bool(item.get("offer", False))))
    accepted = [x for x in (raw.get("offers_accepted") or []) if isinstance(x, str)]
    kept = [x for x in (raw.get("promises_kept") or []) if isinstance(x, str)]
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
    """Apply the three rules. Returns a copy of the reflection with only surviving entries."""
    if enabled is None:
        enabled = promise_ledger_enabled()
    if not enabled:
        cleared = reflection.model_copy(update={"promises_made": [], "offers_accepted": [], "promises_kept": []})
        return PromiseGateResult(reflection=cleared)

    result = PromiseGateResult(reflection=reflection)
    made: list[PromiseV1] = []
    seen: set[str] = set()
    for promise in reflection.promises_made:
        norm = normalize_for_quote_match(promise.quote)
        if len(promise.quote) > PROMISE_QUOTE_MAX_CHARS:
            result.dropped_too_long += 1
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
            "promise_gate corr=%s made=%d offers=%d accepted=%d kept=%d dropped_not_in_text=%d "
            "dropped_too_long=%d dropped_unknown_offer=%d dropped_unknown_kept=%d dropped_over_cap=%d",
            correlation_id, len(made), sum(1 for p in made if p.offer), len(accepted), len(kept),
            len(result.dropped_not_in_text), result.dropped_too_long, len(result.dropped_unknown_offer),
            len(result.dropped_unknown_kept), result.dropped_over_cap,
        )
    return result


def counted_promise_quotes(reflection: FinalizeReflectionV1, *, accepted_later: Iterable[str] = ()) -> list[str]:
    """The promises this reply counts for the ledger: firm promises, plus offers a later
    judge call accepted (Juniper's 2026-11-12 answer 1). Offers alone never count."""
    accepted = {normalize_for_quote_match(q) for q in accepted_later}
    return [p.quote for p in reflection.promises_made
            if not p.offer or normalize_for_quote_match(p.quote) in accepted]
