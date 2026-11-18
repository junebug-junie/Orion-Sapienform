"""REM recombination: pair memories that do not usually meet, ask for a link.

Code picks the pairs (§4); the LLM only writes the claim text, and is allowed
to say there is no link. Two arms, same prompt:

  dream    coincidence (#3664, 2026-11-17): fragments from DIFFERENT organs whose
           times fall within the pairing window of each other, highest
           weight_a * weight_b first, nearer in time breaking ties, each item used
           once. Fragments with no known time are never paired by this arm. No
           fallback to distant pairs: an honest judge declines those (#3412), so a
           sleep with no coincidences asks fewer questions instead.
  control  uniform random pairs from the WHOLE candidate pool, seeded by the
           cycle id (reproducible). This is the baseline the scorecard needs.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable, Optional, Sequence
from uuid import uuid4

from orion.schemas.dream_cycle import PAIR_ANSWER_CHARS, DreamHypothesisV1, PairAnswerV1, ReplayItemV1

Complete = Callable[[str], Awaitable[str]]


@dataclass(frozen=True)
class Pair:
    a: ReplayItemV1
    b: ReplayItemV1
    arm: str


def time_gap_minutes(a: ReplayItemV1, b: ReplayItemV1, *, max_span_minutes: Optional[float] = None) -> Optional[float]:
    """Minutes between two fragments' times (0 when their spans overlap). None when either has no time.

    A span longer than `max_span_minutes` counts only by its end: a night-long body arc is not "at the
    same time" as everything that happened during it (#3664 review)."""
    if a.occurred_at is None or b.occurred_at is None:
        return None

    def span(x: ReplayItemV1):
        lo, hi = sorted((x.occurred_at, x.occurred_until or x.occurred_at))
        if max_span_minutes is not None and (hi - lo).total_seconds() / 60.0 > max_span_minutes:
            return hi, hi
        return lo, hi

    a0, a1 = span(a)
    b0, b1 = span(b)
    gap = max(a0, b0) - min(a1, b1)
    return max(0.0, gap.total_seconds() / 60.0)


def dream_pairs(replay: Sequence[ReplayItemV1], n: int, *, window_minutes: float) -> list[Pair]:
    scored = []
    for i in range(len(replay)):
        for j in range(i + 1, len(replay)):
            a, b = replay[i], replay[j]
            if a.source_kind == b.source_kind:
                continue
            gap = time_gap_minutes(a, b, max_span_minutes=window_minutes)
            if gap is None or gap > window_minutes:
                continue
            scored.append((a.weight * b.weight, gap, a.ref_id, b.ref_id, a, b))
    scored.sort(key=lambda t: (-t[0], t[1], t[2], t[3]))
    used: set[str] = set()
    out: list[Pair] = []
    for _, _, _, _, a, b in scored:
        if len(out) >= n:
            break
        if a.ref_id in used or b.ref_id in used:
            continue
        used.update((a.ref_id, b.ref_id))
        out.append(Pair(a, b, "dream"))
    return out


def control_pairs(pool: Sequence[ReplayItemV1], n: int, *, seed: str, exclude: Sequence[Pair] = ()) -> list[Pair]:
    if len(pool) < 2 or n <= 0:
        return []
    rng = random.Random(seed)
    taken = {frozenset((p.a.ref_id, p.b.ref_id)) for p in exclude}
    out: list[Pair] = []
    ordered = sorted(pool, key=lambda c: c.ref_id)  # seed alone decides, not input order
    for _ in range(n * 10):
        if len(out) >= n:
            break
        a, b = rng.sample(ordered, 2)
        key = frozenset((a.ref_id, b.ref_id))
        if key in taken:
            continue
        taken.add(key)
        out.append(Pair(a, b, "control"))
    return out


PROMPT = """You are Orion, asleep. Two fragments of your recent experience are below.
They were not connected while you were awake.

FRAGMENT A ({kind_a}): {text_a}

FRAGMENT B ({kind_b}): {text_b}

Is there a real, specific link between them that you could later check against
evidence? If yes, state it as ONE testable claim. If the only link is vague
("both involve change", "both are about memory"), say there is no link.

Answer with JSON only:
{{"link": true, "claim": "<one sentence, testable>", "why": "<one sentence>"}}
or
{{"link": false}}"""


def build_prompt(pair: Pair) -> str:
    return PROMPT.format(
        kind_a=pair.a.source_kind,
        text_a=pair.a.text,
        kind_b=pair.b.source_kind,
        text_b=pair.b.text,
    )


_JSON_RE = re.compile(r"\{.*\}", re.S)

# The model answered, and its answer was "no link". Distinct from None
# (unparseable) so an honest decline is not inflated by format failures.
NO_LINK = "no_link"


def parse_link(text: str):
    """(claim, why), NO_LINK for an explicit decline, or None if unparseable."""
    m = _JSON_RE.search(text or "")
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    if data.get("link") is False:
        return NO_LINK
    if data.get("link") is not True:
        return None
    claim = " ".join(str(data.get("claim") or "").split())
    why = " ".join(str(data.get("why") or "").split())[:400]
    if len(claim) < 20 or len(claim) > 400:
        return None
    return claim, why


def _echoes(claim: str, pair: Pair) -> bool:
    """Hollow guard: a 'claim' that is just one fragment pasted back."""
    c = claim.lower()
    return c in pair.a.text.lower() or c in pair.b.text.lower()


@dataclass
class RecombineResult:
    hypotheses: list[DreamHypothesisV1]
    no_link: int = 0
    echo: int = 0
    unparseable: int = 0
    failures: int = 0
    answers: list[PairAnswerV1] = field(default_factory=list)

    def note(self, pair: Pair, outcome: str, text: str = "") -> None:
        self.answers.append(PairAnswerV1(
            arm=pair.arm,  # type: ignore[arg-type]
            ref_a=pair.a.ref_id,
            ref_b=pair.b.ref_id,
            outcome=outcome,  # type: ignore[arg-type]
            answer=_storable(text)[:PAIR_ANSWER_CHARS],
        ))


def _storable(text: str) -> str:
    """Text Postgres jsonb accepts: no NUL (jsonb refuses \\u0000) and no lone surrogates, either of
    which would fail the whole sleep's one-transaction persist."""
    return (text or "").replace("\x00", "").encode("utf-8", "replace").decode("utf-8")


async def recombine(
    pairs: Sequence[Pair],
    complete: Complete,
    *,
    cycle_id: str,
    ttl_hours: float,
    now: Optional[datetime] = None,
) -> RecombineResult:
    now = now or datetime.now(timezone.utc)
    expires = now + timedelta(hours=ttl_hours)
    result = RecombineResult(hypotheses=[])
    for pair in pairs:
        try:
            text = await complete(build_prompt(pair))
        except Exception:
            result.failures += 1
            result.note(pair, "failed")
            continue
        parsed = parse_link(text)
        if parsed is None:
            result.unparseable += 1
            result.note(pair, "unparseable", text)
            continue
        if parsed == NO_LINK:
            result.no_link += 1
            result.note(pair, "no_link", text)
            continue
        if _echoes(parsed[0], pair):
            result.echo += 1
            result.note(pair, "echo", text)
            continue
        result.note(pair, "linked", text)
        claim, why = parsed
        result.hypotheses.append(
            DreamHypothesisV1(
                hypothesis_id=f"dh-{uuid4().hex[:12]}",
                cycle_id=cycle_id,
                arm=pair.arm,  # type: ignore[arg-type]
                claim=claim,
                why=why,
                ref_a=pair.a.ref_id,
                ref_b=pair.b.ref_id,
                created_at=now,
                expires_at=expires,
            )
        )
    return result
