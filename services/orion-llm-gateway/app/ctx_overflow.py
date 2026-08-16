"""Retry a context-overflowed request on a lane that can actually hold it.

WHY THIS EXISTS
---------------
Orion's journaling (`journal.compose`, from the world-pulse pipeline) has been running on
circe's `chat` lane -- 131,072 tokens of context -- to carry prompts whose median is **1,749
tokens**. Measured over 24 h, 90 samples:

    min 915    p50 1,749    p95 1,941    MAX 4,966

97.8% of it fits atlas's 4,096-token lanes with room for the response. It was on a 32x-oversized
lane by inheritance, not by need -- and that lane is circe's SINGLE slot, measured full 8.10% of
the time, versus atlas's 4-slot lanes at 4.01% and 0.00%.

The 2.2% tail is real, though: two of ninety prompts exceeded 4,096. Moving journaling to atlas
without handling those would silently fail a fiftieth of Orion's reflection, which is exactly the
class of quiet degradation this arc keeps finding. So: move it, and retry the overflow somewhere
that fits.

WHY RETRY RATHER THAN PRE-FLIGHT
--------------------------------
The obvious alternative is to count tokens before sending and pick the lane accordingly. That
needs a token estimate, and a chars-per-token estimate is precisely the assumption that has been
wrong here before. A retry uses GROUND TRUTH -- the upstream itself said the prompt did not fit.
It costs one wasted round trip on 2.2% of journal traffic, which is cheaper than being
confidently wrong about the other 97.8%.

WHY THE LADDER IS DISCOVERED, NOT HARDCODED
-------------------------------------------
`metacog` is the natural fallback: atlas, 4 slots, measured **0.00% all-busy over 27.74 h**, and
a better quantisation (Q5_K_M vs `quick`'s Q4_K_M). But it is currently **n_ctx=4,096 -- the same
as `quick`** -- so escalating there for an overflow would fail identically and burn an attempt.

Hardcoding `quick -> chat` would work today and be wrong the moment metacog's context is raised.
So the ladder is built from each lane's ACTUAL `n_ctx`, read from its `/props` at startup: escalate
to the smallest lane that is strictly larger than the one that just failed. Raise metacog's
`n_ctx` past ~5k and it becomes the fallback automatically, with no code change and circe drops
out of the journaling path entirely.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger("orion-llm-gateway.ctx")

# Juniper's cap. Counts TOTAL attempts, not retries after the first -- so at most 3 upstreams are
# ever tried for one request, however deep the ladder gets.
MAX_ATTEMPTS = 3

# llama.cpp's phrasing varies by build; match on the substrings that survive across them rather
# than an exact message. Deliberately narrow: a generic 400 must NOT be treated as an overflow,
# or a malformed request would be retried up every lane in the fleet.
_OVERFLOW_MARKERS = (
    "exceed",              # "the request exceeds the available context size"
    "context size",
    "context length",
    "n_ctx",
    "too many tokens",
    "prompt is too long",
)


def is_context_overflow(status_code: int, body: Any) -> bool:
    """True only for a prompt that did not fit. Narrow on purpose.

    A context overflow is DETERMINISTIC -- the same prompt on the same lane fails identically
    forever -- so it is the one error worth escalating rather than repeating. Every other 4xx is
    left alone: retrying a malformed request on a bigger model just wastes a bigger model.
    """
    if status_code not in (400, 413, 422, 500):
        return False
    text = ""
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            text = str(err.get("message") or err.get("type") or "")
        elif err is not None:
            text = str(err)
        if not text:
            text = str(body.get("message") or "")
    else:
        text = str(body or "")
    low = text.lower()
    return any(marker in low for marker in _OVERFLOW_MARKERS)


def probe_context_size(base_url: str, *, timeout_sec: float = 5.0) -> Optional[int]:
    """This lane's real `n_ctx`, from llama.cpp's own `/props`. None if unreachable.

    None, never a default. A guessed context window is how a lane ends up carrying prompts it
    cannot hold -- an unreachable lane is simply left out of the ladder.
    """
    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/props", timeout=timeout_sec)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001 -- unreachable means "cannot rank", not fatal
        logger.warning("[LLM-GW ctx] could not read /props at %s: %s", base_url, exc)
        return None
    if not isinstance(data, dict):
        return None
    gen = data.get("default_generation_settings")
    for source in (gen if isinstance(gen, dict) else {}, data):
        value = source.get("n_ctx")
        if isinstance(value, int) and value > 0:
            return value
    return None


def build_escalation_ladder(route_urls: Dict[str, str]) -> List[Tuple[str, int]]:
    """Routes ordered by real context size, smallest first. Unprobeable routes are omitted.

    Measured 2026-08-16:
        quick    4,096    metacog  4,096    agent  32,768    chat  131,072
    """
    sized: List[Tuple[str, int]] = []
    for route, url in sorted(route_urls.items()):
        n_ctx = probe_context_size(url)
        if n_ctx is None:
            continue
        sized.append((route, n_ctx))
    sized.sort(key=lambda item: (item[1], item[0]))
    return sized


def next_larger_route(
    ladder: List[Tuple[str, int]], current_route: str
) -> Optional[str]:
    """The smallest lane strictly larger than `current_route`. None if nothing is bigger.

    STRICTLY larger is the load-bearing word. metacog and quick are both 4,096 today, so
    escalating quick -> metacog would fail identically and spend an attempt for nothing. When
    metacog's n_ctx is raised past quick's, this starts returning it with no code change.
    """
    current = next((n for route, n in ladder if route == current_route), None)
    if current is None:
        # The failing route is not in the ladder (unprobeable). Escalate to the largest lane
        # available rather than giving up -- something is better than a silent failure.
        return ladder[-1][0] if ladder else None
    for route, n_ctx in ladder:
        if n_ctx > current:
            return route
    return None
