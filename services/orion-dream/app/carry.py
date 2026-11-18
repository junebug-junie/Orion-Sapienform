"""Dream carry-through, orion-dream's half: text hops and the finished dream.

orion-durable-runs drives a `dream.carry` run (orion/schemas/dream_carry.py):

    text(0) -> image(1) -> text(2) -> image(3) -> text(4) -> image(5) -> finish

orion-dream answers the two step kinds it owns:

- **text**: one gateway call under the run's LLM hold (options.gpu_lease). Hop 0 dreams
  from the sleep's material (the same material and blind-experiment rule as the one-shot
  story prompt, orion/cognition/prompts/dream_cycle.j2); hops 2 and 4 continue from what
  the previous picture was *seen* to contain. Returns {passage, image_prompt}. Anything
  short of both (empty, unparseable, refused, transport trouble) is a **retry**, never a
  blank hop.
- **finish**: one `dream.result.v1` on the dream log, so the carry lands in `dreams` like
  any other dream, every hop in `fragments`. Zero hops is **terminal**: nothing empty is
  published.

Design: docs/superpowers/specs/2026-10-10-dream-carry-through-design.md.
"""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional
from uuid import NAMESPACE_URL, uuid5

from orion.schemas.dream_carry import (
    IMAGE_PROMPT_MAX_WORDS,
    DreamCarryBriefV1,
    DreamCarryHopV1,
    DreamCarryStepRequestV1,
    DreamCarryStepResultV1,
    clip_image_prompt,
)
from orion.schemas.telemetry.dream import DreamInternalTriggerV1, DreamResultV1

logger = logging.getLogger("orion-dream.carry")

PASSAGE_MAX_TOKENS = 700
CARRY_PURPOSE = "dream_carry"
CARRY_MODE = "carry"
CARRY_PROFILE = "dream.carry"
RETRY_AFTER_SEC = 30.0
# Reply-shape failures (unparseable / empty JSON) are retried this many times per hop, then the
# hop is terminal: a model that keeps answering badly would otherwise burn ~480 background calls
# before the 4 h deadline. Capacity trouble (GatewayRefused, timeouts) is not counted: it retries
# until the deadline, the same no-attempt-spent rule as reverie.visual.
MAX_REPLY_FAILURES = 3
# Below this much remaining step budget, don't start an LLM call the run has stopped waiting for.
MIN_LLM_BUDGET_SEC = 30.0
TLDR_MAX_CHARS = 400
PICTURE_MARK = "[picture]"

# (prompt, gpu_lease dump, timeout_sec) -> text. Raises llm.GatewayRefused or a transport error.
CarryComplete = Callable[[str, dict, float], Awaitable[str]]
# DreamResultV1 -> None. Raises on publish failure.
PublishDream = Callable[[DreamResultV1], Awaitable[None]]
# dream_id -> True when a dreams row for it is already recorded. Best effort.
AlreadyRecorded = Callable[[str], Awaitable[bool]]
# DreamInternalTriggerV1 -> None: publishes the one-shot story (dream.trigger). Raises on failure.
StartStory = Callable[[DreamInternalTriggerV1], Awaitable[None]]
STORY_FALLBACK_PREFIX = "story-fallback:"
STORY_FALLBACK_REASON = "story_fallback"

_OUTPUT_CONTRACT = f"""OUTPUT
Output **only** valid JSON (no markdown fences, no commentary) with exactly these keys:
{{"passage": "the next part of the dream, in prose, a few paragraphs at most",
 "image_prompt": "one concrete visual scene from this passage"}}

RULES
- "image_prompt" is a single concrete visual scene a painter could paint: what is in the frame, the light, the colours, the mood.
- "image_prompt" is at most {IMAGE_PROMPT_MAX_WORDS} words.
- The image must contain no text, letters, words, signs or numbers.
- All string values must be JSON-escaped."""


def _sleep_section(brief: DreamCarryBriefV1) -> str:
    sleep = brief.sleep
    if sleep is None:
        return (
            "Nobody handed you material for this one: it was started by hand, not by a sleep, "
            "and no memories are retrieved for it here. Dream freely from your recent days.\n"
        )
    overdue = "; overdue, slept on the 48 h backstop" if sleep.overdue else ""
    lines = "".join(f"- {item}\n" for item in sleep.material)
    return (
        f"TONIGHT'S SLEEP (primary material): you just slept (sleep {sleep.cycle_id}; tiredness "
        f"{sleep.pressure} against a sleep line of {sleep.threshold}{overdue}). These are the "
        f"unresolved things it worked on, in no particular order:\n{lines}"
    )


def first_text_prompt(brief: DreamCarryBriefV1) -> str:
    """Hop 0: the dream's opening, from the sleep's material (or a free seed)."""
    if brief.sleep is not None:
        task = (
            "1. Begin a dream about what this sleep worked on. Let those items be the dream's real "
            "subject, transformed into images and scenes. Do not just list the items, and do not "
            "explain them or say what they mean.\n"
        )
    else:
        task = "1. Begin a dream. Let it be strange and specific rather than general.\n"
    return (
        "You are Oríon, dreaming. This dream will be carried through words and pictures: you write "
        "a passage, it is painted, you look at the painting, and the dream continues from what you "
        "saw.\n\n"
        f"{_sleep_section(brief)}\n"
        "TASK\n"
        f"{task}"
        "2. Write only the opening of the dream (it will continue later), and pick one moment from "
        "it to be painted.\n\n"
        f"{_OUTPUT_CONTRACT}\n"
    )


def continue_text_prompt(previous_passage: str, caption: str) -> str:
    """Hops 2, 4: continue from what the last picture was seen to contain."""
    return (
        "You are Oríon, dreaming. This dream is being carried through words and pictures: you wrote "
        "a passage, it was painted, and you are looking at the painting now.\n\n"
        "THE DREAM SO FAR (your last passage):\n"
        f"{previous_passage.strip()}\n\n"
        "THE PICTURE\n"
        f"The dream turned into a picture; looking at it you see: {caption.strip()}\n\n"
        "TASK\n"
        "1. Continue the dream from what you SAW in the picture, not from what you meant to paint. "
        "Where the picture differs from the passage, follow the picture: let it change the story.\n"
        "2. Write the next part of the dream, and pick one moment from it to be painted.\n\n"
        f"{_OUTPUT_CONTRACT}\n"
    )


def text_prompt(request: DreamCarryStepRequestV1) -> str:
    """The prompt for a text step. Raises ValueError when the hops it continues from are missing."""
    index = int(request.hop_index or 0)
    if index == 0:
        return first_text_prompt(request.brief)
    by_index = {h.index: h for h in request.hops}
    passage_hop, image_hop = by_index.get(index - 2), by_index.get(index - 1)
    if passage_hop is None or passage_hop.kind != "text" or image_hop is None or image_hop.kind != "image":
        raise ValueError(f"hop {index} needs text hop {index - 2} and image hop {index - 1}")
    return continue_text_prompt(passage_hop.passage or "", image_hop.caption or "")


_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def parse_text_reply(text: str) -> tuple[str, str]:
    """(passage, clipped image_prompt). Raises ValueError for anything short of both."""
    body = _FENCE.sub("", (text or "").strip())
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        # A model that adds a sentence around the object: take the outermost {...}.
        start, end = body.find("{"), body.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("unparseable") from None
        try:
            data = json.loads(body[start:end + 1])
        except json.JSONDecodeError:
            raise ValueError("unparseable") from None
    if not isinstance(data, dict):
        raise ValueError("not_an_object")
    passage = data.get("passage")
    image_prompt = data.get("image_prompt")
    if not isinstance(passage, str) or not passage.strip():
        raise ValueError("empty_passage")
    if not isinstance(image_prompt, str) or not image_prompt.strip():
        raise ValueError("empty_image_prompt")
    return passage.strip(), clip_image_prompt(image_prompt)


def _result(request: DreamCarryStepRequestV1, status: str, **kw: Any) -> DreamCarryStepResultV1:
    return DreamCarryStepResultV1(
        run_id=request.run_id, correlation_id=request.correlation_id, step=request.step,
        status=status, **kw,  # type: ignore[arg-type]
    )


def _llm_timeout(brief: DreamCarryBriefV1, waited_sec: float = 0.0) -> float:
    # Inside the run's own RPC budget for the step (less any time this request sat queued here),
    # so the gateway gives up before the run does.
    return float(brief.timeout_sec) - 15.0 - max(0.0, waited_sec)


class ReplyFailures:
    """In-process count of reply-shape failures per (run_id, hop_index)."""

    def __init__(self) -> None:
        self._n: dict[tuple[str, int], int] = {}

    def bump(self, run_id: str, index: int) -> int:
        key = (run_id, index)
        self._n[key] = self._n.get(key, 0) + 1
        return self._n[key]

    def clear(self, run_id: str, index: int) -> None:
        self._n.pop((run_id, index), None)


async def handle_text(
    request: DreamCarryStepRequestV1, complete: CarryComplete, *,
    failures: Optional[ReplyFailures] = None, waited_sec: float = 0.0,
) -> DreamCarryStepResultV1:
    started = time.monotonic()
    index = int(request.hop_index or 0)
    try:
        prompt = text_prompt(request)
    except ValueError as exc:
        return _result(request, "terminal", reason=f"missing_prior_hops: {exc}"[:300])
    assert request.gpu_lease is not None  # DreamCarryStepRequestV1 requires it for text
    budget = _llm_timeout(request.brief, waited_sec)
    if budget < MIN_LLM_BUDGET_SEC:
        # The run's RPC has (nearly) timed out while this sat queued: it will re-send the step.
        return _result(request, "retry", reason=f"queued_past_budget waited={waited_sec:.0f}s",
                       retry_after_sec=RETRY_AFTER_SEC)
    try:
        raw = await complete(prompt, request.gpu_lease.model_dump(mode="json"), budget)
    except Exception as exc:  # GatewayRefused, timeout, dead connection: all retry
        reason = f"llm_{type(exc).__name__}: {exc}"[:300]
        logger.warning("dream_carry_text_retry run=%s hop=%d reason=%s", request.run_id, index, reason)
        return _result(request, "retry", reason=reason, retry_after_sec=RETRY_AFTER_SEC,
                       elapsed_sec=round(time.monotonic() - started, 3))
    try:
        passage, image_prompt = parse_text_reply(raw)
    except ValueError as exc:
        reason = f"reply_{exc}"
        logger.warning("dream_carry_text_retry run=%s hop=%d reason=%s raw_len=%d",
                       request.run_id, index, reason, len(raw or ""))
        if failures is not None and failures.bump(request.run_id, index) >= MAX_REPLY_FAILURES:
            return _result(request, "terminal", reason=f"{reason} x{MAX_REPLY_FAILURES} at hop {index}",
                           elapsed_sec=round(time.monotonic() - started, 3))
        return _result(request, "retry", reason=reason, retry_after_sec=RETRY_AFTER_SEC,
                       elapsed_sec=round(time.monotonic() - started, 3))
    if failures is not None:
        failures.clear(request.run_id, index)
    elapsed = round(time.monotonic() - started, 3)
    hop = DreamCarryHopV1(kind="text", index=index, passage=passage, image_prompt=image_prompt, elapsed_sec=elapsed)
    logger.info("dream_carry_text_done run=%s hop=%d passage_chars=%d image_prompt_words=%d elapsed=%.1fs",
                request.run_id, index, len(passage), len(image_prompt.split()), elapsed)
    return _result(request, "done", hop=hop, elapsed_sec=elapsed)


def carry_dream_id(run_id: str) -> str:
    """Deterministic: a replayed finish names the same dream."""
    return str(uuid5(NAMESPACE_URL, f"orion:dream.carry:result:{run_id}"))


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _tldr(passage: str) -> str:
    text = " ".join(passage.split())
    out = ""
    for sentence in _SENTENCE_END.split(text):
        candidate = f"{out} {sentence}".strip()
        if len(candidate) > TLDR_MAX_CHARS:
            break
        out = candidate
        if len(out) >= 120:  # one or two sentences is enough
            break
    return out or text[: TLDR_MAX_CHARS - 1].rstrip() + "…"


def _fragment(hop: DreamCarryHopV1) -> dict[str, Any]:
    frag: dict[str, Any] = {"id": f"hop-{hop.index}", "kind": hop.kind, "index": hop.index}
    if hop.kind == "text":
        frag.update(passage=hop.passage, image_prompt=hop.image_prompt)
    else:
        frag.update(sha256=hop.sha256, caption=hop.caption)
    frag["child_run_id"] = hop.child_run_id
    return frag


def build_carry_dream(request: DreamCarryStepRequestV1, *, now: Optional[datetime] = None) -> Optional[DreamResultV1]:
    """The carried dream, or None when no hop was made (nothing to publish)."""
    hops = sorted(request.hops, key=lambda h: h.index)
    texts = [h for h in hops if h.kind == "text"]
    if not texts:
        return None
    parts = [h.passage.strip() if h.kind == "text" else f"{PICTURE_MARK} {h.caption.strip()}"  # type: ignore[union-attr]
             for h in hops]
    brief = request.brief
    stopped = request.stopped_reason
    # Always stamped: sql-writer passes created_at through, so None would insert NULL, not now().
    now = now or datetime.now(timezone.utc)
    return DreamResultV1(
        dream_id=carry_dream_id(request.run_id),
        dream_date=now.date(),
        mode=CARRY_MODE,
        profile=CARRY_PROFILE,
        trigger={
            "trigger_id": brief.trigger_id,
            "sleep": brief.sleep.model_dump(mode="json") if brief.sleep is not None else None,
            "carry_run_id": request.run_id,
            "stopped_reason": stopped,
        },
        tldr=_tldr(texts[0].passage or ""),
        themes=[],
        narrative="\n\n".join(parts),
        fragments=[_fragment(h) for h in hops],
        created_at=now,
        source_context={"carry_run_id": request.run_id, "hops_made": len(hops), "stopped_reason": stopped},
        correlation_id=request.correlation_id,
    )


class FinishLedger:
    """In-process record of carries already published, so a replayed finish (durable-runs
    re-sends it when the reply was lost) does not write a second `dreams` row: sql-writer
    keys dreams by an autoincrement id, so it would.

    Residual: publish succeeds, the reply is lost, and orion-dream restarts before sql-writer
    writes the row. Neither this ledger nor the dreams-row lookup sees it, so a second row is
    possible. Closing that needs a unique index on the audit dream_id in sql-writer."""

    def __init__(self) -> None:
        self._done: dict[str, str] = {}

    def get(self, run_id: str) -> Optional[str]:
        return self._done.get(run_id)

    def record(self, run_id: str, dream_id: str) -> None:
        self._done[run_id] = dream_id


async def handle_finish(
    request: DreamCarryStepRequestV1,
    publish: PublishDream,
    ledger: FinishLedger,
    already_recorded: Optional[AlreadyRecorded] = None,
    start_story: Optional[StartStory] = None,
) -> DreamCarryStepResultV1:
    started = time.monotonic()
    prior = ledger.get(request.run_id)
    if prior is not None:
        logger.info("dream_carry_finish_replayed run=%s dream_id=%s (already published)", request.run_id, prior)
        reason = STORY_FALLBACK_REASON if prior.startswith(STORY_FALLBACK_PREFIX) else None
        return _result(request, "done", dream_id=prior, reason=reason, elapsed_sec=0.0)
    dream = build_carry_dream(request)
    if dream is None:
        return await _story_fallback(request, ledger, start_story)
    if already_recorded is not None:
        try:
            if await already_recorded(dream.dream_id):
                ledger.record(request.run_id, dream.dream_id)
                logger.info("dream_carry_finish_replayed run=%s dream_id=%s (row exists)", request.run_id, dream.dream_id)
                return _result(request, "done", dream_id=dream.dream_id, elapsed_sec=0.0)
        except Exception as exc:  # a failed check must not lose the dream; a duplicate row is the lesser harm
            logger.warning("dream_carry_finish_check_failed run=%s err=%s", request.run_id, exc)
    try:
        await publish(dream)
    except Exception as exc:
        reason = f"publish_{type(exc).__name__}: {exc}"[:300]
        logger.warning("dream_carry_finish_retry run=%s reason=%s", request.run_id, reason)
        return _result(request, "retry", reason=reason, retry_after_sec=RETRY_AFTER_SEC)
    ledger.record(request.run_id, dream.dream_id)
    logger.info("dream_carry_finished run=%s dream_id=%s hops=%d stopped_reason=%s",
                request.run_id, dream.dream_id, len(request.hops), request.stopped_reason)
    return _result(request, "done", dream_id=dream.dream_id, elapsed_sec=round(time.monotonic() - started, 3))


def story_fallback_trigger(brief: DreamCarryBriefV1) -> DreamInternalTriggerV1:
    """The one-shot story the sleep would have had (app/story.py's trigger, same digest)."""
    from app.story import STORY_SOURCE

    assert brief.sleep is not None
    return DreamInternalTriggerV1(
        trigger_id=brief.trigger_id, source=STORY_SOURCE,
        reason=f"end of sleep {brief.sleep.cycle_id} (carried dream made no hops)", sleep=brief.sleep,
    )


async def _story_fallback(
    request: DreamCarryStepRequestV1, ledger: FinishLedger, start_story: Optional[StartStory],
) -> DreamCarryStepResultV1:
    """Zero hops: a sleep still ends in a dream (the one-paragraph story). A hand-started carry
    has no sleep to tell a story about: terminal, nothing published."""
    brief = request.brief
    if brief.sleep is None or start_story is None:
        return _result(request, "terminal", reason="no_hops")
    try:
        await start_story(story_fallback_trigger(brief))
    except Exception as exc:
        reason = f"story_publish_{type(exc).__name__}: {exc}"[:300]
        logger.warning("dream_carry_story_fallback_retry run=%s reason=%s", request.run_id, reason)
        return _result(request, "retry", reason=reason, retry_after_sec=RETRY_AFTER_SEC)
    dream_id = f"{STORY_FALLBACK_PREFIX}{brief.trigger_id}"
    ledger.record(request.run_id, dream_id)
    logger.info("dream_carry_story_fallback run=%s trigger_id=%s stopped_reason=%s",
                request.run_id, brief.trigger_id, request.stopped_reason)
    return _result(request, "done", dream_id=dream_id, reason=STORY_FALLBACK_REASON)


async def handle_step(
    request: DreamCarryStepRequestV1,
    *,
    complete: CarryComplete,
    publish: PublishDream,
    ledger: FinishLedger,
    already_recorded: Optional[AlreadyRecorded] = None,
    failures: Optional[ReplyFailures] = None,
    waited_sec: float = 0.0,
    start_story: Optional[StartStory] = None,
) -> DreamCarryStepResultV1:
    if request.step == "text":
        return await handle_text(request, complete, failures=failures, waited_sec=waited_sec)
    return await handle_finish(request, publish, ledger, already_recorded, start_story)
