"""Same-turn LLM novelty/salience judgment for the chat-scoped attention/
curiosity pipeline (`orion.substrate.attention_frame.build_attention_frame`,
read from `chat_stance.py::build_chat_stance_inputs` on every real chat
turn, gated by `ORION_CURIOSITY_FRAME_ENABLED`).

Replaces `orion/substrate/attention/detectors/legacy_regex.py`'s
`LegacyRegexSignalDetector` (deleted in the same patch), whose `_PROPER_RE`
regex (`r"\\b([A-Z][A-Za-z0-9_-]{2,}...)\\b"`) matched ANY capitalized word
as a "proper noun" candidate -- including sentence-initial interjections
like "Heck" in "Heck yeah!" -- filtered only by a 9-word STOP_PHRASES
allowlist with no interjection/filler coverage. Confirmed live: those
garbage candidates could be selected as an "ask" action
(`orion/substrate/attention/policy.py::select_actions`) and threaded into
the live chat-turn stance brief, meaning Orion could literally ask the user
a clarifying question about a garbage interjection in a real reply.

Architecture: `AttentionSignalDetector.detect()`
(`orion/substrate/attention/detectors/base.py`) is a synchronous Protocol
method called synchronously inside `build_open_loops`/`build_attention_frame`
from both the chat path (here) and the unrelated substrate-broadcast path
(`orion-thought` reverie). Making the Protocol itself async would ripple
across every detector and every caller for a change that only needs to
affect one call site. Instead: the actual `await`-based LLM RPC call happens
HERE, called from `chat_stance.py::build_chat_stance_inputs` BEFORE
`build_attention_frame()` runs, and the judged candidates are stashed into
`ctx["current_turn_llm_signals"]` (a plain list of `{"phrase", "type"}`
dicts). `CurrentTurnSignalDetector.detect()`
(`orion/substrate/attention/detectors/current_turn.py`) is then a pure,
synchronous reader of that precomputed ctx list -- no network call inside
the detector itself.

Call-pattern precedent: mirrors
`services/orion-cortex-exec/app/pre_turn_appraisal.py::_llm_probe_call`
(RPC via `bus.rpc_request(settings.channel_llm_intake, ...)`,
`ChatRequestPayload`, tight timeout, decode+fail-open) and
`services/orion-memory-consolidation/app/classify.py::_llm_classify` (same
bus RPC glue, `route`-driven, small `max_tokens` for a quick-lane
classification call rather than a generation call). Does NOT hook into
`orion-memory-consolidation`'s post-hoc turn-classification pipeline
(`orion:memory:turn:persisted` -> `classify.py`): that pipeline's trigger
event is produced by `orion-sql-writer` only AFTER the turn is already
persisted -- i.e. after the reply is generated -- so it is structurally
unable to gate the same turn's decision.

Fail-open by contract: never raises. `ctx["current_turn_llm_signals"]` is
always left as a list (possibly empty) -- RPC failure/timeout, an unbound
bus, and malformed LLM output are each logged with a distinguishable
WARNING message (not conflated with "the LLM genuinely found nothing",
which is a clean empty list with no warning at all).
"""

from __future__ import annotations

import json
import logging
from typing import Any
from uuid import uuid4

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ChatRequestPayload, LLMMessage, ServiceRef

from .settings import settings

logger = logging.getLogger("orion.cortex.current_turn_llm_signals")

_MAX_USER_TEXT = 600
_MAX_PHRASE_LEN = 80
_MAX_CANDIDATES = 8
_ALLOWED_TYPES = {"person", "place", "plan", "belief", "concept", "activity", "other"}

def _balanced_json_array_spans(text: str) -> list[str]:
    """Find every *balanced* top-level `[...]` span in `text`, in order.

    A naive greedy regex (`r"\[.*\]"`) spans from the FIRST `[` to the LAST
    `]` anywhere in the response -- any stray bracket before or after the
    real array (a footnote like "see item [1]", trailing commentary with a
    bracketed aside, or two separate arrays) makes the match invalid JSON
    and the whole response gets discarded as malformed, even though a
    clean array was actually present. This walks bracket depth (skipping
    bracket characters inside quoted strings) to find every genuinely
    balanced span instead of just the outermost first-to-last one -- the
    caller tries each in order and uses the first that actually parses into
    the expected shape (a stray "[1]"-style footnote parses as valid JSON
    too, just not as a list of candidate objects, so scanning must not stop
    at the first syntactically-valid-but-wrong-shaped span).
    """
    spans: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "[":
            i += 1
            continue
        start = i
        depth = 0
        in_string = False
        escape = False
        end = None
        for j in range(start, n):
            ch = text[j]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end is None:
            break
        spans.append(text[start : end + 1])
        i = end + 1
    return spans


def _source() -> ServiceRef:
    return ServiceRef(name=settings.service_name, version=settings.service_version, node=settings.node_name)


def build_current_turn_llm_prompt(user_text: str) -> str:
    """Strict, short-output prompt -- a quick-lane classification call, not
    a generation call. Explicitly excludes filler/interjections so the
    model does the filtering the old regex could not."""
    return (
        "From the single user message below, list genuinely new, trackable "
        "things worth following up on later in conversation: a real "
        "person's name, a real place, a concrete plan, or a specific "
        "belief/claim. Do NOT include filler words, interjections, "
        "exclamations (for example \"heck\", \"yeah\", \"yep\", \"wow\", "
        "\"lol\", \"ok\"), greetings, or generic chat filler. If nothing "
        "qualifies, return an empty array.\n\n"
        "Respond with ONLY a JSON array, no prose, no markdown fences, no "
        "explanation. Each item must be an object with exactly two keys: "
        "\"phrase\" (short string, the exact trackable thing) and \"type\" "
        "(one of: person, place, plan, belief, concept, activity, other). "
        "Return at most 4 items.\n\n"
        f"User message: {user_text}\n\n"
        "JSON array:"
    )


def parse_current_turn_llm_signals(raw_text: str) -> list[dict[str, str]] | None:
    """Parse the LLM's response into a list of `{"phrase", "type"}` dicts.

    Returns `[]` for a genuinely empty result (LLM found nothing -- a clean,
    expected outcome, not a failure). Returns `None` when the text could not
    be parsed as the expected shape at all -- callers must log this
    distinctly from a genuine empty result, per CLAUDE.md's fail-open
    logging convention for this call shape.
    """
    text = (raw_text or "").strip()
    if not text:
        return None

    data: list | None = None
    for span in _balanced_json_array_spans(text):
        try:
            candidate = json.loads(span)
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
        if not isinstance(candidate, list):
            continue
        # Empty list ([]) is a genuine "found nothing" result -- accept it.
        # A non-empty list must contain at least one dict to be plausibly
        # our expected {"phrase", "type"} shape (skips stray syntactically-
        # valid-but-wrong spans like a footnote's "[1]").
        if not candidate or any(isinstance(item, dict) for item in candidate):
            data = candidate
            break
    if data is None:
        return None

    out: list[dict[str, str]] = []
    for item in data[:_MAX_CANDIDATES]:
        if not isinstance(item, dict):
            continue
        phrase = str(item.get("phrase") or "").strip(" ,:;()[]{}\"'")
        if len(phrase) < 2:
            continue
        type_hint = str(item.get("type") or "other").strip().lower()
        if type_hint not in _ALLOWED_TYPES:
            type_hint = "other"
        out.append({"phrase": phrase[:_MAX_PHRASE_LEN], "type": type_hint})
    return out


_BUS: OrionBusAsync | None = None


def bind_current_turn_llm_signals_bus(bus: OrionBusAsync) -> None:
    global _BUS
    _BUS = bus


def reset_current_turn_llm_signals_bus_for_tests() -> None:
    global _BUS
    _BUS = None


async def _llm_call(bus: OrionBusAsync, *, prompt: str) -> str:
    rpc_corr = str(uuid4())
    reply_channel = f"orion:exec:result:LLMGatewayService:{rpc_corr}"
    payload = ChatRequestPayload(
        messages=[LLMMessage(role="user", content=prompt)],
        route=settings.current_turn_signal_probe_route,
        options={
            "max_tokens": settings.current_turn_signal_probe_max_tokens,
            "purpose": "current_turn_signal_probe",
            "skip_spark_candidate_publish": True,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    env = BaseEnvelope(
        kind="llm.chat.request",
        source=_source(),
        correlation_id=rpc_corr,
        reply_to=reply_channel,
        payload=payload.model_dump(mode="json"),
    )
    msg = await bus.rpc_request(
        settings.channel_llm_intake,
        env,
        reply_channel=reply_channel,
        timeout_sec=settings.current_turn_signal_probe_timeout_sec,
    )
    decoded = bus.codec.decode(msg.get("data"))
    if not decoded.ok or not isinstance(decoded.envelope.payload, dict):
        raise RuntimeError(f"current_turn_llm_signal_decode_failed ok={decoded.ok}")
    payload_obj = decoded.envelope.payload
    return str(payload_obj.get("content") or payload_obj.get("text") or "")


async def populate_current_turn_llm_signals(ctx: dict[str, Any]) -> None:
    """Best-effort, bounded, fail-open. Always leaves
    `ctx["current_turn_llm_signals"]` set to a list (possibly empty) -- never
    raises, never delays a chat turn beyond its own bounded RPC timeout.
    """
    ctx["current_turn_llm_signals"] = []
    user_text = str(ctx.get("user_message") or ctx.get("raw_user_text") or "").strip()[:_MAX_USER_TEXT]
    if not user_text:
        return

    bus = _BUS
    if bus is None:
        logger.warning("current_turn_llm_signals_bus_unbound")
        return

    prompt = build_current_turn_llm_prompt(user_text)
    try:
        raw_text = await _llm_call(bus, prompt=prompt)
    except Exception as exc:  # noqa: BLE001 -- fail-open by contract
        logger.warning(
            "current_turn_llm_signals_rpc_failed exc_type=%s err=%s",
            type(exc).__name__,
            exc,
        )
        return

    signals = parse_current_turn_llm_signals(raw_text)
    if signals is None:
        logger.warning(
            "current_turn_llm_signals_malformed_output raw_preview=%s",
            raw_text[:120],
        )
        return

    ctx["current_turn_llm_signals"] = signals
