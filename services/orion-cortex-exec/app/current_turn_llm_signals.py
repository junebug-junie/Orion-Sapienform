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
bus RPC glue, `route`-driven, small `max_tokens` for a short
classification call rather than a generation call; route is `chat`, see
settings.current_turn_signal_probe_route for the eval that moved it off `quick`). Does NOT hook into
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

from orion.cognition.fast_chat_verbs import FAST_SINGLE_PASS_CHAT_VERBS
from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ChatRequestPayload, LLMMessage, ServiceRef

from .settings import settings

logger = logging.getLogger("orion.cortex.current_turn_llm_signals")

_MAX_USER_TEXT = 600
_MAX_PHRASE_LEN = 80
_MAX_QUESTION_LEN = 160
_MAX_CANDIDATES = 8
_ALLOWED_TYPES = {"person", "place", "plan", "belief", "concept", "activity", "other"}

def _balanced_json_spans(text: str, open_ch: str, close_ch: str) -> list[str]:
    r"""Find every *balanced* top-level `open_ch...close_ch` span in `text`, in
    order (`[...]` for the candidate array, `{...}` for the read object).

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
        if text[i] != open_ch:
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
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end is None:
            break
        spans.append(text[start : end + 1])
        i = end + 1
    return spans


# rpc-health hop key suffix for this probe's RPC ("<channel>#current_turn_probe").
PROBE_HEALTH_LABEL = "current_turn_probe"


# The verbs cortex-orch already classifies as interactive chat
# (services/orion-cortex-orch/app/execution_lanes.py::resolve_execution_lane,
# reason="verb_chat"): the legacy Hub -> cortex-gateway -> orch chat path.
_CHAT_ENTRY_VERBS = frozenset({"chat_general"}) | FAST_SINGLE_PASS_CHAT_VERBS
# Reason stamped into ctx["current_turn_llm_read"]["skipped"] when the probe is
# not run because the turn carries no human message. Distinct from every
# failure path (ok=False with no "skipped" key) and from a real read (ok=True).
SKIPPED_NOT_HUMAN_TURN = "not_human_turn"


def human_chat_turn_reason(ctx: dict[str, Any]) -> tuple[bool, str]:
    """Whether this turn carries a real human message, and the evidence used.

    Reuses signals the callers already set rather than a new flag:

    - ``stance_inputs["utterance_origin"]`` -- set by the Hub's unified turn
      (orion/hub/turn_orchestrator.py). Only the two human entry points pass
      "juniper" (websocket chat and HTTP /api/chat); curiosity passes "orion";
      endogenous outreach and autonomous reading pass nothing. Collapse-mirror
      replies also pass nothing: Juniper wrote the entry, but it is a form
      submission framed into a prompt, not a chat message, so it is skipped on
      purpose. Rides to cortex-exec inside the stance_react request context
      (services/orion-thought/app/bus_listener.py::build_stance_react_context).
    - ``ctx["verb"]`` (the plan verb, set by router.py) for the legacy chat path,
      which predates utterance_origin: a chat entry verb is a human turn unless
      Orion's own dispatch machinery sent it (``policy_dispatch_only``, set by
      orion-actions' scheduler, cortex-orch workflows, durable runs, the
      journaler and the capability bridge).

    Everything else (journal.compose, log_orion_metacognition, render_scene,
    harness_finalize_reflect, reverie, ...) is treated as not-a-chat-turn.
    Known consequence: legacy-lane turns whose plan verb is rewritten away from
    a chat verb (Hub auto-route depth 1/2, single-verb override) and the
    harness finalize leg of a Juniper turn also skip; their frames carry
    debug.turn_read_skipped so they are not mistaken for probe failures.
    """
    stance_inputs = ctx.get("stance_inputs") if isinstance(ctx.get("stance_inputs"), dict) else {}
    origin = str(stance_inputs.get("utterance_origin") or "").strip().lower()
    if origin == "juniper":
        return True, "utterance_origin_juniper"
    if origin == "orion":
        return False, "utterance_origin_orion"
    verb = str(ctx.get("verb") or "").strip().lower()
    if verb == "stance_react":
        return False, "unified_turn_without_human_origin"
    if verb not in _CHAT_ENTRY_VERBS:
        return False, f"non_chat_verb:{verb or 'none'}"
    opts = ctx.get("options") if isinstance(ctx.get("options"), dict) else {}
    if bool(opts.get("policy_dispatch_only") or ctx.get("policy_dispatch_only")):
        return False, "policy_dispatch"
    return True, "chat_entry_verb"


def mark_current_turn_llm_skipped(ctx: dict[str, Any], reason: str) -> None:
    """Leave the probe's ctx keys in the explicit skipped state.

    ok stays False, so orion.substrate.attention.policy.direct_answer_cause
    fails closed exactly as it does for any missing read (never "found nothing",
    never "wants a direct answer"); the "skipped" key is what tells it apart
    from a timeout or malformed reply.
    """
    ctx["current_turn_llm_signals"] = []
    ctx["current_turn_llm_read"] = {
        "ok": False,
        "wants_direct_answer": None,
        "skipped": SKIPPED_NOT_HUMAN_TURN,
        "skip_reason": reason,
    }


def _source() -> ServiceRef:
    return ServiceRef(name=settings.service_name, version=settings.service_version, node=settings.node_name)


def build_current_turn_llm_prompt(user_text: str) -> str:
    """Short-output read of one user turn: what they shared that a
    friend would naturally follow up on, and whether they are asking for work or
    an answer.

    The previous contract ("a real person's name, a real place, a concrete plan,
    or a specific belief/claim") was entity-shaped: vague life news carried no
    name or specifics, so it was dropped -- confirmed live on corr beab81a3
    ("busy the next few days with work travel" -> []), and measured at 1/10 on
    the quick lane. Vagueness is the reason to ask, not a reason to skip. No
    topical vocabulary here on purpose: the eval
    (evals/run_current_turn_disclosure_live_eval.py) checks unrelated kinds of
    life news and task/status controls so this cannot degrade into a topic list.
    """
    return (
        "The user below is someone you know well. Read their single message and "
        "answer two things.\n\n"
        "1. wants_direct_answer: true if they are asking you to do something or "
        "to answer a question; false if they are sharing, reacting, or chatting.\n"
        "2. items: things they shared about their own life, plans, people, "
        "feelings, or experiences that a friend who cares about them would "
        "naturally want to hear more about. A thing counts even if it is vague -- "
        "vagueness is what makes it worth asking about. For each, write the "
        "casual, specific question a friend would ask them next, addressed to "
        "them. Do NOT include the task or question they asked you, anything "
        "about software or system status, filler, greetings, interjections, "
        "exclamations (for example \"heck\", \"yeah\", \"wow\", \"lol\", "
        "\"ok\"), or acknowledgments. If nothing qualifies, items is an empty "
        "array.\n\n"
        "Respond with ONLY one JSON object, no prose, no markdown fences:\n"
        '{"wants_direct_answer": true or false, "items": [{"phrase": "<short '
        'string naming the thing they shared>", "type": "<one of person, place, '
        'plan, belief, concept, activity, other>", "question": "<the friend\'s '
        'follow-up question>"}]}\n'
        "Every string value is in double quotes.\n"
        "At most 3 items.\n\n"
        f"User message: {user_text}\n\n"
        "JSON object:"
    )


def _read_object(text: str) -> dict[str, Any] | None:
    """First balanced top-level `{...}` span that is the read object (has an
    `items` list). A bare-array response's first object span is one of its
    items, which has no `items` key, so it is skipped rather than mistaken
    for the read."""
    for span in _balanced_json_spans(text, "{", "}"):
        try:
            candidate = json.loads(span)
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
        if isinstance(candidate, dict) and isinstance(candidate.get("items"), list):
            return candidate
    return None


def parse_current_turn_llm_read(raw_text: str) -> dict[str, Any] | None:
    """Parse the probe's response into `{"wants_direct_answer", "signals"}`.

    `wants_direct_answer` is True/False only when the model returned a real
    boolean; anything else (including the legacy bare-array shape) is None --
    unknown, which the attention policy treats as fail-closed. `signals` is the
    same floor-filtered list `parse_current_turn_llm_signals` returns, with an
    optional bounded `natural_question` per item. Returns None when the text is
    not either shape at all.
    """
    text = (raw_text or "").strip()
    if not text:
        return None
    obj = _read_object(text)
    if obj is not None:
        wants = obj.get("wants_direct_answer")
        return {
            "wants_direct_answer": wants if isinstance(wants, bool) else None,
            "signals": _filter_candidates(obj["items"]),
        }
    data = _candidate_array(text)
    if data is None:
        return None
    return {"wants_direct_answer": None, "signals": _filter_candidates(data)}


def parse_current_turn_llm_signals(raw_text: str) -> list[dict[str, str]] | None:
    """Parse the LLM's response into a list of `{"phrase", "type"}` dicts
    (plus `natural_question` when the model supplied one).

    Returns `[]` for a genuinely empty result (LLM found nothing -- a clean,
    expected outcome, not a failure). Returns `None` when the text could not
    be parsed as the expected shape at all -- callers must log this
    distinctly from a genuine empty result, per CLAUDE.md's fail-open
    logging convention for this call shape.
    """
    read = parse_current_turn_llm_read(raw_text)
    return None if read is None else read["signals"]


def _candidate_array(text: str) -> list | None:
    data: list | None = None
    for span in _balanced_json_spans(text, "[", "]"):
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
    return data


def _filter_candidates(data: list) -> list[dict[str, str]]:
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
        # Structural floor under the prompt's own filtering instruction, not a
        # replacement for it. Confirmed live 2026-08-21 (after the prompt's
        # interjection ban already shipped): a quick-lane model under a tight
        # max_tokens budget still returns bare single words as "concept"/"other"
        # candidates ("bus", "Glad", "Compact", "Interesting") -- the same
        # unactionable-garbage failure mode the deleted regex detector had, one
        # step removed. A single bare token is only ever a real trackable thing
        # when it names a person or place (see run_current_turn_signal_eval.py
        # for the labeled fixture this threshold is measured against) -- a bare
        # "concept"/"belief"/"activity"/"plan" is essentially never expressible
        # in one word ("the reactor rollout plan" is; "plan" alone is not).
        # `.split()` (not `" " not in phrase`, second review pass) -- str.split()
        # with no separator splits on any whitespace str.isspace() recognizes,
        # including non-breaking space (U+00A0) and other Unicode whitespace a
        # literal ASCII-space check would miss, misclassifying a real multi-word
        # phrase as a bare single token.
        is_bare_word = len(phrase.split()) < 2
        # Confirmed live 2026-08-22 (hours after this floor shipped): "bus" --
        # the exact garbage string this floor was built to stop -- got through
        # again, because the model typed it "place" that time instead of
        # "concept"/"other". The type/person-place carve-out alone trusts the
        # model's own classification, and that classification isn't reliably
        # consistent call to call for the same bare word. A genuine name/place
        # is capitalized by ordinary English convention ("Sarah", "Paris" in
        # every fixture below); a bare LOWERCASE word claimed to be a person
        # or place is essentially always a mistyped common noun, not a real
        # entity -- requiring capitalization on top of the type check is a
        # second, independent signal, not just re-deriving the same one.
        #
        # `not phrase[:1].islower()` -- NOT `phrase[:1].isupper()` (review
        # caught this): isupper() is False for any uncased script (CJK,
        # Arabic, Hebrew, Thai, ...), so a real bare name like "東京" or
        # "محمد" would be wrongly dropped -- a regression the pre-floor code
        # never had, since it only checked type_hint. islower() is equally
        # False for those scripts (no case distinction exists), so "not
        # islower()" correctly treats "no case signal available" as
        # non-disqualifying while still catching an affirmatively-lowercase
        # Latin word like "bus"/"glad".
        looks_like_a_name = not phrase[:1].islower() if phrase else False
        if is_bare_word and (type_hint not in {"person", "place"} or not looks_like_a_name):
            logger.debug(
                "current_turn_llm_signal_dropped_bare_word phrase=%r type=%s",
                phrase, type_hint,
            )
            continue
        if is_bare_word:
            # Disclosed, NOT fixed here (review, 2026-08-22): a sentence-
            # initial interjection is capitalized by ordinary English
            # convention too ("Heck", "Glad" were both capitalized in the
            # original live-garbage batch) -- if the model ever mistypes one
            # of those as person/place instead of other/activity/belief (not
            # yet observed, but this diff's own two incidents already prove
            # the type field is unreliable call-to-call for an identical
            # word), it reproduces the deleted LegacyRegexSignalDetector's
            # exact known failure mode one layer up. A static interjection
            # denylist would close this, but nothing has actually been
            # observed hitting it yet -- CLAUDE.md's metric-quality-gate
            # ("live-data sanity check" before wiring a new mechanism in)
            # argues against pre-building one on spec. Logged at INFO (not
            # DEBUG, unlike the drop case) specifically so this exact
            # highest-risk acceptance path is greppable/auditable if it ever
            # does start happening -- instrumentation first, per that gate.
            logger.info(
                "current_turn_llm_signal_bare_word_name_accepted phrase=%r type=%s",
                phrase, type_hint,
            )
        entry = {"phrase": phrase[:_MAX_PHRASE_LEN], "type": type_hint}
        raw_question = item.get("question")
        question = (
            " ".join(raw_question.split())[:_MAX_QUESTION_LEN].strip() if isinstance(raw_question, str) else ""
        )
        if question:
            entry["natural_question"] = question
        out.append(entry)
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
            "temperature": settings.current_turn_signal_probe_temperature,
            "purpose": "current_turn_signal_probe",
            "skip_spark_candidate_publish": True,
            "chat_template_kwargs": {"enable_thinking": False},
            # Tell the gateway how long this caller actually waits. Without it
            # the gateway assumed its default budget (700 s live) for a probe the
            # caller abandons after a few seconds, and admission queued it as if
            # someone were still listening (2026-10-06, turn-latency L3).
            "gateway_read_timeout_sec": float(settings.current_turn_signal_probe_timeout_sec),
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
        # Own rpc-health hop key. This probe is fail-open with a deadline set
        # below normal LLM latency on purpose, so its timeouts are a budget
        # choice, not a delivery failure; the label lets rpc-health consumers
        # (orion/substrate/rpc_delivery.py) tell it apart from real
        # LLMGatewayService traffic on the same channel.
        health_label=PROBE_HEALTH_LABEL,
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

    Also leaves `ctx["current_turn_llm_read"] = {"ok", "wants_direct_answer"}`,
    which `orion.substrate.attention.policy.direct_answer_cause` reads: ok=False
    on every failure path, so the policy can fail closed instead of guessing.
    """
    ctx["current_turn_llm_signals"] = []
    ctx["current_turn_llm_read"] = {"ok": False, "wants_direct_answer": None}
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

    read = parse_current_turn_llm_read(raw_text)
    if read is None:
        logger.warning(
            "current_turn_llm_signals_malformed_output raw_preview=%s",
            raw_text[:120],
        )
        return

    ctx["current_turn_llm_signals"] = read["signals"]
    ctx["current_turn_llm_read"] = {"ok": True, "wants_direct_answer": read["wants_direct_answer"]}
    logger.info(
        "current_turn_llm_read corr=%s wants_direct_answer=%s items=%d with_question=%d",
        ctx.get("correlation_id") or ctx.get("trace_id"),
        read["wants_direct_answer"],
        len(read["signals"]),
        sum(1 for s in read["signals"] if s.get("natural_question")),
    )
