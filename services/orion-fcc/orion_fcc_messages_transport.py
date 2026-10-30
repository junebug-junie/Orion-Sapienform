"""Wire-format repair for the pinned FCC Messages endpoint.

Provider routing/recovery stays upstream. Pings indicate transport liveness,
never model progress. Non-streaming replies require a complete message.
"""

import asyncio
import contextvars
import json
import logging
import re
from contextlib import suppress
from copy import deepcopy

import anyio
from core.anthropic.stream_contracts import parse_sse_text
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.requests import ClientDisconnect

PING = 'event: ping\ndata: {"type":"ping"}\n\n'

# uvicorn's logger is the one whose INFO lines reach `docker logs` in this container.
logger = logging.getLogger("uvicorn.error")

# Fixed text of Claude Code's compaction prompts (2.1.304, the CLI in the governor and Hub):
# the no-tools preamble, and the opening shared by its full, partial ("RECENT portion") and
# "this conversation" wordings. The CLI sends that prompt as the last user turn WITH the turn's tools still attached
# and only asks, in text, for no tool calls. Claude models obey; the local Qwen lanes often
# call a tool, and the CLI keeps the text before the call ("Now let me read the remaining
# files...") as the summary, so the task is lost: 33 of 51 live compactions on 2026-10-17.
COMPACTION_PROMPT_MARKS = (
    "CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.",
    "Your task is to create a detailed summary of ",
)


def _text_of(content):
    if isinstance(content, str):
        return content
    parts = []
    for block in content or ():
        text = getattr(block, "text", None)
        if text is None and isinstance(block, dict):
            text = block.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts)


def is_compaction_request(request_data):
    """True when the last turn is the CLI's own compaction prompt from the user role."""
    messages = getattr(request_data, "messages", None) or []
    if not messages:
        return False
    last = messages[-1]
    if getattr(last, "role", None) != "user":
        return False
    text = _text_of(getattr(last, "content", None))
    return any(mark in text for mark in COMPACTION_PROMPT_MARKS)


def _field(block, name):
    return block.get(name) if isinstance(block, dict) else getattr(block, name, None)


def _blocks(content):
    return [content] if isinstance(content, str) else list(content or ())


def _render_block(block):
    """One content block as plain transcript text; None for blocks a summary does not need."""
    if isinstance(block, str):
        return block
    kind = _field(block, "type")
    if kind == "text":
        return _field(block, "text")
    if kind in ("tool_use", "server_tool_use"):
        args = json.dumps(jsonable_encoder(_field(block, "input")), ensure_ascii=False)
        return f"[called tool {_field(block, 'name')} with {args}]"
    if kind == "tool_result":
        body = _field(block, "content")
        parts = [_render_block(b) for b in _blocks(body if isinstance(body, (str, list)) else [body])]
        text = "\n".join(p for p in parts if p)
        return f"[tool result{' (error)' if _field(block, 'is_error') else ''}]\n{text}"
    if kind in ("web_search_tool_result", "web_fetch_tool_result"):
        body = json.dumps(jsonable_encoder(_field(block, "content")), ensure_ascii=False)
        return f"[{kind}]\n{body[:_WEB_RESULT_CHARS]}"
    if kind in ("image", "document"):
        return f"[{kind} omitted]"
    return None  # thinking / redacted_thinking: the visible turns already carry the work


_WEB_RESULT_CHARS = 4000  # raw search/fetch JSON is mostly markup; the summary needs the gist
_TRANSCRIPT_CLOSE = re.compile(r"</\s*transcript\s*>", re.IGNORECASE)


def _quoted(text):
    """Text that cannot close the transcript quote early (agents read files holding the tag)."""
    return _TRANSCRIPT_CLOSE.sub("<\\/transcript>", text)


def _section(role, texts):
    return f"### {str(role).upper()}\n" + "\n".join(_quoted(t) for t in texts)


def flatten_compaction_text(messages):
    """The whole compaction request as one text: a quoted transcript, then the CLI's prompt.

    Stripping tools was not enough. The history still ends mid agent loop (tool calls,
    tool results), so the local Qwen keeps working and writes its next tool call as plain
    text, and the CLI keeps that text as the summary. Live 2026-10-30, transcript
    7e25128b: tools stripped, model called for 41 s, "summary" = `<tool_call><function=Bash>`.
    A transcript inside one user turn leaves no agent turn to continue, only a document
    to summarize."""
    *history, last = messages
    lines = []
    for message in history:
        rendered = [t for t in map(_render_block, _blocks(_field(message, "content"))) if t]
        if rendered:
            lines.append(_section(_field(message, "role"), rendered))
    # The CLI merges adjacent user turns, so the compaction turn can also carry tool results,
    # reminders or a queued message. Only the prompt itself stays outside the quote.
    prompt, rest = [], []
    for block in _blocks(_field(last, "content")):
        text = _render_block(block)
        if text:
            is_prompt = _field(block, "type") != "tool_result" and any(m in text for m in COMPACTION_PROMPT_MARKS)
            (prompt if is_prompt else rest).append(text)
    if rest:
        lines.append(_section("user", rest))
    transcript = "\n\n".join(lines)
    return (
        "Below is the transcript of an earlier agent conversation, quoted for you to read. "
        "It is finished: do not continue it and do not call any tools.\n\n"
        f"<transcript>\n{transcript}\n</transcript>\n\n" + "\n".join(prompt)
    )


def prepare_request(request_data):
    """Compaction requests go upstream with no tools and the history as one quoted transcript.

    No tools: the model can only answer in text. One transcript (`flatten_compaction_text`):
    there is no agent loop left for it to carry on.
    Side effect: upstream FCC's no-tools local shortcuts (api/optimization_handlers.py) now see
    this request, and its title detector DOES match the CLI's system prompt -- see
    `guard_local_optimizations`, which must be installed alongside this.
    Every other request is returned unchanged (same object)."""
    if not is_compaction_request(request_data):
        return request_data
    messages = request_data.messages
    logger.info(
        "fcc_compaction_flattened model=%s tools=%d messages=%d",
        getattr(request_data, "model", None), len(getattr(request_data, "tools", None) or ()),
        len(messages),
    )
    turn = type(messages[-1]).model_validate({"role": "user", "content": flatten_compaction_text(messages)})
    return request_data.model_copy(update={"tools": None, "tool_choice": None, "messages": [turn]})


def guard_local_optimizations():
    """Keep FCC's local shortcuts away from compaction requests.

    Once `prepare_request` strips their tools, the CLI's own system prompt satisfies FCC's
    title-generation detector, which answers "Conversation" in ~20 ms instead of calling the
    model. That one word became every compaction summary from 2026-10-21 (19 of 19 live), so
    each long turn lost its task after compacting. Idempotent; fails loudly if upstream moves
    the hook, so a pin bump cannot silently drop the guard."""
    import api.handlers.messages as handlers

    original = handlers.try_optimizations
    if getattr(original, "orion_compaction_guard", False):
        return

    def guarded(request_data, settings):
        if is_compaction_request(request_data):
            logger.info("fcc_compaction_local_shortcut_skipped model=%s", getattr(request_data, "model", None))
            return None
        return original(request_data, settings)

    guarded.orion_compaction_guard = True
    handlers.try_optimizations = guarded


# The harness turn id. Claude Code sends it on every model call (ANTHROPIC_CUSTOM_HEADERS, set by
# orion/harness/fcc_motor._build_subprocess_env); FCC builds a brand-new upstream request, so
# without this the header stops here and llm-gateway's per-call token ledger (llm_call_usage)
# records every harness call with correlation_id NULL -- live 2026-10-29: 17 of 17 rows.
CORRELATION_HEADER = "X-Orion-Correlation-Id"
_CORRELATION_MAX = 128  # llm-gateway clips to the GPU pool lease's turn_correlation_id limit
_correlation = contextvars.ContextVar("orion_correlation_id", default=None)


def bind_correlation(request):
    """Remember this request's turn id for the upstream call it causes.

    Set in the route's own task; the provider's upstream send runs in tasks spawned after this
    (the response body iterator, the heartbeat reader), which copy the context, so the value
    reaches them and never another request. A missing or unsafe value clears it."""
    value = str(request.headers.get(CORRELATION_HEADER) or "").strip()
    if not value or "\r" in value or "\n" in value:
        value = None
    _correlation.set(value[:_CORRELATION_MAX] if value else None)


def forward_correlation_header():
    """Add the bound turn id to FCC's native Anthropic upstream request headers.

    The llama.cpp provider (``LLAMACPP_BASE_URL`` = llm-gateway) is an
    ``AnthropicMessagesTransport``; every messages call takes its headers from
    ``_request_headers``. Idempotent; fails loudly if upstream moves the hook, so a pin bump
    cannot silently drop the forward."""
    from providers.transports.anthropic_messages.transport import AnthropicMessagesTransport

    original = AnthropicMessagesTransport._request_headers
    if getattr(original, "orion_correlation_forward", False):
        return

    def with_correlation(self):
        headers = dict(original(self))
        corr = _correlation.get()
        if corr:
            headers[CORRELATION_HEADER] = corr
        return headers

    with_correlation.orion_correlation_forward = True
    AnthropicMessagesTransport._request_headers = with_correlation


async def with_heartbeats(body, interval=15.0):
    iterator = body.__aiter__()
    pending = None
    try:
        yield PING
        while True:
            pending = asyncio.create_task(anext(iterator))
            while not (await asyncio.wait({pending}, timeout=interval))[0]:
                yield PING
            try:
                chunk = pending.result()
            except StopAsyncIteration:
                return
            pending = None
            yield chunk
    finally:
        # Starlette disconnects under an already-cancelled AnyIO scope. Allow
        # the provider's async socket/lease cleanup to finish after cancellation.
        with anyio.CancelScope(shield=True):
            if pending is not None:
                pending.cancel()
                with suppress(asyncio.CancelledError, StopAsyncIteration):
                    await pending
            if hasattr(iterator, "aclose"):
                await iterator.aclose()


class MessageAssemblyError(ValueError):
    pass


class ProviderStreamError(MessageAssemblyError):
    def __init__(self, payload):
        super().__init__("Provider returned an error event")
        self.payload = deepcopy(payload)


async def collect_message(body):
    message = None
    blocks = {}
    open_blocks = set()
    partial_inputs = {}
    stopped = False
    try:
        async for chunk in body:
            # FCC provider iterators yield complete SSE events, not HTTP chunks.
            for event in parse_sse_text(chunk):
                data = event.data
                kind = data.get("type")
                if kind == "ping":
                    continue
                if kind == "error":
                    raise ProviderStreamError(data)
                if stopped:
                    raise MessageAssemblyError("Event after message_stop")
                if kind == "message_start":
                    if message is not None:
                        raise MessageAssemblyError("Duplicate message_start")
                    message = deepcopy(data["message"])
                    continue
                if message is None:
                    raise MessageAssemblyError("Missing message_start")
                if kind == "content_block_start":
                    index = data["index"]
                    if index in blocks:
                        raise MessageAssemblyError("Duplicate content block")
                    blocks[index] = deepcopy(data["content_block"])
                    open_blocks.add(index)
                elif kind == "content_block_delta":
                    index = data["index"]
                    if index not in open_blocks:
                        raise MessageAssemblyError("Delta outside open block")
                    block, delta = blocks[index], data["delta"]
                    delta_type = delta["type"]
                    fields = {
                        "text_delta": "text",
                        "thinking_delta": "thinking",
                        "signature_delta": "signature",
                    }
                    if delta_type in fields:
                        field = fields[delta_type]
                        block[field] = block.get(field, "") + delta[field]
                    elif delta_type == "input_json_delta":
                        partial_inputs[index] = (
                            partial_inputs.get(index, "") + delta["partial_json"]
                        )
                    elif delta_type == "citations_delta":
                        block.setdefault("citations", []).append(
                            deepcopy(delta["citation"])
                        )
                    else:
                        raise MessageAssemblyError("Unsupported content delta")
                elif kind == "content_block_stop":
                    index = data["index"]
                    if index not in open_blocks:
                        raise MessageAssemblyError("Stop outside open block")
                    open_blocks.remove(index)
                    if index in partial_inputs:
                        value = json.loads(partial_inputs.pop(index))
                        if not isinstance(value, dict):
                            raise MessageAssemblyError("Tool input must be an object")
                        blocks[index]["input"] = value
                elif kind == "message_delta":
                    message.update(deepcopy(data["delta"]))
                    message.setdefault("usage", {}).update(data.get("usage", {}))
                elif kind == "message_stop":
                    stopped = True
                else:
                    raise MessageAssemblyError("Unsupported message event")
        if not stopped or open_blocks or not message or not message.get("stop_reason"):
            raise MessageAssemblyError("Incomplete provider message")
        message["content"] = [blocks[index] for index in sorted(blocks)]
        return message
    finally:
        if hasattr(body, "aclose"):
            await body.aclose()


async def collect_until_disconnect(body, disconnected):
    pending = asyncio.create_task(collect_message(body))
    try:
        while not (await asyncio.wait({pending}, timeout=0.25))[0]:
            if disconnected is not None and await disconnected():
                raise ClientDisconnect()
        return pending.result()
    finally:
        with anyio.CancelScope(shield=True):
            if not pending.done():
                pending.cancel()
                with suppress(asyncio.CancelledError):
                    await pending


async def adapt_message_response(response, *, stream, disconnected=None):
    if not isinstance(response, StreamingResponse):
        # Keep upstream local optimization responses unchanged.
        return response
    if stream:
        response.body_iterator = with_heartbeats(response.body_iterator)
        return response
    try:
        message = await collect_until_disconnect(response.body_iterator, disconnected)
    except ProviderStreamError as exc:
        return JSONResponse(status_code=502, content=exc.payload)
    except (MessageAssemblyError, KeyError, TypeError, ValueError):
        return JSONResponse(
            status_code=502,
            content={
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": "FCC could not assemble a complete upstream message.",
                },
            },
        )
    return JSONResponse(
        content=jsonable_encoder(message), background=response.background
    )
