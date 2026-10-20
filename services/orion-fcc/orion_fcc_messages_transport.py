"""Wire-format repair for the pinned FCC Messages endpoint.

Provider routing/recovery stays upstream. Pings indicate transport liveness,
never model progress. Non-streaming replies require a complete message.
"""

import asyncio
import json
import logging
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

# Opening sentence of Claude Code's compaction prompt (2.1.304, the CLI in the governor and
# Hub). The CLI sends that prompt as the last user turn WITH the turn's tools still attached
# and only asks, in text, for no tool calls. Claude models obey; the local Qwen lanes often
# call a tool, and the CLI keeps the text before the call ("Now let me read the remaining
# files...") as the summary, so the task is lost: 33 of 51 live compactions on 2026-10-17.
COMPACTION_PROMPT_MARK = "Your task is to create a detailed summary of the conversation so far"


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
    return COMPACTION_PROMPT_MARK in _text_of(getattr(last, "content", None))


def prepare_request(request_data):
    """Compaction requests go upstream with no tools, so the model can only answer in text.

    Earlier tool calls in the history stay as they are; only the offer of new ones is removed.
    Every other request is returned unchanged (same object)."""
    if not is_compaction_request(request_data) or not getattr(request_data, "tools", None):
        return request_data
    logger.info(
        "fcc_compaction_tools_stripped model=%s tools=%d messages=%d",
        getattr(request_data, "model", None), len(request_data.tools), len(request_data.messages),
    )
    return request_data.model_copy(update={"tools": None, "tool_choice": None})


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
