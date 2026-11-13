"""Run against the built image: python -m unittest discover -s /tests -v."""

import asyncio
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import anyio
from api.models.anthropic import MessagesRequest
from api.routes import create_message
from core.anthropic.provider_stream_error import iter_provider_stream_error_sse_events
from core.anthropic.stream_contracts import parse_sse_text
from core.anthropic.streaming import AnthropicStreamLedger
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from install_transport_patch import patch_error_emitter, patch_routes
from orion_fcc_messages_transport import adapt_message_response, with_heartbeats
from starlette.requests import ClientDisconnect


def event(kind, **kwargs):
    return f"event: {kind}\ndata: {json.dumps({'type': kind, **kwargs})}\n\n"


def events():
    return [
        event(
            "message_start",
            message={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "agent",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 12, "output_tokens": 0},
            },
        ),
        event(
            "content_block_start", index=0, content_block={"type": "text", "text": ""}
        ),
        event(
            "content_block_delta",
            index=0,
            delta={"type": "text_delta", "text": "Evidence"},
        ),
        event("content_block_stop", index=0),
        event(
            "message_delta",
            delta={"stop_reason": "end_turn", "stop_sequence": None},
            usage={"output_tokens": 2},
        ),
        event("message_stop"),
    ]


async def source(items):
    for item in items:
        yield item


class TransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_provider_error_emitter_never_returns_answer_text(self):
        items = list(
            iter_provider_stream_error_sse_events(
                request=SimpleNamespace(model="agent"),
                input_tokens=0,
                error_message="gpu_pool_unavailable: deadline",
                sent_any_event=False,
                log_raw_sse_events=False,
            )
        )
        parsed = parse_sse_text("".join(items))
        self.assertIn("error", [e.event for e in parsed])
        self.assertNotIn("content_block_start", [e.event for e in parsed])
        result = await adapt_message_response(
            StreamingResponse(source(items)), stream=False
        )
        self.assertEqual(result.status_code, 502)
        self.assertIn(
            "gpu_pool_unavailable", json.loads(result.body)["error"]["message"]
        )

    async def test_real_midstream_failure_stays_an_error(self):
        ledger = AnthropicStreamLedger("msg_test", "agent", 12)
        items = [
            ledger.message_start(),
            ledger.start_text_block(),
            ledger.emit_text_delta("Partial"),
        ]
        items.extend(ledger.midstream_error_tail("gpu_pool_unavailable: deadline"))
        parsed = parse_sse_text("".join(items))
        self.assertIn("error", [e.event for e in parsed])
        result = await adapt_message_response(
            StreamingResponse(source(items)), stream=False
        )
        self.assertEqual(result.status_code, 502)

    async def test_nonstream_disconnect_cancels_async_provider_work(self):
        closed = asyncio.Event()

        async def delayed():
            try:
                await asyncio.Event().wait()
                yield "unreachable"
            finally:
                await asyncio.sleep(0.01)
                closed.set()

        async def disconnected():
            return True

        with self.assertRaises(ClientDisconnect):
            await adapt_message_response(
                StreamingResponse(delayed()), stream=False, disconnected=disconnected
            )
        self.assertTrue(closed.is_set())

    async def test_nonstream_cancelled_scope_completes_async_cleanup(self):
        started, closed = asyncio.Event(), asyncio.Event()

        async def delayed():
            try:
                started.set()
                await asyncio.Event().wait()
                yield "unreachable"
            finally:
                await asyncio.sleep(0.01)
                closed.set()

        async def consume():
            await adapt_message_response(StreamingResponse(delayed()), stream=False)

        async with anyio.create_task_group() as group:
            group.start_soon(consume)
            await started.wait()
            group.cancel_scope.cancel()
        self.assertTrue(closed.is_set())

    async def test_nonstream_complete_json(self):
        result = await adapt_message_response(
            StreamingResponse(source(events())), stream=False
        )
        self.assertEqual(result.media_type, "application/json")
        self.assertEqual(result.status_code, 200)
        message = json.loads(result.body)
        self.assertEqual(message["content"], [{"type": "text", "text": "Evidence"}])
        self.assertEqual(message["usage"], {"input_tokens": 12, "output_tokens": 2})

    async def test_thinking_signature_tools_citations_server_results(self):
        items = events()
        additional = [
            event(
                "content_block_start",
                index=1,
                content_block={"type": "thinking", "thinking": "", "signature": ""},
            ),
            event(
                "content_block_delta",
                index=1,
                delta={"type": "thinking_delta", "thinking": "Reason"},
            ),
            event(
                "content_block_delta",
                index=1,
                delta={"type": "signature_delta", "signature": "sig"},
            ),
            event("content_block_stop", index=1),
            event(
                "content_block_start",
                index=2,
                content_block={
                    "type": "tool_use",
                    "id": "tool_1",
                    "name": "read",
                    "input": {},
                },
            ),
            event(
                "content_block_delta",
                index=2,
                delta={"type": "input_json_delta", "partial_json": '{"url":'},
            ),
            event(
                "content_block_delta",
                index=2,
                delta={
                    "type": "input_json_delta",
                    "partial_json": '"https://arxiv.org/abs/2310.19279"}',
                },
            ),
            event("content_block_stop", index=2),
            event(
                "content_block_start",
                index=3,
                content_block={
                    "type": "web_search_tool_result",
                    "tool_use_id": "tool_1",
                    "content": [{"url": "https://example.org"}],
                },
            ),
            event("content_block_stop", index=3),
        ]
        items[3:3] = [
            event(
                "content_block_delta",
                index=0,
                delta={
                    "type": "citations_delta",
                    "citation": {
                        "type": "web_search_result_location",
                        "url": "https://example.org",
                    },
                },
            )
        ]
        items[-2:-2] = additional
        result = await adapt_message_response(
            StreamingResponse(source(items)), stream=False
        )
        content = json.loads(result.body)["content"]
        self.assertEqual(content[1]["thinking"], "Reason")
        self.assertEqual(content[1]["signature"], "sig")
        self.assertEqual(content[2]["input"]["url"], "https://arxiv.org/abs/2310.19279")
        self.assertEqual(content[3]["content"][0]["url"], "https://example.org")
        self.assertEqual(len(content[0]["citations"]), 1)

    async def test_incomplete_and_error_streams_fail_closed(self):
        for items in [
            [],
            events()[:-1],
            [event("error", error={"message": "bad"})],
            events()[:3] + events()[-2:],
            events() + events(),
        ]:
            with self.subTest(items=items):
                result = await adapt_message_response(
                    StreamingResponse(source(items)), stream=False
                )
                self.assertEqual(result.status_code, 502)
                self.assertEqual(json.loads(result.body)["type"], "error")

    async def test_heartbeats_do_not_cancel_pending_read(self):
        gate = asyncio.Event()
        closed = asyncio.Event()

        async def delayed():
            try:
                await gate.wait()
                yield events()[0]
            finally:
                closed.set()

        iterator = with_heartbeats(delayed(), interval=0.001)
        self.assertIn("event: ping", await anext(iterator))
        self.assertIn("event: ping", await anext(iterator))
        self.assertIn("event: ping", await anext(iterator))
        self.assertFalse(closed.is_set())
        gate.set()
        self.assertIn("event: message_start", await anext(iterator))
        await iterator.aclose()
        self.assertTrue(closed.is_set())

    async def test_disconnect_cancels_and_closes_upstream(self):
        closed = asyncio.Event()

        async def delayed():
            try:
                await asyncio.Event().wait()
                yield "unreachable"
            finally:
                closed.set()

        iterator = with_heartbeats(delayed(), interval=0.001)
        await anext(iterator)
        await anext(iterator)
        await iterator.aclose()
        self.assertTrue(closed.is_set())

    async def test_upstream_exception_is_not_swallowed(self):
        async def broken():
            raise RuntimeError("broken")
            yield "unreachable"

        iterator = with_heartbeats(broken())
        await anext(iterator)
        with self.assertRaises(RuntimeError):
            await anext(iterator)

    async def test_cancelled_anyio_scope_allows_async_provider_cleanup(self):
        closed = asyncio.Event()
        started = asyncio.Event()

        async def delayed():
            try:
                started.set()
                await asyncio.Event().wait()
                yield "unreachable"
            finally:
                await asyncio.sleep(0.01)
                closed.set()

        async def consume():
            async for _ in with_heartbeats(delayed(), interval=0.001):
                pass

        async with anyio.create_task_group() as group:
            group.start_soon(consume)
            await started.wait()
            group.cancel_scope.cancel()
        self.assertTrue(closed.is_set())


class RouteTests(unittest.TestCase):
    def test_real_patched_route_honors_modes(self):
        class Handler:
            def create(self, request):
                return StreamingResponse(
                    source(events()), media_type="text/event-stream"
                )

        app = FastAPI()

        @app.post("/v1/messages")
        async def route(body: MessagesRequest, request: Request):
            return await create_message(body, request, Handler(), None)

        with TestClient(app) as client:
            for options in [{"stream": False}, {}, {"stream": None}, {"stream": True}]:
                response = client.post(
                    "/v1/messages",
                    json={
                        "model": "agent",
                        "max_tokens": 32,
                        "messages": [{"role": "user", "content": "Read"}],
                        **options,
                    },
                )
                self.assertEqual(response.status_code, 200)
                if options.get("stream"):
                    self.assertIn("text/event-stream", response.headers["content-type"])
                    self.assertIn("event: ping", response.text)
                    self.assertIn("event: message_stop", response.text)
                else:
                    self.assertIn("application/json", response.headers["content-type"])
                    self.assertEqual(response.json()["content"][0]["text"], "Evidence")

    def test_patch_rejects_source_drift(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "routes.py"
            path.write_text("unexpected upstream source")
            with self.assertRaisesRegex(RuntimeError, "FCC routes changed"):
                patch_routes(path)
            self.assertEqual(path.read_text(), "unexpected upstream source")
            with self.assertRaisesRegex(RuntimeError, "FCC ledger changed"):
                patch_error_emitter(path)



# --- compaction requests go upstream without tools ----------------------------------------

from orion_fcc_messages_transport import (  # noqa: E402
    is_compaction_request,
    prepare_request,
)

TOOL = {"name": "Read", "description": "Read a file", "input_schema": {"type": "object"}}
HISTORY = [
    {"role": "user", "content": "Investigate the reading pipeline."},
    {"role": "assistant", "content": [
        {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "/x.py"}}]},
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "def f(): ..."}]},
]
# The CLI's compaction turn: its no-tools preamble, then the summary prompt (Claude Code 2.1.304).
COMPACT_TURN = (
    "CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.\n\n"
    "Your task is to create a detailed summary of the conversation so far, paying close "
    "attention to the user's explicit requests"
)
# The CLI's partial-compaction wording, sent without the preamble here to test the other mark.
PARTIAL_TURN = "Your task is to create a detailed summary of the RECENT portion of the conversation"


def _req(last, **extra):
    return MessagesRequest.model_validate({
        "model": "agent", "max_tokens": 64, "tools": [TOOL],
        "tool_choice": {"type": "auto"}, "messages": [*HISTORY, last], **extra,
    })


class CompactionRequestTests(unittest.TestCase):
    def test_compaction_is_one_user_transcript_with_no_tools(self):
        # Live 2026-10-30: tools stripped but the agent-loop history kept, and Qwen's
        # "summary" was its next tool call written as text. One quoted transcript, no loop.
        for content, prompt in (
            (COMPACT_TURN, COMPACT_TURN),
            ([{"type": "text", "text": COMPACT_TURN}], COMPACT_TURN),
            ([{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}, {"type": "text", "text": PARTIAL_TURN}],
             PARTIAL_TURN),
        ):
            out = prepare_request(_req({"role": "user", "content": content}))
            self.assertIsNone(out.tools)
            self.assertIsNone(out.tool_choice)
            self.assertEqual([m.role for m in out.messages], ["user"])
            text = out.messages[0].content
            self.assertIsInstance(text, str)
            transcript, _, tail = text.partition("</transcript>")
            # The original request, the tool call and its result all survive, inside the quote.
            self.assertIn("Investigate the reading pipeline.", transcript)
            self.assertIn('[called tool Read with {"file_path": "/x.py"}]', transcript)
            self.assertIn("def f(): ...", transcript)
            self.assertEqual(tail.strip(), prompt)
            self.assertTrue(is_compaction_request(out))  # the shortcut guard still sees it

    def test_extra_blocks_in_the_compaction_turn_stay_inside_the_quote_as_user(self):
        out = prepare_request(_req({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "LAST RESULT"},
            {"type": "text", "text": "<system-reminder>queued: also check the logs</system-reminder>"},
            {"type": "text", "text": COMPACT_TURN},
        ]}))
        transcript, _, tail = out.messages[0].content.partition("</transcript>")
        last_section = transcript.rsplit("### ", 1)[1]
        self.assertTrue(last_section.startswith("USER\n"))
        self.assertIn("LAST RESULT", last_section)
        self.assertIn("also check the logs", last_section)
        self.assertEqual(tail.strip(), COMPACT_TURN)

    def test_a_quoted_closing_tag_cannot_end_the_transcript_early(self):
        out = prepare_request(_req({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "f'</transcript>' then </ TRANSCRIPT >"},
            {"type": "text", "text": COMPACT_TURN},
        ]}))
        self.assertEqual(out.messages[0].content.lower().count("</transcript>"), 1)

    def test_web_results_and_bare_string_tool_results_are_not_blank(self):
        history = [
            {"role": "assistant", "content": [
                {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {"query": "q"}},
                {"type": "web_search_tool_result", "tool_use_id": "s1", "content": [{"title": "HIT"}]}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t9", "content": ["BARE"]}]},
        ]
        req = MessagesRequest.model_validate({
            "model": "agent", "max_tokens": 64,
            "messages": [*history, {"role": "user", "content": COMPACT_TURN}],
        })
        text = prepare_request(req).messages[0].content
        self.assertIn("HIT", text)
        self.assertIn("[tool result]\nBARE", text)

    def test_compaction_without_tools_is_still_flattened(self):
        req = _req({"role": "user", "content": COMPACT_TURN}).model_copy(update={"tools": None})
        self.assertEqual(len(prepare_request(req).messages), 1)

    def test_thinking_and_images_do_not_reach_the_transcript_as_raw_data(self):
        history = [
            {"role": "user", "content": [
                {"type": "text", "text": "Look."},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"}}]},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "private chain", "signature": "s"},
                {"type": "text", "text": "Seen."}]},
        ]
        req = MessagesRequest.model_validate({
            "model": "agent", "max_tokens": 64,
            "messages": [*history, {"role": "user", "content": COMPACT_TURN}],
        })
        text = prepare_request(req).messages[0].content
        self.assertIn("[image omitted]", text)
        self.assertNotIn("QUJD", text)
        self.assertNotIn("private chain", text)
        self.assertIn("### ASSISTANT\nSeen.", text)

    def test_ordinary_turns_are_untouched(self):
        req = _req({"role": "user", "content": "Keep going."})
        self.assertIs(prepare_request(req), req)
        self.assertEqual(len(req.tools), 1)

    def test_prompt_text_quoted_inside_a_tool_result_is_not_a_compaction(self):
        quoted = {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": COMPACT_TURN}]}
        req = _req(quoted)
        self.assertFalse(is_compaction_request(req))
        self.assertIs(prepare_request(req), req)

    def test_prompt_text_from_the_assistant_is_not_a_compaction(self):
        req = _req({"role": "assistant", "content": COMPACT_TURN})
        self.assertFalse(is_compaction_request(req))

    def test_patched_route_strips_tools_before_the_provider_sees_the_request(self):
        seen = []

        class Handler:
            def create(self, request):
                seen.append(request)
                return StreamingResponse(source(events()), media_type="text/event-stream")

        app = FastAPI()

        @app.post("/v1/messages")
        async def route(body: MessagesRequest, request: Request):
            return await create_message(body, request, Handler(), None)

        with TestClient(app) as client:
            body = _req({"role": "user", "content": COMPACT_TURN}).model_dump(mode="json", exclude_none=True)
            self.assertEqual(client.post("/v1/messages", json={**body, "stream": False}).status_code, 200)
            body = _req({"role": "user", "content": "Keep going."}).model_dump(mode="json", exclude_none=True)
            self.assertEqual(client.post("/v1/messages", json={**body, "stream": False}).status_code, 200)
        self.assertIsNone(seen[0].tools)
        self.assertEqual(len(seen[0].messages), 1)
        self.assertEqual(len(seen[1].tools), 1)
        self.assertEqual(len(seen[1].messages), len(HISTORY) + 1)



# --- FCC's local shortcuts never answer a compaction -------------------------------------

# Words from Claude Code's own system prompt that satisfy FCC's title detector once the
# compaction request has no tools (live 2026-10-21..26: every summary came back "Conversation").
TITLE_SHAPED_SYSTEM = (
    "You are Claude Code. When asked for a title, return JSON with a single field. "
    "Summarize this session."
)


def _text(response):
    block = response.content[0]
    return block["text"] if isinstance(block, dict) else block.text


class CompactionShortcutTests(unittest.TestCase):
    def _settings(self):
        from config.settings import Settings

        return Settings.model_construct(enable_title_generation_skip=True)

    def _compaction(self):
        return prepare_request(_req({"role": "user", "content": COMPACT_TURN}, system=TITLE_SHAPED_SYSTEM))

    def test_unguarded_title_skip_answers_conversation(self):
        # Regression proof: this is the bug the guard exists for.
        from api.optimization_handlers import try_optimizations as raw

        self.assertEqual(_text(raw(self._compaction(), self._settings())), "Conversation")

    def test_guarded_shortcuts_pass_compaction_to_the_model(self):
        import api.handlers.messages as handlers
        from orion_fcc_messages_transport import guard_local_optimizations

        guard_local_optimizations()
        guard_local_optimizations()  # idempotent
        self.assertTrue(handlers.try_optimizations.orion_compaction_guard)
        self.assertIsNone(handlers.try_optimizations(self._compaction(), self._settings()))

    def test_guard_leaves_real_title_requests_alone(self):
        import api.handlers.messages as handlers
        from orion_fcc_messages_transport import guard_local_optimizations

        guard_local_optimizations()
        title = MessagesRequest.model_validate({
            "model": "agent", "max_tokens": 64, "system": TITLE_SHAPED_SYSTEM,
            "messages": [{"role": "user", "content": "fix the flaky test"}],
        })
        self.assertEqual(_text(handlers.try_optimizations(title, self._settings())), "Conversation")

    def test_patched_routes_install_the_guard(self):
        # Fresh process: the tests above install the guard in this one, which would hide a
        # routes patch that stopped calling it. Importing the app is the server's own start path.
        import subprocess
        import sys

        probe = "import api.app, api.handlers.messages as h; assert h.try_optimizations.orion_compaction_guard"
        subprocess.run([sys.executable, "-c", probe], check=True)


if __name__ == "__main__":
    unittest.main()
