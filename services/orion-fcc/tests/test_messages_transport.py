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


if __name__ == "__main__":
    unittest.main()
