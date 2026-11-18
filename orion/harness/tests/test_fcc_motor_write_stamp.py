"""The FCC motor stamps `written_at` on graph nodes Orion's turn wrote without it.

The model writes `orion_worldview` from inside `claude -p` via `redis-cli
GRAPH.QUERY`; the motor's stream loop is the code that sees each such command
finish. These tests drive that loop with a fake subprocess whose Bash "write"
lands in a fake graph, and check the stamp happens right after the tool result
-- and never touches a node that was unstamped before the turn began.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, List

import pytest

from orion.harness import fcc_motor as motor
from orion.curiosity.tests.fake_worldview_graph import FakeGraph


class _Stream:
    def __init__(self, lines: List[str], on_line=None) -> None:
        self._lines = [ln.encode() + b"\n" for ln in lines]
        self._on_line = on_line or (lambda _i: None)
        self._i = 0

    async def readline(self) -> bytes:
        await asyncio.sleep(0)
        if self._i >= len(self._lines):
            return b""
        self._on_line(self._i)
        line = self._lines[self._i]
        self._i += 1
        return line

    async def read(self) -> bytes:
        return b""


class _Proc:
    def __init__(self, stdout: _Stream) -> None:
        self.stdout = stdout
        self.stderr = _Stream([])
        self.returncode = 0

    def kill(self) -> None:
        self.returncode = -9

    async def wait(self) -> int:
        return self.returncode


WRITE_CMD = 'redis-cli -u "$U" GRAPH.QUERY orion_worldview "CREATE (:Hop {run_id: \\"new\\", n: 1, note: \\"x\\"})"'
LINES = [
    json.dumps({"type": "assistant", "message": {"content": [
        {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": WRITE_CMD}}]}}),
    json.dumps({"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": "Nodes created: 1"}]}}),
    json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "done"}]}}),
    json.dumps({"type": "result", "result": "done", "session_id": "s1"}),
]


def _patch_motor(monkeypatch: pytest.MonkeyPatch, proc: _Proc, graph: FakeGraph | None) -> None:
    async def fake_exec(*_a: Any, **_k: Any) -> _Proc:
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(motor, "_preflight_fcc_server", lambda *a, **k: None)
    monkeypatch.setattr(motor, "load_fcc_env", lambda _p: {"MODEL_CHAT": "x"})
    monkeypatch.setattr(motor, "label_to_claude_model_id", lambda *_a, **_k: "llamacpp/chat")
    monkeypatch.setattr(motor, "_maybe_render_mcp_config", lambda **_k: None)
    monkeypatch.setattr(
        motor.WriteStamper, "from_env",
        classmethod(lambda cls, _env, **_k: None if graph is None else cls(client=graph, graph_name="orion_worldview")),
    )


async def _drain(**kwargs: Any) -> list:
    events = []
    async for ev in motor.run_fcc_turn(
        prompt="hi", correlation_id="corr-stamp", workspace="/tmp",
        fcc_server_url="http://127.0.0.1:8082", auth_token="tok", claude_bin="claude",
        timeout_sec=5.0, **kwargs,
    ):
        events.append(ev)
    return events


@pytest.mark.asyncio
async def test_hop_written_without_written_at_is_stamped_after_its_tool_result(monkeypatch) -> None:
    graph = FakeGraph()
    legacy = graph.create("Hop", run_id="old", n=1, note="from an earlier run")
    created: dict[str, int] = {}
    stamped_before_next_step: dict[str, Any] = {}

    def on_line(i: int) -> None:
        if i == 1:  # the Bash command ran: its CREATE landed before its result line
            created["hop"] = graph.create("Hop", run_id="new", n=1, note="x")
        if i == 2:  # next model step: the stamp must already be there
            stamped_before_next_step["v"] = graph.props(created["hop"]).get("written_at")

    proc = _Proc(_Stream(LINES, on_line))
    _patch_motor(monkeypatch, proc, graph)
    events = await _drain()

    assert events[-1]["type"] == "final"
    assert graph.closed is True
    assert stamped_before_next_step["v"] is not None
    assert graph.props(created["hop"])["written_at_source"] == "harness_stamp"
    assert graph.props(legacy).get("written_at") is None


@pytest.mark.asyncio
async def test_unflagged_write_still_stamped_at_turn_end(monkeypatch) -> None:
    """A write the GRAPH.QUERY check cannot see (e.g. a script) gets the end-of-turn stamp."""
    graph = FakeGraph()
    legacy = graph.create("Hop", run_id="old", n=1)
    lines = [json.dumps({"type": "assistant", "message": {"content": [
                 {"type": "tool_use", "id": "s1", "name": "Bash", "input": {"command": "python3 write_graph.py"}}]}}),
             json.dumps({"type": "user", "message": {"content": [
                 {"type": "tool_result", "tool_use_id": "s1", "content": "ok"}]}}),
             json.dumps({"type": "result", "result": "hi", "session_id": "s"})]
    created: dict[str, int] = {}
    proc = _Proc(_Stream(lines, lambda i: created.setdefault("hop", graph.create("Finding", run_id="new")) if i == 0 else None))
    _patch_motor(monkeypatch, proc, graph)
    await _drain()
    assert graph.props(created["hop"])["written_at"] is not None
    assert graph.props(legacy).get("written_at") is None


@pytest.mark.asyncio
async def test_reading_only_turn_never_touches_the_graph(monkeypatch) -> None:
    graph = FakeGraph()
    proc = _Proc(_Stream(LINES))
    _patch_motor(monkeypatch, proc, graph)
    await _drain(reading_only=True)
    assert graph.calls == []


@pytest.mark.asyncio
async def test_graph_unreachable_at_turn_start_fails_closed(monkeypatch) -> None:
    """No baseline -> no stamp ever this turn, even once the graph comes back."""
    graph = FakeGraph()
    graph.fail = True
    created: dict[str, int] = {}

    def on_line(i: int) -> None:
        if i == 0:
            graph.fail = False  # recovered after the baseline already failed
        if i == 1:
            created["hop"] = graph.create("Hop", run_id="new", n=1)

    proc = _Proc(_Stream(LINES, on_line))
    _patch_motor(monkeypatch, proc, graph)
    events = await _drain()
    assert events[-1]["type"] == "final"
    assert [c for c in graph.calls if c[0] == "GRAPH.QUERY"] == []
    assert graph.props(created["hop"]).get("written_at") is None


@pytest.mark.asyncio
async def test_timed_out_turn_still_stamps_at_turn_end(monkeypatch) -> None:
    """The kill path: a node written before the stream stalls is stamped on the way out."""
    graph = FakeGraph()
    legacy = graph.create("Hop", run_id="old", n=1)
    created: dict[str, int] = {}

    class _Hang(_Stream):
        async def readline(self) -> bytes:
            if self._i >= len(self._lines):
                await asyncio.sleep(3600)
            return await super().readline()

    tool_use_no_graph = json.dumps({"type": "assistant", "message": {"content": [
        {"type": "tool_use", "id": "t9", "name": "Bash", "input": {"command": "python3 write_graph.py"}}]}})

    def on_line(i: int) -> None:
        if i == 0:
            created["hop"] = graph.create("Hop", run_id="new", n=1)

    proc = _Proc(_Hang([tool_use_no_graph], on_line))
    _patch_motor(monkeypatch, proc, graph)
    events = []
    async for ev in motor.run_fcc_turn(
        prompt="hi", correlation_id="corr-stamp-to", workspace="/tmp",
        fcc_server_url="http://127.0.0.1:8082", auth_token="tok", claude_bin="claude",
        timeout_sec=0.3,
    ):
        events.append(ev)
    assert events[-1]["type"] == "error"
    assert graph.props(created["hop"])["written_at_source"] == "harness_stamp"
    assert graph.props(legacy).get("written_at") is None


@pytest.mark.asyncio
async def test_toolless_turn_skips_the_turn_end_write(monkeypatch) -> None:
    graph = FakeGraph()
    lines = [json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "hi"}]}}),
             json.dumps({"type": "result", "result": "hi", "session_id": "s"})]
    proc = _Proc(_Stream(lines))
    _patch_motor(monkeypatch, proc, graph)
    await _drain()
    assert [c for c in graph.calls if c[0] == "GRAPH.QUERY"] == []
