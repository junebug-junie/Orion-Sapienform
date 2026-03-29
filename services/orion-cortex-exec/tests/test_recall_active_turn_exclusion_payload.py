from __future__ import annotations

import asyncio

from app import executor


class _FakeRecallClient:
    captured = None

    def __init__(self, bus):
        self.bus = bus

    async def query(self, *, source, req, correlation_id, reply_to, timeout_sec):
        _FakeRecallClient.captured = req

        class _Item:
            def model_dump(self, mode="json"):
                return {"id": "x", "source": "vector", "snippet": "memory"}

            id = "x"
            snippet = "memory"
            score = 0.5
            tags = []
            source = "vector"
            source_ref = None
            uri = None

        class _Bundle:
            items = [_Item()]
            rendered = "ok"

            def model_dump(self, mode="json"):
                return {"items": [], "rendered": "ok", "stats": {}}

        class _Res:
            bundle = _Bundle()
            debug = {
                "decision": {
                    "dropped": {"transcript_novelty": 1},
                    "recall_debug": {
                        "source_gating": {"vector": "enabled", "sql_timeline": "disabled_by_profile_or_global"},
                        "selected_summary": [{"id": "x", "source": "vector", "score": 0.5}],
                    },
                }
            }

        return _Res()


def test_run_recall_step_populates_active_turn_exclusion(monkeypatch):
    monkeypatch.setattr(executor, "RecallClient", _FakeRecallClient)
    monkeypatch.setattr(executor.settings, "step_timeout_ms", 1000)

    ctx = {
        "verb": "chat_general",
        "messages": [{"role": "user", "content": "Teddy loves Addy"}],
        "trace_id": "trace-1",
        "request_id": "req-1",
    }

    step, debug, _ = asyncio.run(executor.run_recall_step(
        bus=object(),
        source=type("Source", (), {"name": "test", "version": "0", "node": "n"})(),
        ctx=ctx,
        correlation_id="corr-1",
        recall_cfg={},
        recall_profile="reflect.v1",
    ))

    assert step.status == "success"
    req = _FakeRecallClient.captured
    assert req.exclude is not None
    assert req.exclude.get("active_turn_text") == "Teddy loves Addy"
    assert "corr-1" in req.exclude.get("active_turn_ids")
    assert "trace-1" in req.exclude.get("active_turn_ids")
    assert debug.get("drop_counts") == {"transcript_novelty": 1}
    assert debug.get("source_gating", {}).get("vector") == "enabled"
