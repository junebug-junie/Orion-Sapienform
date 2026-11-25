"""Prime reads the attention schema once per thought (choice-as-EFE Step 2a, S2.1 and S2.3's frame half)."""
from __future__ import annotations

import asyncio
from datetime import timedelta
from types import SimpleNamespace

from orion.prime.attention_line import HEADER
from test_prime_loop import MOVE, T0, runtime

LIVE_MODEL = {
    "attention_reason": "bottom_up_salience", "confidence": 0.6,
    "confidence_basis": "broadcast.coalition_stability_score (fresh, bottom-up)",
    "prediction_error_confidence": None,
    "broadcast_selected_description": "Orion is reading (2 accepted links in the last 15 min)",
    "broadcast_selected_open_loop_id": "open-loop-f24f69eaa370",
}


def _rt(*, on=True, dsn="postgresql://x/y", read=None):
    rt = runtime(answer=MOVE)
    rt.s = SimpleNamespace(**{**vars(rt.s), "PRIME_ATTENTION_SCHEMA_LINE_ENABLED": on,
                              "PRIME_ATTENTION_SCHEMA_PG_DSN": dsn})
    reads = []

    def default_read():
        reads.append(1)
        return {"generated_at": T0 - timedelta(minutes=2), "self_model_json": LIVE_MODEL}

    rt.read_attention_schema_row = read or default_read
    rt.schema_reads = reads
    return rt


def _prompt(rt) -> str:
    [call] = rt.llm.calls
    return call["user_prompt"]


def test_the_frame_shows_the_schema_line_from_the_latest_row():
    rt = _rt()
    asyncio.run(rt.tick())
    prompt = _prompt(rt)
    assert f'{HEADER}\nread 2 min ago: most salient: "Orion is reading' in prompt
    assert "bottom-up: salience alone chose it" in prompt and "confidence 0.60" in prompt
    assert rt.schema_reads == [1]  # once per thought


def test_a_stale_row_shows_absent():
    rt = _rt(read=lambda: {"generated_at": T0 - timedelta(minutes=11), "self_model_json": LIVE_MODEL})
    asyncio.run(rt.tick())
    assert f"{HEADER}\n(absent: the latest schema reading is 11 min old" in _prompt(rt)


def test_a_failed_read_shows_absent_and_the_thought_goes_on():
    def boom():
        raise ConnectionError("db down")

    rt = _rt(read=boom)
    asyncio.run(rt.tick())
    assert f"{HEADER}\n(absent: the schema could not be read: ConnectionError: db down)" in _prompt(rt)
    assert any(a["verb"] == "infer" for _, a in rt.bus.acts())


def test_an_empty_dsn_says_so_without_reading():
    rt = _rt(dsn="")
    asyncio.run(rt.tick())
    assert "PRIME_ATTENTION_SCHEMA_PG_DSN is empty" in _prompt(rt) and rt.schema_reads == []


def test_rollback_flag_off_removes_the_section_and_the_read():
    rt = _rt(on=False)
    asyncio.run(rt.tick())
    assert HEADER not in _prompt(rt) and rt.schema_reads == []


def test_the_shipped_default_is_on():
    from app.settings import Settings

    field = Settings.model_fields["PRIME_ATTENTION_SCHEMA_LINE_ENABLED"]
    assert field.default is True
