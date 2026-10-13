"""The letter's "What I read" section: the piece, Orion's take, what changed -- or one line."""
from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timedelta, timezone

from orion.orion_day.gather import gather_orion_day
from orion.orion_day.tests import fixtures as fx
from orion.schemas.orion_day import OrionDayLetterSourcesV1, OrionDayLetterV1

AFTER_SLOT = datetime(2026, 9, 30, 15, 0, tzinfo=timezone.utc)
ARTICLE = ("Hugging Face transformers can now load GGUF checkpoints directly. "
           "It dequantizes them to fp16 or fp32 on load. Fine-tuning then runs in plain PyTorch.")


def _letter(conn):
    material = asyncio.run(gather_orion_day(conn, fx.LETTER_DATE, now=AFTER_SLOT))
    return OrionDayLetterV1(
        letter_date=fx.LETTER_DATE, run_id="orion-day-2026-09-29-1",
        window_start=material.window_start, window_end=material.window_end,
        note_md="NOTE", carry_forward_md="1. CARRY", material=material,
        sources=OrionDayLetterSourcesV1(by_source=material.sources),
        created_at=AFTER_SLOT, carry_forward_expires_at=AFTER_SLOT + timedelta(hours=36),
    )


def _section(html: str, name: str) -> str:
    m = re.search(rf'data-section="{name}">(.*?)</td>', html, flags=re.S)
    assert m, name
    return m.group(1)


def test_what_i_read_shows_the_piece_the_take_and_the_link(monkeypatch):
    from scripts import orion_day_email as email

    row = {**fx.READING_ROWS[0], "stage2_result_json": json.dumps({
        "summary": "GGUF quants load through transformers now. That changes my serving options. More.",
        "article_summary": ARTICLE})}
    monkeypatch.setattr(fx, "READING_ROWS", [row])
    letter = _letter(fx.FakeConn())
    html = email.render_html(letter, [])
    section = _section(html, "what-i-read")
    assert ARTICLE in section
    assert "GGUF quants load through transformers now. That changes my serving options." in section
    assert "Link made: GGUF quantization is now loadable by the transformers library." in section
    assert "transformers can now load GGUF quantized checkpoints directly" in section  # the quote
    # Above the full material, below the carry-forward.
    assert html.index('data-section="carry-forward"') < html.index('data-section="what-i-read"') < html.index("What the day held")
    text = email.render_text(letter, [])
    assert "## What I read" in text and f"What the piece says: {ARTICLE}" in text
    assert "  Link made: GGUF quantization" in text


def test_a_read_that_changed_nothing_says_so_and_an_old_read_names_the_missing_summary():
    from scripts import orion_day_email as email

    letter = _letter(fx.FakeConn(read_links=[]))
    section = _section(email.render_html(letter, []), "what-i-read")
    assert "(no summary of the piece was recorded)" in section
    assert "Changed: no link accepted, no belief moved." in section
    text = email.render_text(letter, [])
    assert "  Changed: no link accepted, no belief moved." in text
