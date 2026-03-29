from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_RECALL_ROOT = _REPO / "services" / "orion-recall"
if str(_RECALL_ROOT) not in sys.path:
    sys.path.insert(0, str(_RECALL_ROOT))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from app.sql_timeline import _filter_excluded_rows


def test_filter_excluded_rows_removes_id_and_text_echoes():
    rows = [
        {"id": "corr-1", "text": "User: Teddy loves Addy\nOrion: ..."},
        {"id": "row-2", "text": "User: unrelated\nOrion: ..."},
    ]

    out = _filter_excluded_rows(
        rows,
        exclude_ids=["corr-1"],
        exclude_text="Teddy loves Addy",
    )

    assert len(out) == 1
    assert out[0]["id"] == "row-2"
