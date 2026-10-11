"""Contract of the concept-relation Chroma backfill: idempotent skip of ids already
present, per-row failure never aborts the job, doc ids match the live projector,
and progress lines carry every section-14 field."""
from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "backfill_concept_relation_chroma.py"
spec = importlib.util.spec_from_file_location("backfill_concept_relation_chroma", MODULE_PATH)
bf = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bf  # dataclasses resolve their module via sys.modules
spec.loader.exec_module(bf)


def _row(cid: str):
    return SimpleNamespace(crystallization_id=cid)


def test_doc_id_matches_live_projector() -> None:
    from orion.memory.crystallization.projection_chroma import build_chroma_upsert
    from orion.memory.crystallization.schemas import MemoryCrystallizationV1

    fields = MemoryCrystallizationV1.model_fields
    assert "crystallization_id" in fields
    # build_chroma_upsert is the source of truth for the doc id shape.
    import inspect

    assert 'f"crys_{crystallization.crystallization_id}"' in inspect.getsource(build_chroma_upsert)
    assert bf.doc_id_for("abc") == "crys_abc"


def test_skips_existing_publishes_rest_and_survives_errors() -> None:
    calls: list[str] = []

    async def publish(row):
        calls.append(row.crystallization_id)
        if row.crystallization_id == "boom":
            raise RuntimeError("x")
        if row.crystallization_id == "noemb":
            return {"published": False, "reason": "no_embedding"}
        return {"published": True}

    lines: list[str] = []
    prog, results = asyncio.run(
        bf.run_backfill(
            [_row("a"), _row("b"), _row("boom"), _row("noemb")],
            existing_ids={"crys_a"},
            publish=publish,
            emit=lines.append,
            sleep_sec=0,
            report_every=1,
        )
    )
    assert calls == ["b", "boom", "noemb"]
    assert results == {
        "a": "skipped_existing",
        "b": "published",
        "boom": "error:RuntimeError",
        "noemb": "error:no_embedding",
    }
    assert (prog.processed, prog.published, prog.skipped_existing, prog.errors) == (4, 1, 1, 2)
    assert len(lines) == 4
    assert "100.0%" in lines[-1]


def test_force_republishes_existing() -> None:
    seen: list[str] = []

    async def publish(row):
        seen.append(row.crystallization_id)
        return {"published": True}

    asyncio.run(bf.run_backfill([_row("a")], existing_ids={"crys_a"}, publish=publish, emit=lambda _: None, sleep_sec=0, force=True))
    assert seen == ["a"]


def test_progress_line_has_section14_fields() -> None:
    p = bf.Progress(total=10, started=0.0)
    p.processed, p.errors = 5, 1
    p.anomalies.append("x:no_embedding")
    line = p.line(now=10.0)
    for needle in (bf.TITLE, "50.0%", "ETA 10s", "5/10", "0.50 rows/s", "errors=1", "anomalies=x:no_embedding"):
        assert needle in line


def test_before_after_rows() -> None:
    rows = bf.build_before_after_rows(["a", "b"], {"crys_a"}, {"crys_a", "crys_b"}, {"b": "published"})
    assert rows == [
        {"crystallization_id": "a", "doc_id": "crys_a", "in_chroma_before": "true", "in_chroma_after": "true", "result": "not_attempted"},
        {"crystallization_id": "b", "doc_id": "crys_b", "in_chroma_before": "false", "in_chroma_after": "true", "result": "published"},
    ]
