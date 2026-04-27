from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from .anchors import extract_anchors
from .blocks import extract_blocks
from .claims import mine_claims_and_resolutions
from .episodes import segment_episodes
from .graph import build_turn_graph
from .indexer import build_chat_turn_index
from .io_utils import write_json, write_jsonl
from .renderer import render_pageindex_markdown
from .repository import fetch_chat_turn_rows
from .types import PipelineManifest, stable_iso

PIPELINE_VERSION = "chat-corpus-builder-v1"


def run_pipeline(
    *,
    output_dir: Path,
    markdown_path: Path,
    start_at: datetime | None = None,
    end_at: datetime | None = None,
    max_rows: int = 10000,
    include_reasoning: bool = False,
) -> dict[str, str]:
    end_dt = end_at or datetime.now(timezone.utc)
    start_dt = start_at or (end_dt - timedelta(days=1))
    run_id = str(uuid4())
    stage_stats: dict[str, dict[str, int | float]] = {}

    t0 = perf_counter()
    rows = fetch_chat_turn_rows(start_at=start_dt, end_at=end_dt, limit=max_rows)
    stage_stats["fetch_rows"] = {"count": len(rows), "duration_ms": _ms(t0)}

    t0 = perf_counter()
    turns = build_chat_turn_index(rows)
    stage_stats["chat_turn_index"] = {"count": len(turns), "duration_ms": _ms(t0)}

    t0 = perf_counter()
    anchors = extract_anchors(turns)
    stage_stats["anchors"] = {"count": len(anchors), "duration_ms": _ms(t0)}

    t0 = perf_counter()
    edges = build_turn_graph(turns, anchors)
    stage_stats["turn_graph"] = {"count": len(edges), "duration_ms": _ms(t0)}

    t0 = perf_counter()
    episodes = segment_episodes(turns, anchors, edges)
    stage_stats["episodes"] = {"count": len(episodes), "duration_ms": _ms(t0)}

    t0 = perf_counter()
    blocks = extract_blocks(turns, include_reasoning=include_reasoning)
    stage_stats["blocks"] = {"count": len(blocks), "duration_ms": _ms(t0)}

    t0 = perf_counter()
    claims = mine_claims_and_resolutions(episodes, blocks)
    stage_stats["claims"] = {"count": len(claims), "duration_ms": _ms(t0)}

    t0 = perf_counter()
    markdown = render_pageindex_markdown(episodes=episodes, blocks=blocks, claims=claims)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown, encoding="utf-8")
    stage_stats["render"] = {"count": len(markdown), "duration_ms": _ms(t0)}

    artifacts = {
        "chat_turn_index": output_dir / "chat_turn_index.jsonl",
        "anchors": output_dir / "anchors.jsonl",
        "turn_graph": output_dir / "turn_graph.jsonl",
        "episodes": output_dir / "episodes.jsonl",
        "claims": output_dir / "claims.jsonl",
        "blocks": output_dir / "blocks.jsonl",
        "manifest": output_dir / "run_manifest.json",
        "markdown": markdown_path,
    }
    write_jsonl(artifacts["chat_turn_index"], turns)
    write_jsonl(artifacts["anchors"], anchors)
    write_jsonl(artifacts["turn_graph"], edges)
    write_jsonl(artifacts["episodes"], episodes)
    write_jsonl(artifacts["claims"], claims)
    write_jsonl(artifacts["blocks"], blocks)

    manifest = PipelineManifest(
        pipeline_version=PIPELINE_VERSION,
        run_id=run_id,
        date_window_start=stable_iso(start_dt),
        date_window_end=stable_iso(end_dt),
        stage_stats=stage_stats,
        generated_at=stable_iso(datetime.now(timezone.utc)),
    )
    write_json(artifacts["manifest"], asdict(manifest))
    return {key: str(value) for key, value in artifacts.items()}


def _ms(start: float) -> float:
    return round((perf_counter() - start) * 1000, 2)
