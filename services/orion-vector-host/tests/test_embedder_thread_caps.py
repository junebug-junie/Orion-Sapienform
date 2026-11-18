"""Regression: HF embeds must run one at a time with a capped thread count.

Live 2026-10-30: `asyncio.to_thread` let ~10 embeds run at once, each with its
own 48-thread OpenMP team (454 torch threads, 965% CPU, 4.6 s per embed).
"""

from __future__ import annotations

import asyncio
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.embedder import Embedder

SERVICE_ROOT = Path(__file__).resolve().parents[1]


def _hf_embedder_without_model() -> tuple[Embedder, dict[str, int]]:
    embedder = Embedder.__new__(Embedder)
    embedder._settings = type("S", (), {"VECTOR_HOST_EMBED_BACKEND": "hf"})()
    embedder._client = None
    embedder._hf_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hf-embed")
    stats = {"active": 0, "peak": 0}
    lock = threading.Lock()

    def fake_embed_hf(text: str):
        with lock:
            stats["active"] += 1
            stats["peak"] = max(stats["peak"], stats["active"])
        time.sleep(0.02)
        with lock:
            stats["active"] -= 1
        return [0.0], "m", 1

    embedder._embed_hf = fake_embed_hf
    return embedder, stats


def test_concurrent_hf_embeds_never_overlap() -> None:
    embedder, stats = _hf_embedder_without_model()

    async def run() -> None:
        await asyncio.gather(*(embedder.embed(f"t{i}") for i in range(8)))
        await embedder.close()

    asyncio.run(run())
    assert stats["peak"] == 1


def test_embed_uses_the_single_worker_executor_not_to_thread() -> None:
    source = (SERVICE_ROOT / "app" / "embedder.py").read_text()
    assert "asyncio.to_thread(self._embed_hf" not in source
    assert re.search(r"ThreadPoolExecutor\(max_workers=1", source)


def test_dockerfile_caps_torch_threads() -> None:
    dockerfile = (SERVICE_ROOT / "Dockerfile").read_text()
    match = re.search(r"^ENV OMP_NUM_THREADS=(\d+)$", dockerfile, re.MULTILINE)
    assert match and 1 <= int(match.group(1)) <= 16
