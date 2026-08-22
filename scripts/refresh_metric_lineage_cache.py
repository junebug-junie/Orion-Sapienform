#!/usr/bin/env python3
"""Refresh the metric semantic layer's cache for `scripts/hooks/metric_lineage_nudge.py`.

Phase 3 of docs/superpowers/specs/2026-08-12-metric-semantic-layer-design.md.

`orion.metrics.consumers.scan_repo()` walks ~3900 files and takes ~13s
(measured in the design doc) -- too slow to run synchronously inside a
PreToolUse hook that fires on every Edit/Write. This script does that work
once and writes the joined graph (`orion.metrics.lineage.build_graph()`) plus
the consumer scan (`orion.metrics.consumers.scan_repo()`) to a plain JSON
cache the hook can read in milliseconds. No new computation: both calls are
the exact same ones `scripts/check_metric_lineage.py` already makes -- this
just persists their output instead of discarding it after one CLI run.

Written atomically (temp file + os.replace) so a hook reading the cache
mid-refresh never sees a half-written file.

Run manually (`make check-metric-lineage-cache-refresh`), or invoked in the
background, non-blocking, by the nudge hook itself when the cache is stale --
see that script's `_maybe_refresh_in_background()`.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] == _SCRIPT_DIR:
    # scripts/ on sys.path[0] shadows stdlib `platform` via scripts/platform/
    # and breaks pydantic -- same fix as check_metric_lineage.py.
    sys.path.pop(0)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CACHE_PATH = REPO_ROOT / ".cache" / "metric_lineage.json"


def refresh() -> Path:
    from orion.metrics.consumers import scan_repo
    from orion.metrics.lineage import build_graph, to_dict

    graph = build_graph()
    scan = scan_repo(graph.scan_tokens().keys())

    payload = {
        "generated_at": time.time(),
        "nodes": [to_dict(n) for n in graph.nodes.values()],
        "hits": [h.to_dict() for h in scan.hits],
        "files_scanned": scan.files_scanned,
    }

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = CACHE_PATH.with_suffix(f".tmp.{os.getpid()}")
    tmp_path.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp_path, CACHE_PATH)  # atomic on the same filesystem
    return CACHE_PATH


def main() -> int:
    path = refresh()
    print(f"metric lineage cache refreshed: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
