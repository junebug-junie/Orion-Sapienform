#!/usr/bin/env python3
"""Seed graphify's local semantic-extraction cache from the committed graph.json.

graphify's semantic (LLM) extraction pass skips a document/paper/image file
only if it has a matching entry in the local, per-file content-hash cache
under graphify-out/cache/semantic/ (graphify.cache.check_semantic_cache).
That cache had never been committed in this repo (only graph.json,
GRAPH_REPORT.md, and manifest.json were tracked under graphify-out/), so a
fresh worktree or clone starts with an empty cache even when graph.json
already contains correctly-extracted nodes for those files. A full-repo
`/graphify .` run there would re-dispatch subagents to re-extract
already-covered files -- not destructive (build_merge()'s dedup-by-id keeps
the existing node) but wasteful of tokens and time.

Run this from repo root any time graph.json gains new document/paper/image
nodes (after a semantic extraction merge), then commit the updated
graphify-out/cache/semantic/ directory alongside graph.json. See CLAUDE.md's
`## graphify` section, which points every future agent at this script before
a full-repo semantic run.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GRAPH_PATH = REPO_ROOT / "graphify-out" / "graph.json"
SEMANTIC_FILE_TYPES = {"document", "paper", "image"}


def _find_graphify_python() -> str:
    marker = REPO_ROOT / "graphify-out" / ".graphify_python"
    if marker.is_file():
        candidate = marker.read_text(encoding="utf-8").strip()
        if candidate:
            return candidate
    result = subprocess.run(
        ["uv", "tool", "run", "--from", "graphifyy", "python", "-c",
         "import sys; print(sys.executable)"],
        capture_output=True, text=True, check=False,
    )
    candidate = result.stdout.strip()
    return candidate if candidate else sys.executable


def main() -> int:
    if not GRAPH_PATH.is_file():
        print(f"[seed-semantic-cache] ERROR: {GRAPH_PATH} not found", file=sys.stderr)
        return 1

    graphify_python = _find_graphify_python()
    probe = subprocess.run(
        [graphify_python, "-c", "import graphify.cache"],
        capture_output=True, text=True, check=False,
    )
    if probe.returncode != 0:
        print(f"[seed-semantic-cache] ERROR: could not import graphify.cache with "
              f"interpreter {graphify_python}: {probe.stderr}", file=sys.stderr)
        return 1

    data = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    nodes = data.get("nodes", [])
    edges = data.get("edges", data.get("links", []))
    hyperedges = data.get("hyperedges", [])

    eligible_sources = {
        n["source_file"] for n in nodes
        if n.get("file_type") in SEMANTIC_FILE_TYPES and n.get("source_file")
    }
    if not eligible_sources:
        print("[seed-semantic-cache] No document/paper/image nodes found in graph.json -- nothing to seed.")
        return 0

    def _keep(item: dict) -> bool:
        return item.get("source_file") in eligible_sources

    sem_nodes = [n for n in nodes if _keep(n)]
    sem_edges = [e for e in edges if _keep(e)]
    sem_hyperedges = [h for h in hyperedges if _keep(h)]

    payload_path = REPO_ROOT / "graphify-out" / ".graphify_seed_payload.json"
    payload_path.write_text(
        json.dumps({"nodes": sem_nodes, "edges": sem_edges, "hyperedges": sem_hyperedges}),
        encoding="utf-8",
    )

    seed_script = (
        "import json\n"
        "from pathlib import Path\n"
        "from graphify.cache import save_semantic_cache\n"
        f"payload = json.loads(Path({str(payload_path)!r}).read_text(encoding='utf-8'))\n"
        "saved = save_semantic_cache(\n"
        "    payload['nodes'], payload['edges'], payload['hyperedges'],\n"
        f"    root=Path({str(REPO_ROOT)!r}), merge_existing=True,\n"
        ")\n"
        "print(f'[seed-semantic-cache] wrote cache entries for {saved} files')\n"
    )
    result = subprocess.run([graphify_python, "-c", seed_script], cwd=str(REPO_ROOT))
    payload_path.unlink(missing_ok=True)
    if result.returncode != 0:
        print("[seed-semantic-cache] ERROR: seeding failed", file=sys.stderr)
        return 1

    print(f"[seed-semantic-cache] {len(eligible_sources)} source files eligible "
          f"({len(sem_nodes)} nodes, {len(sem_edges)} edges, {len(sem_hyperedges)} hyperedges)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
