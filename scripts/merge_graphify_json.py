#!/usr/bin/env python3
"""Three-way union for Orion's large Graphify node-link JSON.

Graphify 0.9.15's CLI merger caps inputs at 50 MiB and drops top-level
metadata. Keep node/edge/hyperedge identities from both parents, retain
one-sided attribute edits, and fail on competing non-derived scalar edits.
Community labels are derived; callers regenerate the report after merging.
The existing shell driver owns LFS resolution and re-cleaning.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

MISSING = object()
DERIVED = {"community", "community_name", "built_at_commit"}


def merge_value(base, ours, theirs, path="root"):
    if ours is MISSING:
        return theirs
    if theirs is MISSING:
        return ours
    if isinstance(ours, dict) and isinstance(theirs, dict):
        previous = base if isinstance(base, dict) else {}
        return {key: merge_value(previous.get(key, MISSING), ours.get(key, MISSING),
                                 theirs.get(key, MISSING), f"{path}.{key}")
                for key in sorted(ours.keys() | theirs.keys())}
    if isinstance(ours, list) and isinstance(theirs, list):
        if path.endswith(".hyperedges"):
            return merge_records(base if isinstance(base, list) else [], ours, theirs,
                                 lambda row: row["id"], path)
        unique = {json.dumps(value, sort_keys=True): value for value in ours + theirs}
        return list(unique.values())
    if ours == theirs or theirs == base:
        return ours
    if ours == base:
        return theirs
    if path.rsplit(".", 1)[-1] in DERIVED:
        return ours
    raise ValueError(f"competing edits at {path}")


def merge_records(base, ours, theirs, identity, path):
    def index(rows):
        out = {}
        for row in rows:
            key = identity(row)
            if key in out:
                raise ValueError(f"duplicate identity in {path}: {key}")
            out[key] = row
        return out
    previous, left, right = map(index, (base, ours, theirs))
    return [merge_value(previous.get(key, MISSING), left.get(key, MISSING),
                        right.get(key, MISSING), f"{path}[{key}]")
            for key in dict.fromkeys((*left, *right))]


def validate(data):
    if not isinstance(data, dict):
        raise ValueError("graph must be an object")
    if not all(isinstance(data.get(key), bool) for key in ("directed", "multigraph")):
        raise ValueError("directed and multigraph must be booleans")
    if not all(isinstance(data.get(key), list) for key in ("nodes", "links")):
        raise ValueError("nodes and links must be lists")
    ids = [row["id"] for row in data["nodes"]]
    if not all(isinstance(key, str) for key in ids) or len(ids) != len(set(ids)):
        raise ValueError("node IDs must be unique strings")
    members = set(ids)
    for edge in data["links"]:
        if edge["source"] not in members or edge["target"] not in members:
            raise ValueError("edge endpoint is absent from nodes")
        if data["multigraph"] and "key" not in edge:
            raise ValueError("multigraph edges require keys")
    for container in (data, data.get("graph", {})):
        if not isinstance(container, dict):
            raise ValueError("graph metadata must be an object")
        hyperedges = container.get("hyperedges", [])
        if not isinstance(hyperedges, list):
            raise ValueError("hyperedges must be a list")
        if len({row["id"] for row in hyperedges}) != len(hyperedges):
            raise ValueError("hyperedge IDs must be unique")
        for row in hyperedges:
            if not isinstance(row.get("id"), str) or not isinstance(row.get("nodes"), list):
                raise ValueError("hyperedges require string IDs and node lists")
            if not all(node in members for node in row["nodes"]):
                raise ValueError("hyperedge member is absent from nodes")


def merge_graphs(base, ours, theirs):
    normalized = []
    for data in (base, ours, theirs):
        validate(data)
        data = dict(data)
        metadata = dict(data.get("graph", {}))
        hyperedges = merge_records([], data.get("hyperedges", []), metadata.get("hyperedges", []),
                                   lambda row: row["id"], "hyperedges")
        data["hyperedges"] = metadata["hyperedges"] = hyperedges
        data["graph"] = metadata
        normalized.append(data)
    base, ours, theirs = normalized
    for key in ("directed", "multigraph"):
        if ours[key] != theirs[key]:
            raise ValueError(f"incompatible {key} flags")

    def edge_id(row):
        ends = (row["source"], row["target"])
        if not ours["directed"]:
            ends = tuple(sorted(ends))
        return ends + ((row["key"],) if ours["multigraph"] else ())

    # Endpoint order carries no meaning for undirected edges. Canonicalize
    # before merging attributes so a reversed serialization is not a conflict.
    if not ours["directed"]:
        for data in (base, ours, theirs):
            data["links"] = [dict(row, source=min(row["source"], row["target"]),
                                  target=max(row["source"], row["target"])) for row in data["links"]]
    metadata = lambda data: {key: value for key, value in data.items() if key not in {"nodes", "links"}}
    result = merge_value(metadata(base), metadata(ours), metadata(theirs))
    result["nodes"] = merge_records(base["nodes"], ours["nodes"], theirs["nodes"], lambda row: row["id"], "nodes")
    result["links"] = merge_records(base["links"], ours["links"], theirs["links"], edge_id, "links")
    validate(result)
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 3:
        raise ValueError("usage: merge_graphify_json.py BASE CURRENT OTHER")
    inputs = []
    for name in args:
        path = Path(name)
        inputs.append(json.loads(path.read_text()))
    result = merge_graphs(*inputs)
    destination = Path(args[1])
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=destination.parent, delete=False) as handle:
            temp = Path(handle.name)
            json.dump(result, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.chmod(temp, destination.stat().st_mode & 0o777)
        os.replace(temp, destination)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)
    print(f"graph union: {len(result['nodes'])} nodes, {len(result['links'])} links", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(f"graph merge refused: {exc}", file=sys.stderr)
        sys.exit(1)
