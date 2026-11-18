#!/usr/bin/env python3
"""Copy the retired Self Atlas FalkorDB graph to an archive graph. Never deletes.

Self Atlas was retired in 2026-11 (spec:
docs/superpowers/specs/2026-10-30-self-atlas-replacement-proposal.md, Patch B). Its graph
``orion_substrate_self`` (about 22k nodes) has no reader and no writer after that deploy.
Juniper has not approved deleting it. This script only makes a copy, so a later delete
loses nothing.

Default is a dry run: it reads the node and edge counts and says what it would do.
``--apply`` runs ``GRAPH.COPY <source> <archive>``, then checks that the archive has the
same node and edge counts as the source. It refuses when the archive name already exists,
and it never runs ``GRAPH.DELETE``. It prints the delete command for Juniper to run by hand.

Run it after Patch B is deployed (so nothing writes the source during the copy):

    .venv/bin/python scripts/archive_self_atlas_graph.py            # dry run
    .venv/bin/python scripts/archive_self_atlas_graph.py --apply    # copy + verify

Connection: ``FALKORDB_URI`` (default ``redis://127.0.0.1:6380``, the host-mapped port of
orion-athena-falkordb).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Any, Callable

SOURCE_GRAPH = "orion_substrate_self"
DEFAULT_ARCHIVE_GRAPH = "orion_substrate_self_archive_2026_11"
DEFAULT_URI = "redis://127.0.0.1:6380"

NODE_COUNT_QUERY = "MATCH (n) RETURN count(n)"
EDGE_COUNT_QUERY = "MATCH ()-[r]->() RETURN count(r)"

Execute = Callable[..., Any]


def _scalar(reply: Any) -> int:
    """``GRAPH.RO_QUERY`` reply is ``[header, rows, stats]``; take rows[0][0]."""
    try:
        return int(reply[1][0][0])
    except (IndexError, TypeError, ValueError) as exc:
        raise RuntimeError(f"unexpected GRAPH.RO_QUERY reply: {reply!r}") from exc


def graph_exists(execute: Execute, graph: str) -> bool:
    return bool(int(execute("EXISTS", graph)))


def counts(execute: Execute, graph: str) -> tuple[int, int]:
    nodes = _scalar(execute("GRAPH.RO_QUERY", graph, NODE_COUNT_QUERY))
    edges = _scalar(execute("GRAPH.RO_QUERY", graph, EDGE_COUNT_QUERY))
    return nodes, edges


def mask_uri(uri: str) -> str:
    """Hide any user:password in the URI before printing it."""
    return re.sub(r"//[^@/]*@", "//***@", uri)


def delete_command(uri: str, graph: str = SOURCE_GRAPH) -> str:
    return f"redis-cli -u {mask_uri(uri)} GRAPH.DELETE {graph}"


def archive(execute: Execute, *, archive_graph: str, apply: bool, out=print) -> int:
    """Return a process exit code. Never deletes anything."""
    if archive_graph == SOURCE_GRAPH:
        out("REFUSED: the archive name is the source name.")
        return 2
    if not graph_exists(execute, SOURCE_GRAPH):
        out(f"REFUSED: source graph {SOURCE_GRAPH} does not exist (already deleted?).")
        return 2
    if graph_exists(execute, archive_graph):
        out(f"REFUSED: archive graph {archive_graph} already exists. Nothing copied. "
            "Pick another --archive name, or check that archive by hand.")
        return 2
    src_nodes, src_edges = counts(execute, SOURCE_GRAPH)
    out(f"source  {SOURCE_GRAPH}: {src_nodes} nodes, {src_edges} edges")
    if not apply:
        out(f"DRY RUN: would run GRAPH.COPY {SOURCE_GRAPH} {archive_graph}, then compare counts.")
        out("Re-run with --apply to copy.")
        return 0
    try:
        execute("GRAPH.COPY", SOURCE_GRAPH, archive_graph)
        dst_nodes, dst_edges = counts(execute, archive_graph)
        # Re-read the source: if something wrote it during the copy, the counts drift.
        src_nodes_after, src_edges_after = counts(execute, SOURCE_GRAPH)
    except Exception as exc:  # noqa: BLE001 -- report, never a raw traceback
        out(f"COPY FAILED: {exc}; source {SOURCE_GRAPH} untouched.")
        return 1
    out(f"archive {archive_graph}: {dst_nodes} nodes, {dst_edges} edges")
    if (dst_nodes, dst_edges) != (src_nodes_after, src_edges_after):
        out("MISMATCH: archive counts differ from the source. The source was NOT deleted. "
            "Is Hub still writing it (Patch B not deployed yet)? Check before any delete.")
        return 1
    out("VERIFIED: archive matches the source. The source was NOT deleted.")
    out("Note: the body atlas lists every FalkorDB graph, so the archive shows there as an unlinked store. That is expected.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="run GRAPH.COPY (default: dry run)")
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE_GRAPH, help="archive graph name")
    parser.add_argument("--uri", default=os.environ.get("FALKORDB_URI") or DEFAULT_URI)
    args = parser.parse_args(argv)

    import redis  # deferred: tests inject a fake executor

    client = redis.Redis.from_url(args.uri, decode_responses=True)
    code = archive(client.execute_command, archive_graph=args.archive, apply=args.apply)
    print()
    print("This script never deletes. After Juniper approves, the delete is (run by hand):")
    print(f"  {delete_command(args.uri)}")
    return code


if __name__ == "__main__":
    sys.exit(main())
