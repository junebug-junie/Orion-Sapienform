#!/usr/bin/env python3
"""Index + unique rule on Prior.prior_id in Orion's worldview graph (#3395, Patch F).

Three subcommands. Only `scratch` writes, and only to a throwaway FalkorDB
container it starts itself from the SAME image the live server runs.

  scratch    Start a scratch FalkorDB, then (1) replay one taught write batch twice
             and check nothing doubled, (2) add the index + rule over a graph that
             holds a Prior with no prior_id, and check a duplicate CREATE fails.
  preflight  Read-only checks on a live graph: duplicate prior ids, id-less priors,
             existing constraints. Exit 1 when a duplicate would make the rule FAIL.
  commands   Print the exact operator commands to apply (and roll back) the rule.

This script never applies the rule to a live graph. Juniper approves that write.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
import uuid

LIVE_CONTAINER = "orion-athena-falkordb"
LABEL = "Prior"
PROP = "prior_id"
INDEX_CYPHER = f"CREATE INDEX FOR (p:{LABEL}) ON (p.{PROP})"


def constraint_args(graph: str, verb: str = "CREATE") -> list[str]:
    return ["GRAPH.CONSTRAINT", verb, graph, "UNIQUE", "NODE", LABEL, "PROPERTIES", "1", PROP]


def apply_commands(graph: str = "orion_worldview", container: str = LIVE_CONTAINER) -> list[str]:
    cli = f"docker exec {container} redis-cli"
    return [
        f'{cli} GRAPH.QUERY {graph} "{INDEX_CYPHER}"',
        f"{cli} {' '.join(constraint_args(graph))}",
        f'{cli} GRAPH.RO_QUERY {graph} "CALL db.constraints()"',
    ]


def rollback_commands(graph: str = "orion_worldview", container: str = LIVE_CONTAINER) -> list[str]:
    return [f"docker exec {container} redis-cli {' '.join(constraint_args(graph, 'DROP'))}"]


# One taught batch (orion/curiosity/kickoff_prompt.py), RUN_ID substituted: a new
# prior, a test of an old prior (one SET that updates and counts once per run), two different
# revisions of one prior in the same run, a finding and its edge.
def taught_batch(run_id: str) -> list[str]:
    return [
        f'MERGE (p:Prior {{prior_id: "new-claim"}}) ON CREATE SET p.claim = "c", p.confidence = 0.55, '
        f'p.status = "open", p.times_tested = 0, p.run_id = "{run_id}"',
        f'MATCH (p:Prior {{prior_id: "old-claim"}}) SET p.times_tested = coalesce(p.times_tested, 0) + '
        f'CASE WHEN coalesce(p.last_run_id, "") <> "{run_id}" THEN 1 ELSE 0 END, '
        f'p.status = "revised", p.confidence = 0.75, p.last_tested_at = "t", p.last_run_id = "{run_id}"',
        f'MERGE (v:PriorRevision {{prior_id: "old-claim", run_id: "{run_id}", from_confidence: 0.92, '
        'to_confidence: 0.9, from_status: "open", to_status: "revised"}) ON CREATE SET v.written_at = timestamp()',
        f'MERGE (v:PriorRevision {{prior_id: "old-claim", run_id: "{run_id}", from_confidence: 0.9, '
        'to_confidence: 0.75, from_status: "revised", to_status: "revised"}) ON CREATE SET v.written_at = timestamp()',
        f'MERGE (f:Finding {{finding_id: "f-1"}}) ON CREATE SET f.text = "t", f.evidence = "e", f.run_id = "{run_id}"',
        f'MATCH (f:Finding {{finding_id: "f-1"}}), (p:Prior {{prior_id: "old-claim"}}) '
        f'MERGE (f)-[e:SUPPORTS]->(p) ON CREATE SET e.run_id = "{run_id}"',
    ]


class Cli:
    def __init__(self, container: str):
        self.container = container

    def run(self, *args: str, check: bool = False) -> tuple[int, str]:
        p = subprocess.run(["docker", "exec", self.container, "redis-cli", *args],
                           capture_output=True, text=True, timeout=60)
        out = (p.stdout + p.stderr).strip()
        if check and (p.returncode != 0 or "error" in out.lower() or "violation" in out.lower()):
            raise RuntimeError(f"{args!r} -> {out}")
        return p.returncode, out

    def scalar_rows(self, graph: str, cypher: str) -> list[str]:
        """Header then values, for a ONE-column query (redis-cli --csv prints one flat line)."""
        _, out = self.run("--csv", "GRAPH.RO_QUERY", graph, cypher)
        cells = next(csv.reader([out.replace("\n", ",")]), [])
        return [c for c in cells if c and not c.startswith(("Cached execution", "Query internal"))]


def counts(cli: Cli, graph: str) -> str:
    q = ("MATCH (n) WITH labels(n)[0] AS l, count(n) AS c "
         "OPTIONAL MATCH ()-[r]->() WITH l, c, count(r) AS e "
         "RETURN l + ':' + toString(c) + '/edges:' + toString(e) ORDER BY 1")
    tt = cli.scalar_rows(graph, 'MATCH (p:Prior {prior_id: "old-claim"}) RETURN toString(p.times_tested)')
    return " ".join(cli.scalar_rows(graph, q)[1:]) + f" old-claim.times_tested={tt[1:]}"


def scratch(image: str | None) -> int:
    image = image or subprocess.run(["docker", "inspect", "-f", "{{.Image}}", LIVE_CONTAINER],
                                    capture_output=True, text=True, check=True).stdout.strip()
    name = f"orion-patchf-scratch-{uuid.uuid4().hex[:8]}"
    subprocess.run(["docker", "run", "-d", "--rm", "--name", name, "--label", "orion.model_replay=1", image],
                   check=True, capture_output=True, text=True, timeout=120)
    cli, ok = Cli(name), True
    try:
        for _ in range(60):
            if cli.run("PING")[1] == "PONG":
                break
            time.sleep(0.5)
        print("module:", cli.run("MODULE", "LIST")[1].replace("\n", " ")[:120])
        # 1. Replay the taught batch twice.
        g = "replay"
        cli.run("GRAPH.QUERY", g, 'CREATE (:Prior {prior_id: "old-claim", confidence: 0.92, status: "open", '
                'times_tested: 2})', check=True)  # no last_run_id: the coalesce case
        seen, finding_created = [], []
        for _ in range(2):
            for cmd in taught_batch("aaaaaaaaaaaa"):
                out = cli.run("GRAPH.QUERY", g, cmd, check=True)[1]
                if cmd.startswith("MERGE (f:Finding"):
                    finding_created.append("Nodes created: 1" in out)
            seen.append(counts(cli, g))
            print("after batch:", seen[-1])
        # Same graph after the replay; the id-less-last_run_id prior moved 2 -> 3 once.
        ok &= seen[0] == seen[1] and "times_tested=['3']" in seen[1] and "PriorRevision:2" in seen[1]
        # Created vs matched, not just "no duplicate": batch 1 creates the finding, batch 2 matches it.
        print("finding MERGE created per batch:", finding_created)
        ok &= finding_created == [True, False]
        # The dangerous fold: a NEW finding sent under an existing id creates nothing and keeps the old text.
        out = cli.run("GRAPH.QUERY", g, 'MERGE (f:Finding {finding_id: "f-1"}) ON CREATE SET f.text = "new claim"')[1]
        kept = cli.scalar_rows(g, 'MATCH (f:Finding {finding_id: "f-1"}) RETURN f.text')[1:]
        print(f"new finding under an existing id: created={'Nodes created' in out}, text now={kept}")
        ok &= "Nodes created" not in out and kept == ["t"]
        # 2. Unique rule over a graph holding an id-less Prior.
        g = "rule"
        cli.run("GRAPH.QUERY", g, 'CREATE (:Prior {prior_id: "a"}), (:Prior {name: "no-id"}), (:Prior {prior_id: "b"})',
                check=True)
        print("index:", cli.run("GRAPH.QUERY", g, INDEX_CYPHER)[1].splitlines()[0])
        print("constraint create:", cli.run(*constraint_args(g))[1])
        status = ""
        for _ in range(20):
            rows = cli.scalar_rows(g, "CALL db.constraints() YIELD status RETURN status")
            status = rows[-1] if len(rows) > 1 else ""
            if status != "PENDING":
                break
            time.sleep(0.25)
        print("constraint status:", status)
        ok &= status == "OPERATIONAL"
        dup = cli.run("GRAPH.QUERY", g, 'CREATE (:Prior {prior_id: "a"})')[1]
        print("duplicate CREATE:", dup.splitlines()[0])
        ok &= "unique constraint violation" in dup
        idless = cli.run("GRAPH.QUERY", g, 'CREATE (:Prior {name: "no-id-2"})')[1]
        print("second id-less CREATE:", idless.splitlines()[1] if len(idless.splitlines()) > 1 else idless)
        ok &= "Nodes created: 1" in idless
        merge = cli.run("GRAPH.QUERY", g, 'MERGE (p:Prior {prior_id: "a"}) ON CREATE SET p.claim = "x"')[1]
        ok &= "Nodes created" not in merge
        print("MERGE on existing id: no node created" if "Nodes created" not in merge else f"MERGE: {merge}")
        # 3. Hazards the prompt now teaches around (Orion's Q9, Q10).
        for _ in range(2):
            cli.run("GRAPH.QUERY", g, 'MERGE (f:Finding {finding_id: ""}) ON CREATE SET f.text = "t"')
        fused = cli.scalar_rows(g, 'MATCH (f:Finding {finding_id: ""}) RETURN toString(count(f))')[1:]
        print("empty finding_id MERGE x2 -> nodes:", fused, "(one node: empty ids fuse)")
        ok &= fused == ["1"]
        null = cli.run("GRAPH.QUERY", g, 'MERGE (f:Finding {finding_id: null}) ON CREATE SET f.text = "t"')[1]
        print("null finding_id MERGE:", null.splitlines()[0])
        ok &= "null" in null.lower()
        rc, part = cli.run("GRAPH.QUERY", g, 'CREATE (:Marker {m: 1}) CREATE (:Prior {prior_id: "a"})')
        markers = cli.scalar_rows(g, "MATCH (m:Marker) RETURN toString(count(m))")[1:]
        print(f"one query, write then violation: redis-cli rc={rc}, earlier write kept={markers}")
        ok &= markers == ["0"]
        print("final priors:", cli.scalar_rows(g, "MATCH (p:Prior) RETURN coalesce(p.prior_id, '(none:' + p.name + ')') "
                                                  "ORDER BY 1")[1:])
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, text=True, timeout=60)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def preflight(container: str, graph: str) -> int:
    cli = Cli(container)
    dups = cli.scalar_rows(graph, f"MATCH (p:{LABEL}) WHERE p.{PROP} IS NOT NULL WITH p.{PROP} AS id, count(*) AS c "
                                  "WHERE c > 1 RETURN id")[1:]
    idless = cli.scalar_rows(graph, f"MATCH (p:{LABEL}) WHERE p.{PROP} IS NULL RETURN toString(id(p))")[1:]
    cons = cli.scalar_rows(graph, "CALL db.constraints() YIELD label, properties, status "
                                  "RETURN label + ' ' + properties[0] + ' ' + status")[1:]
    print(f"duplicate {PROP}s: {len(dups)} {dups[:10]}")
    print(f"{LABEL} nodes with no {PROP} (not held to the rule): {len(idless)} node ids {idless[:10]}")
    print(f"constraints: {cons or 'none'}")
    stuck = [c for c in cons if not c.endswith("OPERATIONAL")]
    if stuck:
        print(f"NOT OPERATIONAL (UNDER CONSTRUCTION or FAILED): {stuck}")
    return 1 if dups or stuck else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scratch")
    s.add_argument("--image", help="FalkorDB image (default: the live container's image id)")
    p = sub.add_parser("preflight")
    p.add_argument("--container", default=LIVE_CONTAINER)
    p.add_argument("--graph", default="orion_worldview")
    c = sub.add_parser("commands")
    c.add_argument("--graph", default="orion_worldview")
    args = ap.parse_args(argv)
    if args.cmd == "scratch":
        return scratch(args.image)
    if args.cmd == "preflight":
        return preflight(args.container, args.graph)
    print("# apply (operator, after Juniper approves; run preflight first):")
    print("\n".join(apply_commands(args.graph)))
    print("# roll back:")
    print("\n".join(rollback_commands(args.graph)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
