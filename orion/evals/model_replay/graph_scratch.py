"""A throwaway FalkorDB holding copies of Orion's graphs, so a replayed turn's graph writes land
somewhere real that is not production.

  * ``ProdGraphSource``: the ONLY object that talks to production FalkorDB. Its command allowlist
    is DUMP / PING / EXISTS -- enforced in code (``READ_COMMANDS``), tested. It is used once, at
    replay start, to copy the graphs.
  * ``ScratchServer``: a local FalkorDB container (same image as production, published on
    127.0.0.1 only), removed on exit.
  * ``ScratchGraphs``: one model's view. Graph names in the model's commands map to private keys
    (``<model>__<graph>``), restored fresh from the dump before every task, with production's
    curiosity ACL emulated (atlas: GRAPH.RO_QUERY only; own graph: read+write; others: NOPERM).
  * ``prior_snapshot`` / ``diff_snapshots``: what the turn actually changed in its own graph --
    the ground truth the write-claim check compares the write-up against.
"""

from __future__ import annotations

import subprocess
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

OWN_GRAPH = "orion_worldview"
ATLAS_GRAPH = "orion_substrate"
COPIED_GRAPHS = (OWN_GRAPH, ATLAS_GRAPH)
PROD_CONTAINER = "orion-athena-falkordb"


class ForbiddenCommand(RuntimeError):
    pass


class ProdGraphSource:
    """Read-only access to production FalkorDB. Anything but DUMP/PING/EXISTS raises before sending."""

    READ_COMMANDS = frozenset({"DUMP", "PING", "EXISTS"})

    def __init__(self, client: Any) -> None:
        self._client = client

    @classmethod
    def connect(cls, url: str) -> "ProdGraphSource":
        import redis

        return cls(redis.Redis.from_url(url, decode_responses=False, socket_timeout=60))

    def execute(self, *args: Any) -> Any:
        cmd = str(args[0]).upper() if args else ""
        if cmd not in self.READ_COMMANDS:
            raise ForbiddenCommand(f"production FalkorDB: {cmd} is not a read command this replay may send")
        return self._client.execute_command(*args)

    def dump(self, key: str) -> bytes:
        payload = self.execute("DUMP", key)
        if payload is None:
            raise KeyError(f"graph {key} not found in production FalkorDB")
        return payload


def prod_falkordb_image(container: str = PROD_CONTAINER) -> str:
    """The exact image id production runs, so RESTORE accepts its DUMP payloads."""
    return subprocess.run(["docker", "inspect", container, "--format", "{{.Image}}"], check=True,
                          capture_output=True, text=True, timeout=30).stdout.strip()


@dataclass
class ScratchServer:
    image: str
    name: str = field(default_factory=lambda: f"orion-replay-falkordb-{uuid.uuid4().hex[:8]}")
    port: Optional[int] = None
    client: Any = None

    def start(self) -> None:
        subprocess.run(["docker", "run", "-d", "--rm", "--name", self.name, "--label", "orion.model_replay=1",
                        "-p", "127.0.0.1::6379", self.image], check=True, capture_output=True, text=True, timeout=120)
        out = subprocess.run(["docker", "port", self.name, "6379/tcp"], check=True, capture_output=True,
                             text=True, timeout=30).stdout.strip().splitlines()[0]
        self.port = int(out.rsplit(":", 1)[1])
        import redis

        self.client = redis.Redis(host="127.0.0.1", port=self.port, decode_responses=False, socket_timeout=300)
        for _ in range(60):
            try:
                if self.client.ping():
                    return
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
        raise RuntimeError("scratch FalkorDB did not answer PING")

    def stop(self) -> None:
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True, text=True, timeout=60)


def format_reply(reply: Any) -> str:
    """Approximately what `redis-cli` prints for a GRAPH.QUERY reply (non-tty: one value per line)."""
    lines: list[str] = []

    def walk(v: Any) -> None:
        if isinstance(v, (list, tuple)):
            for x in v:
                walk(x)
        elif isinstance(v, bytes):
            lines.append(v.decode("utf-8", "replace"))
        elif v is None:
            lines.append("")
        else:
            lines.append(str(v))

    walk(reply)
    return "\n".join(lines) + "\n"


@dataclass
class ScratchGraphs:
    """One (task, model)'s graphs: its OWN scratch server, keys under their production names.
    Implements sandbox.GraphLane.

    One server per (task, model), not shared keys: FalkorDB keeps the graph's name inside the
    payload, so a copy RESTOREd under another key fails on its first write ("empty key when opened
    key orion_worldview"), and a RESTORE ... REPLACE over a live graph key crashed the server --
    both seen in the 2026-10-09 local smoke against falkordb/falkordb:latest."""

    client: Any
    dumps: dict[str, bytes]
    server: Optional["ScratchServer"] = None

    @classmethod
    def fresh(cls, image: str, dumps: dict[str, bytes], name: Optional[str] = None) -> "ScratchGraphs":
        server = ScratchServer(image=image, **({"name": name} if name else {}))
        server.start()
        try:
            graphs = cls(server.client, dumps, server)
            graphs.load()
        except BaseException:
            server.stop()
            raise
        return graphs

    def key(self, graph: str) -> str:
        return graph

    def load(self) -> None:
        for graph, payload in self.dumps.items():
            self.client.execute_command("RESTORE", graph, 0, payload)

    def stop(self) -> None:
        if self.server is not None:
            self.server.stop()

    def list_graphs(self) -> list[str]:
        return sorted(self.dumps)

    def query(self, graph: str, verb: str, cypher: str) -> tuple[str, bool]:
        verb = verb.upper()
        if graph not in self.dumps:
            return f"(error) NOPERM No permissions to access a key ({graph})\n", False
        if graph == ATLAS_GRAPH and verb != "GRAPH.RO_QUERY":
            return f"(error) NOPERM User orion_curiosity has no permissions to run the '{verb.lower()}' command\n", False
        if verb not in ("GRAPH.QUERY", "GRAPH.RO_QUERY", "GRAPH.EXPLAIN"):
            return f"(error) NOPERM User orion_curiosity has no permissions to run the '{verb.lower()}' command\n", False
        try:
            return format_reply(self.client.execute_command(verb, self.key(graph), cypher)), True
        except Exception as exc:  # noqa: BLE001 -- FalkorDB's own error, shown as redis-cli would
            return f"(error) {exc}\n", False

    def read(self, graph: str, cypher: str) -> list[list[Any]]:
        reply = self.client.execute_command("GRAPH.RO_QUERY", self.key(graph), cypher)
        rows = reply[1] if isinstance(reply, list) and len(reply) >= 2 else []
        return [[_plain(v) for v in row] for row in rows]


def _plain(v: Any) -> Any:
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    return v


def prior_snapshot(graphs: ScratchGraphs) -> dict[str, Any]:
    """Priors (id -> confidence/status/times_tested), PriorRevision ids, and node counts per label."""
    priors: dict[str, dict[str, Any]] = {}
    for pid, conf, status, tested in graphs.read(
            OWN_GRAPH, "MATCH (p:Prior) RETURN p.prior_id, p.confidence, p.status, p.times_tested"):
        if pid is None:
            continue
        priors.setdefault(str(pid), {"confidence": _num(conf), "status": status, "times_tested": tested, "copies": 0})
        priors[str(pid)]["copies"] += 1
    revisions = {int(i): {"prior_id": pid, "from": _num(f), "to": _num(t)} for i, pid, f, t in graphs.read(
        OWN_GRAPH, "MATCH (r:PriorRevision) RETURN id(r), r.prior_id, r.from_confidence, r.to_confidence")}
    labels = {str(lab): int(n) for lab, n in graphs.read(OWN_GRAPH, "MATCH (n) RETURN labels(n)[0], count(n)")}
    return {"priors": priors, "revisions": revisions, "labels": labels}


def _num(v: Any) -> Optional[float]:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def diff_snapshots(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """What landed. ``prior_moves``: id -> {from, to, status_from, status_to, new}."""
    moves: dict[str, dict[str, Any]] = {}
    for pid, a in after["priors"].items():
        b = before["priors"].get(pid)
        if b is None:
            moves[pid] = {"new": True, "from": None, "to": a["confidence"], "status_to": a["status"]}
        elif (a["confidence"], a["status"], a["times_tested"], a["copies"]) != \
                (b["confidence"], b["status"], b["times_tested"], b["copies"]):
            moves[pid] = {"new": False, "from": b["confidence"], "to": a["confidence"],
                          "status_from": b["status"], "status_to": a["status"],
                          "tested_from": b["times_tested"], "tested_to": a["times_tested"]}
    new_revisions = [r for i, r in after["revisions"].items() if i not in before["revisions"]]
    label_delta = {k: after["labels"].get(k, 0) - before["labels"].get(k, 0)
                   for k in set(before["labels"]) | set(after["labels"])
                   if after["labels"].get(k, 0) != before["labels"].get(k, 0)}
    return {"prior_moves": moves, "new_revisions": new_revisions, "label_delta": label_delta}
