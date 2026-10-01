"""Guarantee `written_at` on the run nodes Orion writes into its own graph.

WHY THIS EXISTS. Orion writes `orion_worldview` itself, in-turn, with Cypher it
composes and runs through `redis-cli GRAPH.QUERY` from the `claude -p`
subprocess (`orion/curiosity/kickoff_prompt.py`). Whether a `:Hop` carries
`written_at` therefore depended on the model copying the prompt's example.
Live 2026-10-01: 186 of 594 Hops, 126 Findings and 1 InvestigationRole had no
`written_at`, and `atlas.run_ids_since_cypher` cannot see a run with no dated
node at all. Commit e88766059 fixed the example; nothing enforced it.

WHERE THE INVARIANT IS ENFORCED. There is no code-side write call to wrap --
the write is a shell command inside the subprocess. The one piece of code that
sees every one of those commands complete, in order, is the FCC motor's
stream-json loop (`orion/harness/fcc_motor.py`). So the motor:

  1. takes a BASELINE before spawning the turn: the internal ids of every
     run node that already lacks `written_at` (the legacy ones);
  2. after each Bash tool result whose command ran `GRAPH.QUERY`, stamps
     `written_at = timestamp()` on run nodes that lack it AND are not in the
     baseline -- i.e. only nodes created during this turn, seconds after the
     write landed;
  3. runs the same stamp once more when the turn ends (catches a write the
     command check missed, e.g. a script; that stamp is turn-end time).

Known gaps, all in the safe direction (a node stays unstamped, never faked):
a write still in flight when a killed turn's end stamp runs (`redis-cli` is a
grandchild of `claude`), a node whose in-turn stamp failed on a FalkorDB error,
and a new node that reuses the internal id of a legacy node deleted mid-turn.
Each is then in every later turn's baseline and stays unstamped for good.

Legacy unstamped nodes are never touched: their real write time is unknown and
stamping them now would fake it. Without a baseline (graph unreachable at turn
start) nothing is stamped at all -- fail closed, never guess.

A stamped node also gets `written_at_source = 'harness_stamp'`, so a reader can
tell the harness's clock from one Orion wrote itself. Nodes that already carry
`written_at` -- including an ISO string -- are left exactly as written.

Model Cypher is never rewritten; the stamp is a separate, deterministic query.
Credentials are Orion's own curiosity ACL user (RW on `orion_worldview` only),
already handed to the subprocess by `orion/curiosity/sandbox_env.py`; absent
keys mean no stamper, same kill switch as that module.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping, Optional

from orion.curiosity.atlas import RUN_NODE_FIELDS

logger = logging.getLogger("orion.curiosity.write_stamp")

# Every per-run label the atlas reads `written_at` from. Derived, not restated,
# so a new run-node label the atlas starts reading gets the guarantee too.
STAMPED_LABELS: tuple[str, ...] = tuple(sorted(RUN_NODE_FIELDS))

STAMP_SOURCE = "harness_stamp"

_LABEL_PREDICATE = (
    "any(l IN labels(n) WHERE l IN ["
    + ", ".join(f"'{label}'" for label in STAMPED_LABELS)
    + "])"
)


def baseline_cypher() -> str:
    """Ids of every run node that lacks `written_at` right now (read-only)."""
    return f"MATCH (n) WHERE n.written_at IS NULL AND {_LABEL_PREDICATE} RETURN id(n) AS id"


def stamp_cypher(baseline_ids: Iterable[int]) -> str:
    """Stamp run nodes lacking `written_at` that are NOT in the baseline.

    Ids are coerced to int before they reach the query, and passed through
    FalkorDB's `CYPHER k=v` parameter prefix rather than spliced into the
    pattern."""
    ids = ",".join(str(int(i)) for i in sorted(set(baseline_ids)))
    return (
        f"CYPHER baseline=[{ids}] "
        f"MATCH (n) WHERE n.written_at IS NULL AND {_LABEL_PREDICATE} "
        "AND NOT id(n) IN $baseline "
        f"SET n.written_at = timestamp(), n.written_at_source = '{STAMP_SOURCE}' "
        "RETURN count(n) AS stamped"
    )


def _first_column(reply: Any) -> list[Any]:
    """`[header, rows, stats]` -> first value of each row."""
    if not isinstance(reply, (list, tuple)) or len(reply) < 2:
        return []
    return [row[0] for row in (reply[1] or []) if isinstance(row, (list, tuple)) and row]


class WriteStamper:
    """Per-turn baseline + stamp. One instance per FCC turn."""

    def __init__(self, *, client: Any, graph_name: str) -> None:
        self._client = client
        self.graph_name = graph_name
        self._baseline: Optional[frozenset[int]] = None
        self.stamped_total = 0

    @classmethod
    def from_env(cls, env: Mapping[str, str], *, socket_timeout: float = 1.0) -> Optional["WriteStamper"]:
        """None unless every curiosity graph key is present (the kill switch)."""
        host = str(env.get("ORION_CURIOSITY_GRAPH_HOST") or "").strip()
        port = str(env.get("ORION_CURIOSITY_GRAPH_PORT") or "").strip()
        user = str(env.get("ORION_CURIOSITY_GRAPH_USER") or "").strip()
        password = str(env.get("ORION_CURIOSITY_GRAPH_PASSWORD") or "").strip()
        graph = str(env.get("ORION_CURIOSITY_GRAPH_OWN") or "").strip()
        if not (host and port and user and password and graph):
            return None
        try:
            import redis  # local import: keeps this module importable in unit tests

            client = redis.Redis(
                host=host,
                port=int(port),
                username=user,
                password=password,
                decode_responses=True,
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_timeout,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("write_stamp_client_failed err=%s", exc)
            return None
        return cls(client=client, graph_name=graph)

    def close(self) -> None:
        """Release the per-turn connection pool; never raises."""
        try:
            close = getattr(self._client, "close", None)
            if callable(close):
                close()
        except Exception:  # noqa: BLE001
            pass

    @property
    def armed(self) -> bool:
        return self._baseline is not None

    def take_baseline(self) -> bool:
        """Record the legacy unstamped set. False (and disarmed) on any failure."""
        try:
            reply = self._client.execute_command("GRAPH.RO_QUERY", self.graph_name, baseline_cypher())
            self._baseline = frozenset(int(i) for i in _first_column(reply))
        except Exception as exc:  # noqa: BLE001
            self._baseline = None
            logger.warning("write_stamp_baseline_failed graph=%s err=%s", self.graph_name, exc)
            return False
        return True

    def stamp(self) -> Optional[int]:
        """Stamp this turn's unstamped run nodes. None when disarmed or on error."""
        if self._baseline is None:
            return None
        try:
            reply = self._client.execute_command(
                "GRAPH.QUERY", self.graph_name, stamp_cypher(self._baseline)
            )
            values = _first_column(reply)
            count = int(values[0]) if values else 0
        except Exception as exc:  # noqa: BLE001
            logger.warning("write_stamp_failed graph=%s err=%s", self.graph_name, exc)
            return None
        self.stamped_total += count
        return count


def graph_write_tool_use_ids(event: Mapping[str, Any]) -> set[str]:
    """Ids of Bash tool calls in an assistant event whose command runs GRAPH.QUERY.

    Read-only inspection of the command, used only to decide WHEN to stamp. A
    miss costs accuracy (the end-of-turn stamp is later), never legacy safety.
    `GRAPH.RO_QUERY` does not contain the substring and cannot write."""
    if str(event.get("type") or "") != "assistant":
        return set()
    content = (event.get("message") or {}).get("content") if isinstance(event.get("message"), dict) else None
    out: set[str] = set()
    for block in content if isinstance(content, list) else []:
        if not isinstance(block, dict) or block.get("type") != "tool_use" or block.get("name") != "Bash":
            continue
        command = (block.get("input") or {}).get("command") if isinstance(block.get("input"), dict) else None
        # redis-cli commands are case-insensitive; `graph.ro_query` still does not match.
        if isinstance(command, str) and "graph.query" in command.lower() and block.get("id"):
            out.add(str(block["id"]))
    return out


def completed_tool_use_ids(event: Mapping[str, Any]) -> set[str]:
    """Ids of tool calls whose results arrive in this (user) event."""
    if str(event.get("type") or "") != "user":
        return set()
    content = (event.get("message") or {}).get("content") if isinstance(event.get("message"), dict) else None
    return {
        str(block["tool_use_id"])
        for block in (content if isinstance(content, list) else [])
        if isinstance(block, dict) and block.get("type") == "tool_result" and block.get("tool_use_id")
    }
