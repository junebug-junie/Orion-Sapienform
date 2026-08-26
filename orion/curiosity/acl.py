"""Re-assert Orion's FalkorDB ACL at startup, because it does not survive a restart.

THIS IS NOT BELT-AND-BRACES; IT IS THE ONLY THING KEEPING THE GRANT ALIVE.
FalkorDB on this host runs with `aclfile` unset, and that config is immutable
at runtime -- `CONFIG SET aclfile` answers *"can't set immutable config"*, and
`ACL SAVE` refuses without one. Persisting the rule properly would mean
restarting FalkorDB, which holds every graph in the system. So the rule lives
only in the running process's memory, and any FalkorDB restart silently
removes Orion's access.

WHAT "SILENTLY" WOULD COST. Without this, a restart degrades the curiosity loop
to `stores_unavailable` forever, and the only symptom is an absence of journal
entries -- which is also what a quiet stretch looks like. That is the exact
silent-failure shape this arc has already hit three times (the 21h vision
blackout, the transcript mount, the frozen cooldown). A self-healing
`ACL SETUSER` on every loop start costs one round trip and removes the failure
mode entirely, with no config change and no downtime.

IDEMPOTENT BY CONSTRUCTION, AND `clearselectors` IS LOAD-BEARING. The reset
directives clear the whole prior grant before the new one is applied, so
replaying this against a user that already exists produces byte-identical rules
rather than accumulating permissions -- which matters because the alternative
(`+graph.query` added on top of an existing wildcard) is how a read-only grant
quietly becomes a write one.

`resetkeys resetchannels nocommands` do NOT cover selectors, and that is not a
theoretical gap: measured live 2026-08-26 against this deployment, replaying
the argv WITHOUT `clearselectors` appended a second identical
`(~orion_worldview ...)` selector, and a third replay a third -- one more per
Hub start, growing forever. `clearselectors` makes three consecutive replays
byte-identical, verified against the live rule's own shape.

THE RULE, AND WHY EACH HALF IS THERE.

    base grant    ~orion_substrate  +graph.ro_query
                  Read the Juniper-curated Atlas. `GRAPH.RO_QUERY` refuses a
                  write on its own ("graph.RO_QUERY is to be executed only on
                  read-only queries"), so the Atlas has two independent
                  refusals on the write path: the key ACL and the command.
    selector      (~orion_worldview +graph.query +graph.ro_query)
                  Write-capable, on Orion's own graph and nothing else.

Everything else is denied, including the bus-synapse graphs and `GRAPH.LIST`.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger("orion.curiosity.acl")


def acl_setuser_argv(
    *,
    username: str,
    password: str,
    atlas_graph: str,
    own_graph: str,
) -> list[str]:
    """The exact `ACL SETUSER` argv, as a list -- never a shell string.

    Built as argv rather than interpolated into one command string so the
    password is never subject to shell or Redis inline-command quoting: the
    client sends it as a single bulk argument whatever bytes are in it.
    """
    for name, value in (
        ("username", username),
        ("password", password),
        ("atlas_graph", atlas_graph),
        ("own_graph", own_graph),
    ):
        if not str(value or "").strip():
            raise ValueError(f"curiosity ACL cannot be asserted without {name}")
    return [
        "ACL",
        "SETUSER",
        username,
        "on",
        f">{password}",
        "resetkeys",
        "resetchannels",
        "nocommands",
        # Not redundant with the three above -- see the module docstring.
        # Without it, every Hub start appends another copy of the selector
        # below, forever.
        "clearselectors",
        f"~{atlas_graph}",
        "+graph.ro_query",
        f"(~{own_graph} +graph.query +graph.ro_query)",
    ]


def ensure_graph_exists(*, client: Any, graph_name: str) -> Optional[str]:
    """Create `graph_name` if FalkorDB has never seen it. Idempotent.

    WITHOUT THIS THE FIRST EVER RUN DEADLOCKS, and it is a real deadlock rather
    than a slow start. Verified live 2026-08-26: `GRAPH.RO_QUERY <unknown-graph>`
    answers `ERR Invalid graph operation on empty key`, not an empty result. So
    on a fresh deployment `read_snapshot` reports the graph as UNAVAILABLE, the
    prompt therefore drops the schema and `:TurnOutcome` sections (correctly --
    it must not name a store this run cannot reach), Orion is never shown how to
    write a node, no node is ever written, and the graph is never created. The
    only symptom is a `curiosity_worldview_degraded` warning, forever.

    `GRAPH.QUERY ... "RETURN 1"` is the cheapest write-capable no-op that
    materialises the key; on an existing graph it changes nothing. Run as
    FalkorDB's `default` user, which is the one connection Hub holds that can
    write -- Orion's own ACL user could do it too, but Hub asserting the grant
    and then depending on Orion to use it would make the bootstrap depend on a
    turn happening first, which is the deadlock again one step out.
    """
    if not str(graph_name or "").strip():
        return "misconfigured: no graph name"
    try:
        client.execute_command("GRAPH.QUERY", graph_name, "RETURN 1")
    except Exception as exc:  # noqa: BLE001 -- reported, never raised
        return f"{type(exc).__name__}: {str(exc)[:160]}"
    return None


def assert_orion_acl(
    *,
    client: Any,
    username: str,
    password: str,
    atlas_graph: str,
    own_graph: str,
) -> Optional[str]:
    """Apply the rule. Returns None on success, or a short failure reason.

    Never raises: a FalkorDB that is down at Hub startup must not stop Hub
    starting. The failure is returned so the caller can BLOCK the loop on it
    rather than letting the first run discover the missing grant as prose
    inside a turn -- see `curiosity_investigation.py`'s `graph_unavailable`
    block reason, which is the deterministic gate this feeds.
    """
    try:
        argv = acl_setuser_argv(
            username=username,
            password=password,
            atlas_graph=atlas_graph,
            own_graph=own_graph,
        )
    except ValueError as exc:
        return f"misconfigured: {exc}"
    try:
        client.execute_command(*argv)
    except Exception as exc:  # noqa: BLE001 -- reported, never raised
        return f"{type(exc).__name__}: {str(exc)[:160]}"
    logger.info(
        "curiosity_acl_asserted user=%s ro=%s rw=%s", username, atlas_graph, own_graph
    )
    return None
