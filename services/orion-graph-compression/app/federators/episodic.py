from __future__ import annotations

import logging
from typing import List, Tuple

import httpx

logger = logging.getLogger("orion.graph-compression.federator.episodic")

EPISODIC_GRAPHS = [
    "http://conjourney.net/graph/orion/chat",
    "http://conjourney.net/graph/orion/enrichment",
    "http://conjourney.net/graph/orion/cognition",
    "http://conjourney.net/graph/orion/metacog",
    # orion/collapse removed 2026-07-23: live SPARQL COUNT confirmed exactly
    # 0 triples, ever -- not "low," zero, and structurally guaranteed to
    # stay that way, not just rare. orion-rdf-writer's collapse.mirror.entry
    # handler (_build_raw_collapse_graph) exists and has its own
    # observer==Juniper/dense gate, but that gate is unreachable: rdf-writer
    # only subscribes to orion:collapse:intake (CHANNEL_EVENTS_COLLAPSE),
    # which carries kind="collapse.mirror.intake" from orion-cortex-exec.
    # The only real producer of kind="collapse.mirror.entry" is
    # orion-collapse-mirror/app/bus_runtime.py, which publishes it to a
    # DIFFERENT channel, orion:collapse:triage -- registered in
    # channels.yaml with consumer_services orion-meta-tags/orion-vector-
    # writer/orion-sql-writer/orion-actions, NOT orion-rdf-writer. So the
    # dispatch branch that would write this graph never actually receives
    # a matching envelope at all, independent of the observer/dense gate.
    # Graph-name resolution confirmed correct separately
    # (rdf_store.py::normalize_graph_name maps "orion:collapse" to this
    # exact URI) -- not a naming-mismatch bug either. Since it's provably
    # always been empty, EpisodicFederator's read of it has never
    # contributed any clustering signal; removing it changes nothing about
    # today's real clusters. (The channel/kind mismatch itself is a
    # separate, real bug in orion-rdf-writer worth its own follow-up if
    # anyone ever wants this write path alive -- not fixed here, since the
    # whole point of this series is retiring Fuseki reads, not reviving
    # Fuseki writes.)
    # orion/chat/social removed 2026-07-23: live-verified pure redundancy.
    # Postgres social_room_turns already owns richer content (33 rows,
    # actual prompt/response/text -- the Fuseki triples were metadata-only,
    # no text at all). Nothing else read this graph back. Social-turn
    # tag/entity clustering signal is covered going forward by the
    # already-live (if currently thin) Falkor write in orion-meta-tags
    # (unconditional for both chat.history and social.turn.stored.v1 since
    # 2026-07-18) via FalkorEpisodicFederator.
    # Autonomy graphs are written under graph/autonomy/* (NOT graph/orion/autonomy/*)
    # — see orion/autonomy/constants.py.
    "http://conjourney.net/graph/autonomy/identity",
    "http://conjourney.net/graph/autonomy/drives",
    "http://conjourney.net/graph/autonomy/goals",
]

Triple = Tuple[str, str, str]


class EpisodicFederator:
    def __init__(
        self,
        *,
        query_url: str,
        user: str,
        password: str,
        timeout_sec: float,
    ) -> None:
        self._query_url = query_url
        self._auth = (user, password)
        self._timeout = timeout_sec

    def _build_sparql(self, max_nodes: int = 2000) -> str:
        # UNION (not conjunction) across graphs: a subject/triple may live in any
        # one of the episodic graphs. Concatenating the GRAPH clauses would require
        # the same triple to exist in ALL graphs simultaneously (≈ empty result).
        union_block = "\n      UNION\n      ".join(
            f"{{ GRAPH <{g}> {{ ?s ?p ?o }} }}" for g in EPISODIC_GRAPHS
        )
        return f"""
SELECT ?s ?p ?o WHERE {{
  {{
    SELECT DISTINCT ?s WHERE {{
      {union_block}
    }}
    LIMIT {max_nodes}
  }}
  {union_block}
}}
"""

    def fetch(self, *, max_nodes: int = 2000) -> List[Triple]:
        query = self._build_sparql(max_nodes)
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(
                    self._query_url,
                    data={"query": query},
                    headers={"Accept": "application/sparql-results+json"},
                    auth=self._auth,
                )
                resp.raise_for_status()
                bindings = resp.json().get("results", {}).get("bindings", [])
        except Exception as exc:
            logger.warning("episodic_federator_fetch_failed reason=%s", exc)
            return []
        # Keep only object terms that are graph nodes (IRIs/bnodes). Literal
        # objects (chat text, timestamps) are not entities and must not become
        # cluster nodes — they would later be serialized as invalid <IRI>s.
        return [
            (b["s"]["value"], b["p"]["value"], b["o"]["value"])
            for b in bindings
            if "s" in b and "p" in b and "o" in b
            and b["o"].get("type") in ("uri", "bnode")
        ]
