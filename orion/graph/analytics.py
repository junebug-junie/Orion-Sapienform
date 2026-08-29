"""Structural reads over any FalkorDB graph, using the engine's own algorithms.

WHY THIS EXISTS. FalkorDB ships graph algorithms as stored procedures --
``CALL dbms.procedures()`` on the live instance (module ``graph`` v4.18.11)
lists ``algo.pageRank``, ``algo.WCC``, ``algo.betweenness``,
``algo.labelPropagation``, ``algo.HarmonicCentrality``, ``algo.BFS``,
``algo.SPpaths``, ``algo.SSpaths``, ``algo.MSF`` and ``algo.maxFlow``. As of
2026-08-29 this repo called **none** of them: ``rg "algo\\."`` returned zero
hits, while ``services/orion-hub/scripts/concept_atlas_routes.py`` hand-rolled
union-find and a degree loop in Python, per request, inside a 1732-line route
module no other service can import. This is the shared seam those reads
belong in.

GRAPH-AGNOSTIC BY CONSTRUCTION. Everything here takes a client, never a graph
name of its own, so ``orion_substrate``, ``orion_worldview``,
``orion_bus_synapse`` and ``orion_recall`` are one constructor argument apart
(``RedisGraphQueryClient(uri=..., graph_name=...)``). There is no substrate
vocabulary in this module -- no ``ConceptNodeV1``, no ``node_kind`` semantics
beyond what a caller passes in as a plain label string.

WHAT ACTUALLY WORKS, VERIFIED LIVE 2026-08-29 against FalkorDB 4.18.11 holding
the real ``orion_substrate`` (136 nodes / 461 edges). Being listed by
``dbms.procedures()`` is NOT evidence that a procedure runs -- runtime truth
beats config truth, and three of them are listed but inert:

  WORKS
    algo.WCC                  9.4ms   connected components
    algo.pageRank             4.6ms   influence, edge-direction sensitive
    algo.betweenness          4.5ms   bridge-ness
    algo.HarmonicCentrality   5.1ms   closeness under disconnection
    algo.labelPropagation    18.6ms   communities (config-map form)
    (a)-[*1..n]-(b) varlen    1.6ms   undirected neighbourhood

  RETURNS ZERO ROWS -- deliberately NOT wrapped here
    algo.BFS         empty for every argument form tried:
                     (s, 2, NULL), (s, 1, NULL), (s, 2, 'co_occurs_with'),
                     YIELD nodes / YIELD nodes, edges
    algo.SPpaths     empty; accepts YIELD path, rejects YIELD pathLen
    algo.SSpaths     empty
    shortestPath()   "does not currently support undirected shortestPath
                     traversals"; the directed form returns no path on a
                     semantically-undirected graph

  So ``neighborhood()`` and ``path()`` below are plain variable-length Cypher,
  not procedure wrappers. That is a deliberate downgrade to the primitive that
  demonstrably returns rows, not an oversight.

COST. ``path()`` is the one expensive call: undirected variable-length
enumeration measured **143ms at depth 4** on a 136-node graph, against 1.6ms
for a depth-2 ``neighborhood()``. It grows badly with depth and graph size, so
``max_depth`` is capped at ``MAX_PATH_DEPTH`` and callers should treat it as an
on-demand operator query, never something a page render fans out over.

READ-ONLY. Every string this module emits is a ``CALL``/``MATCH``/``RETURN``
read. ``test_analytics.py`` asserts that against the emitted Cypher rather
than trusting review, and the intended client is built by
``read_only_client()`` so the enforcement is also at the wire (``GRAPH.RO_QUERY``),
the same belt-and-braces worldview.py uses on Orion's own graph.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from orion.graph.falkor_client import FalkorGraphClient

logger = logging.getLogger(__name__)

# measure name -> the exact verified CALL form. A dict lookup rather than string
# interpolation of a caller-supplied name: the procedure name is the one part of
# these queries that cannot be a bound parameter, so it must never come from
# outside.
#
# THE ARITIES ARE NOT CONSISTENT AND GETTING ONE WRONG IS NOT ALWAYS LOUD.
# Verified live 2026-08-29 on FalkorDB 4.18.11:
#   algo.pageRank(null, null)   works; algo.pageRank()      -> ResponseError,
#                               "requires 2 arguments, got 0"
#   algo.betweenness()          works; algo.betweenness(null, null)
#                               -> NO ERROR, zero rows, header only
# So the same mistake raises on one procedure and silently returns an empty
# ranking on the next. That asymmetry is why each form is pinned here as a
# whole string rather than assembled from a bare name plus assumed arguments,
# and why test_analytics.py pins the emitted text.
MEASURES: dict[str, str] = {
    "pagerank": "algo.pageRank(null, null)",
    "betweenness": "algo.betweenness()",
    "harmonic": "algo.HarmonicCentrality()",
}

# Relationship types are likewise not parameterizable in Cypher. FalkorDB's own
# identifier rules are narrower than this, so anything failing it was never a
# real type in the first place.
_REL_TYPE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

MAX_PATH_DEPTH = 4
MAX_NEIGHBOURHOOD_DEPTH = 3
_DEFAULT_TOP_N = 10
_SAMPLE_LABELS = 6


class GraphAnalyticsError(RuntimeError):
    """A structural read could not be answered."""


@dataclass(frozen=True)
class RankedNode:
    node_id: str
    label: Optional[str]
    score: float


@dataclass(frozen=True)
class Component:
    """One connected component.

    ``sample_labels`` is a bounded sample, not the membership: a component can
    hold thousands of nodes and this dataclass is built to be rendered.
    """

    component_id: str
    size: int
    sample_labels: tuple[str, ...] = ()

    @property
    def is_singleton(self) -> bool:
        return self.size == 1


@dataclass(frozen=True)
class StructureSummary:
    """The whole-graph read: what shape is this graph actually in.

    Distinct from any per-request/filtered view a UI computes over the slice it
    happens to be showing -- this is the graph itself, unfiltered.
    """

    node_count: int
    edge_count: int
    edge_type_counts: dict[str, int] = field(default_factory=dict)
    components: tuple[Component, ...] = ()

    @property
    def largest_component_size(self) -> int:
        return max((c.size for c in self.components), default=0)

    @property
    def singleton_count(self) -> int:
        return sum(1 for c in self.components if c.is_singleton)

    @property
    def component_count(self) -> int:
        return len(self.components)

    @property
    def dominant_edge_type(self) -> Optional[str]:
        if not self.edge_type_counts:
            return None
        return max(self.edge_type_counts.items(), key=lambda kv: kv[1])[0]

    def saturation(self, node_count: Optional[int] = None) -> Optional[float]:
        """Edges as a fraction of all possible undirected pairs.

        The legibility number. A graph whose edges are a co-occurrence proxy
        saturates: on live ``orion_substrate`` 2026-08-29, 307 ``co_occurs_with``
        edges over 56 concepts is 307/1540 = **19.9% of every possible pair**,
        which is why ``communities()`` returns a single community there. High
        saturation means no algorithm will find structure, because there is
        none to find -- the fix is upstream, in what the producer calls an edge.

        ``node_count`` overrides the graph-wide count so a caller can ask the
        question about the subpopulation an edge type actually connects (e.g.
        concepts only), which is the honest denominator.
        """
        n = self.node_count if node_count is None else node_count
        if n < 2:
            return None
        return self.edge_count / (n * (n - 1) / 2)


def _rows(client: FalkorGraphClient, cypher: str, params: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
    try:
        result = client.graph_query(cypher, params or None)
    except Exception as exc:  # noqa: BLE001 - surfaced as a typed error below
        raise GraphAnalyticsError(f"graph query failed: {exc}") from exc
    return list(result or [])


def _validated_rel_types(rel_types: Optional[Sequence[str]]) -> list[str]:
    """Normalise and validate relationship type names.

    A relationship type cannot be a bound Cypher parameter, so it is
    interpolated -- which makes this the injection boundary for every call
    below that accepts one. Rejects rather than silently drops, so a typo
    surfaces as an error instead of as a query that quietly means something
    else than the caller asked for.
    """
    if not rel_types:
        return []
    cleaned = [str(t).strip() for t in rel_types if str(t).strip()]
    bad = [t for t in cleaned if not _REL_TYPE_RE.match(t)]
    if bad:
        raise ValueError(f"refusing to build Cypher for non-identifier relationship types: {bad!r}")
    return cleaned


def _rel_filter(rel_types: Optional[Sequence[str]]) -> str:
    """Build a ``:A|B`` relationship filter, or empty for "any type"."""
    cleaned = _validated_rel_types(rel_types)
    if not cleaned:
        return ""
    return ":" + "|".join(cleaned)


class GraphAnalytics:
    """Structural reads over one FalkorDB graph.

    Holds no graph name: the client it is given already points at one, which is
    what makes the same object usable against every graph on the instance.
    """

    def __init__(
        self,
        client: FalkorGraphClient,
        *,
        id_property: str = "node_id",
        label_property: str = "label",
    ) -> None:
        """
        ``id_property``/``label_property`` are what make this genuinely
        graph-agnostic rather than substrate-shaped. Every graph on this
        FalkorDB instance names them differently -- verified live 2026-08-29:

            orion_substrate     node_id     label
            orion_recall        turn_id / session_id     (no label property)
            orion_bus_synapse   organ_id / channel       (no label property)

        Hardcoding ``node.label`` would have returned ``label=None`` for all
        3864 nodes of ``orion_recall`` while looking like it worked. The
        defaults keep substrate callers unchanged.

        Neither can be a bound Cypher parameter (they are property accessors),
        so both are validated as plain identifiers here, once, rather than at
        each query site.
        """
        self._client = client
        for name, value in (("id_property", id_property), ("label_property", label_property)):
            if not _REL_TYPE_RE.match(str(value)):
                raise ValueError(f"refusing to build Cypher for a non-identifier {name}: {value!r}")
        self._id = str(id_property)
        self._label = str(label_property)

    # --- census -------------------------------------------------------------

    def node_count(self, label: Optional[str] = None) -> int:
        if label is not None and not _REL_TYPE_RE.match(str(label)):
            raise ValueError(f"refusing to build Cypher for a non-identifier label: {label!r}")
        match = f"MATCH (n:{label})" if label else "MATCH (n)"
        rows = _rows(self._client, f"{match} RETURN count(n) AS n")
        return int(rows[0].get("n", 0)) if rows else 0

    def edge_type_counts(self) -> dict[str, int]:
        rows = _rows(
            self._client,
            "MATCH ()-[r]->() RETURN type(r) AS edge_type, count(r) AS n ORDER BY n DESC",
        )
        return {str(r.get("edge_type")): int(r.get("n", 0)) for r in rows if r.get("edge_type")}

    # --- structure ----------------------------------------------------------

    def components(self, *, sample_labels: int = _SAMPLE_LABELS) -> tuple[Component, ...]:
        """Connected components over the whole graph, largest first.

        ``algo.WCC`` ignores edge direction, which is the right reading for a
        semantic graph: ``a -[co_occurs_with]-> b`` does not mean b is
        unreachable from a to a human looking at the picture.
        """
        rows = _rows(
            self._client,
            "CALL algo.WCC(null) YIELD node, componentId "
            f"WITH componentId, count(node) AS n, collect(node.{self._label}) AS labels "
            f"RETURN componentId, n, [x IN labels WHERE x IS NOT NULL][0..{int(sample_labels)}] AS sample "
            "ORDER BY n DESC",
        )
        return tuple(
            Component(
                component_id=str(r.get("componentId")),
                size=int(r.get("n", 0)),
                sample_labels=tuple(str(x) for x in (r.get("sample") or [])),
            )
            for r in rows
        )

    def communities(self, *, rel_types: Optional[Sequence[str]] = None, min_size: int = 2) -> tuple[Component, ...]:
        """Label-propagation communities, largest first.

        A single community spanning the whole connected graph is a real and
        common answer -- see ``StructureSummary.saturation``. Read it as "this
        graph has no community structure", not as a failed call.
        """
        cleaned = _validated_rel_types(rel_types)
        if cleaned:
            # verified live: labelPropagation takes a config map, unlike the
            # positional (label, relType) form pageRank/WCC use.
            types = ", ".join(f"'{t}'" for t in cleaned)
            call = f"CALL algo.labelPropagation({{relationshipTypes:[{types}]}})"
        else:
            call = "CALL algo.labelPropagation(null)"
        rows = _rows(
            self._client,
            f"{call} YIELD node, communityId "
            f"WITH communityId, count(node) AS n, collect(node.{self._label}) AS labels "
            f"WHERE n >= {int(min_size)} "
            f"RETURN communityId, n, [x IN labels WHERE x IS NOT NULL][0..{_SAMPLE_LABELS}] AS sample "
            "ORDER BY n DESC",
        )
        return tuple(
            Component(
                component_id=str(r.get("communityId")),
                size=int(r.get("n", 0)),
                sample_labels=tuple(str(x) for x in (r.get("sample") or [])),
            )
            for r in rows
        )

    # --- ranking ------------------------------------------------------------

    def rank(self, measure: str, *, top_n: int = _DEFAULT_TOP_N) -> tuple[RankedNode, ...]:
        """Top nodes by one centrality measure.

        The measures disagree, and the disagreement is the point. Live on
        ``orion_substrate`` 2026-08-29: Orion and Juniper are pageRank #1/#2 and
        absent from the betweenness top 8 -- connected to everything, bridging
        nothing. Reading only one measure hides that.
        """
        procedure = MEASURES.get(str(measure).strip().lower())
        if procedure is None:
            raise ValueError(f"unknown measure {measure!r}; known: {sorted(MEASURES)}")
        limit = max(1, int(top_n))
        rows = _rows(
            self._client,
            f"CALL {procedure} YIELD node, score "
            f"RETURN node.{self._id} AS node_id, node.{self._label} AS label, score "
            f"ORDER BY score DESC LIMIT {limit}",
        )
        return tuple(
            RankedNode(
                node_id=str(r.get("node_id") or ""),
                label=(str(r["label"]) if r.get("label") else None),
                score=float(r.get("score") or 0.0),
            )
            for r in rows
        )

    # --- traversal ----------------------------------------------------------

    def neighborhood(
        self,
        node_id: str,
        *,
        depth: int = 1,
        rel_types: Optional[Sequence[str]] = None,
        limit: int = 200,
    ) -> tuple[RankedNode, ...]:
        """Everything within ``depth`` undirected hops, nearest first.

        ``score`` carries the hop distance, so a caller can render rings
        without a second query.
        """
        d = max(1, min(int(depth), MAX_NEIGHBOURHOOD_DEPTH))
        rel = _rel_filter(rel_types)
        rows = _rows(
            self._client,
            f"MATCH (s) WHERE s.{self._id} = $node_id "
            f"MATCH p = (s)-[{rel}*1..{d}]-(m) "
            "WITH m, min(length(p)) AS hops "
            f"RETURN m.{self._id} AS node_id, m.{self._label} AS label, hops "
            f"ORDER BY hops ASC, label ASC LIMIT {max(1, int(limit))}",
            {"node_id": str(node_id)},
        )
        return tuple(
            RankedNode(
                node_id=str(r.get("node_id") or ""),
                label=(str(r["label"]) if r.get("label") else None),
                score=float(r.get("hops") or 0.0),
            )
            for r in rows
        )

    def path(
        self,
        source_node_id: str,
        target_node_id: str,
        *,
        max_depth: int = 3,
        rel_types: Optional[Sequence[str]] = None,
    ) -> tuple[RankedNode, ...]:
        """Shortest undirected path, or empty when none within ``max_depth``.

        Plain variable-length Cypher because every path *procedure* on this
        build returns zero rows (see the module docstring). Expensive -- 143ms
        at depth 4 on a 136-node graph -- so ``max_depth`` is capped at
        ``MAX_PATH_DEPTH`` and this is an operator query, not a render-path one.

        Empty means "no path within max_depth", which is NOT the same as "not
        connected"; raise the depth or check ``components()`` before concluding.
        """
        d = max(1, min(int(max_depth), MAX_PATH_DEPTH))
        rel = _rel_filter(rel_types)
        rows = _rows(
            self._client,
            f"MATCH (a), (b) WHERE a.{self._id} = $source AND b.{self._id} = $target "
            f"MATCH p = (a)-[{rel}*1..{d}]-(b) "
            "WITH p ORDER BY length(p) ASC LIMIT 1 "
            f"RETURN [n IN nodes(p) | n.{self._id}] AS ids, [n IN nodes(p) | n.{self._label}] AS labels",
            {"source": str(source_node_id), "target": str(target_node_id)},
        )
        if not rows:
            return ()
        ids = list(rows[0].get("ids") or [])
        labels = list(rows[0].get("labels") or [])
        out: list[RankedNode] = []
        for index, node_id in enumerate(ids):
            raw_label = labels[index] if index < len(labels) else None
            out.append(
                RankedNode(
                    node_id=str(node_id or ""),
                    label=(str(raw_label) if raw_label else None),
                    score=float(index),
                )
            )
        return tuple(out)

    # --- assembled ----------------------------------------------------------

    def summary(self) -> StructureSummary:
        """One call for "what shape is this graph in"."""
        edge_types = self.edge_type_counts()
        return StructureSummary(
            node_count=self.node_count(),
            edge_count=sum(edge_types.values()),
            edge_type_counts=edge_types,
            components=self.components(),
        )
