"""Durable neighborhood adapters. All queries are reads and bypass store caches."""
from __future__ import annotations

from .neighborhood import ACCEPTED_ASSERTION_STATES, NeighborhoodRequestV1, read_neighborhood

# Cypher form of neighborhood.walkable_edge(): legacy edges walk as before; a
# semantic projection walks only while its Assertion is accepted at the edge's
# revision. Provenance and assertion-structure edges never walk. Appended AFTER
# the focal/endpoint filter so the OPTIONAL MATCH runs only for candidate edges.
_WALKABLE_EDGE_TAIL = (
    "OPTIONAL MATCH (assertion:SubstrateNode) WHERE assertion.node_id = e.assertion_id "
    "WITH source, e, target, assertion WHERE "
    "(coalesce(e.edge_role, 'legacy_unreviewed') = 'legacy_unreviewed' "
    "OR (e.edge_role = 'semantic_projection' AND assertion.node_kind = 'assertion' "
    "AND assertion.promotion_state IN $accepted_assertion_states "
    "AND assertion.assertion_revision = e.assertion_revision)) "
)


def _groups(ids, request, predicates):
    result = []
    for focal in ids:
        for direction in ("incoming", "outgoing"):
            if request.direction not in {"both", direction}:
                continue
            after = ""
            while True:
                page = predicates(ids, focal, direction, after)
                if not page:
                    break
                for predicate in page:
                    if not isinstance(predicate, str) or predicate <= after:
                        raise ValueError("invalid_or_nonadvancing_predicate")
                    after = predicate
                    result.append((focal, direction, predicate))
                    if len(result) > 16 * 2 * 15:
                        raise ValueError("too_many_boundary_groups")
    return result


def read_falkor_neighborhood(store, request: NeighborhoodRequestV1):
    from .falkor_store import (
        NATIVE_NODE_RETURN_FIELDS, NATIVE_EDGE_RETURN_FIELDS,
        _normalize_rows, _return_clause, _edge_hydrate_return_clause,
    )
    from .falkor_codec import decode_node, decode_edge

    def query(text, params, fields):
        return _normalize_rows(store._client.graph_query(text, params=params), fields=fields)

    def nodes(ids):
        result = []
        for node_id in ids:
            fields = NATIVE_NODE_RETURN_FIELDS
            rows = query("MATCH (n:SubstrateNode) WHERE n.node_id = $node_id RETURN "
                + _return_clause("n", fields) + " LIMIT 2", {"node_id": node_id}, fields)
            for row in rows:
                node = decode_node(row)
                if node is None:
                    raise ValueError("invalid_node")
                result.append(node)
        return result

    def where(ids, group=None):
        params = {"ids": ids, "states": list(request.semantic_states),
                  "scopes": list(request.anchor_scopes),
                  "accepted_assertion_states": list(ACCEPTED_ASSERTION_STATES)}
        condition = ("e.substrate_edge = true "
            "AND source.node_kind IN ['concept', 'entity'] "
            "AND target.node_kind IN ['concept', 'entity'] "
            "AND source.promotion_state IN $states AND target.promotion_state IN $states "
            "AND source.anchor_scope IN $scopes AND target.anchor_scope IN $scopes ")
        if group is None:
            condition += "AND source.node_id IN $ids AND target.node_id IN $ids "
        else:
            focal, direction, predicate = group
            inside, outside = ("target", "source") if direction == "incoming" else ("source", "target")
            condition += f"AND {inside}.node_id = $focal AND NOT ({outside}.node_id IN $ids) "
            params["focal"] = focal
            if predicate is not None:
                condition += "AND e.predicate = $predicate "
                params["predicate"] = predicate
        return condition, params

    match = "MATCH (source:SubstrateNode)-[e]->(target:SubstrateNode) WHERE "

    def predicates(ids, focal, direction, after):
        condition, params = where(ids, (focal, direction, None))
        params["after"] = after
        rows = query(match + condition + "AND e.predicate > $after " + _WALKABLE_EDGE_TAIL
            + "RETURN DISTINCT e.predicate AS predicate ORDER BY predicate LIMIT 16",
            params, ("predicate",))
        return [row["predicate"] for row in rows]

    def edges(ids, group, after, limit):
        condition, params = where(ids, group)
        params.update(after=after)
        fields = NATIVE_EDGE_RETURN_FIELDS
        rows = query(match + condition + "AND e.edge_id > $after " + _WALKABLE_EDGE_TAIL + "RETURN "
            + _edge_hydrate_return_clause(fields) + f" ORDER BY e.edge_id LIMIT {limit}", params, fields)
        result = []
        for row in rows:
            edge = decode_edge(row)
            if edge is None:
                raise ValueError("invalid_edge")
            result.append(edge)
        return result

    return read_neighborhood(request, source_kind="falkor", nodes=nodes,
        groups=lambda ids: _groups(ids, request, predicates), edges=edges)


def read_sparql_neighborhood(store, request: NeighborhoodRequestV1):
    from .graphdb_store import NODE_ADAPTER, ORION_SUBSTRATE_NS
    from orion.core.schemas.cognitive_substrate import SubstrateEdgeV1

    prefix = f"PREFIX orion: <{ORION_SUBSTRATE_NS}>\n"
    graph = f"GRAPH <{store._cfg.graph_uri}>"
    lit = store._lit

    def values(items):
        return ", ".join(lit(item) for item in items)

    def nodes(ids):
        result = []
        for node_id in ids:
            rows = store._select(prefix + f"SELECT ?payload_json WHERE {{ {graph} {{ "
                f"?n orion:nodeId {lit(node_id)} ; orion:payloadJson ?payload_json . }} }} LIMIT 2")
            for row in rows:
                result.append(NODE_ADAPTER.validate_json(store._binding_str(row, "payload_json")))
        return result

    def pattern(ids, group=None):
        # Empty IN lists are not portable SPARQL. The driver has no eligible
        # focal nodes when states/scopes are empty, so this path is not entered.
        clause = ("?edge a orion:SubstrateEdge ; orion:edgeId ?edge_id ; "
            "orion:sourceNodeId ?source_id ; orion:targetNodeId ?target_id ; "
            "orion:predicate ?predicate ; orion:payloadJson ?payload_json . "
            "?source orion:nodeId ?source_id ; orion:nodeKind ?source_kind ; "
            "orion:promotionState ?source_state ; orion:anchorScope ?source_scope . "
            "?target orion:nodeId ?target_id ; orion:nodeKind ?target_kind ; "
            "orion:promotionState ?target_state ; orion:anchorScope ?target_scope . "
            'FILTER(?source_kind IN ("concept", "entity") && ?target_kind IN ("concept", "entity")) '
            # This backend stores no Assertion nodes, so it cannot verify a
            # semantic projection: it walks legacy edges only (fail closed).
            "OPTIONAL { ?edge orion:edgeRole ?edge_role . } "
            'FILTER(!BOUND(?edge_role) || ?edge_role = "legacy_unreviewed") '
            f"FILTER(?source_state IN ({values(request.semantic_states)}) && "
            f"?target_state IN ({values(request.semantic_states)})) "
            f"FILTER(?source_scope IN ({values(request.anchor_scopes)}) && "
            f"?target_scope IN ({values(request.anchor_scopes)})) ")
        if group is None:
            clause += f"FILTER(?source_id IN ({values(ids)}) && ?target_id IN ({values(ids)})) "
        else:
            focal, direction, predicate = group
            inside, outside = ("target_id", "source_id") if direction == "incoming" else ("source_id", "target_id")
            clause += f"FILTER(?{inside} = {lit(focal)} && ?{outside} NOT IN ({values(ids)})) "
            if predicate is not None:
                clause += f"FILTER(?predicate = {lit(predicate)}) "
        return clause

    def predicates(ids, focal, direction, after):
        rows = store._select(prefix + f"SELECT DISTINCT ?predicate WHERE {{ {graph} {{ "
            + pattern(ids, (focal, direction, None)) + f"FILTER(?predicate > {lit(after)}) "
            "} } ORDER BY ?predicate LIMIT 16")
        return [store._binding_str(row, "predicate") for row in rows]

    def edges(ids, group, after, limit):
        rows = store._select(prefix + f"SELECT ?edge_id ?payload_json WHERE {{ {graph} {{ "
            + pattern(ids, group) + f"FILTER(?edge_id > {lit(after)}) "
            + f"}} }} ORDER BY ?edge_id LIMIT {limit}")
        return [SubstrateEdgeV1.model_validate_json(store._binding_str(row, "payload_json")) for row in rows]

    return read_neighborhood(request, source_kind=store._result_source_kind, nodes=nodes,
        groups=lambda ids: _groups(ids, request, predicates), edges=edges)
