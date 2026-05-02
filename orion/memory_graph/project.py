from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import yaml
from rdflib import Graph, URIRef
from rdflib.namespace import RDF, RDFS

from orion.core.contracts.memory_cards import EDGE_TYPES, MemoryCardCreateV1, TimeHorizonV1
from orion.memory_graph.dto import MemoryGraphSubschemaV1, SuggestDraftV1

from .json_to_rdf import ORIONMEM, SCHEMA


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_mapping_path() -> Path:
    return _repo_root() / "config" / "memory_graph" / "projector_mapping.yaml"


def load_mapping(path: Optional[Path] = None) -> Dict[str, Any]:
    p = path or _default_mapping_path()
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw


def _literal_one(g: Graph, s: URIRef, p: URIRef) -> Optional[str]:
    for o in g.objects(s, p):
        return str(o)
    return None


def _card_types_for(node: URIRef, g: Graph, mapping: Dict[str, Any]) -> List[str]:
    types_map: Dict[str, Any] = mapping.get("rdf_type_to_card_types") or {}
    for t in g.objects(node, RDF.type):
        key = str(t)
        if key in types_map:
            val = types_map[key]
            return list(val) if isinstance(val, list) else [str(val)]
    return ["anchor"]


def _anchor_class_for_entity(g: Graph, node: URIRef, mapping: Dict[str, Any]) -> Optional[str]:
    kinds: Dict[str, str] = mapping.get("entity_kind_to_anchor_class") or {}
    ek = _literal_one(g, node, ORIONMEM.entityKind)
    if ek and ek in kinds:
        return kinds[ek]
    return "concept"


def _entity_label_by_id(draft: SuggestDraftV1, entity_id: str | None) -> str:
    if not entity_id:
        return ""
    for e in draft.entities:
        if e.id == entity_id:
            return str(e.label or "").strip()
    return ""


def build_memory_graph_subschema(
    draft: SuggestDraftV1,
    *,
    named_graphs: Sequence[str],
    situation_entity_base: str,
) -> Dict[str, Any]:
    """Appendix D-style projection from draft (facts derived from entities + situations + dispositions)."""
    sit = draft.situations[0] if draft.situations else None
    facts: List[Dict[str, Any]] = []
    if sit:
        stim = _entity_label_by_id(draft, sit.stimulus_entity_id)
        patient_pid = next((p.entity_id for p in sit.participants if p.role == "patient"), None)
        patient_l = _entity_label_by_id(draft, patient_pid)
        pred = (sit.affectLabel or "situation").strip()
        if stim and patient_l:
            row: Dict[str, Any] = {
                "subject": stim,
                "predicate": pred,
                "object": patient_l,
                "source": "situation",
            }
            if sit.timeQualitative:
                row["time"] = sit.timeQualitative
            facts.append(row)
        elif stim or sit.label:
            row = {
                "subject": stim or (sit.label or "situation").split(";")[0].strip()[:120],
                "predicate": pred,
                "object": sit.label or "",
                "source": "situation",
            }
            if sit.timeQualitative:
                row["time"] = sit.timeQualitative
            facts.append(row)
    for disp in draft.dispositions:
        holder_l = next((e.label for e in draft.entities if e.id == disp.holder_id), "subject")
        tgt_l = next((e.label for e in draft.entities if e.id == disp.target_id), "object")
        facts.append(
            {
                "subject": holder_l,
                "predicate": "trust_toward",
                "object": tgt_l,
                "polarity": disp.trustPolarity,
            }
        )
    entity_refs: Dict[str, str] = {}
    for ent in draft.entities:
        iri = situation_entity_base.rstrip("/") + "/" + ent.id.removeprefix("urn:uuid:")
        key = ent.label.lower().replace(" ", "_")[:32]
        if ent.entityKind == "animal":
            entity_refs["joey"] = iri
        elif ent.entityKind == "breed":
            entity_refs["breed"] = iri
        else:
            entity_refs[key] = iri
    inner = MemoryGraphSubschemaV1(
        ontology_version=draft.ontology_version,
        named_graphs=list(named_graphs),
        situation_id=sit.id if sit else None,
        utterance_ids=list(draft.utterance_ids),
        facts=facts,
        entity_refs=entity_refs,
    )
    return {"memory_graph": inner.model_dump(mode="json")}


@dataclass(frozen=True)
class ProjectionPack:
    creates: List[MemoryCardCreateV1]
    """Edges as (from_idx, to_idx, edge_type, metadata)."""

    edge_indices: List[Tuple[int, int, EDGE_TYPES, Dict[str, Any]]]


def project_graph_to_cards(
    g: Graph,
    draft: SuggestDraftV1,
    *,
    mapping: Optional[Dict[str, Any]] = None,
    named_graphs: Optional[Sequence[str]] = None,
) -> ProjectionPack:
    """Map RDF graph + original draft to MemoryCardCreateV1 list + edge index tuples."""
    mapping = mapping or load_mapping()
    entity_base = str(mapping.get("entity_iri_base") or "https://orion.example/ns/mem/entity/")
    ng = list(named_graphs or ["https://orion.example/ns/memory/ng/session/local"])

    nodes_order: List[URIRef] = []
    node_index: Dict[str, int] = {}

    def register(uri: URIRef) -> int:
        key = str(uri)
        if key not in node_index:
            node_index[key] = len(nodes_order)
            nodes_order.append(uri)
        return node_index[key]

    # Typed entities first (sorted by label for stability)
    entity_nodes = sorted(
        {s for s in g.subjects(RDF.type, ORIONMEM.TypedEntity)},
        key=lambda u: (_literal_one(g, u, RDFS.label) or str(u)),
    )
    for n in entity_nodes:
        register(n)

    situation_nodes = sorted({s for s in g.subjects(RDF.type, ORIONMEM.Situation)}, key=str)
    for n in situation_nodes:
        register(n)

    disposition_nodes = sorted({s for s in g.subjects(RDF.type, ORIONMEM.AffectiveDisposition)}, key=str)
    for n in disposition_nodes:
        register(n)

    subschema_blob = build_memory_graph_subschema(draft, named_graphs=ng, situation_entity_base=entity_base)

    creates: List[MemoryCardCreateV1] = []
    for uri in nodes_order:
        ct = _card_types_for(uri, g, mapping)
        title = _literal_one(g, uri, RDFS.label) or str(uri)
        summary_bits: List[str] = []
        if ORIONMEM.TypedEntity in list(g.objects(uri, RDF.type)):
            ek = _literal_one(g, uri, ORIONMEM.entityKind)
            ac = _anchor_class_for_entity(g, uri, mapping)
            summary_bits.append(f"Memory graph entity ({ek or 'unknown'}).")
            sub: Dict[str, Any] = {}
            creates.append(
                MemoryCardCreateV1(
                    types=ct,
                    anchor_class=ac if "anchor" in ct else None,
                    title=title,
                    summary=" ".join(summary_bits),
                    provenance="operator_highlight",
                    evidence=[
                        {
                            "source": draft.utterance_ids[0] if draft.utterance_ids else "memory_graph",
                            "excerpt": (draft.utterance_text_by_id or {}).get(draft.utterance_ids[0], ""),
                            "ts": None,
                        }
                    ]
                    if draft.utterance_ids
                    else [],
                    subschema=sub,
                )
            )
            continue

        if ORIONMEM.Situation in list(g.objects(uri, RDF.type)):
            qual = _literal_one(g, uri, ORIONMEM.timeQualitative)
            aff = _literal_one(g, uri, ORIONMEM.affectLabel)
            od = _literal_one(g, uri, ORIONMEM.occurredAt)
            if aff:
                summary_bits.append(f"Affect: {aff}.")
            if qual:
                summary_bits.append(f"Time: {qual}.")
            th: Optional[TimeHorizonV1] = None
            if od:
                th = TimeHorizonV1(kind="era_bound", start=od, end=None, as_of=None)
            creates.append(
                MemoryCardCreateV1(
                    types=ct,
                    title=title,
                    summary=" ".join(summary_bits) if summary_bits else title,
                    provenance="operator_highlight",
                    time_horizon=th,
                    evidence=[
                        {
                            "source": draft.utterance_ids[0],
                            "excerpt": (draft.utterance_text_by_id or {}).get(draft.utterance_ids[0], ""),
                            "ts": None,
                        }
                    ]
                    if draft.utterance_ids
                    else [],
                    subschema=subschema_blob,
                )
            )
            continue

        if ORIONMEM.AffectiveDisposition in list(g.objects(uri, RDF.type)):
            tp = _literal_one(g, uri, ORIONMEM.trustPolarity)
            desc = _literal_one(g, uri, SCHEMA.description)
            summary_bits.append(desc or (f"Disposition ({tp})."))
            creates.append(
                MemoryCardCreateV1(
                    types=ct,
                    title=title,
                    summary=" ".join(summary_bits),
                    provenance="operator_highlight",
                    evidence=[],
                    subschema={},
                )
            )
            continue

        creates.append(
            MemoryCardCreateV1(
                types=["anchor"],
                title=title,
                summary="Memory graph node.",
                provenance="operator_highlight",
                subschema={},
            )
        )

    pred_map: Dict[str, str] = mapping.get("predicate_to_edge_type") or {}
    edge_indices: List[Tuple[int, int, EDGE_TYPES, Dict[str, Any]]] = []
    for s, p, o in g:
        if not isinstance(s, URIRef) or not isinstance(o, URIRef):
            continue
        ps = str(p)
        if ps not in pred_map:
            continue
        et = pred_map[ps]
        if et not in (
            "relates_to",
            "contradicts",
            "supersedes",
            "supports",
            "parent_of",
            "child_of",
            "precedes",
            "follows",
            "co_occurs_with",
            "derived_from",
            "evidence_for",
            "evidence_against",
            "tagged_as",
            "instance_of",
            "example_of",
            "analogy_of",
            "associated_with",
        ):
            continue
        si = node_index.get(str(s))
        oi = node_index.get(str(o))
        if si is None or oi is None:
            continue
        meta: Dict[str, Any] = {"predicate": ps}
        conf = _literal_one(g, s, ORIONMEM.inferenceConfidence)
        if conf:
            try:
                meta["inferenceConfidence"] = float(conf)
            except ValueError:
                pass
        et_typed = cast(EDGE_TYPES, et)
        if et == "parent_of":
            edge_indices.append((oi, si, et_typed, meta))
        elif et == "child_of":
            edge_indices.append((si, oi, et_typed, meta))
        else:
            edge_indices.append((si, oi, et_typed, meta))

    return ProjectionPack(creates=creates, edge_indices=edge_indices)
