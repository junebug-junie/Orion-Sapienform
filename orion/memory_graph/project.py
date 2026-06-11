from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import yaml
from rdflib import Graph, URIRef
from rdflib.namespace import RDF, RDFS

from orion.core.contracts.memory_cards import EDGE_TYPES, MemoryCardCreateV1, MemoryCardStatus, TimeHorizonV1
from orion.memory_graph.dto import (
    CardProjectionDefaultsV1,
    DispositionDraft,
    MemoryGraphSubschemaV1,
    SituationDraft,
    SuggestDraftV1,
)

from .json_to_rdf import ORIONMEM, SCHEMA


def _repo_root() -> Path:
    override = str(os.environ.get("ORION_REPO_ROOT") or "").strip()
    if override:
        return Path(override)
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


def _ref_from_uri(uri: URIRef, *, entity_base: str) -> Optional[str]:
    s = str(uri)
    base = entity_base.rstrip("/") + "/"
    if not s.startswith(base):
        return None
    tail = s[len(base) :]
    if not tail:
        return None
    return f"urn:uuid:{tail}"


def _situation_for_uri(uri: URIRef, draft: SuggestDraftV1, *, entity_base: str) -> Optional[SituationDraft]:
    ref = _ref_from_uri(uri, entity_base=entity_base)
    if not ref:
        return None
    for sit in draft.situations:
        if sit.id == ref:
            return sit
    return None


def _disposition_for_uri(uri: URIRef, draft: SuggestDraftV1, *, entity_base: str) -> Optional[DispositionDraft]:
    ref = _ref_from_uri(uri, entity_base=entity_base)
    if not ref:
        return None
    for disp in draft.dispositions:
        if disp.id == ref:
            return disp
    return None


def _situation_participant_labels(sit: SituationDraft, draft: SuggestDraftV1) -> List[str]:
    labels: List[str] = []
    seen: set[str] = set()
    for pid in [sit.stimulus_entity_id, *sit.about_entity_ids, *sit.target_entity_ids]:
        if not pid:
            continue
        label = _entity_label_by_id(draft, pid)
        if label and label not in seen:
            seen.add(label)
            labels.append(label)
    for part in sit.participants:
        label = _entity_label_by_id(draft, part.entity_id)
        if label and label not in seen:
            seen.add(label)
            labels.append(label)
    return labels


def _situation_still_true(sit: SituationDraft, draft: SuggestDraftV1) -> List[str]:
    lines: List[str] = []
    if sit.label:
        lines.append(sit.label.strip())
    stim = _entity_label_by_id(draft, sit.stimulus_entity_id)
    about = [_entity_label_by_id(draft, x) for x in sit.about_entity_ids]
    about = [a for a in about if a]
    if stim and about:
        lines.append(f"Involves {stim}; about {', '.join(about)}.")
    for part in sit.participants:
        label = _entity_label_by_id(draft, part.entity_id)
        if label:
            lines.append(f"{label} ({part.role})")
    return lines[:12]


def _situation_summary_text(sit: SituationDraft, draft: SuggestDraftV1) -> str:
    parts: List[str] = []
    if sit.affectLabel:
        parts.append(f"Affect: {sit.affectLabel}.")
    if sit.timeQualitative:
        parts.append(f"Time: {sit.timeQualitative}.")
    participants = ", ".join(_situation_participant_labels(sit, draft))
    if participants:
        parts.append(f"Participants: {participants}.")
    return " ".join(parts) if parts else (sit.label or "")


def _time_horizon_from_situation(sit: SituationDraft) -> Optional[TimeHorizonV1]:
    if sit.occurredAt:
        return TimeHorizonV1(kind="era_bound", start=sit.occurredAt, end=None, as_of=None)
    tq = (sit.timeQualitative or "").strip().lower()
    if tq in ("recent", "current", "now", "today"):
        return TimeHorizonV1(kind="current", start=None, end=None, as_of=None)
    if tq:
        return TimeHorizonV1(kind="era_bound", start=sit.timeQualitative, end=None, as_of=None)
    return TimeHorizonV1(kind="timeless", start=None, end=None, as_of=None)


def _situation_evidence(sit: SituationDraft, draft: SuggestDraftV1) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for uid in sit.utterance_ids:
        excerpt = (draft.utterance_text_by_id or {}).get(uid, "")
        out.append({"source": uid, "excerpt": excerpt or None, "ts": None})
    return out


def _situation_tags(sit: SituationDraft, draft: SuggestDraftV1) -> List[str]:
    tags: List[str] = []
    if sit.affectLabel:
        tags.append(f"affect:{sit.affectLabel}")
    for part in sit.participants:
        ek = next((e.entityKind for e in draft.entities if e.id == part.entity_id), None)
        if ek:
            tags.append(f"entity:{ek}")
    return tags


def apply_card_projection_defaults(
    creates: List[MemoryCardCreateV1],
    defaults: Optional[CardProjectionDefaultsV1],
) -> List[MemoryCardCreateV1]:
    """Merge operator-provided card metadata onto projected creates."""
    if not defaults:
        return creates
    out: List[MemoryCardCreateV1] = []
    for card in creates:
        row = card.model_dump(mode="json")
        if defaults.confidence is not None:
            row["confidence"] = defaults.confidence
        if defaults.sensitivity is not None:
            row["sensitivity"] = defaults.sensitivity
        if defaults.priority is not None:
            row["priority"] = defaults.priority
        if defaults.provenance is not None:
            row["provenance"] = defaults.provenance
        if defaults.visibility_scope is not None:
            row["visibility_scope"] = defaults.visibility_scope
        if defaults.time_horizon is not None:
            row["time_horizon"] = defaults.time_horizon.model_dump(mode="json")
        if defaults.summary and str(defaults.summary).strip():
            row["summary"] = str(defaults.summary).strip()
        if defaults.still_true:
            row["still_true"] = list(defaults.still_true)
        if defaults.evidence:
            row["evidence"] = [e.model_dump(mode="json") for e in defaults.evidence]
        out.append(MemoryCardCreateV1.model_validate(row))
    return out


def build_memory_graph_subschema(
    draft: SuggestDraftV1,
    *,
    named_graphs: Sequence[str],
    situation_entity_base: str,
    situation: Optional[SituationDraft] = None,
) -> Dict[str, Any]:
    """Appendix D-style projection from draft (facts derived from entities + situations + dispositions)."""
    sit = situation or (draft.situations[0] if draft.situations else None)
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
    if situation is None:
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
        utterance_ids=list(sit.utterance_ids) if sit else list(draft.utterance_ids),
        facts=facts,
        entity_refs=entity_refs,
    )
    return {"memory_graph": inner.model_dump(mode="json")}


def _situation_index_for_disposition(disp: DispositionDraft, draft: SuggestDraftV1) -> Optional[int]:
    for i, sit in enumerate(draft.situations):
        participant_ids = {p.entity_id for p in sit.participants}
        if disp.holder_id in participant_ids or disp.target_id in participant_ids:
            return i
    return 0 if draft.situations else None


def _graph_approve_status(mapping: Dict[str, Any]) -> MemoryCardStatus:
    raw = str(mapping.get("graph_approve_card_status") or "active").strip()
    if raw in ("pending_review", "active", "rejected", "superseded", "archived", "deprecated"):
        return cast(MemoryCardStatus, raw)
    return "active"


@dataclass(frozen=True)
class ProjectionPack:
    creates: List[MemoryCardCreateV1]
    """Edges as (from_idx, to_idx, edge_type, metadata)."""

    edge_indices: List[Tuple[int, int, EDGE_TYPES, Dict[str, Any]]]


def _collect_rdf_edges(
    g: Graph,
    node_index: Dict[str, int],
    mapping: Dict[str, Any],
) -> List[Tuple[int, int, EDGE_TYPES, Dict[str, Any]]]:
    pred_map: Dict[str, str] = mapping.get("predicate_to_edge_type") or {}
    edge_indices: List[Tuple[int, int, EDGE_TYPES, Dict[str, Any]]] = []
    allowed = {
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
    }
    for s, p, o in g:
        if not isinstance(s, URIRef) or not isinstance(o, URIRef):
            continue
        ps = str(p)
        if ps not in pred_map:
            continue
        et = pred_map[ps]
        if et not in allowed:
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
    return edge_indices


def _project_situation_centric(
    g: Graph,
    draft: SuggestDraftV1,
    *,
    mapping: Dict[str, Any],
    named_graphs: Sequence[str],
) -> ProjectionPack:
    """One card per situation (+ disposition beliefs); entities become anchors/tags, not orphan cards."""
    entity_base = str(mapping.get("entity_iri_base") or "https://orion.example/ns/mem/entity/")
    ng = list(named_graphs or ["https://orion.example/ns/memory/ng/session/local"])
    card_status = _graph_approve_status(mapping)

    nodes_order: List[URIRef] = []
    node_index: Dict[str, int] = {}
    situation_uri_by_draft_index: Dict[int, URIRef] = {}

    def register(uri: URIRef) -> int:
        key = str(uri)
        if key not in node_index:
            node_index[key] = len(nodes_order)
            nodes_order.append(uri)
        return node_index[key]

    situation_nodes = sorted({s for s in g.subjects(RDF.type, ORIONMEM.Situation)}, key=str)
    for n in situation_nodes:
        register(n)
        sit = _situation_for_uri(n, draft, entity_base=entity_base)
        if sit is not None:
            situation_uri_by_draft_index[draft.situations.index(sit)] = n

    disposition_nodes = sorted({s for s in g.subjects(RDF.type, ORIONMEM.AffectiveDisposition)}, key=str)
    for n in disposition_nodes:
        register(n)

    creates: List[MemoryCardCreateV1] = []
    for uri in nodes_order:
        ct = _card_types_for(uri, g, mapping)
        title = _literal_one(g, uri, RDFS.label) or str(uri)

        if ORIONMEM.Situation in list(g.objects(uri, RDF.type)):
            sit = _situation_for_uri(uri, draft, entity_base=entity_base)
            if sit is None:
                qual = _literal_one(g, uri, ORIONMEM.timeQualitative)
                aff = _literal_one(g, uri, ORIONMEM.affectLabel)
                summary_bits = []
                if aff:
                    summary_bits.append(f"Affect: {aff}.")
                if qual:
                    summary_bits.append(f"Time: {qual}.")
                creates.append(
                    MemoryCardCreateV1(
                        types=ct,
                        title=title,
                        summary=" ".join(summary_bits) if summary_bits else title,
                        provenance="operator_highlight",
                        status=card_status,
                        confidence="likely",
                        priority="high_recall",
                        subschema=build_memory_graph_subschema(
                            draft, named_graphs=ng, situation_entity_base=entity_base
                        ),
                    )
                )
                continue
            anchors = _situation_participant_labels(sit, draft)
            creates.append(
                MemoryCardCreateV1(
                    types=ct,
                    title=title,
                    summary=_situation_summary_text(sit, draft),
                    provenance="operator_highlight",
                    status=card_status,
                    confidence="likely",
                    priority="high_recall",
                    still_true=_situation_still_true(sit, draft),
                    anchors=anchors or None,
                    tags=_situation_tags(sit, draft) or None,
                    time_horizon=_time_horizon_from_situation(sit),
                    evidence=_situation_evidence(sit, draft),
                    subschema=build_memory_graph_subschema(
                        draft, named_graphs=ng, situation_entity_base=entity_base, situation=sit
                    ),
                )
            )
            continue

        if ORIONMEM.AffectiveDisposition in list(g.objects(uri, RDF.type)):
            disp = _disposition_for_uri(uri, draft, entity_base=entity_base)
            if disp is None:
                tp = _literal_one(g, uri, ORIONMEM.trustPolarity)
                desc = _literal_one(g, uri, SCHEMA.description)
                creates.append(
                    MemoryCardCreateV1(
                        types=ct,
                        title=title,
                        summary=desc or (f"Disposition ({tp})."),
                        provenance="operator_highlight",
                        status=card_status,
                        confidence="likely",
                        evidence=[],
                        subschema={},
                    )
                )
                continue
            holder = _entity_label_by_id(draft, disp.holder_id)
            target = _entity_label_by_id(draft, disp.target_id)
            desc = (disp.description or _literal_one(g, uri, SCHEMA.description) or "").strip()
            card_title = desc or f"{holder} toward {target} ({disp.trustPolarity})"
            summary = desc or f"Trust disposition: {holder} toward {target} ({disp.trustPolarity})."
            anchors = [x for x in [holder, target] if x]
            evidence: List[Dict[str, Any]] = []
            sit_idx = _situation_index_for_disposition(disp, draft)
            if sit_idx is not None and draft.situations[sit_idx].utterance_ids:
                uid = draft.situations[sit_idx].utterance_ids[0]
                evidence.append(
                    {
                        "source": uid,
                        "excerpt": (draft.utterance_text_by_id or {}).get(uid, "") or None,
                        "ts": None,
                    }
                )
            creates.append(
                MemoryCardCreateV1(
                    types=ct,
                    title=card_title[:500],
                    summary=summary[:2000],
                    provenance="operator_highlight",
                    status=card_status,
                    confidence="likely",
                    priority="episodic_detail",
                    anchors=anchors or None,
                    tags=[f"trust:{disp.trustPolarity}"],
                    evidence=evidence,
                    subschema={},
                )
            )
            continue

    edge_indices = _collect_rdf_edges(g, node_index, mapping)

    for disp in draft.dispositions:
        sit_idx = _situation_index_for_disposition(disp, draft)
        if sit_idx is None or sit_idx not in situation_uri_by_draft_index:
            continue
        disp_uri: Optional[URIRef] = None
        if disp.id:
            for n in disposition_nodes:
                if _ref_from_uri(n, entity_base=entity_base) == disp.id:
                    disp_uri = n
                    break
        if disp_uri is None:
            continue
        di = node_index.get(str(disp_uri))
        si = node_index.get(str(situation_uri_by_draft_index[sit_idx]))
        if di is None or si is None:
            continue
        edge_indices.append((di, si, "co_occurs_with", {"source": "memory_graph_disposition"}))

    return ProjectionPack(creates=creates, edge_indices=edge_indices)


def _project_entity_per_card(
    g: Graph,
    draft: SuggestDraftV1,
    *,
    mapping: Dict[str, Any],
    named_graphs: Sequence[str],
) -> ProjectionPack:
    """Legacy: one card per typed entity, situation, and disposition."""
    entity_base = str(mapping.get("entity_iri_base") or "https://orion.example/ns/mem/entity/")
    ng = list(named_graphs or ["https://orion.example/ns/memory/ng/session/local"])
    card_status = _graph_approve_status(mapping)

    nodes_order: List[URIRef] = []
    node_index: Dict[str, int] = {}

    def register(uri: URIRef) -> int:
        key = str(uri)
        if key not in node_index:
            node_index[key] = len(nodes_order)
            nodes_order.append(uri)
        return node_index[key]

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
                    status=card_status,
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
                    status=card_status,
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
                    status=card_status,
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
                status=card_status,
                subschema={},
            )
        )

    edge_indices = _collect_rdf_edges(g, node_index, mapping)
    return ProjectionPack(creates=creates, edge_indices=edge_indices)


def project_graph_to_cards(
    g: Graph,
    draft: SuggestDraftV1,
    *,
    mapping: Optional[Dict[str, Any]] = None,
    named_graphs: Optional[Sequence[str]] = None,
    card_defaults: Optional[CardProjectionDefaultsV1] = None,
) -> ProjectionPack:
    """Map RDF graph + original draft to MemoryCardCreateV1 list + edge index tuples."""
    mapping = mapping or load_mapping()
    ng = list(named_graphs or ["https://orion.example/ns/memory/ng/session/local"])
    mode = str(mapping.get("projection_mode") or "situation_centric").strip()
    if mode == "entity_per_card":
        pack = _project_entity_per_card(g, draft, mapping=mapping, named_graphs=ng)
    else:
        pack = _project_situation_centric(g, draft, mapping=mapping, named_graphs=ng)
    if card_defaults:
        pack = ProjectionPack(
            creates=apply_card_projection_defaults(pack.creates, card_defaults),
            edge_indices=pack.edge_indices,
        )
    return pack
