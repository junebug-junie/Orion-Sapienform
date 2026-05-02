from __future__ import annotations

import re
from typing import Dict, Mapping, Optional

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, XSD

from orion.memory_graph.dto import SuggestDraftV1

ORIONMEM = Namespace("https://orion.local/ns/mem/v2026-05#")
PROV = Namespace("http://www.w3.org/ns/prov#")
SCHEMA = Namespace("https://schema.org/")


class _Prefixes:
    """§4 CURIE prefixes for edge `p`."""

    MAP: Dict[str, str] = {
        "orionmem": str(ORIONMEM),
        "prov": str(PROV),
        "schema": str(SCHEMA),
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "owl": "http://www.w3.org/2002/07/owl#",
        "xsd": "http://www.w3.org/2001/XMLSchema#",
    }


def expand_curie(curie: str, *, extra: Optional[Mapping[str, str]] = None) -> str:
    if curie.startswith("http://") or curie.startswith("https://"):
        return curie
    if ":" not in curie:
        raise ValueError(f"not a CURIE or absolute IRI: {curie!r}")
    pfx, rest = curie.split(":", 1)
    base = dict(_Prefixes.MAP)
    if extra:
        base.update(extra)
    if pfx not in base:
        raise ValueError(f"unknown CURIE prefix {pfx!r} in {curie!r}")
    return base[pfx] + rest


_UUID_TAIL = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.I,
)


def entity_uri(ref: str, *, entity_base: str = "https://orion.example/ns/mem/entity/") -> URIRef:
    if ref.startswith("http://") or ref.startswith("https://"):
        return URIRef(ref)
    if ref.startswith("urn:uuid:"):
        tail = ref.removeprefix("urn:uuid:")
        if not _UUID_TAIL.match(tail):
            raise ValueError(f"bad urn:uuid reference {ref!r}")
        return URIRef(entity_base + tail)
    raise ValueError(f"unsupported entity ref {ref!r}")


def utterance_uri(turn_id: str, *, prefix: str = "https://orion.local/ns/mem/utterance/") -> URIRef:
    tid = str(turn_id).strip()
    if not tid:
        raise ValueError("empty turn id")
    return URIRef(prefix + tid)


def draft_to_graph(
    draft: SuggestDraftV1,
    *,
    entity_base: str = "https://orion.example/ns/mem/entity/",
    utterance_prefix: str = "https://orion.local/ns/mem/utterance/",
    revision_batch: Optional[str] = None,
) -> Graph:
    """Appendix C JSON → rdflib Graph (batch tagging optional until approve)."""
    g = Graph()
    g.bind("orionmem", ORIONMEM)
    g.bind("prov", PROV)
    g.bind("schema", SCHEMA)

    def eb(ref: str) -> URIRef:
        return entity_uri(ref, entity_base=entity_base)

    for uid in draft.utterance_ids:
        u = utterance_uri(uid, prefix=utterance_prefix)
        g.add((u, RDF.type, ORIONMEM.UtteranceSlice))
        g.add((u, RDFS.label, Literal(f"turn {uid}")))
        text = (draft.utterance_text_by_id or {}).get(uid)
        if text:
            g.add((u, SCHEMA.text, Literal(text)))
        if revision_batch:
            g.add((u, ORIONMEM.revisionBatch, Literal(revision_batch)))

    for ent in draft.entities:
        node = eb(ent.id)
        g.add((node, RDF.type, SCHEMA.Thing))
        g.add((node, RDF.type, ORIONMEM.TypedEntity))
        g.add((node, RDFS.label, Literal(ent.label)))
        g.add((node, ORIONMEM.entityKind, Literal(ent.entityKind)))
        if ent.generalizes_to:
            g.add((node, ORIONMEM.generalizationOf, eb(ent.generalizes_to)))
        if revision_batch:
            g.add((node, ORIONMEM.revisionBatch, Literal(revision_batch)))

    for sit in draft.situations:
        snode = eb(sit.id)
        g.add((snode, RDF.type, ORIONMEM.Situation))
        if sit.label:
            g.add((snode, RDFS.label, Literal(sit.label)))
        for uid in sit.utterance_ids:
            g.add((snode, PROV.wasDerivedFrom, utterance_uri(uid, prefix=utterance_prefix)))
        if sit.stimulus_entity_id:
            g.add((snode, ORIONMEM.stimulusEntity, eb(sit.stimulus_entity_id)))
        for ae in sit.about_entity_ids:
            g.add((snode, ORIONMEM.aboutEntity, eb(ae)))
        for ta in sit.target_of_negative_affect_ids:
            g.add((snode, ORIONMEM.targetOfNegativeAffect, eb(ta)))
        if sit.affectLabel:
            g.add((snode, ORIONMEM.affectLabel, Literal(sit.affectLabel)))
        if sit.timeQualitative:
            g.add((snode, ORIONMEM.timeQualitative, Literal(sit.timeQualitative)))
        if sit.occurredAt:
            g.add((snode, ORIONMEM.occurredAt, Literal(sit.occurredAt, datatype=XSD.date)))
        for part in sit.participants:
            en = eb(part.entity_id)
            g.add((en, ORIONMEM.inSituation, snode))
            g.add((en, ORIONMEM.participantRole, Literal(part.role)))
        if revision_batch:
            g.add((snode, ORIONMEM.revisionBatch, Literal(revision_batch)))

    for disp in draft.dispositions:
        dnode = eb(disp.id)
        g.add((dnode, RDF.type, ORIONMEM.AffectiveDisposition))
        g.add((dnode, ORIONMEM.trustPolarity, Literal(disp.trustPolarity)))
        g.add((dnode, ORIONMEM.dispositionTarget, eb(disp.target_id)))
        if draft.utterance_ids:
            g.add((dnode, PROV.wasDerivedFrom, utterance_uri(draft.utterance_ids[0], prefix=utterance_prefix)))
        if disp.description:
            g.add((dnode, SCHEMA.description, Literal(disp.description)))
        g.add((eb(disp.holder_id), ORIONMEM.dispositionToward, dnode))
        if revision_batch:
            g.add((dnode, ORIONMEM.revisionBatch, Literal(revision_batch)))

    for e in draft.edges:
        pred = expand_curie(e.p)
        g.add((eb(e.s), URIRef(pred), eb(e.o)))
        if e.confidence is not None:
            g.add((eb(e.s), ORIONMEM.inferenceConfidence, Literal(float(e.confidence))))

    return g
