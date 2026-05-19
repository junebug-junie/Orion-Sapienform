"""JSON Schema contracts for memory-graph suggest (llama.cpp-friendly compact form)."""

from __future__ import annotations

from typing import Any, Dict

from orion.memory_graph.dto import SuggestDraftV1


def suggest_draft_json_schema() -> Dict[str, Any]:
    """Full Pydantic JSON schema (may include $defs / $ref)."""
    return SuggestDraftV1.model_json_schema()


def compact_schema_for_llamacpp(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Strip Pydantic noise; keep inline object shapes only."""
    _ = schema  # reserved for future $ref inlining
    return compact_suggest_draft_json_schema()


def compact_suggest_draft_json_schema() -> Dict[str, Any]:
    """Stage-1 schema: top-level keys + array item shapes without $ref/oneOf."""
    participant = {
        "type": "object",
        "additionalProperties": False,
        "required": ["entity_id", "role"],
        "properties": {
            "entity_id": {"type": "string"},
            "role": {"type": "string"},
        },
    }
    entity = {
        "type": "object",
        "additionalProperties": False,
        "required": ["id", "label", "entityKind", "surfaceForms"],
        "properties": {
            "id": {"type": "string"},
            "label": {"type": "string"},
            "entityKind": {"type": "string"},
            "surfaceForms": {"type": "array", "items": {"type": "string"}},
            "generalizes_to": {"type": "string"},
        },
    }
    situation = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "id",
            "utterance_ids",
            "label",
            "stimulus_entity_id",
            "about_entity_ids",
            "target_entity_ids",
            "affectLabel",
            "timeQualitative",
            "participants",
        ],
        "properties": {
            "id": {"type": "string"},
            "utterance_ids": {"type": "array", "items": {"type": "string"}},
            "label": {"type": "string"},
            "stimulus_entity_id": {"type": "string"},
            "about_entity_ids": {"type": "array", "items": {"type": "string"}},
            "target_entity_ids": {"type": "array", "items": {"type": "string"}},
            "affectLabel": {"type": "string"},
            "timeQualitative": {"type": "string"},
            "occurredAt": {"type": "string"},
            "participants": {"type": "array", "items": participant},
        },
    }
    edge = {
        "type": "object",
        "additionalProperties": False,
        "required": ["s", "p", "o", "confidence"],
        "properties": {
            "s": {"type": "string"},
            "p": {"type": "string"},
            "o": {"type": "string"},
            "confidence": {"type": "number"},
        },
    }
    disposition = {
        "type": "object",
        "additionalProperties": False,
        "required": ["holder_id", "target_id", "trustPolarity", "description"],
        "properties": {
            "id": {"type": "string"},
            "holder_id": {"type": "string"},
            "target_id": {"type": "string"},
            "trustPolarity": {"type": "string"},
            "description": {"type": "string"},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "ontology_version",
            "utterance_ids",
            "entities",
            "situations",
            "edges",
            "dispositions",
        ],
        "properties": {
            "ontology_version": {"type": "string", "enum": ["orionmem-2026-05"]},
            "utterance_ids": {"type": "array", "items": {"type": "string"}},
            "utterance_text_by_id": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
            "entities": {"type": "array", "items": entity},
            "situations": {"type": "array", "items": situation},
            "edges": {"type": "array", "items": edge},
            "dispositions": {"type": "array", "items": disposition},
        },
    }
