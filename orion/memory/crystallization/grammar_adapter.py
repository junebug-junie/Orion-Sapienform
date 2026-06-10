from __future__ import annotations

from typing import Any

from orion.memory.crystallization.schemas import MemoryGrammarEnvelopeV1, MemoryCrystallizationV1


def envelope_from_grammar_event(event: dict[str, Any]) -> MemoryGrammarEnvelopeV1:
    """Build memory grammar envelope from existing GrammarEventV1-shaped dict."""
    atoms = event.get("atoms") or []
    edges = event.get("edges") or []
    atom_ids = [str(a.get("atom_id")) for a in atoms if isinstance(a, dict) and a.get("atom_id")]
    edge_ids = [str(e.get("edge_id")) for e in edges if isinstance(e, dict) and e.get("edge_id")]
    atom_keys = [str(a.get("key")) for a in atoms if isinstance(a, dict) and a.get("key")]

    return MemoryGrammarEnvelopeV1(
        source_grammar_event_ids=[str(event.get("event_id") or "")] if event.get("event_id") else [],
        source_atom_ids=atom_ids,
        source_edge_ids=edge_ids,
        shape=str(event.get("shape") or "") or None,
        atom_keys=atom_keys,
        dimensions=list(event.get("dimensions") or []),
        planning_effects=list(event.get("planning_effects") or []),
        retrieval_affordances=list(event.get("retrieval_affordances") or []),
    )


def attach_grammar_to_crystallization(
    crystallization: MemoryCrystallizationV1,
    *,
    grammar_events: list[dict[str, Any]],
) -> MemoryCrystallizationV1:
    updated = crystallization.model_copy(deep=True)
    if not grammar_events:
        return updated

    merged = MemoryGrammarEnvelopeV1()
    for ev in grammar_events:
        env = envelope_from_grammar_event(ev)
        merged.source_grammar_event_ids.extend(env.source_grammar_event_ids)
        merged.source_atom_ids.extend(env.source_atom_ids)
        merged.source_edge_ids.extend(env.source_edge_ids)
        merged.atom_keys.extend(env.atom_keys)
        merged.dimensions.extend(env.dimensions)
        merged.planning_effects.extend(env.planning_effects)
        merged.retrieval_affordances.extend(env.retrieval_affordances)

    updated.grammar_envelope = merged
    updated.source_grammar_event_ids = list(dict.fromkeys(merged.source_grammar_event_ids))
    updated.source_atom_ids = list(dict.fromkeys(merged.source_atom_ids))
    return updated
