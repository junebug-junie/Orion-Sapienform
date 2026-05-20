from __future__ import annotations

from orion.knowledge_forge.models import ClaimStatusV1, ClaimV1, SpecV1, SpecStatusV1
from orion.knowledge_forge.store import KnowledgeStore


def compile_context_pack_markdown(store: KnowledgeStore, *, spec_id: str, task: str) -> str:
    spec = store.get(spec_id)
    if not isinstance(spec, SpecV1):
        raise ValueError(f"spec not found: {spec_id}")
    if spec.status not in (SpecStatusV1.reviewed, SpecStatusV1.execution_ready):
        raise ValueError(f"spec {spec_id} must be reviewed or execution_ready")

    claims = _accepted_claims_for_spec(store, spec)
    lines = [
        f"# Context Pack: {spec.component}",
        "",
        "## Goal",
        task,
        "",
        "## Current accepted facts",
    ]
    for claim in claims:
        lines.append(f"- {claim.statement} (`{claim.id}`)")
    lines.extend(["", "## Required behavior"])
    for req in spec.requirements:
        lines.append(f"- {req}")
    lines.extend(["", "## Non-goals"])
    for ng in spec.non_goals:
        lines.append(f"- {ng}")
    lines.extend(["", "## Relevant files"])
    for path in spec.likely_files:
        lines.append(f"- `{path}`")
    lines.extend(["", "## Acceptance tests"])
    for test in spec.acceptance_tests:
        lines.append(f"- {test}")
    lines.extend(["", "## Known traps"])
    for trap in spec.known_traps:
        lines.append(f"- {trap}")
    lines.append("")
    return "\n".join(lines)


def _accepted_claims_for_spec(store: KnowledgeStore, spec: SpecV1) -> list[ClaimV1]:
    out: list[ClaimV1] = []
    for claim_id in spec.source_claims:
        doc = store.get(claim_id)
        if not isinstance(doc, ClaimV1):
            continue
        if doc.status != ClaimStatusV1.accepted:
            continue
        out.append(doc)
    return out
