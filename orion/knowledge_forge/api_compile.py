from __future__ import annotations

from datetime import datetime, timezone

from orion.knowledge_forge.models import (
    ClaimStatusV1,
    ClaimV1,
    DecisionStatusV1,
    DecisionV1,
    SpecStatusV1,
    SpecV1,
)
from orion.knowledge_forge.store import KnowledgeStore

_API_TARGET_TO_FOLDER = {
    "cursor": "cursor",
    "codex": "codex",
    "claude_code": "claude_code",
    "orion": "orion",
}


def compile_context_pack_api_v1(
    store: KnowledgeStore,
    *,
    task: str,
    target: str,
    spec_ids: list[str],
    claim_ids: list[str],
    include_disputed: bool = False,
    include_stale: bool = False,
) -> tuple[str, list[str]]:
    warnings: list[str] = []
    specs: list[SpecV1] = []
    for spec_id in spec_ids:
        doc = store.get(spec_id)
        if not isinstance(doc, SpecV1):
            warnings.append(f"spec not found: {spec_id}")
            continue
        if doc.status not in (SpecStatusV1.reviewed, SpecStatusV1.execution_ready):
            warnings.append(f"spec {spec_id} status {doc.status.value} is not reviewed or execution_ready")
        specs.append(doc)

    claim_id_set: set[str] = set(claim_ids)
    for spec in specs:
        claim_id_set.update(spec.source_claims)

    claims: list[ClaimV1] = []
    for claim_id in sorted(claim_id_set):
        doc = store.get(claim_id)
        if not isinstance(doc, ClaimV1):
            warnings.append(f"claim not found: {claim_id}")
            continue
        if not _claim_included(doc, include_disputed=include_disputed, include_stale=include_stale):
            continue
        claims.append(doc)

    decisions = _relevant_decisions(store, specs, claims)
    non_goals = _dedupe_preserve_order([ng for spec in specs for ng in spec.non_goals])
    traps = _dedupe_preserve_order([trap for spec in specs for trap in spec.known_traps])
    impl_paths = _dedupe_preserve_order([path for spec in specs for path in spec.likely_files])
    acceptance = _dedupe_preserve_order([test for spec in specs for test in spec.acceptance_tests])
    lineage = _source_lineage(store, claims, specs, decisions)

    lines = [
        f"# Context Pack ({target})",
        "",
        "## Target",
        target,
        "",
        "## Task",
        task,
        "",
        "## Current accepted facts",
    ]
    if claims:
        for claim in claims:
            lines.append(f"- {claim.statement} (`{claim.id}`, {claim.status.value})")
    else:
        lines.append("- _(none)_")

    lines.extend(["", "## Relevant specs"])
    if specs:
        for spec in specs:
            lines.append(f"### {spec.id} — {spec.component} ({spec.status.value})")
            for req in spec.requirements:
                lines.append(f"- {req}")
    else:
        lines.append("- _(none)_")

    lines.extend(["", "## Relevant decisions"])
    if decisions:
        for decision in decisions:
            lines.append(f"- **{decision.id}**: {decision.decision} ({decision.status.value})")
            if decision.rationale:
                lines.append(f"  - Rationale: {decision.rationale}")
    else:
        lines.append("- _(none)_")

    lines.extend(["", "## Non-goals"])
    if non_goals:
        for ng in non_goals:
            lines.append(f"- {ng}")
    else:
        lines.append("- _(none)_")

    lines.extend(["", "## Known traps"])
    if traps:
        for trap in traps:
            lines.append(f"- {trap}")
    else:
        lines.append("- _(none)_")

    lines.extend(["", "## Suggested implementation path"])
    if impl_paths:
        for path in impl_paths:
            lines.append(f"- `{path}`")
    else:
        lines.append("- _(none)_")

    lines.extend(["", "## Acceptance checks"])
    if acceptance:
        for check in acceptance:
            lines.append(f"- {check}")
    else:
        lines.append("- _(none)_")

    lines.extend(["", "## Source lineage"])
    if lineage:
        for entry in lineage:
            lines.append(f"- {entry}")
    else:
        lines.append("- _(none)_")
    lines.append("")

    return "\n".join(lines), warnings


def build_context_pack_output_path(
    corpus_root,
    *,
    target: str,
    task: str,
    timestamp: datetime | None = None,
) -> "Path":
    from pathlib import Path

    folder = _API_TARGET_TO_FOLDER.get(target, target)
    ts = (timestamp or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    slug = _slugify(task) or "context-pack"
    rel = Path("context_packs") / folder / f"{ts}-{slug}.md"
    return corpus_root / rel


def _claim_included(claim: ClaimV1, *, include_disputed: bool, include_stale: bool) -> bool:
    if claim.status == ClaimStatusV1.accepted:
        return True
    if claim.status == ClaimStatusV1.disputed:
        return include_disputed
    if claim.status == ClaimStatusV1.stale:
        return include_stale
    return False


def _relevant_decisions(
    store: KnowledgeStore,
    specs: list[SpecV1],
    claims: list[ClaimV1],
) -> list[DecisionV1]:
    claim_ids = {claim.id for claim in claims}
    spec_ids = {spec.id for spec in specs}
    out: list[DecisionV1] = []
    seen: set[str] = set()
    for doc in store.by_id.values():
        if not isinstance(doc, DecisionV1):
            continue
        if doc.status != DecisionStatusV1.accepted:
            continue
        related = bool(set(doc.source_claims) & claim_ids)
        related = related or bool(set(doc.implements) & spec_ids)
        related = related or bool(set(doc.motivated_by) & claim_ids)
        if not related:
            continue
        if doc.id in seen:
            continue
        seen.add(doc.id)
        out.append(doc)
    return sorted(out, key=lambda d: d.id)


def _source_lineage(
    store: KnowledgeStore,
    claims: list[ClaimV1],
    specs: list[SpecV1],
    decisions: list[DecisionV1],
) -> list[str]:
    entries: list[str] = []
    seen: set[str] = set()
    for claim in claims:
        for source_ref in claim.source_refs:
            if source_ref in seen:
                continue
            seen.add(source_ref)
            source = store.get(source_ref)
            if source is not None:
                entries.append(f"{source_ref} → {getattr(source, 'path', source_ref)}")
            else:
                entries.append(source_ref)
    for spec in specs:
        if spec.id not in seen:
            seen.add(spec.id)
            entries.append(f"{spec.id} ({spec.component})")
    for decision in decisions:
        if decision.id not in seen:
            seen.add(decision.id)
            entries.append(f"{decision.id} ({decision.status.value})")
    return entries


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _slugify(text: str, *, max_len: int = 48) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in text.strip())
    while "--" in slug:
        slug = slug.replace("--", "-")
    slug = slug.strip("-")
    return slug[:max_len].strip("-")
