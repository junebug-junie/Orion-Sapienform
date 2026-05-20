from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from orion.knowledge_forge.models import SourceKindV1, SpecV1, TrustLevelV1
from orion.knowledge_forge.store import KnowledgeStore
from orion.knowledge_forge.yaml_doc import save_yaml_doc

_CLAIM_SECTIONS = frozenset(
    {
        "requirements",
        "decisions",
        "non-goals",
        "acceptance checks",
        "known traps",
        "implementation path",
    }
)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
_BULLET_RE = re.compile(r"^\s*[-*]\s+(.+)$")
_FORBIDDEN_WRITE_PREFIXES = (
    "claims/accepted/",
    "specs/execution_ready/",
    "decisions/",
)


@dataclass
class ProposedClaim:
    text: str
    provenance: str
    line_ref: int | None = None


@dataclass
class SourceParseResult:
    title: str
    summary_lines: list[str]
    proposed_claims: list[ProposedClaim] = field(default_factory=list)
    section_bullets: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class IngestSourceResult:
    source_id: str
    status: str
    source_path: str | None
    review_path: str | None
    proposed_claims: list[str]
    possibly_affected_specs: list[str]
    warnings: list[str]
    content: str


def slug_from_source_id(source_id: str) -> str:
    slug = source_id.removeprefix("source:").strip()
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", slug).strip("-")
    return slug or "source"


def parse_source_markdown(text: str) -> SourceParseResult:
    lines = text.splitlines()
    title = _infer_title(lines)
    section_bullets: dict[str, list[str]] = {}
    current_section: str | None = None

    for idx, line in enumerate(lines, start=1):
        heading = _HEADING_RE.match(line)
        if heading:
            current_section = heading.group(2).strip()
            section_bullets.setdefault(current_section, [])
            continue
        bullet = _BULLET_RE.match(line)
        if bullet and current_section is not None:
            section_bullets.setdefault(current_section, []).append(bullet.group(1).strip())

    proposed: list[ProposedClaim] = []
    for section, bullets in section_bullets.items():
        if section.casefold() not in _CLAIM_SECTIONS:
            continue
        for bullet_idx, bullet in enumerate(bullets):
            line_ref = _bullet_line_number(lines, section, bullet_idx)
            provenance = f"{section} § {bullet[:80]}"
            if line_ref is not None:
                provenance = f"L{line_ref} · {provenance}"
            proposed.append(ProposedClaim(text=bullet, provenance=provenance, line_ref=line_ref))

    summary_lines = section_bullets.get("Requirements", [])[:3]
    if not summary_lines:
        for bullets in section_bullets.values():
            summary_lines.extend(bullets[:2])
            if summary_lines:
                break

    return SourceParseResult(
        title=title,
        summary_lines=summary_lines,
        proposed_claims=proposed,
        section_bullets=section_bullets,
    )


def build_source_delta_review(
    *,
    source_id: str,
    source_kind: str,
    source_rel_path: str,
    parsed: SourceParseResult,
    possibly_affected_specs: list[str],
) -> str:
    lines = [
        "# Source Delta Review",
        "",
        "## Source",
        f"- id: `{source_id}`",
        f"- kind: `{source_kind}`",
        f"- path: `{source_rel_path}`",
        "",
        "## Source summary",
        f"**{parsed.title}**",
        "",
    ]
    if parsed.summary_lines:
        for item in parsed.summary_lines:
            lines.append(f"- {item}")
    else:
        lines.append("- (no summary bullets extracted)")

    lines.extend(["", "## Proposed claims"])
    if parsed.proposed_claims:
        for claim in parsed.proposed_claims:
            suffix = f" _({claim.provenance})_"
            lines.append(f"- [ ] {claim.text}{suffix}")
    else:
        lines.append("- [ ] (no claim candidates extracted)")

    lines.extend(["", "## Possibly affected specs"])
    if possibly_affected_specs:
        for spec_id in possibly_affected_specs:
            lines.append(f"- `{spec_id}`")
    else:
        lines.append("- (none identified)")

    lines.extend(["", "## Suggested context packs"])
    if possibly_affected_specs:
        for spec_id in possibly_affected_specs:
            lines.append(f"- compile context pack for `{spec_id}` after claims are accepted")
    else:
        lines.append("- compile after review once related specs are identified")

    lines.extend(
        [
            "",
            "## Human action needed",
            "- accept/reject proposed claims",
            "- update affected specs if needed",
            "",
        ]
    )
    return "\n".join(lines)


def find_possibly_affected_specs(store: KnowledgeStore, proposed_claims: list[ProposedClaim]) -> list[str]:
    if not proposed_claims:
        return []
    needles = [claim.text.casefold() for claim in proposed_claims]
    hits: list[str] = []
    for spec in store.specs():
        if not isinstance(spec, SpecV1):
            continue
        haystacks = [spec.component.casefold(), *[req.casefold() for req in spec.requirements]]
        haystacks.extend(ng.casefold() for ng in spec.non_goals)
        if any(
            needle in hay or any(word in hay for word in needle.split() if len(word) > 4)
            for needle in needles
            for hay in haystacks
        ):
            hits.append(spec.id)
    return sorted(set(hits))


def ingest_source(
    corpus_root: Path,
    *,
    source_path: Path,
    source_id: str,
    kind: str,
    write_review: bool = False,
    dry_run: bool = False,
    write_enabled: bool = True,
    store: KnowledgeStore | None = None,
    now: datetime | None = None,
) -> IngestSourceResult:
    if not source_id.startswith("source:"):
        raise ValueError("source_id must start with source:")

    resolved = source_path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"source path not found: {source_path}")

    try:
        source_kind = SourceKindV1(kind)
    except ValueError as exc:
        raise ValueError(f"unsupported source kind: {kind}") from exc

    text = resolved.read_text(encoding="utf-8")
    parsed = parse_source_markdown(text)

    active_store = store or KnowledgeStore(corpus_root)
    if store is None:
        active_store.load()
    affected = find_possibly_affected_specs(active_store, parsed.proposed_claims)

    slug = slug_from_source_id(source_id)
    dest_md_name = f"{slug}.md"
    source_rel_path = f"raw/sources/{dest_md_name}"
    review_rel_path: str | None = None
    warnings: list[str] = []

    content = build_source_delta_review(
        source_id=source_id,
        source_kind=source_kind.value,
        source_rel_path=source_rel_path,
        parsed=parsed,
        possibly_affected_specs=affected,
    )

    should_write = write_review and not dry_run
    if write_review and dry_run:
        warnings.append("dry run: no files written")
    elif write_review and not write_enabled:
        warnings.append("write disabled: KNOWLEDGE_FORGE_WRITE_ENABLED is false")
        should_write = False
    elif not write_review:
        warnings.append("write disabled: write_review is false")

    if should_write:
        _assert_write_targets_allowed(corpus_root, source_rel_path)
        sources_dir = corpus_root / "raw" / "sources"
        sources_dir.mkdir(parents=True, exist_ok=True)
        dest_md = sources_dir / dest_md_name
        shutil.copy2(resolved, dest_md)

        registry_name = f"source-{slug}.yaml"
        registry_rel = f"raw/sources/{registry_name}"
        _assert_write_targets_allowed(corpus_root, registry_rel)
        save_yaml_doc(
            corpus_root / registry_rel,
            {
                "type": "source",
                "id": source_id,
                "kind": source_kind.value,
                "path": source_rel_path,
                "trust_level": TrustLevelV1.primary.value,
            },
        )

        ts = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
        review_name = f"source-delta-{ts}-{slug}.md"
        review_rel_path = f"reviews/pending/{review_name}"
        _assert_write_targets_allowed(corpus_root, review_rel_path)
        pending_dir = corpus_root / "reviews" / "pending"
        pending_dir.mkdir(parents=True, exist_ok=True)
        (pending_dir / review_name).write_text(content, encoding="utf-8")

    return IngestSourceResult(
        source_id=source_id,
        status="proposed",
        source_path=source_rel_path if should_write else None,
        review_path=review_rel_path,
        proposed_claims=[c.text for c in parsed.proposed_claims],
        possibly_affected_specs=affected,
        warnings=warnings,
        content=content,
    )


def _infer_title(lines: list[str]) -> str:
    for line in lines:
        m = _HEADING_RE.match(line)
        if m:
            return m.group(2).strip()
    return "Untitled source"


def _bullet_line_number(lines: list[str], section: str, bullet_index: int) -> int | None:
    in_section = False
    seen = 0
    for idx, line in enumerate(lines, start=1):
        heading = _HEADING_RE.match(line)
        if heading:
            name = heading.group(2).strip()
            in_section = name == section
            continue
        if in_section and _BULLET_RE.match(line):
            if seen == bullet_index:
                return idx
            seen += 1
    return None


def _assert_write_targets_allowed(corpus_root: Path, rel_path: str) -> None:
    normalized = rel_path.replace("\\", "/")
    for forbidden in _FORBIDDEN_WRITE_PREFIXES:
        if normalized.startswith(forbidden):
            raise ValueError(f"refusing to write to protected path: {rel_path}")
    target = (corpus_root / rel_path).resolve()
    root = corpus_root.resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"path escapes corpus root: {rel_path}")
