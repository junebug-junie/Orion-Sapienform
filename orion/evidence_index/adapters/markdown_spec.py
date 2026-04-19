from __future__ import annotations

from typing import Any

from orion.schemas.evidence_index import EvidenceUnitV1, MarkdownSpecIngestV1


def _strip_lines(text: str) -> list[str]:
    return [line.rstrip() for line in text.splitlines()]


def _is_table_line(line: str) -> bool:
    return line.count("|") >= 2


def _summarize(text: str, limit: int = 180) -> str | None:
    cleaned = " ".join(part.strip() for part in text.splitlines() if part.strip())
    if not cleaned:
        return None
    return cleaned[:limit]


def _link_siblings(units: list[EvidenceUnitV1]) -> None:
    for index, unit in enumerate(units):
        unit.sibling_prev_id = units[index - 1].unit_id if index > 0 else None
        unit.sibling_next_id = units[index + 1].unit_id if index + 1 < len(units) else None


class MarkdownSpecEvidenceAdapter:
    source_family = "markdown_spec"

    def to_units(self, payload: Any, *, kind: str, correlation_id: str | None = None) -> list[EvidenceUnitV1]:
        data = payload if isinstance(payload, dict) else {}
        container = data.get("payload") if isinstance(data.get("payload"), dict) else data
        doc = MarkdownSpecIngestV1.model_validate(container)

        document_unit = EvidenceUnitV1(
            unit_id=doc.doc_id,
            unit_kind="document",
            source_family=self.source_family,
            source_kind=doc.source_kind,
            source_ref=doc.source_ref,
            correlation_id=doc.correlation_id or correlation_id,
            title=doc.title,
            summary=_summarize(doc.body),
            body=doc.body,
            facets=["artifact:document", "format:markdown"],
            metadata={"adapter": "markdown_spec"},
            created_at=doc.created_at,
        )

        sections: list[dict[str, Any]] = []
        current_title = "Overview"
        current_lines: list[str] = []
        section_level = 1
        for line in _strip_lines(doc.body):
            if line.startswith("#"):
                heading = line.lstrip("#").strip()
                level = len(line) - len(line.lstrip("#"))
                if current_lines:
                    sections.append({"title": current_title, "level": section_level, "lines": current_lines})
                current_title = heading or "Untitled Section"
                current_lines = []
                section_level = level
            else:
                current_lines.append(line)
        if current_lines:
            sections.append({"title": current_title, "level": section_level, "lines": current_lines})
        if not sections:
            sections = [{"title": "Overview", "level": 1, "lines": _strip_lines(doc.body)}]

        section_units: list[EvidenceUnitV1] = []
        leaf_units: list[EvidenceUnitV1] = []
        for section_index, section in enumerate(sections, start=1):
            section_id = f"{doc.doc_id}::section::{section_index}"
            section_body = "\n".join(section["lines"]).strip()
            section_unit = EvidenceUnitV1(
                unit_id=section_id,
                unit_kind="document_section",
                source_family=self.source_family,
                source_kind=doc.source_kind,
                source_ref=doc.source_ref,
                correlation_id=doc.correlation_id or correlation_id,
                parent_unit_id=doc.doc_id,
                title=section["title"],
                summary=_summarize(section_body),
                body=section_body or None,
                facets=["artifact:section", f"heading_level:{section['level']}"],
                metadata={"adapter": "markdown_spec", "section_index": section_index},
                created_at=doc.created_at,
            )
            section_units.append(section_unit)

            leaves_for_section: list[EvidenceUnitV1] = []
            lines = section["lines"]
            cursor = 0
            leaf_index = 0
            while cursor < len(lines):
                line = lines[cursor]
                if not line.strip():
                    cursor += 1
                    continue

                block_type = "paragraph"
                block_lines: list[str] = []
                if line.strip().startswith("```"):
                    block_type = "code"
                    block_lines.append(line)
                    cursor += 1
                    while cursor < len(lines):
                        block_lines.append(lines[cursor])
                        if lines[cursor].strip().startswith("```"):
                            cursor += 1
                            break
                        cursor += 1
                elif _is_table_line(line):
                    block_type = "table"
                    while cursor < len(lines) and _is_table_line(lines[cursor]):
                        block_lines.append(lines[cursor])
                        cursor += 1
                else:
                    while cursor < len(lines) and lines[cursor].strip() and not lines[cursor].strip().startswith("```") and not _is_table_line(lines[cursor]):
                        block_lines.append(lines[cursor])
                        cursor += 1

                block_body = "\n".join(block_lines).strip()
                if not block_body:
                    continue
                leaf_index += 1
                leaf_id = f"{section_id}::leaf::{leaf_index}"
                leaves_for_section.append(
                    EvidenceUnitV1(
                        unit_id=leaf_id,
                        unit_kind="document_leaf",
                        source_family=self.source_family,
                        source_kind=doc.source_kind,
                        source_ref=doc.source_ref,
                        correlation_id=doc.correlation_id or correlation_id,
                        parent_unit_id=section_id,
                        title=f"{section['title']} [{block_type}]",
                        summary=_summarize(block_body),
                        body=block_body,
                        facets=["artifact:leaf", f"block_type:{block_type}"],
                        metadata={"adapter": "markdown_spec", "section_index": section_index, "block_type": block_type},
                        created_at=doc.created_at,
                    )
                )

            _link_siblings(leaves_for_section)
            leaf_units.extend(leaves_for_section)

        _link_siblings(section_units)
        return [document_unit, *section_units, *leaf_units]
