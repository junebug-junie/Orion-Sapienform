from __future__ import annotations

from typing import Any

from orion.schemas.evidence_index import EvidenceUnitV1, ParsedDocumentIngestV1


def _link_siblings(units: list[EvidenceUnitV1]) -> None:
    for index, unit in enumerate(units):
        unit.sibling_prev_id = units[index - 1].unit_id if index > 0 else None
        unit.sibling_next_id = units[index + 1].unit_id if index + 1 < len(units) else None


class ParsedDocumentEvidenceAdapter:
    source_family = "parsed_document"

    def to_units(self, payload: Any, *, kind: str, correlation_id: str | None = None) -> list[EvidenceUnitV1]:
        data = payload if isinstance(payload, dict) else {}
        container = data.get("payload") if isinstance(data.get("payload"), dict) else data
        parsed = ParsedDocumentIngestV1.model_validate(container)

        section_page_starts = [s.page_start for s in parsed.sections if s.page_start is not None]
        section_page_ends = [s.page_end for s in parsed.sections if s.page_end is not None]
        page_span = None
        if section_page_starts and section_page_ends:
            page_span = {"page_start": min(section_page_starts), "page_end": max(section_page_ends)}

        document_unit = EvidenceUnitV1(
            unit_id=parsed.doc_id,
            unit_kind="document",
            source_family=self.source_family,
            source_kind=parsed.source_kind,
            source_ref=parsed.source_ref,
            correlation_id=parsed.correlation_id or correlation_id,
            title=parsed.title,
            summary=parsed.summary,
            body=parsed.body,
            facets=["artifact:document", "format:parsed_document", *parsed.facets],
            metadata={"source_provenance": parsed.source_provenance, "page_span": page_span},
            created_at=parsed.created_at,
        )

        section_units: list[EvidenceUnitV1] = []
        leaf_units: list[EvidenceUnitV1] = []
        for section in parsed.sections:
            section_unit_id = f"{parsed.doc_id}::section::{section.section_id}"
            block_types = sorted({block.block_type for block in section.blocks if block.block_type})
            section_unit = EvidenceUnitV1(
                unit_id=section_unit_id,
                unit_kind="document_section",
                source_family=self.source_family,
                source_kind=parsed.source_kind,
                source_ref=parsed.source_ref,
                correlation_id=parsed.correlation_id or correlation_id,
                parent_unit_id=parsed.doc_id,
                title=section.title,
                summary=section.summary,
                body=section.body,
                facets=["artifact:section", *section.facets, *[f"block_type:{item}" for item in block_types]],
                metadata={
                    "heading_path": section.heading_path,
                    "page_start": section.page_start,
                    "page_end": section.page_end,
                    "block_types": block_types,
                },
                created_at=parsed.created_at,
            )
            section_units.append(section_unit)

            leaves_for_section: list[EvidenceUnitV1] = []
            for block in section.blocks:
                leaf_unit_id = f"{section_unit_id}::leaf::{block.block_id}"
                leaves_for_section.append(
                    EvidenceUnitV1(
                        unit_id=leaf_unit_id,
                        unit_kind="document_leaf",
                        source_family=self.source_family,
                        source_kind=parsed.source_kind,
                        source_ref=parsed.source_ref,
                        correlation_id=parsed.correlation_id or correlation_id,
                        parent_unit_id=section_unit_id,
                        title=block.title or f"{section.title} [{block.block_type}]",
                        summary=block.summary,
                        body=block.body,
                        facets=["artifact:leaf", f"block_type:{block.block_type}", *block.facets],
                        metadata={
                            "heading_path": block.heading_path or section.heading_path,
                            "page_start": block.page_start,
                            "page_end": block.page_end,
                            "block_type": block.block_type,
                            "source_provenance": block.source_provenance or parsed.source_provenance,
                        },
                        created_at=parsed.created_at,
                    )
                )
            _link_siblings(leaves_for_section)
            leaf_units.extend(leaves_for_section)

        _link_siblings(section_units)
        return [document_unit, *section_units, *leaf_units]
