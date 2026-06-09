from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Optional

import httpx

from orion.schemas.graph_compression import (
    CompressionRegionV1,
    GraphCompressionRegionMaterializedV1,
)

if TYPE_CHECKING:
    from orion.core.bus.async_service import OrionBusAsync

logger = logging.getLogger("orion.graph-compression.writer")

_COMPRESSIONS_GRAPH_URI = "http://conjourney.net/graph/orion/compressions"
_ORN_NS = "http://orion.conjourney.net/ns/compression#"


class CompressionWriter:
    def __init__(
        self,
        *,
        update_url: str,
        user: str,
        password: str,
        timeout_sec: float,
        bus: Optional["OrionBusAsync"],
        service_name: str,
        service_version: str,
        channel_events: str,
        channel_pressure: str,
    ) -> None:
        self._update_url = update_url
        self._auth = (user, password)
        self._timeout = timeout_sec
        self._bus = bus
        self._service_name = service_name
        self._service_version = service_version
        self._channel_events = channel_events
        self._channel_pressure = channel_pressure

    def _escape_literal(self, value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

    def _build_sparql_update(self, region: CompressionRegionV1) -> str:
        rid = region.region_id
        triples = [
            f'<{rid}> <{_ORN_NS}scope> "{self._escape_literal(region.scope)}" .',
            f'<{rid}> <{_ORN_NS}kind> "{self._escape_literal(region.kind)}" .',
            f'<{rid}> <{_ORN_NS}summary> "{self._escape_literal(region.summary)}" .',
            f'<{rid}> <{_ORN_NS}summaryKind> "{self._escape_literal(region.summary_kind)}" .',
            f'<{rid}> <{_ORN_NS}salience> "{region.salience}"^^<http://www.w3.org/2001/XMLSchema#decimal> .',
            f'<{rid}> <{_ORN_NS}trustTier> "{self._escape_literal(region.trust_tier)}" .',
            f'<{rid}> <{_ORN_NS}compressionVersion> "{self._escape_literal(region.compression_version)}" .',
            f'<{rid}> <{_ORN_NS}generatedAt> "{region.generated_at.isoformat()}"^^<http://www.w3.org/2001/XMLSchema#dateTime> .',
        ]
        for exemplar in region.exemplar_ids:
            triples.append(f'<{rid}> <{_ORN_NS}exemplarId> <{exemplar}> .')
        for src in region.derived_from:
            triples.append(f'<{rid}> <{_ORN_NS}derivedFrom> <{src}> .')

        triples_block = "\n    ".join(triples)
        return (
            f"DELETE {{ GRAPH <{_COMPRESSIONS_GRAPH_URI}> {{ <{rid}> ?p ?o }} }}\n"
            f"WHERE  {{ GRAPH <{_COMPRESSIONS_GRAPH_URI}> {{ <{rid}> ?p ?o }} }} ;\n"
            f"INSERT DATA {{\n"
            f"  GRAPH <{_COMPRESSIONS_GRAPH_URI}> {{\n"
            f"    {triples_block}\n"
            f"  }}\n"
            f"}}"
        )

    def write(self, region: CompressionRegionV1) -> bool:
        """Write region to Fuseki. Returns True on success."""
        sparql = self._build_sparql_update(region)
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(
                    self._update_url,
                    data={"update": sparql},
                    auth=self._auth,
                )
                resp.raise_for_status()
            return True
        except Exception as exc:
            logger.warning(
                "compression_write_failed region_id=%s reason=%s",
                region.region_id,
                exc,
            )
            return False

    async def _emit_grammar_hook(self, region: CompressionRegionV1) -> None:
        """Emit bus events after successful write."""
        if self._bus is None:
            return

        from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

        materialized = GraphCompressionRegionMaterializedV1(
            region_id=region.region_id,
            scope=region.scope,
            kind=region.kind,
            salience=region.salience,
            trust_tier=region.trust_tier,
            summary_kind=region.summary_kind,
            compression_version=region.compression_version,
            ts=time.time(),
        )
        await self._bus.publish(
            self._channel_events,
            BaseEnvelope(
                kind="graph.compression.region.materialized.v1",
                source=ServiceRef(name=self._service_name, version=self._service_version),
                payload=materialized.model_dump(mode="json"),
            ),
        )

        if region.kind == "contradiction":
            from orion.core.schemas.substrate_mutation import MutationPressureEvidenceV1

            pressure = MutationPressureEvidenceV1(
                source_service=self._service_name,
                source_event_id=region.region_id,
                pressure_category="unsupported_memory_claim",
                confidence=region.salience,
                evidence_refs=[region.region_id] + region.derived_from[:4],
                metadata={"compression_kind": "contradiction", "scope": region.scope},
            )
            await self._bus.publish(
                self._channel_pressure,
                BaseEnvelope(
                    kind="substrate.mutation.pressure.v1",
                    source=ServiceRef(name=self._service_name, version=self._service_version),
                    payload=pressure.model_dump(mode="json"),
                ),
            )
