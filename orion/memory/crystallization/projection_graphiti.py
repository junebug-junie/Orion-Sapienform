from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

from orion.memory.crystallization.schemas import MemoryCrystallizationV1

logger = logging.getLogger(__name__)


@dataclass
class GraphitiProjectionResult:
    episode_ids: list[str] = field(default_factory=list)
    entity_ids: list[str] = field(default_factory=list)
    edge_ids: list[str] = field(default_factory=list)
    synced_at: datetime | None = None
    canonical_mutated: bool = False
    remote_response: dict[str, Any] | None = None


class GraphitiAdapter:
    """HTTP client to orion-graphiti-adapter — cannot mutate canonical crystallizations."""

    def __init__(self, *, enabled: bool = False, url: str | None = None, falkordb_uri: str | None = None, timeout_sec: float = 10.0):
        self.enabled = enabled
        self.url = (url or "").strip().rstrip("/")
        self.falkordb_uri = (falkordb_uri or "").strip()
        self.timeout_sec = timeout_sec

    def can_sync(self, crystallization: MemoryCrystallizationV1) -> bool:
        if not self.enabled or not self.url:
            return False
        return crystallization.status in ("active", "proposed")

    async def sync_crystallization_async(self, crystallization: MemoryCrystallizationV1) -> GraphitiProjectionResult:
        if not self.can_sync(crystallization):
            return GraphitiProjectionResult()

        payload = {
            "crystallization_id": crystallization.crystallization_id,
            "kind": crystallization.kind,
            "subject": crystallization.subject,
            "summary": crystallization.summary,
            "status": crystallization.status,
            "metadata": {
                "scope": crystallization.scope,
                "salience": crystallization.salience,
                "confidence": crystallization.confidence,
                "sensitivity": crystallization.governance.sensitivity,
            },
            "links": [
                {
                    "target_crystallization_id": l.target_crystallization_id,
                    "relation": l.relation,
                    "confidence": l.confidence,
                }
                for l in crystallization.links
            ],
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
                resp = await client.post(f"{self.url}/v1/episodes", json=payload)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            logger.warning("graphiti_sync_failed id=%s error=%s", crystallization.crystallization_id, exc)
            return GraphitiProjectionResult()

        if data.get("skipped"):
            return GraphitiProjectionResult(
                canonical_mutated=bool(data.get("canonical_mutated")),
                remote_response=data,
            )

        now = datetime.now(timezone.utc)
        return GraphitiProjectionResult(
            episode_ids=[str(data.get("episode_id"))] if data.get("episode_id") else [],
            entity_ids=[str(data.get("entity_id"))] if data.get("entity_id") else [],
            edge_ids=[str(data.get("edge_id"))] if data.get("edge_id") else [],
            synced_at=now,
            canonical_mutated=bool(data.get("canonical_mutated")),
            remote_response=data,
        )

    def sync_crystallization(self, crystallization: MemoryCrystallizationV1) -> GraphitiProjectionResult:
        """Sync wrapper for sync call sites; uses httpx sync client."""
        if not self.can_sync(crystallization):
            return GraphitiProjectionResult()
        payload = {
            "crystallization_id": crystallization.crystallization_id,
            "kind": crystallization.kind,
            "subject": crystallization.subject,
            "summary": crystallization.summary,
            "status": crystallization.status,
            "metadata": {
                "scope": crystallization.scope,
                "salience": crystallization.salience,
                "confidence": crystallization.confidence,
                "sensitivity": crystallization.governance.sensitivity,
            },
            "links": [
                {
                    "target_crystallization_id": l.target_crystallization_id,
                    "relation": l.relation,
                    "confidence": l.confidence,
                }
                for l in crystallization.links
            ],
        }
        try:
            with httpx.Client(timeout=self.timeout_sec) as client:
                resp = client.post(f"{self.url}/v1/episodes", json=payload)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            logger.warning("graphiti_sync_failed id=%s error=%s", crystallization.crystallization_id, exc)
            return GraphitiProjectionResult()

        if data.get("skipped"):
            return GraphitiProjectionResult(
                canonical_mutated=bool(data.get("canonical_mutated")),
                remote_response=data,
            )

        now = datetime.now(timezone.utc)
        return GraphitiProjectionResult(
            episode_ids=[str(data.get("episode_id"))] if data.get("episode_id") else [],
            entity_ids=[str(data.get("entity_id"))] if data.get("entity_id") else [],
            edge_ids=[str(data.get("edge_id"))] if data.get("edge_id") else [],
            synced_at=now,
            canonical_mutated=bool(data.get("canonical_mutated")),
            remote_response=data,
        )

    def health(self) -> dict[str, Any]:
        if not self.enabled or not self.url:
            return {"enabled": False, "backend": "unknown"}
        try:
            with httpx.Client(timeout=self.timeout_sec) as client:
                resp = client.get(f"{self.url}/health")
                resp.raise_for_status()
                return resp.json()
        except Exception as exc:
            logger.warning("graphiti_health_failed error=%s", exc)
            return {"enabled": True, "backend": "unknown", "error": str(exc)}

    def search(
        self,
        query: str,
        *,
        seed_crystallization_id: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        if not self.enabled or not self.url:
            return {"crystallization_ids": [], "trace": {}}
        try:
            with httpx.Client(timeout=self.timeout_sec) as client:
                resp = client.post(
                    f"{self.url}/v1/search",
                    json={
                        "query": query,
                        "seed_crystallization_id": seed_crystallization_id,
                        "limit": limit,
                    },
                )
                resp.raise_for_status()
                return resp.json()
        except Exception as exc:
            logger.warning("graphiti_search_failed query=%s error=%s", query, exc)
            return {"crystallization_ids": [], "trace": {"error": str(exc)}}

    def neighborhood(self, crystallization_id: str, *, depth: int = 1) -> dict[str, Any]:
        if not self.enabled or not self.url:
            return {"enabled": False, "nodes": [], "edges": []}
        try:
            with httpx.Client(timeout=self.timeout_sec) as client:
                resp = client.get(
                    f"{self.url}/v1/neighborhood/{crystallization_id}",
                    params={"depth": depth},
                )
                resp.raise_for_status()
                data = resp.json()
                data["enabled"] = True
                return data
        except Exception as exc:
            logger.warning("graphiti_neighborhood_failed id=%s error=%s", crystallization_id, exc)
            return {"enabled": True, "crystallization_id": crystallization_id, "nodes": [], "edges": [], "error": str(exc)}

    def apply_projection_refs(
        self,
        crystallization: MemoryCrystallizationV1,
        result: GraphitiProjectionResult,
    ) -> MemoryCrystallizationV1:
        updated = crystallization.model_copy(deep=True)
        updated.projection_refs.graphiti_episode_ids = list(result.episode_ids)
        updated.projection_refs.graphiti_entity_ids = list(result.entity_ids)
        updated.projection_refs.graphiti_edge_ids = list(result.edge_ids)
        updated.projection_refs.synced_at = result.synced_at
        updated.updated_at = datetime.now(timezone.utc)
        return updated
