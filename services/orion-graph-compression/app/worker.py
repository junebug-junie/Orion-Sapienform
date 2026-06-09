from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from app.clustering.leiden import build_graph_from_triples, leiden_cluster
from app.clustering.region_builder import build_region
from app.federators.episodic import EpisodicFederator
from app.federators.self_study import SelfStudyFederator
from app.federators.substrate import SubstrateFederator
from app.settings import get_settings

if TYPE_CHECKING:
    from orion.core.bus.async_service import OrionBusAsync
    from app.store import CompressionStore

logger = logging.getLogger("orion.graph-compression.worker")


class CompressionWorker:
    def __init__(
        self,
        *,
        store: "CompressionStore",
        bus: "OrionBusAsync",
    ) -> None:
        self._settings = get_settings()
        self._store = store
        self._bus = bus
        self._stop = asyncio.Event()
        self._poll_task: asyncio.Task | None = None
        self._policy = self._load_policy()
        try:
            self._loop: asyncio.AbstractEventLoop | None = asyncio.get_event_loop()
        except RuntimeError:
            self._loop = None

    def _load_policy(self) -> dict[str, Any]:
        try:
            return yaml.safe_load(Path(self._settings.compression_policy_path).read_text())
        except Exception as exc:
            logger.warning("policy_load_failed reason=%s — using defaults", exc)
            return {
                "clustering": {
                    "resolution": 1.0,
                    "n_iterations": 10,
                    "min_community_size": 3,
                    "max_communities_per_scope": 20,
                }
            }

    async def start(self) -> None:
        self._poll_task = asyncio.create_task(self._poll_loop(), name="graph-compression-poll")

    async def stop(self) -> None:
        self._stop.set()
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass

    async def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.to_thread(self._tick)
            except Exception:
                logger.exception("compression_tick_failed")
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=float(self._settings.compression_poll_interval_sec),
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    def _tick(self) -> None:
        if not self._settings.enable_compression_runtime:
            return

        items = self._store.drain_stale_queue(batch_size=self._settings.compression_batch_size)
        if not items:
            return

        llm_tokens_used = 0
        budget = self._settings.compression_llm_budget_per_tick

        scopes_to_process = list({item.get("scope") for item in items if item.get("scope")})
        if not scopes_to_process:
            scopes_to_process = ["episodic", "substrate", "self_study"]

        for scope in scopes_to_process:
            try:
                self._process_scope(scope=scope, llm_tokens_used=llm_tokens_used, budget=budget)
            except Exception:
                logger.exception("scope_process_failed scope=%s", scope)

        queue_ids = [item["id"] for item in items if "id" in item]
        self._store.delete_stale_queue_items(queue_ids)
        logger.info(
            "compression_tick_complete scopes=%s queue_items_drained=%d",
            scopes_to_process,
            len(queue_ids),
        )

    def _process_scope(self, *, scope: str, llm_tokens_used: int, budget: int) -> None:
        cluster_cfg = (self._policy or {}).get("clustering", {})
        resolution = float(cluster_cfg.get("resolution", 1.0))
        n_iter = int(cluster_cfg.get("n_iterations", 10))
        min_size = int(cluster_cfg.get("min_community_size", 3))
        max_communities = int(cluster_cfg.get("max_communities_per_scope", 20))

        s = self._settings

        federator_kwargs = dict(
            query_url=s.rdf_store_query_url,
            user=s.rdf_store_user,
            password=s.rdf_store_pass,
            timeout_sec=s.rdf_store_timeout_sec,
        )

        if scope == "episodic":
            triples = EpisodicFederator(**federator_kwargs).fetch()
            kind = "community"
        elif scope == "substrate":
            triples = SubstrateFederator(**federator_kwargs).fetch()
            kind = "contradiction"
        elif scope == "self_study":
            triples = SelfStudyFederator(**federator_kwargs).fetch()
            kind = "self_study_cluster"
        else:
            return

        if not triples:
            logger.debug("scope_empty scope=%s — skipping", scope)
            return

        G = build_graph_from_triples(triples)
        communities = leiden_cluster(G, resolution=resolution, n_iterations=n_iter)
        communities = [c for c in communities if len(c) >= min_size][:max_communities]

        if not communities:
            logger.debug("no_viable_communities scope=%s nodes=%d", scope, G.number_of_nodes())
            return

        from app.writer import CompressionWriter

        writer = CompressionWriter(
            update_url=s.rdf_store_update_url,
            user=s.rdf_store_user,
            password=s.rdf_store_pass,
            timeout_sec=s.rdf_store_timeout_sec,
            bus=self._bus,
            service_name=s.service_name,
            service_version=s.service_version,
            channel_events=s.channel_graph_compression_events,
            channel_pressure=s.channel_substrate_mutation_pressure,
        )

        for community in communities:
            summary = f"[structural] {scope} {kind} cluster: {len(community)} nodes."
            salience = min(1.0, len(community) / max(1, G.number_of_nodes()))
            region = build_region(
                nodes=community,
                scope=scope,
                kind=kind,
                summary=summary,
                summary_kind="structural",
                salience=salience,
                trust_tier="unverified",
                compression_version="1.0.0",
            )
            if writer.write(region):
                self._store.upsert_artifact(
                    region_id=region.region_id,
                    scope=region.scope,
                    kind=region.kind,
                    summary_kind=region.summary_kind,
                    salience=region.salience,
                    trust_tier=region.trust_tier,
                    compression_version=region.compression_version,
                    generated_at=region.generated_at,
                )
                if self._loop and self._loop.is_running():
                    asyncio.run_coroutine_threadsafe(
                        writer._emit_grammar_hook(region), self._loop
                    )
                logger.info(
                    "region_written region_id=%s scope=%s kind=%s nodes=%d",
                    region.region_id,
                    scope,
                    kind,
                    len(community),
                )
