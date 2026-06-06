from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import yaml

import app.settings as _app_settings
from app.clustering.leiden import build_graph_from_triples, leiden_cluster
from app.clustering.region_builder import build_region
from app.federators.episodic import EpisodicFederator
from app.federators.self_study import SelfStudyFederator
from app.federators.substrate import SubstrateFederator

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
        self._settings = _app_settings.get_settings()
        self._store = store
        self._bus = bus
        self._stop = asyncio.Event()
        self._policy = self._load_policy()
        # Captured in start() so the sync _tick thread can schedule async bus
        # emits back onto the running event loop. None outside the running app
        # (e.g. unit tests calling _tick directly), in which case emits are skipped.
        self._loop: asyncio.AbstractEventLoop | None = None
        # Shared LLM token budget for the current tick (reset each tick).
        self._llm_budget_remaining: int = 0

    def _load_policy(self) -> dict[str, Any]:
        try:
            return yaml.safe_load(Path(self._settings.compression_policy_path).read_text())
        except Exception as exc:
            logger.warning("policy_load_failed reason=%s — using defaults", exc)
            return {"clustering": {"resolution": 1.0, "n_iterations": 10, "min_community_size": 3, "max_communities_per_scope": 20}}

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        asyncio.create_task(self._poll_loop(), name="graph-compression-poll")

    async def stop(self) -> None:
        self._stop.set()

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

        # LLM token budget is shared across every scope processed this tick.
        self._llm_budget_remaining = int(self._settings.compression_llm_budget_per_tick)

        scopes_to_process = list({item.get("scope") for item in items if item.get("scope")})
        if not scopes_to_process:
            scopes_to_process = ["episodic", "substrate", "self_study"]

        for scope in scopes_to_process:
            try:
                self._process_scope(scope=scope)
            except Exception:
                logger.exception("scope_process_failed scope=%s", scope)

        queue_ids = [item["id"] for item in items if "id" in item]
        self._store.delete_stale_queue_items(queue_ids)
        logger.info("compression_tick_complete scopes=%s queue_items_drained=%d", scopes_to_process, len(queue_ids))

    def _process_scope(self, *, scope: str) -> None:
        cluster_cfg = (self._policy or {}).get("clustering", {})
        resolution = float(cluster_cfg.get("resolution", 1.0))
        n_iter = int(cluster_cfg.get("n_iterations", 10))
        min_size = int(cluster_cfg.get("min_community_size", 3))
        max_communities = int(cluster_cfg.get("max_communities_per_scope", 20))

        s = _app_settings.get_settings()

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
            # Default substrate clusters to "hotspot". We do not (yet) detect
            # genuine contradictions, so we must not blanket-label every cluster
            # "contradiction" — that would spuriously flood substrate mutation
            # pressure. Contradiction detection is a future enhancement.
            kind = "hotspot"
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

        summarizer = self._build_summarizer(s)

        for community in communities:
            summary, summary_kind = self._summarize_community(
                summarizer=summarizer, scope=scope, kind=kind, community=community
            )
            salience = min(1.0, len(community) / max(1, G.number_of_nodes()))
            region = build_region(
                nodes=community,
                scope=scope,
                kind=kind,
                summary=summary,
                summary_kind=summary_kind,
                salience=salience,
                trust_tier="unverified",
                compression_version="1.0.0",
            )
            write_ok = writer.write(region)
            # Persist the artifact index regardless of Fuseki write outcome:
            # Postgres is the source of truth for which regions exist; the
            # Fuseki copy is best-effort and may be unavailable (degraded mode).
            self._store.upsert_artifact(
                region_id=region.region_id,
                scope=region.scope,
                kind=region.kind,
                summary_kind=region.summary_kind,
                salience=region.salience,
                trust_tier=region.trust_tier,
                compression_version=region.compression_version,
                generated_at=region.generated_at,
                stale=not write_ok,
            )
            if write_ok:
                self._emit_events(writer, region)
            logger.info(
                "region_written region_id=%s scope=%s kind=%s nodes=%d fuseki_ok=%s",
                region.region_id, scope, kind, len(community), write_ok,
            )

    def _build_summarizer(self, s: Any):
        """Construct a RegionSummarizer when LLM summarization is viable, else None."""
        if not self._can_use_llm():
            return None
        from app.summarizer import RegionSummarizer

        return RegionSummarizer(
            bus=self._bus,
            llm_channel=s.llm_gateway_bus_channel,
            service_name=s.service_name,
            service_version=s.service_version,
            max_tokens=s.compression_max_tokens_per_summary,
        )

    def _can_use_llm(self) -> bool:
        """LLM summaries require the feature flag, a bus, a running loop, and budget."""
        return (
            bool(self._settings.enable_llm_summaries)
            and self._bus is not None
            and self._loop is not None
            and self._llm_budget_remaining >= int(self._settings.compression_max_tokens_per_summary)
        )

    def _summarize_community(
        self, *, summarizer: Any, scope: str, kind: str, community: Any
    ) -> tuple[str, str]:
        """Return (summary_text, summary_kind). Uses the LLM gateway when budget
        allows, otherwise a deterministic structural summary. Always falls back to
        structural on any error so the tick never fails on summarization."""
        structural = f"[structural] {scope} {kind} cluster: {len(community)} nodes."
        if summarizer is None or not self._can_use_llm():
            return structural, "structural"
        try:
            fut = asyncio.run_coroutine_threadsafe(
                summarizer.summarize(scope=scope, kind=kind, nodes=community),
                self._loop,
            )
            text, summary_kind = fut.result(
                timeout=float(self._settings.rdf_store_timeout_sec) + 15.0
            )
            if summary_kind == "llm":
                self._llm_budget_remaining -= int(
                    self._settings.compression_max_tokens_per_summary
                )
            return text, summary_kind
        except Exception:
            logger.warning("community_summarize_failed scope=%s kind=%s", scope, kind)
            return structural, "structural"

    def _emit_events(self, writer: "CompressionWriter", region: Any) -> None:
        """Bridge the async post-write bus emit from the sync tick thread to the
        event loop. No-op when there is no bus or no running loop (unit tests)."""
        if self._bus is None or self._loop is None:
            return
        try:
            fut = asyncio.run_coroutine_threadsafe(
                writer.emit_post_write(region), self._loop
            )
            fut.result(timeout=float(self._settings.rdf_store_timeout_sec) + 5.0)
        except Exception:
            logger.warning("compression_emit_events_failed region_id=%s", region.region_id)
