from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

from orion.field_coherence import check_field_coherence

from app.graph.lattice import load_lattice
from app.health_monitor import HealthMonitor
from app.ingest.state_deltas import Perturbation, delta_to_perturbations
from app.settings import get_settings
from app.store import FieldDigesterStore, PendingDelta
from app.tensor.field_state import empty_field_state, new_tick_id
from app.tensor.reconcile import reconcile_field_state_with_lattice
from app.tensor.update_rules import run_digestion_tick

logger = logging.getLogger("orion.field.digester")


class FieldDigesterWorker:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._store = FieldDigesterStore(self._settings.postgres_uri)
        self._lattice = load_lattice(Path(self._settings.lattice_path))
        self._health_monitor = HealthMonitor(self._store, self._settings)
        self._stop = asyncio.Event()

    async def start(self) -> None:
        asyncio.create_task(self._poll_loop(), name="field-digester-poll")
        asyncio.create_task(self._prune_loop(), name="field-digester-prune")
        asyncio.create_task(self._health_loop(), name="field-digester-health")

    async def stop(self) -> None:
        self._stop.set()

    async def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.to_thread(self._tick)
            except Exception:
                logger.exception("field_digester_tick_failed")
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=float(self._settings.receipt_poll_interval_sec),
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    def _prune_tick(self) -> None:
        retention = float(self._settings.field_state_retention_hours)
        if retention > 0:
            deleted = self._store.prune_field_state(retention_hours=retention)
            if deleted:
                logger.info(
                    "field_state_pruned deleted=%d retention_hours=%.1f", deleted, retention
                )

        min_age = float(self._settings.field_applied_deltas_prune_min_age_hours)
        deleted_deltas = self._store.prune_applied_deltas(min_age_hours=min_age)
        if deleted_deltas:
            logger.info(
                "applied_deltas_pruned deleted=%d min_age_hours=%.1f",
                deleted_deltas,
                min_age,
            )

    async def _prune_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.to_thread(self._prune_tick)
            except Exception:
                logger.exception("field_state_prune_failed")
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=float(self._settings.field_state_prune_interval_sec),
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    async def _health_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.to_thread(self._health_monitor.run_tick)
            except Exception:
                logger.exception("field_digester_health_check_failed")
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=float(self._settings.health_check_interval_sec),
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    def _tick(self) -> None:
        fetched = self._store.fetch_new_receipts(limit=50)
        if not fetched and not self._settings.enable_idle_tick:
            return

        now = datetime.now(timezone.utc)
        state = self._store.load_latest_field()
        if state is None:
            state = empty_field_state(
                lattice=self._lattice,
                now=now,
                tick_id=new_tick_id(),
            )
        state = reconcile_field_state_with_lattice(state, lattice=self._lattice)
        state.topology_loaded_from = self._settings.lattice_path
        state.topology_id = "orion_field_topology"
        state.topology_version = "v1"

        perturbations: list[Perturbation] = []
        pending_deltas: list[PendingDelta] = []
        for item in fetched:
            receipt = item.receipt
            for delta in receipt.state_deltas:
                if self._store.is_delta_applied(delta.delta_id):
                    continue
                if (
                    delta.target_kind == "transport_bus"
                    and not self._settings.enable_transport_field_digestion
                ):
                    continue
                perturbations.extend(delta_to_perturbations(delta))
                pending_deltas.append(
                    PendingDelta(delta_id=delta.delta_id, receipt_id=receipt.receipt_id)
                )

        state.generated_at = now
        state.tick_id = new_tick_id()
        run_digestion_tick(
            state,
            perturbations=perturbations,
            decay_rate=self._settings.biometrics_field_decay_rate,
            diffusion_rate=self._settings.biometrics_field_diffusion_rate,
        )

        for node_id, suspicion in check_field_coherence(state).items():
            state.node_vectors.setdefault(node_id, {})["field_coherence_warning"] = suspicion

        if not fetched:
            self._store.save_field(state)
            return

        last = fetched[-1]
        self._store.commit_digest_tick(
            state=state,
            pending_deltas=pending_deltas,
            cursor_receipt_id=last.receipt.receipt_id,
            cursor_created_at=last.created_at,
        )
