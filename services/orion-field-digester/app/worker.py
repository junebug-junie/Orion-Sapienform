from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

from app.graph.lattice import load_lattice
from app.ingest.state_deltas import Perturbation, delta_to_perturbations
from app.settings import get_settings
from app.store import FieldDigesterStore, PendingDelta
from app.tensor.field_state import empty_field_state, new_tick_id
from app.tensor.update_rules import run_digestion_tick

logger = logging.getLogger("orion.field.digester")


class FieldDigesterWorker:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._store = FieldDigesterStore(self._settings.postgres_uri)
        self._lattice = load_lattice(Path(self._settings.lattice_path))
        self._stop = asyncio.Event()

    async def start(self) -> None:
        asyncio.create_task(self._poll_loop(), name="field-digester-poll")

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

    def _tick(self) -> None:
        fetched = self._store.fetch_new_receipts(limit=50)
        if not fetched:
            return

        now = datetime.now(timezone.utc)
        state = self._store.load_latest_field()
        if state is None:
            state = empty_field_state(
                lattice=self._lattice,
                now=now,
                tick_id=new_tick_id(),
            )

        perturbations: list[Perturbation] = []
        pending_deltas: list[PendingDelta] = []
        for item in fetched:
            receipt = item.receipt
            for delta in receipt.state_deltas:
                if self._store.is_delta_applied(delta.delta_id):
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

        last = fetched[-1]
        self._store.commit_digest_tick(
            state=state,
            pending_deltas=pending_deltas,
            cursor_receipt_id=last.receipt.receipt_id,
            cursor_created_at=last.created_at,
        )
