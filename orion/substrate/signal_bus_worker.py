"""Substrate signal bus worker.

Subscribes (via an external bus driver) to signal envelopes emitted by the
orion-signal-gateway and bridges the supported subset into the substrate
molecule store. The worker does not mutate gateway behavior; it runs alongside.

The handle_envelope() seam is bus-driver agnostic so tests can drive it without
spinning up a real OrionBusAsync instance.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from orion.signals.models import OrionSignalV1
from orion.substrate.experiment.harness import SubstrateExperimentHarness
from orion.substrate.molecule_store import MoleculeJsonlStore
from orion.substrate.signal_bridge import signal_to_molecule, supports_signal


logger = logging.getLogger(__name__)


class SubstrateSignalBusWorker:
    def __init__(
        self,
        *,
        store: MoleculeJsonlStore,
        harness: Optional[SubstrateExperimentHarness] = None,
    ) -> None:
        self._store = store
        self._harness = harness

    async def handle_envelope(self, env: Any) -> None:
        payload = env.payload
        if hasattr(payload, "model_dump"):
            payload = payload.model_dump()
        if not isinstance(payload, dict):
            return

        try:
            signal = OrionSignalV1.model_validate(payload)
        except Exception as exc:
            logger.debug("substrate signal bus worker: payload not OrionSignalV1: %s", exc)
            return

        if not supports_signal(signal):
            return

        try:
            molecule = signal_to_molecule(signal)
            self._store.add(molecule)
            if self._harness is not None:
                self._harness.record_emit(molecule, organ=signal.organ_id)
        except Exception as exc:
            logger.warning(
                "substrate signal bus worker: bridge/store failed organ=%s kind=%s: %s",
                signal.organ_id,
                signal.signal_kind,
                exc,
            )
