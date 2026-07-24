from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict
from typing import Any, Optional

from orion.core.bus.bus_service_chassis import BaseChassis, ChassisConfig
from orion.core.bus.codec import OrionCodec

from .settings import settings
from .substrate.mps_state import HeartbeatSubstrate
from .substrate.reconstruction import H1ResultV1, compute_h1
from .substrate.routing import (
    ORGAN_SITE_MAP,
    UnroutableAtomTypeError,
    UnroutableOrganError,
    route_atom,
)

logger = logging.getLogger("orion-heartbeat.service")

_GRAMMAR_EVENT_KIND = "grammar.event.v1"  # orion/bus/channels.yaml's
# orion:grammar:event message_kind -- checked against the live channel
# registry, not guessed.


class HeartbeatService(BaseChassis):
    """Additive, read-only consumer of the existing orion:grammar:event
    stream (see design doc's "Current architecture" for why this reuses
    orion-substrate-runtime's already-solved standardization work instead of
    the 2026-05-01 charter's bespoke per-organ reducers). Publishes nothing
    in v0 -- no downstream consumer, no phi broadcast, ablation-safe by
    construction (nothing depends on this service's output yet).
    """

    def __init__(self) -> None:
        super().__init__(
            ChassisConfig(
                service_name=settings.service_name,
                service_version=settings.service_version,
                node_name=settings.node_name or "unknown",
                instance_id=settings.instance_id,
                bus_url=settings.orion_bus_url,
                bus_enabled=settings.orion_bus_enabled,
                heartbeat_interval_sec=settings.heartbeat_interval_sec,
                health_channel=settings.health_channel,
            )
        )
        self.codec = OrionCodec()
        self.substrate = HeartbeatSubstrate(seed=settings.substrate_seed)
        self.latest_h1: Optional[H1ResultV1] = None
        # Counters surfaced on /health -- not published anywhere, purely for
        # the debug HTTP surface (design doc's inspectability commitment).
        self.events_seen = 0
        self.events_absorbed = 0
        self.events_skipped_organ = 0
        self.events_skipped_atom_type = 0
        self.events_skipped_no_atom = 0
        self.events_skipped_malformed = 0

    def latest_h1_dict(self) -> dict[str, Any] | None:
        if self.latest_h1 is None:
            return None
        d = asdict(self.latest_h1)
        d["generated_at"] = self.latest_h1.generated_at.isoformat()
        return d

    def stats(self) -> dict[str, Any]:
        return {
            "events_seen": self.events_seen,
            "events_absorbed": self.events_absorbed,
            "events_skipped_organ": self.events_skipped_organ,
            "events_skipped_atom_type": self.events_skipped_atom_type,
            "events_skipped_no_atom": self.events_skipped_no_atom,
            "events_skipped_malformed": self.events_skipped_malformed,
            "tick_count": self.substrate.tick_count,
            "max_bond": self.substrate.max_bond(),
            "norm": self.substrate.norm(),
            "allowlisted_organs": sorted(ORGAN_SITE_MAP.keys()),
        }

    def _handle_atom_payload(self, atom: dict[str, Any]) -> None:
        """One GrammarAtomV1 dict (already extracted from a decoded
        GrammarEventV1 payload). Skips (does not raise) on anything outside
        v0's organ/atom_type allowlist -- a malformed or out-of-scope event
        must not crash the tick loop; see routing.route_atom's own
        docstring for why routing itself still raises.
        """
        provenance = atom.get("_provenance") or {}
        source_service = str(provenance.get("source_service") or "")
        atom_type = str(atom.get("atom_type") or "")

        try:
            assignment = route_atom(
                source_service=source_service,
                atom_type=atom_type,
                confidence=atom.get("confidence"),
                salience=atom.get("salience"),
                uncertainty=atom.get("uncertainty"),
            )
        except UnroutableOrganError:
            self.events_skipped_organ += 1
            return
        except UnroutableAtomTypeError:
            self.events_skipped_atom_type += 1
            logger.warning(
                "heartbeat_unroutable_atom_type atom_type=%s source_service=%s "
                "-- orion/schemas/grammar.py may have gained a new AtomType not "
                "yet covered in routing.ATOM_TYPE_OPERATOR_KIND",
                atom_type,
                source_service,
            )
            return
        except (ValueError, TypeError) as exc:
            # route_atom()'s confidence/salience/uncertainty float() coercion
            # can raise these for a malformed atom (heartbeat works directly
            # off the decoded dict, not a re-validated GrammarAtomV1, so a
            # non-numeric value isn't ruled out upstream). Found by review:
            # this was previously falling through to _run()'s generic
            # except-Exception handler under a bare "heartbeat_handle_failed"
            # log line, uncounted -- now has its own counter and log line,
            # same as the two named routing exceptions above.
            self.events_skipped_malformed += 1
            logger.warning(
                "heartbeat_malformed_atom_fields source_service=%s atom_type=%s err=%s",
                source_service,
                atom_type,
                exc,
            )
            return

        self.substrate.absorb(assignment)
        self.events_absorbed += 1

    async def _handle_grammar_message(self, payload_dict: dict[str, Any]) -> None:
        self.events_seen += 1
        if payload_dict.get("event_kind") != "atom_emitted":
            return
        atom = payload_dict.get("atom")
        if not isinstance(atom, dict):
            self.events_skipped_no_atom += 1
            return
        # provenance lives at the GrammarEventV1 level, not on the atom
        # itself (orion/schemas/grammar.py) -- stash it under the atom dict
        # so _handle_atom_payload has a single argument to work with.
        atom = dict(atom)
        atom["_provenance"] = payload_dict.get("provenance") or {}
        self._handle_atom_payload(atom)

    async def _h1_loop(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(settings.h1_interval_sec)
            try:
                self.latest_h1 = compute_h1(self.substrate)
                logger.info(
                    "heartbeat_h1_computed tick_count=%d boundary_bulk_entropy=%.4f "
                    "ratio=%.4f verdict=%s",
                    self.latest_h1.tick_count,
                    self.latest_h1.boundary_bulk_entropy,
                    self.latest_h1.ratio,
                    self.latest_h1.verdict,
                )
            except Exception as exc:  # noqa: BLE001 - must not kill the loop
                logger.warning("heartbeat_h1_computation_failed err=%s", exc)

    async def _run(self) -> None:
        h1_task = asyncio.create_task(self._h1_loop(), name="heartbeat-h1-loop")
        try:
            async with self.bus.subscribe(settings.channel_grammar_event) as pubsub:
                async for msg in self.bus.iter_messages(pubsub):
                    if self._stop.is_set():
                        break

                    decoded = self.codec.decode(msg.get("data"))
                    if not decoded.ok or decoded.envelope is None:
                        logger.warning(
                            "heartbeat_decode_failed channel=%s error=%s",
                            settings.channel_grammar_event,
                            decoded.error,
                        )
                        continue

                    env = decoded.envelope
                    if env.kind != _GRAMMAR_EVENT_KIND:
                        continue
                    payload_dict = env.payload if isinstance(env.payload, dict) else {}

                    try:
                        await self._handle_grammar_message(payload_dict)
                    except Exception as exc:  # noqa: BLE001 - one bad event must not kill the subscriber
                        logger.warning("heartbeat_handle_failed err=%s", exc)
        finally:
            h1_task.cancel()
            try:
                await h1_task
            except asyncio.CancelledError:
                pass
