from __future__ import annotations

import hashlib
import json
import logging
import random
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger("orion.substrate.store")

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from psycopg2.extras import Json

from app.settings import Settings, get_settings
from app.cursor_gaps import TailSeedRecord, record_tail_seed
from orion.schemas.biometrics_projection import (
    ActiveNodePressureProjectionV1,
    NodeBiometricsProjectionV1,
)
from orion.schemas.execution_projection import ExecutionTrajectoryProjectionV1
from orion.schemas.transport_projection import TransportBusProjectionV1
from orion.schemas.grammar import GrammarEventV1
from orion.schemas.organ_emission import OrganEmissionV1
from orion.schemas.reduction_receipt import ReductionReceiptV1
from orion.core.schemas.substrate_episodes import EpisodeSummaryV1
from orion.schemas.attention_frame import AttentionBroadcastProjectionV1

from orion.substrate.biometrics_loop.constants import GRAMMAR_CURSOR_NAME
from orion.substrate.execution_loop.constants import (
    EXECUTION_GRAMMAR_CURSOR_NAME,
    EXECUTION_SOURCE_SERVICES,
)
from orion.substrate.transport_loop.constants import (
    TRANSPORT_BUS_PROJECTION_ID,
    TRANSPORT_GRAMMAR_CURSOR_NAME,
)
from orion.substrate.chat_loop.constants import (
    CHAT_GRAMMAR_CURSOR_NAME,
    CHAT_SOURCE_SERVICE,
    CHAT_TRACE_PREFIX,
)
from orion.schemas.chat_projection import ChatSessionProjectionV1

EXECUTION_GRAMMAR_SOURCE_SERVICES = tuple(EXECUTION_SOURCE_SERVICES)

GRAMMAR_CURSOR_REGISTRY: dict[str, tuple[tuple[str, ...], str]] = {
    GRAMMAR_CURSOR_NAME: (("orion-biometrics",), "biometrics.node:"),
    EXECUTION_GRAMMAR_CURSOR_NAME: (EXECUTION_GRAMMAR_SOURCE_SERVICES, "cortex.exec:"),
    TRANSPORT_GRAMMAR_CURSOR_NAME: (("orion-bus",), "bus.transport:"),
    CHAT_GRAMMAR_CURSOR_NAME: ((CHAT_SOURCE_SERVICE,), "hub.chat:"),
}
from orion.substrate.biometrics_loop.lineage import emission_touches_node, receipt_touches_node
from orion.substrate.receipts.retention import (
    ReceiptRetentionSettings,
    classify_receipt,
    compact_receipt_json,
    payload_byte_length,
    payload_fingerprint,
    primary_delta_id,
    primary_event_id,
    primary_reducer_name,
    retention_expires_at,
)


def _retention_settings_from_app(settings: Settings) -> ReceiptRetentionSettings:
    return ReceiptRetentionSettings(
        success_minutes=settings.receipt_retention_success_minutes,
        error_hours=settings.receipt_retention_error_hours,
        full_payload_success=settings.receipt_full_payload_success,
        full_payload_sample_rate=settings.receipt_full_payload_sample_rate,
    )


def _build_receipt_insert_params(
    receipt: ReductionReceiptV1,
    *,
    retention_settings: ReceiptRetentionSettings,
    rng_value: float,
    force_metadata: bool,
    now: datetime | None = None,
) -> dict[str, Any]:
    clock = now or datetime.now(timezone.utc)
    if clock.tzinfo is None:
        clock = clock.replace(tzinfo=timezone.utc)

    classification = classify_receipt(
        receipt, settings=retention_settings, rng_value=rng_value
    )
    is_full = classification.is_full_payload and not force_metadata
    receipt_json = compact_receipt_json(receipt, is_full_payload=is_full)
    if force_metadata and classification.receipt_kind != "error":
        is_full = False
        receipt_json = compact_receipt_json(receipt, is_full_payload=False)

    expires_at = retention_expires_at(
        classification, settings=retention_settings, now=clock
    )

    return {
        "receipt_id": receipt.receipt_id,
        "organ_id": receipt.organ_id,
        "emission_id": receipt.emission_id,
        "receipt_json": receipt_json,
        "created_at": clock,
        "receipt_kind": classification.receipt_kind,
        "receipt_status": classification.receipt_status,
        "event_id": primary_event_id(receipt),
        "delta_id": primary_delta_id(receipt),
        "reducer_name": primary_reducer_name(receipt),
        "stream_name": None,
        "payload_hash": payload_fingerprint(receipt),
        "payload_bytes": payload_byte_length(receipt_json),
        "is_full_payload": is_full,
        "expires_at": expires_at,
    }


def _trace_id_like_pattern(trace_prefix: str) -> str:
    return f"{trace_prefix}%"


def _cursor_lag_seconds(last_created_at: datetime | None) -> float:
    if last_created_at is None:
        return float("inf")
    ts = last_created_at
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - ts).total_seconds())


class BiometricsSubstrateStore:
    def __init__(self, postgres_uri: str) -> None:
        self._engine: Engine = create_engine(
            postgres_uri,
            pool_pre_ping=True,
            json_serializer=json.dumps,
            json_deserializer=json.loads,
        )

    def _seed_grammar_cursor_at_tail(
        self,
        conn: Any,
        *,
        cursor_name: str,
        source_services: tuple[str, ...],
        trace_prefix: str,
        reason: str,
        prior_created_at: datetime | None = None,
        prior_event_id: str | None = None,
        operator_initiated: bool = False,
    ) -> None:
        trace_like = _trace_id_like_pattern(trace_prefix)
        tail = conn.execute(
            text(
                """
                SELECT created_at, event_id
                FROM grammar_events
                WHERE source_service = ANY(:source_services)
                  AND trace_id LIKE :trace_like
                ORDER BY created_at DESC, event_id DESC
                LIMIT 1
                """
            ),
            {
                "source_services": list(source_services),
                "trace_like": trace_like,
            },
        ).mappings().first()

        now = datetime.now(timezone.utc)
        seeded_created_at = tail["created_at"] if tail else now
        seeded_event_id = tail["event_id"] if tail else ""
        self._write_grammar_cursor(
            conn,
            cursor_name=cursor_name,
            created_at=seeded_created_at,
            event_id=seeded_event_id,
        )
        if not operator_initiated:
            record_tail_seed(
                TailSeedRecord(
                    cursor_name=cursor_name,
                    reason=reason,
                    at=now,
                    prior_created_at=prior_created_at,
                    prior_event_id=prior_event_id,
                    seeded_created_at=seeded_created_at,
                    seeded_event_id=seeded_event_id,
                )
            )

    def _write_grammar_cursor(
        self,
        conn: Any,
        *,
        cursor_name: str,
        created_at: datetime,
        event_id: str,
    ) -> None:
        now = datetime.now(timezone.utc)
        conn.execute(
            text(
                """
                INSERT INTO substrate_reduction_cursor (
                    cursor_name, last_event_created_at, last_event_id, updated_at
                ) VALUES (
                    :cursor_name, :created_at, :event_id, :updated_at
                )
                ON CONFLICT (cursor_name) DO UPDATE SET
                    last_event_created_at = EXCLUDED.last_event_created_at,
                    last_event_id = EXCLUDED.last_event_id,
                    updated_at = EXCLUDED.updated_at
                """
            ),
            {
                "cursor_name": cursor_name,
                "created_at": created_at,
                "event_id": event_id,
                "updated_at": now,
            },
        )

    def _ensure_grammar_cursor_at_tail(
        self,
        conn: Any,
        *,
        cursor_name: str,
        source_services: tuple[str, ...],
        trace_prefix: str,
        max_lag_sec: float,
        tail_seed_on_lag: bool,
    ) -> None:
        row = conn.execute(
            text(
                """
                SELECT last_event_created_at, last_event_id
                FROM substrate_reduction_cursor
                WHERE cursor_name = :cursor_name
                """
            ),
            {"cursor_name": cursor_name},
        ).mappings().first()
        if not row or not row["last_event_created_at"]:
            self._seed_grammar_cursor_at_tail(
                conn,
                cursor_name=cursor_name,
                source_services=source_services,
                trace_prefix=trace_prefix,
                reason="cold_start",
            )
            return

        lag_sec = _cursor_lag_seconds(row["last_event_created_at"])
        if tail_seed_on_lag and lag_sec > max_lag_sec:
            self._seed_grammar_cursor_at_tail(
                conn,
                cursor_name=cursor_name,
                source_services=source_services,
                trace_prefix=trace_prefix,
                reason="lag_exceeded",
                prior_created_at=row["last_event_created_at"],
                prior_event_id=row["last_event_id"],
            )
        elif lag_sec > max_lag_sec:
            logger.warning(
                "substrate_cursor_lag cursor=%s lag_sec=%.0f max_lag_sec=%.0f "
                "tail_seed_on_lag=false continuing_from_stored_cursor",
                cursor_name,
                lag_sec,
                max_lag_sec,
            )

    def _fetch_grammar_events(
        self,
        *,
        cursor_name: str,
        source_services: tuple[str, ...],
        trace_prefix: str,
        limit: int,
    ) -> list[GrammarEventV1]:
        trace_like = _trace_id_like_pattern(trace_prefix)
        app_settings = get_settings()
        max_lag_sec = float(app_settings.substrate_cursor_lag_resync_hours) * 3600.0
        with self._engine.begin() as conn:
            self._ensure_grammar_cursor_at_tail(
                conn,
                cursor_name=cursor_name,
                source_services=source_services,
                trace_prefix=trace_prefix,
                max_lag_sec=max_lag_sec,
                tail_seed_on_lag=app_settings.substrate_cursor_tail_seed_on_lag,
            )
            row = conn.execute(
                text(
                    """
                    SELECT last_event_created_at, last_event_id
                    FROM substrate_reduction_cursor
                    WHERE cursor_name = :cursor_name
                    """
                ),
                {"cursor_name": cursor_name},
            ).mappings().first()
            if not row or not row["last_event_created_at"]:
                return []

            rows = conn.execute(
                text(
                    """
                    SELECT event_id, event_json, created_at
                    FROM grammar_events
                    WHERE source_service = ANY(:source_services)
                      AND trace_id LIKE :trace_like
                      AND (
                        created_at > :cursor_ts
                        OR (created_at = :cursor_ts AND event_id > :cursor_id)
                      )
                    ORDER BY created_at ASC, event_id ASC
                    LIMIT :limit
                    """
                ),
                {
                    "source_services": list(source_services),
                    "trace_like": trace_like,
                    "cursor_ts": row["last_event_created_at"],
                    "cursor_id": row["last_event_id"] or "",
                    "limit": limit,
                },
            ).mappings().all()

        events: list[GrammarEventV1] = []
        for r in rows:
            payload = r["event_json"]
            if isinstance(payload, str):
                payload = json.loads(payload)
            events.append(GrammarEventV1.model_validate(payload))
        return events

    def fetch_biometrics_grammar_events(self, *, limit: int = 50) -> list[GrammarEventV1]:
        return self._fetch_grammar_events(
            cursor_name=GRAMMAR_CURSOR_NAME,
            source_services=("orion-biometrics",),
            trace_prefix="biometrics.node:",
            limit=limit,
        )

    def fetch_execution_grammar_events(self, *, limit: int = 50) -> list[GrammarEventV1]:
        return self._fetch_grammar_events(
            cursor_name=EXECUTION_GRAMMAR_CURSOR_NAME,
            source_services=EXECUTION_GRAMMAR_SOURCE_SERVICES,
            trace_prefix="cortex.exec:",
            limit=limit,
        )

    def fetch_transport_grammar_events(self, *, limit: int = 50) -> list[GrammarEventV1]:
        return self._fetch_grammar_events(
            cursor_name=TRANSPORT_GRAMMAR_CURSOR_NAME,
            source_services=("orion-bus",),
            trace_prefix="bus.transport:",
            limit=limit,
        )

    def fetch_chat_grammar_events(self, *, limit: int = 100) -> list[GrammarEventV1]:
        return self._fetch_grammar_events(
            cursor_name=CHAT_GRAMMAR_CURSOR_NAME,
            source_services=(CHAT_SOURCE_SERVICE,),
            trace_prefix=CHAT_TRACE_PREFIX,
            limit=limit,
        )

    def advance_cursor(self, *, event_id: str, created_at: datetime) -> None:
        now = datetime.now(timezone.utc)
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO substrate_reduction_cursor (
                        cursor_name, last_event_created_at, last_event_id, updated_at
                    ) VALUES (
                        :cursor_name, :created_at, :event_id, :updated_at
                    )
                    ON CONFLICT (cursor_name) DO UPDATE SET
                        last_event_created_at = EXCLUDED.last_event_created_at,
                        last_event_id = EXCLUDED.last_event_id,
                        updated_at = EXCLUDED.updated_at
                    """
                ),
                {
                    "cursor_name": GRAMMAR_CURSOR_NAME,
                    "created_at": created_at,
                    "event_id": event_id,
                    "updated_at": now,
                },
            )

    def advance_execution_cursor(self, *, event_id: str, created_at: datetime) -> None:
        now = datetime.now(timezone.utc)
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO substrate_reduction_cursor (
                        cursor_name, last_event_created_at, last_event_id, updated_at
                    ) VALUES (
                        :cursor_name, :created_at, :event_id, :updated_at
                    )
                    ON CONFLICT (cursor_name) DO UPDATE SET
                        last_event_created_at = EXCLUDED.last_event_created_at,
                        last_event_id = EXCLUDED.last_event_id,
                        updated_at = EXCLUDED.updated_at
                    """
                ),
                {
                    "cursor_name": EXECUTION_GRAMMAR_CURSOR_NAME,
                    "created_at": created_at,
                    "event_id": event_id,
                    "updated_at": now,
                },
            )

    def advance_transport_cursor(self, *, event_id: str, created_at: datetime) -> None:
        now = datetime.now(timezone.utc)
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO substrate_reduction_cursor (
                        cursor_name, last_event_created_at, last_event_id, updated_at
                    ) VALUES (
                        :cursor_name, :created_at, :event_id, :updated_at
                    )
                    ON CONFLICT (cursor_name) DO UPDATE SET
                        last_event_created_at = EXCLUDED.last_event_created_at,
                        last_event_id = EXCLUDED.last_event_id,
                        updated_at = EXCLUDED.updated_at
                    """
                ),
                {
                    "cursor_name": TRANSPORT_GRAMMAR_CURSOR_NAME,
                    "created_at": created_at,
                    "event_id": event_id,
                    "updated_at": now,
                },
            )

    def advance_chat_cursor(self, *, event_id: str, created_at: datetime) -> None:
        now = datetime.now(timezone.utc)
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO substrate_reduction_cursor (
                        cursor_name, last_event_created_at, last_event_id, updated_at
                    ) VALUES (
                        :cursor_name, :created_at, :event_id, :updated_at
                    )
                    ON CONFLICT (cursor_name) DO UPDATE SET
                        last_event_created_at = EXCLUDED.last_event_created_at,
                        last_event_id = EXCLUDED.last_event_id,
                        updated_at = EXCLUDED.updated_at
                    """
                ),
                {
                    "cursor_name": CHAT_GRAMMAR_CURSOR_NAME,
                    "created_at": created_at,
                    "event_id": event_id,
                    "updated_at": now,
                },
            )

    def _load_projection(self, table: str, projection_id: str, model: type) -> Any:
        with self._engine.connect() as conn:
            row = conn.execute(
                text(
                    f"""
                    SELECT projection_json FROM {table}
                    WHERE projection_id = :projection_id
                    """
                ),
                {"projection_id": projection_id},
            ).mappings().first()
        if not row:
            return None
        payload = row["projection_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        return model.model_validate(payload)

    def _save_projection(self, table: str, projection: Any) -> None:
        now = datetime.now(timezone.utc)
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    f"""
                    INSERT INTO {table} (projection_id, generated_at, projection_json, created_at)
                    VALUES (:projection_id, :generated_at, :projection_json, :created_at)
                    ON CONFLICT (projection_id) DO UPDATE SET
                        generated_at = EXCLUDED.generated_at,
                        projection_json = EXCLUDED.projection_json
                    """
                ),
                {
                    "projection_id": projection.projection_id,
                    "generated_at": projection.generated_at,
                    "projection_json": Json(projection.model_dump(mode="json")),
                    "created_at": now,
                },
            )

    def load_attention_broadcast(
        self, projection_id: str = "substrate.attention.broadcast.v1"
    ) -> AttentionBroadcastProjectionV1 | None:
        return self._load_projection(
            "substrate_attention_broadcast_projection",
            projection_id,
            AttentionBroadcastProjectionV1,
        )

    def save_attention_broadcast(self, projection: AttentionBroadcastProjectionV1) -> None:
        self._save_projection("substrate_attention_broadcast_projection", projection)

    def save_endogenous_curiosity_candidates(self, signals: list[Any]) -> None:
        """Persist one bounded candidate set for the felt-state curiosity lane.

        Inserts a single row whose ``candidates_json`` is the JSON array of the
        provided signals, then prunes rows older than 24h so the table stays
        bounded. Caller caps the list; this method is a plain writer.
        """
        now = datetime.now(timezone.utc)
        candidates = [sig.model_dump(mode="json") for sig in signals]
        digest = hashlib.sha256(
            "|".join([now.isoformat()] + sorted(str(c.get("signal_id", "")) for c in candidates)).encode("utf-8")
        ).hexdigest()[:24]
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO substrate_endogenous_curiosity_candidates (
                        candidate_set_id, generated_at, candidates_json, created_at
                    ) VALUES (
                        :candidate_set_id, :generated_at, :candidates_json, :created_at
                    )
                    ON CONFLICT (candidate_set_id) DO NOTHING
                    """
                ),
                {
                    "candidate_set_id": f"curiosity-{digest}",
                    "generated_at": now,
                    "candidates_json": Json(candidates),
                    "created_at": now,
                },
            )
            conn.execute(
                text(
                    """
                    DELETE FROM substrate_endogenous_curiosity_candidates
                    WHERE generated_at < now() - interval '24 hours'
                    """
                ),
            )

    def save_coalition_dwell(self, projection: AttentionBroadcastProjectionV1) -> None:
        """Append one dwell row per broadcast tick; prunes rows older than 24h."""
        generated_at = projection.generated_at
        attended = sorted(str(node_id) for node_id in projection.attended_node_ids)
        digest = hashlib.sha256(
            "|".join([generated_at.isoformat()] + attended).encode("utf-8")
        ).hexdigest()[:24]
        dwell_ticks = int(projection.dwell_ticks)
        now = datetime.now(timezone.utc)
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO substrate_coalition_dwell_log (
                        dwell_id, generated_at, coalition_ids, candidate_ticks,
                        active, dwell_ticks, salience_trend, created_at
                    ) VALUES (
                        :dwell_id, :generated_at, :coalition_ids, :candidate_ticks,
                        :active, :dwell_ticks, :salience_trend, :created_at
                    )
                    ON CONFLICT (dwell_id) DO NOTHING
                    """
                ),
                {
                    "dwell_id": f"dwell-{digest}",
                    "generated_at": generated_at,
                    "coalition_ids": Json(list(projection.attended_node_ids)),
                    "candidate_ticks": dwell_ticks,
                    "active": dwell_ticks > 0,
                    "dwell_ticks": dwell_ticks,
                    "salience_trend": float(projection.coalition_stability_score),
                    "created_at": now,
                },
            )
            conn.execute(
                text(
                    """
                    DELETE FROM substrate_coalition_dwell_log
                    WHERE generated_at < now() - interval '24 hours'
                    """
                ),
            )

    def save_brain_frame(self, frame, retention_hours: int = 24) -> None:
        """Append one brain-frame row per tick; prune rows beyond retention."""
        now = datetime.now(timezone.utc)
        payload = frame.model_dump(mode="json")
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO substrate_brain_frame_log (
                        frame_id, tick_seq, generated_at, phase, frame_json, created_at
                    ) VALUES (
                        :frame_id, :tick_seq, :generated_at, :phase, :frame_json, :created_at
                    )
                    ON CONFLICT (frame_id) DO NOTHING
                    """
                ),
                {
                    "frame_id": frame.frame_id,
                    "tick_seq": int(frame.tick_seq),
                    "generated_at": frame.generated_at,
                    "phase": frame.phase,
                    "frame_json": Json(payload),
                    "created_at": now,
                },
            )
            conn.execute(
                text(
                    f"""
                    DELETE FROM substrate_brain_frame_log
                    WHERE generated_at < now() - interval '{int(retention_hours)} hours'
                    """
                ),
            )

    def load_brain_frames_tail(self, limit: int = 1) -> list[dict]:
        """Return the most-recent N frame payloads, ascending by generated_at."""
        limit = max(1, min(int(limit), 120))
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT frame_json FROM substrate_brain_frame_log
                    ORDER BY generated_at DESC LIMIT :limit
                    """
                ),
                {"limit": limit},
            ).mappings().all()
        frames = [self._coerce_frame_json(r["frame_json"]) for r in rows]
        frames.reverse()
        return frames

    def load_brain_frames_range(self, start, end, max_frames: int = 240) -> list[dict]:
        """Return frames in [start, end], downsampled to at most max_frames, ascending."""
        max_frames = max(1, min(int(max_frames), 2000))
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT frame_json FROM substrate_brain_frame_log
                    WHERE generated_at >= :start AND generated_at <= :end
                    ORDER BY generated_at ASC
                    """
                ),
                {"start": start, "end": end},
            ).mappings().all()
        frames = [self._coerce_frame_json(r["frame_json"]) for r in rows]
        if len(frames) <= max_frames:
            return frames
        step = len(frames) / max_frames
        return [frames[int(i * step)] for i in range(max_frames)]

    def brain_frame_window(self) -> dict:
        """Return retention bounds + earliest/latest frame ts + current phase."""
        with self._engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT
                      min(generated_at) AS earliest,
                      max(generated_at) AS latest,
                      count(*) AS n
                    FROM substrate_brain_frame_log
                    """
                ),
            ).mappings().first()
            phase_row = conn.execute(
                text(
                    """
                    SELECT phase FROM substrate_brain_frame_log
                    ORDER BY generated_at DESC LIMIT 1
                    """
                ),
            ).mappings().first()
        earliest = row["earliest"] if row else None
        latest = row["latest"] if row else None
        return {
            "earliest": earliest.isoformat() if hasattr(earliest, "isoformat") else earliest,
            "latest": latest.isoformat() if hasattr(latest, "isoformat") else latest,
            "frame_count": int(row["n"]) if row else 0,
            "phase": (phase_row["phase"] if phase_row else None),
        }

    @staticmethod
    def _coerce_frame_json(value):
        if isinstance(value, str):
            return json.loads(value)
        return value

    def load_node_biometrics(self, projection_id: str) -> NodeBiometricsProjectionV1 | None:
        return self._load_projection(
            "substrate_node_biometrics_projection",
            projection_id,
            NodeBiometricsProjectionV1,
        )

    def save_node_biometrics(self, projection: NodeBiometricsProjectionV1) -> None:
        self._save_projection("substrate_node_biometrics_projection", projection)

    def load_active_pressure(self, projection_id: str) -> ActiveNodePressureProjectionV1 | None:
        return self._load_projection(
            "substrate_active_node_pressure_projection",
            projection_id,
            ActiveNodePressureProjectionV1,
        )

    def save_active_pressure(self, projection: ActiveNodePressureProjectionV1) -> None:
        self._save_projection("substrate_active_node_pressure_projection", projection)

    def load_execution_trajectory(
        self, projection_id: str
    ) -> ExecutionTrajectoryProjectionV1 | None:
        return self._load_projection(
            "substrate_execution_trajectory_projection",
            projection_id,
            ExecutionTrajectoryProjectionV1,
        )

    def save_execution_trajectory(self, projection: ExecutionTrajectoryProjectionV1) -> None:
        self._save_projection("substrate_execution_trajectory_projection", projection)

    def load_chat_session_projection(
        self, projection_id: str
    ) -> ChatSessionProjectionV1 | None:
        return self._load_projection(
            "substrate_chat_session_projection",
            projection_id,
            ChatSessionProjectionV1,
        )

    def save_chat_session_projection(self, projection: ChatSessionProjectionV1) -> None:
        self._save_projection("substrate_chat_session_projection", projection)

    def load_transport_bus_projection(
        self, projection_id: str = TRANSPORT_BUS_PROJECTION_ID
    ) -> TransportBusProjectionV1 | None:
        with self._engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT projection_json FROM substrate_transport_bus_projection
                    WHERE projection_id = :projection_id
                    """
                ),
                {"projection_id": projection_id},
            ).mappings().first()
        if not row:
            return None
        payload = row["projection_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        return TransportBusProjectionV1.model_validate(payload)

    def save_transport_bus_projection(self, projection: TransportBusProjectionV1) -> None:
        now = datetime.now(timezone.utc)
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO substrate_transport_bus_projection (
                        projection_id, projection_json, updated_at
                    ) VALUES (
                        :projection_id, :projection_json, :updated_at
                    )
                    ON CONFLICT (projection_id) DO UPDATE SET
                        projection_json = EXCLUDED.projection_json,
                        updated_at = EXCLUDED.updated_at
                    """
                ),
                {
                    "projection_id": projection.projection_id,
                    "projection_json": Json(projection.model_dump(mode="json")),
                    "updated_at": projection.updated_at,
                },
            )

    def save_receipt(self, receipt: ReductionReceiptV1) -> None:
        from app.receipt_pruner import get_cached_pressure_state

        s = get_settings()
        retention_settings = _retention_settings_from_app(s)
        disk_critical, table_critical = get_cached_pressure_state()
        force_metadata = (
            s.receipt_emergency_metadata_only and (disk_critical or table_critical)
        )

        params = _build_receipt_insert_params(
            receipt,
            retention_settings=retention_settings,
            rng_value=random.random(),
            force_metadata=force_metadata,
        )
        insert_params = {
            **params,
            "receipt_json": Json(params["receipt_json"]),
        }

        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO substrate_reduction_receipts (
                        receipt_id, organ_id, emission_id, receipt_json, created_at,
                        receipt_kind, receipt_status, event_id, delta_id, reducer_name,
                        stream_name, payload_hash, payload_bytes, is_full_payload,
                        expires_at
                    ) VALUES (
                        :receipt_id, :organ_id, :emission_id, :receipt_json, :created_at,
                        :receipt_kind, :receipt_status, :event_id, :delta_id, :reducer_name,
                        :stream_name, :payload_hash, :payload_bytes, :is_full_payload,
                        :expires_at
                    )
                    ON CONFLICT (receipt_id) DO NOTHING
                    """
                ),
                insert_params,
            )

    def fetch_receipts_between(
        self,
        *,
        start: datetime,
        end: datetime,
        limit: int = 500,
    ) -> list[ReductionReceiptV1]:
        """Receipts in [start, end] for episodic consolidation, oldest first.

        Metadata-only rows (compact receipt_json) still validate — absent event
        lists default to empty. Unparseable rows are skipped, not fatal.
        """
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT receipt_json FROM substrate_reduction_receipts
                    WHERE created_at >= :start AND created_at <= :end
                    ORDER BY created_at ASC, receipt_id ASC
                    LIMIT :limit
                    """
                ),
                {"start": start, "end": end, "limit": limit},
            ).mappings().all()
        receipts: list[ReductionReceiptV1] = []
        for row in rows:
            payload = row["receipt_json"]
            if isinstance(payload, str):
                payload = json.loads(payload)
            try:
                receipts.append(ReductionReceiptV1.model_validate(payload))
            except Exception:
                logger.warning("substrate_receipt_parse_failed", exc_info=True)
        return receipts

    def save_episode_summary(self, episode: EpisodeSummaryV1) -> bool:
        """Insert a proposal-marked episode; idempotent on episode_id."""
        with self._engine.begin() as conn:
            result = conn.execute(
                text(
                    """
                    INSERT INTO substrate_episode_summaries (
                        episode_id, status, window_start, window_end,
                        episode_json, created_at
                    ) VALUES (
                        :episode_id, :status, :window_start, :window_end,
                        :episode_json, :created_at
                    )
                    ON CONFLICT (episode_id) DO NOTHING
                    """
                ),
                {
                    "episode_id": episode.episode_id,
                    "status": episode.status,
                    "window_start": episode.window_start,
                    "window_end": episode.window_end,
                    "episode_json": Json(episode.model_dump(mode="json")),
                    "created_at": episode.created_at,
                },
            )
            return bool(result.rowcount)

    def fetch_recent_episode_summaries(self, *, limit: int = 20) -> list[EpisodeSummaryV1]:
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT episode_json FROM substrate_episode_summaries
                    ORDER BY window_end DESC
                    LIMIT :limit
                    """
                ),
                {"limit": limit},
            ).mappings().all()
        episodes: list[EpisodeSummaryV1] = []
        for row in rows:
            payload = row["episode_json"]
            if isinstance(payload, str):
                payload = json.loads(payload)
            try:
                episodes.append(EpisodeSummaryV1.model_validate(payload))
            except Exception:
                logger.warning("substrate_episode_parse_failed", exc_info=True)
        return episodes

    def prune_episode_summaries(self, *, older_than: datetime) -> int:
        with self._engine.begin() as conn:
            result = conn.execute(
                text(
                    """
                    DELETE FROM substrate_episode_summaries
                    WHERE created_at < :older_than
                    """
                ),
                {"older_than": older_than},
            )
            return int(result.rowcount or 0)

    def save_emission(self, emission: OrganEmissionV1) -> None:
        now = datetime.now(timezone.utc)
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO substrate_organ_emissions (
                        emission_id, organ_id, invocation_id, emission_json, created_at
                    ) VALUES (
                        :emission_id, :organ_id, :invocation_id, :emission_json, :created_at
                    )
                    ON CONFLICT (emission_id) DO NOTHING
                    """
                ),
                {
                    "emission_id": emission.emission_id,
                    "organ_id": emission.organ_id,
                    "invocation_id": emission.invocation_id,
                    "emission_json": Json(emission.model_dump(mode="json")),
                    "created_at": now,
                },
            )

    def latest_emission_for_node(self, node_id: str) -> OrganEmissionV1 | None:
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT emission_json FROM substrate_organ_emissions
                    WHERE organ_id = 'biometrics_pressure'
                    ORDER BY created_at DESC
                    LIMIT 50
                    """
                ),
            ).mappings().all()
        for row in rows:
            payload = row["emission_json"]
            if isinstance(payload, str):
                payload = json.loads(payload)
            emission = OrganEmissionV1.model_validate(payload)
            if emission_touches_node(emission, node_id):
                return emission
        return None

    def latest_receipt_for_node(self, node_id: str) -> ReductionReceiptV1 | None:
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT receipt_json FROM substrate_reduction_receipts
                    ORDER BY created_at DESC
                    LIMIT 50
                    """
                ),
            ).mappings().all()
        for row in rows:
            payload = row["receipt_json"]
            if isinstance(payload, str):
                payload = json.loads(payload)
            receipt = ReductionReceiptV1.model_validate(payload)
            if receipt_touches_node(receipt, node_id):
                return receipt
        return None

    def load_receipt(self, receipt_id: str) -> ReductionReceiptV1 | None:
        with self._engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT receipt_json FROM substrate_reduction_receipts
                    WHERE receipt_id = :receipt_id
                    """
                ),
                {"receipt_id": receipt_id},
            ).mappings().first()
        if not row:
            return None
        payload = row["receipt_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        return ReductionReceiptV1.model_validate(payload)

    def grammar_cursor_metrics(self, cursor_name: str) -> dict[str, Any]:
        """Pending backlog and stream lag for a grammar reducer cursor."""
        spec = GRAMMAR_CURSOR_REGISTRY.get(cursor_name)
        if spec is None:
            raise ValueError(f"unknown cursor_name: {cursor_name}")
        source_services, trace_prefix = spec
        trace_like = _trace_id_like_pattern(trace_prefix)

        with self._engine.connect() as conn:
            cursor_row = conn.execute(
                text(
                    """
                    SELECT last_event_created_at, last_event_id
                    FROM substrate_reduction_cursor
                    WHERE cursor_name = :cursor_name
                    """
                ),
                {"cursor_name": cursor_name},
            ).mappings().first()
            head_row = conn.execute(
                text(
                    """
                    SELECT created_at, event_id
                    FROM grammar_events
                    WHERE source_service = ANY(:source_services)
                      AND trace_id LIKE :trace_like
                    ORDER BY created_at DESC, event_id DESC
                    LIMIT 1
                    """
                ),
                {"source_services": list(source_services), "trace_like": trace_like},
            ).mappings().first()

            pending = 0
            if cursor_row and cursor_row["last_event_created_at"]:
                pending = int(
                    conn.execute(
                        text(
                            """
                            SELECT COUNT(*)
                            FROM grammar_events
                            WHERE source_service = ANY(:source_services)
                              AND trace_id LIKE :trace_like
                              AND (
                                created_at > :cursor_ts
                                OR (
                                  created_at = :cursor_ts
                                  AND event_id > :cursor_id
                                )
                              )
                            """
                        ),
                        {
                            "source_services": list(source_services),
                            "trace_like": trace_like,
                            "cursor_ts": cursor_row["last_event_created_at"],
                            "cursor_id": cursor_row["last_event_id"] or "",
                        },
                    ).scalar()
                    or 0
                )

        cursor_ts = cursor_row["last_event_created_at"] if cursor_row else None
        cursor_wall_lag = _cursor_lag_seconds(cursor_ts) if cursor_ts else None
        stream_lag: float | None = None
        if cursor_ts and head_row and head_row["created_at"]:
            head_ts = head_row["created_at"]
            if head_ts.tzinfo is None:
                head_ts = head_ts.replace(tzinfo=timezone.utc)
            cts = cursor_ts if cursor_ts.tzinfo else cursor_ts.replace(tzinfo=timezone.utc)
            stream_lag = max(0.0, (head_ts - cts).total_seconds())

        return {
            "cursor_name": cursor_name,
            "pending_backlog": pending,
            "stream_lag_sec": None if stream_lag == float("inf") else stream_lag,
            "cursor_wall_lag_sec": None if cursor_wall_lag == float("inf") else cursor_wall_lag,
            "head_event_created_at": (
                head_row["created_at"].isoformat() if head_row and head_row["created_at"] else None
            ),
            "head_event_id": head_row["event_id"] if head_row else None,
        }

    def cursor_positions(self) -> list[dict[str, Any]]:
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT cursor_name, last_event_created_at, last_event_id, updated_at
                    FROM substrate_reduction_cursor
                    ORDER BY cursor_name
                    """
                ),
            ).mappings().all()
        out: list[dict[str, Any]] = []
        for row in rows:
            last_at = row["last_event_created_at"]
            lag_sec = _cursor_lag_seconds(last_at) if last_at else None
            out.append(
                {
                    "cursor_name": row["cursor_name"],
                    "last_event_created_at": last_at.isoformat() if last_at else None,
                    "last_event_id": row["last_event_id"],
                    "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
                    "lag_sec": None if lag_sec == float("inf") else lag_sec,
                }
            )
        return out

    def reset_grammar_cursor(
        self,
        *,
        cursor_name: str,
        mode: str,
        at_timestamp: datetime | None = None,
    ) -> dict[str, Any]:
        """Operator recovery: reset reducer cursor. Caller must validate auth/mode/name."""
        spec = GRAMMAR_CURSOR_REGISTRY.get(cursor_name)
        if spec is None:
            raise ValueError(f"unknown cursor_name: {cursor_name}")
        source_services, trace_prefix = spec
        trace_like = _trace_id_like_pattern(trace_prefix)
        mode_norm = mode.strip().lower()

        def _finish(
            *,
            prior_created_at: str | None,
            prior_event_id: str | None,
            new_created_at: datetime,
            new_event_id: str,
            history_may_be_skipped: bool,
            extra: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            out: dict[str, Any] = {
                "cursor_name": cursor_name,
                "mode": mode_norm,
                "prior_created_at": prior_created_at,
                "prior_event_id": prior_event_id,
                "new_created_at": new_created_at.isoformat(),
                "new_event_id": new_event_id,
                "history_may_be_skipped": history_may_be_skipped,
            }
            if extra:
                out.update(extra)
            return out

        with self._engine.begin() as conn:
            prior_row = conn.execute(
                text(
                    """
                    SELECT last_event_created_at, last_event_id
                    FROM substrate_reduction_cursor
                    WHERE cursor_name = :cursor_name
                    """
                ),
                {"cursor_name": cursor_name},
            ).mappings().first()
            prior_created_at = (
                prior_row["last_event_created_at"].isoformat()
                if prior_row and prior_row["last_event_created_at"]
                else None
            )
            prior_event_id = prior_row["last_event_id"] if prior_row else None

            if mode_norm == "earliest":
                row = conn.execute(
                    text(
                        """
                        SELECT created_at, event_id
                        FROM grammar_events
                        WHERE source_service = ANY(:source_services)
                          AND trace_id LIKE :trace_like
                        ORDER BY created_at ASC, event_id ASC
                        LIMIT 1
                        """
                    ),
                    {"source_services": list(source_services), "trace_like": trace_like},
                ).mappings().first()
                created_at = datetime(1970, 1, 1, tzinfo=timezone.utc)
                event_id = ""
                self._write_grammar_cursor(
                    conn,
                    cursor_name=cursor_name,
                    created_at=created_at,
                    event_id=event_id,
                )
                return _finish(
                    prior_created_at=prior_created_at,
                    prior_event_id=prior_event_id,
                    new_created_at=created_at,
                    new_event_id=event_id,
                    history_may_be_skipped=False,
                    extra={
                        "first_event_created_at": row["created_at"].isoformat() if row else None,
                        "first_event_id": row["event_id"] if row else None,
                    },
                )

            if mode_norm == "tail":
                self._seed_grammar_cursor_at_tail(
                    conn,
                    cursor_name=cursor_name,
                    source_services=source_services,
                    trace_prefix=trace_prefix,
                    reason="operator_tail_reset",
                    operator_initiated=True,
                )
                row = conn.execute(
                    text(
                        """
                        SELECT last_event_created_at, last_event_id
                        FROM substrate_reduction_cursor
                        WHERE cursor_name = :cursor_name
                        """
                    ),
                    {"cursor_name": cursor_name},
                ).mappings().one()
                return _finish(
                    prior_created_at=prior_created_at,
                    prior_event_id=prior_event_id,
                    new_created_at=row["last_event_created_at"],
                    new_event_id=row["last_event_id"] or "",
                    history_may_be_skipped=True,
                )

            if mode_norm == "timestamp":
                if at_timestamp is None:
                    raise ValueError("timestamp mode requires at_timestamp")
                if at_timestamp.tzinfo is None:
                    raise ValueError("timestamp must be timezone-aware")
                row = conn.execute(
                    text(
                        """
                        SELECT created_at, event_id
                        FROM grammar_events
                        WHERE source_service = ANY(:source_services)
                          AND trace_id LIKE :trace_like
                          AND created_at <= :at_ts
                        ORDER BY created_at DESC, event_id DESC
                        LIMIT 1
                        """
                    ),
                    {
                        "source_services": list(source_services),
                        "trace_like": trace_like,
                        "at_ts": at_timestamp,
                    },
                ).mappings().first()
                if not row:
                    created_at = datetime(1970, 1, 1, tzinfo=timezone.utc)
                    event_id = ""
                else:
                    created_at = row["created_at"]
                    event_id = row["event_id"]
                self._write_grammar_cursor(
                    conn,
                    cursor_name=cursor_name,
                    created_at=created_at,
                    event_id=event_id,
                )
                return _finish(
                    prior_created_at=prior_created_at,
                    prior_event_id=prior_event_id,
                    new_created_at=created_at,
                    new_event_id=event_id,
                    history_may_be_skipped=True,
                    extra={"at_timestamp": at_timestamp.isoformat()},
                )

        raise ValueError(f"unsupported cursor reset mode: {mode}")

    def grammar_event_created_at(self, event_id: str) -> datetime | None:
        with self._engine.connect() as conn:
            row = conn.execute(
                text("SELECT created_at FROM grammar_events WHERE event_id = :event_id"),
                {"event_id": event_id},
            ).mappings().first()
        if not row:
            return None
        created_at = row["created_at"]
        if created_at.tzinfo is None:
            return created_at.replace(tzinfo=timezone.utc)
        return created_at

    def latest_receipt(self) -> ReductionReceiptV1 | None:
        with self._engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT receipt_json FROM substrate_reduction_receipts
                    ORDER BY created_at DESC
                    LIMIT 1
                    """
                ),
            ).mappings().first()
        if not row:
            return None
        payload = row["receipt_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        return ReductionReceiptV1.model_validate(payload)

    def save_quarantine(
        self,
        *,
        reducer_key: str,
        cursor_name: str,
        event_id: str,
        trace_id: str | None,
        reason: str,
    ) -> str:
        quarantine_id = f"quarantine:{reducer_key}:{event_id}"
        now = datetime.now(timezone.utc)
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO substrate_reducer_quarantine (
                        quarantine_id, reducer_key, cursor_name, event_id,
                        trace_id, reason, quarantined_at
                    ) VALUES (
                        :quarantine_id, :reducer_key, :cursor_name, :event_id,
                        :trace_id, :reason, :quarantined_at
                    )
                    ON CONFLICT (quarantine_id) DO UPDATE SET
                        reason = EXCLUDED.reason,
                        trace_id = COALESCE(EXCLUDED.trace_id, substrate_reducer_quarantine.trace_id)
                    """
                ),
                {
                    "quarantine_id": quarantine_id,
                    "reducer_key": reducer_key,
                    "cursor_name": cursor_name,
                    "event_id": event_id,
                    "trace_id": trace_id,
                    "reason": reason,
                    "quarantined_at": now,
                },
            )
        return quarantine_id

    def quarantine_summary(
        self,
        *,
        examples_per_reducer: int = 10,
    ) -> dict[str, Any]:
        with self._engine.connect() as conn:
            count_rows = conn.execute(
                text(
                    """
                    SELECT cursor_name, COUNT(*) AS unacked_count
                    FROM substrate_reducer_quarantine
                    WHERE acknowledged_at IS NULL
                    GROUP BY cursor_name
                    """
                ),
            ).mappings().all()
            example_rows = conn.execute(
                text(
                    """
                    SELECT reducer_key, cursor_name, event_id, trace_id, reason,
                           quarantined_at, acknowledged_at
                    FROM substrate_reducer_quarantine
                    WHERE acknowledged_at IS NULL
                    ORDER BY quarantined_at DESC
                    """
                ),
            ).mappings().all()

        unack_by_cursor: dict[str, int] = {
            row["cursor_name"]: int(row["unacked_count"]) for row in count_rows
        }
        unack_by_reducer: dict[str, int] = {}
        examples_by_reducer: dict[str, list[dict[str, Any]]] = {}
        for row in example_rows:
            reducer_key = row["reducer_key"]
            unack_by_reducer[reducer_key] = unack_by_reducer.get(reducer_key, 0) + 1
            bucket = examples_by_reducer.setdefault(reducer_key, [])
            if len(bucket) >= examples_per_reducer:
                continue
            bucket.append(
                {
                    "event_id": row["event_id"],
                    "trace_id": row["trace_id"],
                    "reason": row["reason"],
                    "quarantined_at": (
                        row["quarantined_at"].isoformat() if row["quarantined_at"] else None
                    ),
                }
            )

        quarantine_by_reducer: dict[str, dict[str, Any]] = {}
        for reducer_key, examples in examples_by_reducer.items():
            quarantine_by_reducer[reducer_key] = {
                "unacknowledged_count": unack_by_reducer.get(reducer_key, 0),
                "recent_examples": examples,
            }
        for reducer_key, count in unack_by_reducer.items():
            if reducer_key not in quarantine_by_reducer:
                quarantine_by_reducer[reducer_key] = {
                    "unacknowledged_count": count,
                    "recent_examples": [],
                }

        return {
            "unacknowledged_quarantine_count_by_reducer": unack_by_reducer,
            "unacknowledged_quarantine_count_by_cursor": unack_by_cursor,
            "quarantine_by_reducer": quarantine_by_reducer,
        }

    def acknowledge_quarantine(
        self,
        *,
        cursor_name: str,
        event_id: str | None,
        ack_all: bool,
        actor: str,
    ) -> int:
        now = datetime.now(timezone.utc)
        with self._engine.begin() as conn:
            if ack_all:
                result = conn.execute(
                    text(
                        """
                        UPDATE substrate_reducer_quarantine
                        SET acknowledged_at = :now, acknowledged_by = :actor
                        WHERE cursor_name = :cursor_name
                          AND acknowledged_at IS NULL
                        """
                    ),
                    {
                        "now": now,
                        "actor": actor,
                        "cursor_name": cursor_name,
                    },
                )
                return int(result.rowcount or 0)

            if not event_id:
                raise ValueError("event_id required unless ack_all=true")
            result = conn.execute(
                text(
                    """
                    UPDATE substrate_reducer_quarantine
                    SET acknowledged_at = :now, acknowledged_by = :actor
                    WHERE cursor_name = :cursor_name
                      AND event_id = :event_id
                      AND acknowledged_at IS NULL
                    """
                ),
                {
                    "now": now,
                    "actor": actor,
                    "cursor_name": cursor_name,
                    "event_id": event_id,
                },
            )
            return int(result.rowcount or 0)
