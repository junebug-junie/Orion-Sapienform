from __future__ import annotations

from datetime import datetime, timezone

from orion.schemas.identity_snapshot import IdentitySnapshotV1
from orion.schemas.self_state import SelfStateV1


def _snapshot_id(self_state_id: str, generated_at: datetime) -> str:
    return f"identity:{self_state_id}:{generated_at.isoformat()}"


def build_identity_snapshot(
    *,
    self_state: SelfStateV1,
    dominant_drive: str,
    active_drives: list[str],
    key_unknowns: list[str],
    subject: str = "orion",
    now: datetime | None = None,
) -> IdentitySnapshotV1:
    generated_at = now or datetime.now(timezone.utc)
    return IdentitySnapshotV1(
        snapshot_id=_snapshot_id(self_state.self_state_id, generated_at),
        generated_at=generated_at,
        dominant_drive=dominant_drive,
        active_drives=active_drives,
        self_state_condition=self_state.overall_condition,
        overall_intensity=self_state.overall_intensity,
        summary_labels=self_state.summary_labels,
        key_unknowns=key_unknowns,
        trajectory_condition=self_state.trajectory_condition,
        source_self_state_id=self_state.self_state_id,
        source_autonomy_subject=subject,
    )
