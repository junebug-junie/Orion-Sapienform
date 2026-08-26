from __future__ import annotations

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.vision import VisionFramePointerPayload, VisionTaskRequestPayload


def make_host_task_envelope(
    *,
    frame_env: BaseEnvelope,
    frame: VisionFramePointerPayload,
    task: VisionTaskRequestPayload,
    service_name: str,
    service_version: str,
    reply_to: str,
) -> BaseEnvelope:
    _ = frame  # lineage anchor; payload lives in task
    return frame_env.derive_child(
        kind="vision.task.request",
        source=ServiceRef(name=service_name, version=service_version),
        payload=task,
        reply_to=reply_to,
    )


def make_secondary_task_envelope(
    *,
    frame_env: BaseEnvelope,
    frame: VisionFramePointerPayload,
    task: VisionTaskRequestPayload,
    service_name: str,
    service_version: str,
    reply_to: str,
    correlation_id: str,
) -> BaseEnvelope:
    """Same as make_host_task_envelope, but with an INDEPENDENT
    correlation_id -- for a second task dispatched off the same frame
    (identity_face alongside the frame's primary task). derive_child's
    default reuses frame_env.correlation_id, which is correct for the
    single task per frame that is this router's normal shape; a second
    task sharing that id would collide in RouterState.pending (keyed by
    corr_id) and both replies would land on the same reply_to channel.

    Review finding, 2026-08-26: an earlier version of this function hand-
    duplicated BaseEnvelope.derive_child's body instead of extending it,
    which would silently drift from any future change to that shared
    primitive (30+ other call sites across the repo). Fixed by adding an
    optional correlation_id override param to derive_child itself
    (backward-compatible default of None -- every other caller unaffected)
    and calling it here instead."""
    _ = frame  # lineage anchor; payload lives in task, same as make_host_task_envelope
    return frame_env.derive_child(
        kind="vision.task.request",
        source=ServiceRef(name=service_name, version=service_version),
        payload=task,
        reply_to=reply_to,
        correlation_id=correlation_id,
    )
