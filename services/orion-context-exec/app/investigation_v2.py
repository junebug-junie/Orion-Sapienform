"""PR1 skeleton handler for neutral investigation_v2 mode."""

from __future__ import annotations

from orion.schemas.context_exec import ContextExecRequestV1

INVESTIGATION_V2_SKELETON_MESSAGE = (
    "investigation_v2 skeleton active; evidence sweep not implemented in PR 1"
)
INVESTIGATION_V2_ARTIFACT_TYPE = "InvestigationV2SkeletonV1"


def build_investigation_v2_skeleton_artifact(request: ContextExecRequestV1) -> dict:
    perms = request.permissions
    return {
        "mode": "investigation_v2",
        "permissions_received": perms.model_dump(mode="json"),
        "read_repo": perms.read_repo,
        "message": INVESTIGATION_V2_SKELETON_MESSAGE,
        "text_received": request.text,
    }
