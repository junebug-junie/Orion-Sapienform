"""A dispatch-minted act route, for tests of execution-dispatch's act machinery only.

Until 2026-11-05 the painting route (render_scene -> thought.painting `make`, object_source
"dispatch") lived in config/execution_dispatch/execution_dispatch_policy.v1.yaml, and these tests used
it as their example of an act whose object the dispatch mints. Ladder retirement PR 2 moved painting
out of dispatch (orion-thought app/painting_self_start.py), so the real policy has no such route now
(tests/test_rung3_act_candidates.py pins that). The dispatch code that scores, gates and holds an
act-backed route is unchanged and still serves the reading, retro and AI Town routes; it retires
with the whole service in PR 6. Until then its tests run on this copy of the old route, added to the
worker's policy in memory only. Nothing here is read by any service.
"""
from __future__ import annotations

from orion.execution_dispatch.policy import CortexRouteTemplateV1

RENDER_VERB = "skills.imagination.render_scene.v1"
ROUTE = CortexRouteTemplateV1.model_validate({
    "dispatch_kind": "express", "cortex_verb": RENDER_VERB, "cortex_mode": "brain",
    "allowed_scope": "express_bounded", "rpc_timeout_sec": 170,
    "act": {"verb": "make", "actor": "thought.painting", "signal_id": "artifact_produced",
            "horizon_sec": 5400, "object_source": "dispatch"},
})


def with_dispatch_minted_act_route(worker):
    """The worker, with the old painting route back in its (in-memory) policy."""
    policy = worker._policy
    worker._policy = policy.model_copy(update={"template_to_cortex": {**policy.template_to_cortex,
                                                                      "render_scene": ROUTE}})
    return worker
