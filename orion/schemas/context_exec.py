"""Read/write permission envelope carried on every harness turn.

orion-context-exec (the bounded RLM investigation service this module was named
for) was retired 2026-10-10. Its request/run/artifact/proposal schemas were
deleted with it. ``ContextExecPermissionV1`` survives only because the live
harness contract (``HarnessRunRequestV1.permissions`` in
``orion/schemas/harness_finalize.py``) and the Hub turn orchestrator
(``orion/hub/turn_orchestrator.py``) still carry it on every unified turn.
Renaming it is a separate contract migration, not part of the retirement.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ContextExecPermissionV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    read_memory: bool = True
    read_graph: bool = True
    read_recall: bool = True
    read_repo: bool = False
    read_runtime_logs: bool = False
    read_redis_traces: bool = True

    write_memory: bool = False
    write_graph: bool = False
    write_repo: bool = False
    mutate_runtime: bool = False
    network_enabled: bool = False
    shell_enabled: bool = False
