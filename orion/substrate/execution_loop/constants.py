EXECUTION_TRAJECTORY_PROJECTION_ID = "active_execution_trajectory"
EXECUTION_GRAMMAR_CURSOR_NAME = "execution_grammar_reducer"
EXECUTION_SOURCE_SERVICES = frozenset({
    "orion-cortex-exec",
    "orion-harness-governor",
    # Only ever the source of one event kind: exec_turn_timeout, published by
    # orion/hub/turn_orchestrator.py when a harness-governor RPC never returns (the
    # one unified-turn failure mode where the governor side publishes nothing at all).
    # See services/orion-hub/scripts/grammar_emit.py's build_turn_timeout_grammar_events()
    # and docs/superpowers/specs/2026-07-23-fcc-motor-field-digester-signals-design.md.
    "orion-hub",
})
# Keep EXECUTION_SOURCE_SERVICE for backward compat imports; prefer EXECUTION_SOURCE_SERVICES.
EXECUTION_SOURCE_SERVICE = "orion-cortex-exec"
EXECUTION_TRACE_PREFIX = "cortex.exec:"
EXECUTION_REDUCER_ID = "execution_trajectory_reducer"
# Default caps on the runs dict inside ExecutionTrajectoryProjectionV1. Only the
# freshest ~120s of runs are ever consumed downstream, so the dict is pruned by
# LRU (last_updated_at) on every write to prevent unbounded growth.
EXECUTION_TRAJECTORY_MAX_RUNS = 2000
EXECUTION_TRAJECTORY_MAX_AGE_SEC = 86400
