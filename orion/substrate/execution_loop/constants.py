EXECUTION_TRAJECTORY_PROJECTION_ID = "active_execution_trajectory"
EXECUTION_GRAMMAR_CURSOR_NAME = "execution_grammar_reducer"
EXECUTION_SOURCE_SERVICES = frozenset({
    "orion-cortex-exec",
    "orion-harness-governor",
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
