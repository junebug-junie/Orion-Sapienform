ROUTE_ARBITRATION_PROJECTION_ID = "active_route_arbitration"
ROUTE_PROJECTION_ID = ROUTE_ARBITRATION_PROJECTION_ID
ROUTE_GRAMMAR_CURSOR_NAME = "route_grammar_consumer"
ROUTE_SOURCE_SERVICE = "orion-cortex-orch"
ROUTE_TRACE_PREFIX = "orch.route:"
ROUTE_REDUCER_ID = "route_arbitration_reducer"
# Default caps on the runs dict inside RouteArbitrationProjectionV1. Copied
# verbatim from execution_loop's cap constants -- this reducer runs at
# chat-turn volume (same order of magnitude of traffic), and this codebase
# has already hit unbounded-growth incidents on this exact reducer shape
# three times (see [[feedback_substrate_performance]],
# [[feedback_execution_merge_cap]], and commit 8daeecf7). The cap ships in
# the first commit, not as a follow-up.
ROUTE_ARBITRATION_MAX_RUNS = 2000
ROUTE_ARBITRATION_MAX_AGE_SEC = 86400
