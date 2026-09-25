TRANSPORT_BUS_PROJECTION_ID = "active_transport_bus_projection"
TRANSPORT_GRAMMAR_CURSOR_NAME = "transport_grammar_reducer"
TRANSPORT_SOURCE_SERVICE = "orion-bus"
TRANSPORT_TRACE_PREFIX = "bus.transport:"
TRANSPORT_REDUCER_ID = "transport_bus_reducer"
DEFAULT_STREAM_DEPTH_CRITICAL = 100_000

# Second segments of `bus.transport:<x>:<y>` trace ids that are NOT bus nodes.
#
# orion/core/bus/async_service.py::_emit_rpc_timeout_grammar publishes every
# rpc_request() timeout on `bus.transport:rpc_timeout:<correlation_uuid>` --
# same lane prefix as the bus observer's real `bus.transport:<node>:<window>`
# traces, so parse_bus_transport_trace_id() used to read node_id="rpc_timeout",
# and the reducer minted a fake `bus:rpc_timeout` bus (zero evidence,
# redis_ping_ok=None -> fabricated 0.5 delivery_confidence/reliability_pressure),
# which the field digester turned into node:rpc_timeout -- a dominant attention
# target in 42% of attention frames on 2026-09-19. Fixed consumer-side (not by
# changing the emitter's trace format) because async_service.py is the shared
# bus client inside every service. The timeouts themselves are consumed by
# orion-equilibrium-service's transport_metacog_gate keyed on semantic_role
# "rpc_transport_timeout", and the honest rate-with-denominator lives in
# RpcHealthSnapshotV1 (orion:rpc_health:snapshot) -- this reducer has no
# business turning them into a bus.
NON_BUS_TRANSPORT_NODE_IDS: frozenset[str] = frozenset({"rpc_timeout"})
