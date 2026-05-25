Layer 1: bus-observer emits substrate traces for transport-relevant conditions
Layer 2: traces persist to grammar_events (via orion-sql-writer consumer)
Layer 3: deferred bus_transport_reducer
Layer 4: deferred transport pressure into field lattice
Layer 5: attention may later select transport/backpressure anomalies
Layer 6: self-state may later reflect transport degradation
Layer 7: proposals may later suggest inspect/restart/defer actions
Layer 8: restart/control must require policy/operator review
Layer 9: any bus restart/snapshot/replay dispatch must be dry-run first
Layer 10: transport anomalies and operator actions become feedback
Layer 11: repeated lag/backpressure/schema violations become motifs

Reducer follow-up:
  bus transport traces → bus_transport_reducer → StateDeltaV1(target_kind=transport_bus)
  pressure_hints: bus_health, transport_lag, backpressure, schema_violation_pressure, delivery_confidence
