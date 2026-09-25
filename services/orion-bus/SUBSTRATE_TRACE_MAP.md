# orion-bus substrate trace map

| semantic_role | v1 | atom_type | layer | summary hints |
|---------------|----|-----------|-------|---------------|
| bus_observer_tick_started | yes | signal | transport | sample_window_id, node_id |
| bus_health_observed | yes | observation | transport | redis_ping_ok |
| bus_configured_stream_uncataloged | yes | uncertainty_marker | transport | stream_key; "not declared in channel catalog" |
| bus_schema_validation_failed | yes | uncertainty_marker | transport | stream_key, mismatch_count, sampled_count (bounded XREVRANGE sample vs. declared schema_id; contract_pressure) |
| bus_observer_tick_completed | yes | signal | transport | streams_observed count |
| bus_observer_tick_failed | yes | uncertainty_marker | transport | error_kind |
| bus_stream_depth_observed | retired 2026-09-25 | — | — | XLEN is retained length, not backlog; see docs/superpowers/specs/2026-09-25-bus-observer-stream-depth-retirement.md |
| bus_backpressure_observed | retired 2026-09-25 | — | — | never fired in 24,633 live ticks; same spec |
| bus_stream_lag_observed | not planned | — | — | live SCAN 2026-09-25: 5 streams, 1 live consumer group, lag=0 pending=0 |
| bus_delivery_anomaly_observed | deferred | — | — | no subscriber map |
| bus_metrics_* | deferred | — | — | exporter scrape optional |
