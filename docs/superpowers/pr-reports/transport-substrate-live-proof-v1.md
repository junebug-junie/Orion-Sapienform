# Transport Substrate Live Proof V1

**Date:** 2026-06-08
**Status:** Live, dry-run / read-only mode

## Milestones Green

| Layer | Table | Status |
|-------|-------|--------|
| M3 reducer receipts | `substrate_reduction_receipts` | green |
| M3 transport projection | `substrate_transport_bus_projection` | green |
| M4 field state | `substrate_field_state` | green |
| M5 attention frames | `substrate_attention_frames` | green |
| L6 self-state | `substrate_self_state` | green |
| L7 proposal frames | `substrate_proposal_frames` | green |
| L8 policy decision frames | `substrate_policy_decision_frames` | green |
| L9 execution dispatch frames | `substrate_execution_dispatch_frames` | green |
| L10 feedback frames | `substrate_feedback_frames` | green |
| L11 consolidation frames | `substrate_consolidation_frames` | green |

## Known Constraints

- All dispatch runs in `dry_run` mode. No mutations to Redis, SQL catalog, or compose env.
- All proposals require `read_only` policy gate. No autonomous execution.
- Transport proof confirmed via `scripts/smoke_orion_bus_transport_full_stack.sh` modes: `m3`, `m4`, `m5`, `full-observe`.

## Known Issues

- Orphan policy frame queue poison: a stale policy frame can block downstream frames in the queue. Fix is required before enabling non-dry-run dispatch. Tracked separately.

## Next Step

Build Substrate Lattice Hub Tab V1 to surface this proof chain in the UI and allow threshold simulation.
