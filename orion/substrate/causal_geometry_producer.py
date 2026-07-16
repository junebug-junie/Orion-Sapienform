"""Causal Geometry v1, follow-up rung: the scheduled Phase A -> Phase B producer.

PR #1087 shipped Phase A (`orion/substrate/causal_geometry_engine.py`, observed-
vs-designed measurement) and Phase B (`field_topology_plasticity.py` +
`field_topology_learned_store.py`, HITL-gated weight-patch proposals), but
nothing ever called them together on a schedule -- the HITL proposal queue was
real but permanently empty. This module is that missing link: one function,
`run_causal_geometry_production_cycle()`, that a periodic caller (see
`services/orion-field-digester/app/worker.py`'s `_causal_geometry_producer_loop`)
invokes to measure, propose, and enqueue.

This module only *proposes* into the HITL queue via `store.propose()`. It never
calls `store.adopt()` and never imports `orion.substrate.mutation_apply` --
same hard constraint as `field_topology_plasticity.py` itself.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from orion.schemas.field_state import FieldEdgeV1
from orion.substrate.causal_geometry_engine import (
    DEFAULT_ALPHA,
    DEFAULT_BIOMETRICS_NODE,
    DEFAULT_MIN_SAMPLES,
    DEFAULT_N_SURROGATES,
    DEFAULT_SEED,
    DEFAULT_WINDOW_HOURS,
    build_snapshot,
    fetch_channels,
    load_field_topology,
)
from orion.substrate.field_topology_learned_store import FieldTopologyLearnedWeightsStore
from orion.substrate.field_topology_plasticity import (
    MIN_MEANINGFUL_DELTA,
    propose_field_topology_patches,
)

logger = logging.getLogger(__name__)

# How many pending proposals to scan for de-duplication purposes. Generously
# above any realistic queue size (proposals only exist for diverging cap->cap
# edges with a real delta -- see MIN_MEANINGFUL_DELTA) so a legitimately large
# backlog can't cause a duplicate to slip through list_pending()'s own default
# limit of 50.
DEDUP_SCAN_LIMIT = 1000


def run_causal_geometry_production_cycle(
    *,
    postgres_uri: str,
    topology_path: str,
    field_edges: List[FieldEdgeV1],
    store: FieldTopologyLearnedWeightsStore,
    window_hours: float = DEFAULT_WINDOW_HOURS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    n_surrogates: int = DEFAULT_N_SURROGATES,
    alpha: float = DEFAULT_ALPHA,
    biometrics_node: str = DEFAULT_BIOMETRICS_NODE,
    seed: int = DEFAULT_SEED,
    min_meaningful_delta: float = MIN_MEANINGFUL_DELTA,
    now: datetime | None = None,
) -> Dict[str, Any]:
    """Measure observed-vs-designed field topology and enqueue new HITL proposals.

    Never raises -- every failure mode (Postgres unreachable, topology YAML
    missing/malformed, a candidate-building error) degrades to a returned
    summary dict with `ok: False` and an `error` string, exactly like
    `services/orion-field-digester/app/digestion/diffusion.py`'s
    `_load_learned_overlay()` degrades a read-path failure to an empty
    overlay. A scheduled caller should log this summary, not propagate an
    exception into its own tick loop.

    De-duplication: a proposal is only enqueued for an edge_ref that has no
    existing *pending* proposal already in the store (checked via
    `store.list_pending()`). An edge whose prior proposal was already adopted
    or rejected is eligible to be re-proposed if divergence still exists --
    only an in-flight pending proposal blocks a new one for the same edge.
    """
    moment = now or datetime.now(timezone.utc)
    window_start = moment - timedelta(hours=window_hours)

    try:
        channels, table_row_counts = fetch_channels(
            postgres_uri, window_start, biometrics_node=biometrics_node
        )
        topology = load_field_topology(topology_path)
        snapshot, _pair_results = build_snapshot(
            channels,
            topology,
            window_start=window_start,
            window_end=moment,
            min_samples=min_samples,
            n_surrogates=n_surrogates,
            alpha=alpha,
            seed=seed,
            table_row_counts=table_row_counts,
        )
    except Exception as exc:
        logger.warning("causal_geometry_production_cycle_measurement_failed: %s", exc, exc_info=True)
        return {
            "ok": False,
            "stage": "measurement",
            "error": str(exc),
            "snapshot_id": None,
            "insufficient_data": None,
            "candidates_found": 0,
            "proposals_created": 0,
            "proposals_skipped_pending_duplicate": 0,
        }

    try:
        proposals = propose_field_topology_patches(
            snapshot, field_edges=field_edges, min_meaningful_delta=min_meaningful_delta
        )
        existing_pending_refs = {
            proposal.patch.target_ref for proposal in store.list_pending(limit=DEDUP_SCAN_LIMIT)
        }
        created = 0
        skipped = 0
        for proposal in proposals:
            if proposal.patch.target_ref in existing_pending_refs:
                skipped += 1
                continue
            store.propose(proposal)
            created += 1
    except Exception as exc:
        logger.warning("causal_geometry_production_cycle_proposal_stage_failed: %s", exc, exc_info=True)
        return {
            "ok": False,
            "stage": "proposal",
            "error": str(exc),
            "snapshot_id": snapshot.snapshot_id,
            "insufficient_data": snapshot.insufficient_data,
            "candidates_found": 0,
            "proposals_created": 0,
            "proposals_skipped_pending_duplicate": 0,
        }

    return {
        "ok": True,
        "stage": "completed",
        "error": None,
        "snapshot_id": snapshot.snapshot_id,
        "insufficient_data": snapshot.insufficient_data,
        "candidates_found": len(proposals),
        "proposals_created": created,
        "proposals_skipped_pending_duplicate": skipped,
    }
