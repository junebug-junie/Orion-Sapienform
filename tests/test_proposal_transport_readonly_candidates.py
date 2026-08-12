from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from orion.proposals.builder import build_proposal_frame
from orion.proposals.policy import load_proposal_policy
from orion.proposals.scoring import PRESSURE_DIMENSIONS
from orion.proposals.templates import FORBIDDEN_TRANSPORT_PROPOSAL_KEYS, TRANSPORT_PROPOSAL_TEMPLATE_KEYS
from orion.schemas.field_state import FieldStateV1

REPO_ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 5, 25, 23, 30, 10, tzinfo=timezone.utc)


def _field() -> FieldStateV1:
    """2026-07-22 (SelfStateV1 burn): broadly-high pressure across every
    real (non-composite) category, mirroring the old fixture's "all dims 0.8"
    intent -- transport template scoring never depended on contract_pressure
    itself (it was already always 0.0 under self_state, unmapped then and
    unmapped now), only on base_priority/policy weights clearing thresholds."""
    return FieldStateV1(
        generated_at=NOW,
        tick_id="tick",
        node_vectors={
            "node:transport-test": {
                "execution_pressure": 0.8,
                "reasoning_pressure": 0.8,
                "reliability_pressure": 0.9,
                "pressure": 0.8,
                "staleness": 0.8,
                "repair_pressure": 0.8,
                "egress_confidence_deficit": 0.8,
            },
        },
        # 2026-08-12 (action_warrant tick gate): real precision-baseline
        # state, elevated. A fixture claiming to be a LOADED field must look
        # loaded to the tick gate as well as to per-template scoring. These
        # dicts were absent entirely, which the gate reads -- correctly -- as
        # every dimension `pinned`/`no_reading`, i.e. "cannot tell", so the
        # frame came back empty. It passed before only because nothing had
        # ever asked the field whether its state warranted acting at all;
        # `base_priority` carried every candidate over `min_priority`
        # regardless of what the field said.
        dimension_precision_ewma_n={dim: 128 for dim in PRESSURE_DIMENSIONS},
        dimension_precision_zscore={dim: 2.5 for dim in PRESSURE_DIMENSIONS},
        dimension_precision_ewma_var={dim: 1.0 for dim in PRESSURE_DIMENSIONS},
    )


def test_transport_readonly_templates_present() -> None:
    policy = load_proposal_policy(REPO_ROOT / "config" / "proposals" / "proposal_policy.v1.yaml")
    frame = build_proposal_frame(field=_field(), attention=None, policy=policy, now=NOW)
    keys = {c.proposal_id.split(":")[1] for c in frame.candidates}
    assert TRANSPORT_PROPOSAL_TEMPLATE_KEYS & keys
    assert not (FORBIDDEN_TRANSPORT_PROPOSAL_KEYS & keys)
    transport_candidates = [
        c for c in frame.candidates if c.proposal_id.split(":")[1] in TRANSPORT_PROPOSAL_TEMPLATE_KEYS
    ]
    assert all(c.required_policy_gate == "read_only" for c in transport_candidates)
