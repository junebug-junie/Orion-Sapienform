"""L6 eval: a seed concept 23 h past observed_at, ticked every 30 s for 30 min
with no re-save, must follow the configured half-life curve (since_last) rather
than the ~2%/tick compounding cliff (legacy). Mirrors the spec's live proof
("smooth half-life curve, not a 2%/tick cliff") on a deterministic fixture.

Run: python -m orion.substrate.evals.run_decay_since_last_eval
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from orion.core.schemas.cognitive_substrate import (
    DEFAULT_CONCEPT_ACTIVATION_HALF_LIFE_SECONDS,
    ConceptNodeV1,
    SubstrateActivationV1,
    SubstrateProvenanceV1,
    SubstrateSignalBundleV1,
)
from orion.substrate.dynamics import SubstrateDynamicsEngine
from orion.substrate.store import InMemorySubstrateGraphStore

T0 = datetime(2026, 10, 6, 3, 0, 0, tzinfo=timezone.utc)
HL = DEFAULT_CONCEPT_ACTIVATION_HALF_LIFE_SECONDS
TICK_S = 30
TICKS = 60  # 30 min


def _store() -> InMemorySubstrateGraphStore:
    store = InMemorySubstrateGraphStore()
    node = ConceptNodeV1(
        node_id="sub-concept-seed-eval",
        label="seed",
        anchor_scope="orion",
        temporal={"observed_at": T0 - timedelta(hours=23)},
        signals=SubstrateSignalBundleV1(
            salience=0.0,
            activation=SubstrateActivationV1(activation=1.0, decay_half_life_seconds=HL, decay_floor=0.0),
        ),
        provenance=SubstrateProvenanceV1(
            authority="local_inferred", source_kind="eval", source_channel="eval", producer="eval"
        ),
        # Already decayed up to T0 (as after the first live tick post-deploy).
        metadata={"activation_decayed_at": T0.isoformat()},
    )
    store.upsert_node(identity_key="id:seed", node=node)
    return store


def _curve(mode: str) -> list[float]:
    store = _store()
    engine = SubstrateDynamicsEngine(store=store, decay_mode=mode)
    out = []
    for i in range(1, TICKS + 1):
        engine.tick(now=T0 + timedelta(seconds=TICK_S * i))
        out.append(store.get_node_by_id("sub-concept-seed-eval").signals.activation.activation)
    return out


def run() -> dict:
    since_last = _curve("since_last")
    legacy = _curve("legacy")
    ideal = [0.5 ** (TICK_S * i / HL) for i in range(1, TICKS + 1)]
    max_err = max(abs(a - b) for a, b in zip(since_last, ideal))
    report = {
        "half_life_seconds": HL,
        "ticks": TICKS,
        "since_last_end": round(since_last[-1], 9),
        "ideal_end": round(ideal[-1], 9),
        "since_last_max_abs_error_vs_ideal": max_err,
        "legacy_end": round(legacy[-1], 6),
        "legacy_first_three": [round(v, 4) for v in legacy[:3]],
    }
    assert max_err < 1e-9, report
    assert legacy[-1] < 0.5, report  # legacy compounds to below half in 30 min
    report["passed"] = True
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
