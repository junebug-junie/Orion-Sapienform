"""The reducer's domain map must not outlive a domain's producer.

CLAUDE.md section 0A: "When a metric is replaced by a better one, retire the old
one completely -- do not leave a partial exclusion in place." The transport
domain is the worked example of the failure this guards:

- 2026-07-26: `node:substrate.transport`'s producer write was removed.
- 2026-07-31: `_PREDICTION_ERROR_DOMAIN_NODE_IDS` was STILL listing it, so the
  brain-frame reducer read that node every tick and handed `transport` to
  `reduce_attention_self_model()`. Live value at removal: `prediction_error=0.556`,
  `observed_at=2026-07-24` -- seven days stale.

Nothing failed for those five days, which is exactly why a comment is not
sufficient here. This is the failing gate CLAUDE.md asks for instead of a
reminder.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from orion.substrate.attention_self_model import ACTIVE_INFERENCE_DOMAINS

WORKER = Path(__file__).resolve().parents[1] / "app" / "worker.py"
PREDICTION_ERROR = (
    Path(__file__).resolve().parents[3] / "orion" / "substrate" / "prediction_error.py"
)


def _domain_map() -> dict[str, str]:
    """Parse the map out of worker.py without importing it.

    `app.worker` pulls in the whole substrate runtime (Redis, Postgres, FalkorDB
    clients) at import time, which this gate has no reason to require.
    """
    tree = ast.parse(WORKER.read_text())
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(t, ast.Name) and t.id == "_PREDICTION_ERROR_DOMAIN_NODE_IDS"
            for t in node.targets
        ):
            continue
        return ast.literal_eval(node.value)
    pytest.fail("_PREDICTION_ERROR_DOMAIN_NODE_IDS not found in worker.py")


def test_domain_map_matches_active_inference_domains_exactly():
    """Set equality, not subset.

    Subset in one direction alone misses the real bug: a retired domain that
    still has a reader is a superset violation, and a live domain the reducer
    silently stopped reading is a subset violation. Both are the same class of
    "the map and the reducer disagree about what is real".
    """
    mapped = set(_domain_map().values())
    assert mapped == set(ACTIVE_INFERENCE_DOMAINS), (
        "worker.py's _PREDICTION_ERROR_DOMAIN_NODE_IDS and "
        "attention_self_model.ACTIVE_INFERENCE_DOMAINS disagree.\n"
        f"  only in the node map: {sorted(mapped - set(ACTIVE_INFERENCE_DOMAINS))}\n"
        f"  only in the reducer:  {sorted(set(ACTIVE_INFERENCE_DOMAINS) - mapped)}\n"
        "If a domain is being retired, remove it from BOTH in the same patch. "
        "If one is being added, wire it into both, or add it here with a comment "
        "explaining the deliberate asymmetry (see harness_closure/codebase)."
    )


def test_every_mapped_node_id_uses_the_domain_naming_convention():
    for node_id, domain in _domain_map().items():
        assert node_id == f"node:substrate.{domain}", (
            f"{node_id!r} does not follow node:substrate.<domain>; "
            "_write_prediction_error_node() upserts that fixed identity, so a "
            "mismatch here reads a node that is never written."
        )


def test_retired_transport_producer_stays_deleted():
    """`transport_prediction_error()` was deleted 2026-07-31, not just unwired.

    It was previously "kept, not deleted -- deleting it buys nothing", which is
    how a retired instrument gets wired back in. A live-importable symbol is an
    invitation; git history is the archive.
    """
    source = PREDICTION_ERROR.read_text()
    tree = ast.parse(source)
    defined = {
        n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "transport_prediction_error" not in defined, (
        "transport_prediction_error() is back in orion/substrate/prediction_error.py. "
        "It measured a 2-Redis-Stream 'world_pulse' census, not inter-service bus "
        "traffic. The successor is bus_synaptic_prediction_error()."
    )
