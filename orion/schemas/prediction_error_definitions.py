"""Which *definition* of each domain's prediction error is live right now.

A prediction-error receipt carries a number, and a number only means something
against the formula that produced it. When a formula changes, every running
average built from the old formula's numbers describes a different quantity.
Candidate A's persisted precision baseline
(``substrate_node_prediction_error_baseline``, advanced by
``services/orion-attention-runtime/app/store.py::advance_node_prediction_error_baseline``)
is exactly such an average, so it must restart when the formula it summarises
changes -- not by a hand-run SQL write, but by this version number moving.

Producer: ``services/orion-substrate-runtime/app/worker.py::_prediction_error_receipt``
stamps ``after.definition_version`` on every prediction-error receipt.
Consumer: the attention store resets a target's baseline when its persisted
``definition_version`` differs from the live one here, and folds only receipts
whose stamped version matches (a receipt with no stamp is version 1).

Bump a domain's number in the same patch that changes what its prediction-error
function measures. Kept dependency-free so the attention runtime can import it
without pulling in ``orion.substrate.prediction_error``'s producer imports.
"""

from __future__ import annotations

# Receipts written before this module existed carry no stamp; they are version 1.
UNSTAMPED_DEFINITION_VERSION = 1

# Keyed by the reducer_key a receipt is written under
# (``reducer_name = "substrate.<reducer_key>"``).
#
# chat_session v2 (2026-09-25): ``topic_coherence`` removed from the chat
#   pressure hints -- it was ``1 - repair_pressure`` and double-weighted repair.
# route_arbitration v2 (2026-09-25): the decision-mismatch rate averages over
#   only the runs a batch actually touched, not every run in the projection.
PREDICTION_ERROR_DEFINITION_VERSIONS: dict[str, int] = {
    "chat_session": 2,
    "route_arbitration": 2,
}


def prediction_error_definition_version(reducer_key: str) -> int:
    """Live definition version for ``reducer_key``; unlisted domains are 1."""
    return PREDICTION_ERROR_DEFINITION_VERSIONS.get(reducer_key, UNSTAMPED_DEFINITION_VERSION)
