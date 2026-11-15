"""Hermetic node catalog for the biometrics reducer/organ/sweep suites.

Why this exists (2026-11-14). These suites used to call
`NodeCatalog.load("config/biometrics/node_catalog.yaml")` and lean on hecate as
the standing example of a node the catalog declares `expected_online: false`. That
made a *behaviour* suite depend on *fleet state*: when Juniper brought hecate back
up with a second V100-SXM2 and its biometrics service started answering, PR #3362
flipped that flag to `true` -- correct for the body, and it broke 8 tests in CI
that had nothing to do with the change.

So the expected-offline contract is now exercised against a synthetic box built
here, and the production file is no longer a fixture. The live catalog is still
pinned where pinning it is the actual subject:
`tests/test_fleet_roster_coverage.py` and `tests/test_static_body_descriptions.py`.

Shape notes: `circe` mirrors the real entry's role and declared-true capabilities
(`CIRCE_CAPS` in test_capability_absence_signal.py), because several tests assert
the *real capability names* reaching the projection -- a fixture with empty caps
would make Rule F vacuously pass. `ONDEMAND_BOX` is deliberately not a real host
name, so it can never be mistaken for a machine in the fleet.
"""
from __future__ import annotations

from orion.biometrics.node_catalog import NodeCatalog, NodeProfile

# The expected-offline exemplar. Synthetic on purpose: no such host exists, so no
# future fleet change can move the flag these tests depend on.
ONDEMAND_BOX = "on-demand-testbox"

CIRCE_ROLE = "primary_cognition_gpu"
CIRCE_CAPS_TRUE = [
    "local_llm_heavy",
    "local_llm_quick",
    "training",
    "batch_inference",
    "dream_batch",
]


def make_catalog(*, retired: tuple[str, ...] = ()) -> NodeCatalog:
    """A catalog with one always-online GPU box and one expected-offline box.

    `retired` adds hosts under `retired_nodes:` (always `expected_online=False`,
    dropped from the live roster) for suites that need that shape too.
    """
    profiles = {
        "circe": NodeProfile(
            node_id="circe",
            raw_node=None,
            role=CIRCE_ROLE,
            expected_online=True,
            capabilities={c: True for c in CIRCE_CAPS_TRUE}
            | {"graphdb": False, "postgres": False, "hub": False},
            known=True,
        ),
        ONDEMAND_BOX: NodeProfile(
            node_id=ONDEMAND_BOX,
            raw_node=None,
            role="gpu_server",
            expected_online=False,
            capabilities={},
            known=True,
        ),
        # Mirrors the real entry's misspelled alias, which two suites assert on.
        "prometheus": NodeProfile(
            node_id="prometheus",
            raw_node=None,
            role="observability",
            expected_online=True,
            capabilities={"monitoring": True, "logs": True, "metrics": True},
            known=True,
        ),
    }
    # NodeCatalog.load() registers every canonical id as its own alias; a hand-built
    # catalog must do the same or resolve() treats the node as unknown.
    aliases = {
        "circe": "circe",
        "circe.tail348bbe.ts.net": "circe",
        "prometheus": "prometheus",
        "prometheous": "prometheus",
        "prometheus.tail348bbe.ts.net": "prometheus",
        "prometheous.tail348bbe.ts.net": "prometheus",
        ONDEMAND_BOX: ONDEMAND_BOX,
    }
    for node_id in retired:
        aliases[node_id] = node_id
    return NodeCatalog(
        profiles=profiles,
        aliases=aliases,
        defaults={"expected_online": True, "role": "unknown", "capabilities": {}},
        retired={
            node_id: NodeProfile(
                node_id=node_id,
                raw_node=None,
                role="retired",
                expected_online=False,
                capabilities={},
                known=True,
                retired=True,
            )
            for node_id in retired
        },
    )
