"""Contract tests for `SubstrateReadService` (design doc option B(2)).

This is the contract patch of a §5 split (contract → producer → consumer), so
these tests assert the contract itself: that the schemas are registered, that
the channels exist and point at those schemas, and that the two states a read
can be in stay distinguishable.

That last one is the load-bearing test. An unavailable read and an empty read
must never look the same — this repo has shipped decayed-to-zero-looks-like-calm
before (`node:substrate.route`'s prediction_error, `CLAUDE.md` §0A), and the
whole reason this service exists is to give an autonomous action a measurement
it can act on.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from orion.core.contracts.substrate_read import (
    HostStorageReadV1,
    SubstrateReadQueryV1,
    SubstrateReadReplyV1,
)
from orion.schemas.registry import _REGISTRY, SCHEMA_REGISTRY

REPO = Path(__file__).resolve().parents[1]
REQUEST_CHANNEL = "orion:exec:request:SubstrateReadService"
RESULT_CHANNEL = "orion:exec:result:SubstrateReadService:*"
NOW = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)


def _channels() -> dict[str, dict]:
    data = yaml.safe_load((REPO / "orion" / "bus" / "channels.yaml").read_text())
    return {c["name"]: c for c in data["channels"]}


# --- registration (CLAUDE.md §6: never publish an unregistered event shape) ---


def test_both_schemas_are_registered_in_both_registries():
    """`orion/schemas/registry.py` holds TWO separate maps — `_REGISTRY`
    (bare model classes) and `SCHEMA_REGISTRY` (`SchemaRegistration`, carrying
    the event `kind`). A schema present in only one is half-registered, which
    is the mistake this test caught while it was being written."""
    assert _REGISTRY["SubstrateReadQueryV1"] is SubstrateReadQueryV1
    assert _REGISTRY["SubstrateReadReplyV1"] is SubstrateReadReplyV1

    assert SCHEMA_REGISTRY["SubstrateReadQueryV1"].model is SubstrateReadQueryV1
    assert SCHEMA_REGISTRY["SubstrateReadReplyV1"].model is SubstrateReadReplyV1

    # The registered kind must match the model's own schema_version literal,
    # or a consumer routing on `kind` gets a payload it cannot parse.
    assert SCHEMA_REGISTRY["SubstrateReadQueryV1"].kind == (
        SubstrateReadQueryV1(read_kind="host_storage").schema_version
    )
    assert SCHEMA_REGISTRY["SubstrateReadReplyV1"].kind == (
        SubstrateReadReplyV1(
            read_kind="host_storage", observed_at=NOW, available=False, reason="x"
        ).schema_version
    )


def test_channels_exist_and_reference_the_registered_schemas():
    ch = _channels()
    assert ch[REQUEST_CHANNEL]["schema_id"] == "SubstrateReadQueryV1"
    assert ch[RESULT_CHANNEL]["schema_id"] == "SubstrateReadReplyV1"
    for name in (REQUEST_CHANNEL, RESULT_CHANNEL):
        assert ch[name]["schema_id"] in SCHEMA_REGISTRY


def test_request_is_single_consumer_like_every_sibling():
    """Matches RecallService/ContextExecService. A second consumer would
    silently split replies between two responders."""
    ch = _channels()
    assert ch[REQUEST_CHANNEL]["single_consumer"] is True
    assert ch[REQUEST_CHANNEL]["consumer_services"] == ["orion-biometrics"]
    assert ch[REQUEST_CHANNEL]["producer_services"] == ["orion-cortex-exec"]


def test_result_flows_back_the_other_way():
    ch = _channels()
    assert ch[RESULT_CHANNEL]["producer_services"] == ["orion-biometrics"]
    assert ch[RESULT_CHANNEL]["consumer_services"] == ["orion-cortex-exec"]


# --- the contract's own invariants ---


def test_read_kind_is_closed_not_a_free_string():
    """A new read kind must be a deliberate schema change, not something a
    caller can invent at runtime (§0A: a new name needs a schema, a producer,
    a consumer and a test in the same patch)."""
    with pytest.raises(Exception):
        SubstrateReadQueryV1(read_kind="whatever_i_felt_like")  # type: ignore[arg-type]


def test_query_rejects_unknown_fields():
    with pytest.raises(Exception):
        SubstrateReadQueryV1(read_kind="host_storage", sneaky="value")  # type: ignore[call-arg]


def test_query_defaults_to_the_mount_under_pressure():
    assert SubstrateReadQueryV1(read_kind="host_storage").mount_path == "/mnt/docker"


def test_unavailable_read_is_distinguishable_from_an_empty_one():
    """The load-bearing test. `available=False` carries no payload and a
    reason; a real read of an idle host carries a payload of zeros. An
    autonomous action must be able to tell "I could not look" from "I looked
    and there is nothing there" -- they imply opposite decisions."""
    broken = SubstrateReadReplyV1(
        read_kind="host_storage",
        observed_at=NOW,
        available=False,
        reason="docker daemon unreachable",
    )
    assert broken.payload is None
    assert broken.reason

    empty = SubstrateReadReplyV1(
        read_kind="host_storage",
        observed_at=NOW,
        available=True,
        payload=HostStorageReadV1(
            mount_path="/mnt/docker",
            total_bytes=1_000,
            used_bytes=0,
            free_bytes=1_000,
            used_pct=0.0,
        ),
    )
    assert empty.payload is not None
    assert empty.reason is None
    assert broken.available != empty.available


def test_contract_carries_no_docker_payload():
    """Scope boundary, asserted so it is not quietly re-added.

    The first version of this contract carried Docker build-cache accounting,
    because the motivating consumer is autonomous cache pruning. But
    `orion-biometrics` -- the only registered consumer -- has no
    `/var/run/docker.sock` mount and could never have populated it. A contract
    promising a payload its responder cannot produce is worse than one that
    omits it: the omission is visible, the promise is not.

    `orion-cortex-exec` has the socket (which is why the docker skills already
    work there as local adapters), and `CortexRouteTemplateV1.cortex_verb` is
    an unconstrained str, so a dispatch route can name a `skills.*` verb
    directly. The prune action needs no bus service at all.
    """
    fields = set(HostStorageReadV1.model_fields)
    assert "docker" not in fields
    assert fields == {
        "mount_path",
        "total_bytes",
        "used_bytes",
        "free_bytes",
        "used_pct",
    }


def test_percentages_and_sizes_are_bounded():
    with pytest.raises(Exception):
        HostStorageReadV1(
            mount_path="/x", total_bytes=1, used_bytes=1, free_bytes=0, used_pct=101.0
        )
    with pytest.raises(Exception):
        HostStorageReadV1(
            mount_path="/x", total_bytes=1, used_bytes=-1, free_bytes=0, used_pct=0.0
        )
