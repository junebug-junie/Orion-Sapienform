"""Exercise the lookup used by real bus publication, not just the kind map."""
from orion.schemas.registry import SCHEMA_REGISTRY, resolve
from orion.schemas.resource_admission import RESOURCE_EVENT_KIND, ResourceEventV1


def test_resource_event_resolves_for_bus_publication():
    assert resolve("ResourceEventV1") is ResourceEventV1
    assert SCHEMA_REGISTRY["ResourceEventV1"].kind == RESOURCE_EVENT_KIND
