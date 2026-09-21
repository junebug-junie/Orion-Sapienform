from orion.schemas.curiosity_supervisor import READING_CHANNEL, READING_KIND
from orion.schemas.registry import SCHEMA_REGISTRY, _REGISTRY


def test_hop_reading_v1_registered_in_both_maps() -> None:
    assert "HopReadingV1" in _REGISTRY
    assert SCHEMA_REGISTRY["HopReadingV1"].kind == READING_KIND


def test_reading_channel_and_kind_match_bus_catalog_entry() -> None:
    import yaml

    with open("orion/bus/channels.yaml") as f:
        catalog = yaml.safe_load(f)
    entries = [c for c in catalog["channels"] if c["name"] == READING_CHANNEL]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["schema_id"] == "HopReadingV1"
    assert entry["message_kind"] == READING_KIND
