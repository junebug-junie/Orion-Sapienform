from app.zwave_client import (
    extract_meter_watts,
    extract_switch_on,
    ingest_node_values,
)


def test_extract_meter_watts_from_value_id():
    values = {
        "50-0-value-65537": {"commandClass": 50, "property": "value", "propertyKey": 65537, "value": 412.5},
    }
    assert extract_meter_watts(values) == 412.5


def test_extract_meter_watts_absent():
    assert extract_meter_watts({}) is None


def test_extract_switch_on():
    values = {
        "37-0-currentValue": {"commandClass": 37, "property": "currentValue", "value": True},
    }
    assert extract_switch_on(values) is True


def test_ingest_node_values_list_seeds_meter_and_switch():
    """start_listening returns values as a list — must become a keyed map."""
    by_node: dict = {}
    ingest_node_values(
        by_node,
        2,
        [
            {"commandClass": 50, "endpoint": 0, "property": "value", "propertyKey": 65537, "value": 33.11},
            {"commandClass": 37, "endpoint": 0, "property": "currentValue", "value": True},
        ],
    )
    values = by_node[2]
    assert extract_meter_watts(values) == 33.11
    assert extract_switch_on(values) is True
