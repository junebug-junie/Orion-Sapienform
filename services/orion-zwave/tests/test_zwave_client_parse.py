from app.zwave_client import (
    extract_meter_amps,
    extract_meter_volts,
    extract_meter_watts,
    extract_switch_on,
    ingest_node_values,
    normalize_value_entry,
)


def test_extract_meter_watts_prefers_electric_w_not_kwh():
    """Shelly Wave Plug: 65537 is kWh energy; 66049 is instantaneous watts."""
    values = {
        "50-0-value-65537": {
            "commandClass": 50,
            "property": "value",
            "propertyKey": 65537,
            "propertyKeyName": "Electric_kWh_Consumed",
            "value": 33.25,
        },
        "50-0-value-66049": {
            "commandClass": 50,
            "property": "value",
            "propertyKey": 66049,
            "propertyKeyName": "Electric_W_Consumed",
            "value": 727.3,
        },
    }
    assert extract_meter_watts(values) == 727.3


def test_extract_meter_watts_ignores_kwh_only():
    values = {
        "50-0-value-65537": {
            "commandClass": 50,
            "property": "value",
            "propertyKey": 65537,
            "propertyKeyName": "Electric_kWh_Consumed",
            "value": 33.25,
        },
    }
    assert extract_meter_watts(values) is None


def test_extract_meter_volts_and_amps():
    values = {
        "50-0-value-66561": {
            "commandClass": 50,
            "property": "value",
            "propertyKey": 66561,
            "propertyKeyName": "Electric_V_Consumed",
            "value": 122.3,
        },
        "50-0-value-66817": {
            "commandClass": 50,
            "property": "value",
            "propertyKey": 66817,
            "propertyKeyName": "Electric_A_Consumed",
            "value": 6.073,
        },
    }
    assert extract_meter_volts(values) == 122.3
    assert extract_meter_amps(values) == 6.073


def test_extract_meter_watts_absent():
    assert extract_meter_watts({}) is None


def test_extract_switch_on():
    values = {
        "37-0-currentValue": {"commandClass": 37, "property": "currentValue", "value": True},
    }
    assert extract_switch_on(values) is True


def test_normalize_value_updated_event_copies_new_value():
    event_args = {
        "commandClass": 50,
        "property": "value",
        "propertyKey": 66049,
        "endpoint": 0,
        "newValue": 710.1,
        "prevValue": 727.3,
        "propertyKeyName": "Electric_W_Consumed",
    }
    normalized = normalize_value_entry(event_args)
    assert normalized["value"] == 710.1
    assert extract_meter_watts({"k": normalized}) == 710.1


def test_ingest_node_values_list_seeds_meter_and_switch():
    """start_listening returns values as a list — must become a keyed map."""
    by_node: dict = {}
    ingest_node_values(
        by_node,
        2,
        [
            {
                "commandClass": 50,
                "endpoint": 0,
                "property": "value",
                "propertyKey": 66049,
                "propertyKeyName": "Electric_W_Consumed",
                "value": 412.5,
            },
            {"commandClass": 37, "endpoint": 0, "property": "currentValue", "value": True},
        ],
    )
    values = by_node[2]
    assert extract_meter_watts(values) == 412.5
    assert extract_switch_on(values) is True
