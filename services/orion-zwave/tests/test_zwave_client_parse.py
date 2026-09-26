from app.zwave_client import extract_meter_watts, extract_switch_on


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
