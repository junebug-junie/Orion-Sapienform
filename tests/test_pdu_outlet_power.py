"""Per-outlet PDU power gives circe a chassis figure (ROADMAP B6).

circe has no reachable BMC. Every fleet power total this arc has produced has carried
`measurements_missing: {"chassis_watts": ["circe"]}` -- honest, and a whole machine wide. The
rack PDU meters every outlet, so it closes that gap without touching the BMC.

Values below are the device's own admin-panel readout, 2026-08-15:

    OUTLET19  0.64 A  206.1 V  132 VA  129 W  PF 0.98   } circe, 3 PSUs
    OUTLET25  0.63 A  206.0 V  130 VA  125 W  PF 0.95   }
    OUTLET31  0.81 A  206.4 V  167 VA  165 W  PF 0.99   }
    OUTLET34  0.77 A  206.4 V  159 VA  156 W  PF 0.98   } atlas, 2 PSUs
    OUTLET35  0.55 A  206.4 V  114 VA  110 W  PF 0.96   }
"""
from __future__ import annotations

import pytest

from orion.telemetry.biometrics_pipeline import FLEET_SUM_KEYS, extract_measurements

# Panel readout, exact.
CIRCE_OUTLETS = {19: 129.0, 25: 125.0, 31: 165.0}   # sum 419
ATLAS_OUTLETS = {34: 156.0, 35: 110.0}              # sum 266


def _sample(pdu=None, ilo=None):
    s = {}
    if pdu is not None:
        s["pdu"] = pdu
    if ilo is not None:
        s["ilo"] = ilo
    return s


# ------------------------------------------------------------ the gap this closes

def test_a_node_with_no_bmc_gets_chassis_watts_from_the_pdu():
    """THE POINT. circe has no iLO, so without this its chassis_watts simply does not exist."""
    m = extract_measurements(_sample(pdu={"pdu_watts": 419.0}))
    assert m["chassis_watts"] == 419.0
    assert m["pdu_watts"] == 419.0


def test_the_sum_is_over_every_psu_not_one_outlet():
    """A 3-PSU machine on one outlet's reading would understate by two thirds. 129+125+165."""
    assert sum(CIRCE_OUTLETS.values()) == 419.0
    m = extract_measurements(_sample(pdu={"pdu_watts": sum(CIRCE_OUTLETS.values())}))
    assert m["chassis_watts"] == 419.0


# ------------------------------------------------------------ containment

def test_a_node_with_both_meters_keeps_the_bmc_as_chassis_watts():
    """atlas has iLO AND outlets. They measure THE SAME WATTS, so chassis_watts must not
    become the PDU value -- and must never be their sum."""
    m = extract_measurements(_sample(
        pdu={"pdu_watts": 266.0}, ilo={"ilo_power_watts": 291.0}))
    assert m["chassis_watts"] == 291.0       # iLO wins, deliberately
    assert m["pdu_watts"] == 266.0           # kept separately, so they stay comparable
    assert m["chassis_watts"] != m["chassis_watts"] + m["pdu_watts"]


def test_the_pdu_never_overwrites_a_bmc_reading_even_when_they_disagree():
    """Measured live mid-burst on atlas: PDU 737 W against iLO 484 W. iLO polls every 60 s and
    the PDU read is instantaneous, so under moving load they diverge. Whichever wrote last
    would win at random unless the ordering is fixed."""
    m = extract_measurements(_sample(
        pdu={"pdu_watts": 737.0}, ilo={"ilo_power_watts": 484.0}))
    assert m["chassis_watts"] == 484.0
    assert m["pdu_watts"] == 737.0


def test_a_zero_bmc_reading_still_beats_the_pdu():
    """0.0 W from a BMC is a reading, not an absence. It must not fall through to the PDU."""
    m = extract_measurements(_sample(
        pdu={"pdu_watts": 419.0}, ilo={"ilo_power_watts": 0.0}))
    assert m["chassis_watts"] == 0.0
    assert m["pdu_watts"] == 419.0


# ------------------------------------------------------------ absence

def test_no_pdu_blob_leaves_everything_untouched():
    """athena is not on this PDU. It must not gain a key, and must not gain a zero."""
    m = extract_measurements(_sample(ilo={"ilo_power_watts": 470.0}))
    assert m["chassis_watts"] == 470.0
    assert "pdu_watts" not in m


def test_an_unreachable_pdu_is_absent_not_zero():
    """The poller reports an error and no watts. chassis_watts must stay absent so the fleet
    aggregate names the node in measurements_missing -- exactly today's circe behaviour."""
    m = extract_measurements(_sample(pdu={"pdu_error": "timeout", "pdu_fetched_at": "x"}))
    assert "pdu_watts" not in m
    assert "chassis_watts" not in m


def test_a_negative_reading_is_rejected_as_a_sensor_fault():
    m = extract_measurements(_sample(pdu={"pdu_watts": -5.0}))
    assert "pdu_watts" not in m and "chassis_watts" not in m


@pytest.mark.parametrize("bad", [None, True, "lots", float("nan"), float("inf")])
def test_nonsense_pdu_values_do_not_reach_measurements(bad):
    m = extract_measurements(_sample(pdu={"pdu_watts": bad}))
    assert "pdu_watts" not in m and "chassis_watts" not in m


def test_zero_watts_from_the_pdu_is_a_real_reading():
    """A powered-off machine genuinely draws ~0 at the outlet. That is data, not absence --
    circe read exactly 0 W on both its outlets while it was shut down."""
    m = extract_measurements(_sample(pdu={"pdu_watts": 0.0}))
    assert m["pdu_watts"] == 0.0
    assert m["chassis_watts"] == 0.0


# ------------------------------------------------------------ fleet composition

def test_pdu_watts_is_extensive():
    assert "pdu_watts" in FLEET_SUM_KEYS


def test_the_fleet_chassis_total_finally_includes_circe():
    """The regression this whole step exists to produce: circe stops being named missing."""
    from orion.telemetry.biometrics_pipeline import aggregate_fleet_measurements

    per_node = {
        "athena": {"chassis_watts": 470.0},
        "atlas": {"chassis_watts": 291.0, "pdu_watts": 266.0},
        "circe": {"chassis_watts": 419.0, "pdu_watts": 419.0},
    }
    totals, missing = aggregate_fleet_measurements(per_node)
    assert totals["chassis_watts"] == pytest.approx(1180.0)
    assert "chassis_watts" not in missing        # <- circe no longer absent
    assert missing["pdu_watts"] == ["athena"]    # athena is not on this PDU, and says so


# ------------------------------------------------------------ outlet parsing

def test_outlet_lists_parse():
    from app.pdu import parse_outlets

    assert parse_outlets("19,25,31") == [19, 25, 31]
    assert parse_outlets(" 34 , 35 ") == [34, 35]
    assert parse_outlets("19;25") == [19, 25]


def test_an_empty_outlet_list_disables_the_feature():
    """athena's value. Empty must mean off, not 'all outlets'."""
    from app.pdu import parse_outlets

    assert parse_outlets("") == []
    assert parse_outlets("   ") == []


def test_garbage_outlets_are_skipped_not_fatal():
    """A malformed env value must not take the whole collector down with it."""
    from app.pdu import parse_outlets

    assert parse_outlets("19,banana,31") == [19, 31]
    assert parse_outlets("19,0,-4,31") == [19, 31]


def test_duplicate_outlets_are_counted_once():
    """A duplicated outlet would double that PSU's contribution to the chassis sum."""
    from app.pdu import parse_outlets

    assert parse_outlets("19,19,25") == [19, 25]


def test_an_unconfigured_poller_reports_nothing():
    from app.pdu import PduPoller

    p = PduPoller("", [])
    assert p.enabled is False
    assert p.details() == {}


def test_a_configured_poller_is_enabled():
    from app.pdu import PduPoller

    p = PduPoller("192.168.1.39", [19, 25, 31])
    assert p.enabled is True
