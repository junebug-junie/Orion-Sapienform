from pathlib import Path

from orion.vision.zones import DEFAULT_ZONES_PATH, load_zones, may_embed, zone_for_box


def test_shipped_config_loads_and_patio_is_never_embeddable():
    zones = load_zones(DEFAULT_ZONES_PATH)
    walkway = zones["walkway"]
    patio = [z for z in walkway if z.name == "patio"]
    assert patio and patio[0].embed is False


def test_bottom_center_decides_zone_and_first_match_wins():
    zones = load_zones(DEFAULT_ZONES_PATH)["walkway"]
    # feet at (0.1, 0.9): inside patio and inside walkway; patio listed first.
    z = zone_for_box(zones, [40, 200, 88, 324], 640, 360)
    assert z is not None and z.name == "patio" and not may_embed(z)
    # feet at (0.6, 0.9): walkway only.
    z = zone_for_box(zones, [360, 200, 408, 324], 640, 360)
    assert z is not None and z.name == "walkway" and may_embed(z)
    # feet in the sky: no zone, embeddable.
    z = zone_for_box(zones, [300, 10, 340, 50], 640, 360)
    assert z is None and may_embed(z)


def test_missing_file_means_no_zones(tmp_path: Path):
    assert load_zones(tmp_path / "nope.yaml") == {}
