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


def test_box_touching_bottom_edge_stays_in_patio():
    zones = load_zones(DEFAULT_ZONES_PATH)["walkway"]
    for y2 in (360, 400):
        z = zone_for_box(zones, [40, 200, 88, y2], 640, 360)
        assert z is not None and z.name == "patio" and not may_embed(z)


# --- intersects_no_embed: any overlap with a no-embed zone -------------------
from orion.vision.zones import Zone as _Z, intersects_no_embed as _ine

_PATIO = _Z(name="patio", polygon=((0.0, 0.7), (0.35, 0.7), (0.35, 1.0), (0.0, 1.0)), embed=False)
_WALK = _Z(name="walkway", polygon=((0.0, 0.35), (1.0, 0.35), (1.0, 1.0), (0.0, 1.0)), embed=True)
_TRI = _Z(name="tri", polygon=((0.5, 0.1), (0.6, 0.3), (0.4, 0.3)), embed=False)


def test_box_fully_outside_the_patio_does_not_intersect():
    assert _ine([_PATIO, _WALK], [600, 600, 700, 900], 1000, 1000) is False


def test_box_centered_outside_but_overlapping_the_patio_edge_intersects():
    # bottom-center (400, 950) is walkway, but the box reaches x=300 < 350.
    assert _ine([_PATIO, _WALK], [300, 750, 500, 950], 1000, 1000) is True


def test_box_inside_the_patio_and_patio_inside_the_box_intersect():
    assert _ine([_PATIO], [100, 800, 200, 900], 1000, 1000) is True
    assert _ine([_PATIO], [0, 0, 1000, 1000], 1000, 1000) is True


def test_edge_crossing_without_contained_vertices():
    # A thin horizontal band crossing the triangle: no triangle vertex in it,
    # no band corner in the triangle.
    assert _ine([_TRI], [0, 200, 1000, 220], 1000, 1000) is True
    assert _ine([_TRI], [0, 350, 1000, 400], 1000, 1000) is False


def test_no_no_embed_zone_means_no_intersection_and_bad_input_fails_closed():
    assert _ine([_WALK], [0, 0, 10, 10], 1000, 1000) is False
    assert _ine([_PATIO], [0, 0, 10, 10], 0, 1000) is True
    assert _ine([_PATIO], [0, 0, 10], 1000, 1000) is True
