from orion.town_cast import ORION_DISPLAY_NAME, TOWN_PARTICIPANT_SLUGS, slug_for_name, thread_id_for


def test_slug_map_is_explicit_six_rows():
    assert TOWN_PARTICIPANT_SLUGS == {
        "Mara Vale": "mara-vale",
        "Nico Sable": "nico-sable",
        "Sofia Bell": "sofia-bell",
        "Cam Lin": "cam-lin",
        "Juniper Feld": "juniper-feld",
        "Orion": "orion",
    }
    assert ORION_DISPLAY_NAME == "Orion"


def test_slug_for_name_unknown_is_none():
    assert slug_for_name("Dr. Elian Cross") is None
    assert slug_for_name("") is None


def test_thread_id_is_sorted_slugs():
    assert thread_id_for("Sofia Bell", "Cam Lin") == "cam-lin--sofia-bell"
    assert thread_id_for("Cam Lin", "Sofia Bell") == "cam-lin--sofia-bell"


def test_thread_id_unknown_is_none():
    assert thread_id_for("Sofia Bell", "Juno Park") is None
