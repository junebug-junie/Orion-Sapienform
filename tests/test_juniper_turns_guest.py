"""A guest's chat-box message from the game page (game company patch 3) is never "Juniper spoke": not in the logged
turn rule (dream idle gate, arousal, the act and promise ledgers), not in the ingress rule, not in the bus rule."""
from orion.regulation import juniper_turns as J


def test_the_bus_rule_skips_a_guest_turn():
    assert J.is_juniper_turn({"prompt": "hi", "source": "hub_orion"})
    assert not J.is_juniper_turn({"prompt": "hi", "source": "retro_guest"})


def test_the_ingress_rule_skips_a_guest_turn_from_the_game_page():
    assert J.is_browser_surface_turn({"surface_context": {"surface": "retro"}})          # Juniper stated on the page
    assert not J.is_browser_surface_turn({"surface_context": {"surface": "retro"}, "chat_history_source": "retro_guest"})


def test_both_sql_rules_name_the_guest_source():
    assert "COALESCE(source, '') <> 'retro_guest'" in J.JUNIPER_TURN_PREDICATE
    assert "raw->>'chat_history_source'" in J.JUNIPER_INGRESS_PREDICATE and "retro_guest" in J.JUNIPER_INGRESS_PREDICATE
