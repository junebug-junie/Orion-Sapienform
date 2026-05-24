from app.settings import DEFAULT_ROUTE_MAP, settings


def test_route_map_includes_grammar_event() -> None:
    assert DEFAULT_ROUTE_MAP["grammar.event.v1"] == "GrammarEventSQL"


def test_subscribe_channels_include_grammar() -> None:
    assert "orion:grammar:event" in settings.sql_writer_subscribe_channels
