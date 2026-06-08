import pytest


def _load_profile(name: str) -> dict:
    """Load a recall profile YAML by name using the existing profiles loader."""
    from app.profiles import get_profile
    return get_profile(name)


def test_global_profile_loads_and_has_required_fields():
    p = _load_profile("graph.compressions.global.v1")
    assert p.get("enable_graph_compression") is True
    assert p.get("compression_mode") == "global"
    assert int(p.get("compression_global_top_k", 0)) >= 1
    assert isinstance(p.get("compression_scopes"), list)
    assert "episodic" in p["compression_scopes"]


def test_local_profile_loads_and_has_required_fields():
    p = _load_profile("graph.compressions.local.v1")
    assert p.get("enable_graph_compression") is True
    assert p.get("compression_mode") == "local"
    assert int(p.get("compression_local_top_k", 0)) >= 1


def test_unified_profile_loads_and_has_required_fields():
    p = _load_profile("graph.compressions.v1")
    assert p.get("enable_graph_compression") is True
    assert p.get("compression_mode") == "unified"
    assert int(p.get("compression_global_top_k", 0)) >= 1
    assert int(p.get("compression_local_top_k", 0)) >= 1


def test_profiles_do_not_conflict_with_existing_profiles():
    """Loading a compression profile must not affect loading a standard profile."""
    _load_profile("graph.compressions.v1")
    from app.profiles import get_profile
    existing = get_profile("reflect.v1")
    assert existing is not None
