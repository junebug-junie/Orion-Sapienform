from pathlib import Path
from orion_cognition.planner.rdf_sync import generate_turtle_for_all

def test_turtle_generation():
    base = Path(__file__).resolve().parents[1]
    ttl = generate_turtle_for_all(base)

    assert "@prefix orion:" in ttl
    assert "rdf:type" in ttl
    assert "Verb" in ttl or "orion:Verb" in ttl
