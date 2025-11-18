from pathlib import Path
from orion_cognition.planner.rdf_sync import generate_turtle_for_all

if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1]
    ttl = generate_turtle_for_all(base)
    print(ttl)
