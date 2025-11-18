from pathlib import Path
from orion_cognition.packs_loader import PackManager

if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1]
    pm = PackManager(base)
    pm.load_packs()

    pack = pm.get_pack("emergent_pack")
    print(f"{pack.label}: {pack.description}")
    print("Verbs:", ", ".join(pack.verbs))
