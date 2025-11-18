from pathlib import Path
from orion_cognition.packs_loader import PackManager

def test_packs_load():
    base = Path(__file__).resolve().parents[1]
    pm = PackManager(base)
    pm.load_packs()

    packs = pm.list_packs()
    assert "memory_pack" in packs
    assert "executive_pack" in packs
    assert "emergent_pack" in packs

def test_pack_validation_memory_pack():
    base = Path(__file__).resolve().parents[1]
    pm = PackManager(base)
    pm.load_packs()

    result = pm.verify_pack("memory_pack")
    assert result["missing"] == []
