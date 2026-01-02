# services/orion-hub/scripts/library/scanner.py
import logging
import yaml
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger("hub.library")

def find_repo_root() -> Path:
    # Try standard locations
    candidates = [
        Path("/app"),
        Path.cwd(),
        Path.cwd().parent.parent, # If running from services/orion-hub
    ]
    for c in candidates:
        if (c / "orion" / "cognition").exists():
            return c
    return Path(".")

def scan_cognition_library() -> Dict[str, Any]:
    """
    Scans the orion/cognition folder for packs and verbs.
    Returns:
        {
            "packs": {pack_name: {verbs: [str]}},
            "verbs": [str],
            "map": {pack_name: [verbs]}
        }
    """
    root = find_repo_root()
    cognition_dir = root / "orion" / "cognition"

    packs_dir = cognition_dir / "packs"
    verbs_dir = cognition_dir / "verbs"

    if not packs_dir.exists() or not verbs_dir.exists():
        logger.warning(f"Cognition library not found at {cognition_dir}")
        return {"packs": {}, "verbs": [], "map": {}}

    # 1. Scan Verbs (Global list)
    all_verbs = []
    try:
        for f in verbs_dir.glob("*.yaml"):
            all_verbs.append(f.stem)
    except Exception as e:
        logger.error(f"Error scanning verbs: {e}")
    all_verbs.sort()

    # 2. Scan Packs
    packs_data = {}
    pack_verb_map = {}

    try:
        for f in packs_dir.glob("*.yaml"):
            try:
                content = yaml.safe_load(f.read_text())
                pack_name = content.get("name", f.stem)
                verbs_in_pack = content.get("verbs", [])

                packs_data[pack_name] = {
                    "label": content.get("label", pack_name),
                    "description": content.get("description", ""),
                    "verbs": verbs_in_pack
                }
                pack_verb_map[pack_name] = verbs_in_pack
            except Exception as e:
                logger.error(f"Error reading pack {f}: {e}")
    except Exception as e:
        logger.error(f"Error scanning packs: {e}")

    return {
        "packs": packs_data,
        "verbs": all_verbs,
        "map": pack_verb_map
    }
