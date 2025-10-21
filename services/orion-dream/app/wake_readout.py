# ==================================================
# wake_readout.py
# ==================================================
from datetime import date

def build_readout(dream_date: date):
    return {
        "dream_date": str(dream_date),
        "tldr": "A soft voice whispered across Orion's bus, threads of meaning weaving themselves into reflection.",
        "themes": ["connection", "continuity", "self-observation"],
        "symbols": {"mirror": "reflection", "bus": "connection"},
        "prompts": ["What signals carried your meaning today?", "Where did reflection turn to insight?"]
    }

def write_readout(dream_id, readout):
    print(f"🪞 Wake readout for {dream_id}: {readout['tldr']}")
