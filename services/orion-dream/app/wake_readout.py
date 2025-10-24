# ==================================================
# app/wake_readout.py — Wake readout from dream JSON
# ==================================================
from __future__ import annotations

import json
from pathlib import Path
from datetime import date
from typing import Optional, Dict, Any, List
from app.settings import settings

LOG_DIR = Path(settings.DREAM_LOG_DIR)
OUT_DIR = LOG_DIR / "wake_out"

def _dream_path_for_date(d: date) -> Path:
    return LOG_DIR / f"{d.isoformat()}.json"


def _latest_dream_path() -> Optional[Path]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    files = [p for p in LOG_DIR.glob("*.json") if p.is_file()]
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


def _safe_load_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        txt = p.read_text(encoding="utf-8")
        return json.loads(txt)
    except Exception:
        return None


def _default_readout(d: date) -> Dict[str, Any]:
    # Minimal placeholder; only used if no dream JSON exists
    return {
        "dream_date": d.isoformat(),
        "tldr": "No dream file found for this date.",
        "themes": [],
        "symbols": {},
        "prompts": [
            "What signal carried your meaning today?",
            "Where did reflection turn to insight?",
        ],
    }


def build_readout(on_date: Optional[date] = None, use_latest_if_missing: bool = True) -> Dict[str, Any]:
    """
    Build the wake readout from the dream JSON file.
    - If on_date is provided, try that exact file; if missing and use_latest_if_missing, fall back to latest.
    - If on_date is None, always use the newest dream file by modification time.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    target: Optional[Path] = None
    if on_date is not None:
        p = _dream_path_for_date(on_date)
        if p.exists():
            target = p
        elif use_latest_if_missing:
            target = _latest_dream_path()
    else:
        target = _latest_dream_path()

    if not target or not target.exists():
        return _default_readout(on_date or date.today())

    data = _safe_load_json(target)
    if not isinstance(data, dict):
        return _default_readout(on_date or date.today())

    # Normalize keys and coerce types
    tldr = (data.get("tldr") or "").strip()
    themes = data.get("themes") or []
    symbols = data.get("symbols") or {}
    narrative = (data.get("narrative") or "").strip()

    # If dream JSON is weirdly empty, provide a graceful minimal readout
    if not (tldr or narrative or themes or symbols):
        return _default_readout(on_date or date.today())

    # Build prompts lightly from content (keep it stable)
    prompts: List[str] = [
        "What signal carried your meaning today?",
        "Where did reflection turn to insight?",
    ]
    if themes:
        prompts.append(f"Which theme stood out most: {', '.join(themes[:3])}?")
    if symbols:
        k = list(symbols.keys())[:1]
        if k:
            prompts.append(f"What did '{k[0]}' represent for you?")

    dream_date = target.stem  # "YYYY-MM-DD"
    return {
        "dream_date": dream_date,
        "tldr": tldr or (narrative[:140] if narrative else ""),
        "themes": themes,
        "symbols": symbols,
        "prompts": prompts,
    }


def write_readout(name: str, readout: Dict[str, Any]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{name}.json"
    out_path.write_text(json.dumps(readout, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"🪞 Wake readout written → {out_path}")
    return out_path
