# ==================================================
# app/utils.py — helpers for Orion Dream
# ==================================================

from __future__ import annotations

import re
import json
import ast
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Tuple, Any

import numpy as np
from json_repair import repair_json

# ---------------------------
# Fragment model
# ---------------------------
@dataclass
class Fragment:
    id: str
    kind: str
    text: str
    tags: List[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None
    salience: float = 0.0
    ts: float = 0.0
    valence: float = 0.0
    arousal: float = 0.0


# ---------------------------
# Text utils
# ---------------------------
def clean_text(val: str | list | dict | None) -> str:
    if val is None: return ""
    if isinstance(val, (list, tuple)): val = ", ".join(map(str, val))
    elif isinstance(val, dict): val = json.dumps(val, ensure_ascii=False)
    val = " ".join(str(val).split())
    return val.strip()

def sanitize_text_for_prompt(s: str) -> str:
    if not s: return ""
    t = str(s).replace("\r", "")
    t = (t.replace("“", '"').replace("”", '"')
          .replace("’", "'").replace("‘", "'"))
    t = re.sub(r"^```[^\n]*\n", "", t, flags=re.M)
    t = t.replace("```", "")
    t = re.sub(r'"{3,}', '"', t)
    t = re.sub(r"'{3,}", "'", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t

def rewrite_chat_roles(text: str, style: str = "arrow") -> str:
    if not text: return ""
    out = []
    for line in text.splitlines():
        raw = line.strip()
        if not raw: continue
        low = raw.lower()
        if style == "arrow":
            if low.startswith("user:"):
                raw = raw.split(":", 1)[-1].strip()
                raw = f"→ {raw}"
            elif low.startswith(("assistant:", "orion:")):
                raw = raw.split(":", 1)[-1].strip()
                raw = f"← {raw}"
        elif style == "abstract":
            if low.startswith(("user:", "assistant:", "orion:")):
                raw = raw.split(":", 1)[-1].strip()
        out.append(raw)
    return "\n".join(out)


# ---------------------------
# JSON repair / parsing
# ---------------------------
_COMMENT_RE = re.compile(r'\s*//.*')

def coerce_json(s: str) -> dict:
    """
    Uses the json_repair library to fix common JSON errors from LLMs.
    1. Strips // comments (as json_repair might not handle them).
    2. Calls repair_json to fix syntax.
    3. Uses json.loads on the repaired string.
    """
    if not isinstance(s, str):
        raise ValueError("coerce_json expects a string")

    t = s.strip()

    # 1. Strip code fences
    if t.startswith("```json"): t = t[7:].strip()
    elif t.startswith("```"): t = t[3:].strip()
    if t.endswith("```"): t = t[:-3].strip()

    # 2. Strip // comments before repairing
    t = _COMMENT_RE.sub('', t)

    try:
        print("\n" + "="*30 + " TEXT BEFORE json_repair " + "="*30)
        print(repr(t))
        print("="*80 + "\n")

        # 3. Use repair_json to fix the string
        # It handles missing commas, quotes, brackets, control chars, etc.
        repaired_t = repair_json(t)

        # --- Debug Print (See what repair_json outputs) ---
        print("\n" + "="*30 + " TEXT AFTER json_repair " + "="*30)
        print(repr(repaired_t))
        print("="*80 + "\n")

        # 4. Parse the repaired string using standard json.loads
        obj = json.loads(repaired_t)

        if not isinstance(obj, dict):
            print(f"⚠️ json_repair result was not a dict (was {type(obj)}), wrapping.")
            obj = {"parsed_result": obj}
        return obj

    except Exception as e:
        print("\n" + "="*30 + " FAILING TEXT (json_repair) " + "="*30)
        print(repr(t))
        print("="*80 + "\n")

        raise ValueError(
            f"Failed to repair and parse JSON string. Original error: {e}. Text was: {t}"
        )

# ---------------------------
# Biometrics parsing
# ---------------------------
_METRIC_RE = re.compile(
    r"GPU\s*power\s*=\s*(?P<gpu_w>[0-9]+(?:\.[0-9]+)?)W.*?"
    r"util\s*=\s*(?P<util>[0-9]+(?:\.[0-9]+)?)%.*?"
    r"mem\s*=\s*(?P<mem_mb>[0-9]+(?:\.[0-9]+)?)MB.*?"
    r"CPU\s*avg\s*temp\s*=\s*(?P<cpu_c>[0-9]+(?:\.[0-9]+)?)"
    r"(?:°C|C| deg C)?",
    re.I | re.S,
)

def extract_metrics(text: str) -> Dict[str, float]:
    """Pull GPU power/util/mem and CPU °C from the summary block."""
    m = _METRIC_RE.search(text or "")
    if not m:
        return {}
    try:
        return {
            "gpu_w": float(m.group("gpu_w")),
            "util": float(m.group("util")),
            "mem_mb": float(m.group("mem_mb")),
            "cpu_c": float(m.group("cpu_c")),
        }
    except (ValueError, IndexError):
        # Handle cases where conversion to float might fail or group doesn't exist
        print("⚠️ Failed to parse metrics string.")
        return {}

# ---------------------------
# Anchor helper
# ---------------------------
def collect_anchors(frags: List[Fragment], max_tags: int = 6, max_chat: int = 3) -> Tuple[List[str], List[str]]:
    tags: List[str] = []
    for f in frags:
        for t in getattr(f, "tags", []) or []:
            t_norm = clean_text(t).lower()
            if not t_norm: continue
            tags.append(f"tag:{t_norm}")
    seen = set()
    tag_anchors: List[str] = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            tag_anchors.append(t)
        if len(tag_anchors) >= max_tags: break
    chat_anchors: List[str] = []
    for f in frags:
        if f.kind != "chat": continue
        line = sanitize_text_for_prompt(f.text).splitlines()[0] if f.text else ""
        line = line.strip()
        if line and len(line) <= 140 and line not in chat_anchors:
            chat_anchors.append(line)
        if len(chat_anchors) >= max_chat: break
    return tag_anchors, chat_anchors


# ---------------------------
# Salience & timestamp
# ---------------------------
def compute_salience(f: Fragment) -> float:
    v = abs(getattr(f, "valence", 0.0) or 0.0)
    a = float(getattr(f, "arousal", 0.0) or 0.0)
    novelty = random.random() * 0.3
    recurrence = random.random() * 0.2
    score = 0.5 * min(v, 1.0) + 0.3 * min(a, 1.0) + 0.2 * novelty + 0.1 * recurrence
    return float(max(0.0, min(1.0, score)))


def now_ts() -> float:
    return datetime.utcnow().timestamp()
