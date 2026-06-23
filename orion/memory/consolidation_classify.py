import math
import re
from typing import Any

_MEMORY_LINE = re.compile(r"^MEMORY:\s*(YES|NO)\s*$", re.I | re.M)
_BOUNDARY_LINE = re.compile(r"^BOUNDARY:\s*(YES|NO)\s*$", re.I | re.M)


def binary_score_from_top_logprobs(top_logprobs: list[dict]) -> float | None:
    yes_lp = no_lp = None
    for entry in top_logprobs or []:
        tok = str(entry.get("token") or "").strip().upper()
        lp = entry.get("logprob")
        if not isinstance(lp, (int, float)):
            continue
        if tok == "YES":
            yes_lp = float(lp)
        elif tok == "NO":
            no_lp = float(lp)
    if yes_lp is None or no_lp is None:
        return None
    ey, en = math.exp(yes_lp), math.exp(no_lp)
    return ey / (ey + en)


def parse_classify_lines(text: str) -> tuple[str | None, str | None]:
    mem = _MEMORY_LINE.search(text or "")
    bnd = _BOUNDARY_LINE.search(text or "")
    return (
        mem.group(1).upper() if mem else None,
        bnd.group(1).upper() if bnd else None,
    )


from orion.memory.turn_change_classify import build_turn_change_prompt


def build_classify_prompt(
    *,
    prompt: str,
    response: str,
    spark_meta: dict[str, Any],
    baseline_mode: str = "none",
    baseline_text: str = "",
) -> str:
    phase = (
        (spark_meta.get("conversation_phase") or {}).get("phase_change")
        or spark_meta.get("temporal_phase")
        or "unknown"
    )
    return build_turn_change_prompt(
        prompt=prompt,
        response=response,
        baseline_mode=baseline_mode,
        baseline_text=baseline_text,
        phase=str(phase),
    )
