from __future__ import annotations

import re


def _normalize_whitespace(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def is_low_info_social(text: str) -> bool:
    candidate = _normalize_whitespace(text).lower()
    if not candidate:
        return True
    normalized_candidate = re.sub(r"[^a-z0-9' ]+", " ", candidate)
    tokens = re.findall(r"[a-z']{2,}", normalized_candidate)
    if len(tokens) > 18:
        return False
    courtesy_patterns = [
        r"^(hi|hey|hello|yo|sup|hiya|good (morning|afternoon|evening))( there| friend| orion| juniper)?[!. ]*$",
        r"^(thanks|thank you|awesome|cool|nice|sounds good|all good|doing good|doing well|glad to hear)[!. ]*$",
        r"^(how are you|how's it going|hope you're well|hope you are well)[?.! ]*$",
    ]
    if any(re.match(pattern, candidate, flags=re.I) for pattern in courtesy_patterns):
        return True
    if len(tokens) <= 6:
        low_info_terms = {
            "hi",
            "hey",
            "hello",
            "thanks",
            "thank",
            "good",
            "great",
            "cool",
            "nice",
            "fine",
            "well",
            "friend",
            "orion",
            "juniper",
            "all",
            "doing",
            "okay",
            "ok",
            "for",
            "now",
        }
        if all(token in low_info_terms for token in tokens):
            return True
    return False
