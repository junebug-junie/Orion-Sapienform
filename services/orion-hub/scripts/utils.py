import re

_sentence_regex = re.compile(r'(?<=[.!?])\s+')

def split_sentences(text: str):
    parts = _sentence_regex.split(text or "")
    return [p.replace("\n", " ").strip() for p in parts if p and p.strip()]
