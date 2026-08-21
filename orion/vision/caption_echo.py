from __future__ import annotations

import re

CAPTION_PROMPT = (
    "List visible objects and people. "
    "State only what is directly visible. "
    "No guesses about activity."
)

_LEGACY_PROMPT_ECHO = re.compile(r"describe this image", re.I)


def _normalize(text: str) -> str:
    return " ".join(text.lower().strip().split()).strip(".,!?")


def is_caption_prompt_echo(text: str, *, prompt: str = CAPTION_PROMPT) -> bool:
    """True when the model's output is its own input prompt echoed back, not a
    real answer -- a known BLIP-family failure mode, not scene/question
    description.

    ``prompt`` defaults to the fixed captioning prompt (every pre-existing
    caller keeps its exact prior behavior unchanged) but is a real keyword
    parameter, not hardcoded: VQA (added alongside this parameter) prompts
    the same model family with a caller-supplied *question* instead of
    ``CAPTION_PROMPT``, and an echo there means the question came back
    verbatim, not that the caption prompt did -- checking against the wrong
    fixed string would silently never catch a real VQA echo.
    """
    raw = (text or "").strip()
    if not raw:
        return False
    if _LEGACY_PROMPT_ECHO.search(raw):
        return True
    norm = _normalize(raw)
    prompt_norm = _normalize(prompt)
    if norm == prompt_norm:
        return True
    # BLIP often echoes the prompt with minor suffix noise (e.g. trailing "s").
    if norm.startswith(prompt_norm) and len(norm) <= len(prompt_norm) + 4:
        return True
    prompt_tokens = prompt_norm.split()
    text_tokens = norm.split()
    if len(text_tokens) <= len(prompt_tokens) + 1 and text_tokens[: len(prompt_tokens)] == prompt_tokens:
        return True
    return False


def strip_echoed_prompt_prefix(text: str, *, prompt: str = CAPTION_PROMPT) -> str:
    """Removes a *leading* echo of ``prompt`` from ``text``, case-insensitively.

    Confirmed live 2026-08-20 (VQA's first real smoke test): a caller-supplied
    question ``"What color is the door?"`` (capital W) came back from BLIP as
    ``"what color is the door? a black and white photo of a room..."`` --
    lowercased by the model regardless of the caller's input casing. The
    prior code here (both this module's caller in ``_run_caption_frame`` and
    the new ``_run_vlm_vqa``) used a plain, case-sensitive ``str.replace()``,
    which silently failed to strip that prefix -- not because the echo was
    hard to detect (``is_caption_prompt_echo`` above already handles
    case-insensitively via ``_normalize``), but because nothing was actually
    trying to *remove* a partial-prefix echo before this function existed:
    the raw generation output went straight to the sanitizer with the
    echoed prefix still attached, and a mixed "echo + real content" response
    correctly is NOT a pure echo (so the sanitizer rightly keeps it) but
    still shipped with an ugly, redundant echoed prefix glued to the front
    of an otherwise-real answer.

    Only strips a genuine *prefix* match (the very start of ``text``) --
    deliberately not doing a global find-and-replace anywhere in the
    string, which risks mangling a real answer that happens to reuse a
    prompt word mid-sentence.
    """
    raw = (text or "").strip()
    if not raw:
        return raw
    prompt_stripped = (prompt or "").strip()
    if not prompt_stripped:
        return raw
    if raw.lower().startswith(prompt_stripped.lower()):
        return raw[len(prompt_stripped):].strip()
    return raw
