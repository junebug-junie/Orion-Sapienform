from __future__ import annotations

"""Single source of truth for "which VLM family does this model_id belong
to" -- both ``model_manager.py`` (which transformers class to instantiate)
and ``runner.py`` (how to build the prompt and decode the reply) need the
same answer for the same model_id, and a substring check duplicated in two
places is exactly the kind of thing that quietly drifts (one file adds a
new Qwen alias, the other doesn't, and half the pipeline silently falls
back to the wrong prompt format). One function, imported by both.
"""

from orion.vision.caption_echo import CAPTION_PROMPT

# "qwen2.5-vl" does not contain the substring "qwen2-vl", so checking order
# against QWEN2_VL_MARKERS is not load-bearing -- kept as two explicit tuples
# anyway so a caller that only cares about one generation doesn't have to
# also match the other.
QWEN2_5_VL_MARKERS = ("qwen2.5-vl", "qwen2_5_vl")
QWEN2_VL_MARKERS = ("qwen2-vl", "qwen2_vl")


def is_qwen2_5_vl_model(model_id: str) -> bool:
    mid = (model_id or "").lower()
    return any(m in mid for m in QWEN2_5_VL_MARKERS)


def is_qwen2_vl_model(model_id: str) -> bool:
    mid = (model_id or "").lower()
    return any(m in mid for m in QWEN2_VL_MARKERS)


def is_chat_template_vlm(model_id: str) -> bool:
    """True for any VLM family that expects its prompt built via
    ``processor.apply_chat_template`` and its reply decoded from just the
    newly generated tokens (sliced by input length), rather than BLIP/BLIP2's
    plain ``processor(images=, text=)`` call and full-sequence decode.
    Qwen2-VL and Qwen2.5-VL are the only members today; add new chat-template
    families here, not as a third parallel branch in runner.py.

    Known, accepted limitation (review finding, 2026-08-25): this is a
    closed substring allowlist, same as the pre-existing blip2/blip checks
    it sits alongside in ``model_manager.py`` -- a chat-template VLM whose
    model_id doesn't match one of these markers (a differently-named Qwen
    fork, or a future non-Qwen chat-tuned family) silently falls through to
    the generic ``AutoModelForVision2Seq`` + BLIP-style prompt path instead
    of erroring, which produces the model's whole echoed prompt glued to
    its answer rather than a clean one. Not a regression this PR
    introduces -- every model_id outside the pre-existing blip/blip2/generic
    split already had this same "silently wrong branch, not an error"
    property before Qwen support existed. Add the new family's markers here
    when it's actually needed, rather than building a registry/plugin
    abstraction for a single-operator, single-node deployment with no
    second chat-template family in sight.
    """
    return is_qwen2_5_vl_model(model_id) or is_qwen2_vl_model(model_id)


def caption_prompt_for(model_id: str) -> str | None:
    """The text a caption call sends with the frame, chosen by model family.

    Chat-template VLMs follow instructions, so they get ``CAPTION_PROMPT``.
    Everything else (BLIP/BLIP2 and unknown families on the plain
    ``processor(images=, text=)`` path) gets no prompt: those models continue
    the text they are given rather than obeying it. Confirmed live
    2026-10-08 on cam0 frames: BLIP-base handed ``CAPTION_PROMPT`` returns
    the prompt itself (the echo filter rejects it, the caption publishes
    blank); with no prompt it returns "a woman sitting at a desk in an
    office". See docs/superpowers/specs/2026-10-08-vision-blank-captions-
    and-on-demand-look-design.md.
    """
    return CAPTION_PROMPT if is_chat_template_vlm(model_id) else None
