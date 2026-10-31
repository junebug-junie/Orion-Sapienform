"""Stitch an FCC answer back together when the CLI's output cap split it.

When one model response hits ``CLAUDE_CODE_MAX_OUTPUT_TOKENS`` the Claude Code
CLI does not fail the turn. It sends the model a fixed user message
("Output token limit hit. Resume directly ...") and the model writes the rest
of its answer in a NEW assistant message. The CLI's ``result`` event, and every
"last assistant text wins" reader in this repo, then holds only that last
piece. The head of the answer is silently dropped.

Live evidence (2026-10-31, reading stage-2 turn ``c5df42da``): three responses
of 16384 output tokens each; the second ended with 3,876 chars of the JSON
answer, the third carried the last 7,993 chars. The harness draft was the
tail alone, starting mid-string (``0.047, arithmetic on ...``), so
``parse_json_object`` failed. Reproduced against the real CLI (2.1.316) with a
fake Messages server: stream-json carried ``stop_reason: null`` on both
assistant events, the resume message as a plain ``user`` event, and
``result`` = the tail only. So the resume message is the one reliable marker.

Rule: a text piece followed by the resume message is a HEAD; the next assistant
text continues it. Any other user event (a tool result, a compaction summary,
a nudge) ends the continuation: the earlier piece was mid-work prose, not the
answer. A next piece that opens the way the head opened is a rewrite from the
top, not a continuation, and replaces the head (the model often ignores
"resume directly": 9 of the 23 cut JSON answers in the governor's transcripts,
2026-10-20..31; the other 14 continued mid-string).
"""

from __future__ import annotations

from typing import Any, Optional

#: Fixed opening of the CLI's output-cap recovery message (Claude Code 2.1.x).
OUTPUT_CAP_RESUME_PREFIX = "Output token limit hit."


def _raw(event: Any) -> Optional[dict]:
    if not isinstance(event, dict):
        return None
    inner = event.get("raw")
    return inner if isinstance(inner, dict) else event


def _message(raw: dict) -> dict:
    message = raw.get("message")
    return message if isinstance(message, dict) else {}


def _text_of(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(
        b["text"] for b in content
        if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str)
    )


def is_output_cap_resume_event(event: Any) -> bool:
    """True for the CLI's synthetic 'Output token limit hit' user message."""
    raw = _raw(event)
    if raw is None or raw.get("type") != "user":
        return False
    return _text_of(_message(raw)).lstrip().startswith(OUTPUT_CAP_RESUME_PREFIX)


#: A continuation that opens with the same characters the answer opened with is a
#: rewrite from the top, not a continuation (a real cut lands mid-answer).
RESTART_PREFIX_CHARS = 32


def _is_restart(head: str, tail: str) -> bool:
    a, b = head.lstrip(), tail.lstrip()
    k = min(RESTART_PREFIX_CHARS, len(a), len(b))
    return k >= 8 and a[:k] == b[:k]


class OutputCapStitcher:
    """Feed every stream event to ``observe``; pass the draft through ``stitch``."""

    def __init__(self) -> None:
        self._head = ""
        self._cur_id: Optional[str] = None
        self._cur_text = ""
        #: How many output-cap continuations this turn stitched (telemetry).
        self.continuations = 0

    def observe(self, event: Any) -> None:
        raw = _raw(event)
        if raw is None:
            return
        etype = raw.get("type")
        if etype == "assistant":
            message = _message(raw)
            text = _text_of(message)
            msg_id = message.get("id")
            msg_id = str(msg_id) if msg_id else None
            if msg_id is None or msg_id != self._cur_id:
                # A new response. The CLI emits one assistant event per content
                # block, so a thinking block and its text block share an id.
                self._cur_id = msg_id
                self._cur_text = text
            else:
                self._cur_text += text
        elif etype == "user":
            if is_output_cap_resume_event(raw):
                self._head += self._cur_text
                self.continuations += 1
            else:
                self._head = ""
            self._cur_id = None
            self._cur_text = ""

    @property
    def head(self) -> str:
        return self._head

    def stitch(self, text: str) -> str:
        """Prepend the cut-off head(s) to the final piece of the answer.

        ``text`` is the caller's stripped draft. When it is the current
        response's own text, the raw (unstripped) piece is used instead: the
        cut can land mid-word or mid-string and the model is told to "pick up
        mid-thought", so a stripped leading space would glue two words. The
        caller re-applies its own secret redaction to the stitched result.
        """
        head = self._head.strip()
        if not head or not text or text.lstrip().startswith(head):
            return text
        tail = self._cur_text if self._cur_text.strip() == text.strip() else text
        if _is_restart(head, tail):
            # The model ignored "resume directly" and wrote the whole answer again
            # (9 of 23 cut JSON answers in the governor's transcripts, 2026-10-20..31).
            # The new piece IS the answer; gluing the old head on would break it.
            self._head = ""
            return text
        return (self._head + tail).strip()

    def draft_for(self, event: Any, text: str) -> str:
        """The turn's draft after ``event``, whose extracted text is ``text``.

        Only a result event, or an assistant event with its own text block,
        carries new answer text. The resume message yields the pieces cut off
        so far. Any other event hands back ``text`` unchanged.
        """
        raw = _raw(event)
        if raw is None:
            return text
        if is_output_cap_resume_event(raw):
            # The CLI's own instruction to the model, never Orion's answer: the
            # extractor reads text blocks from user events too. The draft so far
            # is every piece cut off until now.
            return self._head.strip() or text
        etype = raw.get("type")
        if etype == "result" or (etype == "assistant" and _text_of(_message(raw)).strip()):
            return self.stitch(text)
        return text
