"""Structured output on Strata lanes, done in the gateway so a bad answer is repaired or explained.

Why (2026-11-13; supersedes the finalize judge-only retry #3292, ticket #3289): Strata 0.2.7 has no constrained decoding. Its capabilities say
``"structured_output": {"method": "prompt_and_validate"}``: ``response_format`` only appends a
"return one JSON object" directive to the system message, then runs a strict ``json.loads`` on the
finished answer (``/opt/strata/terminal/serve/structured.py`` ``prepare_format`` /
``validated_json``). Grammar fields (``grammar``, ``guided_json``, top-level ``json_schema``) are
ignored: probed live on circe-worker-chat, all still returned ``{'mood': 'calm'}``. Strata's image has
no ``jsonschema``, so even ``json_schema`` requests are checked as "one JSON object" only.

When the strict parse fails, Strata answers HTTP 502 ``structured_output_failed`` and the text is
gone. Gateway log 2026-11-13 08:31-23:05Z: 21 such 502s on the chat/agent lanes (agent 11, prime 7,
judge 2, chat 1 -- a Juniper chat turn deferred unanswered at stance_react). Kinds:

* 13 ``Expecting ',' delimiter`` -- most likely a bare ``"`` inside a string value (Prime's 7 all break
  at char 445-483 of a one-line answer);
* 6 ``Expecting property name enclosed in double quotes: line 1 column 2`` -- a Python-literal dict,
  ``{'key': ...}`` (reproduced exactly with a single-quote bait);
* 2 ``Unterminated string``.

So the gateway takes over the same two steps, keeping the text:

1. ``take_over``: drop ``response_format`` and append Strata's own directive, word for word.
2. ``check``: Strata's own extraction and strict parse (duplicate keys and NaN refused); a reply that
   did not finish (``finish_reason != "stop"``) is incomplete, as Strata says.
3. ``_python_literal_text``: only when the strict error is the single-quote signature ("Expecting
   property name enclosed in double quotes"), a reply that parses as a Python literal dict with string
   keys is converted. That signature also covers a trailing comma, a tuple, a comment, a ``1_000``:
   their meaning is the same in both. Anything whose meaning could differ is refused: adjacent
   strings, escapes Python decodes differently, prefixed or triple-quoted strings, repeated keys.
   ``ast.literal_eval`` evaluates literals only.
4. One corrective retry (``corrective_messages``): the model sees its own reply and the parser's error.
5. Still bad: the same 502 message shape callers already handle, and the raw text in the log.
"""

from __future__ import annotations

import ast
import io
import json
import math
import re
import tokenize
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

#: Strata's own words (structured.py prepare_format), so a valid reply looks exactly as it did.
_DIRECTIVE_HEAD = (
    "OUTPUT FORMAT REQUIREMENT: Return exactly one JSON object matching the JSON Schema below. "
    "No Markdown, headings, code fences, commentary, or text outside the JSON. "
    "Use every required field, correct types, and only allowed fields. "
    "Put all requested writing inside the appropriate JSON string fields.\nJSON Schema:\n"
)
RAW_LOG_CHARS = 1500


def _schema_of(response_format: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    kind = response_format.get("type")
    if kind == "json_object":
        return {"type": "object"}
    if kind == "json_schema":
        spec = response_format.get("json_schema")
        if isinstance(spec, dict) and isinstance(spec.get("schema"), dict):
            return spec["schema"]
    return None


def directive(schema: Dict[str, Any]) -> str:
    return _DIRECTIVE_HEAD + json.dumps(schema, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def take_over(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """(payload to send, taken). Taken only for a json_object / json_schema format Strata accepts and
    no tools (Strata refuses structured output with tools); anything else is sent unchanged."""
    rf = payload.get("response_format")
    if not isinstance(rf, dict) or payload.get("tools"):
        return payload, False
    schema = _schema_of(rf)
    if schema is None:
        return payload, False
    try:
        text = directive(schema)
    except ValueError:  # a schema json.dumps refuses (NaN): Strata answers that with its own 400
        return payload, False
    messages = [dict(m) for m in payload.get("messages") or []]
    if messages and messages[0].get("role") == "system":
        content = messages[0].get("content") or ""
        if isinstance(content, list):
            messages[0]["content"] = content + [{"type": "text", "text": text}]
        else:
            messages[0]["content"] = content + "\n\n" + text
    else:
        messages.insert(0, {"role": "system", "content": text})
    out = {k: v for k, v in payload.items() if k != "response_format"}
    out["messages"] = messages
    return out, True


def extract_json(text: str) -> str:
    """Strata's extract_json: a ```-fenced block's inside, else from the first ``{`` to its match."""
    s = (text or "").strip()
    if not s:
        return ""
    if s.startswith("```"):
        first = s.find("\n")
        if first != -1:
            end = s.find("```", first + 1)
            if end != -1:
                s = s[first + 1:end].strip()
    start = s.find("{")
    if start == -1:
        return s
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return s[start:]


def _pairs(items: List[Tuple[str, Any]]) -> Dict[str, Any]:
    obj: Dict[str, Any] = {}
    for key, value in items:
        if key in obj:
            raise ValueError(f"duplicate JSON key: {key}")
        obj[key] = value
    return obj


def _constant(value: str) -> Any:
    raise ValueError(f"invalid JSON constant: {value}")


def _strict(text: str) -> Any:
    return json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant)


def _python_literal_text(text: str) -> Optional[str]:
    """``{'a': 1}`` (and True/False/None) as JSON, or None. Literals only; no names, calls or code."""
    candidate = (text or "").strip()
    if not candidate.startswith("{"):
        return None
    # The literal may run past Strata's JSON-aware brace matcher (a single-quoted string holding "}"),
    # so take everything from the first "{" to the last "}".
    end = candidate.rfind("}")
    if end == -1:
        return None
    source = candidate[:end + 1]
    # Python accepts things JSON does not that would silently change the meaning: adjacent strings
    # join, escapes such as \a or \x41 decode differently, a repeated key keeps only its last value.
    # All refused. Same-meaning extras (a trailing comma, a tuple, a comment) pass.
    if _unsafe_strings(source) or _has_duplicate_keys(source):
        return None
    try:
        value = ast.literal_eval(source)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None
    if not isinstance(value, dict):
        return None

    def ok(v: Any) -> bool:
        if isinstance(v, dict):
            return all(isinstance(k, str) and ok(x) for k, x in v.items())
        if isinstance(v, (list, tuple)):
            return all(ok(x) for x in v)
        if isinstance(v, float):
            return math.isfinite(v)
        return v is None or isinstance(v, (str, int, bool))

    if not ok(value):
        return None
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


# Escapes Python and JSON read the same way. Any other backslash (\a, \x41, \0, \N{...}, \/, a line
# continuation) means something else to Python, so a string holding one is never repaired.
_SAME_ESCAPES = re.compile(r"""\\(?:[\\'"ntrbf]|u[0-9a-fA-F]{4})""")


def _unsafe_strings(source: str) -> bool:
    """True when Python could read a string differently from JSON: adjacent strings (joined: "x" "b"
    -> "xb", the bare-quote failure turned into the wrong text), a prefixed (r/b/f/u) or triple-quoted
    string, or an escape outside ``_SAME_ESCAPES``."""
    try:
        tokens = [t for t in tokenize.generate_tokens(io.StringIO(source).readline)
                  if t.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT, tokenize.INDENT,
                                    tokenize.DEDENT, tokenize.ENDMARKER)]
    except (tokenize.TokenError, SyntaxError, IndentationError):
        return True  # cannot tell: no repair
    if any(a.type == tokenize.STRING and b.type == tokenize.STRING for a, b in zip(tokens, tokens[1:])):
        return True
    for tok in tokens:
        if tok.type != tokenize.STRING:
            continue
        text = tok.string
        if text[0] not in ("'", '"') or text[:3] in ("'''", '"""'):
            return True
        if "\\" in _SAME_ESCAPES.sub("", text[1:-1]):
            return True
    return False


def _has_duplicate_keys(source: str) -> bool:
    try:
        tree = ast.parse(source, mode="eval")
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return True
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
            if len(keys) != len(set(map(repr, keys))):
                return True
    return False


#: The one strict-parse error a Python-literal dict produces: a key that is not double-quoted.
_SINGLE_QUOTE_SIGNATURE = "Expecting property name enclosed in double quotes"


@dataclass(frozen=True)
class Verdict:
    text: Optional[str]          # canonical JSON when ok
    error: Optional[str]         # parser's words when not
    repaired: Optional[str] = None  # "python_literal" when a repair made it ok
    retryable: bool = True

    @property
    def ok(self) -> bool:
        return self.text is not None


def check(text: Any, finish_reason: Any) -> Verdict:
    """Strata's validated_json for "one JSON object", plus the one exact repair. Strata always sends a
    finish_reason; a reply with none is not called incomplete (that would fail every JSON call)."""
    if finish_reason is not None and finish_reason != "stop":
        return Verdict(None, f"structured output was incomplete (finish_reason={finish_reason}); "
                             "increase the output budget", retryable=False)
    raw = text if isinstance(text, str) else ""
    extracted = extract_json(raw)
    try:
        value = _strict(extracted)
        if not isinstance(value, dict):
            return Verdict(None, "model output failed the JSON Schema at /: the answer is not a JSON object")
        return Verdict(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")), None)
    except (ValueError, TypeError) as exc:
        error = f"model did not return valid JSON: {exc}"
    if _SINGLE_QUOTE_SIGNATURE in error:  # only the failure the repair is for
        start = raw.find("{")
        repaired = _python_literal_text(raw[start:] if start != -1 else raw)
        if repaired is not None:
            return Verdict(repaired, None, repaired="python_literal")
    return Verdict(None, error)


def corrective_messages(messages: List[Dict[str, Any]], bad_reply: str, error: str) -> List[Dict[str, Any]]:
    """The one retry: the model sees what it wrote and why it was refused."""
    return [*messages, {"role": "assistant", "content": bad_reply or ""}, {"role": "user", "content": (
        f"Your reply was not one valid JSON object ({error}). Reply again with only the corrected JSON "
        "object: double quotes around every key and string, every double quote inside a string escaped "
        "as \\\", no text outside the object."
    )}]


def with_content(raw_data: Dict[str, Any], text: str) -> Dict[str, Any]:
    """The upstream reply with its first choice's content replaced by the canonical JSON."""
    choices = list(raw_data.get("choices") or [])
    if not choices or not isinstance(choices[0], dict):
        return raw_data
    first = dict(choices[0])
    message = dict(first.get("message") or {})
    message["content"] = text
    first["message"] = message
    choices[0] = first
    return {**raw_data, "choices": choices}


def first_content(raw_data: Any) -> Tuple[str, Any]:
    """(content string, finish_reason) of the first choice; content "" when not a string."""
    choices = raw_data.get("choices") if isinstance(raw_data, dict) else None
    first = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first.get("message"), dict) else {}
    content = message.get("content")
    return (content if isinstance(content, str) else ""), first.get("finish_reason")
