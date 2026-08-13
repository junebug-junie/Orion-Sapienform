from __future__ import annotations

import json
from typing import Any

# Ordered, explicit allowlist of fields a structured (non-observation) verb
# result may be summarised from. Deliberately short and generic: every entry is
# a field a verb author writes to say what it decided, and the observation is
# built by JOINING the ones actually present -- never by inventing prose for a
# payload that carries none. A payload with none of these keeps an empty
# `observation` and is still reported as a real result via `result_kind`, which
# is the point of separating the two.
STRUCTURED_SUMMARY_FIELDS = ("decision", "status", "summary", "reason")

# Cap on the synthesised observation so a verbose skill payload cannot blow up
# the stored row. Matches nothing else in particular -- it exists only to bound
# a string that is a convenience rendering, not the record. The record is
# `structured_result`, stored in full.
MAX_SYNTHESISED_OBSERVATION_CHARS = 1000

RESULT_KIND_OBSERVATION = "observation"
RESULT_KIND_STRUCTURED = "structured"
RESULT_KIND_EMPTY = "empty"


def extract_final_text(result_payload: dict[str, Any]) -> str:
    """Pull the plan's final_text out of a decoded CortexExecResultPayload.

    Mirrors services/orion-actions/app/main.py's _extract_plan_final_text --
    duplicated rather than imported since that function is private to
    orion-actions and not a shared library export.
    """
    result = result_payload.get("result") if isinstance(result_payload, dict) else None
    if isinstance(result, dict):
        final = result.get("final_text")
        if isinstance(final, str) and final.strip():
            return final.strip()
        steps = result.get("steps")
        if isinstance(steps, list):
            for step in reversed(steps):
                step_result = step.get("result") if isinstance(step, dict) else None
                if not isinstance(step_result, dict):
                    continue
                for payload in step_result.values():
                    if not isinstance(payload, dict):
                        continue
                    text = payload.get("text") or payload.get("content")
                    if isinstance(text, str) and text.strip():
                        return text.strip()
                    raw = payload.get("raw") if isinstance(payload.get("raw"), dict) else {}
                    raw_text = raw.get("text")
                    if isinstance(raw_text, str) and raw_text.strip():
                        return raw_text.strip()
    return ""


def plan_execution_status(result_payload: dict[str, Any]) -> tuple[str | None, str | None]:
    """The dispatched plan's OWN verdict, and why, straight from the payload.

    Returns `(status, reason)` where status is `PlanExecutionResult.status`
    (`"success"` / `"partial"` / `"fail"`), or the synthetic `"blocked"` when
    the plan reports `blocked=True`. `(None, None)` when the payload carries no
    readable status at all -- an unknown shape must not be silently graded.

    This exists because NOTHING upstream of here surfaces a verb failure:

      - `services/orion-cortex-exec/app/main.py:696` builds
        `CortexExecResultPayload(ok=True, ...)` with `ok` hardcoded, regardless
        of `res.status`.
      - `orion/execution_dispatch/cortex_client.py` only checks the *codec*'s
        `decoded.ok`, so it never raises for a failed plan.
      - `services/orion-cortex-exec/app/router.py:1396` sets
        `overall_status = "fail"` but still populates `final_text` with the
        skill's own JSON error payload.

    So a verb that failed arrives here looking exactly like one that succeeded,
    carrying a perfectly well-formed structured result whose `decision` field
    literally reads `"failed"`. The plan's real status is sitting in the same
    payload and was simply never read.
    """
    result = result_payload.get("result") if isinstance(result_payload, dict) else None
    if not isinstance(result, dict):
        return None, None
    if result.get("blocked") is True:
        reason = result.get("blocked_reason")
        return "blocked", str(reason) if isinstance(reason, str) and reason else "blocked"
    status = result.get("status")
    if not isinstance(status, str) or not status:
        return None, None
    return status, None


def is_failed_plan_status(status: str | None) -> bool:
    """True when a plan's own status means the dispatched work did not complete.

    `None` is NOT a failure: an older or unrecognised payload shape has no
    readable verdict, and inventing one would be a fabricated grade. Those keep
    the pre-existing content-based classification.

    `"partial"` counts as a failure. Dispatch plans are single-step, so
    `router.py:1396`'s `"partial" if len(step_results) > 1 else "fail"` cannot
    produce it for this caller today -- but if a multi-step dispatch plan ever
    exists, a plan that did not finish what it was asked is not a success, and
    the safe default is the honest one.
    """
    return status is not None and status not in ("success", "ok", "completed")


def _empty_result() -> dict[str, Any]:
    return {
        "observation": "",
        "salient_facts": [],
        "confidence": 0.0,
        "result_kind": RESULT_KIND_EMPTY,
        "structured_result": None,
    }


def _synthesise_observation(data: dict[str, Any]) -> str:
    """A truthful one-line rendering of a structured verb result.

    Built only from `STRUCTURED_SUMMARY_FIELDS` that are actually present and
    are actually strings. Returns "" when the payload carries none of them --
    an empty string here means "this verb did not summarise itself", NOT "this
    verb returned nothing", and the two are kept distinguishable by
    `result_kind`.
    """
    parts: list[str] = []
    for field in STRUCTURED_SUMMARY_FIELDS:
        value = data.get(field)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    if not parts:
        return ""
    return ": ".join(parts)[:MAX_SYNTHESISED_OBSERVATION_CHARS]


def parse_structured_observation(final_text: str) -> dict[str, Any]:
    """Parse a dispatched verb's structured JSON output.

    Two producer families reach this function, and until 2026-08-13 only one of
    them was readable:

    1. **Observation verbs** (`substrate.inspect`, `substrate.summarize`,
       `substrate.observe`) emit `{"observation": str, "salient_facts": list,
       "confidence": float}`. Unchanged.

    2. **Skill verbs** (`skills.runtime.*`) emit their own result dict via
       `services/orion-cortex-exec/app/verb_adapters.py::_skill_result_output`,
       which sets `final_text = json.dumps(result)`. That dict has no
       `observation` key at all.

    Family 2 used to fall into the `observation = ""` branch, which the caller
    read as `raw_len == 0` and therefore `status="empty"`. Confirmed live: all
    four `skills.runtime.builder_prune.v1` dispatches on 2026-08-12 stored
    `status=empty, raw_len=0, observation=""` while the cortex-exec log for the
    same correlation ids recorded `raw_len=739 final_len=739` -- the skill ran,
    measured the disk, made a real decision, and the entire result was
    discarded at this boundary. Orion's only mutating action was structurally
    incapable of reporting an outcome.

    Worse than losing the bytes: the verb's own carefully separated verdicts
    (`declined_no_pressure`, `declined_nothing_to_reclaim`, `would_act`,
    `pruned` with a real `bytes_reclaimed`) all collapsed into the single
    indistinguishable state `empty` -- the same state a dead executor produces.

    Returns, in addition to the three original keys:

      `result_kind`        one of `observation` / `structured` / `empty`.
                           This, not `len(observation)`, is what the caller
                           should use to decide whether a real result came
                           back. A skill that honestly declined to act
                           returned a REAL result and must not be recorded as
                           empty.
      `structured_result`  the verb's own payload, verbatim, when it is not an
                           observation. This is where a real outcome such as
                           `bytes_reclaimed` survives to the stored row and
                           becomes queryable.

    Never raises. An unparseable payload, a non-object payload, or a
    JSON object with no keys still degrades to `empty` -- the empty-shell-
    cognition rule stays in force for the cases where it is actually true.

    KNOWN UNCHANGED, deliberately out of scope: a verb returning non-JSON plain
    text still degrades to `empty`. That family does not occur in live data
    (24h window, 2026-08-12: 12,782 `success` results, all valid observation
    JSON; the 30 `failed` rows are send failures that never reach this
    function), so changing it here would be an unmeasured second mechanism in
    a patch that already carries one.
    """
    if not final_text.strip():
        return _empty_result()
    try:
        data = json.loads(final_text)
    except (json.JSONDecodeError, ValueError):
        return _empty_result()
    if not isinstance(data, dict) or not data:
        return _empty_result()

    if "observation" in data:
        observation = data.get("observation")
        if not isinstance(observation, str):
            observation = ""
        salient_facts = data.get("salient_facts")
        if not isinstance(salient_facts, list):
            salient_facts = []
        confidence = data.get("confidence")
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            confidence = 0.0
        stripped = observation.strip()
        return {
            "observation": stripped,
            "salient_facts": [str(f) for f in salient_facts],
            "confidence": float(confidence),
            # An observation verb that returned an `observation` key holding an
            # empty string genuinely returned nothing -- that is the original
            # empty-shell case and it stays `empty`.
            "result_kind": RESULT_KIND_OBSERVATION if stripped else RESULT_KIND_EMPTY,
            "structured_result": None,
        }

    # Family 2: a real structured result from a skill verb.
    #
    # `confidence` stays 0.0 rather than being invented. A skill verb does not
    # report an epistemic confidence, and fabricating a mid-range number here
    # would be exactly the "schema-valid payload with meaningless content"
    # failure this repo bans. Consumers that care can read `result_kind` to see
    # that 0.0 means "not applicable", not "measured as zero".
    return {
        "observation": _synthesise_observation(data),
        "salient_facts": [],
        "confidence": 0.0,
        "result_kind": RESULT_KIND_STRUCTURED,
        "structured_result": data,
    }
