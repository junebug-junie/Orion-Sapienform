# PlannerReact Salvage and Repair Manifest

## Root causes fixed
- Primary planner responses were being rejected when the LLM returned otherwise-usable JSON wrapped in bare language markers (`bash`, `sh`, `plaintext`, `json`), fenced code blocks, or explanatory text around the JSON object.
- Some valid final responses used `final_answer` as an object with obvious text-bearing fields such as `content`, `text`, or `answer`, but strict handling could still treat them as schema-invalid or over-normalize them.
- The repair path could receive a valid `finish=true` payload with a usable `final_answer`, yet still degrade into a synthetic `finish=false requires an action` failure after parsing/validation fallout.

## Salvage rules
PlannerReact now attempts a narrow salvage pass before declaring the planner output non-JSON:
1. Parse the raw text directly first.
2. If that fails, strip outer fenced code block wrappers.
3. Strip a leading bare language marker line when it is exactly one of: `bash`, `sh`, `plaintext`, `json`.
4. Attempt extraction of the first balanced top-level JSON object from the raw text and from salvaged candidates.
5. Re-parse the salvaged candidate and record whether salvage succeeded for logging.

The salvage pass remains intentionally conservative:
- it only targets object-shaped JSON;
- it does not redesign routing;
- it still raises a parse failure when no recoverable JSON object exists.

## Repair validation rules
- `finish=true` plus a usable `final_answer` is accepted even when `action` is `null`.
- `final_answer` is accepted directly when it is a string.
- When `final_answer` is a dict, obvious string fields are extracted in priority order: `content`, `text`, `answer`.
- When `finish=true` and `final_answer` is still a list/dict without an obvious text field, it is compactly stringified instead of being discarded.
- `finish=false` still requires an action.
- If the repair output remains unusable after salvage and normalization, fallback behavior is preserved.

## Files changed
- `services/orion-planner-react/app/api.py`
- `tests/test_planner_react_contract.py`
- `docs/postflight/planner_react_salvage_and_repair_manifest.md`

## Tests added
- Salvage for a raw payload prefixed by `bash`.
- Salvage for a raw payload prefixed by `plaintext` plus fenced JSON.
- Salvage for JSON embedded in leading/trailing explanatory text.
- Acceptance of `finish=true` with string `final_answer`.
- Acceptance of `finish=true` with dict-based `final_answer` using `content`.
- Acceptance of `finish=true` with dict-based `final_answer` using `text`.
- Repair-path acceptance of `finish=true` final answers without an action.
- Preservation of fallback behavior for unrecoverable garbage.

## Expected impact on live Discord deployment proof
- PlannerReact should reject far fewer planner completions that already contain a usable final answer.
- Repair attempts should stop misclassifying valid `finish=true` payloads as missing-action failures.
- Logs should now make it clear whether a response was salvaged, normalized, schema-invalid, or forced into fallback.
- The live Discord deployment flow should therefore hit the synthetic repair fallback less often and surface final deployment instructions more reliably.
