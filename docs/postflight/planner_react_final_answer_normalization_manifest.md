# PlannerReact Final Answer Normalization Manifest

## Root cause
PlannerReact could receive a semantically valid planner completion with `finish=true` while `final_answer` arrived as a structured object or list instead of a plain string. The transport and LLM gateway path were healthy, but the planner response handling treated these cases as schema-adjacent failures or unusable completions, which unnecessarily pushed some requests into repair/fallback behavior.

## Files changed
- `services/orion-planner-react/app/api.py`
- `tests/test_planner_react_contract.py`
- `docs/postflight/planner_react_final_answer_normalization_manifest.md`

## Normalization rules added
- If `finish=true` and `final_answer` is already a string, PlannerReact accepts it unchanged.
- If `finish=true` and `final_answer` is a dict:
  - use `content` directly when it is a non-empty string
  - otherwise render the object into readable markdown-style sections while preserving input field order
  - fallback to compact JSON serialization if markdown rendering would be empty
- If `finish=true` and `final_answer` is a list:
  - render it as readable bullet/section markdown
  - fallback to compact JSON serialization if necessary
- If `finish=true` and `final_answer` cannot be normalized into usable text, the response still fails validation and continues into the existing repair/fallback path.
- `action=null` is accepted for `finish=true` responses once `final_answer` is usable after normalization.

## Logging changes
- Distinguished planner failures into:
  - LLM transport/request failure
  - planner response parse failure
  - planner response schema/validation failure
  - normalization applied successfully
  - repair/fallback invoked because normalization could not recover
- Added structured log context including:
  - `corr_id`
  - failure category
  - `final_answer` type when present
  - truncated raw response snippet when safe
  - whether normalization succeeded

## Tests added
- `finish=true` + string `final_answer` accepted unchanged
- `finish=true` + dict `final_answer` normalized and accepted
- `finish=true` + list `final_answer` normalized and accepted
- malformed planner response still triggers repair path
- normalization logging classification recorded with `corr_id` and `final_answer_type`

## Expected effect on live Discord deployment proof
When the planner returns HTTP 200 with a valid completion payload whose `final_answer` is object-shaped or list-shaped, PlannerReact should now convert that structured value into usable answer text instead of unnecessarily rejecting it. Real RPC/gateway failures and truly malformed planner outputs still remain visible and continue through repair/fallback, so the deployment should preserve existing resilience while eliminating this narrow false-negative planner rejection mode.
