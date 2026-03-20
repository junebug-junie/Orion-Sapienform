# Supervisor Meta-Plan Expected Evidence (Pass 3)

Target behavior:

- When Cortex exec `Supervisor` receives a planner `final_answer` that looks like a shallow meta-plan
  for an instructional `output_mode` (e.g. `implementation_guide`), it must invoke `finalize_response`.
- The resulting `final_text` must not leak meta-plan scaffolding phrases.

## Expected evidence (high level)

- Evidence file: `supervisor_meta_plan_finalize_evidence.json` / `.md`
- Expected `called_tool_ids` includes:
  - `finalize_response`
- Expected final answer does **not** include:
  - `gather requirements`
  - `create a guide`
  - `review and refine`

