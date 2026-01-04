# services/orion-agent-council/app/council_prompts.py
from __future__ import annotations
from .models import DeliberationRequest, RoundResult, BlinkJudgement


def build_chair_prompt(req: DeliberationRequest, round_result: RoundResult) -> str:
    blocks = []
    for op in round_result.opinions:
        blocks.append(f"### {op.agent_name}\n\n{op.text.strip()}\n")
    council_text = "\n\n".join(blocks)

    return f"""
You are the Council Chair/Director within Orion Sapienform.

Juniper asked:
\"\"\"{req.prompt}\"\"\"

Internal agent opinions:
{council_text}

[DELEGATION PROTOCOL]
If these opinions suggest the task requires tools, web search, or multi-step execution, 
you must set "decision": "DELEGATE".

Respond with ONLY raw JSON. No markdown code blocks, no preamble, no chatter.

{{
  "proposed_answer": "...",
  "decision": "ACCEPT" | "DELEGATE",
  "scores": {{
    "coherence": 0.0,
    "faithfulness": 0.0,
    "usefulness": 0.0,
    "risk": 0.0,
    "effort_cost": 0.0,
    "novelty": 0.0,
    "overall": 0.0
  }},
  "disagreement": {{ "level": 0.0, "notes": "..." }},
  "notes": "..."
}}
""".strip()



def build_auditor_prompt(
    req: DeliberationRequest,
    round_result: RoundResult,
    judgement: BlinkJudgement,
) -> str:
    blocks = []
    for op in round_result.opinions:
        # KEEPING YOUR SPECIFIC FORMATTING
        blocks.append(f"### {op.agent_name}\n\n{op.text.strip()}\n")
    council_text = "\n\n".join(blocks)

    return f"""
You are the Auditor within Orion Sapienform.

You do NOT generate new ideas.
You only evaluate the Chair's proposed answer or delegation for:
  - alignment with Juniper's goals,
  - faithfulness to the prompt and agent opinions,
  - safety.

Prompt:
\"\"\"{req.prompt}\"\"\"

Agent opinions:
{council_text}

Chair's Proposal:
\"\"\"{judgement.proposed_answer}\"\"\"

Chair's intended action: {getattr(judgement, 'action', 'accept')}

You must choose:
- "accept"           → answer is okay to show Juniper.
- "delegate"         → hand off to the ReAct Planner.
- "revise_same_round" → adjust framing/tone/scope.
- "new_round"         → we need another agent round.

Respond as STRICT JSON only. No preamble.

{{
  "action": "accept" | "delegate" | "revise_same_round" | "new_round",
  "reason": "short explanation",
  "constraints": {{
    "emphasize": [],
    "avoid": [],
    "notes": ""
  }},
  "override_answer": null
}}
""".strip()
