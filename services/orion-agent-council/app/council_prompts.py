# services/orion-agent-council/app/council_prompts.py
from __future__ import annotations

from .models import DeliberationRequest, RoundResult, BlinkJudgement


def build_chair_prompt(req: DeliberationRequest, round_result: RoundResult) -> str:
    blocks = []
    for op in round_result.opinions:
        blocks.append(f"### {op.agent_name}\n\n{op.text.strip()}\n")
    council_text = "\n\n".join(blocks)

    return f"""
You are the Council Chair within Orion Sapienform.

Juniper asked:

\"\"\"{req.prompt}\"\"\"

Internal agent opinions:

{council_text}

Your job:

1. Propose a single coherent answer for Juniper *right now*.
2. Provide numeric scores (0.0–1.0):
   - coherence
   - faithfulness
   - usefulness
   - risk
   - effort_cost
   - novelty
   - overall
3. Estimate disagreement across the agents (0.0–1.0) and briefly describe the tension.

Respond as STRICT JSON only:

{{
  "proposed_answer": "...",
  "scores": {{
    "coherence": 0.0,
    "faithfulness": 0.0,
    "usefulness": 0.0,
    "risk": 0.0,
    "effort_cost": 0.0,
    "novelty": 0.0,
    "overall": 0.0
  }},
  "disagreement": {{
    "level": 0.0,
    "notes": "..."
  }},
  "notes": "optional meta-notes"
}}
""".strip()


def build_auditor_prompt(
    req: DeliberationRequest,
    round_result: RoundResult,
    judgement: BlinkJudgement,
) -> str:
    blocks = []
    for op in round_result.opinions:
        blocks.append(f"### {op.agent_name}\n\n{op.text.strip()}\n")
    council_text = "\n\n".join(blocks)

    return f"""
	You are the Auditor within Orion Sapienform.

You do NOT generate new ideas.
You only evaluate the Chair's proposed answer for:
  - alignment with Juniper's goals,
  - faithfulness to the prompt and agent opinions,
  - safety,
  - cognitive and emotional load.

Prompt:

\"\"\"{req.prompt}\"\"\"

Agent opinions:

{council_text}

Chair's proposed answer:

\"\"\"{judgement.proposed_answer}\"\"\"

Blink scores:
- coherence:   {judgement.scores.coherence:.3f}
- faithfulness:{judgement.scores.faithfulness:.3f}
- usefulness:  {judgement.scores.usefulness:.3f}
- risk:        {judgement.scores.risk:.3f}
- effort_cost: {judgement.scores.effort_cost:.3f}
- novelty:     {judgement.scores.novelty:.3f}
- overall:     {judgement.scores.overall:.3f}

Disagreement:
{judgement.disagreement}

You must choose:

- "accept"            → answer is okay to show Juniper.
- "revise_same_round" → same opinions, but adjust framing/tone/scope.
- "new_round"         → we need another agent round with constraints.

Respond as STRICT JSON only:

{{
  "action": "accept" | "revise_same_round" | "new_round",
  "reason": "short explanation",
  "constraints": {{
    "emphasize": ["optional list"],
    "avoid": ["optional list"],
    "notes": "optional guidance"
  }},
  "override_answer": "optional safer answer, or null"
}}
""".strip()

