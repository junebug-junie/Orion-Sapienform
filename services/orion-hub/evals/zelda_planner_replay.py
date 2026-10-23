"""Replay a recorded Zelda decision through the real planner (local GPU) and grade the goal.

Fixture: a live decision where Orion asked "explore S" eight times against a solid wall
(evals/fixtures/zelda_room119_south_wall.json). PASS when every run returns a goal that
can move Link: not an exit through a closed side, not the move that just did nothing.

  PYTHONPATH=. ORION_BUS_URL=redis://100.92.216.81:6379/0 \
    .venv/bin/python services/orion-hub/evals/zelda_planner_replay.py --runs 5
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

HUB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HUB))
from scripts import retro_planner  # noqa: E402


def dead_move(goal: dict, decision: dict) -> str | None:
    sides, previous = decision.get('open_sides') or [], decision.get('previous') or {}
    if goal['kind'] == 'exit' and goal['dir'] not in sides:
        return 'exit through a closed side'
    if goal == previous.get('goal') and (previous.get('applied_frames') == 0 or goal.get('dir') not in sides):
        return 'repeats the move that did nothing'
    return None


async def main(args) -> int:
    fixture = json.loads(Path(args.fixture).read_text())
    config = SimpleNamespace(ORION_BUS_URL=os.environ['ORION_BUS_URL'], NODE_NAME='eval',
                             LLM_GATEWAY_SERVICE_NAME='LLMGatewayService',
                             CHANNEL_LLM_INTAKE='orion:exec:request:LLMGatewayService')
    history = [fixture['decision']['previous']]
    results = []
    for run in range(args.runs):
        try:
            goal = await retro_planner.plan_zelda(config, fixture['decision'], fixture['atlas'], history)
            bad = dead_move(goal['goal'], fixture['decision'])
            results.append(dict(run=run, goal=goal['goal'], prediction=goal['prediction'], dead=bad,
                                rationale=goal['rationale'][:200]))
        except retro_planner.PlannerError as exc:
            results.append(dict(run=run, refused=exc.reason, details=exc.details))
    for r in results:
        print(json.dumps(r))
    goals = [r for r in results if 'goal' in r]
    verdict = 'PASS' if goals and len(goals) == len(results) and not any(r['dead'] for r in goals) else 'FAIL'
    print(f'{verdict}: {len(goals)}/{len(results)} runs returned a goal; '
          f'{sum(1 for r in goals if r["dead"])} dead moves')
    return 0 if verdict == 'PASS' else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--fixture', default=str(HUB / 'evals/fixtures/zelda_room119_south_wall.json'))
    p.add_argument('--runs', type=int, default=5)
    sys.exit(asyncio.run(main(p.parse_args())))
