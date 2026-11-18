"""Native Nova mechanic + real Orion planner eval; no live session is touched.

Uses the operator-supplied pinned ROM/core. No ROM/RAM/image is exported. The only
LLM input is bounded decoded game facts. This is an offline accelerated trial,
not the independent 30-minute, client-latency, learning or leisure qualification.
"""
import argparse
import asyncio
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'services/orion-retro'))
sys.path.insert(0, str(ROOT/'services/orion-hub'))
from orion.retro.probe import CoreProbe
from app.observation import Observer, outcome
from app.scripted import Scripted
from app.control import Arbiter
from scripts.retro_planner import choose


async def run(args):
    rows = []
    core = CoreProbe(args.core, args.rom)
    try:
        observer = Observer(core)
        assert observer.supported, 'unqualified ROM/core'
        # Real controller inputs navigate menus; no RAM writes or savestate cheats.
        for frame in range(3000):
            mask = 8 if frame % 360 < 10 else (256 if 180 <= frame % 360 < 190 else 0)
            core.step(mask)
        initial = observer.read()
        assert initial['status'] == 'observed', initial
        for _ in range(300): core.step(128)
        before = observer.read()
        for _ in range(300): core.step(128)
        blocked = observer.read()
        assert outcome(before, blocked, 'walk_right') == 'no_horizontal_progress', (before, blocked)
        # Verify genuine rest: neutral and a movement attempt are not the same verdict.
        for _ in range(120): core.step(0)
        rest = observer.read()
        assert outcome(blocked, rest, 'wait') == 'unknown'
        a = Arbiter([0])
        script = Scripted(a, lambda kind, **data: rows.append(dict(kind=kind, **data)))
        human = a.control('eval', 0, 'acquire', 0)
        script.start('eval', 0, 60, 0, 3720, human['epoch'], True)
        policy = script.active[0]; goals = policy['goals']
        goals.accepted = dict(goal='walk_right', attempt_id='preflight')
        goals.offer(0, policy['grant']['epoch'], 3720, blocked, 0, 'no_horizontal_progress')
        config = SimpleNamespace(ORION_BUS_URL=args.bus_url, NODE_NAME='athena',
            LLM_GATEWAY_SERVICE_NAME='LLMGatewayService', CHANNEL_LLM_INTAKE='orion:exec:request:LLMGatewayService')
        if args.mechanic_only:
            choice = dict(goal='jump_right', rationale='Explicit mechanic eval fixture; not Orion cognition.',
                          planner_id='mechanic-fixture')
        else:
            try:
                choice = await choose(config, goals.attempt)
            except Exception as exc:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(dict(passed=False, identity=core.identity,
                    initial=initial, before=before, blocked=blocked, rest=rest,
                    attempt=goals.attempt, ack=None, outcome='unknown',
                    planner_error=str(exc)[:300]), indent=2)+'\n')
                raise
        data = choice | dict(port=0, expected_epoch=policy['grant']['epoch'], attempt_id=goals.attempt['attempt_id'])
        ack = script.goal('eval', data, 0)
        applied = []
        for i in range(300):
            now = i/60
            script.tick(now)
            masks, events, _ = a.frame(now)
            for event in events:
                if goals.accepted_current(): goals.applied_start(3721+i, observer.read())
            core.step(*masks)
            for event in events:
                script.annotate(event)
                applied.append(event | {'frame': 3721+i})
        after = observer.read()
        verdict = outcome(goals.start_observation or {}, after, ack['goal'])
        result = dict(real_planner=not args.mechanic_only, identity=core.identity, initial=initial, before=before, blocked=blocked,
                      rest=rest, attempt=goals.attempt, ack=ack, applied=applied,
                      action_start=goals.start_observation, after=after, outcome=verdict, journal=rows,
                      passed=ack['goal'] != 'walk_right' and any(e['mask'] != 128 for e in applied)
                             and verdict in ('horizontal_position_changed', 'health_decreased'))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({k:result[k] for k in ('ack','after','outcome','passed')}))
        assert result['passed'], 'real planner did not demonstrate the cognition gate; see artifact'
    finally:
        core.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mechanic-only', action='store_true', help='Fixture goal; never claims cognition')
    p.add_argument('--core', type=Path, required=True)
    p.add_argument('--rom', type=Path, required=True)
    p.add_argument('--bus-url', required=True)
    p.add_argument('--output', type=Path, required=True)
    asyncio.run(run(p.parse_args()))
