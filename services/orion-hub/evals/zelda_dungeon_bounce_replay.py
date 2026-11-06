"""Replay eval for the 2026-11-06 dungeon two-room bounce (retro_planner.RoomMemory).

Live 2026-11-06: solo sessions 3bf6973a and 0d7e21de each played 30 min inside a dungeon and walked
36 -> N -> 20 -> S -> 36 for 337 and 339 decisions (0 new rooms, every prediction 'unknown'). Retro
reports level 'unknown' in dungeons, its atlas then has no neighbours, every side looks new, and
the menu breaks the value tie by N-S-E-W order.

This replays the recorded decisions (evals/fixtures/zelda_dungeon_bounce_3bf6973a.json, a read-only
export of Retro's evidence file) through the Hub's real menu (scripts/retro_experience.py), quoted by
orion-prediction's real model (services/orion-prediction/app/action_bets.py, via the counts store of
evals/zelda_experience_replay.py), with every traverse settled the way it went live (success: the
room changed). Then a closed loop in a synthetic 2 x 3 dungeon on Retro's room grid takes option 1
each time, as the planner model did live (chosen_rank=1 on every decision).

PASS when:
  1. without the memory the recorded decision 2:3 (back in room 36) ranks exit N first: the bug
     reproduces from the recorded inputs;
  2. with the memory 2:3 ranks exit S first (room 52, never visited) and predicts reach_room 52;
  3. the closed loop without the memory visits only 2 rooms (the bounce);
  4. the closed loop with the memory visits all 6 rooms within 24 decisions.

  PYTHONPATH=. .venv/bin/python services/orion-hub/evals/zelda_dungeon_bounce_replay.py
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

HUB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import zelda_experience_replay as R  # noqa: E402  (loads orion-prediction's real model)
from scripts import retro_experience as X  # noqa: E402
from scripts import retro_planner as P  # noqa: E402


def run_menu(counts, decision, atlas):
    options = X.build_options(decision, atlas)
    return X.build_menu(options, counts.quotes(options))


def replay(fixture, memory):
    """Recorded decisions 2:1..2:3 in order. Returns {decision_id: (menu, atlas)}."""
    counts, out, bets = R.Counts(), {}, {}
    sid = fixture['session_id']
    for row in fixture['rows']:
        if row['kind'] == 'decision_offered':
            prev = (row.get('previous') or {}).get('decision_id')
            if prev in bets:                       # the settle table, from this decision's observation
                option, start = bets.pop(prev)
                body = X.settle_body(option.skill, start, row['previous'], row['observation'])
                if body is not None:
                    counts.add(option.skill, option.context, option.place, body.result)
            atlas = dict(copy.deepcopy(fixture['retro_atlas']), room=row['observation']['room'])
            if memory is not None:
                atlas = memory.atlas_for(sid, row, atlas)
            out[row['decision_id']] = (run_menu(counts, row, atlas), atlas, row)
        elif row['kind'] == 'goal_committed' and row['decision_id'] in out:
            _, atlas, decision = out[row['decision_id']]
            option = X.option_for_goal(row['goal'], decision, atlas)
            bets[row['decision_id']] = (option, X.start_facts(decision['observation']))
    return out


def closed_loop(fixture, memory, steps=24):
    rooms = {20, 21, 22, 36, 37, 38}
    template = next(r for r in fixture['rows'] if r['kind'] == 'decision_offered')
    counts, room, came, path = R.Counts(), 36, None, []
    for n in range(1, steps + 1):
        d = copy.deepcopy(template)
        d['decision_id'] = f'9:{n}'
        d['observation']['room'] = d['observation']['next_room'] = room
        d['open_sides'] = [s for s in 'NSEW' if P._neighbour(room, s) in rooms]
        d['previous'] = None if came is None else dict(decision_id=f'9:{n - 1}', goal={'kind': 'exit', 'dir': came},
                                                       outcome='completed', verdict='unknown')
        atlas = dict(copy.deepcopy(fixture['retro_atlas']), room=room)
        if memory is not None:
            atlas = memory.atlas_for(fixture['session_id'], d, atlas)
        top = run_menu(counts, d, atlas).entries[0].option
        counts.add(top.skill, top.context, top.place, 'success')   # every exit worked live
        path.append(room)
        room, came = P._neighbour(room, top.goal['dir']), top.goal['dir']
    return path


def main(args) -> int:
    fixture = json.loads(Path(args.fixture).read_text())
    checks = []
    before = replay(fixture, None)
    after = replay(fixture, P.RoomMemory())
    for name, runs in (('without memory', before), ('with memory', after)):
        print(f'menu 2:3 {name} (room 36, back from room 20):')
        for e in runs['2:3'][0].entries:
            print('  ' + e.line, f'[score {e.score:.3f}]')
    b, a = before['2:3'][0].entries[0].option, after['2:3'][0].entries[0].option
    checks.append(('bug reproduces: without memory 2:3 ranks exit N first', b.goal == {'kind': 'exit', 'dir': 'N'},
                   f'first={b.option_id}'))
    checks.append(('with memory 2:3 ranks exit S first, predicting reach_room 52',
                   a.goal == {'kind': 'exit', 'dir': 'S'} and a.default_plan[0] == {'kind': 'reach_room', 'value': 52},
                   f'first={a.option_id} predicts={a.default_plan[0]}'))
    path0, path1 = closed_loop(fixture, None), closed_loop(fixture, P.RoomMemory())
    print('closed loop without memory:', path0)
    print('closed loop with memory:   ', path1)
    checks.append(('closed loop without memory bounces between 2 rooms', set(path0) == {20, 36},
                   f'rooms={sorted(set(path0))}'))
    checks.append(('closed loop with memory visits all 6 rooms', set(path1) == {20, 21, 22, 36, 37, 38},
                   f'rooms={sorted(set(path1))}'))
    for name, ok, detail in checks:
        print(('ok   ' if ok else 'FAIL ') + name + ': ' + detail)
    verdict = all(ok for _, ok, _ in checks)
    print(('PASS' if verdict else 'FAIL') + f': {sum(ok for _, ok, _ in checks)}/{len(checks)} checks '
          f'(model {R.MODEL} from services/orion-prediction/app/action_bets.py)')
    return 0 if verdict else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--fixture', default=str(HUB / 'evals/fixtures/zelda_dungeon_bounce_3bf6973a.json'))
    sys.exit(main(p.parse_args()))
