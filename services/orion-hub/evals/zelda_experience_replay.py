"""Offline learning eval for action bets (spec 2026-10-23 embodied experience, acceptance check 4).

Replays the 2026-10-23 co-play session (evals/fixtures/zelda_session_c6041e2d_goals.json, exported
read-only from Retro's evidence file) through the Hub's Zelda adapter (scripts/retro_experience.py):
every committed goal becomes a bet, every next decision settles it with the adapter's settle table,
and the settled results feed an in-process count store read by orion-prediction's own model
(services/orion-prediction/app/action_bets.py ``backoff``). No GPU, no network, no database.

PASS when, after the replay:
  1. room 119's south side (place zelda:0:119:S) reads p < 0.25;
  2. room 119's west and east sides read above it;
  3. the menu for decision 15:9 does not rank "explore S" first -- both in the evidence's own
     shape (pre-#2870, no open_sides) and in the #2870 shape (evals/fixtures/zelda_room119_south_wall.json).

  PYTHONPATH=. .venv/bin/python services/orion-hub/evals/zelda_experience_replay.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HUB = Path(__file__).resolve().parents[1]
PREDICTION = HUB.parent / 'orion-prediction'


def _load_model():
    """orion-prediction's backoff(skill, situation, place, has_place). Its package is also named
    ``app``, so it is imported first with its own directory on the path, then the path is restored."""
    sys.path.insert(0, str(PREDICTION))
    try:
        from app.action_bets import MODEL, backoff  # noqa: PLC0415
        return MODEL, backoff
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f'FAIL: cannot import services/orion-prediction/app/action_bets.py ({type(exc).__name__}: '
                         f'{exc}); this eval grades the real model, not a copy') from exc
    finally:
        sys.path.remove(str(PREDICTION))


MODEL, backoff = _load_model()
sys.path.insert(0, str(HUB))
from orion.schemas.prediction import ActionBetQuoteV1  # noqa: E402
from scripts import retro_experience as X  # noqa: E402

COLS, ROWS = 16, 8   # the overworld grid: room id = row * 16 + column


class Counts:
    """The settled-bet counts orion-prediction would read (void never counts)."""

    def __init__(self):
        self.rows = []
        self.skill, self.situation, self.place = {}, {}, {}

    def add(self, skill, context, place, result):
        if result not in ('success', 'failure'):
            return
        self.rows.append((skill, tuple(sorted(context.items())), place, result))
        # Mirror services/orion-prediction/app/store.py: above the place level each place is ONE
        # unit (each of its n bets weighs 1/n); the place level keeps raw counts.
        n = {}
        for _, _, pl, _ in self.rows:
            if pl is not None: n[pl] = n.get(pl, 0) + 1
        self.skill, self.situation, self.place = {}, {}, {}
        for sk, ctx, pl, res in self.rows:
            key = 'successes' if res == 'success' else 'failures'
            w = 1.0 if pl is None else 1.0 / n[pl]
            for table, k, inc in ((self.skill, sk, w), (self.situation, (sk, ctx), w), (self.place, pl, 1)):
                if k is not None:
                    table.setdefault(k, {'successes': 0, 'failures': 0})[key] += inc

    def quote(self, option):
        q = backoff(self.skill.get(option.skill), self.situation.get((option.skill, tuple(sorted(option.context.items())))),
                    self.place.get(option.place) if option.place else None, has_place=bool(option.place))
        return ActionBetQuoteV1(option_id=option.option_id, model=MODEL, **q)

    def quotes(self, options):
        return {o.option_id: self.quote(o) for o in options}


def overworld_atlas(level, room, visited):
    """The atlas neighbourhood Retro would serve, rebuilt from the rooms seen so far in the replay
    (off-grid sides absent, as in services/orion-retro/app/zelda/atlas.py)."""
    if level != 0 or not isinstance(room, int):
        return None
    r, c = divmod(room, COLS)
    steps = {'N': (r - 1, c), 'S': (r + 1, c), 'E': (r, c + 1), 'W': (r, c - 1)}
    adjacent = {d: {'room': rr * COLS + cc, 'known': rr * COLS + cc in visited, 'seen_exit': None}
                for d, (rr, cc) in steps.items() if 0 <= rr < ROWS and 0 <= cc < COLS}
    return {'level': level, 'room': room, 'adjacent': adjacent, 'caves': []}


def replay(rows):
    counts, visited, open_bets, settled, decisions = Counts(), set(), {}, [], {}
    for row in rows:
        kind = row['kind']
        if kind == 'decision_offered':
            obs = row['observation']
            previous = row.get('previous') or {}
            bet = open_bets.pop(previous.get('decision_id'), None)
            if bet:
                body = X.settle_body(bet['option'].skill, bet['start'], previous, obs)
                if body is None:
                    open_bets[previous['decision_id']] = bet
                else:
                    counts.add(bet['option'].skill, bet['option'].context, bet['option'].place, body.result)
                    settled.append(dict(decision_id=previous['decision_id'], goal=bet['option'].goal,
                                        place=bet['option'].place, p=round(bet['p'], 3), result=body.result,
                                        outcome=body.outcome, rule=body.detail.get('rule')))
            if obs.get('status') == 'observed':
                visited.add(obs.get('room'))
            decision = dict(row, open_sides=row.get('open_sides'))
            decisions[row['decision_id']] = (decision, overworld_atlas(obs.get('level'), obs.get('room'), visited))
        elif kind == 'goal_committed':
            decision, atlas = decisions[row['decision_id']]
            option = X.option_for_goal(row['goal'], decision, atlas)
            open_bets[row['decision_id']] = dict(option=option, start=X.start_facts(decision['observation']),
                                                 p=counts.quote(option).p)
        elif kind == 'policy_stopped':
            for decision_id, bet in list(open_bets.items()):   # session end with no finished row: void
                settled.append(dict(decision_id=decision_id, goal=bet['option'].goal, result='void',
                                    outcome='session_ended'))
                open_bets.pop(decision_id)
    return counts, decisions, settled


def main(args) -> int:
    fixture = json.loads(Path(args.fixture).read_text())
    counts, decisions, settled = replay(fixture['rows'])
    for s in settled:
        print('settled', json.dumps(s))
    checks = []
    decision, atlas = decisions['15:9']          # the evidence shape: no open_sides
    options = X.build_options(decision, atlas)
    by_id = {o.option_id: o for o in options}
    south, west, east = (counts.quote(by_id[f'explore:{d}']) for d in 'SWE')
    checks.append(('room 119 S place p < 0.25', south.p < 0.25, f'p={south.p:.3f} level={south.level}'))
    checks.append(('room 119 W and E above S', west.p > south.p and east.p > south.p,
                   f'W={west.p:.3f} E={east.p:.3f} S={south.p:.3f}'))
    menu = X.build_menu(options, counts.quotes(options))
    print('menu 15:9 (evidence shape, open_sides not recorded):')
    for e in menu.entries:
        print('  ' + e.line, f'[score {e.score:.3f}]')
    checks.append(('menu 15:9 does not rank explore S first', menu.entries[0].option.goal != {'kind': 'explore', 'dir': 'S'},
                   f'first={menu.entries[0].option.option_id}'))
    wall = json.loads(Path(args.wall_fixture).read_text())   # the #2870 shape: open_sides N, W, E
    menu2 = X.build_menu(X.build_options(wall['decision'], wall['atlas']),
                         counts.quotes(X.build_options(wall['decision'], wall['atlas'])))
    print('menu 15:9 (#2870 shape, open_sides N, W, E):')
    for e in menu2.entries:
        print('  ' + e.line, f'[score {e.score:.3f}]')
    checks.append(('menu 15:9 (#2870 shape) does not rank explore S first',
                   menu2.entries[0].option.goal != {'kind': 'explore', 'dir': 'S'},
                   f'first={menu2.entries[0].option.option_id}'))
    graded = [s for s in settled if s['result'] in ('success', 'failure')]
    checks.append(('every committed goal settled', len(settled) == 8 and len(graded) == 8,
                   f'settled={len(settled)} graded={len(graded)}'))
    for name, ok, detail in checks:
        print(('ok   ' if ok else 'FAIL ') + name + ': ' + detail)
    verdict = all(ok for _, ok, _ in checks)
    print(('PASS' if verdict else 'FAIL') + f': {sum(ok for _, ok, _ in checks)}/{len(checks)} checks '
          f'(model {MODEL} from services/orion-prediction/app/action_bets.py)')
    return 0 if verdict else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--fixture', default=str(HUB / 'evals/fixtures/zelda_session_c6041e2d_goals.json'))
    p.add_argument('--wall-fixture', default=str(HUB / 'evals/fixtures/zelda_room119_south_wall.json'))
    sys.exit(main(p.parse_args()))
