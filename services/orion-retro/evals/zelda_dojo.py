"""Zelda dojo: Go-Explore over the real core, the real reader and the real goal motor.

Orion's live play gave 223 finished goals in total; no world model learns from that. Here the
emulator runs headless (about 12x real time per process) and snapshots restore in about 0.04 ms,
so experience is branched instead of lived once:

  archive = cells (room, coarse Link tile, hearts, items) -> a saved core state
  loop: pick a rarely chosen cell, restore it, offer the decision Retro would offer there, run
        every goal the motor supports there (exit per open side, fight, collect, wait) from the
        same state, record (state, goal, outcome, next state), add any new cell to the archive.

(Ecoffet et al. 2019, "Go-Explore": remember states, return to them, explore from them.)
No LLM, no Hub, no new game rules: the options are exactly the decision's open_sides, enemies
and drops, and every goal goes through ZeldaGoals as in live play. Real controller input only.

Writes <out>/transitions.jsonl (one row per goal tried), <out>/archive.json, <out>/progress.log.
The ROM and core are operator-supplied and never copied or printed; rows hold RAM-derived facts.

Run inside the retro image (core at /opt/retro/core.so, catalog at /catalog):
  python services/orion-retro/evals/zelda_dojo.py --core /opt/retro/core.so \
      --rom /catalog/the-legend-of-zelda-prg1.nes --minutes 10 --out /out/w0 --seed 0
"""
import argparse
import json
import math
from pathlib import Path
import random
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from zelda_play import FPS, Harness  # noqa: E402
from app.control import Arbiter  # noqa: E402
from app.scripted import Scripted  # noqa: E402
from orion.embodiment.transition import zelda_facts  # noqa: E402

GOAL_BUDGET = 10  # seconds of game time per goal


def cell_key(obs):
    """Place, coarse Link tile, hearts, and which items are held (not how many rupees)."""
    link = obs['link']
    held = tuple(sorted(k for k, v in obs['items'].items() if isinstance(v, int) and v > 0))
    return (obs['mode'], str(obs['level']), obs['room'], link['tile'][0] // 8, link['tile'][1] // 6,
            obs['hearts']['current'], held)


def facts(obs):
    return zelda_facts(obs)   # the shared transition contract (orion/embodiment/transition.py)


def options(decision):
    """The goals this decision supports, from Retro's own offer (no Hub scoring)."""
    obs, sides = decision['observation'], decision.get('open_sides')
    goals = [{'kind': 'exit', 'dir': d} for d in (sides or [])]
    if sides is None:
        goals += [{'kind': 'explore', 'dir': d} for d in 'NSWE']
    if obs.get('enemies'):
        goals += [{'kind': 'fight', 'target': 'nearest'}, {'kind': 'wait'}]
    for d in obs.get('drops') or []:
        goals.append({'kind': 'collect', 'tile': {'x': min(d['tile'][0], 30), 'y': max(1, min(d['tile'][1], 21))}})
    return goals


class Dojo(Harness):
    rng = random.Random(0)

    def rewind(self, state):
        """Restore a state, forget the frame before it, and idle a random 0-60 frames so a cell
        chosen again does not replay the same rows (the core is deterministic from a snapshot)."""
        self.core.restore(state)
        self.runtime.on_load(self.runtime.lineage_id())     # same lineage: clears the atlas's last frame
        self.rows.clear()
        for _ in range(self.rng.randrange(61)): self.step()

    def fresh_motor(self):
        self.authority = Arbiter([0])
        self.scripted = Scripted(self.authority, self.record, goals_factory=self.runtime.goals)
        self.hand_off()

    def decision_at(self, state):
        """Restore a state and wait for the decision the motor offers there."""
        self.rewind(state); self.fresh_motor()
        for _ in range(600):
            if self.goals.attempt: return self.goals.attempt
            self.tick()
        return None

    def try_goal(self, state, goal):
        """Run one goal from ``state``. Returns (goal_finished row, offer used, state after) or a
        ``dojo:*`` outcome with no state: the goal never ran cleanly, so it says nothing."""
        self.rewind(state); self.fresh_motor()
        try:
            row = self.goal_once(goal, {'kind': 'unknown'}, GOAL_BUDGET, 'dojo')
        except AssertionError as exc:
            return dict(outcome=f'dojo:{exc}'), None, None
        except ValueError as exc:       # Retro refused the goal (e.g. no_enemies_present)
            return dict(outcome=f'dojo:rejected:{exc}'), None, None
        offer = next((r for r in self.rows if r['kind'] == 'decision_offered'), {})
        return row, offer, self.core.snapshot()


def run(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    Dojo.rng = random.Random(args.seed + 1)
    progress = (out / 'progress.log').open('a')
    h = Dojo(args.core, args.rom, Path(tempfile.mkdtemp(prefix='zelda-dojo-')), f'dojo-{args.seed}')
    h.boot()
    if args.opening == 'sword':
        # zelda_play's fixture opening: the start cave's sword, then back out to room $77.
        h.hand_off()
        for goal, budget in (({'kind': 'enter', 'tile': {'x': 8, 'y': 2}}, 20),
                             ({'kind': 'collect', 'tile': {'x': 15, 'y': 12}}, 20), ({'kind': 'exit', 'dir': 'S'}, 20)):
            h.goal(goal, {'kind': 'unknown'}, budget)
        assert h.obs['items']['sword'] == 1 and h.obs['room'] == 0x77, 'sword opening failed'
    archive = {}   # cell -> dict(state, seen, chosen, facts)
    def add(obs, state):
        k = cell_key(obs)
        if k not in archive:
            archive[k] = dict(state=state, seen=0, chosen=0, facts=facts(obs), found_frame=h.frame)
        archive[k]['seen'] += 1
        return k
    add(h.obs, h.core.snapshot())
    t0, tried, frames = time.time(), 0, 0
    rooms = {(str(h.obs['level']), h.obs['room'])}
    with (out / 'transitions.jsonl').open('a') as sink:
        while time.time() - t0 < args.minutes * 60:
            cells = list(archive)
            weights = [1 / math.sqrt(1 + archive[c]['chosen']) + 1 / math.sqrt(1 + archive[c]['seen']) for c in cells]
            key = rng.choices(cells, weights)[0]
            cell = archive[key]; cell['chosen'] += 1
            decision = h.decision_at(cell['state'])
            if not decision or decision['observation'].get('status') != 'observed':
                continue
            before = facts(decision['observation'])
            for goal in options(decision):
                row, offer, state = h.try_goal(cell['state'], goal)
                tried += 1
                ran = state is not None and h.obs.get('status') == 'observed'
                after = facts(h.obs) if ran else None
                used = (row['end_frame'] - row['start_frame']) if ran and isinstance(row.get('start_frame'), int) \
                    and isinstance(row.get('end_frame'), int) else None
                frames += used or 0
                died = row.get('outcome') == 'death' or bool(after and after['mode'] in ('death', 'menu'))
                # The offer this goal committed against (atlas blocks learned in an earlier branch
                # can change it), and its observation as "before".
                offered = (offer or {}).get('observation') or decision['observation']
                sink.write(json.dumps(dict(seed=args.seed, before=facts(offered) if offered.get('status') == 'observed' else before,
                                           open_sides=(offer or decision).get('open_sides'), goal=goal,
                                           outcome=row.get('outcome'), verdict=row.get('verdict'),
                                           frames=used, after=after, died=died), default=str) + '\n')
                if state is not None and after and not died and h.obs['mode'] in ('play', 'cave'):
                    add(h.obs, state)
                    rooms.add((str(after['level']), after['room']))
            sink.flush()
            elapsed = time.time() - t0
            progress.write(json.dumps(dict(elapsed_s=round(elapsed), pct=round(100 * elapsed / (args.minutes * 60)),
                                           goals_tried=tried, cells=len(archive), rooms=len(rooms),
                                           game_hours=round(frames / FPS / 3600, 2),
                                           speedup=round(frames / FPS / max(elapsed, 1e-9), 1))) + '\n')
            progress.flush()
    summary = dict(seed=args.seed, goals_tried=tried, cells=len(archive), rooms=sorted(rooms),
                   game_hours=round(frames / FPS / 3600, 2), wall_minutes=args.minutes,
                   cells_by_room={f'{l}:{r}': sum(1 for k in archive if (k[1], k[2]) == (l, r)) for l, r in rooms})
    (out / 'archive.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != 'cells_by_room'}))
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--core', required=True); p.add_argument('--rom', required=True)
    p.add_argument('--out', default='/tmp/zelda-dojo'); p.add_argument('--minutes', type=float, default=10)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--opening', choices=('none', 'sword'), default='sword')
    sys.exit(run(p.parse_args()))
