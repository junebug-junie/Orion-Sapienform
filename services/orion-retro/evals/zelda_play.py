"""Native Zelda navigator + atlas eval (acceptance checks 1-3). No LLM, no live session.

From power-on, with real controller input only (no RAM writes, no cheats), a fixed
goal sequence runs through the same Scripted/Arbiter/ZeldaGoals path the worker uses:
enter the start cave -> collect the sword -> exit S -> exit E -> fight an Octorok.
A bump is elicited where the room map is open but the cave's invisible barrier
stops Link. Then a second session restores the save state on the same atlas lineage.
Evidence (RAM facts, goal_finished records, atlas) goes to --output (default /tmp/zelda-play).
The ROM and core are operator-supplied and never copied or printed.

Run: python services/orion-retro/evals/zelda_play.py --core <fceumm ac0d00a7 .so> --rom <zelda .nes>
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'services/orion-retro'))
from orion.retro.probe import CoreProbe  # noqa: E402
from app.control import Arbiter  # noqa: E402
from app.scripted import Scripted  # noqa: E402
from app.zelda.runtime import ZeldaRuntime  # noqa: E402
from app.zelda.atlas import load as load_atlas, neighbourhood  # noqa: E402

B, SEL, START, UP, DOWN, LEFT, RIGHT, A = 1, 4, 8, 16, 32, 64, 128, 256
FPS = 60.0988


class Harness:
    """One native core, driven frame by frame on a simulated clock (frame / fps)."""

    def __init__(self, core, rom, saves, session_id, lineage_id=None):
        self.rows = []
        self.core = CoreProbe(Path(core), Path(rom))
        self.frame = 0
        self.runtime = ZeldaRuntime(self.core, saves, session_id, self.record, lineage_id,
                                    now=lambda: self.now)
        assert self.runtime.supported, 'unqualified ROM/core fingerprint'
        self.authority = Arbiter([0])
        self.scripted = Scripted(self.authority, self.record, goals_factory=self.runtime.goals)

    @property
    def now(self):
        return self.frame / FPS

    def record(self, kind, **fields):
        self.rows.append(dict(kind=kind, frame=self.frame, **fields))

    def ram(self, address):
        import ctypes as C
        lib = self.core.lib
        return C.string_at(lib.retro_get_memory_data(2), 2048)[address]

    def step(self, mask=0, n=1):
        for _ in range(n):
            self.core.step(mask); self.frame += 1
            self.obs = self.runtime.after_frame(self.frame, self.now, {})

    def wait_mode(self, code, limit):
        start = self.frame
        while self.ram(0x12) != code and self.frame - start < limit: self.step()
        return self.ram(0x12) == code

    def press(self, mask, wait=20, hold=2):
        self.step(mask, hold); self.step(0, wait)

    def boot(self):
        """Power-on -> register one name -> start slot 1. Reads the menu cursor ($16)."""
        self.step(0, 120)
        self.press(START, 60)
        assert self.wait_mode(1, 300), 'file select'
        while self.ram(0x16) != 3 and self.frame < 1000: self.press(SEL, 10)
        self.press(START, 60)
        assert self.wait_mode(0x0E, 300), 'register'
        self.press(A, 10)
        while self.ram(0x16) != 3 and self.frame < 3000: self.press(SEL, 10)
        self.press(START, 60)
        assert self.wait_mode(1, 300), 'back to file select'
        while self.ram(0x16) != 0 and self.frame < 4000: self.press(SEL, 10)
        self.press(START, 10)
        assert self.wait_mode(5, 600), 'gameplay'
        self.step(0, 2)

    def hand_off(self, seconds=1800):
        """Same transfer as the worker: a live human lease hands its port to the motor."""
        grant = self.authority.control('eval', 0, 'acquire', self.now)
        self.scripted.start('eval', 0, seconds, self.now, self.frame, grant['epoch'], True)
        self.policy = self.scripted.active[0]
        self.goals = self.policy['goals']

    def tick(self):
        self.scripted.tick(self.now)
        masks, applied, _ = self.authority.frame(self.now)
        self.core.step(*masks); self.frame += 1
        self.obs = self.runtime.after_frame(self.frame, self.now, self.scripted.active)
        for event in applied: self.scripted.annotate(event)  # same order as the worker

    def goal(self, goal, prediction, budget=20, rationale='native eval fixture sequence', retries=3):
        """Commit one goal and run it to its end; re-issue it after an interruption (a hit)."""
        for _ in range(retries):
            row = self.goal_once(goal, prediction, budget, rationale)
            if not row['outcome'].startswith('interrupted:hit'): return row
        return row

    def goal_once(self, goal, prediction, budget, rationale):
        """Wait for the next offered decision, commit one goal, run it to its end."""
        for _ in range(600):
            if self.goals.attempt and not self.goals.accepted: break
            self.tick()
        decision = self.goals.attempt
        assert decision, 'no decision offered'
        data = dict(port=0, expected_epoch=decision['expected_epoch'], decision_id=decision['decision_id'],
                    goal=goal, prediction=prediction, rationale=rationale, planner_id='zelda-eval-fixture',
                    budget_seconds=budget)
        self.scripted.goal('eval', data, self.now)
        before = len([r for r in self.rows if r['kind'] == 'goal_finished'])
        for _ in range(int(budget * FPS) + 120):
            self.tick()
            done = [r for r in self.rows if r['kind'] == 'goal_finished']
            if len(done) > before: return done[-1]
        raise AssertionError('goal never finished')

    def facts(self):
        o = self.obs
        return dict(mode=o['mode'], level=o['level'], room=o['room'], link=o['link'],
                    hearts=o['hearts'], items=o['items'], enemies=o['enemies'])


def check(results, name, ok, **evidence):
    results.append(dict(check=name, ok=bool(ok), **evidence))
    print(('PASS ' if ok else 'FAIL ') + name, json.dumps(evidence, default=str)[:300])


def run(args):
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    journals = {}
    try:
        return _run(args, out, journals)
    finally:
        for name, rows in journals.items():
            (out / f'journal_{name}.json').write_text(json.dumps(rows, default=str))


def _run(args, out, journals):
    saves = Path(args.saves or tempfile.mkdtemp(prefix='zelda-atlas-'))
    results = []; evidence_drops = []
    h = Harness(args.core, args.rom, saves, 'eval-session-1')
    journals['session_1'] = h.rows
    lineage = h.runtime.lineage_id()
    h.boot()
    start = h.facts()
    check(results, 'reader: boot reaches play at start room $77', start['mode'] == 'play' and start['room'] == 0x77,
          facts=start)
    h.press(SEL, 10)
    paused = h.facts() | {'paused': h.obs['paused']}
    h.press(SEL, 10)
    check(results, 'reader: SELECT pause reads $E0 and freezes control',
          paused['paused'] is True and not paused['link']['controllable'] and h.obs['paused'] is False
          and h.obs['link']['controllable'], paused=paused['paused'], after=h.obs['paused'])
    h.hand_off()
    seq = []
    r = h.goal({'kind': 'enter', 'tile': {'x': 8, 'y': 2}}, {'kind': 'enter_cave'}); seq.append(r)
    check(results, 'enter cave', r['verdict'] == 'met', finished=r)
    # The cave's invisible barrier: the room map is open above the item row, the game is not.
    r = h.goal({'kind': 'collect', 'tile': {'x': 15, 'y': 6}}, {'kind': 'unknown'}, budget=8); seq.append(r)
    bumps = [x for x in h.rows if x['kind'] == 'bump']
    blocked = h.runtime.atlas.blocked(0, 0x77, 'cave', h.obs['walls_hash'])
    check(results, 'bump marks a learned-blocked tile', bumps and blocked, bumps=bumps[:3], blocked=sorted(blocked))
    r = h.goal({'kind': 'collect', 'tile': {'x': 15, 'y': 12}}, {'kind': 'item_gained'}); seq.append(r)
    check(results, 'collect sword (RAM $657 = 1)', r['verdict'] == 'met' and h.obs['items']['sword'] == 1,
          finished=r, sword=h.obs['items']['sword'])
    r = h.goal({'kind': 'exit', 'dir': 'S'}, {'kind': 'reach_room', 'value': 0x77}); seq.append(r)
    check(results, 'exit S leaves the cave to room $77', r['verdict'] == 'met' and h.obs['room'] == 0x77, finished=r)
    r = h.goal({'kind': 'exit', 'dir': 'E'}, {'kind': 'reach_room', 'value': 0x78}); seq.append(r)
    check(results, 'exit E reaches room $78', r['verdict'] == 'met' and h.obs['room'] == 0x78, finished=r)
    for _ in range(90): h.tick()  # let the room's enemies spawn
    kills, before, fights = [], {}, 0
    while not kills and fights < 4 and h.obs['mode'] == 'play':
        before = {e['slot']: e for e in h.obs['enemies']}
        if not before:
            for _ in range(60): h.tick()
            fights += 1; continue
        link = h.obs['link']
        target = min(before.values(), key=lambda e: abs(e['x'] - link['x']) + abs(e['y'] - link['y']))
        r = h.goal({'kind': 'fight', 'target': target['slot']}, {'kind': 'survive'}, budget=30); seq.append(r)
        fights += 1
        alive = {x['slot']: x for x in h.obs['enemies']}
        if target['slot'] not in alive or alive[target['slot']]['type'] != target['type']:
            kills.append(target)
    check(results, 'fight kills an Octorok (slot hp>0 -> gone)', kills and kills[0]['type'] in (7, 8),
          killed=kills, after=h.obs['enemies'], finished=r)
    # Evidence only (drops are random): collect any drop and record the inventory delta.
    for d in list(h.obs['drops'])[:2]:
        before_items = dict(h.obs['items'])
        g = h.goal({'kind': 'collect', 'tile': {'x': min(d['tile'][0], 30), 'y': max(1, min(d['tile'][1], 21))}},
                   {'kind': 'item_gained'}, budget=8)
        evidence_drops.append(dict(drop=d, before=before_items, after=dict(h.obs['items']), finished=g))
    # Forced death: stand still among the Octoroks until Link dies, then the motor's own
    # death recovery must pick Continue on the menu (cursor read from $13) and play resumes.
    died = None
    for _ in range(8):
        if h.obs['mode'] != 'play': break
        r = h.goal({'kind': 'wait'}, {'kind': 'survive'}, budget=30); seq.append(r)
        if r['outcome'] == 'death': died = r; break
    menus = []
    for _ in range(1500):
        h.tick()
        if h.obs.get('menu'): menus.append(h.obs['menu'])
        if any(x['kind'] == 'death_recovered' for x in h.rows) and h.goals.attempt: break
    recovered = [x for x in h.rows if x['kind'] == 'death_recovered']
    check(results, 'forced death -> Continue (cursor $13) -> back in play with a new decision',
          died and died['verdict'] == 'refuted' and recovered and recovered[0]['decision_id'] == died['decision_id']
          and menus and menus[0]['cursor'] == 'continue' and h.obs['mode'] == 'play' and h.obs['room'] == 0x77
          and h.goals.attempt is not None and h.obs['hearts']['current'] == 3,
          death=died, recovered=recovered, menu_first=menus[:1], menu_last=menus[-1:], room=h.obs['room'],
          hearts=h.obs['hearts'], new_decision=(h.goals.attempt or {}).get('decision_id'))
    h.scripted.stop(0, 'eval_end')
    state = h.core.snapshot()
    facts1 = h.facts()
    h.runtime.close(); h.core.close()
    atlas1 = load_atlas(saves, lineage)
    s1 = atlas1['sessions'][-1]
    check(results, 'atlas session 1: start, east room, cave and sword recorded',
          {'0:119', '0:120'} <= set(atlas1['rooms']) and atlas1['rooms']['0:119']['exits_seen']['E'] == 120
          and atlas1['rooms']['0:119']['caves'] and 'sword' in atlas1['rooms']['0:119']['caves'][0]['contents_seen']
          and s1['rooms_new'] == 2 and any(i['item'] == 'sword' for i in atlas1['items_gained']),
          session=s1, rooms=sorted(atlas1['rooms']), caves=atlas1['rooms']['0:119']['caves'])

    # Session 2: same lineage, restored from the session-1 save state.
    h2 = Harness(args.core, args.rom, saves, 'eval-session-2', lineage)
    journals['session_2'] = h2.rows
    h2.step(0, 1)
    h2.core.restore(state)
    h2.step(0, 2)
    facts2 = h2.facts()
    check(results, 'save-state continuity: sword and room survive restore',
          facts2['items']['sword'] == 1 and facts2['room'] == facts1['room'] == 0x77, before=facts1, after=facts2)
    h2.hand_off()
    seq2 = []
    r = h2.goal({'kind': 'exit', 'dir': 'E'}, {'kind': 'reach_room', 'value': 0x78}); seq2.append(r)
    r = h2.goal({'kind': 'exit', 'dir': 'N'}, {'kind': 'reach_room', 'value': 0x68}, budget=25); seq2.append(r)
    h2.scripted.stop(0, 'eval_end')
    h2.runtime.close(); h2.core.close()
    atlas2 = load_atlas(saves, lineage)
    s2 = atlas2['sessions'][-1]
    new_in_2 = sorted(k for k, v in atlas2['rooms'].items() if v['first_session'] == 'eval-session-2')
    check(results, 'atlas session 2: known rooms are not new, the new one is',
          new_in_2 == ['0:104'] and s2['rooms_new'] == 1
          and atlas2['rooms']['0:120']['visits'] > atlas1['rooms']['0:120']['visits']
          and atlas2['rooms']['0:120']['exits_seen']['N'] == 104,
          session2=s2, new_in_session_2=new_in_2, finished=seq2)
    hood = neighbourhood(atlas2, 0, 0x68)
    report = dict(results=results, ok=all(r['ok'] for r in results), lineage_id=lineage,
                  drop_pickups=evidence_drops, goals_session_1=seq, goals_session_2=seq2, neighbourhood=hood,
                  atlas_bytes=len(json.dumps(atlas2)), identity=h2.core.identity)
    (out / 'report.json').write_text(json.dumps(report, indent=1, default=str))
    (out / 'atlas.json').write_text(json.dumps(atlas2, indent=1))
    print('OK' if report['ok'] else 'FAILED', out / 'report.json')
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--core', required=True)
    p.add_argument('--rom', required=True)
    p.add_argument('--output', default='/tmp/zelda-play')
    p.add_argument('--saves', default=None, help='atlas root (default: a fresh temp dir)')
    sys.exit(run(p.parse_args()))
