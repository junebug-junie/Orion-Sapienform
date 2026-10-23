"""Zelda reader / GoalV2 motor / atlas: deterministic, synthetic RAM only (no ROM, no core)."""
import ctypes as C
import json
from types import SimpleNamespace
import pytest
from pydantic import ValidationError
from app.zelda import observe
from app.zelda.observe import decode, ZeldaObserver, tile_of, ROMS, CORES
from app.zelda.navigate import ZeldaGoals, astar, judge, tile_xy, UP, DOWN, LEFT, RIGHT, A, STUCK_FRAMES
from app.zelda.atlas import Atlas, load, neighbourhood, path_for, MAX_BYTES
from app.contracts import GoalV2, LeisureControl
from app.control import Arbiter
from app.scripted import Scripted

ROM, CORE = next(iter(ROMS)), next(iter(CORES))
IDENTITY = dict(rom_sha256=ROM, core_sha256=CORE)


def snapshot(mode=5, sub=0, room=0x77, next_room=0, x=120, y=141, facing=8, hv=0x22, partial=0xFF, sword=0,
             rupees=0, enemies=(), blocked=None, ad=0, lift=0, paused=0, level=0, inv=0, drops=()):
    ram, save = bytearray(2048), bytearray(8192)
    for addr, value in {0x12: mode, 0x13: sub, 0xEB: room, 0xEC: next_room, 0x70: x, 0x84: y, 0x98: facing,
                        0x66F: hv, 0x670: partial, 0x657: sword, 0x66D: rupees, 0x34A: 0x89,
                        0xAD: ad, 0x506: lift, 0xE0: paused, 0x10: level, 0x4F0: inv}.items():
        ram[addr] = value
    for slot, kind, ex, ey, hp in enemies:
        ram[0x34F + slot], ram[0x70 + slot], ram[0x84 + slot], ram[0x485 + slot] = kind, ex, ey, hp
    for slot, dx, dy in drops:
        ram[0x34F + slot], ram[0x70 + slot], ram[0x84 + slot] = 0x60, dx, dy
    blocked = {(c, r) for c in range(32) for r in range(18, 22)} if blocked is None else blocked
    for c in range(32):
        for r in range(22):
            save[0x530 + c * 22 + r] = 0x90 if (c, r) in blocked else 0x24
    return bytes(ram), bytes(save)


def obs(**kw):
    return decode(*snapshot(**kw))


# ---- reader ------------------------------------------------------------------------
def test_fixture_snapshot_decodes_to_expected_observation():
    o = obs(enemies=[(3, 7, 200, 125, 16), (4, 7, 10, 10, 0)], drops=[(5, 96, 141)])
    assert o['status'] == 'observed' and o['profile'] == 'zelda-1-ram-v1'
    assert (o['mode'], o['level'], o['room']) == ('play', 0, 0x77)
    assert o['link'] == dict(x=120, y=141, tile=[15, 11], facing='N', invincible_frames=0, controllable=True)
    assert o['hearts'] == dict(current=3, containers=3)
    assert o['enemies'] == [dict(slot=3, type=7, x=200, y=125, tile=[25, 9], hp=16)]  # hp 0 = not alive
    assert o['drops'] == [dict(slot=5, x=96, y=141, tile=[12, 11])]
    assert len(o['walls']) == 22 and len(o['walls'][0]) == 32
    assert o['walls'][18][0] is True and o['walls'][11][15] is False
    assert 'walls' not in observe.summary(o) and o['walls_hash'] == observe.summary(o)['walls_hash']


@pytest.mark.parametrize('hv,partial,current', [(0x22, 0xFF, 3), (0x22, 0x7F, 2.5), (0x21, 0xFE, 2),
                                                 (0x20, 0x7D, .5), (0x20, 0, 0)])
def test_hearts_formula_matches_live_damage_ladder(hv, partial, current):
    assert obs(hv=hv, partial=partial)['hearts'] == dict(current=current, containers=3)


def test_unsettled_fields_read_unknown_never_zero():
    o = obs(level=3)
    for name in ('bombs', 'keys', 'candle', 'raft'):
        assert o['items'][name] == 'unknown'
    assert o['level'] == 'unknown'  # dungeon level has no live probe yet
    assert o['items']['sword'] == 0 and o['items']['rupees'] == 0 and o['paused'] is False


def test_unknown_fingerprint_and_bad_sizes_are_unknown():
    assert ZeldaObserver(SimpleNamespace(identity={'rom_sha256': ROM, 'core_sha256': '0' * 64})).read()['status'] == 'unknown'
    assert ZeldaObserver(SimpleNamespace(identity={})).read()['reason'] == 'unsupported_rom_or_core'
    assert decode(bytes(10), bytes(8192))['status'] == 'unknown'
    ram, save = snapshot(); assert decode(ram[:2047], save)['status'] == 'unknown'
    ram = bytearray(ram); ram[0x34A] = 0
    assert decode(bytes(ram), save)['reason'] == 'unqualified_game_state'


def test_walls_only_in_map_modes_and_control_gates():
    assert obs(mode=7)['walls_hash'] is None and 'walls' not in obs(mode=7)
    assert obs(mode=0x0B, sub=4)['mode'] == 'transition'          # cave still loading
    assert not obs(mode=0x0B, ad=1)['link']['controllable']        # cave text typing
    assert not obs(mode=0x0B, ad=0)['link']['controllable']        # cave prices transfer (live: frozen)
    assert obs(mode=0x0B, ad=2)['link']['controllable']
    assert not obs(lift=40)['link']['controllable']                # item held up
    assert not obs(paused=1)['link']['controllable'] and obs(paused=1)['paused'] is True
    assert obs(mode=0x11)['mode'] == 'death'


def test_observer_reads_memory_without_writing():
    ram, save = snapshot()
    bufs = {2: C.create_string_buffer(ram, 2048), 0: C.create_string_buffer(save, 8192)}
    lib = SimpleNamespace(retro_get_memory_size=lambda r: len(bufs[r]), retro_get_memory_data=lambda r: C.addressof(bufs[r]))
    o = ZeldaObserver(SimpleNamespace(identity=IDENTITY, lib=lib)).read(7)
    assert o['status'] == 'observed' and o['frame'] == 7
    assert bufs[2].raw == ram and bufs[0].raw == save


# ---- motor ---------------------------------------------------------------------------
class World:
    """Tiny deterministic stand-in for the game: 2 px/frame, exits at the screen edge."""

    def __init__(self, hidden=(), **kw):
        self.kw = dict(kw); self.hidden = set(hidden)

    def step(self, mask):
        k = self.kw; x, y = k.get('x', 120), k.get('y', 141)
        dx = 2 if mask & RIGHT else -2 if mask & LEFT else 0
        dy = 2 if mask & DOWN else -2 if mask & UP else 0
        if mask & RIGHT and x >= 240:
            k.update(room=k.get('room', 0x77) + 1, x=0); return
        # The game's own probe (GetCollidableTile): hotspot Y+$B, then -8 up / +8 down,
        # X-8 left / X+16 right; the probed tile must be open.
        px = x - 8 if dx < 0 else x + 16 if dx > 0 else x
        py = y + 11 + (-8 if dy < 0 else 8 if dy > 0 else 0)
        c, r = min(max(px // 8, 0), 31), (py - 64) // 8
        walls = k.get('blocked')
        walls = {(c, r) for c in range(32) for r in range(18, 22)} if walls is None else walls
        if (c, r) not in self.hidden and (c, r) not in walls:
            k.update(x=x + dx, y=y + dy)

    def obs(self):
        return obs(**self.kw)


def run_goal(goal, prediction, world=None, frames=900, atlas=None, budget=30, setup=None):
    rows = []
    g = ZeldaGoals(lambda kind, **f: rows.append(dict(kind=kind, **f)), atlas)
    world = world or World()
    g.observe_frame(0, 2, 1, world.obs(), 0)
    offered = g.attempt
    data = dict(port=0, expected_epoch=2, decision_id=offered['decision_id'], goal=goal, prediction=prediction,
                rationale='test', planner_id='test', budget_seconds=budget)
    g.accept(data, 0)
    for f in range(frames):
        if setup: setup(f, world)
        mask = g.mask(0, f)
        if g.accepted: g.note_applied(g.accepted['decision_id'], mask)  # the arbiter applied it
        world.step(mask)
        g.observe_frame(0, 2, f + 2, world.obs(), (f + 2) / 60)
        done = [r for r in rows if r['kind'] == 'goal_finished']
        if done: return done[-1], rows, g
    return None, rows, g


def test_astar_routes_around_walls():
    walls = obs(blocked={(c, 10) for c in range(4, 32)})['walls']
    path = astar(walls, set(), (10, 11), [(10, 9)])
    assert path and path[-1] == (10, 9) and all(not walls[r][c] for c, r in path)
    assert astar(walls, {(c, r) for c in range(32) for r in range(22)} - {(10, 11)}, (10, 11), [(10, 9)]) is None
    assert tile_of(*tile_xy(15, 11)) == (15, 11)


def test_exit_east_meets_reach_room():
    row, rows, _ = run_goal({'kind': 'exit', 'dir': 'E'}, {'kind': 'reach_room', 'value': 0x78})
    assert row['outcome'] == 'prediction_met' and row['verdict'] == 'met'
    assert row['applied_frames'] > 0 and row['human_intervention'] is False
    assert [r['kind'] for r in rows][:3] == ['decision_offered', 'goal_committed', 'goal_action_started']


def test_wrong_room_prediction_is_refuted():
    row, _, _ = run_goal({'kind': 'exit', 'dir': 'E'}, {'kind': 'reach_room', 'value': 0x10})
    assert row['outcome'] == 'prediction_refuted' and row['verdict'] == 'refuted'


def test_bump_marks_learned_blocked_tile_and_motor_replans(tmp_path):
    atlas = Atlas(tmp_path, Atlas.new_lineage(), IDENTITY, 's1', now=lambda: 0)
    atlas.observe(obs(), 1)
    hidden = {(c, r) for c in range(19, 22) for r in range(10, 13)}  # open on the map, solid in "game"
    row, rows, _ = run_goal({'kind': 'exit', 'dir': 'E'}, {'kind': 'reach_room', 'value': 0x78},
                            World(hidden=hidden), atlas=atlas, frames=3000)
    bumps = [r for r in rows if r['kind'] == 'bump']
    assert bumps and row['bumps'] == len(bumps)
    learned = atlas.blocked(0, 0x77, 'play', obs()['walls_hash'])
    assert learned and learned <= {tuple(b['tile']) for b in bumps}
    assert atlas.blocked(0, 0x77, 'play', 'another-layout') == set()  # never inherited by a new layout
    assert row['verdict'] == 'met'  # it found a way round


def test_stuck_and_no_path_end_the_goal_never_spin():
    boxed = {(c, r) for c in range(32) for r in range(22)} - {(15, 11), (16, 11)}
    row, rows, _ = run_goal({'kind': 'exit', 'dir': 'N'}, {'kind': 'unknown'}, World(blocked=boxed))
    assert row['outcome'] == 'interrupted:stuck' and row['verdict'] == 'unknown'
    w = World(hidden={(c, r) for c in range(32) for r in range(22)} - {(15, 11)})
    row, _, _ = run_goal({'kind': 'exit', 'dir': 'E'}, {'kind': 'unknown'}, w, frames=4000)
    assert row['outcome'] == 'interrupted:stuck' and row['verdict'] == 'unknown'


def test_hit_interrupts_navigation_but_not_combat():
    hurt = lambda f, w: w.kw.update(partial=0x7F) if f == 5 else None
    row, _, _ = run_goal({'kind': 'exit', 'dir': 'E'}, {'kind': 'reach_room', 'value': 0x78}, setup=hurt)
    assert row['outcome'] == 'interrupted:hit' and row['verdict'] == 'unknown'
    w = World(enemies=[(3, 7, 200, 141, 16)])
    row, _, _ = run_goal({'kind': 'fight', 'target': 3}, {'kind': 'survive'}, w, frames=20, setup=hurt)
    assert row is None  # still fighting after the hit


def test_death_refutes_and_budget_closes_predictions():
    die = lambda f, w: w.kw.update(mode=0x11) if f == 3 else None
    row, _, _ = run_goal({'kind': 'wait'}, {'kind': 'survive'}, setup=die)
    assert row['outcome'] == 'death' and row['verdict'] == 'refuted'
    row, _, _ = run_goal({'kind': 'wait'}, {'kind': 'survive'}, budget=3, frames=400)
    assert row['outcome'] == 'budget_expired' and row['verdict'] == 'met'
    row, _, _ = run_goal({'kind': 'wait'}, {'kind': 'enemies_cleared'}, World(enemies=[(3, 7, 40, 60, 16)]), budget=3, frames=400)
    assert row['verdict'] == 'refuted'


def test_fight_faces_and_swings_only_inside_the_goal():
    w = World(enemies=[(3, 7, 160, 141, 16)], facing=1, sword=1)
    g = ZeldaGoals(lambda *a, **k: None)
    g.observe_frame(0, 2, 1, w.obs(), 0)
    assert g.mask(0, 0) == 0  # an enemy in reach, no accepted goal: no reflex
    _, rows, g = run_goal({'kind': 'fight', 'target': 'nearest'}, {'kind': 'enemies_cleared'}, w, frames=1)
    assert g.current_mask == A
    kill = lambda f, w: w.kw.update(enemies=[]) if f == 4 else None
    row, _, _ = run_goal({'kind': 'fight', 'target': 3}, {'kind': 'survive'}, World(enemies=[(3, 7, 160, 141, 16)], facing=1), setup=kill)
    assert row['outcome'] == 'completed' and row['verdict'] == 'met'


def test_accept_validates_decision_target_and_observation():
    rows = []
    g = ZeldaGoals(lambda kind, **f: rows.append(kind))
    g.observe_frame(0, 2, 1, obs(enemies=[(3, 7, 40, 60, 16)]), 0)
    d = g.attempt['decision_id']
    base = dict(port=0, expected_epoch=2, decision_id=d, goal={'kind': 'wait'}, prediction={'kind': 'unknown', 'value': None},
                rationale='t', planner_id='t', budget_seconds=5)
    for bad, reason in ((dict(decision_id='9:9'), 'stale_decision'), (dict(expected_epoch=3), 'stale_decision'),
                        (dict(goal={'kind': 'fight', 'target': 7}), 'target_not_present'),
                        (dict(prediction={'kind': 'reach_room', 'value': None}), 'prediction_value_mismatch')):
        with pytest.raises(ValueError, match=reason): g.accept(base | bad, 1)
    with pytest.raises(ValueError, match='expired'): g.accept(base, 31)
    assert g.accept(base, 1)['goal'] == {'kind': 'wait'}
    assert g.accept(base, 2)['decision_id'] == d          # idempotent retry
    assert g.deadline == 6                                  # retry never extends the budget
    with pytest.raises(ValueError, match='conflict'): g.accept(base | {'rationale': 'other'}, 2)
    g2 = ZeldaGoals(lambda *a, **k: None)
    g2.observe_frame(0, 2, 1, obs(mode=7), 0)
    assert g2.attempt is None                               # no offer while scrolling


def test_takeover_abort_records_human_intervention():
    rows = []
    g = ZeldaGoals(lambda kind, **f: rows.append(dict(kind=kind, **f)))
    g.observe_frame(0, 2, 1, obs(), 0)
    g.accept(dict(port=0, expected_epoch=2, decision_id=g.attempt['decision_id'], goal={'kind': 'exit', 'dir': 'E'},
                  prediction={'kind': 'unknown', 'value': None}, rationale='t', planner_id='t', budget_seconds=5), 0)
    g.observe_frame(0, 2, 2, obs(), 0)
    g.abort('human_takeover')
    end = rows[-1]
    assert end['kind'] == 'goal_finished' and end['outcome'] == 'stopped:human_takeover' and end['human_intervention']


def test_judge_rules():
    s = obs()
    assert judge({'kind': 'reach_room', 'value': 0x78}, s, obs(room=0x78)) == 'met'
    assert judge({'kind': 'reach_room', 'value': 0x77}, obs(mode=0x0B, ad=2), obs()) == 'met'  # left the cave
    assert judge({'kind': 'enter_cave'}, s, obs(mode=0x10)) == 'met'
    assert judge({'kind': 'item_gained'}, s, obs(rupees=1)) == 'met'
    assert judge({'kind': 'item_gained'}, s, obs(room=0x78)) == 'refuted'
    assert judge({'kind': 'unknown'}, s, obs(mode=0x11)) == 'pending'
    assert judge({'kind': 'survive'}, s, {'status': 'unknown'}) == 'pending'


# ---- atlas ---------------------------------------------------------------------------
def walk(atlas, frames):
    for i, o in enumerate(frames): atlas.observe(o, i)


def test_atlas_records_only_observed_transitions_items_and_deaths(tmp_path):
    lineage = Atlas.new_lineage()
    a = Atlas(tmp_path, lineage, IDENTITY, 's1', now=lambda: 0)
    walk(a, [obs(), obs(x=64, y=77), obs(mode=0x10, x=64, y=77), obs(mode=0x0B, ad=2), obs(mode=0x0B, ad=2, sword=1),
             obs(), obs(mode=7, room=0x77, next_room=0x78), obs(room=0x78), obs(room=0x78, partial=0x7F), obs(room=0x78, mode=0x11)])
    assert a.flush(force=True)
    doc = load(tmp_path, lineage)
    r119, r120 = doc['rooms']['0:119'], doc['rooms']['0:120']
    assert r119['exits_seen']['E'] == 120 and r120['exits_seen']['W'] == 119 and r119['exits_seen']['N'] is None
    assert r119['caves'] == [dict(tile=[8, 2], entered=True, contents_seen=['sword'])]
    assert doc['items_gained'][0]['item'] == 'sword' and r120['deaths'] == 1 and r120['damage_taken'] == 1
    assert doc['sessions'][-1]['rooms_new'] == 2  # the cave interior is not a room
    hood = neighbourhood(doc, 0, 0x78)
    assert hood['adjacent']['W'] == dict(room=119, known=True, seen_exit=119) and hood['rooms_known'] == 2
    assert 'S' not in neighbourhood(doc, 0, 0x77)['adjacent']  # 135 is off the 8x16 overworld


def test_atlas_second_session_does_not_count_known_rooms(tmp_path):
    lineage = Atlas.new_lineage()
    a = Atlas(tmp_path, lineage, IDENTITY, 's1', now=lambda: 0)
    walk(a, [obs(), obs(mode=7, next_room=0x78), obs(room=0x78)]); a.end_session(); a.flush(force=True)
    b = Atlas(tmp_path, lineage, IDENTITY, 's2', now=lambda: 0)
    walk(b, [obs(room=0x78), obs(), obs(room=0x67)]); b.flush(force=True)
    doc = load(tmp_path, lineage)
    assert [s['rooms_new'] for s in doc['sessions']] == [2, 1]
    assert doc['rooms']['0:119']['visits'] == 2 and doc['rooms']['0:103']['first_session'] == 's2'


def test_atlas_identity_mismatch_bounds_and_io(tmp_path):
    lineage = Atlas.new_lineage()
    a = Atlas(tmp_path, lineage, IDENTITY, 's1', now=lambda: 0); walk(a, [obs()]); a.flush(force=True)
    other = Atlas(tmp_path, lineage, dict(rom_sha256='1' * 64, core_sha256=CORE), 's2', now=lambda: 0)
    assert other.doc['rooms'] == {} and other.lineage_id != lineage  # never mixes a different ROM's map
    with pytest.raises(ValueError): path_for(tmp_path, '../etc')
    empty = Atlas(tmp_path, Atlas.new_lineage(), IDENTITY, 's3', now=lambda: 0)
    assert empty.flush(force=True) is False  # nothing observed: no file
    a.doc['items_gained'] = [dict(item='x' * 100)] * (MAX_BYTES // 100)
    a.dirty = True
    assert a.flush(force=True) is False and a.disabled == 'atlas_full'
    ro = Atlas(tmp_path / 'missing' / 'deep', Atlas.new_lineage(), IDENTITY, 's4', now=lambda: 0)
    (tmp_path / 'missing').write_text('a file, not a dir')
    walk(ro, [obs()])
    assert ro.flush(force=True) is False and ro.disabled == 'atlas_io_failed'


def test_flush_is_throttled(tmp_path):
    clock = [0.0]
    a = Atlas(tmp_path, Atlas.new_lineage(), IDENTITY, 's1', now=lambda: clock[0])
    walk(a, [obs()]); assert a.flush()
    walk(a, [obs(room=0x78)]); assert not a.flush()
    clock[0] = 3; assert a.flush()


# ---- contracts and arbiter -------------------------------------------------------------
def goal_body(**kw):
    return dict(dict(connection_id='c' * 36, generation='g' * 36, port=0, expected_epoch=2, decision_id='2:1',
                     goal={'kind': 'exit', 'dir': 'E'}, prediction={'kind': 'reach_room', 'value': 120},
                     rationale='r', planner_id='p', budget_seconds=10), **kw)


def test_goalv2_contract_is_strict():
    assert GoalV2.model_validate_json(json.dumps(goal_body())).sponsor() == 'c' * 36
    assert GoalV2.model_validate_json(json.dumps(goal_body(connection_id=None, leisure_id='run-1'))).sponsor() == 'leisure:run-1'
    for bad in (dict(leisure_id='x'), dict(connection_id=None), dict(goal={'kind': 'fight', 'target': True}),
                dict(goal={'kind': 'fight', 'target': 12}), dict(goal={'kind': 'walk_right'}), dict(budget_seconds=31),
                dict(budget_seconds=2), dict(prediction={'kind': 'survive', 'value': 3}), dict(mask=255),
                dict(goal={'kind': 'enter', 'tile': {'x': 32, 'y': 0}}), dict(leisure_id='a:b', connection_id=None)):
        with pytest.raises(ValidationError): GoalV2.model_validate_json(json.dumps(goal_body(**bad)))
    with pytest.raises(ValidationError): LeisureControl(port=0, action='takeover', leisure_id='x')


def test_human_acquire_preempts_leisure_and_leisure_never_preempts_human():
    a = Arbiter([0])
    grant = a.control('leisure:run', 0, 'acquire', 0, leisure=True)
    assert grant['leisure']
    human = a.control('browser', 0, 'acquire', .1)
    assert human['preempted_leisure'] and a.leases[0]['connection'] == 'browser'
    with pytest.raises(ValueError, match='port_owned'): a.control('leisure:run', 0, 'acquire', .2, leisure=True)
    with pytest.raises(ValueError, match='leisure_cannot'): a.control('leisure:run', 0, 'takeover', .2, leisure=True)
    with pytest.raises(ValueError, match='port_owned'): a.control('other', 0, 'acquire', .2)  # humans still exclusive


def test_leisure_script_grant_stays_preemptible_and_stops_on_human():
    rows = []
    a = Arbiter([0])
    s = Scripted(a, lambda kind, **f: rows.append(dict(kind=kind, **f)), goals_factory=lambda: ZeldaGoals(lambda *x, **y: None))
    g = a.control('leisure:run', 0, 'acquire', 0, leisure=True)
    s.start('leisure:run', 0, 60, 0, 0, g['epoch'], True)
    assert a.leases[0]['leisure'] and s.active[0]['policy_id'] == 'zelda-navigator-v1'
    assert s.active[0]['interval'] == 0
    a.control('browser', 0, 'acquire', .05)
    s.tick(.06)
    assert not s.active and rows[-1]['reason'] == 'human_takeover'


def test_nova_scripted_path_is_unchanged():
    a = Arbiter([0]); s = Scripted(a, lambda *x, **y: None)
    g = a.control('h', 0, 'acquire', 0)
    s.start('h', 0, 60, 0, 0, g['epoch'], True)
    assert s.active[0]['policy_id'] == 'goal-motor-v1' and s.active[0]['interval'] == 1 / 15
    assert not a.leases[0]['leisure']


# ---- worker dispatch -------------------------------------------------------------------
def run_worker(monkeypatch, tmp_path, identity, cognition=True, script=()):
    import queue
    from app import worker
    from settings import Settings
    ram, save = snapshot()
    bufs = {2: C.create_string_buffer(ram, 2048), 0: C.create_string_buffer(save, 8192)}
    clock = SimpleNamespace(now=0., loops=0)
    monkeypatch.setattr(worker, 'time', SimpleNamespace(monotonic=lambda: clock.now,
                        sleep=lambda s: (setattr(clock, 'now', clock.now + max(s, .001)), setattr(clock, 'loops', clock.loops + 1))))

    class Core:
        av = SimpleNamespace(timing=SimpleNamespace(fps=60)); video_frames = audio_samples = 0
        frame_hash = 'f'; buttons = (0, 0)
        lib = SimpleNamespace(retro_get_memory_size=lambda r: len(bufs[r]), retro_get_memory_data=lambda r: C.addressof(bufs[r]))
        def __init__(self, *a): self.identity = identity
        def step(self, *m): pass
        def snapshot(self): return b'state'
        def close(self): pass
    monkeypatch.setattr(worker, 'CoreProbe', Core)
    commands, rows = queue.Queue(), []
    for op, rid, conn, data in script:
        commands.put_nowait(dict(op=op, id=rid, connection=conn, data=data, received_at=0))
    class Stop:
        def is_set(self): return clock.loops >= 40
    class Replies:
        def put_nowait(self, row): rows.append(row)
    (tmp_path / 'saves').mkdir(parents=True)
    worker.run(commands, queue.Queue(), Replies(), Stop(),
               Settings(artifacts=tmp_path / 'ev', cognition_enabled=cognition, saves=tmp_path / 'saves'),
               {'path': tmp_path / 'z.nes', 'ports': [0], 'game_id': 'the-legend-of-zelda'}, 'sess', 'gen')
    return rows, tmp_path


def test_zelda_session_dispatches_reader_motor_atlas_and_lineage(monkeypatch, tmp_path):
    rows, root = run_worker(monkeypatch, tmp_path, IDENTITY, script=[
        ('control', 'c1', 'h', dict(port=0, action='acquire')),
        ('scripted', 's1', 'h', dict(port=0, action='start', cognition=True, duration_seconds=60, expected_epoch=1)),
        ('save', 'sv', '', {})])
    assert next(r for r in rows if r['kind'] == 'ready')['game_profile'] == 'zelda-1-ram-v1'
    status = [r for r in rows if r['kind'] == 'status'][-1]
    assert status['zelda']['observation']['mode'] == 'play' and 'walls' not in status['zelda']['observation']
    lineage = status['zelda']['atlas']['lineage_id']
    assert status['scripted'][0]['policy_id'] == 'zelda-navigator-v1'
    counts = status['zelda']['atlas']['session']['frames_by_controller']
    assert counts['script'] > 0 and set(counts) == {'human', 'script', 'leisure', 'none'}
    evidence = [json.loads(l) for l in (root / 'ev' / 'sess.jsonl').read_text().splitlines()]
    assert {r['controller'] for r in evidence if r['kind'] == 'applied'} <= {'human', 'script', 'leisure'}
    assert any(r['kind'] == 'applied' for r in evidence)
    assert status['scripted'][0]['cognition']['attempt']['decision_id']
    saved = next(r for r in rows if r.get('id') == 'sv')['result']['saved']
    assert saved['lineage_id'] == lineage and load(root / 'saves', lineage)['rooms']


def test_cognition_off_or_other_rom_keeps_the_v1_path(monkeypatch, tmp_path):
    for identity, cognition in ((IDENTITY, False), (dict(rom_sha256='0' * 64, core_sha256=CORE), True)):
        rows, _ = run_worker(monkeypatch, tmp_path / str(cognition), identity, cognition)
        assert next(r for r in rows if r['kind'] == 'ready')['game_profile'] is None
        assert all('zelda' not in r for r in rows if r['kind'] == 'status')


# ---- HTTP surface ----------------------------------------------------------------------
def api(tmp_path, profile, zelda_status=None):
    from fastapi.testclient import TestClient
    from app.main import create_app
    from settings import Settings
    token = tmp_path / 'token'; token.write_text('a' * 48)
    cfg = Settings(enabled=False, cognition_enabled=True, token_file=token, catalog=tmp_path / 'c.json',
                   artifacts=tmp_path / 'ev', core=tmp_path / 'core.so', saves=tmp_path)
    calls = []
    class Manager:
        generation = 'g' * 36; connections = {'c' * 36}
        session = dict(session_id='s' * 36, owner='operator', game_profile=profile, zelda=zelda_status)
        def require(self, sid): pass
        async def command(self, op, connection='', data=None):
            calls.append((op, connection, data)); return {'ok': True}
        async def stop(self, **kw): pass
    app = create_app(cfg)
    return TestClient(app), app, Manager(), calls, {'Authorization': 'Bearer ' + 'a' * 48}


def test_goal_route_uses_goalv2_only_for_zelda_and_strips_sponsor(tmp_path):
    client, app, m, calls, h = api(tmp_path, 'zelda-1-ram-v1')
    with client as c:
        app.state.manager = m
        assert c.post('/sessions/' + 's' * 36 + '/goal', json=goal_body(), headers=h).status_code == 200
        op, conn, data = calls[-1]
        assert (op, conn) == ('goal', 'c' * 36) and 'connection_id' not in data and 'generation' not in data
        body = goal_body(connection_id=None, leisure_id='run-1')
        assert c.post('/sessions/' + 's' * 36 + '/goal', json=body, headers=h).status_code == 200
        assert calls[-1][1] == 'leisure:run-1'
        assert c.post('/sessions/' + 's' * 36 + '/goal', json=goal_body(connection_id='d' * 36), headers=h).json()['detail'] == 'connection_not_live'
        assert c.post('/sessions/' + 's' * 36 + '/goal', json={'goal': 'walk_right'}, headers=h).status_code == 422
        r = c.post('/sessions/' + 's' * 36 + '/control', json={'leisure_id': 'run-1', 'port': 0}, headers=h)
        assert r.status_code == 200 and calls[-1] == ('control', 'leisure:run-1', dict(port=0, action='acquire', leisure=True))


def test_nova_rejects_goalv2_and_leisure(tmp_path):
    client, app, m, calls, h = api(tmp_path, None)
    with client as c:
        app.state.manager = m
        assert c.post('/sessions/' + 's' * 36 + '/goal', json=goal_body(), headers=h).status_code == 422  # v1 Goal only
        assert c.post('/sessions/' + 's' * 36 + '/control', json={'leisure_id': 'run-1', 'port': 0},
                      headers=h).json()['detail'] == 'leisure_requires_zelda'
        assert c.get('/sessions/' + 's' * 36 + '/atlas', headers=h).status_code == 404
    assert not calls


def test_atlas_endpoint_returns_neighbourhood(tmp_path):
    lineage = Atlas.new_lineage()
    a = Atlas(tmp_path, lineage, IDENTITY, 's1', now=lambda: 0); walk(a, [obs(), obs(mode=7, next_room=0x78), obs(room=0x78)]); a.flush(force=True)
    status = dict(observation=observe.summary(obs(room=0x78)), atlas=dict(lineage_id=lineage))
    client, app, m, _, h = api(tmp_path, 'zelda-1-ram-v1', status)
    with client as c:
        app.state.manager = m
        hood = c.get('/sessions/' + 's' * 36 + '/atlas', headers=h).json()
    assert hood['room'] == 0x78 and hood['adjacent']['W']['known'] and set(hood['map']) == {'0:119', '0:120'}


# ---- review regressions (2026-10-22) -----------------------------------------------------
def test_motor_sends_every_native_frame_and_counts_only_applied_inputs():
    # Review: a 1/60 s interval against 60.0988 fps sent every other frame and dropped swings.
    a = Arbiter([0]); g = a.control('h', 0, 'acquire', 0)
    s = Scripted(a, lambda *x, **y: None, goals_factory=lambda: ZeldaGoals(lambda *x, **y: None))
    s.start('h', 0, 60, 0, 0, g['epoch'], True)
    goals = s.active[0]['goals']
    goals.observe_frame(0, 2, 0, obs(), 0)
    goals.accept(dict(port=0, expected_epoch=2, decision_id=goals.attempt['decision_id'], goal={'kind': 'exit', 'dir': 'E'},
                      prediction={'kind': 'unknown', 'value': None}, rationale='t', planner_id='t', budget_seconds=30), 0)
    w = World(); sends = frames = 0
    for f in range(600):
        now = f / 60.0988
        seq = s.active[0]['seq']; s.tick(now); sends += s.active[0]['seq'] > seq; frames += 1
        masks, applied, _ = a.frame(now)
        w.step(masks[0])
        goals.observe_frame(0, 2, f + 1, w.obs(), now)
        for e in applied: s.annotate(e)
        if not goals.accepted: break
    assert sends == frames  # one send per frame while the goal runs
    assert 0 < goals.last_finished['applied_frames'] <= sends


def test_idle_motor_sends_no_rows_and_leisure_without_goals_stops():
    a = Arbiter([0]); g = a.control('leisure:run', 0, 'acquire', 0, leisure=True); rows = []
    s = Scripted(a, lambda kind, **f: rows.append(dict(kind=kind, **f)), goals_factory=lambda: ZeldaGoals(lambda *x, **y: None))
    s.start('leisure:run', 0, 600, 0, 0, g['epoch'], True)
    goals = s.active[0]['goals']
    s.tick(0); a.frame(0); first = s.active[0]['seq']
    for f in range(1, 100): s.tick(f / 60); a.frame(f / 60)
    assert s.active[0]['seq'] == first  # idle and neutral: hold refreshed, no new input rows
    for t in (1, 32, 63, 94):
        goals.observe_frame(0, g['epoch'], t, obs(), t)
        if s.active: s.active[0]['until'] = 1e9; a.leases[0]['expires'] = t + 1
        s.tick(t)
    assert not s.active and rows[-1]['reason'] == 'leisure_sponsor_lost'


def test_reach_room_across_a_level_change_is_refuted_not_met():
    assert judge({'kind': 'reach_room', 'value': 0x77}, obs(room=0x10), obs(room=0x77)) == 'met'
    observe.VERDICTS['level_dungeon'] = 'CONFIRMED'
    try:
        assert judge({'kind': 'reach_room', 'value': 0x77}, obs(), obs(room=0x77, level=1)) == 'refuted'
    finally:
        observe.VERDICTS['level_dungeon'] = 'UNSETTLED'


def test_fight_or_enemies_cleared_refused_on_an_empty_room_and_unknown_cannot_hang():
    g = ZeldaGoals(lambda *x, **y: None)
    g.observe_frame(0, 2, 1, obs(), 0)
    base = dict(port=0, expected_epoch=2, decision_id=g.attempt['decision_id'], goal={'kind': 'fight', 'target': 'nearest'},
                prediction={'kind': 'survive', 'value': None}, rationale='t', planner_id='t', budget_seconds=3)
    with pytest.raises(ValueError, match='no_enemies_present'): g.accept(base, 0)
    with pytest.raises(ValueError, match='no_enemies_present'):
        g.accept(base | {'goal': {'kind': 'wait'}, 'prediction': {'kind': 'enemies_cleared', 'value': None}}, 0)
    with pytest.raises(ValueError, match='stale_decision|unsupported_port'): g.accept(base | {'port': 1}, 0)
    g.accept(base | {'goal': {'kind': 'wait'}}, 0)
    for t in range(1, 5): g.observe_frame(0, 2, t, {'status': 'unknown'}, t)
    assert g.last_finished['outcome'] == 'budget_expired' and g.last_finished['verdict'] == 'unknown'


def test_atlas_never_invents_exits_lineages_or_places(tmp_path):
    a = Atlas(tmp_path, 'not-a-lineage', IDENTITY, 's1', now=lambda: 0)
    assert len(a.lineage_id) == 32  # a damaged save field gets a fresh lineage
    walk(a, [obs(room=0x76), obs(mode=0x11, room=0x76), obs(room=0x77)])  # died, respawned next door
    assert a.doc['rooms']['0:118']['exits_seen']['E'] is None
    a.flush(force=True)
    hood = neighbourhood(a.doc, 'unknown', 5)
    assert hood['level'] == 'unknown' and hood['current'] is None and hood['adjacent'] == {}
    assert neighbourhood(a.doc, 0, None)['adjacent'] == {}
    mismatch = Atlas(tmp_path, a.lineage_id, dict(rom_sha256='1' * 64, core_sha256=CORE), 's2', now=lambda: 0)
    assert mismatch.lineage_id != a.lineage_id  # never overwrites another build's file
    bad = path_for(tmp_path, a.lineage_id); doc = json.loads(bad.read_text())
    doc['rooms']['0:119'].pop('damage_taken'); bad.write_text(json.dumps(doc))
    assert load(tmp_path, a.lineage_id) is None  # a malformed shape is refused, not trusted


def test_death_recovery_selects_continue_by_cursor_and_offers_again():
    from app.zelda.navigate import SELECT, START
    rows = []
    g = ZeldaGoals(lambda kind, **f: rows.append(dict(kind=kind, **f)))
    g.observe_frame(0, 2, 1, obs(room=0x78), 0)
    g.accept(dict(port=0, expected_epoch=2, decision_id=g.attempt['decision_id'], goal={'kind': 'wait'},
                  prediction={'kind': 'unknown', 'value': None}, rationale='t', planner_id='t', budget_seconds=30), 0)
    g.observe_frame(0, 2, 2, obs(room=0x78), .1)
    g.observe_frame(0, 2, 3, obs(room=0x78, mode=0x11), .2)
    end = [r for r in rows if r['kind'] == 'goal_finished'][-1]
    assert end['outcome'] == 'death' and end['verdict'] == 'refuted'  # even an 'unknown' prediction
    assert g.attempt is None and g.mask(0, 0) == 0
    masks = []
    for f, snap in enumerate([dict(mode=8, sub=1), dict(mode=8, sub=1), dict(mode=8, sub=2), dict(mode=8, sub=2),
                              dict(mode=8, sub=0), dict(mode=8, sub=0x80), dict(mode=3)]):
        g.observe_frame(0, 2, 10 + f, obs(room=0x78, **snap), 1 + f / 60)
        masks.append(g.mask(0, f))
    # Cursor read from $13: Save -> SELECT, release, Retry -> SELECT, release, Continue -> START, then hands off.
    assert masks == [SELECT, 0, SELECT, 0, START, 0, 0]
    assert obs(mode=8, sub=0x81)['menu'] == dict(kind='continue', cursor='save', activated=True)
    g.observe_frame(0, 2, 20, obs(room=0x77), 2)
    rec = [r for r in rows if r['kind'] == 'death_recovered']
    assert rec == [dict(kind='death_recovered', decision_id=end['decision_id'], room=0x78, level=0,
                        death_frame=3, room_after=0x77)]
    assert g.attempt and g.mask(0, 0) == 0  # a fresh decision once play returns


def test_death_without_a_goal_recovers_with_null_decision_and_humans_are_untouched():
    rows = []
    g = ZeldaGoals(lambda kind, **f: rows.append(dict(kind=kind, **f)))
    for f, snap in enumerate([dict(), dict(mode=0x11), dict(mode=8, sub=0), dict(mode=8, sub=0x80), dict()]):
        g.observe_frame(0, 2, f, obs(**snap), f)
    assert [r for r in rows if r['kind'] == 'death_recovered'][0]['decision_id'] is None
    # Human play has no script grant, so no motor exists to press anything: the arbiter
    # applies only the human's own mask on the Continue screen.
    a = Arbiter([0]); grant = a.control('human', 0, 'acquire', 0)
    a.submit('human', dict(port=0, lease=grant['lease'], epoch=grant['epoch'], seq=1, mask=0), 0)
    a.frame(0)
    assert a.frame(.01)[0] == [0, 0]


def test_offered_decision_carries_bounded_terrain_but_status_does_not(tmp_path):
    from app.zelda.runtime import ZeldaRuntime
    a = Atlas(tmp_path, Atlas.new_lineage(), IDENTITY, 's1', now=lambda: 0)
    walk(a, [obs(), obs(x=64, y=77), obs(mode=0x10, x=64, y=77)])        # cave at tile (8, 2)
    here = obs(blocked={(c, r) for c in range(32) for r in range(18, 22)} | {(8, 2)})
    a.mark_blocked(0, 0x77, (20, 11), 'play', here['walls_hash'])
    a.mark_blocked(0, 0x77, (21, 11), 'play', 'old-layout')               # other layout: ignored
    g = ZeldaGoals(lambda *x, **y: None, a)
    g.observe_frame(0, 2, 1, here, 0)
    t = g.attempt['observation']['terrain']
    assert len(t) == 22 and all(len(row) == 32 and set(row) <= set('#.xC') for row in t)
    assert t[2][8] == 'C' and t[11][20] == 'x' and t[11][21] == '.' and t[18][0] == '#' and t[0][0] == '.'
    assert len(json.dumps(t)) <= 800 and 'walls' not in g.attempt['observation']
    rt = ZeldaRuntime.__new__(ZeldaRuntime); rt.latest = obs(); rt.atlas = None
    assert 'terrain' not in rt.status()['observation'] and 'walls' not in rt.status()['observation']


def test_applied_rows_and_frames_are_attributed_to_their_controller(tmp_path):
    a = Arbiter([0]); s = Scripted(a, lambda *x, **y: None, goals_factory=lambda: ZeldaGoals(lambda *x, **y: None))
    atlas = Atlas(tmp_path, Atlas.new_lineage(), IDENTITY, 's1', now=lambda: 0)
    assert s.controller(0) == 'none'
    def first(now):  # the first frame after an acquire/takeover is a neutral barrier
        for dt in (0, .001):
            got = a.frame(now + dt)[1]
            if got: return got[0]
    h = a.control('browser', 0, 'acquire', 0)
    a.submit('browser', dict(port=0, lease=h['lease'], epoch=h['epoch'], seq=1, mask=128), 0)
    event = first(0); s.annotate(event)
    assert event['controller'] == 'human' and s.controller(0) == 'human'
    atlas.count_frame(s.controller(0))
    s.start('browser', 0, 60, 0, 0, h['epoch'], True)
    assert s.controller(0) == 'script'
    s.tick(.01); event = first(.01); s.annotate(event)
    assert event['controller'] == 'script'
    atlas.count_frame(s.controller(0)); atlas.count_frame(s.controller(0))
    s.stop(0, 'test')
    lz = a.control('leisure:run', 0, 'acquire', 1, leisure=True)
    assert s.controller(0) == 'leisure'
    s.start('leisure:run', 0, 60, 1, 0, lz['epoch'], True)
    s.active[0]['goals'].recovering = {'last': 0}  # force one non-idle send
    s.active[0]['goals'].current_mask = 8
    s.tick(1.01); event = first(1.01); s.annotate(event)
    assert event['controller'] == 'leisure' and s.controller(0) == 'leisure'
    atlas.count_frame(s.controller(0)); atlas.count_frame('bogus')
    assert atlas.session['frames_by_controller'] == dict(human=1, script=2, leisure=1, none=1)
    assert atlas.public()['session']['frames_by_controller']['script'] == 2


def test_offer_lists_only_sides_link_can_walk_off():
    # Live 2026-10-23: the start room's south side is solid; the planner asked "explore S"
    # eight times. The offer now names the sides an exit can use.
    rows = []
    g = ZeldaGoals(lambda kind, **f: rows.append(dict(kind=kind, **f)))
    g.observe_frame(0, 2, 1, World().obs(), 0)
    assert g.attempt['open_sides'] == ['N', 'W', 'E']
    boxed = {(c, r) for c in range(32) for r in range(22)} - {(15, 11), (16, 11)}
    g = ZeldaGoals(lambda kind, **f: None)
    g.observe_frame(0, 2, 1, World(blocked=boxed).obs(), 0)
    assert g.attempt['open_sides'] == []


def test_open_sides_ignore_the_last_goals_bumps_and_skip_dungeons():
    g = ZeldaGoals(lambda kind, **f: None)
    g.blocked = {(c, r) for c in range(32) for r in range(1, 10)}   # bumps left over from another room
    g.observe_frame(0, 2, 1, World().obs(), 0)
    assert g.attempt['open_sides'] == ['N', 'W', 'E']
    g = ZeldaGoals(lambda kind, **f: None)
    g.observe_frame(0, 2, 1, World(level=1).obs(), 0)                # doorways not verified: no claim
    assert g.attempt['open_sides'] is None


def test_learned_blocked_tile_heals_once_link_stands_on_it(tmp_path):
    a = Atlas(tmp_path, Atlas.new_lineage(), IDENTITY, 's1', now=lambda: 0)
    here = obs()                                                 # Link stands on tile (15, 11)
    assert tile_of(here['link']['x'], here['link']['y']) == (15, 11)
    away = obs(x=40)                                             # same room, tile (5, 11)
    a.observe(away, 0)
    h = here['walls_hash']; learned = lambda: a.doc['rooms']['0:119']['learned_blocked']
    for tile, wh in (((15, 11), h), ((16, 11), h), ((15, 11), 'old-layout'), ((20, 11), h), ((15, 10), h)):
        a.mark_blocked(0, 0x77, tile, 'play', wh)
    a.dirty = False
    a.observe(away, 1)                                           # not standing on any: all kept
    assert len(learned()) == 5 and a.dirty is False
    frozen = dict(here, link=dict(here['link'], controllable=False))
    a.observe(frozen, 2)                                         # not controllable: no change
    assert len(learned()) == 5 and a.dirty is False
    a.observe(here, 3)
    # Only Link's own position heals: (16, 11) is the position east of him (a bump he just made).
    assert a.blocked(0, 0x77, 'play', h) == {(16, 11), (20, 11), (15, 10)}
    assert [15, 11, 'old-layout'] in learned()                    # another layout's bump is kept
    assert a.dirty is True
    a.dirty = False
    a.observe(here, 4)                                           # nothing left to remove
    assert a.dirty is False and len(learned()) == 4


def test_learned_blocked_tile_heals_in_the_cave_key_only(tmp_path):
    a = Atlas(tmp_path, Atlas.new_lineage(), IDENTITY, 's1', now=lambda: 0)
    cave = obs(mode=0x0B, ad=2)
    assert cave['mode'] == 'cave' and cave['link']['controllable']
    a.observe(obs(), 0); a.observe(dict(cave, link=dict(cave['link'], controllable=False)), 1)
    a.mark_blocked(0, 0x77, (15, 11), 'cave', cave['walls_hash'])
    a.mark_blocked(0, 0x77, (15, 11), 'play', cave['walls_hash'])
    a.observe(cave, 2)
    assert a.blocked(0, 0x77, 'cave', cave['walls_hash']) == set()
    assert a.blocked(0, 0x77, 'play', cave['walls_hash']) == {(15, 11)}   # overworld entry untouched
