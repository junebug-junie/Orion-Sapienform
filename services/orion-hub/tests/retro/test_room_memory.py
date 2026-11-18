"""Dungeon room memory (retro_planner.RoomMemory): the 2026-11-06 two-room bounce.

Live 2026-11-06, solo session 3bf6973a: inside a dungeon Retro reports level 'unknown' (the dungeon
number in RAM is not settled), so its atlas has no neighbours for the room. Every side then looked
never visited, both sides scored the same, the menu broke the tie by N-S-E-W order, and Link walked
36 -> N -> 20 -> S -> 36 for 337 decisions (0 new rooms, every prediction unknown).

The recorded decisions come from evals/fixtures/zelda_dungeon_bounce_3bf6973a.json (read-only export
of Retro's evidence file). Quotes are equal for every option, as they were live (p 0.959 for both
sides): the order then comes from value alone. No network, no GPU.
"""
import asyncio
import copy
import json
from pathlib import Path

from orion.schemas.prediction import ActionBetQuoteV1
from scripts import retro_experience as X
from scripts import retro_planner as P

HUB = Path(__file__).resolve().parents[2]
FIXTURE = json.loads((HUB / 'evals/fixtures/zelda_dungeon_bounce_3bf6973a.json').read_text())
SID = FIXTURE['session_id']
OFFERS = {r['decision_id']: r for r in FIXTURE['rows'] if r['kind'] == 'decision_offered'}


def retro_atlas(room):
    return dict(copy.deepcopy(FIXTURE['retro_atlas']), room=room)


def offer(decision_id):
    return copy.deepcopy(OFFERS[decision_id])


def equal_quotes(options):
    return {o.option_id: ActionBetQuoteV1(option_id=o.option_id, model='m', p=0.959, p_baseline=0.9,
                                          interval80=(0.9, 0.99), eig_nats=0.01, level='situation',
                                          counts=[dict(level='skill', successes=20, failures=1),
                                                  dict(level='situation', successes=20, failures=1)])
            for o in options}


def first_choice(decision, atlas):
    options = X.build_options(decision, atlas)
    menu = X.build_menu(options, equal_quotes(options))
    return menu.for_planner()[0]


def test_recorded_decisions_bounce_without_memory():
    """The bug, from the recorded inputs: room 36 again (2:3) still ranks N (back to 20) first."""
    for decision_id, room, side in (('2:1', 36, 'N'), ('2:2', 20, 'S'), ('2:3', 36, 'N')):
        d = offer(decision_id)
        assert d['observation']['level'] == 'unknown' and d['observation']['room'] == room
        top = first_choice(d, retro_atlas(room))
        assert top['goal'] == {'kind': 'exit', 'dir': side}
        assert top['prediction'] == {'kind': 'unknown'}       # no neighbour room: nothing to check
    assert 'first visit (not in the atlas yet)' in P.atlas_neighbourhood(retro_atlas(36))


def test_recorded_decisions_leave_by_the_new_side_with_memory():
    memory = P.RoomMemory()
    picks = {}
    for decision_id in ('2:1', '2:2', '2:3'):
        d = offer(decision_id)
        atlas = memory.atlas_for(SID, d, retro_atlas(d['observation']['room']))
        picks[decision_id] = (first_choice(d, atlas), atlas)
    assert picks['2:2'][0]['goal'] == {'kind': 'exit', 'dir': 'S'}       # room 20: S is the only way
    top, atlas = picks['2:3']
    assert top['goal'] == {'kind': 'exit', 'dir': 'S'}                    # room 52, never visited
    assert top['prediction'] == {'kind': 'reach_room', 'value': 52}       # checkable now
    assert atlas['adjacent']['N'] == {'room': 20, 'known': True, 'seen_exit': 20}
    assert atlas['current']['visits'] == 2 and atlas['map']['unknown:20']['visits'] == 1
    text = P.atlas_neighbourhood(atlas)
    assert 'this session' in text and 'visits 2' in text and 'N unknown:20: known, reached from here before' in text
    assert 'S unknown:52: never visited' in text
    line = next(m['line'] for m in X.build_menu(X.build_options(offer('2:3'), atlas),
                                                 equal_quotes(X.build_options(offer('2:3'), atlas))).for_planner()
                if m['goal'] == {'kind': 'exit', 'dir': 'N'})
    assert 'room 20 (visited 1 time)' in line


def test_known_level_keeps_retros_atlas_and_clears_the_dungeon():
    memory = P.RoomMemory()
    memory.atlas_for(SID, offer('2:1'), retro_atlas(36))
    overworld = offer('2:2')
    overworld['decision_id'] = '2:9'
    overworld['observation'].update(level=0, room=119)
    atlas = dict(lineage_id='l', level=0, room=119, adjacent={'N': {'room': 103, 'known': True, 'seen_exit': 103}})
    assert memory.atlas_for(SID, overworld, atlas) is atlas            # Retro knows the level: untouched
    assert memory.visits == {} and memory.last_room is None             # left the dungeon: forget it
    # Settled 'level_dungeon' (Retro sends an int level) also keeps Retro's own atlas.
    d = offer('2:3'); d['observation']['level'] = 1
    retro = dict(lineage_id='l', level=1, room=36, adjacent={})
    assert memory.atlas_for(SID, d, retro) is retro


def test_a_new_session_starts_empty_and_each_decision_counts_once():
    memory = P.RoomMemory()
    for _ in range(3):
        memory.observe(SID, offer('2:1'))
    assert memory.visits == {36: 1}
    memory.observe('other-session', offer('2:2'))
    assert memory.visits == {20: 1} and memory.exits == {}


def test_a_side_is_crossed_only_by_the_goal_that_walked_it():
    """Co-play: a human may walk rooms between two decisions; grid adjacency alone proves no door."""
    memory = P.RoomMemory()
    memory.observe(SID, offer('2:1'))
    walked = offer('2:2')
    walked['previous'] = dict(walked['previous'], goal={'kind': 'fight', 'target': 'nearest'})
    memory.observe(SID, walked)
    assert memory.visits == {36: 1, 20: 1} and memory.exits == {}
    atlas = memory.fill(walked, retro_atlas(20))
    assert atlas['adjacent']['S'] == {'room': 36, 'known': True, 'seen_exit': None}


def test_a_stale_or_interrupted_previous_goal_proves_no_side():
    """Retro's ``previous`` stays the same until another goal finishes. Co-play: Orion exits N from
    36 to 20 (2:1 -> 2:2), then a human walks 20 -> E 21 -> N 5 -> W 4 with no goal finishing. The
    next offer, in room 4 (north of 20 on the grid), still carries 2:1's "exit N": no door proven."""
    memory = P.RoomMemory()
    memory.observe(SID, offer('2:1'))
    memory.observe(SID, offer('2:2'))
    assert memory.exits == {36: {'N': 20}, 20: {'S': 36}}
    stale = offer('2:3')
    stale['decision_id'] = '2:7'
    stale['observation']['room'] = 4
    stale['previous'] = offer('2:2')['previous']          # 2:1's exit N, offered back in room 36
    memory.observe(SID, stale)
    assert memory.visits[4] == 1 and 4 not in memory.exits and 'N' not in memory.exits[20]
    for change in ({'human_intervention': True}, {'outcome': 'interrupted:stuck'}):
        m = P.RoomMemory()
        m.observe(SID, offer('2:1'))
        d = offer('2:2'); d['previous'] = dict(d['previous'], **change)
        m.observe(SID, d)
        assert m.exits == {}, change


def test_non_play_and_unknown_observations_are_ignored():
    memory = P.RoomMemory()
    d = offer('2:1'); d['observation']['mode'] = 'stairs'
    assert memory.atlas_for(SID, d, retro_atlas(36)) == retro_atlas(36) and memory.visits == {}
    d = offer('2:2'); d['observation'] = {'status': 'unknown'}
    assert memory.atlas_for(SID, d, None) is None


def _dungeon(rooms, doors):
    """A synthetic dungeon on Retro's grid: ``doors`` = {(room, dir)} that are open."""
    template = offer('2:1')
    def at(room, n, came):
        d = copy.deepcopy(template)
        d['decision_id'] = f'2:{n}'
        d['observation']['room'] = d['observation']['next_room'] = room
        d['open_sides'] = [s for s in 'NSEW' if (room, s) in doors]
        d['previous'] = None if came is None else dict(template['previous'] or {}, decision_id=f'2:{n - 1}',
                                                        goal={'kind': 'exit', 'dir': came})
        return d
    return at


def walk(memory, steps=24):
    # A 2 x 3 dungeon: rooms 20 21 22 over 36 37 38, every inner door open.
    rooms = {20, 21, 22, 36, 37, 38}
    doors = {(r, d) for r in rooms for d in 'NSEW' if P._neighbour(r, d) in rooms}
    at = _dungeon(rooms, doors)
    room, came, seen, path = 36, None, set(), []
    for n in range(1, steps + 1):
        d = at(room, n, came)
        atlas = memory.atlas_for(SID, d, retro_atlas(room)) if memory else retro_atlas(room)
        side = first_choice(d, atlas)['goal']['dir']
        seen.add(room); path.append(room)
        room, came = P._neighbour(room, side), side
    return seen, path


def test_closed_loop_explores_the_whole_dungeon_instead_of_two_rooms():
    seen, path = walk(None)
    assert seen == {20, 36} and path[:4] == [36, 20, 36, 20]           # the live bounce
    seen, path = walk(P.RoomMemory())
    assert seen == {20, 21, 22, 36, 37, 38}
    assert len(set(path[:8])) >= 5                                      # not a two-room ping-pong


def test_leisure_plans_with_the_dungeon_memory():
    from scripts import retro_leisure as L
    seen = []
    async def plan(decision, atlas, history, **kw):
        seen.append(atlas)
        return {}
    deps = L.LeisureDeps(store=None, retro=None, regulation=None, plan=plan, experience=None)
    run = L._Run(leisure_id='l', act_id=None, game_id='g', port=0, started_at=0.0, session_id=SID)
    for decision_id in ('2:1', '2:2', '2:3'):
        d = offer(decision_id)
        asyncio.run(L._plan(deps, run, d, retro_atlas(d['observation']['room'])))
    assert seen[-1]['session_memory'] is True and seen[-1]['adjacent']['N']['known'] is True


def test_dungeon_bounce_replay_eval_passes(capsys):
    # Grades with orion-prediction's real model (numpy/scipy); jobs without them skip.
    import importlib.util
    import types
    import pytest
    pytest.importorskip('scipy')
    spec = importlib.util.spec_from_file_location('zelda_dungeon_bounce_replay', HUB / 'evals/zelda_dungeon_bounce_replay.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    args = types.SimpleNamespace(fixture=str(HUB / 'evals/fixtures/zelda_dungeon_bounce_3bf6973a.json'))
    assert mod.main(args) == 0, capsys.readouterr().out


def test_coplay_route_plans_with_the_dungeon_memory(tmp_path, monkeypatch):
    import importlib.util
    from fastapi.testclient import TestClient
    spec = importlib.util.spec_from_file_location('planner_v2_harness', HUB / 'tests/retro/test_planner_v2.py')
    H = importlib.util.module_from_spec(spec); spec.loader.exec_module(H)
    from scripts import retro_planner
    monkeypatch.setattr(retro_planner, '_in_flight', False)
    seen = []
    async def plan(config, decision, atlas, history, **kw):
        seen.append(atlas)
        raise retro_planner.PlannerRejected('planner_invalid_goal')
    monkeypatch.setattr(retro_planner, 'plan_zelda', plan)
    state = H._state(attempt=offer('2:1'))
    app = H._app(tmp_path, monkeypatch, state, [], atlas=retro_atlas(36))
    with TestClient(app) as c:
        for decision_id in ('2:1', '2:2', '2:3'):
            state['scripted'][0]['cognition']['attempt'] = offer(decision_id)
            c.post(H.PATH, json=H.BODY, headers=H.H)
    assert seen[0]['session_memory'] is True and seen[0]['current']['visits'] == 1
    # One session's memory carries across plan calls: back in 36, N (room 20) is known.
    assert seen[2]['current']['visits'] == 2 and seen[2]['adjacent']['N']['known'] is True
