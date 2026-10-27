"""Action bets, Hub side (spec 2026-10-23 embodied experience): options, settle table, menu, place.

Deterministic: the prediction service is a fake ``http`` callable. No network, no GPU.
"""
import asyncio
import copy
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import TypeAdapter

from orion.schemas.prediction import ActionBetPlaceV1, ActionBetQuoteRequestV1, ActionBetSettleV1
from scripts import retro_experience as X
from scripts import retro_planner as P

HUB = Path(__file__).resolve().parents[2]
WALL = json.loads((HUB / 'evals/fixtures/zelda_room119_south_wall.json').read_text())
SID = 'c6041e2d-ea4f-452a-9f61-5514df7e985a'


def live_decision(**over):
    """Decision 15:9 as Retro offered it live (room 119, open N, W, E) and its atlas."""
    d = copy.deepcopy(WALL['decision'])
    d.update(over)
    return d, copy.deepcopy(WALL['atlas'])


def watched_decision():
    """The room 119 decision with an enemy next to Link, so waiting is offered."""
    d, atlas = live_decision()
    link = d['observation']['link']
    d['observation']['enemies'] = [dict(slot=2, x=link['x'] + 16, y=link['y'])]
    return d, atlas


# Real goal_finished rows (evidence c6041e2d, 2026-10-23) and Retro's _finish shape for the rest.
def finished(outcome, goal=None, prediction=None, verdict='unknown', decision_id='15:2'):
    return dict(decision_id=decision_id, goal=goal or {'kind': 'explore', 'dir': 'S'},
                prediction=prediction or {'kind': 'reach_room', 'value': 120},
                planner_id='llm-gateway:metacog_background:e1d4', outcome=outcome, verdict=verdict,
                start_digest='5dab0dba7e51081d', end_digest='ea8443189cdd1bb1', start_frame=7166, end_frame=7263,
                applied_frames=96, human_intervention=outcome == 'stopped:human_takeover', bumps=0)


def obs(**over):
    o = copy.deepcopy(WALL['decision']['observation'])
    o.pop('terrain', None)
    o.update(over)
    return o


START = X.start_facts(obs())   # room 119, 3/3 hearts, rupees 3, sword 1


class FakePrediction:
    """The orion-prediction HTTP API as built (services/orion-prediction/app/main.py)."""

    def __init__(self, fail=None, p=0.5, eig=0.193, level='prior', counts=None):
        self.calls, self.fail, self.p, self.eig, self.level = [], fail, p, eig, level
        self.counts = counts or [{'level': 'skill', 'successes': 0, 'failures': 0}]
        self.settle_status = 200

    async def __call__(self, method, path, json=None, params=None):
        self.calls.append((method, path, json, params))
        if self.fail == 'raise':
            raise ConnectionRefusedError('prediction down')
        if self.fail == '503':
            return 503, {'detail': 'action bet store unavailable'}
        if method == 'GET':
            return 200, ({'session_id': (params or {}).get('session_id'), 'bets': []} if path == '/action-bets'
                         else {'skills': [], 'situations': [], 'places': [], 'grading': {}})
        if path == '/action-bets/quote':
            ActionBetQuoteRequestV1.model_validate(json)
            p = self.p if callable(self.p) else (lambda o: self.p)
            return 200, {'domain': json['domain'], 'body': json['body'], 'served_model': 'action_beta_backoff',
                         'quotes': [dict(option_id=o['option_id'], model='action_beta_backoff', p=p(o),
                                         p_baseline=0.5, interval80=[0.1, 0.9], eig_nats=self.eig, level=self.level,
                                         counts=self.counts) for o in json['options']]}
        if path == '/action-bets':
            ActionBetPlaceV1.model_validate(json)
            return 200, dict(json, model='action_beta_backoff', p=0.5, p_baseline=0.5, level='prior', counts=[],
                             placed_at='2026-10-23T16:17:11+00:00', detail={})
        if path.endswith('/settle'):
            ActionBetSettleV1.model_validate(json)
            return self.settle_status, {}
        return 404, {'detail': 'not_found'}

    def paths(self, method='POST'):
        return [p for m, p, _, _ in self.calls if m == method]


def experience(fake=None, enabled=True):
    return X.ZeldaExperience(enabled=enabled, http=fake or FakePrediction())


# --- options ------------------------------------------------------------------------------------

def test_options_for_the_live_room_119_decision():
    d, atlas = live_decision()
    opts = X.build_options(d, atlas)
    # One option per side: exit and explore toward one side are the same bet.
    # No wait: the room is empty, so waiting cannot change anything.
    assert [o.option_id for o in opts] == ['exit:N', 'exit:E', 'exit:W']
    watched = X.build_options(*watched_decision())
    assert [o.option_id for o in watched] == ['exit:N', 'exit:E', 'exit:W', 'fight:nearest', 'retreat', 'wait']
    by = {o.option_id: o for o in opts + watched[-1:]}   # the calm room's exits, the watched room's wait
    assert by['exit:N'].context == {'health': 'full', 'side': 'open_visited', 'threat': 'none'}
    assert by['exit:W'].context == {'health': 'full', 'side': 'open_unvisited', 'threat': 'none'}
    assert by['exit:W'].place == 'zelda:0:119:W' and by['exit:W'].skill == 'traverse'
    assert by['exit:N'].value == 0.3 and by['exit:E'].value == 1.0     # visited vs new room
    assert by['wait'].skill == 'hold' and by['wait'].place is None and by['wait'].value == 0.05
    goal = TypeAdapter(P.GoalSpec)
    for o in opts:   # every option is a goal Retro's GoalV2 accepts
        assert goal.validate_python(o.goal).model_dump(mode='json') == o.goal


def test_open_sides_unknown_offers_all_four_and_off_grid_south_reads_closed():
    d, atlas = live_decision(open_sides=None)
    by = {o.option_id: o for o in X.build_options(d, atlas)}
    assert {k for k in by if k.startswith('explore')} == {'explore:N', 'explore:S', 'explore:E', 'explore:W'}
    assert by['explore:S'].context['side'] == 'closed' and by['explore:W'].context['side'] == 'unknown'
    # A stale atlas (another room) says nothing about these sides.
    stale = dict(atlas, room=103)
    assert X.side_state(d, stale, 'S') == 'unknown'


def test_enemies_drops_and_caves_build_fight_retreat_collect_and_enter():
    o = obs(enemies=[dict(slot=2, x=130, y=200, tile=[16, 18], hp=1), dict(slot=3, x=10, y=70)],
            drops=[dict(slot=4, x=200, y=150)], hearts={'current': 1.5, 'containers': 3})
    d, atlas = live_decision(observation=o)
    atlas['caves'] = [dict(room='0:119', tile=[8, 2], entered=True)]
    by = {x.option_id: x for x in X.build_options(d, atlas)}
    assert by['fight:nearest'].context == {'health': 'hurt', 'threat': 'many'} and by['fight:nearest'].value == 0.6
    assert by['retreat'].context == {'health': 'hurt', 'threat': 'near'} and by['retreat'].value == 0.4
    assert by['collect:25,12'].goal == {'kind': 'collect', 'tile': {'x': 25, 'y': 12}}   # Retro's hotspot rule
    assert by['enter:8,2'].goal == {'kind': 'enter', 'tile': {'x': 8, 'y': 2}} and by['enter:8,2'].context == {'threat': 'near'}
    assert by['exit:W'].context['threat'] == 'near'
    far = obs(enemies=[dict(slot=2, x=10, y=70)])   # Manhattan 102+119 px: no threat, no retreat
    assert 'retreat' not in {x.option_id for x in X.build_options(live_decision(observation=far)[0], atlas)}


def test_dungeon_traverse_has_no_place_belief():
    d, atlas = live_decision(observation=obs(level=3, room=45))
    assert all(o.place is None for o in X.build_options(d, None))


# --- settle table --------------------------------------------------------------------------------

def settle(skill, row, nxt=None, start=START):
    body = X.settle_body(skill, start, row, nxt)
    return None if body is None else (body.result, body.detail['rule'])


ITEM_UP = obs(items=dict(obs()['items'], rupees=8))
HURT = obs(hearts={'current': 2.5, 'containers': 3})
ROOM_118 = obs(room=118)

SETTLE_TABLE = [
    # traverse: room changed / death / interrupted / same room
    ('traverse', finished('prediction_met', verdict='met'), obs(room=119), X.start_facts(obs(room=103)), ('success', 'room_changed')),
    ('traverse', finished('completed', verdict='refuted'), ROOM_118, START, ('success', 'room_changed')),
    ('traverse', finished('completed', verdict='refuted'), obs(), START, ('failure', 'same_room')),
    ('traverse', finished('budget_expired', verdict='refuted'), obs(), START, ('failure', 'same_room')),
    ('traverse', finished('interrupted:stuck'), obs(), START, ('failure', 'interrupted')),
    ('traverse', finished('interrupted:hit'), None, START, ('failure', 'interrupted')),
    ('traverse', finished('death', verdict='refuted'), None, START, ('failure', 'death')),
    # engage
    ('engage', finished('completed', goal={'kind': 'fight', 'target': 'nearest'}), obs(), START, ('success', 'completed')),
    ('engage', finished('prediction_met', goal={'kind': 'fight', 'target': 'nearest'},
                        prediction={'kind': 'enemies_cleared'}, verdict='met'), obs(), START, ('success', 'completed')),
    ('engage', finished('budget_expired', goal={'kind': 'fight', 'target': 2}), obs(), START, ('failure', 'budget_expired')),
    ('engage', finished('interrupted:stuck', goal={'kind': 'fight', 'target': 2}), obs(), START, ('failure', 'interrupted:stuck')),
    ('engage', finished('death', goal={'kind': 'fight', 'target': 2}), None, START, ('failure', 'death')),
    # withdraw
    ('withdraw', finished('completed', goal={'kind': 'retreat'}), obs(), START, ('success', 'completed')),
    ('withdraw', finished('interrupted:stuck', goal={'kind': 'retreat'}), obs(), START, ('failure', 'interrupted')),
    ('withdraw', finished('death', goal={'kind': 'retreat'}), None, START, ('failure', 'death')),
    # acquire
    ('acquire', finished('completed', goal={'kind': 'collect', 'tile': {'x': 25, 'y': 10}}), ITEM_UP, START, ('success', 'item_count_up')),
    ('acquire', finished('prediction_met', goal={'kind': 'collect', 'tile': {'x': 25, 'y': 10}},
                         prediction={'kind': 'item_gained'}, verdict='met'), None, START, ('success', 'item_gained_met')),
    ('acquire', finished('completed', goal={'kind': 'collect', 'tile': {'x': 25, 'y': 10}}), obs(), START, ('failure', 'no_item')),
    ('acquire', finished('budget_expired', goal={'kind': 'collect', 'tile': {'x': 25, 'y': 10}}), obs(), START, ('failure', 'no_item')),
    # enter
    ('enter', finished('completed', goal={'kind': 'enter', 'tile': {'x': 8, 'y': 2}}), obs(mode='cave'), START, ('success', 'completed')),
    ('enter', finished('prediction_met', goal={'kind': 'enter', 'tile': {'x': 8, 'y': 2}},
                       prediction={'kind': 'enter_cave'}, verdict='met'), obs(mode='cave'), START, ('success', 'completed')),
    ('enter', finished('budget_expired', goal={'kind': 'enter', 'tile': {'x': 8, 'y': 2}}), obs(), START, ('failure', 'not_entered')),
    ('enter', finished('interrupted:hit', goal={'kind': 'enter', 'tile': {'x': 8, 'y': 2}}), obs(), START, ('failure', 'not_entered')),
    # hold
    ('hold', finished('budget_expired', goal={'kind': 'wait'}, prediction={'kind': 'survive'}, verdict='met'), obs(), START, ('success', 'no_heart_lost')),
    ('hold', finished('budget_expired', goal={'kind': 'wait'}, prediction={'kind': 'survive'}, verdict='met'), HURT, START, ('failure', 'heart_lost')),
    ('hold', finished('death', goal={'kind': 'wait'}), None, START, ('failure', 'death')),
]


@pytest.mark.parametrize('skill,row,nxt,start,want', SETTLE_TABLE,
                         ids=[f'{s}-{r["outcome"]}-{w[1]}' for s, r, _, _, w in SETTLE_TABLE])
def test_settle_table_row(skill, row, nxt, start, want):
    assert settle(skill, row, nxt, start) == want


@pytest.mark.parametrize('skill', ['traverse', 'engage', 'withdraw', 'acquire', 'enter', 'hold'])
@pytest.mark.parametrize('reason', ['human_takeover', 'pause', 'sponsor_disconnected'])
def test_stopped_is_void_never_a_failure(skill, reason):
    body = X.settle_body(skill, START, finished(f'stopped:{reason}'), HURT)
    assert body.result == 'void' and body.outcome == f'stopped:{reason}'


def test_cannot_tell_without_the_next_observation():
    assert X.settle_body('traverse', START, finished('completed'), None) is None
    assert X.settle_body('traverse', START, finished('completed'), {'status': 'unknown'}) is None
    assert X.settle_body('hold', START, finished('budget_expired', goal={'kind': 'wait'}), None) is None
    assert X.settle_body('traverse', START, None, obs()) is None


# --- scoring and menu text ------------------------------------------------------------------------

def test_menu_line_matches_the_spec_style():
    d, atlas = live_decision()
    w = {o.option_id: o for o in X.build_options(dict(d, open_sides=None), atlas)}['explore:W']
    q = X.ActionBetQuoteV1(option_id='explore:W', model='action_beta_backoff', p=0.71, p_baseline=0.6,
                           interval80=(0.5, 0.9), eig_nats=0.05, level='place',
                           counts=[dict(level='skill', successes=9, failures=5),
                                   dict(level='situation', successes=4, failures=1),
                                   dict(level='place', successes=2, failures=0)])
    assert X.menu_line(1, w, q) == ('1. explore W -> room 118 (never visited). Chance it works 0.71 '
                                    '(2 of 2 here; moving overall 9 of 14). Learning value medium. Predicts reach_room 118, 15 s.')
    prior = X.ActionBetQuoteV1.model_validate(q.model_dump() | dict(level='prior', eig_nats=0.193,
                                            counts=[dict(level='skill', successes=0, failures=0)]))
    assert X.counts_phrase('traverse', prior) == 'no tries yet' and X.learning_word(0.193) == 'high'
    assert X.learning_word(0.02) == 'low'


def test_scoring_is_p_times_value_plus_two_eig_and_keeps_the_top_five():
    d, atlas = live_decision()
    opts = X.build_options(d, atlas)
    ps = {'exit:N': 0.9, 'exit:W': 0.4, 'exit:E': 0.1}
    fake = FakePrediction(p=lambda o: ps[o['option_id']], eig=0.0, level='skill')
    quotes = asyncio.run(experience(fake).quote(opts))
    menu = X.build_menu(opts, quotes)
    # 0.4 * 1.0 (new room) beats 0.9 * 0.3 (visited).
    assert [e.option.option_id for e in menu.entries] == ['exit:W', 'exit:N', 'exit:E']
    assert [e.rank for e in menu.entries] == [1, 2, 3]
    assert menu.entries[0].score == pytest.approx(0.4)
    # value x (p + 2 EIG): learning is worth only as much as the action.
    assert X.score(opts[0], quotes['exit:N'].model_copy(update={'eig_nats': 0.1})) == pytest.approx(0.3 * (0.9 + 0.2))
    assert menu.rank_of({'kind': 'exit', 'dir': 'N'}) == 2 and menu.rank_of({'kind': 'explore', 'dir': 'N'}) is None


def test_an_untried_wait_does_not_outrank_a_likely_exit_on_curiosity():
    # Review 2026-10-23: under p*value + 2*EIG an untried wait (p 0.5, EIG 0.193) scored 0.41 and
    # beat exits that had started failing (0.29). Learning value is now scaled by the action's value.
    d, atlas = watched_decision()
    opts = {o.option_id: o for o in X.build_options(d, atlas)}
    untried = X.ActionBetQuoteV1(option_id='wait', model='m', p=0.5, p_baseline=0.5, interval80=(0.2, 0.8),
                                 eig_nats=0.193, level='prior', counts=[dict(level='skill', successes=0, failures=0)])
    failing = untried.model_copy(update={'option_id': 'exit:W', 'p': 0.29, 'eig_nats': 0.02, 'level': 'place'})
    assert X.score(opts['exit:W'], failing) > X.score(opts['wait'], untried)


# --- quote failure, flag off -----------------------------------------------------------------------

@pytest.mark.parametrize('fail', ['raise', '503'])
def test_quote_failure_means_no_menu_no_bet_and_a_log_line(fail, caplog):
    fake = FakePrediction(fail=fail)
    exp = experience(fake)
    d, atlas = live_decision()
    with caplog.at_level(logging.WARNING, logger='orion-hub.retro.experience'):
        menu = asyncio.run(exp.prepare(SID, d, atlas))
    assert menu is None and 'action_quote_unavailable' in caplog.text
    goal = dict(goal={'kind': 'exit', 'dir': 'W'}, prediction={'kind': 'reach_room', 'value': 118}, decision_id='15:9')
    assert asyncio.run(exp.placed(SID, d, atlas, goal, menu)) is None
    assert '/action-bets' not in fake.paths() and exp.open == {}


def test_a_quote_missing_an_option_is_no_menu(caplog):
    class Short(FakePrediction):
        async def __call__(self, method, path, json=None, params=None):
            status, data = await super().__call__(method, path, json=json, params=params)
            if path == '/action-bets/quote':
                data['quotes'] = data['quotes'][:-1]
            return status, data
    d, atlas = live_decision()
    with caplog.at_level(logging.WARNING, logger='orion-hub.retro.experience'):
        assert asyncio.run(experience(Short()).prepare(SID, d, atlas)) is None
    assert 'missing_quotes' in caplog.text


def test_flag_off_touches_nothing_and_the_prompt_is_the_2870_prompt():
    fake = FakePrediction()
    exp = experience(fake, enabled=False)
    d, atlas = live_decision()
    assert asyncio.run(exp.prepare(SID, d, atlas)) is None
    asyncio.run(exp.end_session(SID, [finished('completed')], None))
    assert asyncio.run(exp.learned_lines(SID)) == [] and fake.calls == []
    assert P.build_zelda_prompt(d, atlas, []) == P.build_zelda_prompt(d, atlas, [], None)
    assert P.MENU_INSTRUCTION not in P.build_zelda_prompt(d, atlas, [])
    # A config without the key is off (Settings ships it on).
    assert X.ZeldaExperience.from_config(SimpleNamespace()).enabled is False
    assert X.ZeldaExperience.from_config(SimpleNamespace(HUB_ACTION_BETS_ENABLED=True)).enabled is True


# --- the planner with a menu --------------------------------------------------------------------------

@pytest.fixture
def planner(monkeypatch):
    # Imported here, not at module top: conftest re-imports ``scripts`` per test, and the route
    # under test must see the same module object this fixture patches.
    from scripts import retro_planner
    from orion.gpu_pool.shed_backoff import ShedBackoff
    monkeypatch.setattr(retro_planner, 'SHED_BACKOFF', ShedBackoff(name='test', clock=lambda: 1000.0, rng=lambda: 0.5,
                                                                   log=lambda m: None))
    monkeypatch.setattr(retro_planner, '_in_flight', False)
    return retro_planner


def _menu():
    d, atlas = live_decision()
    opts = X.build_options(d, atlas)
    quotes = asyncio.run(experience(FakePrediction(eig=0.0, p=0.5)).quote(opts))
    return d, atlas, X.build_menu(opts, quotes)


def test_menu_goes_after_the_sides_line_and_the_choice_carries_its_rank(planner):
    d, atlas, menu = _menu()
    prompt = P.build_zelda_prompt(d, atlas, [], menu.for_planner())
    lines = prompt.split('\n')
    sides = next(i for i, l in enumerate(lines) if l.startswith('Sides Link can walk off'))
    assert lines[sides + 2].startswith('1. exit E -> room 120 (never visited). Chance it works 0.50')
    assert lines[sides + 2 + len(menu.for_planner())] == P.MENU_INSTRUCTION
    calls = []
    async def chat(config, prompt, *, max_tokens):
        calls.append(prompt)
        return {'content': json.dumps(dict(goal={'kind': 'exit', 'dir': 'W'}, prediction={'kind': 'reach_room', 'value': 118},
                                           rationale='west is new', budget_seconds=12))}, 'c1'
    goal = asyncio.run(planner.plan_zelda(None, d, atlas, [], chat=chat, menu=menu.for_planner()))
    assert goal['chosen_rank'] == menu.rank_of({'kind': 'exit', 'dir': 'W'}) is not None
    assert 'chosen_rank' not in P.goal_v2_body(goal) and P.goal_v2_body(goal)['goal'] == {'kind': 'exit', 'dir': 'W'}
    async def off_menu(config, prompt, *, max_tokens):
        return {'content': json.dumps(dict(goal={'kind': 'wait'}, prediction={'kind': 'survive'},
                                           rationale='hold', budget_seconds=5))}, 'c2'
    small = menu.for_planner()[:2]
    assert asyncio.run(planner.plan_zelda(None, d, atlas, [], chat=off_menu, menu=small))['chosen_rank'] is None
    # Without a menu the result has exactly the #2870 keys.
    assert 'chosen_rank' not in asyncio.run(planner.plan_zelda(None, d, atlas, [], chat=off_menu))


# --- place and settle ----------------------------------------------------------------------------------

GOAL_W = dict(goal={'kind': 'explore', 'dir': 'W'}, prediction={'kind': 'reach_room', 'value': 118},
              rationale='west is new', budget_seconds=12, decision_id='15:9', planner_id='llm:c1', chosen_rank=2)


def test_place_then_settle_at_the_next_plan():
    fake = FakePrediction()
    exp = experience(fake)
    d, atlas = live_decision()
    menu = asyncio.run(exp.prepare(SID, d, atlas))
    assert menu is not None and fake.paths() == ['/action-bets/quote']
    asyncio.run(exp.placed(SID, d, atlas, GOAL_W, menu))
    placed = [j for m, p, j, _ in fake.calls if p == '/action-bets'][0]
    assert placed['bet_id'] == f'retro:zelda:{SID}:15:9' and placed['chosen_rank'] == 2
    assert placed['skill'] == 'traverse' and placed['place'] == 'zelda:0:119:W'
    assert placed['context'] == {'health': 'full', 'side': 'open_unvisited', 'threat': 'none'}
    assert 'p' not in placed   # the service computes p
    # Next decision: Link walked into room 118.
    nxt = dict(decision_id='15:10', observation=obs(room=118), supported_goals=d['supported_goals'], open_sides=['E'],
               previous=finished('prediction_met', goal=GOAL_W['goal'], verdict='met', decision_id='15:9'))
    asyncio.run(exp.prepare(SID, nxt, None))
    settled = [(p, j) for m, p, j, _ in fake.calls if p.endswith('/settle')]
    assert settled == [(f'/action-bets/retro:zelda:{SID}:15:9/settle',
                        dict(result='success', outcome='prediction_met', detail=settled[0][1]['detail']))]
    assert settled[0][1]['detail']['rule'] == 'room_changed' and exp.open == {}


def test_off_menu_goal_is_placed_with_no_rank():
    fake = FakePrediction()
    exp = experience(fake)
    d, atlas = live_decision()
    menu = asyncio.run(exp.prepare(SID, d, atlas))
    goal = dict(GOAL_W, goal={'kind': 'explore', 'dir': 'S'}, chosen_rank=None)
    asyncio.run(exp.placed(SID, d, atlas, goal, menu))
    placed = [j for m, p, j, _ in fake.calls if p == '/action-bets'][0]
    assert placed['chosen_rank'] is None and placed['context']['side'] == 'closed' and placed['place'] == 'zelda:0:119:S'


def test_a_failed_settle_is_retried_at_the_next_plan_and_409_counts_as_settled():
    fake = FakePrediction()
    exp = experience(fake)
    d, atlas = live_decision()
    menu = asyncio.run(exp.prepare(SID, d, atlas))
    asyncio.run(exp.placed(SID, d, atlas, GOAL_W, menu))
    fake.settle_status = 503
    nxt = dict(decision_id='15:10', observation=obs(), supported_goals=['wait'],
               previous=finished('completed', goal=GOAL_W['goal'], decision_id='15:9'))
    asyncio.run(exp.prepare(SID, nxt, None))
    assert len(exp.open) == 1
    fake.settle_status = 409   # already settled (e.g. expired never_settled): stop retrying
    asyncio.run(exp.prepare(SID, dict(nxt, decision_id='15:11'), None))
    assert exp.open == {} and len([p for p in fake.paths() if p.endswith('/settle')]) == 2


def test_session_end_settles_from_the_last_finished_row_else_void_session_ended():
    fake = FakePrediction()
    exp = experience(fake)
    d, atlas = live_decision()
    menu = asyncio.run(exp.prepare(SID, d, atlas))
    asyncio.run(exp.placed(SID, d, atlas, GOAL_W, menu))
    d2 = dict(d, decision_id='15:10')
    asyncio.run(exp.placed(SID, d2, atlas, dict(GOAL_W, decision_id='15:10'), menu))
    rows = [finished('interrupted:stuck', goal=GOAL_W['goal'], decision_id='15:9')]
    asyncio.run(exp.end_session(SID, rows, None))
    bodies = {p.split('/')[2]: j for m, p, j, _ in fake.calls if p.endswith('/settle')}
    assert bodies[f'retro:zelda:{SID}:15:9']['result'] == 'failure'
    assert bodies[f'retro:zelda:{SID}:15:10'] == dict(result='void', outcome='session_ended',
                                                      detail={'decision_id': '15:10', 'rule': 'no_finished_row'})
    assert exp.open == {}


# --- co-play route: place only after Retro accepts ---------------------------------------------------

def _route_app(tmp_path, monkeypatch, goal_status, posted, exp):
    import importlib.util
    from fastapi import FastAPI
    spec = importlib.util.spec_from_file_location('plan_routes_bets', HUB / 'scripts/retro_routes.py')
    routes = importlib.util.module_from_spec(spec); spec.loader.exec_module(routes)
    (tmp_path / 'hub.token').write_text('x' * 48)
    cfg = SimpleNamespace(HUB_RETRO_ENABLED=True, HUB_RETRO_COGNITION_ENABLED=True,
                          HUB_RETRO_ORIGIN='http://testserver', HUB_RETRO_AUTH_DIR=str(tmp_path), HUB_RETRO_SOCKET='/unused')
    d, atlas = live_decision(expected_epoch=2)
    state = dict(generation='g' * 36, scripted=[dict(port=0, epoch=2, policy_id='zelda-navigator-v1',
                 cognition=dict(policy_id='zelda-navigator-v1', attempt=d, goal=None, last_finished=d['previous']))])
    class Response:
        def __init__(self, data, status=200): self.data = data; self.status = status
        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass
        async def json(self): return self.data
    class Client:
        def __init__(self, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass
        def get(self, url, **kw): return Response(atlas if url.endswith('/atlas') else state)
        def request(self, method, url, **kw): return self.get(url)
        def post(self, url, **kw):
            posted.append((url, kw['json'])); return Response(kw['json'] if goal_status == 200 else {'detail': 'stale_decision'}, goal_status)
    monkeypatch.setattr(routes.aiohttp, 'ClientSession', Client)
    class Store:
        async def mark_human(self): pass
    app = FastAPI(); app.include_router(routes.create_router(cfg, presence=Store(), experience=exp))
    return app


@pytest.mark.parametrize('goal_status', [200, 409])
def test_route_places_only_after_retro_accepts_and_strips_chosen_rank(tmp_path, monkeypatch, planner, goal_status):
    from fastapi.testclient import TestClient
    async def chat(config, prompt, *, max_tokens, route=None):
        assert P.MENU_INSTRUCTION in prompt
        return {'content': json.dumps(dict(goal={'kind': 'exit', 'dir': 'W'}, prediction={'kind': 'reach_room', 'value': 118},
                                           rationale='west is new', budget_seconds=12))}, 'c9'
    monkeypatch.setattr(planner, '_gateway_chat', chat)
    fake, posted = FakePrediction(eig=0.0), []
    exp = experience(fake)
    body = dict(connection_id='c' * 36, generation='g' * 36, port=0, expected_epoch=2)
    with TestClient(_route_app(tmp_path, monkeypatch, goal_status, posted, exp)) as c:
        r = c.post('/api/retro/sessions/' + 'a' * 36 + '/plan', json=body, headers={'Origin': 'http://testserver'})
    assert r.status_code == goal_status, r.text
    url, sent = posted[0]
    assert url.endswith('/goal') and 'chosen_rank' not in sent and sent['goal'] == {'kind': 'exit', 'dir': 'W'}
    places = [j for m, p, j, _ in fake.calls if m == 'POST' and p == '/action-bets']
    if goal_status == 200:
        assert len(places) == 1 and places[0]['bet_id'] == 'retro:zelda:' + 'a' * 36 + ':15:9'
        assert places[0]['chosen_rank'] is not None
    else:
        assert places == [] and exp.open == {}


# --- the replay eval runs offline -----------------------------------------------------------------------

def test_offline_replay_eval_passes(capsys):
    # The eval grades orion-prediction's real model (numpy/scipy); jobs without them skip, the
    # orion-prediction job and a local venv run it.
    pytest.importorskip('scipy')
    import importlib.util
    spec = importlib.util.spec_from_file_location('zelda_experience_replay', HUB / 'evals/zelda_experience_replay.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    args = SimpleNamespace(fixture=str(HUB / 'evals/fixtures/zelda_session_c6041e2d_goals.json'),
                           wall_fixture=str(HUB / 'evals/fixtures/zelda_room119_south_wall.json'))
    assert mod.main(args) == 0, capsys.readouterr().out


# --- leisure: plan with a menu, place after accept, settle, session end ---------------------------------

def test_leisure_places_after_accept_settles_and_closes_out_at_session_end():
    import importlib.util
    spec = importlib.util.spec_from_file_location('leisure_harness', HUB / 'tests/retro/test_leisure.py')
    H = importlib.util.module_from_spec(spec); spec.loader.exec_module(H)
    from scripts import retro_leisure as L
    from scripts.retro_presence import PresenceStore
    clock, redis = H.Clock(), H.FakeRedis()
    retro = H.FakeRetro(clock)
    real_call = retro.call
    async def call(method, path, body=None):   # Retro's real goal_finished rows carry an outcome
        status, data = await real_call(method, path, body)
        if method == 'POST' and path.endswith('/goal') and status == 200:
            retro.session['scripted'][0]['cognition']['last_finished']['outcome'] = 'completed'
            retro.session['scripted'][0]['cognition']['attempt']['previous'] = \
                retro.session['scripted'][0]['cognition']['last_finished']
        return status, data
    retro.call = call
    menus = []
    async def plan(decision, atlas, history, menu=None):
        menus.append(menu)
        return dict(H.GOAL, decision_id=decision['decision_id'])
    async def regulation(): return ('idle', 'idle', None)
    fake = FakePrediction()
    deps = L.LeisureDeps(store=PresenceStore(redis, clock=clock), retro=retro, regulation=regulation, plan=plan,
                         clock=clock, sleep=clock.sleep, experience=experience(fake))
    L._running_here = False
    out = asyncio.run(L.run_leisure_session(H.config(RETRO_LEISURE_SESSION_MAX_SEC=60), deps=deps))
    assert out.started and retro.goals and all(m for m in menus)
    assert all('chosen_rank' not in g for g in retro.goals)          # Retro's GoalV2 forbids extra keys
    places = [j for m, p, j, _ in fake.calls if m == 'POST' and p == '/action-bets']
    settles = [(p, j['result'], j['outcome']) for m, p, j, _ in fake.calls if p.endswith('/settle')]
    assert len(places) == len(retro.goals) and places[0]['chosen_rank'] == 1   # exit N is the only option
    assert {p for p, _, _ in settles} == {f'/action-bets/{b["bet_id"]}/settle' for b in places}
    # Room never changes in the fake: every settled exit is a same-room failure, none left open.
    assert {(r, o) for _, r, o in settles} <= {('failure', 'completed'), ('void', 'session_ended')}
    assert deps.experience.open == {}
    assert [p for m, p, _, _ in fake.calls if m == 'GET'] == ['/action-bets', '/action-bets/live']


def test_leisure_without_experience_plans_exactly_as_before():
    from scripts import retro_leisure as L
    seen = []
    async def plan(decision, atlas, history):   # the #2870 signature: no menu keyword at all
        seen.append(decision['decision_id']); return {}
    deps = SimpleNamespace(experience=None, plan=plan)
    run = L._Run(leisure_id='l', act_id=None, game_id='g', port=0, started_at=0.0, session_id='s')
    asyncio.run(L._plan(deps, run, {'decision_id': '1:1'}, None))
    assert seen == ['1:1'] and run.plan_inputs == ({'decision_id': '1:1'}, None, None)


# --- learned lines on the memory card ---------------------------------------------------------------------

def _bet(place, result, skill='traverse', side=None):
    return dict(bet_id=f'b-{place}-{result}', skill=skill, place=place, result=result,
                context={'side': side} if side else {})


# Counts arrive as floats from /action-bets/live (each place weighs as one unit above place level).
LIVE = {'skills': [{'skill': 'traverse', 'successes': 6.0, 'failures': 8.0, 'mean': 0.44}],
        'situations': [{'domain': 'retro:zelda', 'skill': 'traverse', 'successes': 4, 'failures': 1,
                        'context': {'side': 'open_unvisited', 'threat': 'none', 'health': 'full'}},
                       {'domain': 'retro:zelda', 'skill': 'traverse', 'successes': 1, 'failures': 0,
                        'context': {'side': 'open_unvisited', 'threat': 'near', 'health': 'full'}},
                       {'domain': 'retro:zelda', 'skill': 'traverse', 'successes': 0, 'failures': 2,
                        'context': {'side': 'closed', 'threat': 'none', 'health': 'full'}}],
        'grading': {'served': {'n': 12, 'brier': 0.18, 'brier_base_rate': 0.2412}, 'baseline': {'n': 12, 'brier': 0.2}},
        'open': 0, 'never_settled_24h': 0}


def test_learned_lines_are_facts_from_the_bets():
    from scripts.retro_memory_export import learned_lines_from
    bets = {'session_id': SID, 'bets': [_bet('zelda:0:119:S', 'failure', side='closed')] * 3
            + [_bet(f'zelda:0:1{i}:W', 'success', side='open_unvisited') for i in range(5)]
            + [_bet('zelda:0:120:E', 'failure', side='open_unvisited'), _bet('zelda:0:119:N', 'void')]}
    # Line 2 counts THIS session's tries (raw), never the service's place-weighted situation counts;
    # line 3's "average" is grade()'s base-rate Brier, not the skill-rate baseline model.
    assert learned_lines_from(bets, LIVE) == [
        "Room 119's south side has not led anywhere (0 of 3 tries).",
        'Moving toward rooms I had not visited worked 5 of 6 times.',
        'My guesses about my own moves scored 0.18. Always guessing my average would score 0.24 (lower is better).']


def test_no_calibration_line_under_ten_settled_bets_and_odd_shapes_drop_lines():
    from scripts.retro_memory_export import learned_lines_from
    few = dict(LIVE, grading={'served': {'n': 9, 'brier': 0.1, 'brier_base_rate': 0.2}}, situations=[])
    assert learned_lines_from([_bet(None, 'success', skill='engage')] * 3, few) == ['Fighting worked 3 of 3 times.']
    assert learned_lines_from(None, None) == [] and learned_lines_from({'bets': 'x'}, {'grading': 3}) == []


def test_unreachable_prediction_service_gives_no_lines_and_a_log_line(caplog):
    from scripts.retro_memory_export import fetch_learned_lines
    async def down(path, params=None): raise ConnectionRefusedError('down')
    with caplog.at_level(logging.WARNING, logger='orion-hub.retro_memory_export'):
        assert asyncio.run(fetch_learned_lines(SID, get=down)) == []
    assert 'retro_memory_export_learned_unavailable' in caplog.text
    async def ok(path, params=None):
        return (200, {'bets': [_bet('zelda:0:119:S', 'failure')] * 3}) if path == '/action-bets' else (200, LIVE)
    assert len(asyncio.run(fetch_learned_lines(SID, get=ok))) == 3


def test_card_carries_the_learned_lines_and_is_written_without_them(monkeypatch):
    from scripts import retro_memory_export as rx
    from scripts.retro_session_end import RetroSessionEnded
    ended = RetroSessionEnded(session_id=SID, lineage_id='lin', leisure_id='l1', game_id='the-legend-of-zelda',
                              started_at=1792699320.0, ended_at=1792700640.0, end_reason='session_cap',
                              learned_lines=("Room 119's south side is not a way out (0 of 3 tries).",))
    s = rx.build_session_summary(**rx.session_end_inputs(ended))
    card = rx.memory_card_for(s, ended.learned_lines)
    assert card.summary == s.summary + " From my own moves: Room 119's south side is not a way out (0 of 3 tries)."
    assert card.subschema['retro_play']['learned_lines'] == list(ended.learned_lines)
    assert rx.memory_card_for(s).summary == s.summary
    seen = {}
    async def fake_export(**kw):
        seen.update(kw); return rx.ExportResult('exported', kw['session_id'])
    monkeypatch.setattr(rx, 'export_session', fake_export)
    asyncio.run(rx.export_session_end(pool=None, bus=None, ended=ended))
    assert seen['learned'] == list(ended.learned_lines)


def test_explore_and_exit_toward_one_side_share_a_menu_rank():
    from scripts.retro_planner import menu_rank
    menu = [{'rank': 1, 'goal': {'kind': 'exit', 'dir': 'W'}, 'line': 'x'}]
    assert menu_rank(menu, {'kind': 'explore', 'dir': 'W'}) == 1 and menu_rank(menu, {'kind': 'exit', 'dir': 'E'}) is None


def test_a_cave_never_bets_on_the_room_above_and_hearts_count_as_a_pickup():
    d, atlas = live_decision()
    cave = dict(d, observation=dict(d['observation'], mode='cave'), open_sides=None)
    o = X.option_for_goal({'kind': 'explore', 'dir': 'S'}, cave, atlas)
    assert o.place is None and o.context['side'] == 'unknown'
    start = X.start_facts(dict(d['observation'], hearts={'current': 2, 'containers': 3}))
    after = dict(d['observation'], hearts={'current': 3, 'containers': 3})
    assert X._gained(start, after)


def test_menu_counts_below_one_unit_are_not_shown():
    d, atlas = live_decision()
    o = {x.option_id: x for x in X.build_options(d, atlas)}['exit:E']
    q = X.ActionBetQuoteV1(option_id='exit:E', model='m', p=0.74, p_baseline=0.7, interval80=(0.5, 0.9), eig_nats=0.05,
                           level='situation', counts=[dict(level='skill', successes=7.5, failures=2.5),
                                                      dict(level='situation', successes=0.5, failures=0.0),
                                                      dict(level='place', successes=0, failures=0)])
    assert X.counts_phrase('traverse', q) == 'moving overall 7.5 of 10'   # live 10-23: "0.5 of 0.5 in this situation"


def test_gateway_registers_the_coplay_exporter():
    # Live 2026-10-23: the gateway relays every handoff (watch_coplay runs there) but only Hub's
    # main.py registered the exporter, so 7 co-play handoffs wrote 0 session rows.
    from scripts import retro_gateway as G
    app = SimpleNamespace(state=SimpleNamespace())
    gate = asyncio.Event()
    async def create_pool(**kw):
        await gate.wait()          # a slow database
        return 'pool'
    async def run():
        task = G.register_session_end_sink(app, SimpleNamespace(RECALL_PG_DSN='postgresql://x'), create_pool=create_pool)
        # The sink exists before the pool opens: the relay never waits for the database.
        assert callable(app.state.retro_session_end_sink) and app.state.memory_pg_pool is None
        gate.set(); await task
        assert app.state.memory_pg_pool == 'pool'
    asyncio.run(run())
    bare = SimpleNamespace(state=SimpleNamespace())
    assert G.register_session_end_sink(bare, SimpleNamespace(RECALL_PG_DSN='')) is None
    assert not hasattr(bare.state, 'retro_session_end_sink') and bare.state.memory_pg_pool is None


def test_gateway_opens_a_bus_for_the_coplay_play_act():
    # Regression (2026-10-23): the gateway passed bus_provider=lambda: None, so co-play acts could
    # never be published from the process that runs co-play.
    from scripts import retro_gateway as G
    app = SimpleNamespace(state=SimpleNamespace())
    gate = asyncio.Event()
    class Bus:
        async def connect(self): await gate.wait()   # a slow Redis
    async def run():
        task = G.register_session_start_sink(app, SimpleNamespace(ORION_BUS_URL='redis://x:6379/0',
                                                                  ORION_BUS_ENABLED=True),
                                             bus_factory=lambda url: Bus())
        # The opener exists before the bus connects: the relay never waits for Redis.
        assert callable(app.state.retro_session_start_sink) and app.state.retro_act_bus is None
        gate.set(); await task
        assert isinstance(app.state.retro_act_bus, Bus)
    asyncio.run(run())
    off = SimpleNamespace(state=SimpleNamespace())
    assert G.register_session_start_sink(off, SimpleNamespace(ORION_BUS_URL='', ORION_BUS_ENABLED=True)) is None
    assert off.state.retro_act_bus is None


def test_gateway_act_bus_retries_and_closes_a_failed_client(monkeypatch):
    # Review finding: one failed connect at boot left the gateway without play acts for life, and
    # the half-built client leaked.
    from scripts import retro_gateway as G
    monkeypatch.setattr(G, 'BUS_RETRY_FIRST_SEC', 0.0)
    app = SimpleNamespace(state=SimpleNamespace())
    made = []
    class Bus:
        def __init__(self): self.closed = False; made.append(self)
        async def connect(self):
            if len(made) < 3: raise ConnectionError('redis down')
        async def close(self): self.closed = True
    async def run():
        await G.register_session_start_sink(app, SimpleNamespace(ORION_BUS_URL='redis://x:6379/0',
                                                                 ORION_BUS_ENABLED=True),
                                            bus_factory=lambda url: Bus())
    asyncio.run(run())
    assert len(made) == 3 and app.state.retro_act_bus is made[2]
    assert [b.closed for b in made] == [True, True, False]


def test_each_handoff_card_counts_only_its_own_bets():
    from scripts.retro_memory_export import bets_since, learned_lines_from
    old = [dict(_bet(f'zelda:0:1{i}:W', 'success', side='open_unvisited'), placed_at='2026-10-23T19:00:00+00:00')
           for i in range(5)]
    new = [dict(_bet(f'zelda:0:2{i}:E', 'failure', side='open_unvisited'), placed_at='2026-10-23T19:40:00+00:00')
           for i in range(3)]
    from datetime import datetime
    since = datetime.fromisoformat('2026-10-23T19:39:00+00:00').timestamp()
    assert [b['place'] for b in bets_since({'bets': old + new}, since)] == [b['place'] for b in new]
    assert 'worked 0 of 3 times.' in learned_lines_from(bets_since(old + new, since), {})[0]
    assert bets_since(old, None) == old


def test_menu_entries_carry_the_plan_a_number_pick_uses():
    d, atlas = watched_decision()
    opts = {o.option_id: o for o in X.build_options(d, atlas)}
    assert opts['exit:W'].default_plan == ({'kind': 'reach_room', 'value': 118}, 15)   # the room on that side
    assert opts['wait'].default_plan == ({'kind': 'survive'}, 5)
    q = X.ActionBetQuoteV1(option_id='x', model='m', p=0.5, p_baseline=0.5, interval80=(0.1, 0.9), eig_nats=0.1,
                           level='prior', counts=[dict(level='skill', successes=0, failures=0)])
    menu = X.build_menu(list(opts.values()), {o: q.model_copy(update={'option_id': o}) for o in opts})
    entry = next(m for m in menu.for_planner(limit=None) if m['goal'] == {'kind': 'exit', 'dir': 'W'})
    assert entry['prediction'] == {'kind': 'reach_room', 'value': 118} and entry['budget_seconds'] == 15


def test_a_fight_with_several_enemies_predicts_survival_not_a_clear_room():
    # Retro ends fight-nearest when that enemy dies; "enemies_cleared" with 2+ would be refuted.
    d, atlas = live_decision()
    many = dict(d, observation=dict(d['observation'], enemies=[dict(slot=1, x=40, y=100), dict(slot=2, x=200, y=150)]))
    one = dict(d, observation=dict(d['observation'], enemies=[dict(slot=1, x=40, y=100)]))
    fight = {'kind': 'fight', 'target': 'nearest'}
    assert X.option_for_goal(fight, many, atlas).default_plan == ({'kind': 'survive'}, 20)
    assert X.option_for_goal(fight, one, atlas).default_plan == ({'kind': 'enemies_cleared'}, 20)


def test_wait_is_offered_only_when_something_on_screen_can_change():
    # Live 2026-10-27 (session 6c359e66): empty room 124, sides unknown. "wait" was picked 41 of 41
    # times, met every time, and its chance climbed to 37 of 37 while Link never left the room.
    d, atlas = live_decision()
    assert 'wait' not in [o.option_id for o in X.build_options(d, atlas)]
    assert 'wait' in [o.option_id for o in X.build_options(*watched_decision())]
    far = dict(d, observation=dict(d['observation'], enemies=[dict(slot=2, x=10, y=70)]))
    assert 'wait' not in [o.option_id for o in X.build_options(far, atlas)]    # a far enemy keeps it met forever
    frozen = dict(d, observation=dict(d['observation'], link=dict(d['observation']['link'], controllable=False)))
    assert 'wait' in [o.option_id for o in X.build_options(frozen, atlas)]     # the only goal Retro accepts then
    boxed = dict(d, open_sides=[], supported_goals=['exit', 'wait'])   # nothing else to offer: wait stays
    assert [o.option_id for o in X.build_options(boxed, atlas)] == ['wait']


def test_a_room_crossed_many_times_is_worth_less_than_a_rarely_seen_one():
    # Live 2026-10-27 (session bae052c0): 115 <-> 99 thirty times. Room 99's sides: S to 115 (41
    # visits, met 28 of 28), W to 98 (4 visits, never tried here).
    d, atlas = live_decision()
    atlas = dict(atlas, adjacent=dict(atlas['adjacent'], W=dict(room=118, known=True, seen_exit=None)),
                 map=dict(atlas['map'], **{'0:103': dict(visits=41), '0:118': dict(visits=4)}))
    opts = {o.option_id: o for o in X.build_options(d, atlas)}
    assert opts['exit:N'].value == pytest.approx(0.3 / 41) and opts['exit:W'].value == pytest.approx(0.3 / 4)
    assert opts['exit:E'].value == 1.0                                    # never seen: full worth
    q = lambda p, eig: X.ActionBetQuoteV1(option_id='x', model='m', p=p, p_baseline=p, interval80=(0.1, 0.9),
                                          eig_nats=eig, level='place', counts=[dict(level='place', successes=28, failures=0)])
    assert X.score(opts['exit:W'], q(0.25, 0.02)) > X.score(opts['exit:N'], q(0.91, 0.0))
    assert 'exit N -> room 103 (visited 41 times)' in opts['exit:N'].text   # the model sees why it ranks low
    assert X.room_value(None, 0, 103) == 0.3 and X.room_value(atlas, 0, None) == 0.3   # no counts: base worth


def test_the_planner_sees_only_the_two_best_options():
    d, atlas = live_decision()
    opts = X.build_options(d, atlas)
    ps = {'exit:N': 0.9, 'exit:W': 0.4, 'exit:E': 0.1}
    menu = X.build_menu(opts, asyncio.run(experience(FakePrediction(p=lambda o: ps[o['option_id']], eig=0.0,
                                                                    level='skill')).quote(opts)))
    assert [m['rank'] for m in menu.for_planner()] == [1, 2] and len(menu.entries) == 3
    assert len(menu.for_planner(limit=None)) == 3
    assert menu.rank_of({'kind': 'exit', 'dir': 'E'}) == 3        # an off-shortlist pick still gets its rank


def test_the_shortlist_always_keeps_a_way_out_next_to_a_fight():
    # Review: near enemy, every side heavily visited -> fight and retreat outscore every exit.
    d, atlas = watched_decision()
    atlas = dict(atlas, map={k: dict(visits=50) for k in ('0:103', '0:118', '0:120')},
                 adjacent={k: dict(v, known=True) for k, v in atlas['adjacent'].items()})
    opts = X.build_options(d, atlas)
    ps = {o.option_id: 0.9 for o in opts}
    menu = X.build_menu(opts, asyncio.run(experience(FakePrediction(p=lambda o: ps[o['option_id']], eig=0.0,
                                                                    level='skill')).quote(opts)))
    top = [e.option.skill for e in menu.entries[:2]]
    assert 'traverse' not in top                                     # a plain top 2 would hide every exit
    shown = menu.for_planner()
    assert len(shown) == 2 and {menu.entries[r['rank'] - 1].option.skill == 'traverse' for r in shown} == {True, False}
    assert [r['rank'] for r in shown] == sorted(r['rank'] for r in shown)


def test_a_pick_outside_the_shortlist_is_placed_with_its_full_menu_rank():
    d, atlas = live_decision()
    opts = X.build_options(d, atlas)
    ps = {'exit:N': 0.9, 'exit:W': 0.4, 'exit:E': 0.1}
    fake = FakePrediction(p=lambda o: ps[o['option_id']], eig=0.0, level='skill')
    exp = experience(fake)
    menu = X.build_menu(opts, asyncio.run(exp.quote(opts)))
    third = menu.entries[2].option.goal
    asyncio.run(exp.placed('s1', d, atlas, dict(goal=third, prediction={'kind': 'unknown'}, decision_id='15:9',
                                                chosen_rank=None), menu))
    placed = [c for c in fake.calls if c[0] == 'POST' and c[1] == '/action-bets']
    assert placed and placed[-1][2]['chosen_rank'] == 3
