// Driver for tests/test_sim_core.py. Runs every scenario against the shipped
// core.ts + seed.ts (copied next to this file) and prints one JSON report.
import { MAX_EFFECTS, MAX_FACTS, MAX_OPEN_QUESTS, MAX_RESOLVED_QUESTS, MAX_RUMORS, SimError, applyEvent, checkEntities,
  checkInjectSource, createState, tick } from './core.ts';
import type { State } from './core.ts';
import { SIM_CAST, SIM_SEED } from './seed.ts';

const out: Record<string, unknown> = {};
const j = (v: unknown) => JSON.stringify(v);
// What Convex does to a saved document: JSON round trip with sorted keys.
function convexRoundTrip<T>(v: T): T {
  const sortDeep = (x: any): any => Array.isArray(x) ? x.map(sortDeep)
    : x && typeof x === 'object' ? Object.fromEntries(Object.keys(x).sort().map((k) => [k, sortDeep(x[k])])) : x;
  return JSON.parse(JSON.stringify(sortDeep(v)));
}
function err(fn: () => unknown): string | null {
  try { fn(); return null; } catch (e) { return (e instanceof SimError ? 'SimError: ' : 'Error: ') + (e as Error).message; }
}
const numbers = (s: State) => j({ stock: s.stock, trust: s.trust,
  needs: Object.fromEntries(Object.keys(s.characters).map((k) => [k, s.characters[k].needs])) });
const s0 = createState(SIM_SEED, SIM_CAST);
const juniper = (rest: object) => ({ source: 'juniper', kind: 'note', title: 'Juniper event', ...rest });

// A1: a seeded 24 game-hour run moves stock, needs and trust; replay is exact.
{
  const a = tick(s0, 24 * 60);
  const b = tick(createState(SIM_SEED, SIM_CAST), 24 * 60);
  let c = convexRoundTrip(s0);
  const cEvents: unknown[] = [];
  for (let i = 0; i < 288; i++) {
    const r = tick(c, 5);
    cEvents.push(...r.events);
    c = convexRoundTrip(r.state);
  }
  const other = tick(createState(SIM_SEED + 1, SIM_CAST), 24 * 60);
  const changed = (pick: (s: State) => Record<string, unknown>) =>
    Object.keys(pick(a.state)).filter((k) => j(pick(a.state)[k]) !== j(pick(s0)[k]));
  const noVersion = (s: State) => j(convexRoundTrip({ ...s, version: 0 }));
  out.a1 = {
    minute: a.state.minute,
    events: a.events.length,
    kinds: [...new Set(a.events.map((e) => e.kind))].sort(),
    stock_changed: changed((s) => s.stock),
    trust_changed: changed((s) => s.trust),
    needs_changed: changed((s) => Object.fromEntries(Object.keys(s.characters).map((k) => [k, s.characters[k].needs]))),
    replay_identical: j(a) === j(b),
    stepped_identical: noVersion(a.state) === noVersion(c) && j(a.events) === j(cEvents),
    other_seed_differs: j(other.state) !== j(a.state),
    input_untouched: j(s0) === j(createState(SIM_SEED, SIM_CAST)),
    sources: [...new Set(a.events.map((e) => e.source))],
  };
}

// Clamping: needs 0..1, trust -1..1, stock >= 0.
{
  const r = applyEvent(s0, juniper({ effects: [
    { op: 'need_delta', target: 'mara_vale.rest', value: 1 },
    { op: 'need_delta', target: 'nico_sable.social', value: -1 },
    { op: 'trust_delta', target: 'cam_lin|orion', value: 2 },
    { op: 'trust_delta', target: 'sofia_bell|nico_sable', value: -2 },
    { op: 'stock_delta', target: 'diner.coffee', value: -1000 },
  ] }));
  out.clamp = {
    rest: r.state.characters.mara_vale.needs!.rest,
    social: r.state.characters.nico_sable.needs!.social,
    trust_up: r.state.trust['cam_lin|orion'],
    trust_down: r.state.trust['sofia_bell|nico_sable'],
    coffee: r.state.stock['diner.coffee'],
  };
}

// Rejections: nothing applies, the input state is untouched.
{
  const before = j(s0);
  out.rejects = {
    unknown_op: err(() => applyEvent(s0, juniper({ effects: [{ op: 'teleport', target: 'diner.coffee', value: 1 }] }))),
    half_valid: err(() => applyEvent(s0, juniper({ effects: [
      { op: 'stock_delta', target: 'diner.coffee', value: 5 }, { op: 'need_delta', target: 'ghost.rest', value: 1 }] }))),
    money_need: err(() => applyEvent(s0, juniper({ effects: [{ op: 'need_delta', target: 'cam_lin.money', value: 0.1 }] }))),
    visitor_need: err(() => applyEvent(s0, juniper({ effects: [{ op: 'need_delta', target: 'orion.rest', value: 0.1 }] }))),
    unknown_item: err(() => applyEvent(s0, juniper({ effects: [{ op: 'stock_delta', target: 'diner.unicorns', value: 1 }] }))),
    unknown_place: err(() => applyEvent(s0, juniper({ place: 'bakery' }))),
    fractional_stock: err(() => applyEvent(s0, juniper({ effects: [{ op: 'stock_delta', target: 'diner.coffee', value: 0.5 }] }))),
    object_value: err(() => applyEvent(s0, juniper({ effects: [{ op: 'fact_set', target: 'x.y', value: { a: 1 } }] }))),
    new_key_wrong_kind: err(() => applyEvent(s0, juniper({ effects: [{ op: 'stock_delta', target: 'diner.cash', value: 1 }] }))),
    resident_place_item: err(() => applyEvent(s0, juniper({ effects: [{ op: 'stock_delta', target: 'cam_lin.parts', value: 1 }] }))),
    place_item_ok: applyEvent(s0, juniper({ effects: [{ op: 'stock_delta', target: 'workshop.coffee', value: 2 }] })).state.stock['workshop.coffee'],
    reserved_fact: err(() => applyEvent(s0, juniper({ effects: [{ op: 'fact_set', target: 'quest.e1', value: 'succeeded' }] }))),
    proto_trust: err(() => applyEvent(s0, juniper({ effects: [{ op: 'trust_delta', target: 'constructor', value: 1 }] }))),
    proto_need: err(() => applyEvent(s0, juniper({ effects: [{ op: 'need_delta', target: '__proto__.rest', value: 1 }] }))),
    effects_not_list: err(() => applyEvent(s0, juniper({ effects: 'lots' }))),
    untouched: before === j(s0),
  };
}

// Free text with no effects becomes a rumor fact and moves no numbers.
{
  const r = applyEvent(s0, { source: 'juniper', kind: 'rumor', title: 'Someone saw lights over the square.' });
  let many = s0;
  for (let i = 0; i < 25; i++) many = applyEvent(many, { source: 'juniper', kind: 'rumor', title: `rumor ${i}` }).state;
  const rumorKeys = Object.keys(many.facts).filter((k) => k.startsWith('rumor.'));
  out.free_text = {
    status: r.event.status,
    fact: r.state.facts[`rumor.${r.event.id}`],
    numbers_same: numbers(r.state) === numbers(s0),
    rng_same: r.state.rng === s0.rng,
    rumors_kept: rumorKeys.length,
    oldest_kept: rumorKeys.map((k) => Number(k.slice(7))).sort((x, y) => x - y)[0],
  };
}

// Sources: sim and juniper only in PR 2.
{
  const ok = applyEvent(s0, juniper({ effects: [{ op: 'fact_set', target: 'diner.open_late', value: true }] }));
  out.sources = {
    juniper_status: ok.event.status,
    juniper_source: ok.event.source,
    juniper_fact: ok.state.facts['diner.open_late'],
    orion: err(() => applyEvent(s0, { source: 'orion', kind: 'note', title: 'x' })),
    mirror: err(() => applyEvent(s0, { source: 'mirror', kind: 'note', title: 'x' })),
    missing: err(() => applyEvent(s0, { kind: 'note', title: 'x' } as any)),
  };
}

// Quests: succeed when the goal is met (reward), fail past the deadline (penalty).
{
  const opened = applyEvent(s0, { source: 'juniper', kind: 'quest', title: 'Fix the generator', quest: {
    goal: [{ kind: 'fact', target: 'generator.fixed', cmp: '==', value: true },
      { kind: 'stock', target: 'workshop.parts', cmp: '>=', value: 1 }],
    participants: ['cam_lin', 'orion'], deadline_in_minutes: 600,
    reward: [{ op: 'trust_delta', target: 'cam_lin|orion', value: 0.2 }, { op: 'stock_delta', target: 'cam_lin.cash', value: 5 }],
    penalty: [{ op: 'trust_delta', target: 'cam_lin|orion', value: -0.2 }],
  } });
  const qid = opened.event.id;
  const stillOpen = tick(opened.state, 60).state;
  const fixed = applyEvent(stillOpen, juniper({ effects: [{ op: 'fact_set', target: 'generator.fixed', value: true }] }));
  const done = tick(fixed.state, 5);
  const failed = tick(applyEvent(s0, { source: 'juniper', kind: 'quest', title: 'Impossible', quest: {
    goal: [{ kind: 'stock', target: 'diner.coffee', cmp: '>=', value: 999 }],
    participants: ['sofia_bell'], deadline_in_minutes: 30,
    penalty: [{ op: 'trust_delta', target: 'sofia_bell|nico_sable', value: -0.3 }],
  } }).state, 30);
  const fq = Object.values(failed.state.quests)[0];
  out.quests = {
    opened_status: opened.event.status,
    open_after_hour: stillOpen.quests[qid].status,
    cam_purpose_rose: stillOpen.characters.cam_lin.needs!.purpose > tick(s0, 60).state.characters.cam_lin.needs!.purpose,
    succeeded: done.state.quests[qid].status,
    success_fact: done.state.facts[`quest.${qid}`],
    reward_trust: done.state.trust['cam_lin|orion'],
    reward_cash: done.state.stock['cam_lin.cash'] - fixed.state.stock['cam_lin.cash'],
    resolved_event: done.events.filter((e) => e.kind === 'quest_resolved').map((e) => e.source),
    failed: fq.status,
    failed_at: fq.resolved_at,
    penalty_trust: failed.state.trust['sofia_bell|nico_sable'] <= -0.3 + 0.2,
    bad_participant: err(() => applyEvent(s0, { source: 'juniper', kind: 'quest', title: 'x', quest: {
      goal: [{ kind: 'fact', target: 'a.b', cmp: '==', value: true }], participants: ['maggie'], deadline_in_minutes: 60 } })),
    bad_cmp: err(() => applyEvent(s0, { source: 'juniper', kind: 'quest', title: 'x', quest: {
      goal: [{ kind: 'fact', target: 'a.b', cmp: '~=', value: true } as any], participants: ['cam_lin'], deadline_in_minutes: 60 } })),
  };
}

// Malformed and hostile quest input is a SimError (a clean rejected row), never a TypeError.
{
  const q = (quest: object) => err(() => applyEvent(s0, { source: 'juniper', kind: 'quest', title: 'q', quest } as any));
  const goal = [{ kind: 'fact', target: 'a.b', cmp: '==', value: true }];
  out.malformed = {
    target_not_string: q({ goal: [{ kind: 'stock', target: 5, cmp: '>=', value: 1 }], participants: ['cam_lin'], deadline_in_minutes: 60 }),
    reward_not_list: q({ goal, participants: ['cam_lin'], deadline_in_minutes: 60, reward: 7 }),
    penalty_object: q({ goal, participants: ['cam_lin'], deadline_in_minutes: 60, penalty: { op: 'x' } }),
    goal_item_null: q({ goal: [null], participants: ['cam_lin'], deadline_in_minutes: 60 }),
    quest_not_object: q('soon' as any),
    stock_no_item: q({ goal: [{ kind: 'stock', target: 'diner', cmp: '>=', value: 1 }], participants: ['cam_lin'], deadline_in_minutes: 60 }),
    proto_trust_goal: q({ goal: [{ kind: 'trust', target: 'constructor', cmp: '>=', value: 0 }], participants: ['cam_lin'], deadline_in_minutes: 60 }),
    proto_need_goal: q({ goal: [{ kind: 'need', target: 'cam_lin.toString', cmp: '>=', value: 0 }], participants: ['cam_lin'], deadline_in_minutes: 60 }),
  };
}

// Extra keys on effects and conditions are dropped: stored and logged shapes
// match the simEvents validator exactly.
{
  const r = applyEvent(s0, juniper({ effects: [{ op: 'stock_delta', target: 'diner.coffee', value: 1, note: 'x', $bad: 1 }] }));
  const q = applyEvent(s0, { source: 'juniper', kind: 'quest', title: 'q', quest: {
    goal: [{ kind: 'fact', target: 'a.b', cmp: '==', value: true, extra: [1, 2] } as any],
    participants: ['cam_lin'], deadline_in_minutes: 60,
    reward: [{ op: 'trust_delta', target: 'cam_lin|orion', value: 0.1, junk: { deep: 1 } } as any] } });
  const quest = q.state.quests[q.event.id];
  const done = tick(applyEvent(q.state, juniper({ effects: [{ op: 'fact_set', target: 'a.b', value: true }] })).state, 5);
  out.shapes = {
    event_effect: Object.keys(r.event.effects[0]).sort(),
    goal: Object.keys(quest.goal[0]).sort(),
    reward: Object.keys(quest.reward[0]).sort(),
    resolved_effects: done.events.filter((e) => e.kind === 'quest_resolved').flatMap((e) => e.effects.map((x) => Object.keys(x).sort().join(','))),
  };
}

// The biggest legal quest resolves both ways; one effect more is refused.
{
  const many = (n: number) => Array.from({ length: n }, () => ({ op: 'need_delta', target: 'cam_lin.rest', value: 0.01 }));
  const open = (title: string, goal: object[], deadline: number) => applyEvent(s0, { source: 'juniper', kind: 'quest', title,
    quest: { goal, participants: ['cam_lin'], deadline_in_minutes: deadline, reward: many(MAX_EFFECTS - 1), penalty: many(MAX_EFFECTS - 1) } } as any);
  const win = open('win', [{ kind: 'stock', target: 'diner.coffee', cmp: '>=', value: 0 }], 600);
  const lose = open('lose', [{ kind: 'stock', target: 'diner.coffee', cmp: '>=', value: 999 }], 10);
  const won = tick(win.state, 5);
  const lost = tick(lose.state, 10);
  out.max_quest = {
    won: won.state.quests[win.event.id].status,
    won_effects: won.events.filter((e) => e.kind === 'quest_resolved')[0]?.effects.length,
    lost: lost.state.quests[lose.event.id].status,
    too_many: err(() => applyEvent(s0, { source: 'juniper', kind: 'quest', title: 'x', quest: {
      goal: [{ kind: 'fact', target: 'a.b', cmp: '==', value: true }], participants: ['cam_lin'], deadline_in_minutes: 60,
      reward: many(MAX_EFFECTS) } } as any)),
  };
}

// A corrupted quest in the state is abandoned with a reason; the tick carries on.
{
  const opened = applyEvent(s0, { source: 'juniper', kind: 'quest', title: 'Broken', quest: {
    goal: [{ kind: 'stock', target: 'diner.coffee', cmp: '>=', value: 0 }], participants: ['cam_lin'], deadline_in_minutes: 60 } });
  const good = applyEvent(opened.state, { source: 'juniper', kind: 'quest', title: 'Fine', quest: {
    goal: [{ kind: 'stock', target: 'diner.coffee', cmp: '>=', value: 0 }], participants: ['mara_vale'], deadline_in_minutes: 60,
    reward: [{ op: 'trust_delta', target: 'mara_vale|orion', value: 0.1 }] } });
  const bad = JSON.parse(JSON.stringify(good.state));
  bad.quests[opened.event.id].reward = [{ op: 'teleport', target: 'diner.coffee', value: 1 }];
  bad.quests.e999 = null;
  const r = tick(bad, 60);
  const rejected = r.events.filter((e) => e.status === 'rejected');
  out.corrupted = {
    minute: r.state.minute,
    status: r.state.quests[opened.event.id].status,
    error: r.state.quests[opened.event.id].error,
    null_quest_gone: !('e999' in r.state.quests),
    rejected: rejected.map((e) => [e.kind, e.source, e.reason]),
    good_quest: r.state.quests[good.event.id].status,
    good_reward: r.state.trust['mara_vale|orion'],
    later_steps_ran: tick(r.state, 1440).state.minute,
  };
}

// Growth stays bounded: quests, facts and rumors over a long run.
{
  let s = s0;
  let factErr: string | null = null;
  let openErr: string | null = null;
  let questsOpened = 0;
  for (let day = 0; day < 30; day++) {
    for (let i = 0; i < 12; i++) {
      try {
        s = applyEvent(s, { source: 'juniper', kind: 'quest', title: `q${day}.${i}`, quest: {
          goal: [{ kind: 'fact', target: `goal.d${day}x${i}`, cmp: '==', value: true }], participants: ['mara_vale'],
          deadline_in_minutes: 120 + i } }).state;
        questsOpened += 1;
      } catch (e) { openErr = (e as Error).message; }
    }
    for (let i = 0; i < 10; i++) {
      try {
        s = applyEvent(s, juniper({ effects: [{ op: 'fact_set', target: `note.d${day}x${i}`, value: i }] })).state;
      } catch (e) { factErr = (e as Error).message; }
      s = applyEvent(s, { source: 'juniper', kind: 'rumor', title: `r${day}.${i}` }).state;
    }
    s = tick(s, 1440).state;
  }
  const f = Object.keys(s.facts);
  out.growth = {
    quests_opened: questsOpened,
    quests_kept: Object.keys(s.quests).length,
    plain_facts: f.filter((k) => !/^(rumor|quest)\./.test(k)).length,
    rumor_facts: f.filter((k) => k.startsWith('rumor.')).length,
    quest_facts: f.filter((k) => k.startsWith('quest.')).length,
    quest_facts_match: f.filter((k) => k.startsWith('quest.')).every((k) => k.slice(6) in s.quests),
    fact_err: factErr,
    open_err: openErr,
    state_bytes: JSON.stringify(s).length,
    limits: { MAX_FACTS, MAX_RUMORS, MAX_RESOLVED_QUESTS, MAX_OPEN_QUESTS },
  };
}

// Only Juniper's events can be injected; the tick alone makes sim events.
out.inject = { juniper: err(() => checkInjectSource('juniper')), sim: err(() => checkInjectSource('sim')),
  orion: err(() => checkInjectSource('orion')) };

// Schedules are hours 0..23, so work_end is always reached and the wage pays.
{
  const cast = JSON.parse(JSON.stringify(SIM_CAST));
  cast.characters[0].sim.schedule.work_end = 24;
  cast.characters[0].sim.schedule.sleep = 24;
  out.schedule_24 = err(() => createState(1, cast));
  const day = tick(s0, 1440);
  out.wages = day.events.filter((e) => e.kind === 'wage').map((e) => e.title.split(' ')[0]).sort();
}

// The entity rule.
out.entities = checkEntities(s0, ['Mara', 'mara vale', 'diner', 'Diner', 'coffee', 'workshop.parts', 'Orion',
  'juniper_feld', 'Maggie', 'blacksmith', 'Tom']);

out.tick_bad_minutes = err(() => tick(s0, 7));
out.tick_too_long = err(() => tick(s0, 1445));
console.log(JSON.stringify(out));
