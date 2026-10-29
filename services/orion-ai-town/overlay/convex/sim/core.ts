// AI Town simulation core (living-world design §1-3). Pure code: no Convex,
// no upstream imports, no clock, no network, no model calls. Node runs this
// file directly with --experimental-strip-types (tests/test_sim_core.py), so
// it uses erasable TypeScript only: no enums, namespaces or parameter
// properties.
//
// Replay rule: every random draw comes from `state.rng`, which is part of the
// state, and every loop walks sorted keys (Convex stores objects with sorted
// keys, so insertion order is not stable across a save). The same state in
// gives the same state out.

export type Value = number | string | boolean;
export type NeedName = 'money' | 'rest' | 'social' | 'purpose';
export type Needs = { money: number; rest: number; social: number; purpose: number };
export type Source = 'sim' | 'juniper';
export type EffectOp = 'stock_delta' | 'need_delta' | 'fact_set' | 'trust_delta' | 'place_flag';
export type Effect = { op: EffectOp; target: string; value: Value };
export type Cmp = '>=' | '<=' | '>' | '<' | '==' | '!=';
export type Condition = { kind: 'stock' | 'fact' | 'need' | 'trust'; target: string; cmp: Cmp; value: Value };
export type Schedule = { wake: number; work_start: number; work_end: number; sleep: number };
export type PurposeWeights = { trade_news: number; persuade: number; deceive: number; ask_favor: number };

export type CharacterSim = {
  // Starting wants. Money is not listed: it follows cash (syncMoney).
  needs: { rest: number; social: number; purpose: number };
  schedule: Schedule;
  home: string;
  work: string;
  cash: number;
  wage: number;
  stock: Record<string, number>;
  purpose_weights: PurposeWeights;
  lying_rate: number;
};
// A character without `sim` is a visitor (Juniper, Orion): a valid name, quest
// participant and trust target, but driven from outside, so no needs or schedule.
export type CharacterSeed = { id: string; name: string; sim?: CharacterSim | null };
export type Cast = {
  places: Record<string, { name: string }>;
  place_stock: Record<string, Record<string, number>>;
  facts?: Record<string, Value>;
  characters: CharacterSeed[];
};

export type Character = {
  name: string;
  visitor: boolean;
  needs: Needs | null;
  schedule: Schedule | null;
  home: string | null;
  work: string | null;
  wage: number;
  purpose_weights: PurposeWeights | null;
  lying_rate: number;
};
export type Quest = {
  id: string;
  title: string;
  goal: Condition[];
  participants: string[];
  deadline: number;
  reward: Effect[];
  penalty: Effect[];
  status: 'open' | 'succeeded' | 'failed';
  opened_at: number;
  resolved_at: number | null;
};
export type State = {
  version: number;
  seed: number;
  rng: number;
  minute: number;
  next_event: number;
  places: Record<string, { name: string; flags: Record<string, Value> }>;
  characters: Record<string, Character>;
  stock: Record<string, number>;
  trust: Record<string, number>;
  facts: Record<string, Value>;
  quests: Record<string, Quest>;
};

export type QuestSpec = {
  goal: Condition[];
  participants: string[];
  deadline_in_minutes: number;
  reward?: Effect[];
  penalty?: Effect[];
};
export type EventInput = {
  source: string;
  kind: string;
  title: string;
  place?: string | null;
  effects?: Effect[];
  quest?: QuestSpec | null;
};
export type EventStatus = 'applied' | 'rumor' | 'quest_opened';
export type AppliedEvent = {
  id: string;
  minute: number;
  source: Source;
  kind: string;
  title: string;
  place: string | null;
  effects: Effect[];
  seed: number;
  status: EventStatus;
};

export const SOURCES: readonly string[] = ['sim', 'juniper'];
export const EFFECT_OPS: readonly string[] = ['stock_delta', 'need_delta', 'fact_set', 'trust_delta', 'place_flag'];
export const STEP_MINUTES = 5;
export const MAX_EFFECTS = 16;
export const MAX_RUMORS = 20;
// Money need follows cash: 0 at MONEY_COMFORT cash or more, 1 at no cash.
export const MONEY_COMFORT = 40;
export const COFFEE_PRICE = 2;
export const MEAL_HOURS: readonly number[] = [8, 12, 18];

export class SimError extends Error {}
function check(ok: unknown, message: string): asserts ok {
  if (!ok) throw new SimError(message);
}
const has = (o: object, k: string) => Object.prototype.hasOwnProperty.call(o, k);
const keys = (o: object) => Object.keys(o).sort();
const clone = <T>(v: T): T => JSON.parse(JSON.stringify(v));
// Round to 6 places so stored numbers stay readable; never return -0
// (Convex encodes -0 as a special float).
const round = (n: number) => {
  const r = Math.round(n * 1e6) / 1e6;
  return r === 0 ? 0 : r;
};
const clamp = (n: number, lo: number, hi: number) => round(Math.max(lo, Math.min(hi, n)));
const isNum = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
const KEY_RE = /^[a-z][a-z0-9_]{0,31}$/;
const FACT_RE = /^[a-z][a-z0-9_]{0,31}(\.[a-z0-9_]{1,32}){0,3}$/;

// mulberry32. The generator state is one uint32 stored in State.rng.
export function draw(s: State): number {
  s.rng = (s.rng + 0x6d2b79f5) >>> 0;
  let t = s.rng;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}

export function hourOf(minute: number): number {
  return Math.floor(minute / 60) % 24;
}

function checkNeeds(n: CharacterSim['needs'], who: string) {
  for (const k of ['purpose', 'rest', 'social'] as const) check(isNum(n[k]) && n[k] >= 0 && n[k] <= 1, `${who}: need ${k} must be 0..1`);
}

export function createState(seed: number, cast: Cast): State {
  check(Number.isInteger(seed) && seed >= 0 && seed <= 0xffffffff, 'seed must be a uint32');
  const s: State = {
    version: 0, seed, rng: seed, minute: 0, next_event: 1,
    places: {}, characters: {}, stock: {}, trust: {}, facts: {}, quests: {},
  };
  for (const id of keys(cast.places)) {
    check(KEY_RE.test(id), `bad place id ${id}`);
    s.places[id] = { name: String(cast.places[id].name), flags: {} };
  }
  for (const place of keys(cast.place_stock)) {
    check(has(s.places, place), `stock for unknown place ${place}`);
    for (const item of keys(cast.place_stock[place])) setStock(s, `${place}.${item}`, cast.place_stock[place][item]);
  }
  for (const fact of keys(cast.facts ?? {})) {
    check(FACT_RE.test(fact), `bad fact key ${fact}`);
    s.facts[fact] = (cast.facts ?? {})[fact];
  }
  for (const c of cast.characters) {
    check(KEY_RE.test(c.id) && !has(s.characters, c.id) && !has(s.places, c.id), `bad or duplicate character id ${c.id}`);
    const sim = c.sim ?? null;
    if (sim) {
      checkNeeds(sim.needs, c.id);
      const sc = sim.schedule;
      check([sc.wake, sc.work_start, sc.work_end, sc.sleep].every((h) => Number.isInteger(h) && h >= 0 && h <= 24)
        && sc.wake <= sc.work_start && sc.work_start < sc.work_end && sc.work_end <= sc.sleep, `${c.id}: bad schedule`);
      check(has(s.places, sim.home) && has(s.places, sim.work), `${c.id}: unknown home/work place`);
      check(isNum(sim.lying_rate) && sim.lying_rate >= 0 && sim.lying_rate <= 1, `${c.id}: lying_rate must be 0..1`);
      check(isNum(sim.wage) && sim.wage >= 0, `${c.id}: wage must be >= 0`);
      const w = sim.purpose_weights;
      check(['trade_news', 'persuade', 'deceive', 'ask_favor'].every((k) => isNum((w as any)[k]) && (w as any)[k] >= 0),
        `${c.id}: purpose weights must be >= 0`);
    }
    s.characters[c.id] = {
      name: String(c.name), visitor: !sim,
      needs: sim ? { money: 0, purpose: sim.needs.purpose, rest: sim.needs.rest, social: sim.needs.social } : null, schedule: sim ? { ...sim.schedule } : null,
      home: sim ? sim.home : null, work: sim ? sim.work : null, wage: sim ? sim.wage : 0,
      purpose_weights: sim ? { ...sim.purpose_weights } : null, lying_rate: sim ? sim.lying_rate : 0,
    };
    if (sim) {
      setStock(s, `${c.id}.cash`, sim.cash);
      for (const item of keys(sim.stock)) setStock(s, `${c.id}.${item}`, sim.stock[item]);
    }
  }
  // Directed trust from every resident toward every other character.
  for (const from of residents(s)) for (const to of keys(s.characters)) if (from !== to) s.trust[`${from}|${to}`] = 0;
  syncMoney(s);
  return s;
}

function setStock(s: State, key: string, n: number) {
  check(Number.isInteger(n) && n >= 0, `stock ${key} must be an integer >= 0`);
  s.stock[key] = n;
}
export function residents(s: State): string[] {
  return keys(s.characters).filter((id) => !s.characters[id].visitor);
}
function itemNames(s: State): Set<string> {
  return new Set(keys(s.stock).map((k) => k.slice(k.indexOf('.') + 1)));
}
function syncMoney(s: State) {
  for (const id of residents(s)) {
    s.characters[id].needs!.money = clamp(1 - (s.stock[`${id}.cash`] ?? 0) / MONEY_COMFORT, 0, 1);
  }
}
function placeOwner(s: State, place: string): string | null {
  return residents(s).find((id) => s.characters[id].work === place) ?? null;
}

// ---- effects: the closed set. validateEffect runs before anything mutates. ----

function validateEffect(s: State, e: Effect): void {
  check(e && typeof e === 'object', 'effect must be an object');
  check(EFFECT_OPS.includes(e.op), `unknown effect op ${String((e as any).op)}`);
  check(typeof e.target === 'string', 'effect target must be a string');
  const v = e.value;
  check(isNum(v) || typeof v === 'string' || typeof v === 'boolean', 'effect value must be number, string or boolean');
  if (e.op === 'stock_delta') {
    const dot = e.target.indexOf('.');
    const owner = e.target.slice(0, dot);
    const item = e.target.slice(dot + 1);
    check(dot > 0 && (has(s.places, owner) || (has(s.characters, owner) && !s.characters[owner].visitor)),
      `stock_delta: unknown owner in ${e.target}`);
    check(has(s.stock, e.target) || itemNames(s).has(item), `stock_delta: unknown item in ${e.target}`);
    check(isNum(v) && Number.isInteger(v) && Math.abs(v) <= 1000, 'stock_delta value must be an integer within ±1000');
  } else if (e.op === 'need_delta') {
    const [id, need] = e.target.split('.');
    check(has(s.characters, id) && !s.characters[id].visitor, `need_delta: unknown resident ${id}`);
    check(need === 'rest' || need === 'social' || need === 'purpose',
      `need_delta: need must be rest, social or purpose (money follows cash: use stock_delta on ${id}.cash)`);
    check(isNum(v) && Math.abs(v) <= 1, 'need_delta value must be a number within ±1');
  } else if (e.op === 'trust_delta') {
    check(has(s.trust, e.target), `trust_delta: unknown pair ${e.target} (use from|to)`);
    check(isNum(v) && Math.abs(v) <= 2, 'trust_delta value must be a number within ±2');
  } else if (e.op === 'fact_set') {
    check(FACT_RE.test(e.target), `fact_set: bad fact key ${e.target}`);
    check(typeof v !== 'string' || v.length <= 200, 'fact_set: string value over 200 chars');
  } else {
    const [place, flag] = e.target.split('.');
    check(has(s.places, place), `place_flag: unknown place ${place}`);
    check(KEY_RE.test(flag ?? ''), `place_flag: bad flag name in ${e.target}`);
  }
}

function applyEffect(s: State, e: Effect): void {
  const v = e.value;
  if (e.op === 'stock_delta') s.stock[e.target] = Math.max(0, (s.stock[e.target] ?? 0) + (v as number));
  else if (e.op === 'need_delta') {
    const [id, need] = e.target.split('.');
    const needs = s.characters[id].needs!;
    needs[need as NeedName] = clamp(needs[need as NeedName] + (v as number), 0, 1);
  } else if (e.op === 'trust_delta') s.trust[e.target] = clamp(s.trust[e.target] + (v as number), -1, 1);
  else if (e.op === 'fact_set') s.facts[e.target] = v;
  else {
    const [place, flag] = e.target.split('.');
    s.places[place].flags[flag] = v;
  }
}

// ---- conditions (quest goals): AND of comparisons over the state ----

export function readTarget(s: State, c: Condition): Value | undefined {
  if (c.kind === 'stock') return s.stock[c.target] ?? 0;
  if (c.kind === 'fact') return has(s.facts, c.target) ? s.facts[c.target] : undefined;
  if (c.kind === 'trust') return s.trust[c.target];
  const [id, need] = c.target.split('.');
  return s.characters[id]?.needs?.[need as NeedName];
}

function validateCondition(s: State, c: Condition): void {
  check(c && ['stock', 'fact', 'need', 'trust'].includes(c.kind), `unknown condition kind ${String((c as any)?.kind)}`);
  check(['>=', '<=', '>', '<', '==', '!='].includes(c.cmp), `unknown comparison ${String(c.cmp)}`);
  check(isNum(c.value) || typeof c.value === 'string' || typeof c.value === 'boolean', 'condition value must be a scalar');
  if (c.kind === 'fact') check(FACT_RE.test(c.target), `bad fact key ${c.target}`);
  else {
    check(isNum(c.value) || c.cmp === '==' || c.cmp === '!=', `${c.kind} conditions compare numbers`);
    if (c.kind === 'stock') {
      const owner = c.target.split('.')[0];
      check(has(s.places, owner) || has(s.characters, owner), `stock condition: unknown owner in ${c.target}`);
    } else check(readTarget(s, c) !== undefined, `${c.kind} condition: unknown target ${c.target}`);
  }
}

export function checkCondition(s: State, c: Condition): boolean {
  const a = readTarget(s, c);
  if (c.cmp === '==') return a === c.value;
  if (c.cmp === '!=') return a !== c.value;
  if (!isNum(a) || !isNum(c.value)) return false;
  if (c.cmp === '>=') return a >= c.value;
  if (c.cmp === '<=') return a <= c.value;
  if (c.cmp === '>') return a > c.value;
  return a < c.value;
}

export function goalMet(s: State, goal: Condition[]): boolean {
  return goal.every((c) => checkCondition(s, c));
}

// ---- the entity rule ----

// Names that are not a place, item or character in the state. Matches ids,
// display names and first names, case-insensitively.
export function checkEntities(s: State, names: string[]): string[] {
  const known = new Set<string>();
  const add = (n: string) => known.add(n.trim().toLowerCase());
  for (const id of keys(s.places)) { add(id); add(s.places[id].name); }
  for (const id of keys(s.characters)) {
    const name = s.characters[id].name;
    add(id); add(name); add(name.split(' ')[0]);
  }
  for (const k of keys(s.stock)) add(k);
  for (const item of itemNames(s)) add(item);
  return names.filter((n) => !known.has(String(n).trim().toLowerCase()));
}

// ---- the ONE apply path: every event (tick template, quest result, Juniper) ----

export function applyEvent(state: State, input: EventInput): { state: State; event: AppliedEvent } {
  check(input && typeof input === 'object', 'event must be an object');
  check(SOURCES.includes(input.source), `unknown source ${String(input.source)} (allowed: ${SOURCES.join(', ')})`);
  check(typeof input.kind === 'string' && KEY_RE.test(input.kind), 'kind must be a short snake_case word');
  check(typeof input.title === 'string' && input.title.trim().length > 0 && input.title.length <= 200, 'title must be 1..200 chars');
  const effects = input.effects ?? [];
  check(Array.isArray(effects) && effects.length <= MAX_EFFECTS, `effects must be a list of at most ${MAX_EFFECTS}`);
  const place = input.place ?? null;
  if (place !== null) check(typeof place === 'string' && has(state.places, place), `unknown place ${String(place)}`);
  const s = clone(state);
  for (const e of effects) validateEffect(s, e);
  const quest = input.quest ?? null;
  if (quest) {
    check(Array.isArray(quest.goal) && quest.goal.length >= 1 && quest.goal.length <= 8, 'quest goal needs 1..8 conditions');
    for (const c of quest.goal) validateCondition(s, c);
    check(Array.isArray(quest.participants) && quest.participants.length >= 1 && quest.participants.length <= 8,
      'quest needs 1..8 participants');
    const missing = quest.participants.filter((p) => typeof p !== 'string' || !has(s.characters, p));
    check(missing.length === 0, `unknown quest participants (use character ids): ${missing.join(', ')}`);
    check(isNum(quest.deadline_in_minutes) && quest.deadline_in_minutes >= STEP_MINUTES
      && quest.deadline_in_minutes <= 14 * 1440, 'deadline_in_minutes must be 5..20160');
    for (const e of [...(quest.reward ?? []), ...(quest.penalty ?? [])]) validateEffect(s, e);
    check((quest.reward ?? []).length <= MAX_EFFECTS && (quest.penalty ?? []).length <= MAX_EFFECTS, 'too many quest effects');
  }
  const id = `e${s.next_event}`;
  s.next_event += 1;
  for (const e of effects) applyEffect(s, e);
  let status: EventStatus = 'applied';
  if (quest) {
    s.quests[id] = {
      id, title: input.title, goal: clone(quest.goal), participants: [...quest.participants],
      deadline: s.minute + Math.floor(quest.deadline_in_minutes), reward: clone(quest.reward ?? []),
      penalty: clone(quest.penalty ?? []), status: 'open', opened_at: s.minute, resolved_at: null,
    };
    status = 'quest_opened';
  } else if (effects.length === 0) {
    // Free text with no effects is a rumor: something to talk about, no numbers move.
    s.facts[`rumor.${id}`] = input.title;
    pruneRumors(s);
    status = 'rumor';
  }
  syncMoney(s);
  s.version += 1;
  const event: AppliedEvent = {
    id, minute: s.minute, source: input.source as Source, kind: input.kind, title: input.title,
    place, effects: clone(effects), seed: state.rng, status,
  };
  return { state: s, event };
}

function pruneRumors(s: State) {
  const rumors = Object.keys(s.facts).filter((k) => k.startsWith('rumor.e'))
    .sort((a, b) => Number(a.slice(7)) - Number(b.slice(7)));
  for (const k of rumors.slice(0, Math.max(0, rumors.length - MAX_RUMORS))) delete s.facts[k];
}

// ---- templates: seeded Poisson arrivals ----

type Built = { title: string; place: string | null; effects: Effect[] } | null;
type Template = { kind: string; per_hour: number; build: (s: State) => Built };

function pick<T>(s: State, items: T[], weight: (t: T) => number): T | null {
  const total = items.reduce((sum, t) => sum + weight(t), 0);
  if (!items.length || total <= 0) return null;
  let r = draw(s) * total;
  for (const t of items) {
    r -= weight(t);
    if (r < 0) return t;
  }
  return items[items.length - 1];
}

export const TEMPLATES: readonly Template[] = [
  {
    kind: 'coffee_delivery', per_hour: 1 / 12,
    build: (s) => {
      if (!has(s.stock, 'diner.coffee')) return null;
      if (draw(s) < 0.25) {
        return { title: 'The coffee delivery is running late.', place: 'diner',
          effects: [{ op: 'fact_set', target: 'shipment.coffee_late', value: true }] };
      }
      return { title: 'A coffee delivery arrives at the diner.', place: 'diner', effects: [
        { op: 'stock_delta', target: 'diner.coffee', value: 8 },
        { op: 'fact_set', target: 'shipment.coffee_late', value: false }] };
    },
  },
  {
    kind: 'parts_delivery', per_hour: 1 / 10,
    build: (s) => has(s.stock, 'workshop.parts')
      ? { title: 'A box of parts arrives at the workshop.', place: 'workshop',
        effects: [{ op: 'stock_delta', target: 'workshop.parts', value: 3 }] }
      : null,
  },
  {
    kind: 'breakdown', per_hour: 1 / 16,
    build: (s) => has(s.places, 'workshop') && s.places.workshop.flags.broken !== true
      ? { title: 'The workshop generator breaks down.', place: 'workshop',
        effects: [{ op: 'place_flag', target: 'workshop.broken', value: true }] }
      : null,
  },
  {
    kind: 'repair', per_hour: 1 / 3,
    build: (s) => {
      if (!has(s.places, 'workshop') || s.places.workshop.flags.broken !== true) return null;
      if ((s.stock['workshop.parts'] ?? 0) < 2) return null;
      const effects: Effect[] = [
        { op: 'stock_delta', target: 'workshop.parts', value: -2 },
        { op: 'place_flag', target: 'workshop.broken', value: false }];
      const fixer = placeOwner(s, 'workshop');
      if (fixer) effects.push({ op: 'need_delta', target: `${fixer}.purpose`, value: -0.2 });
      return { title: 'The workshop generator is fixed with two parts.', place: 'workshop', effects };
    },
  },
  {
    kind: 'gossip', per_hour: 1 / 2,
    build: (s) => {
      const who = residents(s);
      const speaker = pick(s, who, (id) => {
        const w = s.characters[id].purpose_weights!;
        return w.trade_news + w.deceive;
      });
      if (!speaker) return null;
      const listener = pick(s, who.filter((id) => id !== speaker), () => 1);
      if (!listener) return null;
      const lie = draw(s) < s.characters[speaker].lying_rate;
      const caught = lie && draw(s) < 0.5;
      const a = s.characters[speaker].name;
      const b = s.characters[listener].name;
      return {
        title: caught ? `${b} catches ${a} spinning a story.` : `${a} passes some news to ${b}.`,
        place: null,
        effects: [
          { op: 'trust_delta', target: `${listener}|${speaker}`, value: caught ? -0.1 : 0.03 },
          { op: 'need_delta', target: `${speaker}.social`, value: -0.05 },
          { op: 'need_delta', target: `${listener}.social`, value: -0.05 }],
      };
    },
  },
  {
    kind: 'favor', per_hour: 1 / 4,
    build: (s) => {
      const who = residents(s);
      const asker = pick(s, who, (id) => s.characters[id].purpose_weights!.ask_favor + s.characters[id].needs!.money);
      if (!asker) return null;
      const helper = pick(s, who.filter((id) => id !== asker && (s.stock[`${id}.cash`] ?? 0) >= 2), () => 1);
      if (!helper) return null;
      return {
        title: `${s.characters[helper].name} lends ${s.characters[asker].name} two cash.`, place: null,
        effects: [
          { op: 'stock_delta', target: `${helper}.cash`, value: -2 },
          { op: 'stock_delta', target: `${asker}.cash`, value: 2 },
          { op: 'trust_delta', target: `${asker}|${helper}`, value: 0.06 }],
      };
    },
  },
];

// ---- the tick ----

function inQuest(s: State, id: string): boolean {
  return keys(s.quests).some((q) => s.quests[q].status === 'open' && s.quests[q].participants.includes(id));
}

// Needs drift by schedule, one STEP_MINUTES step. Needs are wants: 0 means
// satisfied, 1 means urgent.
function drift(s: State) {
  const hour = hourOf(s.minute);
  for (const id of residents(s)) {
    const c = s.characters[id];
    const n = c.needs!;
    const sc = c.schedule!;
    const asleep = hour < sc.wake || hour >= sc.sleep;
    const working = hour >= sc.work_start && hour < sc.work_end;
    n.rest = clamp(n.rest + (asleep ? -0.015 : working ? 0.005 : 0.002), 0, 1);
    n.social = clamp(n.social + (asleep ? 0 : working ? 0.001 : 0.003), 0, 1);
    n.purpose = clamp(n.purpose + (inQuest(s, id) ? 0.004 : -0.0005), 0, 1);
  }
}

// Routine, schedule-driven events at the top of an hour: meals and wages.
function routines(s: State): EventInput[] {
  if (s.minute % 60 !== 0) return [];
  const hour = hourOf(s.minute);
  const out: EventInput[] = [];
  const owner = placeOwner(s, 'diner');
  let coffee = s.stock['diner.coffee'] ?? 0;
  for (const id of residents(s)) {
    const c = s.characters[id];
    const sc = c.schedule!;
    if (MEAL_HOURS.includes(hour) && hour >= sc.wake && hour < sc.sleep && coffee > 0
      && (id === owner || (s.stock[`${id}.cash`] ?? 0) >= COFFEE_PRICE)) {
      coffee -= 1;
      const effects: Effect[] = [
        { op: 'stock_delta', target: 'diner.coffee', value: -1 },
        { op: 'need_delta', target: `${id}.rest`, value: -0.08 },
        { op: 'need_delta', target: `${id}.social`, value: -0.06 }];
      if (id !== owner) {
        effects.push({ op: 'stock_delta', target: `${id}.cash`, value: -COFFEE_PRICE });
        if (owner) effects.push({ op: 'stock_delta', target: `${owner}.cash`, value: COFFEE_PRICE });
      }
      out.push({ source: 'sim', kind: 'meal', title: `${c.name} has a coffee at the diner.`, place: 'diner', effects });
    }
    if (hour === sc.work_end && c.wage > 0) {
      out.push({ source: 'sim', kind: 'wage', title: `${c.name} is paid for the day.`, place: c.work,
        effects: [{ op: 'stock_delta', target: `${id}.cash`, value: c.wage }] });
    }
  }
  return out;
}

function resolveQuests(s0: State, out: AppliedEvent[]): State {
  let s = s0;
  for (const qid of keys(s.quests)) {
    const q = s.quests[qid];
    if (q.status !== 'open') continue;
    const met = goalMet(s, q.goal);
    if (!met && s.minute < q.deadline) continue;
    const outcome = met ? 'succeeded' : 'failed';
    const next = clone(s);
    next.quests[qid].status = outcome;
    next.quests[qid].resolved_at = s.minute;
    // Validated at injection; effects clamp, so applying cannot fail later.
    const r = applyEvent(next, {
      source: 'sim', kind: 'quest_resolved', title: `Quest ${outcome}: ${q.title}`.slice(0, 200), place: null,
      effects: [{ op: 'fact_set', target: `quest.${qid}`, value: outcome }, ...(met ? q.reward : q.penalty)],
    });
    s = r.state;
    out.push(r.event);
  }
  return s;
}

// Advance the world by `minutes` (a positive multiple of STEP_MINUTES).
// tick(s, 60) equals twelve tick(s, 5) calls.
export function tick(state: State, minutes: number): { state: State; events: AppliedEvent[] } {
  check(Number.isInteger(minutes) && minutes > 0 && minutes % STEP_MINUTES === 0,
    `minutes must be a positive multiple of ${STEP_MINUTES}`);
  let s = clone(state);
  const events: AppliedEvent[] = [];
  for (let step = 0; step < minutes / STEP_MINUTES; step++) {
    s.minute += STEP_MINUTES;
    drift(s);
    for (const input of routines(s)) {
      const r = applyEvent(s, input);
      s = r.state;
      events.push(r.event);
    }
    for (const t of TEMPLATES) {
      const p = 1 - Math.exp(-t.per_hour * STEP_MINUTES / 60);
      if (draw(s) >= p) continue;
      const built = t.build(s);
      if (!built) continue;
      const r = applyEvent(s, { source: 'sim', kind: t.kind, ...built });
      s = r.state;
      events.push(r.event);
    }
    s = resolveQuests(s, events);
    syncMoney(s);
  }
  s.version += 1;
  return { state: s, events };
}
