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
// trade_news + deceive pick who gossips; ask_favor (plus money need) picks who asks for a loan.
export type PurposeWeights = { trade_news: number; deceive: number; ask_favor: number };

export type CharacterSim = {
  // Starting wants. Money is not listed: it follows cash (syncMoney).
  needs: { rest: number; social: number; purpose: number };
  schedule: Schedule;
  work: string;
  cash: number;
  wage: number;
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
  status: 'open' | 'succeeded' | 'failed' | 'abandoned';
  opened_at: number;
  resolved_at: number | null;
  // Why the tick abandoned the quest (resolution threw), else null.
  error: string | null;
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
export type EventStatus = 'applied' | 'rumor' | 'quest_opened' | 'rejected';
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
  reason: string | null;
};

export const SOURCES: readonly string[] = ['sim', 'juniper'];
// Sources allowed through sim/events:inject. Only the tick makes `sim` events.
export const INJECT_SOURCES: readonly string[] = ['juniper'];
export const EFFECT_OPS: readonly string[] = ['stock_delta', 'need_delta', 'fact_set', 'trust_delta', 'place_flag'];
export const STEP_MINUTES = 5;
export const MAX_EFFECTS = 16;
export const MAX_RUMORS = 20;
// Bounds that keep the one state document small (Convex: 1 MB per document,
// 1024 fields per object). Past a limit, new keys are refused, never evicted;
// only resolved quests (newest kept) and rumors (newest kept) are pruned.
export const MAX_FACTS = 100; // plain facts (not rumor.* / quest.*)
export const MAX_FLAGS_PER_PLACE = 16;
export const MAX_OPEN_QUESTS = 10;
export const MAX_RESOLVED_QUESTS = 20;
export const MAX_TICK_MINUTES = 24 * 60;
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
      // Hours 0..23 and in order, so work_end is always a reached hour (the wage pays then).
      check(sc && [sc.wake, sc.work_start, sc.work_end, sc.sleep].every((h) => Number.isInteger(h) && h >= 0 && h <= 23)
        && sc.wake <= sc.work_start && sc.work_start < sc.work_end && sc.work_end <= sc.sleep, `${c.id}: bad schedule`);
      check(typeof sim.work === 'string' && has(s.places, sim.work), `${c.id}: unknown work place`);
      check(isNum(sim.lying_rate) && sim.lying_rate >= 0 && sim.lying_rate <= 1, `${c.id}: lying_rate must be 0..1`);
      check(isNum(sim.wage) && sim.wage >= 0, `${c.id}: wage must be >= 0`);
      const w = sim.purpose_weights;
      check(w && ['trade_news', 'deceive', 'ask_favor'].every((k) => isNum((w as any)[k]) && (w as any)[k] >= 0),
        `${c.id}: purpose weights must be >= 0`);
    }
    s.characters[c.id] = {
      name: String(c.name), visitor: !sim,
      needs: sim ? { money: 0, purpose: sim.needs.purpose, rest: sim.needs.rest, social: sim.needs.social } : null,
      schedule: sim ? { wake: sim.schedule.wake, work_start: sim.schedule.work_start, work_end: sim.schedule.work_end,
        sleep: sim.schedule.sleep } : null,
      work: sim ? sim.work : null, wage: sim ? sim.wage : 0,
      purpose_weights: sim ? { trade_news: sim.purpose_weights.trade_news, deceive: sim.purpose_weights.deceive,
        ask_favor: sim.purpose_weights.ask_favor } : null,
      lying_rate: sim ? sim.lying_rate : 0,
    };
    if (sim) setStock(s, `${c.id}.cash`, sim.cash);
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
// Items an owner of this kind (place or resident) already holds somewhere:
// a new stock key is allowed only for those (workshop.coffee yes, diner.cash no).
function itemsFor(s: State, ownerIsPlace: boolean): Set<string> {
  const out = new Set<string>();
  for (const k of keys(s.stock)) {
    const dot = k.indexOf('.');
    if (has(s.places, k.slice(0, dot)) === ownerIsPlace) out.add(k.slice(dot + 1));
  }
  return out;
}
function syncMoney(s: State) {
  for (const id of residents(s)) {
    s.characters[id].needs!.money = clamp(1 - (s.stock[`${id}.cash`] ?? 0) / MONEY_COMFORT, 0, 1);
  }
}
function placeOwner(s: State, place: string): string | null {
  return residents(s).find((id) => s.characters[id].work === place) ?? null;
}

// ---- effects: the closed set. Every effect is validated before anything mutates. ----

// Validate one effect and rebuild it as exactly {op, target, value}: extra
// keys never reach the state or the simEvents row (its validator is strict).
function normEffect(s: State, raw: unknown, reserved: boolean): Effect {
  check(raw !== null && typeof raw === 'object' && !Array.isArray(raw), 'effect must be an object');
  const r = raw as Record<string, unknown>;
  check(typeof r.op === 'string' && EFFECT_OPS.includes(r.op), `unknown effect op ${String(r.op)}`);
  check(typeof r.target === 'string', 'effect target must be a string');
  const v = r.value;
  check(isNum(v) || typeof v === 'string' || typeof v === 'boolean', 'effect value must be number, string or boolean');
  const e: Effect = { op: r.op as EffectOp, target: r.target, value: v };
  if (e.op === 'stock_delta') {
    const dot = e.target.indexOf('.');
    const owner = e.target.slice(0, dot);
    const item = e.target.slice(dot + 1);
    const isPlace = has(s.places, owner);
    check(dot > 0 && (isPlace || (has(s.characters, owner) && !s.characters[owner].visitor)),
      `stock_delta: unknown owner in ${e.target}`);
    check(has(s.stock, e.target) || itemsFor(s, isPlace).has(item),
      `stock_delta: unknown item in ${e.target} (${isPlace ? 'places' : 'residents'} hold ${[...itemsFor(s, isPlace)].join(', ')})`);
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
    check(reserved || !/^(rumor|quest)\./.test(e.target), `fact_set: ${e.target} is reserved (rumor.* and quest.* are written by the sim)`);
    check(typeof v !== 'string' || v.length <= 200, 'fact_set: string value over 200 chars');
  } else {
    const parts = e.target.split('.');
    check(parts.length === 2 && has(s.places, parts[0]), `place_flag: unknown place in ${e.target}`);
    check(KEY_RE.test(parts[1]), `place_flag: bad flag name in ${e.target}`);
  }
  return e;
}

const plainFact = (k: string) => !/^(rumor|quest)\./.test(k);

// Refuse new keys past the fact and flag limits (never evict real facts).
function checkLimits(s: State, effects: Effect[]) {
  const newFacts = new Set(effects.filter((e) => e.op === 'fact_set' && plainFact(e.target) && !has(s.facts, e.target))
    .map((e) => e.target));
  const facts = Object.keys(s.facts).filter(plainFact).length;
  check(facts + newFacts.size <= MAX_FACTS, `fact limit: ${MAX_FACTS} facts already set, refusing new keys`);
  for (const place of keys(s.places)) {
    const flags = s.places[place].flags;
    const added = new Set(effects.filter((e) => e.op === 'place_flag' && e.target.startsWith(`${place}.`)
      && !has(flags, e.target.slice(place.length + 1))).map((e) => e.target));
    check(Object.keys(flags).length + added.size <= MAX_FLAGS_PER_PLACE, `flag limit: ${place} has ${MAX_FLAGS_PER_PLACE} flags`);
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

const NEED_NAMES: readonly string[] = ['money', 'purpose', 'rest', 'social'];

// Own-property reads only, so targets like `constructor` or `__proto__` read
// as missing instead of reaching Object.prototype.
export function readTarget(s: State, c: Condition): Value | undefined {
  if (c.kind === 'stock') return has(s.stock, c.target) ? s.stock[c.target] : 0;
  if (c.kind === 'fact') return has(s.facts, c.target) ? s.facts[c.target] : undefined;
  if (c.kind === 'trust') return has(s.trust, c.target) ? s.trust[c.target] : undefined;
  const [id, need] = c.target.split('.');
  if (!has(s.characters, id) || !NEED_NAMES.includes(need)) return undefined;
  return s.characters[id].needs?.[need as NeedName];
}

// Validate one condition and rebuild it as exactly {kind, target, cmp, value}.
function normCondition(s: State, raw: unknown): Condition {
  check(raw !== null && typeof raw === 'object' && !Array.isArray(raw), 'condition must be an object');
  const r = raw as Record<string, unknown>;
  check(typeof r.kind === 'string' && ['stock', 'fact', 'need', 'trust'].includes(r.kind), `unknown condition kind ${String(r.kind)}`);
  check(typeof r.cmp === 'string' && ['>=', '<=', '>', '<', '==', '!='].includes(r.cmp), `unknown comparison ${String(r.cmp)}`);
  check(typeof r.target === 'string', 'condition target must be a string');
  const v = r.value;
  check(isNum(v) || typeof v === 'string' || typeof v === 'boolean', 'condition value must be a scalar');
  const c: Condition = { kind: r.kind as Condition['kind'], target: r.target, cmp: r.cmp as Cmp, value: v };
  if (c.kind === 'fact') check(FACT_RE.test(c.target), `bad fact key ${c.target}`);
  else {
    check(isNum(c.value) || c.cmp === '==' || c.cmp === '!=', `${c.kind} conditions compare numbers`);
    if (c.kind === 'stock') {
      const dot = c.target.indexOf('.');
      const owner = c.target.slice(0, dot);
      check(dot > 0 && (has(s.places, owner) || has(s.characters, owner)) && KEY_RE.test(c.target.slice(dot + 1)),
        `stock condition: target must be <owner>.<item> with a known owner, got ${c.target}`);
    } else check(readTarget(s, c) !== undefined, `${c.kind} condition: unknown target ${c.target}`);
  }
  return c;
}

function normEffects(s: State, raw: unknown, max: number, what: string, reserved: boolean): Effect[] {
  if (raw === undefined || raw === null) return [];
  check(Array.isArray(raw) && raw.length <= max, `${what} must be a list of at most ${max}`);
  return raw.map((e) => normEffect(s, e, reserved));
}

export function checkInjectSource(source: unknown): void {
  check(typeof source === 'string' && INJECT_SOURCES.includes(source),
    `source ${String(source)} cannot be injected (allowed: ${INJECT_SOURCES.join(', ')}; only the tick makes sim events)`);
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
  return applyCore(state, input, false);
}

// `reserved` lets the tick itself write quest.* facts (quest results). It is
// the same apply path; Juniper's events always go through applyEvent.
function applyCore(state: State, input: EventInput, reserved: boolean): { state: State; event: AppliedEvent } {
  check(input !== null && typeof input === 'object', 'event must be an object');
  check(SOURCES.includes(input.source), `unknown source ${String(input.source)} (allowed: ${SOURCES.join(', ')})`);
  check(typeof input.kind === 'string' && KEY_RE.test(input.kind), 'kind must be a short snake_case word');
  check(typeof input.title === 'string' && input.title.trim().length > 0 && input.title.length <= 200, 'title must be 1..200 chars');
  const place = input.place ?? null;
  if (place !== null) check(typeof place === 'string' && has(state.places, place), `unknown place ${String(place)}`);
  const s = clone(state);
  const effects = normEffects(s, input.effects, MAX_EFFECTS, 'effects', reserved);
  checkLimits(s, effects);
  const rawQuest = input.quest ?? null;
  let quest: Quest | null = null;
  const id = `e${s.next_event}`;
  if (rawQuest !== null) {
    check(typeof rawQuest === 'object' && !Array.isArray(rawQuest), 'quest must be an object');
    check(Array.isArray(rawQuest.goal) && rawQuest.goal.length >= 1 && rawQuest.goal.length <= 8, 'quest goal needs 1..8 conditions');
    const goal = rawQuest.goal.map((c) => normCondition(s, c));
    check(Array.isArray(rawQuest.participants) && rawQuest.participants.length >= 1 && rawQuest.participants.length <= 8,
      'quest needs 1..8 participants');
    const missing = rawQuest.participants.filter((p) => typeof p !== 'string' || !has(s.characters, p));
    check(missing.length === 0, `unknown quest participants (use character ids): ${missing.map(String).join(', ')}`);
    check(isNum(rawQuest.deadline_in_minutes) && rawQuest.deadline_in_minutes >= STEP_MINUTES
      && rawQuest.deadline_in_minutes <= 14 * 1440, 'deadline_in_minutes must be 5..20160');
    // One slot stays free for the quest.<id> result fact the tick adds.
    const reward = normEffects(s, rawQuest.reward, MAX_EFFECTS - 1, 'quest reward', false);
    const penalty = normEffects(s, rawQuest.penalty, MAX_EFFECTS - 1, 'quest penalty', false);
    const open = keys(s.quests).filter((q) => s.quests[q].status === 'open').length;
    check(open < MAX_OPEN_QUESTS, `open quest limit: ${MAX_OPEN_QUESTS} quests already open`);
    quest = {
      id, title: input.title, goal, participants: rawQuest.participants.map(String),
      deadline: s.minute + Math.floor(rawQuest.deadline_in_minutes), reward, penalty,
      status: 'open', opened_at: s.minute, resolved_at: null, error: null,
    };
  }
  s.next_event += 1;
  for (const e of effects) applyEffect(s, e);
  let status: EventStatus = 'applied';
  if (quest) {
    s.quests[id] = quest;
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
    place, effects: clone(effects), seed: state.rng, status, reason: null,
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
    kind: 'coffee_delivery', per_hour: 1 / 16,
    build: (s) => has(s.stock, 'diner.coffee')
      ? { title: 'A coffee delivery arrives at the diner.', place: 'diner',
        effects: [{ op: 'stock_delta', target: 'diner.coffee', value: 8 }] }
      : null,
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
  return keys(s.quests).some((q) => s.quests[q]?.status === 'open' && Array.isArray(s.quests[q].participants)
    && s.quests[q].participants.includes(id));
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

function questNumber(id: string): number {
  return Number(id.slice(1)) || 0;
}

// Resolve open quests whose goal holds (succeeded) or whose deadline passed
// (failed). A quest whose resolution throws is abandoned with the reason and
// logged as a rejected event; the tick carries on (one bad quest must never
// wedge the cron).
function resolveQuests(s0: State, out: AppliedEvent[]): State {
  let s = s0;
  for (const qid of keys(s.quests)) {
    const q = s.quests[qid];
    if (q && typeof q === 'object' && q.status !== 'open') continue;
    try {
      check(q && typeof q === 'object' && Array.isArray(q.goal) && isNum(q.deadline), 'malformed quest');
      const met = goalMet(s, q.goal);
      if (!met && s.minute < q.deadline) continue;
      const outcome = met ? 'succeeded' : 'failed';
      const next = clone(s);
      next.quests[qid].status = outcome;
      next.quests[qid].resolved_at = s.minute;
      const r = applyCore(next, {
        source: 'sim', kind: 'quest_resolved', title: `Quest ${outcome}: ${q.title}`.slice(0, 200), place: null,
        effects: [{ op: 'fact_set', target: `quest.${qid}`, value: outcome }, ...(met ? q.reward : q.penalty)],
      }, true);
      s = r.state;
      out.push(r.event);
    } catch (err) {
      const reason = err instanceof Error ? err.message : String(err);
      s = clone(s);
      const title = typeof q?.title === 'string' ? q.title : qid;
      if (q && typeof q === 'object') {
        s.quests[qid].status = 'abandoned';
        s.quests[qid].resolved_at = s.minute;
        s.quests[qid].error = reason.slice(0, 200);
      } else delete s.quests[qid];
      out.push({
        id: '', minute: s.minute, source: 'sim', kind: 'quest_resolved', title: `Quest abandoned: ${title}`.slice(0, 200),
        place: null, effects: [], seed: s.rng, status: 'rejected', reason: reason.slice(0, 200),
      });
    }
  }
  pruneQuests(s);
  return s;
}

// Keep the newest MAX_RESOLVED_QUESTS resolved quests; drop older ones and their quest.<id> facts.
function pruneQuests(s: State) {
  const done = keys(s.quests).filter((q) => s.quests[q].status !== 'open')
    .sort((a, b) => ((s.quests[a].resolved_at ?? 0) - (s.quests[b].resolved_at ?? 0)) || (questNumber(a) - questNumber(b)));
  for (const q of done.slice(0, Math.max(0, done.length - MAX_RESOLVED_QUESTS))) {
    delete s.quests[q];
    delete s.facts[`quest.${q}`];
  }
}

// Advance the world by `minutes` (a positive multiple of STEP_MINUTES, at most
// MAX_TICK_MINUTES). tick(s, 60) gives the same world and events as twelve
// tick(s, 5) calls, except `version`, which counts calls (+1 per tick call,
// +1 per applied event).
export function tick(state: State, minutes: number): { state: State; events: AppliedEvent[] } {
  check(Number.isInteger(minutes) && minutes > 0 && minutes % STEP_MINUTES === 0 && minutes <= MAX_TICK_MINUTES,
    `minutes must be a positive multiple of ${STEP_MINUTES}, at most ${MAX_TICK_MINUTES}`);
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
