const test = require("node:test");
const assert = require("node:assert");
const fs = require("node:fs");
const path = require("node:path");
const p = require("./personality.js");

const CHOICE = {
  choice_id: "curiosity:r1:self:walkway", lane: "curiosity", occurred_at: "2026-11-13T07:29:39Z",
  chosen: { key: "self:walkway", text: "walkway vehicle reports are mainly my own misnaming", market_rank: 4 },
  passed_over: [
    { key: "a", text: "peer briefs refused budget", market_rank: 1 },
    { key: "b", text: "v2 relative switch", market_rank: 2 },
    { key: "c", text: "wp read record grows", market_rank: 3 },
    { key: "d", text: "four wiring points named", market_rank: null },
  ],
  chosen_market_rank: 4, declined_all: false, care_promoted: false, picks_in_decision: 1,
  tendency: { disposition_id: "disp-1", label: "walkway misnaming over peer briefs, 3 of 3", status: "candidate" },
};

test("choiceView: what was picked, at which market rank, and the top passed-over options", () => {
  const v = p.choiceView(CHOICE);
  assert.strictEqual(v.lane, "What to look into");
  assert.strictEqual(v.picked, "Picked: walkway vehicle reports are mainly my own misnaming (market's #4)");
  assert.deepStrictEqual(v.passedOver, ["#1 peer briefs refused budget", "#2 v2 relative switch", "#3 wp read record grows"]);
  assert.strictEqual(v.more, 1);
  assert.strictEqual(v.allPassedOver[3], "four wiring points named");   // unranked: no number
  assert.strictEqual(v.passedOverLabel, "Passed over");
  assert.strictEqual(v.tendency, "Counts toward: walkway misnaming over peer briefs, 3 of 3");
  assert.deepStrictEqual(v.notes, []);
});

test("choiceView: a decline reads as 'said nothing' / 'made no move', never as a pick", () => {
  const out = p.choiceView({ ...CHOICE, lane: "outreach", chosen: null, declined_all: true, tendency: null });
  assert.strictEqual(out.picked, "Said nothing");
  assert.strictEqual(out.declined, true);
  assert.strictEqual(out.passedOverLabel, "Options Orion turned down");
  assert.strictEqual(out.tendency, "");
  assert.strictEqual(p.choiceView({ ...CHOICE, lane: "prime", chosen: null, declined_all: true }).picked, "Made no move");
  assert.strictEqual(p.choiceView({ ...CHOICE, lane: "reading", chosen: null, declined_all: true }).picked,
    "Picked none of them");
});

test("choiceView notes: market's first pick, care, several picks; tolerates junk", () => {
  const v = p.choiceView({ ...CHOICE, chosen: { text: "x", market_rank: 1 }, care_promoted: true, picks_in_decision: 2 });
  assert.deepStrictEqual(v.notes, ["Same as the market's first pick.", "Orion's care for a tendency moved this up.",
    "One of 2 picks in this decision."]);
  assert.strictEqual(p.choiceView({ ...CHOICE, chosen: { text: "x", market_rank: null } }).picked,
    "Picked: x (not on the market's list)");
  const blank = p.choiceView(null);
  assert.strictEqual(blank.declined, true);
  assert.deepStrictEqual(blank.passedOver, []);
});

test("tendencyView: label without the count, n of m, status in words, Prime reach, examples", () => {
  const t = {
    disposition_id: "disp-1", label: "turn start context is stale over camera layer (+13 more), 21 of 21",
    status: "active", weight: 12.04, n: 21, m: 23, name: null,
    examples: [CHOICE, { ...CHOICE, lane: "outreach", chosen: null, declined_all: true, passed_over: [] }],
    juniper: null,
  };
  const v = p.tendencyView(t);
  assert.strictEqual(v.label, "turn start context is stale over camera layer (+13 more)");
  assert.strictEqual(v.count, "21 of 23 choices went this way");
  assert.strictEqual(v.status, "Orion leans on this now");
  assert.strictEqual(v.strength, "Strength 12.0 (Prime sees it at 10)");
  assert.strictEqual(v.inPrime, true);
  assert.match(v.examples[0], /^Picked walkway vehicle reports .* over #1 peer briefs refused budget, #2 v2 relative switch, #3 wp read record grows and 1 more \(What to look into, /);
  assert.match(v.examples[1], /^Said nothing \(Reaching out to you/);
  assert.strictEqual(v.verdict, "You have not answered yet.");
  assert.strictEqual(p.tendencyView({ ...t, status: "candidate" }).inPrime, false);   // only active reaches Prime
  assert.strictEqual(p.tendencyView({ ...t, weight: 9.9 }).inPrime, false);
  assert.strictEqual(p.tendencyView({ ...t, status: "candidate" }).status, "Still forming");
  assert.strictEqual(p.tendencyView(null).label, "(no label)");
});

test("verdictText and afterVerdictText say what Juniper answered in plain words", () => {
  assert.match(p.verdictText({ verdict: "yes", at: "2026-11-13T10:00:00Z" }), /^You said yes on /);
  assert.strictEqual(p.verdictText({ verdict: "no", at: null }), "You said no.");
  assert.strictEqual(p.afterVerdictText({ retired: true }), "Retired. Orion will not lean on this any more.");
  assert.strictEqual(p.afterVerdictText({ retired: false }), "Saved. You said yes.");
  assert.strictEqual(p.verdictUrl("disp-1:a.b"), "/api/personality/tendencies/disp-1%3Aa.b/verdict");
});

test("armsView: not live yet with no marks; per-arm funded share when live", () => {
  assert.strictEqual(p.armsView({ days: 7, total: 0, arms: {} }).live, false);
  assert.match(p.armsView(null).summary, /^Not live yet/);
  const v = p.armsView({ days: 7, total: 6, arms: { care: { funded: 3, deferred: 1 }, holdback: { funded: 1, deferred: 1 } } });
  assert.strictEqual(v.live, true);
  assert.deepStrictEqual(v.rows, [
    { arm: "With Orion's tendencies counted", text: "4 decisions, 3 funded (75%)" },
    { arm: "Without them (the comparison group)", text: "2 decisions, 1 funded (50%)" },
  ]);
});

test("choicesFrom / tendenciesFrom accept the durable-runs wrapper or a bare list", () => {
  assert.deepStrictEqual(p.choicesFrom({ choices: [1] }), [1]);
  assert.deepStrictEqual(p.choicesFrom([2]), [2]);
  assert.deepStrictEqual(p.choicesFrom(null), []);
  assert.deepStrictEqual(p.tendenciesFrom({ tendencies: [3] }), [3]);
  assert.deepStrictEqual(p.tendenciesFrom({}), []);
});

test("PRIME_MIN_WEIGHT matches orion/prime/tending.py MIN_WEIGHT", () => {
  const src = fs.readFileSync(path.join(__dirname, "../../../../orion/prime/tending.py"), "utf8");
  const m = src.match(/^MIN_WEIGHT = ([0-9.]+)$/m);
  assert.ok(m, "MIN_WEIGHT not found in orion/prime/tending.py");
  assert.strictEqual(Number(m[1]), p.PRIME_MIN_WEIGHT);
});

test("DOM layer never writes HTML: upstream text only via textContent", () => {
  const src = fs.readFileSync(path.join(__dirname, "personality.js"), "utf8");
  assert.deepStrictEqual(src.match(/\.(innerHTML|outerHTML|insertAdjacentHTML)\b/g), null);
});
