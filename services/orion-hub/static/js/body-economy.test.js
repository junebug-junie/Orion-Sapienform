const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");
const fs = require("node:fs");
const tab = require("./body-economy.js");

// The fixture is the exact physiology.report.v1 shape; the Python route test and the browser smoke
// read the same file.
const FIXTURE = JSON.parse(fs.readFileSync(path.join(__dirname, "../../tests/fixtures/body_economy_report.json"), "utf8"));
const START = Date.parse(FIXTURE.window.start);
const END = Date.parse(FIXTURE.window.end);

test("fixture carries every top-level section the page reads", () => {
  for (const key of ["generated_at", "window", "limits", "wants", "rhythm", "money", "unit_costs", "waste",
    "queue_wait", "decisions", "missing", "claude", "vram", "value_per_cost", "services"]) {
    assert.ok(key in FIXTURE, key);
  }
});

test("brushing (H2): a window keeps exactly the decisions inside it, end exclusive", () => {
  // Drag from 70% to 100% of the 24 h window: 2026-10-29 11:48Z .. 19:00Z.
  const w = tab.brushWindow(1.0, 0.7, START, END); // drag right-to-left gives the same window
  assert.equal(w.from, START + Math.round(0.7 * (END - START)));
  assert.equal(w.to, END);
  const kept = tab.filterDecisions(FIXTURE.decisions, { from: w.from, to: w.to });
  const expected = FIXTURE.decisions.filter((d) => {
    const t = Date.parse(d.decided_at);
    return t >= w.from && t < w.to;
  });
  assert.deepEqual(kept, expected);
  assert.equal(kept.length, 7);
  assert.ok(kept.every((d) => Date.parse(d.decided_at) >= w.from));
  // a row exactly on the end edge is outside; exactly on the start edge is inside
  const edge = [{ decided_at: new Date(w.from).toISOString(), family: "a", outcome: "funded" },
    { decided_at: new Date(w.to).toISOString(), family: "b", outcome: "funded" }];
  assert.deepEqual(tab.filterDecisions(edge, w).map((d) => d.family), ["a"]);
});

test("brushing clamps outside the chart and combines with outcome and the family search", () => {
  const w = tab.brushWindow(-0.5, 1.7, START, END);
  assert.deepEqual(w, { from: START, to: END });
  const deferredOutreach = tab.filterDecisions(FIXTURE.decisions, { from: w.from, to: w.to, outcome: "deferred", query: " OUTREACH " });
  assert.deepEqual(deferredOutreach.map((d) => d.decided_at), ["2026-10-29T18:25:00Z"]);
  assert.equal(tab.filterDecisions(FIXTURE.decisions, { outcome: "all" }).length, FIXTURE.decisions.length);
  assert.equal(tab.filterDecisions(FIXTURE.decisions, { from: null, to: null }).length, FIXTURE.decisions.length);
});

test("idle lanes fold away only when slack every hour and now", () => {
  const byKey = Object.fromEntries(FIXTURE.limits.map((l) => [l.constraint, l]));
  assert.equal(tab.isIdleLimit(byKey["lane:fast"]), true);
  assert.equal(tab.isIdleLimit(byKey["lane:diffusion"]), true);
  assert.equal(tab.isIdleLimit(byKey["heat"]), false);
  assert.equal(tab.isIdleLimit(byKey["lane:chat"]), false); // slack now, binding earlier
  assert.equal(tab.isIdleLimit({ hourly: [], now: { bind_state: "no_reading" } }), false);
  // only lanes fold: Juniper's attention was slack all day and still stays in view
  assert.equal(tab.isIdleLimit(byKey["attention:juniper"]), false);
});

test("exogenous hours are hatched (a pattern), never a price colour", () => {
  const energy = FIXTURE.limits.find((l) => l.constraint === "energy");
  const svg = tab.limitSvg(energy, START, END);
  assert.equal((svg.match(/class="c-exo"/g) || []).length, 24);
  assert.ok(!svg.includes("c-bind"));
  const heat = tab.limitSvg(FIXTURE.limits.find((l) => l.constraint === "heat"), START, END);
  assert.equal((heat.match(/class="c-bind"/g) || []).length, 5);
  assert.ok(heat.includes('class="budget"') && heat.includes('class="level"'));
});

test("limit tooltip names the hour, reading and state", () => {
  const heat = FIXTURE.limits.find((l) => l.constraint === "heat");
  const p = tab.pointAt(heat, 23.5 / 24, START, END);
  assert.equal(p.level, 32.1);
  const tip = tab.limitTip(heat, p, "UTC");
  assert.match(tip, /18:00 · 32\.1 °C · binding · price 0\.10/);
});

test("wants: top 4 families plus other, deferred kept apart, every unit counted once", () => {
  const b = tab.bucketWants(FIXTURE.wants, START, END, 6);
  assert.equal(b.buckets.length, 4);
  assert.equal(b.families.length, 5);
  assert.equal(b.families[4], "other");
  let funded = 0, deferred = 0;
  for (const k of b.buckets) {
    deferred += k.deferred;
    for (const f of Object.keys(k.funded)) funded += k.funded[f];
  }
  const raw = FIXTURE.wants.hourly.reduce((a, r) => ({ f: a.f + r.funded, d: a.d + r.deferred }), { f: 0, d: 0 });
  assert.equal(funded, raw.f);
  assert.equal(deferred, raw.d);
  assert.ok(b.buckets[3].deferredBy["hub.outreach"] > 0);
});

test("family colour is fixed by name, not by rank", () => {
  assert.equal(tab.colorForFamily("hub.outreach"), "#199e70");
  assert.equal(tab.colorForFamily("other"), "#6b7280");
  assert.equal(tab.colorForFamily("some.new_family"), tab.colorForFamily("some.new_family"));
});

test("decision life, reason words and prices seen", () => {
  const deferred = FIXTURE.decisions[0];
  assert.equal(tab.blockedBy(deferred), "lane:agent");
  assert.equal(tab.reasonWords(deferred), "deferred: lane:agent is full");
  assert.equal(tab.decisionLife(deferred)[1].text, "bid · deferred: lane:agent is full");
  const sated = FIXTURE.decisions.find((d) => d.reason === "sated");
  assert.equal(tab.blockedBy(sated), "not hungry");
  const funded = FIXTURE.decisions.find((d) => d.outcome === "funded");
  assert.equal(tab.blockedBy(funded), "—");
  assert.deepEqual(tab.pricesList({ a: 0, b: 0.5, c: 1, d: null }).map((p) => p.k), ["c", "b"]);
});

test("empty states say why: missing reason, unreachable since, or a plain default", () => {
  const report = { missing: [{ section: "wants", reason: "no wants yet: want_ledger is empty" }] };
  assert.equal(tab.emptyText(report, "wants", null), "no wants yet: want_ledger is empty");
  assert.equal(tab.emptyText(report, "rhythm", null), "No heavy work recorded in this window.");
  const err = { status: 502, detail: "durable_runs_unavailable: ConnectError", unreachable_since: "2026-10-29T17:00:00Z" };
  assert.equal(tab.emptyText(report, "wants", err, "UTC"), "durable-runs unreachable since 29/10, 17:00: durable_runs_unavailable: ConnectError");
  assert.match(tab.unreachableText({ status: 400, detail: "bad window" }), /refused \(HTTP 400\): bad window/);
});

test("hosts: what is spinning each machine -- now vs rest, a GPU/CPU/other bar, cards with holders", () => {
  const html = tab.hostsHtml(FIXTURE, null, "UTC");
  const hosts = [...html.matchAll(/data-host="([^"]+)"/g)].map((m) => m[1]);
  assert.deepEqual(hosts, FIXTURE.hosts.map((h) => h.host));
  const circe = FIXTURE.hosts.find((h) => h.host === "circe");
  assert.ok(html.includes("<strong>circe</strong> " + Math.round(circe.chassis_w_now) + " W now · rest 541 W"));
  assert.ok(html.includes("+" + Math.round(circe.above_rest_w) + " W above rest"));
  // the stacked bar: three parts, scaled to the biggest machine, with a tick at rest
  assert.match(html, /data-host-stack="circe"><span data-part="gpu"[^>]*style="width:[0-9.]+%;background:#3987e5"/);
  assert.ok(html.includes('data-part="cpu"') && html.includes('data-part="other"') && html.includes('class="rest-tick"'));
  // each card: watts, its rest, and who held it (plain-word labels)
  assert.ok(html.includes('data-host-card="circe:gpu1"'));
  assert.match(html, /held by curiosity\.investigate runs [0-9.]+ h/);
  // sponsored work is not in Orion's top CPU users; it has its own line
  assert.ok(!/Top CPU users[^<]*zelda-dojo/.test(html));
  assert.match(html, /data-sponsored="zelda_dojo">Sponsored by Juniper, not Orion's activity: zelda_dojo · 32 containers/);
  assert.ok(html.includes("Orion's body at rest here costs $49.56/month"));
  // this week's floor 578 W is 37 W over the pinned 541 W: said as new always-on draw
  assert.ok(html.includes("37 W above the pinned rest: new always-on draw"));
});

test("hosts: no machine reading says why; an asleep machine's note is shown", () => {
  const none = tab.hostsHtml({ missing: [{ section: "hosts", reason: "no biometrics row in the last 15 min for hecate" }] }, null, "UTC");
  assert.ok(none.includes("no biometrics row in the last 15 min for hecate"));
  assert.ok(tab.hostsHtml({}, null, "UTC").includes("No machine reading yet."));
  const err = { status: 502, detail: "durable_runs_unavailable", unreachable_since: "2026-10-29T17:00:00Z" };
  assert.match(tab.hostsHtml(null, err, "UTC"), /durable-runs unreachable since/);
  const odd = tab.hostsHtml({ hosts: [{ host: "x", chassis_w_now: null, cards: [{ card: "g", watts: null }] }] }, null, "UTC");
  assert.ok(odd.includes("<strong>x</strong> ? now") && !odd.includes("NaN"));
});

test("energy over its envelope reads as Orion's body at rest, never 'not Orion's doing'", () => {
  assert.equal(tab.stateWordsFor("energy", "exogenous_bind"), "Orion's body at rest alone reaches the envelope");
  assert.equal(tab.stateWordsFor("heat", "exogenous_bind"), "over its line, not Orion's doing");
  assert.equal(tab.stateWordsFor("energy", "slack"), "room to spare");
});

test("formatting never turns unknown into zero", () => {
  assert.equal(tab.fmtUsd(null), "not measured");
  assert.equal(tab.fmtUsd(0.00005), "$0.000050");
  assert.equal(tab.fmtUsd(0.125), "$0.125");
  assert.equal(tab.fmtUsd(181.71), "$181.71");
  assert.equal(tab.fmtLevel(null, "C"), "no reading");
  assert.equal(tab.fmtLevel(32.14, "C"), "32.1 °C");
  assert.equal(tab.fmtLevel(181.71, "USD"), "$181.71");
  assert.equal(tab.fmtSeconds(1840), "31 min");
  assert.equal(tab.fmtSeconds(312.5), "5.2 min");
  assert.equal(tab.fmtCount(9800000), "9.8M");
  assert.equal(tab.esc('<a "x">'), "&lt;a &quot;x&quot;&gt;");
});

test("a 200 that is not a report is refused, not drawn as empty data", () => {
  assert.equal(tab.isReport(FIXTURE), true);
  assert.equal(tab.isReport({}), false);
  assert.equal(tab.isReport([]), false);
  assert.equal(tab.isReport({ schema_version: "physiology.state.v1", window: {}, generated_at: "x" }), false);
  assert.equal(tab.isReport({ window: { start: "a", end: "b" }, generated_at: "x" }), true);
  assert.match(tab.unreachableText({ not_report: true, detail: "expected physiology.report.v1" }), /^the Hub answered, but not with a body report/);
  assert.match(tab.unreachableText({ network: true, detail: "Failed to fetch" }), /^the Hub did not answer: Failed to fetch/);
});

test("odd family values never throw", () => {
  assert.equal(tab.colorForFamily(null), "#6b7280");
  assert.equal(tab.colorForFamily(42), "#6b7280");
  assert.equal(tab.filterDecisions([{ family: null, outcome: "funded", decided_at: "2026-10-29T00:00:00Z" }], { query: "x" }).length, 0);
});
