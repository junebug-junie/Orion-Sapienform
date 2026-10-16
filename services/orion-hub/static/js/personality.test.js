const test = require("node:test");
const assert = require("node:assert");
const p = require("./personality.js");

test("lanesText sorts by count then name and drops zeros and junk", () => {
  assert.strictEqual(p.lanesText({ journal: 3, dream: 4, chat: 0, reading: 3 }), "dream: 4, journal: 3, reading: 3");
  assert.strictEqual(p.lanesText({}), "—");
  assert.strictEqual(p.lanesText(null), "—");
  assert.strictEqual(p.lanesText({ dream: "x" }), "—");
});

test("sparklineModel orders by time, scales min..max, and handles one point and flat history", () => {
  const m = p.sparklineModel([
    { at: "2026-10-16T12:00:00Z", weight: 3 },
    { at: "2026-10-16T10:00:00Z", weight: 1 },
    { at: "2026-10-16T11:00:00Z", weight: "bad" },
  ], 80, 18);
  assert.strictEqual(m.n, 2);
  assert.strictEqual(m.first, 1);
  assert.strictEqual(m.last, 3);
  assert.strictEqual(m.points, "1.0,17.0 79.0,1.0");   // low on the left, high on the right
  const one = p.sparklineModel([{ at: "x", weight: 2 }], 80, 18);
  assert.strictEqual(one.n, 1);
  assert.strictEqual(one.points, "1.0,9.0 79.0,9.0");
  const flat = p.sparklineModel([{ at: "a", weight: 2 }, { at: "b", weight: 2 }, { at: "c", weight: 2 }], 80, 18);
  assert.ok(flat.points.split(" ").every((pt) => pt.endsWith(",9.0")));
  assert.strictEqual(p.sparklineModel([], 80, 18), null);
  assert.strictEqual(p.sparklineModel(null), null);
});

test("sparklineSvg is numbers plus an escaped title, empty without history", () => {
  const svg = p.sparklineSvg([{ at: "a", weight: 0.5 }, { at: "b", weight: 1.25 }]);
  assert.match(svg, /^<svg class="spark"/);
  assert.match(svg, /<title>weight 0\.500 → 1\.250 over 2 points<\/title>/);
  assert.strictEqual(p.sparklineSvg([]), "");
});

test("escapeHtml neutralises markup in Orion's raw words", () => {
  assert.strictEqual(p.escapeHtml(`<img src=x onerror="a('b')">&`),
    "&lt;img src=x onerror=&quot;a(&#39;b&#39;)&quot;&gt;&amp;");
  assert.strictEqual(p.escapeHtml(null), "");
});

test("dispositionRow formats a node and keeps the label verbatim (rendered as text)", () => {
  const r = p.dispositionRow({
    disposition_id: "d1", label: "<b>the gap</b>", status: "active", origin: "recurrence", weight: 2.4,
    evidence_count: 7, lane_counts: { dream: 4, journal: 3 }, chosen_count: 1,
    first_seen_at: "2026-10-16T06:00:00Z", last_reinforced_at: "2026-10-16T12:00:00Z",
    history: [{ at: "2026-10-16T06:00:00Z", weight: 1 }],
  });
  assert.strictEqual(r.id, "d1");
  assert.strictEqual(r.label, "<b>the gap</b>");
  assert.strictEqual(r.weight, "2.400");
  assert.strictEqual(r.evidence, 7);
  assert.strictEqual(r.lanes, "dream: 4, journal: 3");
  assert.strictEqual(r.chosen, 1);
  assert.strictEqual(r.history.length, 1);
  const blank = p.dispositionRow(null);
  assert.strictEqual(blank.label, "(no label)");
  assert.strictEqual(blank.weight, "—");
  assert.strictEqual(blank.first_seen, "—");
});

test("evidenceRow signs the weight delta and tolerates missing numbers", () => {
  const r = p.evidenceRow({ evidence_id: "e1", lane: "dream", occurred_at: "2026-10-16T06:00:00Z",
                            excerpt: "I kept circling it", similarity: 0.8123, weight_delta: 0.25 });
  assert.strictEqual(r.similarity, "0.81");
  assert.strictEqual(r.weight_delta, "+0.250");
  assert.strictEqual(r.excerpt, "I kept circling it");
  assert.strictEqual(p.evidenceRow({ weight_delta: -0.1 }).weight_delta, "-0.100");
  assert.strictEqual(p.evidenceRow({}).similarity, "—");
});

test("nodesFrom / evidenceFrom accept a bare list or a wrapped one", () => {
  assert.deepStrictEqual(p.nodesFrom([{ a: 1 }]), [{ a: 1 }]);
  assert.deepStrictEqual(p.nodesFrom({ dispositions: [{ a: 1 }] }), [{ a: 1 }]);
  assert.deepStrictEqual(p.evidenceFrom({ evidence: [{ b: 2 }] }), [{ b: 2 }]);
  assert.deepStrictEqual(p.evidenceFrom(null), []);
});

test("pageState: unavailable without a frame, empty for a zero graph, ok otherwise", () => {
  const down = p.pageState({ frame: null, error: "durable_runs_url_not_configured", upstream_status: 503 });
  assert.strictEqual(down.kind, "unavailable");
  assert.match(down.message, /^durable-runs unavailable: durable_runs_url_not_configured/);
  const empty = p.pageState({ frame: { active_count: 0, candidate_count: 0, retired_count: 0, evidence_rows: 0, dispositions: [] } });
  assert.strictEqual(empty.kind, "empty");
  assert.strictEqual(empty.message, p.EMPTY_TEXT);
  assert.match(empty.message, /starts from today and grows as Orion writes/);
  assert.strictEqual(p.pageState({ frame: { candidate_count: 1 } }).kind, "ok");
  assert.strictEqual(p.pageState(null).kind, "unavailable");
});

test("countsText, choiceStyleText and skipText read the frame and cue", () => {
  assert.strictEqual(p.countsText({ active_count: 2, candidate_count: 5, retired_count: 1, evidence_rows: 40 }),
    "2 active · 5 candidate · 1 retired · 40 evidence rows");
  assert.strictEqual(p.countsText(null), "0 active · 0 candidate · 0 retired · 0 evidence rows");
  assert.strictEqual(p.choiceStyleText({ choice_style: { window_days: 7, choices: 5, took_top: 2, went_own_way: 3 } }),
    "In curiosity over the last 7 days: took the top suggestion 2 of 5 times, went their own way 3 of 5.");
  assert.strictEqual(p.choiceStyleText({ choice_style: { window_days: 7, choices: 0, took_top: 0, went_own_way: 0 } }),
    "No curiosity choices recorded yet.");
  assert.match(p.skipText({ skip_reason: "stale" }), /^stale: /);
  assert.strictEqual(p.skipText({ text: "x", skip_reason: null }), "");
});

test("sparklineModel orders by instant, not by string, across offset formats", () => {
  const m = p.sparklineModel([
    { at: "2026-10-16T12:00:00+00:00", weight: 3 },     // later instant, sorts first as a string
    { at: "2026-10-16T11:30:00.5Z", weight: 1 },
  ], 80, 18);
  assert.strictEqual(m.first, 1);
  assert.strictEqual(m.last, 3);
});

test("pageState reads an invalid frame as unavailable, never as an empty graph", () => {
  const s = p.pageState({ frame: { nope: 1 }, cue: { skip_reason: "invalid_frame" } });
  assert.strictEqual(s.kind, "unavailable");
  assert.doesNotMatch(s.message, /starts from today/);
});

test("shownCountText marks a capped list and renderParamsText names the defaults", () => {
  assert.strictEqual(p.shownCountText(50, 80), "(50 of 80)");
  assert.strictEqual(p.shownCountText(3, 3), "(3)");
  assert.strictEqual(p.shownCountText(3, undefined), "(3)");
  assert.match(p.renderParamsText({ max_age_sec: 3600, top_k: 5 }), /max age 3600s, top 5/);
  assert.strictEqual(p.renderParamsText(null), "");
  assert.match(p.skipText({ skip_reason: "render_failed" }), /^render_failed: The cue renderer failed/);
});
