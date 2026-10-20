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

test("situationText shows each dimension's top value and share of n", () => {
  const mix = { n: 20, arousal: { idle: 14, engaged: 6 }, day_phase: { night: 11, morning: 9 },
                company: { true: 4, false: 16 } };
  assert.strictEqual(p.situationText(mix), "idle 70% · night 55% · alone 80% (n=20)");
  assert.strictEqual(p.situationText({ ...mix, company: { true: 16, false: 4 } }),
    "idle 70% · night 55% · with company 80% (n=20)");
  assert.match(p.situationDetail(mix), /^arousal: idle 14, engaged 6 \| day phase: night 11, morning 9 \| company: alone 16, with company 4 \| n=20/);
});

test("situationText says so when no evidence carries a situation", () => {
  for (const m of [undefined, null, {}, { n: 0, arousal: {}, day_phase: {}, company: { true: 0, false: 0 } }, { n: "x" }]) {
    assert.strictEqual(p.situationText(m), p.NO_SITUATION_TEXT);
    assert.strictEqual(p.situationDetail(m), p.NO_SITUATION_TEXT);
  }
  assert.strictEqual(p.dispositionRow({ disposition_id: "d" }).when, "no situation data yet");
  // Stamped rows but every dimension empty: the count still shows.
  assert.strictEqual(p.situationText({ n: 2, arousal: {}, day_phase: {}, company: {} }), "(n=2)");
});

test("situationText shows every tied value, by name", () => {
  const mix = { n: 4, arousal: { idle: 2, engaged: 2 }, day_phase: { night: 2, dawn: 2 }, company: { true: 2, false: 2 } };
  assert.strictEqual(p.situationText(mix), "engaged/idle 50% · dawn/night 50% · alone/with company 50% (n=4)");
});

test("situationText keeps upstream values verbatim as text; escapeHtml neutralises them", () => {
  const mix = { n: 1, arousal: { "<img src=x onerror=alert(1)>": 1 }, day_phase: {}, company: {} };
  const text = p.situationText(mix);
  assert.strictEqual(text, "<img src=x onerror=alert(1)> 100% (n=1)");  // written with textContent
  assert.ok(!p.escapeHtml(text).includes("<"));
  const row = p.dispositionRow({ disposition_id: "d", situation_mix: mix });
  assert.strictEqual(row.when, text);
});


test("company share is over known rows only (null company is not 'alone')", () => {
  const mix = { n: 10, arousal: { idle: 10 }, day_phase: { night: 10 }, company: { true: 2, false: 2 } };
  const text = p.situationText(mix);
  assert.ok(text.includes("50%"), text);
  assert.ok(!text.includes("20%"), text);
  const none = p.situationText({ n: 3, arousal: { idle: 3 }, day_phase: { night: 3 }, company: { true: 0, false: 0 } });
  assert.ok(!/alone|with company/.test(none), none);
});


test("choice style names a one-day window in the singular", () => {
  const text = p.choiceStyleText({ choice_style: { window_days: 1, choices: 1, took_top: 0, went_own_way: 1 } });
  assert.ok(text.includes("over the last day:"), text);
  assert.ok(!text.includes("1 days"), text);
});

// --- self-naming -------------------------------------------------------------------------------

test("nameView badges: named, rejected, merged (resolved), not yet shown, not reported", () => {
  const named = p.nameView({ name: " the gap ", name_sentence: "It is where I look.", orion_verdict: null,
                             merged_into: null, name_grounding_rank: 1 });
  assert.deepStrictEqual(named.badge, { kind: "named", text: "named" });
  assert.strictEqual(named.name, "the gap");
  assert.strictEqual(named.sentence, "It is where I look.");
  assert.strictEqual(named.grounding, "");   // rank 1 grounds to its own node
  const rejected = p.nameView({ name: null, orion_verdict: "rejected", weight: 3 });
  assert.deepStrictEqual(rejected.badge, { kind: "rejected", text: "Orion: not me" });
  const merged = p.nameView({ merged_into: "d9" }, (id) => (id === "d9" ? "the watch" : null));
  assert.strictEqual(merged.badge.text, "merged into the watch");
  assert.strictEqual(p.nameView({ merged_into: "d9" }).badge.text, "merged into d9");
  assert.deepStrictEqual(p.nameView({ name: null, orion_verdict: null }).badge, { kind: "unshown", text: "not yet shown" });
  // Older durable-runs: no naming keys at all.
  assert.deepStrictEqual(p.nameView({ label: "x" }).badge, { kind: "unknown", text: "naming not reported" });
  assert.strictEqual(p.nameView(null).badge.kind, "unknown");
});

test("nameView shows grounding only when the name points elsewhere, never hiding the name", () => {
  const v = p.nameView({ name: "the gap", name_grounding_rank: 3, name_grounding_margin: -0.0412 });
  assert.strictEqual(v.name, "the gap");
  assert.strictEqual(v.badge.kind, "named");
  assert.strictEqual(v.grounding, "points elsewhere: rank 3 (margin -0.04)");
  assert.strictEqual(p.nameView({ name: "g", name_grounding_rank: 2 }).grounding, "points elsewhere: rank 2");
  assert.strictEqual(p.nameView({ name: "g", name_grounding_rank: null }).grounding, "");
  assert.strictEqual(p.nameView({ name_grounding_rank: 4 }).grounding, "");   // no name, nothing to ground
});

test("namingRow keeps refused rows with reasons and the packet shown, text verbatim", () => {
  const refused = p.namingRow({ naming_id: "n1", verdict: "refused", name: "<script>x</script>",
    refused_reasons: ["name_too_long", "no_evidence_ref"], evidence_refs: '["e1","e2"]',
    shown: '{"label":"the gap","excerpts":["a"]}', packet_id: "p1", written_at: "2026-10-20T06:00:00Z" });
  assert.strictEqual(refused.verdict, "refused");
  assert.strictEqual(refused.name, "<script>x</script>");   // written with textContent, never innerHTML
  assert.strictEqual(refused.refused, "name_too_long, no_evidence_ref");
  assert.strictEqual(refused.evidence_refs, "e1, e2");
  assert.match(refused.packet, /"label": "the gap"/);
  assert.strictEqual(refused.packet_id, "p1");
  const merge = p.namingRow({ verdict: "merge", merge_with: "d2", shown: {} });
  assert.strictEqual(merge.merge_with, "d2");
  assert.strictEqual(merge.packet, "");
  assert.strictEqual(merge.name, "—");
  const blank = p.namingRow(null);
  assert.strictEqual(blank.verdict, "?");
  assert.strictEqual(blank.refused, "");
  assert.strictEqual(p.packetText("not json"), "not json");
});

test("namingFrom accepts the durable-runs wrapper or a bare list", () => {
  assert.deepStrictEqual(p.namingFrom({ naming: [{ a: 1 }] }), [{ a: 1 }]);
  assert.deepStrictEqual(p.namingFrom([{ a: 1 }]), [{ a: 1 }]);
  assert.deepStrictEqual(p.namingFrom({ rows: [{ b: 2 }] }), [{ b: 2 }]);
  assert.deepStrictEqual(p.namingFrom(null), []);
});

test("echoSummary counts cites_own_name from meta (string or object) and says when the mark is absent", () => {
  const rows = [
    { meta: '{"cites_own_name": true}' }, { meta: { cites_own_name: false } },
    { meta: { cites_own_name: true } }, { meta: "{}" },
  ];
  const e = p.echoSummary(rows);
  assert.strictEqual(e.cites, 2);
  assert.strictEqual(e.marked, 3);
  assert.strictEqual(e.text, "Echo: 2 of 4 loaded evidence rows quote this node's own name (1 row predates the mark).");
  const none = p.echoSummary([{ meta: "{}" }, {}]);
  assert.strictEqual(none.cites, 0);
  assert.match(none.text, /none of the 2 loaded evidence rows carry the cites_own_name mark yet/);
  assert.strictEqual(p.echoSummary([]).text, "");
  assert.strictEqual(p.echoSummary([{ meta: "not json" }]).marked, 0);
});

test("DOM layer writes model-written naming text only via textContent", () => {
  const src = require("node:fs").readFileSync(require("node:path").join(__dirname, "personality.js"), "utf8");
  const writes = src.match(/\.innerHTML\s*=/g) || [];
  assert.strictEqual(writes.length, 1);   // the sparkline only
  assert.match(src, /el\("span", \{ class: "node-name", title: r\.naming\.sentence \|\| "" \}, r\.naming\.name\)/);
});
