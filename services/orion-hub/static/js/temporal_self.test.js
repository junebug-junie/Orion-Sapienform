const test = require("node:test");
const assert = require("node:assert");
const ts = require("./temporal_self.js");

const ARCS = [
  { kind: "interoception", arc_id: "i2", status: "closed", began_at: "2026-10-13T12:00:00Z", ended_at: "2026-10-13T12:10:00Z",
    subject_ref: "node:substrate.route", subject_label: "", cumulative_dwell_sec: 300, attention_returns: 2 },
  { kind: "interoception", arc_id: "i1", status: "closed", began_at: "2026-10-13T10:00:00Z", ended_at: "2026-10-13T10:30:00Z",
    subject_ref: "node:substrate.execution", subject_label: "Execution", cumulative_dwell_sec: 155.5, attention_returns: 1 },
  { kind: "attention", arc_id: "a1", status: "open", began_at: "2026-10-14T03:16:37Z", ended_at: null,
    subject_ref: "world:chat", subject_label: "Juniper is talking", cumulative_dwell_sec: 130.6, attention_returns: 0 },
  { kind: "concern", arc_id: "c1", status: "open", began_at: "2026-10-13T06:00:00Z", carried_from_previous_day: true,
    subject_ref: "open-loop-x", subject_label: "", cumulative_dwell_sec: 0, attention_returns: 5 },
];

test("fmtDuration covers seconds, minutes, hours and junk", () => {
  assert.strictEqual(ts.fmtDuration(42), "42s");
  assert.strictEqual(ts.fmtDuration(155.5), "3m");
  assert.strictEqual(ts.fmtDuration(3600), "1h");
  assert.strictEqual(ts.fmtDuration(5400), "1h 30m");
  assert.strictEqual(ts.fmtDuration(null), "—");
  assert.strictEqual(ts.fmtDuration(-1), "—");
});

test("arcLabel falls back to subject_ref when the label is empty", () => {
  assert.strictEqual(ts.arcLabel(ARCS[0]), "node:substrate.route");
  assert.strictEqual(ts.arcLabel(ARCS[1]), "Execution");
  assert.strictEqual(ts.arcLabel({}), "(no subject)");
});

test("timelineModel groups by kind, biggest first, rows oldest first, counts open", () => {
  const groups = ts.timelineModel(ARCS);
  assert.deepStrictEqual(groups.map((g) => g.kind), ["interoception", "attention", "concern"]);
  assert.deepStrictEqual(groups[0].rows.map((r) => r.arc_id), ["i1", "i2"]);
  assert.strictEqual(groups[0].open, 0);
  assert.strictEqual(groups[0].dwell, "8m");
  assert.strictEqual(groups[1].open, 1);
  assert.match(groups[1].rows[0].span, /–now$/);
  assert.strictEqual(groups[2].rows[0].carried, true);
  assert.strictEqual(groups[2].rows[0].returns, 5);
});

test("timelineModel tolerates junk input", () => {
  assert.deepStrictEqual(ts.timelineModel(null), []);
  assert.deepStrictEqual(ts.timelineModel([null, 3]), []);
});

test("regulationModel reads arousal, rest drive and immune findings (live shapes)", () => {
  const reg = {
    arousal: { arousal_level: "engaged", since: "2026-10-14T03:07:37Z", reasons: ["juniper_turn_recent"] },
    drives: [{ drive: "rest", level: 2.7, threshold: 3.0, state: "building", last_discharge_at: "2026-10-12T13:45:21Z" }],
    warnings: [], immune_findings_open: 0,
  };
  const immune = { immune: { open: 2, findings: [{ summary: "loop never closes" }, "raw"], last_run_at: "2026-10-14T00:27:19Z", skipped: "arousal engaged" } };
  const m = ts.regulationModel(reg, immune);
  assert.strictEqual(m.arousal_level, "engaged");
  assert.deepStrictEqual(m.reasons, ["juniper_turn_recent"]);
  assert.strictEqual(m.rest.state, "building");
  assert.strictEqual(m.rest.level, 2.7);
  assert.strictEqual(m.immune_open, 2);
  assert.deepStrictEqual(m.findings.map(ts.findingText), ["loop never closes", "raw"]);
  assert.strictEqual(m.immune_skipped, "arousal engaged");
});

test("regulationModel falls back to regulation's immune count and survives missing halves", () => {
  assert.strictEqual(ts.regulationModel({ immune_findings_open: 3 }, null).immune_open, 3);
  const empty = ts.regulationModel(null, null);
  assert.strictEqual(empty.arousal_level, "unknown");
  assert.strictEqual(empty.rest, null);
  assert.strictEqual(empty.immune_open, null);
});

test("shiftDay crosses month and year boundaries and rejects junk", () => {
  assert.strictEqual(ts.shiftDay("2026-10-13", -1), "2026-10-12");
  assert.strictEqual(ts.shiftDay("2026-11-01", -1), "2026-10-31");
  assert.strictEqual(ts.shiftDay("2026-12-31", 1), "2027-01-01");
  assert.strictEqual(ts.shiftDay("yesterday", -1), null);
});

test("immune that never ran reads as unknown, not as a calm 0", () => {
  const m = ts.regulationModel({ immune_findings_open: 0 }, { immune: { open: 0, findings: [], last_run_at: null } });
  assert.strictEqual(m.immune_open, null);
  assert.strictEqual(m.immune_never_ran, true);
});

test("fmtOnDay dates a time from an earlier day, keeps same/later-day times short", () => {
  assert.notStrictEqual(ts.fmtOnDay("2026-10-12T06:00:00Z", "2026-10-13"), ts.fmtOnDay("2026-10-13T06:00:00Z", "2026-10-13"));
  assert.strictEqual(ts.fmtOnDay(null, "2026-10-13"), "");
});
