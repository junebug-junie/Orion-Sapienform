const test = require("node:test");
const assert = require("node:assert");
const fs = require("node:fs");
const path = require("node:path");
const rd = require("./reading.js");

function detail(overrides) {
  return Object.assign({
    seed_id: "reading:1", status: "done", stage2_status: "failed", last_error: null,
    duplicate_of: null, durable_turns: [],
    handoff: { read_evidence: [{ tool_name: "WebFetch", url: "https://example.org", content_chars: 900 }] },
  }, overrides || {});
}

test("sourceLabel separates Hub, chat, curiosity and World Pulse requests", () => {
  assert.strictEqual(rd.sourceLabel({ requested_by: "juniper", invocation_context: "operator" }), "Juniper (Hub)");
  assert.strictEqual(rd.sourceLabel({ requested_by: "juniper", invocation_context: "unified_chat" }), "Juniper (chat)");
  assert.strictEqual(rd.sourceLabel({ requested_by: "juniper", request: { invocation_context: "operator" } }), "Juniper (Hub)");
  assert.strictEqual(rd.sourceLabel({ requested_by: "orion" }), "Orion (curiosity)");
  assert.strictEqual(rd.sourceLabel({}), "World Pulse");
});

test("statusLabel speaks plainly and flags unknown states", () => {
  assert.deepStrictEqual(rd.statusLabel("completed"), { text: "done", tone: "good" });
  assert.strictEqual(rd.statusLabel("stage1_completed").tone, "busy");
  assert.deepStrictEqual(rd.statusLabel("weird"), { text: "weird", tone: "warn" });
});

test("allowedActions mirrors the server's cancel and retry rules", () => {
  assert.deepStrictEqual(rd.allowedActions(detail()), { cancel: false, retry1: false, retry2: true });
  assert.strictEqual(rd.allowedActions(detail({ handoff: { read_evidence: [] } })).retry2, false);
  assert.strictEqual(rd.allowedActions(detail({ durable_turns: [{ stage: 2, consumed_at: null }] })).retry2, false);
  assert.deepStrictEqual(rd.allowedActions(detail({ status: "pending", stage2_status: "pending" })),
    { cancel: true, retry1: false, retry2: false });
  assert.strictEqual(rd.allowedActions(detail({ status: "done", stage2_status: "claimed" })).cancel, true);
  assert.strictEqual(rd.allowedActions(detail({ status: "failed", stage2_status: "pending" })).retry1, true);
  assert.strictEqual(rd.allowedActions(detail({ status: "skipped", last_error: "stale_digest_item" })).retry1, false);
  assert.deepStrictEqual(rd.allowedActions(detail({ status: "pending", duplicate_of: "reading:0" })),
    { cancel: false, retry1: false, retry2: false });
});

test("refusal codes become sentences; unknown codes pass through", () => {
  assert.match(rd.refusalText("no_read_evidence"), /never actually fetched/);
  assert.strictEqual(rd.refusalText("something_new"), "something_new");
  assert.strictEqual(rd.refusalText(undefined), "request failed");
});

test("listQuery only sends filters that are set", () => {
  assert.strictEqual(rd.listQuery({ phase: "all", offset: 0 }), "phase=all&limit=50&offset=0");
  assert.strictEqual(rd.listQuery({ phase: "failed", kind: "reading", includeStale: true, offset: 50 }),
    "phase=failed&kind=reading&include_stale=true&limit=50&offset=50");
});

test("walletText shows usage against cap and paused state", () => {
  assert.strictEqual(rd.walletText({ done_today: 3, daily_cap: 12, enabled: true }), "3 of 12 used today");
  assert.strictEqual(rd.walletText({ done_today: 0, daily_cap: 12, enabled: false }), "0 of 12 used today (paused)");
});

test("DOM layer never writes read content as HTML", () => {
  const src = fs.readFileSync(path.join(__dirname, "reading.js"), "utf8");
  assert.doesNotMatch(src, /innerHTML|outerHTML|insertAdjacentHTML|document\.write/);
});
