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

test("rowStatus shows a folded duplicate as merged, not skipped", () => {
  assert.deepStrictEqual(rd.rowStatus({ reading_status: "skipped", duplicate_of: "reading:0" }),
    { text: "merged into another read", tone: "warn" });
  assert.deepStrictEqual(rd.rowStatus({ reading_status: "skipped" }), { text: "skipped", tone: "warn" });
});

test("an already-read URL reads as passed on, not retryable, and submit says it was blocked", () => {
  assert.deepStrictEqual(rd.rowStatus({ reading_status: "skipped", last_error: "already_read" }),
    { text: "already read, passed on", tone: "warn" });
  assert.deepStrictEqual(rd.rowStatus({ reading_status: "stage1_completed", stage2_error: "already_read" }),
    { text: "already read, passed on", tone: "warn" });
  assert.strictEqual(rd.allowedActions(detail({ status: "skipped", last_error: "already_read" })).retry1, false);
  assert.strictEqual(rd.allowedActions(detail({ stage2_status: "skipped", stage2_error: "already_read" })).retry2, false);
  assert.match(rd.submitText({ status: "completed", duplicate: "already_read" }), /blocked as a duplicate by design/);
  assert.match(rd.submitText({ status: "queued", duplicate: "already_queued" }), /joined the read/);
  assert.strictEqual(rd.submitText({ status: "queued", queue_position: 2, queue_depth: 9 }), "Queued: waiting to be read — 2 of 9 in line");
  assert.match(rd.refusalText("already_read"), /duplicate by design/);
});

test("actionResultText does not claim a cancel landed on a finished run", () => {
  assert.match(rd.actionResultText("Cancel read", { action: "cancel_durable_run", durable_status: "completed",
    run_already_finished: true }), /too late, the run had already completed/);
  assert.strictEqual(rd.actionResultText("Cancel read", { action: "cancel_durable_run", durable_status: "cancelled",
    run_already_finished: false }), "Cancel read: cancel_durable_run (run is now cancelled)");
  assert.strictEqual(rd.actionResultText("Retry stage 1", { action: "requeued" }), "Retry stage 1: requeued");
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

test("walletText shows reads today and paused state (no cap)", () => {
  assert.strictEqual(rd.walletText({ done_today: 3, enabled: true }), "3 read today");
  assert.strictEqual(rd.walletText({ done_today: 0, enabled: false }), "0 read today (paused)");
});

test("DOM layer never writes read content as HTML", () => {
  const src = fs.readFileSync(path.join(__dirname, "reading.js"), "utf8");
  assert.doesNotMatch(src, /innerHTML|outerHTML|insertAdjacentHTML|document\.write/);
});

test("documentLabel shows a captured document as its path and short version", () => {
  const sha = "a".repeat(64);
  assert.strictEqual(rd.documentLabel(`file:///mnt/scripts/Orion-Sapienform/docs/my%20spec.md?sha256=${sha}`),
    "/mnt/scripts/Orion-Sapienform/docs/my spec.md (version aaaaaaaaaaaa)");
  assert.strictEqual(rd.documentLabel("file:///srv/notes.md"), "/srv/notes.md");
  assert.strictEqual(rd.documentLabel("https://example.org/a"), null);
});

test("document refusals speak plainly", () => {
  assert.match(rd.refusalText("document_outside_allowed_roots"), /outside the folders/);
  assert.match(rd.refusalText("document_too_large"), /refused rather than read in part/);
});
