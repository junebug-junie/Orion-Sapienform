const test = require("node:test");
const assert = require("node:assert/strict");
const tray = require("./orion-message-tray.js");

const { isOrionMessage, mergeOpenMessages, collapseSelfClearing } = tray;

function alert(kind, minute) {
  return { event_kind: kind, created_at: `2026-10-28T10:${String(minute).padStart(2, "0")}:00Z` };
}

test("isOrionMessage: live type or outreach tag, nothing else", () => {
  assert.equal(isOrionMessage({ notification_type: "endogenous_outreach" }), true);
  assert.equal(isOrionMessage({ tags: ["reading_finding", "endogenous_outreach"] }), true);
  assert.equal(isOrionMessage({ notification_type: "chat_message", tags: ["chat"] }), false);
  assert.equal(isOrionMessage({ notification_type: "collapse_mirror_reply" }), false);
  assert.equal(isOrionMessage(null), false);
});

test("mergeOpenMessages: one per message, newest first, durable full text wins", () => {
  const durable = [
    { message_id: "a", session_id: "s1", preview_text: "short", full_text: "short but full text", created_at: "2026-10-28T08:00:00Z" },
    { message_id: "b", session_id: "s1", preview_text: "older", created_at: "2026-10-27T08:00:00Z" },
  ];
  const live = [
    { message_id: "a", notification_type: "endogenous_outreach", body_text: "short", created_at: "2026-10-28T08:00:00Z" },
    { message_id: "c", notification_type: "endogenous_outreach", body_text: "newest", created_at: "2026-10-28T09:00:00Z" },
    { message_id: "d", notification_type: "chat_message", body_text: "not outreach" },
  ];
  const out = mergeOpenMessages(durable, live, new Set());
  assert.deepEqual(out.map((m) => m.message_id), ["c", "a", "b"]);
  assert.equal(out[1].text, "short but full text");
});

test("mergeOpenMessages: dismissed ids stay gone even if the server still lists them", () => {
  const durable = [{ message_id: "a", preview_text: "x", created_at: "2026-10-28T08:00:00Z" }];
  assert.deepEqual(mergeOpenMessages(durable, [], new Set(["a"])), []);
});

test("collapseSelfClearing: a flapping raise/clear pair disappears entirely", () => {
  const items = [
    alert("execution_dispatch_all_refused", 1),
    alert("execution_dispatch_all_refused_cleared", 2),
    alert("execution_dispatch_all_refused", 3),
    alert("execution_dispatch_all_refused_cleared", 4),
    alert("gpu_pool_seat_failed", 5),
  ];
  const { visible, hidden } = collapseSelfClearing(items);
  assert.deepEqual(visible.map((n) => n.event_kind), ["gpu_pool_seat_failed"]);
  assert.equal(hidden, 4);
});

test("collapseSelfClearing: a still-active condition shows once, its newest item", () => {
  const items = [
    alert("execution_dispatch_theater_tripwire", 1),
    alert("execution_dispatch_theater_tripwire_cleared", 2),
    alert("execution_dispatch_theater_tripwire", 3),
    alert("execution_dispatch_theater_tripwire_still_blocked", 9),
  ];
  const { visible } = collapseSelfClearing(items);
  assert.deepEqual(visible.map((n) => n.event_kind), ["execution_dispatch_theater_tripwire_still_blocked"]);
});

test("collapseSelfClearing: kinds without a clear sibling are untouched", () => {
  const items = [alert("execution_dispatch_all_refused", 1), alert("execution_dispatch_all_refused", 2), alert("x", 3)];
  const { visible, hidden } = collapseSelfClearing(items);
  assert.equal(visible.length, 3);
  assert.equal(hidden, 0);
});

test("collapseSelfClearing: the longest base wins when families nest", () => {
  const items = [
    alert("foo_cleared", 5),
    alert("foo_bar", 6),
    alert("foo_bar_cleared", 4),
  ];
  // foo_bar belongs to family foo_bar (cleared at 4, before it): still active.
  const { visible } = collapseSelfClearing(items);
  assert.deepEqual(visible.map((n) => n.event_kind), ["foo_bar"]);
});

test("collapseSelfClearing: groups map the shown card to its whole family", () => {
  const a = alert("execution_dispatch_theater_tripwire", 1);
  const b = alert("execution_dispatch_theater_tripwire_cleared", 2);
  const c = alert("execution_dispatch_theater_tripwire", 3);
  const other = alert("x", 4);
  const { visible, groups } = collapseSelfClearing([a, b, c, other]);
  assert.deepEqual(visible, [c, other]);
  assert.deepEqual(groups.get(c), [a, b, c]);
  assert.equal(groups.has(other), false);
});
