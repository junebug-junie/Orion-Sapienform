const test = require("node:test");
const assert = require("node:assert/strict");
const { leaseRows, lentCount, badgeText, rowHtml } = require("./gpu_pool_leases.js");

// Shape of a live GET /api/gpu-pool/state (2026-10-09): circe gpu0 + hecate-gpu0 lendable.
const STATE = {
  cards: [
    { card: "gpu0", host: "circe", vram_gb: 32.0, lendable: true, lent: false },
    { card: "gpu1", host: "circe", vram_gb: 32.0, lendable: false, lent: false },
    { card: "hecate-gpu0", host: "hecate", vram_gb: 32.0, lendable: true, lent: true },
  ],
  roles: [
    { role: "chat", cards: ["gpu0"], status: "confirmed" },
    { role: "agent", cards: ["gpu1"], status: "confirmed" },
    { role: "agent-deep", cards: ["hecate-gpu0"], status: "down" },
    { role: "experiment", cards: ["gpu0", "gpu1"], status: "down" },
  ],
};

test("leaseRows lists only lendable cards, with their roles and serving state", () => {
  const rows = leaseRows(STATE);
  assert.deepEqual(rows.map((r) => r.card), ["gpu0", "hecate-gpu0"]);
  assert.deepEqual(rows[0].roles, ["chat", "experiment"]);
  assert.equal(rows[0].serving, true);   // one confirmed role is enough
  assert.equal(rows[1].serving, false);
  assert.equal(rows[1].lent, true);
  assert.equal(rows[1].host, "hecate");
});

test("leaseRows tolerates an empty or missing payload", () => {
  assert.deepEqual(leaseRows(null), []);
  assert.deepEqual(leaseRows({}), []);
  const row = leaseRows({ cards: [{ card: "x", lendable: true }] })[0];
  assert.equal(row.serving, null);   // no role listed is unknown, not down
});

test("badge counts lent cards and is empty when none are lent", () => {
  const rows = leaseRows(STATE);
  assert.equal(lentCount(rows), 1);
  assert.equal(badgeText(rows), "1 lent");
  rows[1].lent = false;
  assert.equal(badgeText(rows), "");
});

test("rowHtml renders an accessible switch reflecting the lent flag", () => {
  const [circe, hecate] = leaseRows(STATE);
  const on = rowHtml(hecate);
  assert.match(on, /role="switch"/);
  assert.match(on, /aria-checked="true"/);
  assert.match(on, /data-lease-card="hecate-gpu0"/);
  assert.match(on, /data-lease-down="1"/);
  const off = rowHtml(circe);
  assert.match(off, /aria-checked="false"/);
  assert.doesNotMatch(off, /data-lease-down/);
});

test("rowHtml escapes card names", () => {
  const html = rowHtml({ card: '<b>"x"', host: "", vramGb: null, lent: false, roles: [], serving: null });
  assert.doesNotMatch(html, /<b>/);
  assert.match(html, /&lt;b&gt;&quot;x&quot;/);
});

// A display utility (flex/grid/block) on the modal root outranks the browser's `[hidden]` rule,
// which would leave the modal permanently open (same failure as runtime-activity-modal-hidden).
test("the modal root in index.html carries no display utility that would defeat `hidden`", () => {
  const fs = require("node:fs");
  const path = require("node:path");
  const html = fs.readFileSync(path.join(__dirname, "..", "..", "templates", "index.html"), "utf8");
  const tag = html.match(/<div id="gpuLeasesModal"[^>]*>/);
  assert.ok(tag, "gpuLeasesModal markup not found");
  assert.match(tag[0], /\shidden[\s>]/);
  const cls = (tag[0].match(/class="([^"]*)"/) || [, ""])[1].split(/\s+/);
  for (const bad of ["flex", "grid", "block", "inline-flex", "inline-block"]) {
    assert.ok(!cls.includes(bad), `modal root has "${bad}", which overrides hidden`);
  }
});
